from __future__ import annotations

import os
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.output import OmniChunkRecvHandle, OmniSchedulerOutput
from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)

# Unified debug prefix for all Omni diagnostic logs (grep-friendly). Sub-labels
# (OmniQueueDump / OmniStageStart / OmniStageDone / OmniStageRecv) follow it.
_OMNI_DEBUG_TAG = "[OmniDebug]"

_STATS_INTERVAL_S = 1.0

# Min seconds between queue-snapshot dumps (schedule() runs every DiT step /
# AR token, so per-tick is too noisy). 0 = dump every schedule, <0 = disabled.
_DUMP_INTERVAL_S = float(os.environ.get("VLLM_OMNI_QUEUE_DUMP_INTERVAL_S", "1.0"))

# Upper bound on how long a request may sit in full-payload-input wait
# (the state ``OmniSchedulingCoordinator`` records via ``_waiting_since``)
# before the scheduler force-fails it.  Defends against stuck consumer-side
# requests when the producer drops a full-payload, send fails, or recv
# never arrives.  Override per-deployment via
# VLLM_OMNI_INPUT_WAIT_TIMEOUT_S; set <=0 to disable the safety net.
#
# Scope: this constant only covers the full-payload coordinator path
# (``input_coordinator``).  The async-chunk path uses
# ``chunk_transfer_adapter`` and is not affected by this constant.
_INPUT_WAIT_TIMEOUT_RAW = os.environ.get("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", "300")
try:
    DEFAULT_INPUT_WAIT_TIMEOUT_S: float = float(_INPUT_WAIT_TIMEOUT_RAW)
except ValueError:
    logger.warning(
        "Invalid VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=%r; falling back to 300 seconds.",
        _INPUT_WAIT_TIMEOUT_RAW,
    )
    DEFAULT_INPUT_WAIT_TIMEOUT_S = 300.0


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    # Per-request monotonic timestamps: stage-admit time and prefill-done time,
    # used by the stage start/done debug logs to compute dwell durations.
    _omni_stage_start_times: dict[str, float]
    _omni_prefill_done_times: dict[str, float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._omni_stage_start_times = {}
        self._omni_prefill_done_times = {}

    def _free_input_coordinator_request(self, request_id: str) -> None:
        """Prune full-payload coordinator state for a completed request."""
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            input_coordinator.free_finished_request(request_id)

    # ------------------------------------------------------------------ #
    #  Shared scheduler/output helpers (lift the AR / generation duplicates)
    # ------------------------------------------------------------------ #

    def _consume_pending_connector_output(self, model_mode: str) -> None:
        """Drain ``self._latest_omni_connector_output`` into the coordinator.

        Called at the top of every ``schedule()`` cycle.  Identical between
        AR and generation schedulers except for the ``model_mode`` argument
        forwarded to ``update_request_metadata``.
        """
        connector_output = getattr(self, "_latest_omni_connector_output", None)
        self._latest_omni_connector_output = None
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        if connector_output and connector_output.request_metadata:
            input_coordinator.update_request_metadata(
                self.requests, connector_output.request_metadata, model_mode=model_mode
            )
        input_coordinator.process_pending_full_payload_inputs(
            self.waiting,
            self.running,
            connector_output.stage_recv_req_ids if connector_output else set(),
        )

    def _process_pending_input_timeouts(self) -> None:
        """Force-fail requests waiting on the full-payload coordinator too long.

        Called at the top of every ``schedule()`` cycle, right after
        ``_consume_pending_connector_output``.  Without this hook, a request
        whose producer dropped a payload would sit in the
        full-payload-input wait state indefinitely (the runner mixin
        protects ``_pending_load_reqs`` from prune sweeps).

        Reads ``_waiting_since`` timestamps maintained by the input
        coordinator and delegates to the base scheduler's
        ``finish_requests`` to mark expired requests FINISHED_ERROR.
        Disabled when ``DEFAULT_INPUT_WAIT_TIMEOUT_S`` is <= 0.

        Scope: only covers ``input_coordinator`` (full-payload path).
        Async-chunk requests park in ``chunk_transfer_adapter`` instead
        and are not handled here -- if a similar safety net is needed
        for the chunk path, it belongs in the chunk adapter.
        """
        if DEFAULT_INPUT_WAIT_TIMEOUT_S <= 0:
            return
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        timed_out_ids = input_coordinator.collect_timed_out_request_ids(timeout_s=DEFAULT_INPUT_WAIT_TIMEOUT_S)
        if not timed_out_ids:
            return
        present_ids = {req_id for req_id in timed_out_ids if req_id in self.requests}
        if not present_ids:
            return
        logger.warning(
            "Marking %d request(s) as FINISHED_ERROR after waiting > %.0fs for connector input: %s",
            len(present_ids),
            DEFAULT_INPUT_WAIT_TIMEOUT_S,
            sorted(present_ids),
        )
        self.finish_requests(present_ids, RequestStatus.FINISHED_ERROR)

    def _capture_omni_connector_output(self, model_runner_output: Any) -> None:
        """Stash the model runner's omni_connector_output for next schedule().

        Called at the tail of every ``update_from_output()`` -- identical
        between AR and generation schedulers.  Only stashes the output;
        applying the metadata is the responsibility of
        ``_consume_pending_connector_output()`` at the start of the next
        ``schedule()`` cycle.  Applying it twice (once here, once on
        consume) is unsafe under ``update_request_metadata`` in
        generation mode, which resets ``prompt_token_ids`` /
        ``_output_token_ids`` / ``num_computed_tokens`` and would
        clobber any progress between the two calls.
        """
        omni_output = getattr(model_runner_output, "omni_connector_output", None)
        if omni_output is None:
            return
        self._latest_omni_connector_output = omni_output

    def _wrap_omni_scheduler_output(
        self,
        base: SchedulerOutput,
        *,
        finished_requests_needing_kv_transfer: dict | None = None,
        pending_input_registrations: list[OmniChunkRecvHandle] | None = None,
    ) -> OmniSchedulerOutput:
        """Wrap a base ``SchedulerOutput`` in ``OmniSchedulerOutput``.

        Pulls each base ``SchedulerOutput`` dataclass field via ``getattr``
        and forwards optional omni-specific fields.  Lifted from 4 separate
        copy-pastes between AR (1) and generation (3) schedulers.
        """
        base_data = {name: getattr(base, name) for name in SchedulerOutput.__dataclass_fields__}
        input_coordinator = getattr(self, "input_coordinator", None)
        if pending_input_registrations is None:
            pending_input_registrations = input_coordinator.pending_input_registrations if input_coordinator else []
        return OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_requests_needing_kv_transfer or {},
            pending_input_registrations=pending_input_registrations,
        )

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        now = time.monotonic()
        if now - getattr(self, "_last_stats_time", 0.0) < _STATS_INTERVAL_S:
            return None
        self._last_stats_time = now
        return super().make_stats(*args, **kwargs)

    # ------------------------------------------------------------------ #
    #  Request stage start / done timing logs (debug)
    # ------------------------------------------------------------------ #

    def _omni_task_label_for_req(self, req) -> str:
        """Infer the task type from ``additional_information``."""
        info = getattr(req, "additional_information", None)
        if info is not None and not isinstance(info, dict):
            try:
                info = deserialize_additional_information(info)
            except Exception:
                info = None
        if isinstance(info, dict):
            fsid = info.get("omni_final_stage_id")
            if fsid == 0:
                return "i2t"
            if fsid is not None and fsid > 0:
                bot = info.get("bot_task")
                return f"it2i/{bot}" if bot else "t2i"
        return "?"

    def _omni_prompt_brief_for_req(self, req) -> str:
        """A short prompt fingerprint to distinguish requests."""
        ptids = getattr(req, "prompt_token_ids", None)
        if ptids and isinstance(ptids, (list, tuple)):
            head = ptids[:8]
            return f"tokens_head={list(head)}"
        return ""

    def _log_stage_start(self, request_id: str, req: Any, stage_label: str) -> None:
        """Log a request's first admission to a stage (WAITING -> RUNNING).

        Idempotent: records the monotonic admit timestamp once per request,
        consumed by :meth:`_log_stage_done` for dwell computation.
        """
        key = request_id
        if key in self._omni_stage_start_times:
            return
        now_mono = time.monotonic()
        now_wall = time.time()
        self._omni_stage_start_times[key] = now_mono
        task = self._omni_task_label_for_req(req)
        prompt = self._omni_prompt_brief_for_req(req)
        arrival = getattr(req, "arrival_time", None)
        wait_ms = int((now_wall - arrival) * 1000) if arrival else -1
        ts_iso = datetime.fromtimestamp(now_wall, tz=timezone.utc).isoformat()
        logger.info(
            "%s OmniStageStart stage=%s requestId=%s task=%s wait=%dms ts=%s mono=%.6f %s",
            _OMNI_DEBUG_TAG, stage_label, request_id, task, wait_ms,
            ts_iso, now_mono, prompt,
        )

    def _log_prefill_done(self, request_id: str, request: Any, stage_label: str) -> None:
        """Log the prefill->decode switch point (first decode token sampled).

        Idempotent. ``prefill_ms`` = this instant minus the stage-admit time.
        """
        if request_id in self._omni_prefill_done_times:
            return
        now_mono = time.monotonic()
        now_wall = time.time()
        self._omni_prefill_done_times[request_id] = now_mono
        start_mono = self._omni_stage_start_times.get(request_id)
        prefill_ms = int((now_mono - start_mono) * 1000) if start_mono is not None else -1
        task = self._omni_task_label_for_req(request)
        prompt = self._omni_prompt_brief_for_req(request)
        ts_iso = datetime.fromtimestamp(now_wall, tz=timezone.utc).isoformat()
        logger.info(
            "%s OmniPrefillDone stage=%s requestId=%s task=%s prefill=%dms ts=%s mono=%.6f %s",
            _OMNI_DEBUG_TAG, stage_label, request_id, task, prefill_ms,
            ts_iso, now_mono, prompt,
        )

    def _log_stage_done(self, request_id: str, request: Any, stage_label: str) -> None:
        """Log a request's completion at a stage, with dwell durations."""
        now_mono = time.monotonic()
        now_wall = time.time()
        start_mono = self._omni_stage_start_times.pop(request_id, None)
        prefill_done_mono = self._omni_prefill_done_times.pop(request_id, None)
        if start_mono is not None:
            dur_ms = int((now_mono - start_mono) * 1000)
        else:
            dur_ms = -1
        if prefill_done_mono is not None:
            prefill_ms = int((prefill_done_mono - start_mono) * 1000) if start_mono is not None else -1
            decode_ms = int((now_mono - prefill_done_mono) * 1000)
        else:
            prefill_ms = -1
            decode_ms = 0
        task = self._omni_task_label_for_req(request)
        prompt = self._omni_prompt_brief_for_req(request)
        status = getattr(request, "status", None)
        arrival = getattr(request, "arrival_time", None)
        e2e_ms = int((now_wall - arrival) * 1000) if arrival else -1
        status_name = status.name if status is not None else "?"
        ts_iso = datetime.fromtimestamp(now_wall, tz=timezone.utc).isoformat()
        logger.info(
            "%s OmniStageDone stage=%s requestId=%s task=%s status=%s dur=%dms e2e=%dms ts=%s mono=%.6f %s",
            _OMNI_DEBUG_TAG, stage_label, request_id, task,
            status_name, dur_ms, e2e_ms, ts_iso, now_mono, prompt,
        )

    def _dump_queue_snapshot(self, stage_label: str) -> None:
        """Dump an internal queue snapshot once per :data:`_DUMP_INTERVAL_S`.

        Prints waiting / running / finished with a per-request brief
        (requestId | task | status | finished | wait_ms | prompt/computed |
        out | prompt_fingerprint). AR additionally dumps the KV-transfer state
        sets; generation (LLM_GENERATION) dumps ``_pending_finish_reqs``.
        Only covers stages running the vLLM V1 scheduler (LLM_AR /
        LLM_GENERATION); diffusion stages are dumped by
        ``_BaseScheduler._dump_diffusion_queue_snapshot``.
        """
        if _DUMP_INTERVAL_S < 0:
            return
        now = time.monotonic()
        if now - getattr(self, "_last_queue_dump_ts", 0.0) < _DUMP_INTERVAL_S:
            return
        self._last_queue_dump_ts = now

        now_wall = time.time()

        def _task_label(req) -> str:
            info = getattr(req, "additional_information", None)
            if info is not None and not isinstance(info, dict):
                try:
                    info = deserialize_additional_information(info)
                except Exception:
                    info = None
            if isinstance(info, dict):
                fsid = info.get("omni_final_stage_id")
                if fsid == 0:
                    return "i2t"
                if fsid is not None and fsid > 0:
                    bot = info.get("bot_task")
                    return f"it2i/{bot}" if bot else "t2i"
                fot = info.get("final_output_type")
                if fot:
                    return str(fot)
            return "?"

        def _prompt_fingerprint(req) -> str:
            ptids = getattr(req, "prompt_token_ids", None)
            if ptids and isinstance(ptids, (list, tuple)):
                head = ptids[:8]
                return f"prompt_head={list(head)}"
            return ""

        def _req_brief(req) -> str:
            rid = getattr(req, "request_id", "?")
            status = getattr(req, "status", None)
            status_name = status.name if status is not None else "?"
            is_fin = getattr(req, "is_finished", None)
            fin_flag = "FINISHED" if (callable(is_fin) and is_fin()) else "active"
            arrival = getattr(req, "arrival_time", None)
            wait_ms = int((now_wall - arrival) * 1000) if arrival else -1
            npt = getattr(req, "num_prompt_tokens", 0)
            nct = getattr(req, "num_computed_tokens", 0)
            out_n = len(getattr(req, "_output_token_ids", []) or [])
            pfp = _prompt_fingerprint(req)
            parts = [
                f"reqId={rid}",
                f"task={_task_label(req)}",
                f"status={status_name}",
                f"finished={fin_flag}",
                f"wait={wait_ms}ms",
                f"prompt={npt}/computed={nct}",
                f"out={out_n}",
            ]
            if pfp:
                parts.append(pfp)
            return "|".join(parts)

        waiting = [r for r in self.waiting] if self.waiting is not None else []
        running = list(self.running) if self.running else []
        finished = (
            list(self.finished_req_ids) if getattr(self, "finished_req_ids", None) else []
        )

        lines = [
            f"{_OMNI_DEBUG_TAG} OmniQueueDump {stage_label} waiting={len(waiting)} "
            f"running={len(running)} finished={len(finished)}"
        ]
        for r in waiting:
            lines.append(f"  WAIT  {_req_brief(r)}")
        for r in running:
            lines.append(f"  RUN   {_req_brief(r)}")
        for fid in finished:
            lines.append(f"  DONE  reqId={fid}")

        # AR-only: KV-transfer state machine.
        kv_sets = {
            "need_xfer": getattr(self, "requests_needing_kv_transfer", None),
            "active": getattr(self, "active_kv_transfers", None),
            "wait_free": getattr(self, "waiting_for_transfer_free", None),
            "pend_stop": getattr(self, "pending_stop_after_extraction", None),
            "triggered": getattr(self, "transfer_triggered_requests", None),
        }
        if any(v for v in kv_sets.values()):
            kv_summary = " ".join(f"{k}={len(v)}" for k, v in kv_sets.items() if v)
            lines.append(f"  KV   {kv_summary}")
            for label in ("active", "wait_free"):
                s = kv_sets[label]
                if s:
                    ids = list(s)
                    lines.append(f"  KV.{label} {ids}")

        # Generation-only: _pending_finish_reqs.
        pending = getattr(self, "_pending_finish_reqs", None)
        if pending:
            ids = [getattr(r, "request_id", "?") for r in pending]
            lines.append(f"  PEND_FINISH ({len(pending)}) {ids}")

        logger.info("\n".join(lines))

    def _realign_request_status_to_queues(
        self,
        request_ids: str | Iterable[str] | None,
    ) -> None:
        """Realign ``request.status`` to actual queue membership.

        ``OmniChunkTransferAdapter._process_chunk_queue`` stamps
        ``requests_origin_status[req.id] = WAITING`` (or ``RUNNING``) when
        first parking a request in a chunk-transfer deque. On the next
        tick, when the chunk arrives, ``_process_chunk_queue`` sets
        ``request.status = target_status`` and continues, but
        ``requests_origin_status`` is left at its first-park value -- no
        hook updates it on the ``waiting → running`` admit transition
        that ``super().schedule()`` later performs. The table stays
        stale until the request makes another deque round-trip.

        If an abort lands in the gap between admit and the next deque
        round-trip, ``chunk_transfer_adapter.finish_requests`` reads the
        stale ``WAITING`` from ``requests_origin_status``, stomps it
        onto ``request.status``, and the upstream
        ``Scheduler.finish_requests`` else branch silently fails to
        remove from ``self.running`` -- the request stays alive in
        ``self.running`` and the worker's ``input_batch`` slot leaks.
        After ``max_num_seqs`` such aborts every new request hangs at
        ``chunks=0`` until the client times out.

        Realign here: if a request lives in ``self.running`` but its
        status is not ``RUNNING``, set it to ``RUNNING``; symmetrically
        flip ``RUNNING → WAITING`` when the request is actually in
        ``self.waiting``. This is a localized safety net for
        ``requests_origin_status`` staleness on the admit transition;
        it does not touch the adapter's invariants and is complementary
        to the chunk-transfer-adapter deque purge that already runs
        inside ``process_pending_chunks`` / ``restore_queues``.

        Note on scope: only the ``async_chunk`` path actually triggers
        the ``requests_origin_status`` staleness this helper repairs.
        When ``async_chunk`` is disabled, no chunk-transfer round-trip
        occurs between admit and finish, so the realignment walk is a
        cheap O(n) no-op over an already-aligned set. The call is kept
        unconditional in ``finish_requests`` to (a) keep the abort path
        uniform and (b) defend any future configuration that re-enables
        chunk transfer from rediscovering the same regression.

        See https://github.com/vllm-project/vllm-omni/pull/3774 and the
        residual-hang reproduction discussed in that PR.
        """
        # Mirror the upstream Scheduler.finish_requests resolution of
        # ``request_ids`` so realignment touches exactly the set that
        # ``super().finish_requests`` will then walk.
        if isinstance(request_ids, str):
            ids_to_align: Iterable[str] = (request_ids,)
        elif request_ids is None:
            ids_to_align = list(self.requests.keys())
        else:
            ids_to_align = list(request_ids)

        if not ids_to_align:
            return

        running_ids = {r.request_id for r in self.running}
        waiting_ids = {r.request_id for r in self.waiting}

        for rid in ids_to_align:
            req = self.requests.get(rid)
            if req is None or req.is_finished():
                continue
            if rid in running_ids and req.status != RequestStatus.RUNNING:
                req.status = RequestStatus.RUNNING
            elif rid in waiting_ids and req.status == RequestStatus.RUNNING:
                req.status = RequestStatus.WAITING

    def _purge_finished_from_running(self) -> None:
        """Defensive post-finish sweep of ``self.running``.

        Belt-and-suspenders to ``_realign_request_status_to_queues``:
        even after status realignment lets upstream
        ``Scheduler.finish_requests`` pick the right removal branch,
        a future regression or an unexpected ``status`` mid-transition
        could still leave already-finished entries in ``self.running``.
        Sweeping here guarantees the worker's ``input_batch`` slot is
        not pinned by a freed request.

        Complementary to ``_realign_request_status_to_queues``: realign
        is preventive (fix ``status`` before ``super().finish_requests``
        so the right branch fires); this purge is defensive (sweep the
        residue after ``super().finish_requests`` so any stale entries
        are reclaimed).

        Scope of the predicate. ``is_finished()`` covers entries the
        upstream ``finish_requests`` already drained from ``self.requests``
        but failed to remove from ``self.running``; the
        ``request_id not in self.requests`` arm catches the same surface
        from a different angle and is the post-cleanup mirror of the
        deque purge ``_purge_untracked_chunk_requests`` already runs at
        the chunk-transfer-adapter layer. It does **not** by itself make
        arbitrary direct deletions of ``self.requests`` safe -- callers
        that pop ``self.requests`` outside the standard finish path
        still have to go through ``_free_request`` (or equivalent) for
        block / connector / coordinator cleanup. This sweep only
        reclaims the ``self.running`` slot reference.

        In-place via ``self.running[:] = ...`` for minor consistency
        with idiomatic vLLM scheduler mutation; upstream
        ``Scheduler.finish_requests`` itself rebinds ``self.running``,
        so list identity across the whole call is not preserved -- the
        slice form is just to avoid an extra rebind inside this helper.

        Assumes the upstream V1 invariant that scheduler ticks are
        serialized on a single thread; in-place mutation here is no more
        racy than the rest of the scheduler under that assumption.

        See https://github.com/vllm-project/vllm-omni/pull/3774
        discussion.
        """
        if not self.running:
            return
        self.running[:] = [req for req in self.running if not req.is_finished() and req.request_id in self.requests]
