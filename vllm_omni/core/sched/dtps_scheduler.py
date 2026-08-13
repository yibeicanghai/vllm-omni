"""DTPS (DiT-priority Type-based Scheduling) — AR-side reorder strategy.

DTPS maximizes throughput for AR+DiT unified deployments serving a mix of two
request families on the same AR stage:

* ``ar_only`` — requests that finish at the AR stage (i2t / t2t). They occupy
  AR for a long decode tail and feed no downstream stage.
* ``ar_downstream`` — requests that continue to a downstream diffusion stage
  (t2i / it2i). They occupy AR only briefly and feed the expensive DiT stage.

The core idea: keep the downstream stage fed by admitting ``ar_downstream``
ahead of ``ar_only`` whenever both are waiting, so DiT never starves while AR
burns a long decode on an i2t request. An aging mechanism promotes
long-waiting ``ar_only`` requests to prevent starvation.

Scope / decoupling contract
---------------------------
DTPS is gated to AR+DiT deployments only. The AR scheduler constructs a
``DTPSScheduler`` solely from the ``omni_dtps_config`` config block — an
INDEPENDENT engine arg kept out of ``omni_kv_config``. That block originates
from the model deploy YAML's top-level ``dtps:`` section, which
``merge_pipeline_deploy`` validates, normalizes, and injects as
``omni_dtps_config`` — but only when the pipeline topology is
``stage 0 == LLM_AR`` and some downstream stage is ``DIFFUSION``. Every other
deployment form gets no ``omni_dtps_config``, so no ``DTPSScheduler`` is
constructed and the AR schedule() hot path is untouched (pure FCFS).
Model-specific knobs (the CoT-weight table, the tag field name, the aging
threshold) live in that deploy YAML block, not here.

Task classification is by the request's declared task type
(``additional_information["omni_task_type"]``), stamped at the API entry point.
i2t / t2t finish at the AR stage (ar_only bucket); t2i / it2i continue to the
downstream diffusion stage (ar_downstream bucket). When that tag is absent,
classification falls back to deriving i2t/t2i/it2i from the generic
``omni_final_stage_id`` + ``bot_task`` fields, so DTPS stays correct for
untagged traffic.

The ``ar_proxy`` ordering key is ``num_prompt_tokens`` (universal on Request,
prefill length) plus two optional terms: a model-specific CoT-length hint
(``cot_weight_table`` / ``cot_tag_key``) and the AR stage's output-token budget
(``req.max_tokens``) scaled by ``max_tokens_divisor``. With both terms off,
``ar_proxy`` degrades to pure ``num_prompt_tokens`` ordering. Nothing
model-specific is hardcoded here.

Module 2 — DiT-stage load awareness
-----------------------------------
On top of the Module 1 layering (L0 starving i2t -> L1 ar_downstream -> L2 rest
i2t), Module 2 caps how many ``ar_downstream`` are admitted ahead of i2t per
batch based on the DiT stage's queue load, fed in from the Orchestrator via a
:class:`DitLoadSharedState` shared-memory segment (each DiT replica reports its
``num_waiting`` every poll tick). The threshold is BOTH the phase boundary AND
a per-batch admission cap. With a single threshold the DiT phase is exactly one
of two complements:

  * **idle** — ANY live DiT replica has ``waiting < dit_load_threshold``
    (``min_waiting < threshold``): admit ``ar_downstream`` ahead of i2t, but
    only up to the DiT admission budget
    ``budget = max(0, threshold - effective_min) * n_reps`` — the first
    ``budget`` downstream (by ar_proxy) form L1 (admitted before i2t), the rest
    form L3 (demoted AFTER i2t). When headroom covers all downstream, L3 is
    empty and the order is the Module 1 L0 -> L1 -> L2 (feed t2i/it2i first).
    When it does not, L0 -> L1(cap) -> L2 -> L3: post-admit DiT load stays
    <= threshold and the leftover slots let some i2t get scheduled.
  * **busy** — EVERY live DiT replica has ``waiting >= dit_load_threshold``
    (``min_waiting >= threshold``): ``budget = 0`` ⟹ L1 empty ⟹ L0 -> L2 -> L3
    (all downstream after i2t) so pure-AR i2t interleaves and finishes early
    instead of piling more work onto an already-long DiT queue.

``threshold <= 0`` disables Module 2 (budget unbounded ⟹ L3 always empty ⟹
pure Module 1 L0 -> L1 -> L2). No live DiT signal yet (no replica tracked) also
yields idle: ``min_waiting`` defaults to 0, which is below any positive
threshold. L0 (starving ar_only) is always the highest layer — aging-based
starvation prevention is never traded away for load awareness.

The shared-memory segment NAME is injected at runtime (before the AR subprocess
is spawned) into the shared ``omni_dtps_config`` dict under
``"_dit_load_shm_name"``; the AR subprocess reattaches to the segment by name
at scheduler construction, so the reader is live from the first schedule()
tick.
"""

from __future__ import annotations

import os
import time
from typing import Any, NamedTuple

from vllm.logger import init_logger

from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)

_OMNI_DEBUG_TAG = "[OmniDebug]"

_DEFAULT_I2T_AGING_S = 5.0
_DEFAULT_COT_TAG_KEY = "bot_task"
# ar_proxy = num_prompt_tokens + cot_weight + max_tokens // max_tokens_divisor.
# <=0 disables the max_tokens term.
_DEFAULT_MAX_TOKENS_DIVISOR = 0

# Module 2 — single DiT-load threshold. <=0 disables Module 2 (pure Module 1).
_DEFAULT_DIT_LOAD_THRESHOLD = 0

# Blind-spot inflight bookkeeping: a downstream request that finished AR is held
# here until a DiT poll reports its id (then de-duped out) or it times out.
# miss counted only on a FRESH shm snapshot (seq advanced + fresh=True) — AR
# ticks are µs-scale, so counting per-tick would evict a live request almost
# instantly. age caps lifetime so a dead DiT can't leak ids forever.
_DIT_INFLIGHT_MAX_MISS = 3
_DIT_INFLIGHT_MAX_AGE_S = 1.0


class _InflightEntry(NamedTuple):
    """One finished-AR-but-not-yet-in-DiT downstream request."""

    miss: int
    added_mono: float

from vllm_omni.core.sched.dit_load_shared import (
    DitLoadSharedState,
    _DIT_LOAD_SHM_NAME_KEY,
)

# i2t / t2t finish at the AR stage -> ar_only; t2i / it2i -> ar_downstream.
_AR_ONLY_TASKS: frozenset[str] = frozenset({"i2t", "t2t"})
_AR_DOWNSTREAM_TASKS: frozenset[str] = frozenset({"t2i", "it2i"})

# Min seconds between DTPS reorder dumps (maybe_reorder_waiting runs every
# schedule() tick). ~1Hz; 0 = every reorder, <0 = disabled.
try:
    _DTPS_DUMP_INTERVAL_S = float(
        os.environ.get("VLLM_OMNI_DTPS_DUMP_INTERVAL_S", "1.0")
    )
except ValueError:
    _DTPS_DUMP_INTERVAL_S = 1.0


class DTPSScheduler:
    """DTPS Module 1 (priority reorder of the AR waiting queue) + Module 2
    (DiT load-aware admission cap). A single instance is owned by
    ``OmniARScheduler`` (one per AR stage replica) and invoked once per
    ``schedule()`` cycle via :meth:`maybe_reorder_waiting`.
    """

    def __init__(
        self,
        *,
        i2t_aging_s: float = _DEFAULT_I2T_AGING_S,
        cot_tag_key: str = _DEFAULT_COT_TAG_KEY,
        cot_weight_table: dict[str, int] | None = None,
        max_tokens_divisor: int = _DEFAULT_MAX_TOKENS_DIVISOR,
        dit_load_threshold: int = _DEFAULT_DIT_LOAD_THRESHOLD,
        dit_load_shm_name: str | None = None,
    ) -> None:
        self.i2t_aging_s: float = i2t_aging_s
        self.cot_tag_key: str = cot_tag_key
        try:
            self.max_tokens_divisor: int = int(max_tokens_divisor)
        except (TypeError, ValueError):
            self.max_tokens_divisor = _DEFAULT_MAX_TOKENS_DIVISOR
        try:
            self.dit_load_threshold: int = int(dit_load_threshold)
        except (TypeError, ValueError):
            self.dit_load_threshold = _DEFAULT_DIT_LOAD_THRESHOLD
        # Attach the cross-process DiT-load segment eagerly. The NAME was
        # injected into omni_dtps_config before this subprocess was spawned, so
        # the segment already exists; a failed attach leaves Module 2 inert
        # (phase stays idle, no flip) — DTPS never breaks the AR scheduler.
        self._dit_load: DitLoadSharedState | None = None
        if dit_load_shm_name and self.dit_load_threshold > 0:
            try:
                self._dit_load = DitLoadSharedState.attach(dit_load_shm_name)
            except Exception:
                logger.debug(
                    "[OmniDTPS] DitLoadSharedState.attach(%r) failed; "
                    "Module 2 will stay idle",
                    dit_load_shm_name, exc_info=True,
                )
        if cot_weight_table is None:
            self.cot_weight_table: dict[str, int] = {}
        elif hasattr(cot_weight_table, "items"):
            self.cot_weight_table = dict(cot_weight_table.items())
        else:
            self.cot_weight_table = dict(cot_weight_table)

        # Module 2 blind-spot set: request_id -> _InflightEntry. Populated by
        # ``register_finished_downstream`` (called from OmniARScheduler._free_request
        # when a downstream req leaves AR's running set). De-duped against DiT's
        # reported waiting/running ids inside ``_dit_phase``; aged out on a dead
        # DiT. ``_last_shm_seq`` gates miss-counting to fresh snapshots only.
        self._dit_inflight_ids: dict[str, _InflightEntry] = {}
        self._last_shm_seq: int = -1
        self._last_phase_stats: dict[str, int | bool] = {}

    @classmethod
    def from_config(cls, dtps_cfg: Any) -> "DTPSScheduler":
        """Build a DTPSScheduler from the ``omni_dtps_config`` block.

        ``dtps_cfg`` may be a dict (the normal case, injected by
        ``merge_pipeline_deploy``) or an object with attribute access. Unknown
        keys are ignored so the config schema can grow without breaking older
        AR schedulers.
        """
        if isinstance(dtps_cfg, dict):
            cfg_get = dtps_cfg.get
        else:
            def cfg_get(key: str, default: Any = None) -> Any:
                return getattr(dtps_cfg, key, default)

        if not cfg_get("enabled", False):
            raise ValueError(
                "DTPS config block present but 'enabled' is not True; "
                "refusing to construct DTPSScheduler."
            )

        # Normalize cot_weight_table via .items() so an OmegaConf DictConfig
        # (which dict() would iterate as keys) coerces cleanly to dict[str,int].
        raw_table = cfg_get("cot_weight_table", None)
        if raw_table is None:
            cot_weight_table: dict[str, int] | None = None
        elif hasattr(raw_table, "items"):
            cot_weight_table = {str(k): int(v) for k, v in raw_table.items()}
        else:
            raise ValueError(
                "DTPS cot_weight_table must be a mapping; got "
                f"{type(raw_table).__name__}."
            )

        raw_divisor = cfg_get("max_tokens_divisor", _DEFAULT_MAX_TOKENS_DIVISOR)
        try:
            max_tokens_divisor = int(raw_divisor)
        except (TypeError, ValueError):
            logger.warning(
                "[OmniDTPS] Invalid max_tokens_divisor=%r; using default %d.",
                raw_divisor, _DEFAULT_MAX_TOKENS_DIVISOR,
            )
            max_tokens_divisor = _DEFAULT_MAX_TOKENS_DIVISOR

        raw_threshold = cfg_get("dit_load_threshold", _DEFAULT_DIT_LOAD_THRESHOLD)
        try:
            dit_load_threshold = int(raw_threshold)
        except (TypeError, ValueError):
            logger.warning(
                "[OmniDTPS] Invalid dit_load_threshold=%r; using default %d.",
                raw_threshold, _DEFAULT_DIT_LOAD_THRESHOLD,
            )
            dit_load_threshold = _DEFAULT_DIT_LOAD_THRESHOLD

        # The SHM name is injected at runtime by StageRuntime before the AR
        # subprocess is spawned; absent here only when Module 2 is disabled.
        dit_load_shm_name = cfg_get(_DIT_LOAD_SHM_NAME_KEY, None)

        return cls(
            i2t_aging_s=float(cfg_get("i2t_aging_s", _DEFAULT_I2T_AGING_S)),
            cot_tag_key=str(cfg_get("cot_tag_key", _DEFAULT_COT_TAG_KEY)),
            cot_weight_table=cot_weight_table,
            max_tokens_divisor=max_tokens_divisor,
            dit_load_threshold=dit_load_threshold,
            dit_load_shm_name=dit_load_shm_name,
        )

    # ------------------------------------------------------------------ #
    #  Task classification (model-agnostic)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deserialize_info(req: Any) -> dict[str, Any] | None:
        info = getattr(req, "additional_information", None)
        if info is None:
            return None
        if isinstance(info, dict):
            return info
        try:
            info = deserialize_additional_information(info)
        except Exception:
            return None
        return info if isinstance(info, dict) else None

    def _classify_task(self, req: Any) -> str:
        """Classify a request by task type: i2t / t2i / it2i / t2t / unknown.

        Primary signal: ``additional_information["omni_task_type"]`` (stamped
        at the API entry). Fallback: derive from ``omni_final_stage_id`` +
        ``bot_task`` when the tag is absent (older clients / untagged models).
        """
        info = self._deserialize_info(req)
        if info is None:
            return "unknown"
        # 1) Explicit API-entry task type — the preferred signal.
        tag = info.get("omni_task_type")
        if isinstance(tag, str) and tag:
            return tag
        # 2) Fallback: derive from final-stage routing + bot_task.
        fsid = info.get("omni_final_stage_id")
        if fsid == 0:
            return "i2t"
        if fsid is not None:
            try:
                if int(fsid) > 0:
                    return "it2i" if info.get("bot_task") else "t2i"
            except (TypeError, ValueError):
                pass
        return "unknown"

    @staticmethod
    def _task_bucket(task: str) -> str:
        """Map a task type to its DTPS bucket: ``ar_only`` or ``ar_downstream``.

        unknown / unrecognized -> ``ar_downstream`` (conservative: never starve
        the downstream stage).
        """
        if task in _AR_ONLY_TASKS:
            return "ar_only"
        if task in _AR_DOWNSTREAM_TASKS:
            return "ar_downstream"
        return "ar_downstream"

    def _ar_proxy_parts(self, req: Any) -> tuple[int, dict[str, Any]]:
        """Compute ``ar_proxy`` and return ``(proxy, parts)``.

        ``parts`` breaks the proxy into its components. Split out from
        :meth:`_ar_proxy` so :meth:`maybe_reorder_waiting` can compute the proxy
        once per reorder and reuse both the value and the breakdown.
        """
        num_prompt = getattr(req, "num_prompt_tokens", 0) or 0
        proxy = num_prompt

        # CoT-length weight from the model-specific tag (e.g. bot_task).
        info = self._deserialize_info(req)
        cot_weight = 0
        cot_tag = None
        if info is not None and self.cot_weight_table:
            cot_tag = info.get(self.cot_tag_key)
            if cot_tag is not None:
                # Unknown tag value -> 0 extra weight (treated as no-CoT).
                cot_weight = int(self.cot_weight_table.get(cot_tag, 0))
        proxy += cot_weight

        # Scaled AR output-token budget. ``req.max_tokens`` is always set on a
        # generative vllm Request; guard for safety in case a non-standard
        # request type lacks it.
        max_tokens = getattr(req, "max_tokens", 0) or 0
        max_tokens_term = 0
        if self.max_tokens_divisor > 0 and max_tokens > 0:
            max_tokens_term = max_tokens // self.max_tokens_divisor
            proxy += max_tokens_term

        return proxy, {
            "num_prompt": num_prompt,
            "cot_weight": cot_weight,
            "cot_tag": cot_tag,
            "max_tokens": max_tokens,
            "max_tokens_term": max_tokens_term,
        }

    # ------------------------------------------------------------------ #
    #  Module 2: DiT-stage load-aware phase
    # ------------------------------------------------------------------ #

    def register_finished_downstream(self, request_id: str) -> None:
        """Record that a downstream (t2i/it2i) request just finished AR.

        Called from ``OmniARScheduler._free_request`` at the top of the KV-
        transfer block (so only downstream requests register). The id lives in
        ``_dit_inflight_ids`` until a DiT poll reports it (de-duped out) or it
        times out (3 fresh misses / age cap). Idempotent: re-registering an id
        already tracked does NOT reset its miss/age. No-op when Module 2 is
        disabled — the set is never read and stays empty.
        """
        if self.dit_load_threshold <= 0:
            return
        if not request_id or request_id in self._dit_inflight_ids:
            return
        self._dit_inflight_ids[request_id] = _InflightEntry(miss=0, added_mono=time.monotonic())

    def _dit_phase(self, inflight: int = 0) -> str:
        """Return the DiT-load phase: ``"idle"`` or ``"busy"``.

        ``inflight`` is the Fix-B feed-forward count of downstream (t2i/it2i)
        requests currently RUNNING on this AR stage — AR knows they will land
        on DiT once they finish here but the polled DiT-load report hasn't
        reflected them yet. ``_dit_inflight_ids`` is the blind-spot set: ids
        that already LEFT AR's running set (t0) but haven't surfaced in a DiT
        poll yet. The two terms are mutually exclusive (a request is in
        AR-running XOR in the blind set), so they sum cleanly.

        De-dup pass: any blind id that appears in DiT's reported waiting OR
        running ids (union across all replicas) has reached DiT — drop it. Ids
        that miss across ``_DIT_INFLIGHT_MAX_MISS`` fresh snapshots (DiT empty
        -> straight to running, or aborted) are dropped, as is anything older
        than ``_DIT_INFLIGHT_MAX_AGE_S`` (guards a dead DiT). Miss is counted
        only on a FRESH snapshot (seq advanced + fresh=True); AR ticks are
        µs-scale, so counting per-tick would evict a live id almost instantly.

        Multi-replica: both inflight terms (running + blind) spread uniformly
        across ``n_reps`` DiT replicas, so only ~1/R of them land on the min
        replica and actually raise ``min_waiting``. Fold them together and
        floor-divide by R (R<=1 -> no fold, single-replica Fix-B behavior
        exact). Floor biases toward idle (feed DiT) — safe per DTPS's goal.

        idle (no flip) when Module 2 is disabled (threshold <= 0) or the
        effective load is below threshold; busy (flip L1/L2) otherwise. With no
        SHM attached the reported term is 0 and inflight alone drives the phase.
        Records intermediate counts in ``_last_phase_stats``.
        """
        inflight_running = max(int(inflight or 0), 0)
        if self.dit_load_threshold <= 0:
            self._last_phase_stats = {}
            return "idle"

        reported_min = 0
        max_waiting = 0
        total_waiting = 0
        total_running = 0
        waiting_ids: frozenset[str] = frozenset()
        running_ids: frozenset[str] = frozenset()
        n_reps = 0
        snap_seq = self._last_shm_seq
        fresh = False
        load = self._dit_load
        if load is not None:
            try:
                read = load.snapshot()
                snap = read.snapshot
                reported_min = int(snap.get("min_waiting", 0))
                max_waiting = int(snap.get("max_waiting", 0))
                total_waiting = int(snap.get("total_waiting", 0))
                total_running = int(snap.get("total_running", 0))
                w_ids = snap.get("waiting_ids")
                r_ids = snap.get("running_ids")
                if isinstance(w_ids, frozenset):
                    waiting_ids = w_ids
                if isinstance(r_ids, frozenset):
                    running_ids = r_ids
                n_reps = int(snap.get("num_replicas", 0))
                snap_seq = int(read.seq)
                fresh = bool(snap.get("fresh", False))
            except Exception:
                logger.debug(
                    "[OmniDTPS] DitLoadSharedState.snapshot() raised; "
                    "using inflight only",
                    exc_info=True,
                )

        # A snapshot is "fresh for miss-counting" only if it advanced (new seq)
        # AND carries live data. Counting misses on a stale (unchanged) snapshot
        # would evict a live id in µs across AR ticks.
        fresh_poll = fresh and snap_seq != self._last_shm_seq
        self._last_shm_seq = snap_seq

        dit_ids = waiting_ids | running_ids
        now_mono = time.monotonic()
        for rid in list(self._dit_inflight_ids):
            entry = self._dit_inflight_ids[rid]
            if rid in dit_ids:
                del self._dit_inflight_ids[rid]
            elif now_mono - entry.added_mono > _DIT_INFLIGHT_MAX_AGE_S:
                del self._dit_inflight_ids[rid]
            elif fresh_poll:
                miss = entry.miss + 1
                if miss >= _DIT_INFLIGHT_MAX_MISS:
                    del self._dit_inflight_ids[rid]
                else:
                    self._dit_inflight_ids[rid] = _InflightEntry(
                        miss=miss, added_mono=entry.added_mono
                    )

        inflight_blind = len(self._dit_inflight_ids)
        if inflight_blind > 0:
            logger.info(
                f"{_OMNI_DEBUG_TAG}[dit_phase] _dit_inflight_ids={self._dit_inflight_ids}"
            )
        inflight_total = inflight_running + inflight_blind
        if n_reps <= 1:
            inflight_reduced = inflight_total
        else:
            inflight_reduced = inflight_total // n_reps
        effective_min = reported_min + inflight_reduced
        phase = "busy" if effective_min >= self.dit_load_threshold else "idle"

        self._last_phase_stats = {
            "reported_min": reported_min,
            "max_waiting": max_waiting,
            "total_waiting": total_waiting,
            "total_running": total_running,
            "inflight_running": inflight_running,
            "inflight_blind": inflight_blind,
            "inflight_total": inflight_total,
            "n_reps": n_reps,
            "inflight_reduced": inflight_reduced,
            "effective_min": effective_min,
            "fresh_poll": fresh_poll,
        }
        return phase

    def _dit_load_summary(self, inflight: int = 0) -> str:
        """One-line DiT-load snapshot for the reorder dump (Module 2 only)."""
        if self.dit_load_threshold <= 0:
            return ""
        infl = max(int(inflight or 0), 0)
        load = self._dit_load
        if load is None:
            return f"dit=none +infl={infl}" if infl else "dit=none"
        stats = self._last_phase_stats
        if not stats:
            suffix = f" +infl={infl}" if infl else ""
            return f"dit=none{suffix}"
        nrun = int(stats.get("inflight_running", 0))
        mblind = int(stats.get("inflight_blind", 0))
        total = int(stats.get("inflight_total", 0))
        n_reps = int(stats.get("n_reps", 0))
        nred = int(stats.get("inflight_reduced", 0))
        rep_min = int(stats.get("reported_min", 0))
        max_w = int(stats.get("max_waiting", 0))
        tot_w = int(stats.get("total_waiting", 0))
        tot_r = int(stats.get("total_running", 0))
        eff = int(stats.get("effective_min", 0))
        if n_reps <= 1:
            fold = f"{total}//{n_reps}={nred}"
        else:
            fold = f"({nrun}+{mblind})={total}//{n_reps}={nred}"
        return (
            f"dit[min_w={rep_min},max_w={max_w},tot_w={tot_w},tot_r={tot_r},"
            f"reps={n_reps}] infl[Nrun={nrun},Mblind={mblind},{fold},eff={eff}]"
        )

    # ------------------------------------------------------------------ #
    #  Module 1: reorder self.waiting
    # ------------------------------------------------------------------ #

    def maybe_reorder_waiting(
        self,
        waiting: Any,
        running: Any = None,
    ) -> None:
        """Reorder the AR ``waiting`` queue by DTPS priority layers.

        Must be called after ``process_pending_chunks`` /
        ``_consume_pending_connector_output`` (so only genuinely schedulable
        WAITING requests remain) and before ``super().schedule()`` (so the
        admission order follows the reorder).

        ``running`` is the AR stage's current running set; its downstream
        (t2i/it2i) members are fed forward as anticipated DiT load (see
        :meth:`_dit_phase`) so the phase decision isn't fooled by the
        poll-lagged DiT-load report. ``None`` -> the running term is 0.

        Priority layers (smaller layer = admitted first):
          L0 — ``ar_only`` requests waiting longer than ``i2t_aging_s``
               (starving; aging boost to prevent starvation). ALWAYS highest.
          L1 — ``ar_downstream`` within the DiT admission budget, by ar_proxy
          L2 — remaining ``ar_only`` requests
          L3 — ``ar_downstream`` beyond the budget, by ar_proxy
        Within a layer, FCFS follows the ``waiting`` queue order (``arrival_time``
        is only read for the L0 starving check).

        Module 2 sets the admission budget from the DiT phase:
          * idle  -> budget = max(0, threshold - effective_min) * n_reps;
            first ``budget`` downstream form L1 (before i2t), the rest L3.
          * busy  -> budget = 0 -> L1 empty -> L0 < L2 < L3 (i2t first, don't
            pile onto a congested DiT queue).
        L0 stays on top in both phases (no starvation). With Module 2 disabled
        (threshold <= 0) the budget is unbounded, L3 is empty, and the order is
        the pure Module 1 L0 -> L1 -> L2.

        ``ar_proxy`` and arrival are read-only; only queue order is mutated.
        FCFSRequestQueue (a deque subclass) is reordered via clear()+extend();
        any other queue type falls back to remove_requests()+add_request().
        """
        inflight_running = 0
        if running is not None:
            inflight_running = sum(
                1 for r in running
                if self._task_bucket(self._classify_task(r)) == "ar_downstream"
            )
        phase = self._dit_phase(inflight_running)
        if self.dit_load_threshold <= 0:
            budget_raw: int | None = None
        else:
            stats = self._last_phase_stats
            eff_min = int(stats.get("effective_min", 0))
            n_reps = max(1, int(stats.get("n_reps", 0)))
            budget_raw = max(0, self.dit_load_threshold - eff_min) * n_reps

        ar_only_reqs: list = []
        downstream_reqs: list[tuple[int, Any, dict[str, Any]]] = []
        starving_ar_only: list = []
        aging_threshold = self.i2t_aging_s
        now = time.time()
        records: list[dict[str, Any]] = []

        for req in list(waiting):
            task = self._classify_task(req)
            bucket = self._task_bucket(task)
            arrival = getattr(req, "arrival_time", None)
            wait = (now - arrival) if arrival is not None else 0.0
            rec: dict[str, Any] = {
                "req": req,
                "rid": getattr(req, "request_id", "?"),
                "task": task,
                "bucket": bucket,
                "arrival": arrival,
                "wait": wait,
            }
            if bucket == "ar_only":
                starving = wait > aging_threshold
                rec["starving"] = starving
                if starving:
                    starving_ar_only.append(req)
                    rec["layer"] = "L0"
                else:
                    ar_only_reqs.append(req)
                    rec["layer"] = "L2"
            else:
                proxy, parts = self._ar_proxy_parts(req)
                rec["ar_proxy"] = proxy
                rec["ar_proxy_parts"] = parts
                downstream_reqs.append((proxy, req, rec))
                rec["layer"] = "L1/L3"
            records.append(rec)

        downstream_reqs.sort(key=lambda t: t[0])

        if budget_raw is None:
            downstream_head = [t[1] for t in downstream_reqs]
            downstream_tail: list = []
            for _proxy, _req, rec in downstream_reqs:
                rec["layer"] = "L1"
        else:
            downstream_head = [t[1] for t in downstream_reqs[:budget_raw]]
            downstream_tail = [t[1] for t in downstream_reqs[budget_raw:]]
            for _proxy, _req, rec in downstream_reqs[:budget_raw]:
                rec["layer"] = "L1"
            for _proxy, _req, rec in downstream_reqs[budget_raw:]:
                rec["layer"] = "L3"

        ordered = starving_ar_only + downstream_head + ar_only_reqs + downstream_tail
        if hasattr(waiting, "clear") and hasattr(waiting, "extend"):
            waiting.clear()
            waiting.extend(ordered)
        else:
            waiting.remove_requests(list(waiting))
            for req in ordered:
                waiting.add_request(req)

        self._dump_reorder(
            records,
            ordered,
            phase=phase,
            inflight=inflight_running,
            budget_raw=budget_raw,
        )

    # ------------------------------------------------------------------ #
    #  Debug dump (computation + result + basis)
    # ------------------------------------------------------------------ #

    def _dump_reorder(
        self,
        records: list[dict[str, Any]],
        ordered: list,
        *,
        phase: str = "idle",
        inflight: int = 0,
        budget_raw: int | None = None,
    ) -> None:
        """Emit a throttled debug dump of one DTPS reorder.

        Throttled by ``_DTPS_DUMP_INTERVAL_S`` (env ``VLLM_OMNI_DTPS_DUMP_INTERVAL_S``):
        ~1 Hz by default, 0 = every reorder, <0 = disabled. For local
        diagnosis only; remove before formal merge.
        """
        if _DTPS_DUMP_INTERVAL_S < 0:
            return
        now_mono = time.monotonic()
        if _DTPS_DUMP_INTERVAL_S > 0 and (
            now_mono - getattr(self, "_last_dtps_dump_ts", 0.0)
            < _DTPS_DUMP_INTERVAL_S
        ):
            return
        self._last_dtps_dump_ts = now_mono

        if not records:
            return

        def _short(rid: Any) -> str:
            rid = str(rid)
            return rid if len(rid) <= 16 else rid[:16] + "…"

        calc_lines = []
        for rec in records:
            parts = [f"id={_short(rec['rid'])}", f"task={rec['task']}",
                     f"bucket={rec['bucket']}", f"layer={rec['layer']}"]
            if rec["bucket"] == "ar_only":
                parts.append(f"wait={rec['wait']:.2f}s")
                parts.append(f"starving={'Y' if rec['starving'] else 'N'}"
                             f"(thr={self.i2t_aging_s:.1f}s)")
            else:
                p = rec["ar_proxy_parts"]
                parts.append(
                    f"ar_proxy={rec['ar_proxy']}"
                    f"(np={p['num_prompt']}+cot={p['cot_weight']}"
                    f"<{p['cot_tag']}>+mt={p['max_tokens_term']}"
                    f"<{p['max_tokens']}//{self.max_tokens_divisor}>)"
                )
                parts.append(f"wait={rec['wait']:.2f}s")
            calc_lines.append("  " + " | ".join(parts))

        order_ids = [_short(getattr(r, "request_id", "?")) for r in ordered]
        before_ids = [_short(rec["rid"]) for rec in records]
        changed = order_ids != before_ids

        n_l0 = len([r for r in records if r["layer"] == "L0"])
        n_l1 = len([r for r in records if r["layer"] == "L1"])
        n_l2 = len([r for r in records if r["layer"] == "L2"])
        n_l3 = len([r for r in records if r["layer"] == "L3"])
        layer_summary = (
            f"L0(starving i2t)={n_l0} "
            f"L1(ds≤budget)={n_l1} "
            f"L2(i2t)={n_l2} "
            f"L3(ds>budget)={n_l3}"
        )
        dit_summary = self._dit_load_summary(inflight)
        phase_str = f" phase={phase}"
        dit_str = f" {dit_summary}" if dit_summary else ""
        budget_str = (
            " budget=none(Module2 off)"
            if budget_raw is None else
            f" budget={budget_raw}(L1 cap)"
        )
        # Three order-label states: busy (budget=0, L1 empty), idle with the
        # budget binding (non-empty L3 tail), idle with headroom covering all
        # downstream (L3 empty).
        if phase == "busy":
            order_label = "result (admission order, L0→L2→L3 [DiT busy: i2t before downstream])"
        elif n_l3 > 0:
            order_label = (f"result (admission order, L0→L1(cap={n_l1})→L2→L3 "
                           f"[DiT idle, downstream capped at DiT budget])")
        else:
            order_label = "result (admission order, L0→L1→L2 [DiT idle, headroom covers all downstream])"
        lines = [
            f"{_OMNI_DEBUG_TAG} OmniDTPSReorder n={len(records)} {layer_summary}"
            f"{phase_str}{dit_str}{budget_str} "
            f"{'REORDERED' if changed else 'no-change'}",
            "  compute (input order):",
        ]
        lines.extend(calc_lines)
        lines.append(f"  {order_label}:")
        lines.append("    " + " -> ".join(order_ids))
        basis = (
            "  basis: L0=starving ar_only(wait>i2t_aging_s) by arrival; "
            "L1=ar_downstream within DiT budget by (ar_proxy,queue order) where "
            "ar_proxy=num_prompt+cot_weight(tag)+max_tokens//divisor; "
            "L2=rest ar_only by queue order(FCFS); "
            "L3=ar_downstream beyond budget by (ar_proxy,queue order)"
        )
        if budget_raw is None:
            basis += (
                "; MODULE2 off: budget unbounded, L3 empty (pure Module 1 "
                "L0<L1<L2, all downstream before i2t)"
            )
        else:
            basis += (
                f"; MODULE2 budget=max(0,thr-effective_min)*max(1,n_reps)="
                f"{budget_raw}: only this many downstream (by ar_proxy) admitted "
                "before i2t (L1); the rest demoted to L3 (after i2t) so post-admit "
                "DiT load stays ≤ threshold. busy ⟹ budget=0 ⟹ L1 empty ⟹ "
                "L0<L2<L3 (i2t first, don't pile onto a congested DiT queue)"
            )
        lines.append(basis)
        logger.info("\n".join(lines))
