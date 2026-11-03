"""Bounded, read-only shadow audit for the ``cmr_myo_empty`` stratum (n=106).

Objective
---------
Sample 30 positive cases plus 10 controls, replay the cached stage summaries
produced by runs 4/5/9/13-18, and report -- as interval bands, never as a new
threshold -- how many positives really are valid slice-vacuum masks.

Safety envelope
---------------
* Read-only: the audit never mutates the input records and never writes labels;
the selected-cohort fingerprint is recomputed afterwards and must match.
* Cache-only replay: a cached summary is reused only when its run is in
  :data:`CACHE_RUNS` and its content hash matches the recorded expectation.
* Hash mismatch: the *only* recovery path is the run-9 B1 extractor
  (:data:`B1_EXTRACTOR_RUN`) and every mismatch is emitted as one report line.
* No production rule, review rule or QC safety check is modified here.

The module has no third-party dependencies and performs no I/O by itself:
cache access and extraction are injected callables, so unit tests can drive the
whole audit with in-memory fixtures.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

CACHE_RUNS: Tuple[int, ...] = (4, 5, 9, 13, 14, 15, 16, 17, 18)
B1_EXTRACTOR_RUN: int = 9
POSITIVE_SAMPLE_SIZE: int = 30
CONTROL_SAMPLE_SIZE: int = 10
MIN_CACHE_REUSE_RATE: float = 0.80
MAX_CONTROL_FALSE_POSITIVE_RATE: float = 0.10
DEFAULT_SEED: int = 20240517

POSITIVE_ARM = "positive"
CONTROL_ARM = "control"

# Reporting bands only: no decision threshold is derived from them.
RATE_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("0-25%", 0.0, 0.25),
    ("25-50%", 0.25, 0.50),
    ("50-75%", 0.50, 0.75),
    ("75-100%", 0.75, 1.000000001),
)

CacheLoader = Callable[[Mapping[str, Any]], Optional[Mapping[str, Any]]]
B1Extractor = Callable[[Mapping[str, Any]], Optional[bool]]


def band_for_rate(rate: Optional[float]) -> str:
    """Return the reporting band label for a rate in ``[0, 1]``.

    ``None`` means the rate could not be resolved.  Bands are half-open
    intervals and are used for reporting only; they never gate anything beyond
    the acceptance metric that was handed to this audit.
    """
    if rate is None:
        return "unresolved"
    value = float(rate)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"rate outside [0, 1]: {rate!r}")
    for label, low, high in RATE_BANDS:
        if low <= value < high:
            return label
    return RATE_BANDS[-1][0]


def _fingerprint(cases: List[Mapping[str, Any]]) -> str:
    """Stable digest of the selected cohort, used to prove read-only replay."""
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda item: str(item.get("case_id", ""))):
        digest.update(
            repr(sorted((str(key), repr(value)) for key, value in case.items())).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _cached_run(cached: Mapping[str, Any]) -> Optional[int]:
    try:
        return int(cached.get("run"))
    except (AttributeError, TypeError, ValueError):
        return None


def hash_verified(cached: Mapping[str, Any], expected_hash: Optional[str]) -> bool:
    """True when the cached summary may be reused as-is.

    A missing expected hash means the upstream manifest gave nothing to verify
    against; the summary is still reused (read-only) and no mismatch is emitted.
    """
    if expected_hash is None:
        return True
    return cached.get("content_hash") == expected_hash


def _mismatch_line(
    case: Mapping[str, Any],
    cached: Mapping[str, Any],
    expected_hash: Optional[str],
) -> str:
    return (
        f"case_id={case.get('case_id')} arm={case.get('arm')} "
        f"cached_run={_cached_run(cached)} expected_hash={expected_hash} "
        f"observed_hash={cached.get('content_hash')} "
        f"action=re-extract via run-{B1_EXTRACTOR_RUN} B1 extractor"
    )


def _default_cache_loader(case: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Inline-fixture adapter: read the cache entry attached to the case."""
    return case.get("cached_summary")


def _default_b1_extractor(case: Mapping[str, Any]) -> Optional[bool]:
    """Adapter for the run-9 B1 extractor (``None`` when unavailable)."""
    value = case.get("b1_slice_vacuum")
    return None if value is None else bool(value)


def select_bounded_cohort(
    cases: List[Mapping[str, Any]] | Tuple[Mapping[str, Any], ...],
    *,
    positives: int = POSITIVE_SAMPLE_SIZE,
    controls: int = CONTROL_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> List[Mapping[str, Any]]:
    """Deterministically sample the bounded cohort (30 positives, 10 controls).

    Selection is stratified by ``arm`` and seeded, so the same pool always
    yields the same cohort.  A stratum smaller than the requested size raises
    ``ValueError`` instead of silently shrinking the audit.
    """
    ordered = sorted(cases, key=lambda item: str(item.get("case_id", "")))
    positive_pool = [case for case in ordered if case.get("arm") == POSITIVE_ARM]
    control_pool = [case for case in ordered if case.get("arm") == CONTROL_ARM]
    if len(positive_pool) < positives:
        raise ValueError(
            f"positive pool too small: {len(positive_pool)} < {positives}"
        )
    if len(control_pool) < controls:
        raise ValueError(
            f"control pool too small: {len(control_pool)} < {controls}"
        )
    rng = random.Random(seed)
    picked = rng.sample(positive_pool, positives) + rng.sample(control_pool, controls)
    picked.sort(key=lambda item: (str(item.get("arm")), str(item.get("case_id"))))
    return list(picked)


@dataclass(frozen=True)
class BoundedShadowAuditReport:
    positives_selected: int
    controls_selected: int
    cache_hits: int
    cache_misses: int
    cache_reuse_rate: float
    cache_reuse_ok: bool
    hash_mismatches: Tuple[str, ...]
    b1_recovered: int
    positive_resolved: int
    positive_valid_slice_vacuum: int
    positive_valid_ratio: Optional[float]
    positive_band: str
    control_resolved: int
    control_false_positives: int
    control_false_positive_rate: float
    control_false_positive_ok: bool
    read_only: bool
    notes: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hash_mismatches"] = list(self.hash_mismatches)
        payload["notes"] = list(self.notes)
        return payload


def run_bounded_shadow_audit(
    cases: List[Mapping[str, Any]] | Tuple[Mapping[str, Any], ...],
    *,
    cache_loader: Optional[CacheLoader] = None,
    b1_extractor: Optional[B1Extractor] = None,
    positives: int = POSITIVE_SAMPLE_SIZE,
    controls: int = CONTROL_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> BoundedShadowAuditReport:
    """Replay the bounded cohort read-only and return a banded report.

    Acceptance metrics evaluated (as given, no new thresholds introduced):

    * control false-positive rate <= :data:`MAX_CONTROL_FALSE_POSITIVE_RATE`
    * cached-summary reuse rate >= :data:`MIN_CACHE_REUSE_RATE`
    * every hash mismatch is reported individually and re-extracted only with
      the run-9 B1 extractor

    ``cache_loader`` and ``b1_extractor`` are injected so the audit can run
    offline in tests and against a real reader in the QC worktree.
    """
    cache_loader = cache_loader or _default_cache_loader
    b1_extractor = b1_extractor or _default_b1_extractor

    cohort = select_bounded_cohort(
        cases, positives=positives, controls=controls, seed=seed
    )
    fingerprint_before = _fingerprint(cohort)

    mismatches: List[str] = []
    notes: List[str] = []
    cache_hits = 0
    cache_misses = 0
    b1_recovered = 0
    positives_resolved = 0
    positive_valid = 0
    controls_resolved = 0
    control_false_positives = 0

    for case in cohort:
        case_id = case.get("case_id")
        arm = case.get("arm")
        cached = cache_loader(case)
        determination: Optional[bool] = None

        if cached is None:
            cache_misses += 1
            notes.append(f"{case_id}: no cached summary in reuse set {CACHE_RUNS}")
            continue

        cached_run = _cached_run(cached)
        if cached_run not in CACHE_RUNS:
            cache_misses += 1
            notes.append(
                f"{case_id}: cached summary from run {cached_run} outside reuse set {CACHE_RUNS}"
            )
            continue

        expected_hash = case.get("expected_hash")
        if hash_verified(cached, expected_hash):
            cache_hits += 1
            determination = bool(cached.get("slice_vacuum"))
        else:
            mismatches.append(_mismatch_line(case, cached, expected_hash))
            extracted = b1_extractor(case)
            if extracted is None:
                notes.append(
                    f"{case_id}: run-{B1_EXTRACTOR_RUN} B1 extractor returned no determination"
                )
            else:
                b1_recovered += 1
                determination = bool(extracted)

        if determination is None:
            continue
        if arm == POSITIVE_ARM:
            positives_resolved += 1
            positive_valid += int(determination)
        elif arm == CONTROL_ARM:
            controls_resolved += 1
            control_false_positives += int(determination)
        else:
            notes.append(f"{case_id}: unknown arm {arm!r}; excluded from metrics")

    total_selected = len(cohort)
    cache_reuse_rate = (cache_hits / total_selected) if total_selected else 0.0
    cache_reuse_ok = total_selected > 0 and cache_reuse_rate >= MIN_CACHE_REUSE_RATE

    positive_valid_ratio = (
        (positive_valid / positives_resolved) if positives_resolved else None
    )

    if controls_resolved:
        control_false_positive_rate = control_false_positives / controls_resolved
        control_false_positive_ok = (
            control_false_positive_rate <= MAX_CONTROL_FALSE_POSITIVE_RATE
        )
    else:
        control_false_positive_rate = 0.0
        control_false_positive_ok = False
        notes.append("no control case resolved; control false-positive gate not evaluable")

    notes.append(
        "positive rate reported as band only; no threshold derived from this audit"
    )

    fingerprint_after = _fingerprint(cohort)
    read_only = fingerprint_before == fingerprint_after
    if not read_only:
        notes.append("cohort fingerprint changed during the audit: read-only violated")

    return BoundedShadowAuditReport(
        positives_selected=sum(1 for case in cohort if case.get("arm") == POSITIVE_ARM),
        controls_selected=sum(1 for case in cohort if case.get("arm") == CONTROL_ARM),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_reuse_rate=cache_reuse_rate,
        cache_reuse_ok=cache_reuse_ok,
        hash_mismatches=tuple(mismatches),
        b1_recovered=b1_recovered,
        positive_resolved=positives_resolved,
        positive_valid_slice_vacuum=positive_valid,
        positive_valid_ratio=positive_valid_ratio,
        positive_band=band_for_rate(positive_valid_ratio),
        control_resolved=controls_resolved,
        control_false_positives=control_false_positives,
        control_false_positive_rate=control_false_positive_rate,
        control_false_positive_ok=control_false_positive_ok,
        read_only=read_only,
        notes=tuple(notes),
    )


def format_report(report: BoundedShadowAuditReport) -> str:
    """Render the report as text, one line per hash mismatch."""
    total = report.positives_selected + report.controls_selected
    lines = [
        "cmr_myo_empty bounded read-only shadow audit",
        f"cases selected: {report.positives_selected} positive + {report.controls_selected} control",
        f"cache reuse: {report.cache_hits}/{total} "
        f"({report.cache_reuse_rate:.2%}) ok={report.cache_reuse_ok}",
        f"cache misses: {report.cache_misses}",
        f"hash mismatches: {len(report.hash_mismatches)} "
        f"(recovered via run-{B1_EXTRACTOR_RUN} B1: {report.b1_recovered})",
        f"positives resolved: {report.positive_resolved}, "
        f"valid slice-vacuum: {report.positive_valid_slice_vacuum}, "
        f"band={report.positive_band}",
        f"control false positives: {report.control_false_positives}/"
        f"{report.control_resolved} ({report.control_false_positive_rate:.2%}) "
        f"ok={report.control_false_positive_ok}",
        f"read-only: {report.read_only}",
    ]
    for mismatch in report.hash_mismatches:
        lines.append(f"HASH MISMATCH {mismatch}")
    for note in report.notes:
        lines.append(f"NOTE {note}")
    return "\n".join(lines)
