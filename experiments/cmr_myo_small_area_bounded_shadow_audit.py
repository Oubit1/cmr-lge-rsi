"""Bounded read-only shadow audit for cmr_myo_small_area (n=11105).

Objective (owner: cmr_specialist):
  * Audit 50 positive (small-area) cases + 10 CMR controls.
  * Reuse cached features from runs 4/5/9/13-18 when the cached content
    hash matches the current payload hash (latest reusable run preferred).
  * On hash mismatch, fall back ONLY to the run-9 B1 extractor and report
    every fallback (cache misses are reported separately).
  * Never modify labels; never add production rules (read-only shadow).

Fully offline and self-contained: operates on in-memory records only and
performs no I/O against datasets, patient images, production labels,
credentials, or runtime databases.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

RULE_ID = "cmr_myo_small_area"
COHORT_SIZE = 11105
POSITIVE_QUOTA = 50
CONTROL_QUOTA = 10

# Runs whose feature caches may be reused by this shadow audit.
REUSABLE_RUN_IDS: Tuple[int, ...] = (4, 5, 9, 13, 14, 15, 16, 17, 18)

# Sole sanctioned fallback: the run-9 B1 extractor.
FALLBACK_RUN_ID = 9
FALLBACK_EXTRACTOR_ID = "B1"

# Nominal small-area threshold (px) used only by the offline default extractor.
SMALL_AREA_THRESHOLD_PX = 50

CACHE_SOURCE = "cache"
FALLBACK_SOURCE = "fallback_b1"


def content_hash(payload: bytes) -> str:
    """SHA-256 hex digest used for cache-validity checks."""
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CaseRecord:
    """Immutable view of one audited case; ``label`` is never written back."""

    case_id: str
    is_positive: bool
    payload: bytes
    label: str = ""


@dataclass(frozen=True)
class CacheEntry:
    """One cached feature row from a previous run."""

    run_id: int
    case_id: str
    content_hash: str
    features: Mapping[str, float]


@dataclass(frozen=True)
class CaseAuditOutcome:
    """Per-case shadow outcome (no production effect)."""

    case_id: str
    is_positive: bool
    source: str  # CACHE_SOURCE or FALLBACK_SOURCE
    run_id: Optional[int]
    hash_mismatch: bool
    features: Mapping[str, float]


@dataclass(frozen=True)
class ShadowAuditReport:
    """Aggregate, JSON-serializable shadow-audit report."""

    rule_id: str
    cohort_size: int
    positives_audited: int
    controls_audited: int
    audited_total: int
    cache_hits: int
    cache_reuse_rate: float
    hash_mismatch_fallbacks: int
    hash_mismatch_case_ids: Tuple[str, ...]
    cache_miss_fallbacks: int
    cache_miss_case_ids: Tuple[str, ...]
    fallback_run_id: int
    fallback_extractor_id: str
    label_modifications: int
    new_production_rules: int
    outcomes: Tuple[CaseAuditOutcome, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "cohort_size": self.cohort_size,
            "positives_audited": self.positives_audited,
            "controls_audited": self.controls_audited,
            "audited_total": self.audited_total,
            "cache_hits": self.cache_hits,
            "cache_reuse_rate": self.cache_reuse_rate,
            "hash_mismatch_fallbacks": self.hash_mismatch_fallbacks,
            "hash_mismatch_case_ids": list(self.hash_mismatch_case_ids),
            "cache_miss_fallbacks": self.cache_miss_fallbacks,
            "cache_miss_case_ids": list(self.cache_miss_case_ids),
            "fallback_run_id": self.fallback_run_id,
            "fallback_extractor_id": self.fallback_extractor_id,
            "label_modifications": self.label_modifications,
            "new_production_rules": self.new_production_rules,
            "outcomes": [
                {
                    "case_id": o.case_id,
                    "is_positive": o.is_positive,
                    "source": o.source,
                    "run_id": o.run_id,
                    "hash_mismatch": o.hash_mismatch,
                    "features": dict(o.features),
                }
                for o in self.outcomes
            ],
        }


def default_b1_extractor(case: CaseRecord) -> Dict[str, float]:
    """Deterministic offline stand-in for the run-9 B1 extractor.

    Real deployments inject the genuine extractor via ``b1_extractor``;
    this default keeps tests and dry-runs offline.
    """
    digest = hashlib.sha256(b"B1:" + case.payload).hexdigest()
    area_px = float(int(digest[:8], 16) % 500)
    return {
        "myo_area_px": area_px,
        "small_area_flag": 1.0 if area_px < SMALL_AREA_THRESHOLD_PX else 0.0,
        "extractor_run_id": float(FALLBACK_RUN_ID),
    }


def latest_reusable_cache_entry(
    cache_manifest: Mapping[int, Mapping[str, CacheEntry]], case_id: str
) -> Optional[CacheEntry]:
    """Cache entry from the latest sanctioned run, or None.

    Runs outside REUSABLE_RUN_IDS are never reused, even if present.
    """
    for run_id in sorted(REUSABLE_RUN_IDS, reverse=True):
        entries = cache_manifest.get(run_id)
        if entries and case_id in entries:
            return entries[case_id]
    return None


def select_audit_sample(
    cases: Iterable[CaseRecord],
    positive_quota: int = POSITIVE_QUOTA,
    control_quota: int = CONTROL_QUOTA,
) -> Tuple[List[CaseRecord], List[CaseRecord]]:
    """Deterministically select up to the quotas (sorted by case_id)."""
    ordered = sorted(cases, key=lambda c: c.case_id)
    positives = [c for c in ordered if c.is_positive][:positive_quota]
    controls = [c for c in ordered if not c.is_positive][:control_quota]
    return positives, controls


def run_shadow_audit(
    cases: Iterable[CaseRecord],
    cache_manifest: Mapping[int, Mapping[str, CacheEntry]],
    b1_extractor: Optional[Callable[[CaseRecord], Mapping[str, float]]] = None,
    positive_quota: int = POSITIVE_QUOTA,
    control_quota: int = CONTROL_QUOTA,
) -> ShadowAuditReport:
    """Execute the bounded read-only shadow audit.

    Read-only guarantees:
      * labels are snapshotted before/after and compared (never written);
      * ``new_production_rules`` is hard-coded to 0;
      * the only extraction path besides sanctioned caches is run-9 B1.
    """
    extractor = b1_extractor if b1_extractor is not None else default_b1_extractor
    positives, controls = select_audit_sample(cases, positive_quota, control_quota)
    sample = sorted(positives + controls, key=lambda c: c.case_id)

    labels_before = {c.case_id: c.label for c in sample}

    outcomes: List[CaseAuditOutcome] = []
    cache_hits = 0
    mismatch_ids: List[str] = []
    miss_ids: List[str] = []

    for case in sample:
        expected = content_hash(case.payload)
        entry = latest_reusable_cache_entry(cache_manifest, case.case_id)
        if entry is not None and entry.content_hash == expected:
            cache_hits += 1
            outcomes.append(
                CaseAuditOutcome(
                    case_id=case.case_id,
                    is_positive=case.is_positive,
                    source=CACHE_SOURCE,
                    run_id=entry.run_id,
                    hash_mismatch=False,
                    features=dict(entry.features),
                )
            )
            continue

        # Sole sanctioned fallback: run-9 B1 extractor.
        features = dict(extractor(case))
        if entry is not None:
            mismatch_ids.append(case.case_id)
            hash_mismatch = True
        else:
            miss_ids.append(case.case_id)
            hash_mismatch = False
        outcomes.append(
            CaseAuditOutcome(
                case_id=case.case_id,
                is_positive=case.is_positive,
                source=FALLBACK_SOURCE,
                run_id=FALLBACK_RUN_ID,
                hash_mismatch=hash_mismatch,
                features=features,
            )
        )

    labels_after = {c.case_id: c.label for c in sample}
    label_modifications = sum(
        1 for cid in labels_before if labels_before[cid] != labels_after.get(cid)
    )

    audited_total = len(sample)
    cache_reuse_rate = (cache_hits / audited_total) if audited_total else 0.0

    return ShadowAuditReport(
        rule_id=RULE_ID,
        cohort_size=COHORT_SIZE,
        positives_audited=len(positives),
        controls_audited=len(controls),
        audited_total=audited_total,
        cache_hits=cache_hits,
        cache_reuse_rate=cache_reuse_rate,
        hash_mismatch_fallbacks=len(mismatch_ids),
        hash_mismatch_case_ids=tuple(mismatch_ids),
        cache_miss_fallbacks=len(miss_ids),
        cache_miss_case_ids=tuple(miss_ids),
        fallback_run_id=FALLBACK_RUN_ID,
        fallback_extractor_id=FALLBACK_EXTRACTOR_ID,
        label_modifications=label_modifications,
        new_production_rules=0,
        outcomes=tuple(outcomes),
    )


def verify_acceptance(report: ShadowAuditReport) -> Dict[str, bool]:
    """Check the three acceptance metrics against a report."""
    checks = {
        "audit_completed_cases>=60_and_cache_reuse_recorded": (
            report.audited_total >= POSITIVE_QUOTA + CONTROL_QUOTA
            and 0.0 <= report.cache_reuse_rate <= 1.0
        ),
        "hash_mismatch_fallbacks_recorded_and_reported": (
            report.hash_mismatch_fallbacks
            == sum(1 for o in report.outcomes if o.hash_mismatch)
            and report.hash_mismatch_fallbacks == len(report.hash_mismatch_case_ids)
            and report.fallback_run_id == FALLBACK_RUN_ID
            and report.fallback_extractor_id == FALLBACK_EXTRACTOR_ID
        ),
        "no_label_modifications_and_no_new_production_rules": (
            report.label_modifications == 0 and report.new_production_rules == 0
        ),
    }
    checks["all_passed"] = all(checks.values())
    return checks
