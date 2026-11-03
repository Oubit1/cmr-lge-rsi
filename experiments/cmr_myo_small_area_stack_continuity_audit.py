"""
CMR small-area severity-stratified stack-continuity audit.

Bounded, import-only, read-only audit using validated cached run-9/run-14
per-case/per-slice digests. Reuses the existing harness if the pipeline/config
hash validates; on mismatch uses exactly one import-only extractor under the
run-9 B1 contract and reports the mismatch.

Stratifies cmr_myo_small_area flagged cases into severity tiers crossed with
co-occurrence patterns, reports temporal/spatial continuity and paired-LGE
support, and samples up to 15 cases per stratum.

No adjudication, label edits, rule activation, or threshold changes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

RUN9_DIGEST_PATH = Path("experiments/.cache/run9_per_case_per_slice_digests.json")
RUN14_DIGEST_PATH = Path("experiments/.cache/run14_per_case_per_slice_digests.json")
HARNESS_HASH_PATH = Path("experiments/.cache/run9_pipeline_config_hash.txt")

# Run-9 B1 contract: import-only, read-only extractor interface
B1_CONTRACT_VERSION = "run-9-b1"
B1_EXPECTED_HASH = "run9_b1_expected_hash_placeholder"

SEVERITY_TIERS = {
    "severe": (0.0, 0.10),
    "moderate": (0.10, 0.25),
    "mild": (0.25, 0.50),
}

COOCCURRENCE_PATTERNS = {
    "pattern_a": ("cmr_myo_excess_components", "cmr_myo_touches_image_border"),
    "pattern_b": ("cmr_myo_missing_temporal_hole", "cmr_centroid_drift"),
    "pattern_c": None,  # isolated small area without co-occurrence
}

MAX_CASES_PER_STRATUM = 15
TOTAL_MAX_CASES = 45

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CaseDigest:
    case_id: str
    area_ratio: float  # actual / expected area
    flags: Tuple[str, ...]
    slices: List[Dict[str, Any]]
    lge_present: bool
    lge_info: Optional[Dict[str, Any]]
    run14_shadow: Optional[Dict[str, Any]] = None


@dataclasses.dataclass(frozen=True)
class StratumKey:
    tier: str
    pattern: str


@dataclasses.dataclass
class TemporalMetrics:
    area_cv: float
    area_cv_ci: Tuple[float, float]
    centroid_drift_persistence: float
    centroid_drift_persistence_ci: Tuple[float, float]
    n_cases: int


@dataclasses.dataclass
class SpatialMetrics:
    slice_area_correlation: float
    slice_area_correlation_ci: Tuple[float, float]
    ring_break_persistence: float
    ring_break_persistence_ci: Tuple[float, float]
    layer_area_jump_persistence: float
    layer_area_jump_persistence_ci: Tuple[float, float]
    n_cases: int


@dataclasses.dataclass
class LGESupportMetrics:
    lge_presence_rate: float
    lge_component_count_mean: float
    cross_modality_area_agreement: float
    cross_modality_area_agreement_ci: Tuple[float, float]
    n_lge_present: int


@dataclasses.dataclass
class StratumReport:
    stratum_key: StratumKey
    case_ids: List[str]
    temporal: TemporalMetrics
    spatial: SpatialMetrics
    lge_support: Optional[LGESupportMetrics]
    shadow_discordance_rate: float
    shadow_discordance_n: int
    shadow_discordance_total: int


@dataclasses.dataclass
class AuditResult:
    stratum_reports: List[StratumReport]
    total_flagged: int
    tier_counts: Dict[str, int]
    pattern_counts: Dict[str, int]
    hash_validated: bool
    hash_mismatch_reported: bool
    fallback_extractor_used: bool


# ---------------------------------------------------------------------------
# Hash validation / harness reuse
# ---------------------------------------------------------------------------

def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _validate_harness_hash() -> Tuple[bool, Optional[str]]:
    """
    Validate pipeline/config hash against cached expected hash.
    Returns (is_valid, actual_hash_or_none).
    """
    if not HARNESS_HASH_PATH.exists():
        return False, None
    expected = HARNESS_HASH_PATH.read_text().strip()
    # In practice, hash the shadow module + config
    actual = _compute_file_hash(Path("experiments/cmr_myo_small_area_shadow.py"))
    return actual == expected, actual


# ---------------------------------------------------------------------------
# Import-only fallback extractor (run-9 B1 contract)
# ---------------------------------------------------------------------------

def _import_only_extractor_b1(digest_data: Dict[str, Any]) -> CaseDigest:
    """
    Single import-only extractor under run-9 B1 contract.
    Transforms raw digest dict into CaseDigest without external I/O.
    """
    return CaseDigest(
        case_id=digest_data.get("case_id", "unknown"),
        area_ratio=digest_data.get("area_ratio", 1.0),
        flags=tuple(digest_data.get("flags", [])),
        slices=digest_data.get("slices", []),
        lge_present=digest_data.get("lge_present", False),
        lge_info=digest_data.get("lge_info"),
        run14_shadow=digest_data.get("run14_shadow"),
    )


# ---------------------------------------------------------------------------
# Digest loading with hash validation
# ---------------------------------------------------------------------------

def load_digests() -> Tuple[List[CaseDigest], bool, bool, bool]:
    """
    Load run-9 and run-14 digests. Validate hash, use fallback if needed.
    Returns (cases, hash_validated, hash_mismatch_reported, fallback_used).
    """
    hash_valid, actual_hash = _validate_harness_hash()
    hash_mismatch_reported = False
    fallback_used = False

    cases: List[CaseDigest] = []

    # Attempt to load run-9 digests
    run9_data: Dict[str, Any] = {}
    if RUN9_DIGEST_PATH.exists():
        try:
            run9_data = json.loads(RUN9_DIGEST_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            run9_data = {}

    # Attempt to load run-14 digests for shadow comparison
    run14_data: Dict[str, Any] = {}
    if RUN14_DIGEST_PATH.exists():
        try:
            run14_data = json.loads(RUN14_DIGEST_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            run14_data = {}

    if not hash_valid:
        hash_mismatch_reported = True
        fallback_used = True
        # Use exactly one import-only extractor under B1 contract
        for case_id, raw in run9_data.get("cases", {}).items():
            digest = _import_only_extractor_b1(raw)
            # Merge run-14 shadow if available
            if case_id in run14_data.get("cases", {}):
                digest = dataclasses.replace(
                    digest,
                    run14_shadow=run14_data["cases"][case_id].get("run14_shadow"),
                )
            cases.append(digest)
    else:
        # Reuse existing harness path: directly construct from cached digests
        for case_id, raw in run9_data.get("cases", {}).items():
            digest = CaseDigest(
                case_id=raw.get("case_id", case_id),
                area_ratio=raw.get("area_ratio", 1.0),
                flags=tuple(raw.get("flags", [])),
                slices=raw.get("slices", []),
                lge_present=raw.get("lge_present", False),
                lge_info=raw.get("lge_info"),
                run14_shadow=run14_data.get("cases", {}).get(case_id, {}).get("run14_shadow")
                if case_id in run14_data.get("cases", {})
                else None,
            )
            cases.append(digest)

    return cases, hash_valid, hash_mismatch_reported, fallback_used


# ---------------------------------------------------------------------------
# Severity tier assignment
# ---------------------------------------------------------------------------

def assign_severity_tier(area_ratio: float) -> Optional[str]:
    """Assign severity tier based on area ratio."""
    for tier, (low, high) in SEVERITY_TIERS.items():
        if low <= area_ratio < high:
            return tier
    # Handle boundary: >= 0.50 is not small area, should not be flagged
    if area_ratio < 0.0:
        return None
    # area_ratio >= 0.50: not a small-area case
    return None


# ---------------------------------------------------------------------------
# Co-occurrence pattern detection
# ---------------------------------------------------------------------------

def detect_cooccurrence_pattern(flags: Tuple[str, ...]) -> Optional[str]:
    """
    Detect which co-occurrence pattern applies.
    Returns pattern key or None if no pattern matches.
    """
    flag_set = set(flags)

    # Pattern A: cmr_myo_excess_components + cmr_myo_touches_image_border
    if set(COOCCURRENCE_PATTERNS["pattern_a"]).issubset(flag_set):
        return "pattern_a"

    # Pattern B: cmr_myo_missing_temporal_hole + cmr_centroid_drift
    if set(COOCCURRENCE_PATTERNS["pattern_b"]).issubset(flag_set):
        return "pattern_b"

    # Pattern C: isolated small area (neither A nor B flags present)
    pattern_a_flags = set(COOCCURRENCE_PATTERNS["pattern_a"])
    pattern_b_flags = set(COOCCURRENCE_PATTERNS["pattern_b"])
    all_pattern_flags = pattern_a_flags | pattern_b_flags

    if not any(f in all_pattern_flags for f in flags):
        return "pattern_c"

    # Partial overlap: default to pattern_c as "isolated-ish"
    return "pattern_c"


# ---------------------------------------------------------------------------
# Sampling with discordance preference
# ---------------------------------------------------------------------------

def _has_shadow_discordance(digest: CaseDigest) -> bool:
    """Check if run-14 A1 no-op vs largest-component-retention discordance exists."""
    if digest.run14_shadow is None:
        return False
    shadow = digest.run14_shadow
    # Discordance: no-op path taken but largest-component retention would differ
    noop_taken = shadow.get("a1_no_op", False)
    largest_retained = shadow.get("largest_component_retained", True)
    return noop_taken and not largest_retained


def sample_cases(
    cases: List[CaseDigest],
    max_per_stratum: int = MAX_CASES_PER_STRATUM,
    total_max: int = TOTAL_MAX_CASES,
) -> Dict[StratumKey, List[CaseDigest]]:
    """
    Sample up to max_per_stratum cases per stratum, preferring discordant cases.
    """
    random.seed(RANDOM_SEED)

    # Group by stratum
    by_stratum: Dict[StratumKey, List[CaseDigest]] = defaultdict(list)
    for case in cases:
        tier = assign_severity_tier(case.area_ratio)
        pattern = detect_cooccurrence_pattern(case.flags)
        if tier is None or pattern is None:
            continue
        key = StratumKey(tier=tier, pattern=pattern)
        by_stratum[key].append(case)

    sampled: Dict[StratumKey, List[CaseDigest]] = {}
    total_selected = 0

    for key in sorted(by_stratum.keys(), key=lambda k: (k.tier, k.pattern)):
        stratum_cases = by_stratum[key]
        # Sort: discordant first, then by case_id for stability
        stratum_cases.sort(key=lambda c: (-int(_has_shadow_discordance(c)), c.case_id))
        n = min(max_per_stratum, len(stratum_cases))
        selected = stratum_cases[:n]
        if total_selected + len(selected) > total_max:
            # Trim to respect total max
            allowed = total_max - total_selected
            selected = selected[:allowed]
        if selected:
            sampled[key] = selected
            total_selected += len(selected)

    return sampled


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def _mean_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """Compute mean and normal-approximation confidence interval."""
    if not values:
        return 0.0, (0.0, 0.0)
    n = len(values)
    mean_val = sum(values) / n
    if n < 2:
        return mean_val, (mean_val, mean_val)
    # Sample standard deviation
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std_dev = math.sqrt(variance)
    # 95% CI using normal approximation (z ≈ 1.96)
    z = 1.96
    margin = z * std_dev / math.sqrt(n)
    return mean_val, (max(0.0, mean_val - margin), mean_val + margin)


def compute_temporal_metrics(cases: List[CaseDigest]) -> TemporalMetrics:
    """Compute frame-to-frame area CV and centroid-drift persistence."""
    area_cvs: List[float] = []
    drift_persistences: List[float] = []

    for case in cases:
        # Frame-to-frame area CV
        areas = [s.get("area", 0.0) for s in case.slices if "area" in s]
        if len(areas) > 1:
            mean_area = sum(areas) / len(areas)
            if mean_area > 0:
                variance = sum((a - mean_area) ** 2 for a in areas) / len(areas)
                cv = math.sqrt(variance) / mean_area
                area_cvs.append(cv)

        # Centroid drift persistence (fraction of frames with drift)
        drift_frames = sum(1 for s in case.slices if s.get("centroid_drift", False))
        total_frames = len(case.slices)
        if total_frames > 0:
            drift_persistences.append(drift_frames / total_frames)

    area_cv_mean, area_cv_ci = _mean_ci(area_cvs)
    drift_mean, drift_ci = _mean_ci(drift_persistences)

    return TemporalMetrics(
        area_cv=area_cv_mean,
        area_cv_ci=area_cv_ci,
        centroid_drift_persistence=drift_mean,
        centroid_drift_persistence_ci=drift_ci,
        n_cases=len(cases),
    )


def compute_spatial_metrics(cases: List[CaseDigest]) -> SpatialMetrics:
    """Compute slice-to-slice area correlation, ring-break and layer-area-jump persistence."""
    correlations: List[float] = []
    ring_break_persistences: List[float] = []
    layer_jump_persistences: List[float] = []

    for case in cases:
        # Slice-to-slice area correlation (adjacent slices)
        areas = [s.get("area", 0.0) for s in case.slices if "area" in s]
        if len(areas) >= 2:
            # Simple correlation between adjacent slices
            x = areas[:-1]
            y = areas[1:]
            n = len(x)
            if n > 1:
                mean_x = sum(x) / n
                mean_y = sum(y) / n
                num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
                den_x = sum((xi - mean_x) ** 2 for xi in x)
                den_y = sum((yi - mean_y) ** 2 for yi in y)
                if den_x > 0 and den_y > 0:
                    corr = num / math.sqrt(den_x * den_y)
                    correlations.append(corr)

        # Ring break persistence
        ring_breaks = sum(1 for s in case.slices if s.get("ring_break", False))
        total_slices = len(case.slices)
        if total_slices > 0:
            ring_break_persistences.append(ring_breaks / total_slices)

        # Layer area jump persistence
        layer_jumps = sum(1 for s in case.slices if s.get("layer_area_jump", False))
        if total_slices > 0:
            layer_jump_persistences.append(layer_jumps / total_slices)

    corr_mean, corr_ci = _mean_ci(correlations)
    ring_mean, ring_ci = _mean_ci(ring_break_persistences)
    jump_mean, jump_ci = _mean_ci(layer_jump_persistences)

    return SpatialMetrics(
        slice_area_correlation=corr_mean,
        slice_area_correlation_ci=corr_ci,
        ring_break_persistence=ring_mean,
        ring_break_persistence_ci=ring_ci,
        layer_area_jump_persistence=jump_mean,
        layer_area_jump_persistence_ci=jump_ci,
        n_cases=len(cases),
    )


def compute_lge_support(cases: List[CaseDigest]) -> Optional[LGESupportMetrics]:
    """Compute paired-LGE support metrics for LGE-present subset."""
    lge_cases = [c for c in cases if c.lge_present and c.lge_info is not None]
    if not lge_cases:
        return None

    presence_count = 0
    component_counts: List[float] = []
    area_agreements: List[float] = []

    for case in lge_cases:
        info = case.lge_info or {}
        if info.get("myocardial_presence", False):
            presence_count += 1
        comp_count = info.get("component_count", 0)
        component_counts.append(float(comp_count))
        agreement = info.get("cross_modality_area_agreement", 0.0)
        area_agreements.append(agreement)

    agreement_mean, agreement_ci = _mean_ci(area_agreements)
    n = len(lge_cases)

    return LGESupportMetrics(
        lge_presence_rate=presence_count / n if n > 0 else 0.0,
        lge_component_count_mean=sum(component_counts) / len(component_counts) if component_counts else 0.0,
        cross_modality_area_agreement=agreement_mean,
        cross_modality_area_agreement_ci=agreement_ci,
        n_lge_present=n,
    )


def compute_shadow_discordance(cases: List[CaseDigest]) -> Tuple[float, int, int]:
    """Return (rate, n_discordant, total)."""
    total = len(cases)
    discordant = sum(1 for c in cases if _has_shadow_discordance(c))
    rate = discordant / total if total > 0 else 0.0
    return rate, discordant, total


# ---------------------------------------------------------------------------
# Main audit execution
# ---------------------------------------------------------------------------

def run_audit() -> AuditResult:
    """Execute the full stack-continuity audit."""
    # Load digests with hash validation
    cases, hash_valid, hash_mismatch, fallback_used = load_digests()

    # Filter to cmr_myo_small_area flagged cases only
    flagged_cases = [c for c in cases if "cmr_myo_small_area" in c.flags]
    total_flagged = len(flagged_cases)

    # Sample cases per stratum
    sampled = sample_cases(flagged_cases)

    # Build reports
    reports: List[StratumReport] = []
    tier_counts: Dict[str, int] = defaultdict(int)
    pattern_counts: Dict[str, int] = defaultdict(int)

    for key, stratum_cases in sampled.items():
        temporal = compute_temporal_metrics(stratum_cases)
        spatial = compute_spatial_metrics(stratum_cases)
        lge_support = compute_lge_support(stratum_cases)
        shadow_rate, shadow_n, shadow_total = compute_shadow_discordance(stratum_cases)

        report = StratumReport(
            stratum_key=key,
            case_ids=[c.case_id for c in stratum_cases],
            temporal=temporal,
            spatial=spatial,
            lge_support=lge_support,
            shadow_discordance_rate=shadow_rate,
            shadow_discordance_n=shadow_n,
            shadow_discordance_total=shadow_total,
        )
        reports.append(report)

        tier_counts[key.tier] += len(stratum_cases)
        pattern_counts[key.pattern] += len(stratum_cases)

    return AuditResult(
        stratum_reports=reports,
        total_flagged=total_flagged,
        tier_counts=dict(tier_counts),
        pattern_counts=dict(pattern_counts),
        hash_validated=hash_valid,
        hash_mismatch_reported=hash_mismatch,
        fallback_extractor_used=fallback_used,
    )


def emit_stratum_table(result: AuditResult) -> str:
    """Emit formatted stratum table for all flagged cases."""
    lines = [
        "=" * 80,
        "CMR Small-Area Severity-Stratified Stack-Continuity Audit",
        "=" * 80,
        f"Total flagged cases (cmr_myo_small_area): {result.total_flagged}",
        f"Hash validated: {result.hash_validated}",
        f"Hash mismatch reported: {result.hash_mismatch_reported}",
        f"Fallback extractor used: {result.fallback_extractor_used}",
        "",
        "Tier counts:",
    ]
    for tier in ["severe", "moderate", "mild"]:
        lines.append(f"  {tier}: {result.tier_counts.get(tier, 0)}")
    lines.append("")
    lines.append("Pattern counts:")
    for pattern in ["pattern_a", "pattern_b", "pattern_c"]:
        lines.append(f"  {pattern}: {result.pattern_counts.get(pattern, 0)}")
    lines.append("")
    lines.append("-" * 80)

    for report in result.stratum_reports:
        key = report.stratum_key
        lines.append(f"\nStratum: {key.tier} x {key.pattern}")
        lines.append(f"  Cases: {len(report.case_ids)}")
        lines.append(f"  Case IDs: {', '.join(report.case_ids[:5])}{'...' if len(report.case_ids) > 5 else ''}")
        lines.append(f"  Temporal continuity:")
        lines.append(f"    Frame-to-frame area CV: {report.temporal.area_cv:.4f} {report.temporal.area_cv_ci}")
        lines.append(f"    Centroid drift persistence: {report.temporal.centroid_drift_persistence:.4f} {report.temporal.centroid_drift_persistence_ci}")
        lines.append(f"  Spatial stack continuity:")
        lines.append(f"    Slice-to-slice area correlation: {report.spatial.slice_area_correlation:.4f} {report.spatial.slice_area_correlation_ci}")
        lines.append(f"    Ring-break persistence: {report.spatial.ring_break_persistence:.4f} {report.spatial.ring_break_persistence_ci}")
        lines.append(f"    Layer-area-jump persistence: {report.spatial.layer_area_jump_persistence:.4f} {report.spatial.layer_area_jump_persistence_ci}")
        if report.lge_support is not None:
            lge = report.lge_support
            lines.append(f"  Paired-LGE support (non-gating):")
            lines.append(f"    LGE presence rate: {lge.lge_presence_rate:.4f}")
            lines.append(f"    LGE component count mean: {lge.lge_component_count_mean:.2f}")
            lines.append(f"    Cross-modality area agreement: {lge.cross_modality_area_agreement:.4f} {lge.cross_modality_area_agreement_ci}")
            lines.append(f"    N LGE-present: {lge.n_lge_present}")
        else:
            lines.append("  Paired-LGE support: No LGE-present cases in stratum")
        lines.append(f"  Shadow discordance rate: {report.shadow_discordance_rate:.4f} ({report.shadow_discordance_n}/{report.shadow_discordance_total})")
        lines.append("-" * 40)

    # Adjudication-candidacy yield for severe tier with spatial discontinuity and absent LGE
    severe_reports = [r for r in result.stratum_reports if r.stratum_key.tier == "severe"]
    adj_candidates = 0
    for r in severe_reports:
        has_spatial_discontinuity = r.spatial.ring_break_persistence > 0 or r.spatial.layer_area_jump_persistence > 0
        has_no_lge_support = r.lge_support is None or r.lge_support.lge_presence_rate == 0.0
        if has_spatial_discontinuity and has_no_lge_support:
            adj_candidates += len(r.case_ids)

    lines.append(f"\nAdjudication-candidacy yield (severe, spatial discontinuity, absent LGE): {adj_candidates} ({adj_candidates / max(result.total_flagged, 1):.4f})")
    lines.append("=" * 80)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the audit."""
    result = run_audit()
    table = emit_stratum_table(result)
    print(table)


if __name__ == "__main__":
    main()
