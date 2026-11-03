"""CMR S1 stratum analysis: review_centroid_outlier persistence and attribution.

This module executes the CMR stratum within S1 only, reusing the S1 cached digest
and shadow arms. It does not include LGE cases or MI fragmentation flags.

Attribution targets for review_centroid_outlier persistence:
- cmr_myo_small_area
- cmr_myo_excess_components
- cmr_myo_touches_image_border
- myo_topology
- cmr_centroid_drift
- cmr_layer_area_jump
- temporal instability (residual)

Reports:
- Pre/post temporal centroid coefficient of variation
- Cross-arm agreement (no-op, largest-component retention, temporal continuity)
- Co-occurrence rates for mask-instability phenotypes
- Attribution fraction for mask-instability phenotypes vs unexplained residual

No production writes, no label edits, no rule activation, no new thresholds.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class S1DigestRecord:
    study_id: str
    arm: str  # 'noop', 'largest_component', 'temporal_continuity'
    review_centroid_outlier: bool
    cmr_myo_small_area: bool = False
    cmr_myo_excess_components: bool = False
    cmr_myo_touches_image_border: bool = False
    myo_topology: bool = False
    cmr_centroid_drift: bool = False
    cmr_layer_area_jump: bool = False
    temporal_instability: bool = False
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    centroid_z: float = 0.0


@dataclasses.dataclass(frozen=True)
class StratumReport:
    total_cmr_cases: int
    outlier_persistence_noop: float
    outlier_persistence_largest: float
    outlier_persistence_temporal: float
    cooccurrence_rates: dict[str, float]
    temporal_centroid_cv_pre: float
    temporal_centroid_cv_post: float
    cross_arm_agreement: dict[str, float]
    attribution_fractions: dict[str, float]
    unexplained_residual: float


# ---------------------------------------------------------------------------
# Shadow arm loaders (stubbed for offline execution)
# ---------------------------------------------------------------------------

def _load_shadow_arm(arm_name: str) -> list[dict[str, Any]]:
    """Load cached shadow arm records. Stubbed for offline use."""
    # In production, reads from S1 cached digest path.
    # Here we return empty list; tests inject data directly.
    return []


def _is_cmr_case(record: dict[str, Any]) -> bool:
    """Return True if record is CMR (not LGE, no MI fragmentation)."""
    modality = record.get("modality", "CMR")
    mi_frag = record.get("mi_fragmentation_flag", False)
    return modality == "CMR" and not mi_frag


def _filter_cmr_stratum(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if _is_cmr_case(r)]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _parse_records(records: list[dict[str, Any]]) -> list[S1DigestRecord]:
    parsed: list[S1DigestRecord] = []
    for r in records:
        parsed.append(
            S1DigestRecord(
                study_id=r.get("study_id", ""),
                arm=r.get("arm", "noop"),
                review_centroid_outlier=bool(r.get("review_centroid_outlier", False)),
                cmr_myo_small_area=bool(r.get("cmr_myo_small_area", False)),
                cmr_myo_excess_components=bool(r.get("cmr_myo_excess_components", False)),
                cmr_myo_touches_image_border=bool(r.get("cmr_myo_touches_image_border", False)),
                myo_topology=bool(r.get("myo_topology", False)),
                cmr_centroid_drift=bool(r.get("cmr_centroid_drift", False)),
                cmr_layer_area_jump=bool(r.get("cmr_layer_area_jump", False)),
                temporal_instability=bool(r.get("temporal_instability", False)),
                centroid_x=float(r.get("centroid_x", 0.0)),
                centroid_y=float(r.get("centroid_y", 0.0)),
                centroid_z=float(r.get("centroid_z", 0.0)),
            )
        )
    return parsed


def _compute_outlier_persistence(
    records: list[S1DigestRecord], arm: str
) -> float:
    arm_records = [r for r in records if r.arm == arm]
    if not arm_records:
        return 0.0
    outliers = [r for r in arm_records if r.review_centroid_outlier]
    return len(outliers) / len(arm_records)


def _compute_cooccurrence_rates(records: list[S1DigestRecord]) -> dict[str, float]:
    total = len(records)
    if total == 0:
        return {
            "cmr_myo_small_area": 0.0,
            "cmr_myo_excess_components": 0.0,
            "cmr_myo_touches_image_border": 0.0,
            "myo_topology": 0.0,
            "cmr_centroid_drift": 0.0,
        }
    return {
        "cmr_myo_small_area": sum(1 for r in records if r.cmr_myo_small_area) / total,
        "cmr_myo_excess_components": sum(1 for r in records if r.cmr_myo_excess_components) / total,
        "cmr_myo_touches_image_border": sum(1 for r in records if r.cmr_myo_touches_image_border) / total,
        "myo_topology": sum(1 for r in records if r.myo_topology) / total,
        "cmr_centroid_drift": sum(1 for r in records if r.cmr_centroid_drift) / total,
    }


def _compute_temporal_centroid_cv(records: list[S1DigestRecord]) -> tuple[float, float]:
    """Return (pre_cv, post_cv) for temporal centroid stability.

    Pre: no-op arm (raw).
    Post: temporal_continuity arm (repaired).
    """
    def _cv(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return math.sqrt(variance) / abs(mean)

    pre = [r.centroid_x + r.centroid_y + r.centroid_z for r in records if r.arm == "noop"]
    post = [r.centroid_x + r.centroid_y + r.centroid_z for r in records if r.arm == "temporal_continuity"]
    return _cv(pre), _cv(post)


def _compute_cross_arm_agreement(records: list[S1DigestRecord]) -> dict[str, float]:
    """Report agreement between arms for outlier flag."""
    study_map: dict[str, dict[str, bool]] = {}
    for r in records:
        study_map.setdefault(r.study_id, {})[r.arm] = r.review_centroid_outlier

    agreements: dict[str, float] = {}
    arms = ["noop", "largest_component", "temporal_continuity"]
    for i, a1 in enumerate(arms):
        for a2 in arms[i + 1 :]:
            matches = 0
            total = 0
            for study, flags in study_map.items():
                if a1 in flags and a2 in flags:
                    total += 1
                    if flags[a1] == flags[a2]:
                        matches += 1
            agreements[f"{a1}_vs_{a2}"] = matches / total if total else 0.0
    return agreements


def _compute_attribution_fractions(records: list[S1DigestRecord]) -> dict[str, Any]:
    """Attribute outlier persistence to mask-instability phenotypes."""
    outlier_records = [r for r in records if r.review_centroid_outlier]
    total_outliers = len(outlier_records)
    if total_outliers == 0:
        return {
            "cmr_myo_small_area": 0.0,
            "cmr_myo_excess_components": 0.0,
            "cmr_myo_touches_image_border": 0.0,
            "myo_topology": 0.0,
            "cmr_centroid_drift": 0.0,
            "cmr_layer_area_jump": 0.0,
            "temporal_instability": 0.0,
            "unexplained_residual": 1.0,
        }

    phenotype_counts: dict[str, int] = {
        "cmr_myo_small_area": 0,
        "cmr_myo_excess_components": 0,
        "cmr_myo_touches_image_border": 0,
        "myo_topology": 0,
        "cmr_centroid_drift": 0,
        "cmr_layer_area_jump": 0,
        "temporal_instability": 0,
    }

    for r in outlier_records:
        if r.cmr_myo_small_area:
            phenotype_counts["cmr_myo_small_area"] += 1
        if r.cmr_myo_excess_components:
            phenotype_counts["cmr_myo_excess_components"] += 1
        if r.cmr_myo_touches_image_border:
            phenotype_counts["cmr_myo_touches_image_border"] += 1
        if r.myo_topology:
            phenotype_counts["myo_topology"] += 1
        if r.cmr_centroid_drift:
            phenotype_counts["cmr_centroid_drift"] += 1
        if r.cmr_layer_area_jump:
            phenotype_counts["cmr_layer_area_jump"] += 1
        if r.temporal_instability:
            phenotype_counts["temporal_instability"] += 1

    # Simple additive attribution (phenotypes can co-occur)
    total_phenotype_hits = sum(phenotype_counts.values())
    # Avoid division by zero; if no phenotypes, all residual
    if total_phenotype_hits == 0:
        fractions = {k: 0.0 for k in phenotype_counts}
        fractions["unexplained_residual"] = 1.0
        return fractions

    fractions = {k: v / total_phenotype_hits for k, v in phenotype_counts.items()}
    # Residual: proportion of outliers not explained by any phenotype
    explained = sum(
        1 for r in outlier_records
        if any([
            r.cmr_myo_small_area,
            r.cmr_myo_excess_components,
            r.cmr_myo_touches_image_border,
            r.myo_topology,
            r.cmr_centroid_drift,
            r.cmr_layer_area_jump,
            r.temporal_instability,
        ])
    )
    fractions["unexplained_residual"] = (total_outliers - explained) / total_outliers if total_outliers else 0.0
    return fractions


def run_s1_cmr_stratum_analysis(
    raw_records: list[dict[str, Any]] | None = None,
) -> StratumReport:
    """Execute CMR-only S1 stratum analysis.

    Args:
        raw_records: Optional injected records for testing. If None, attempts
            to load from cached shadow arms (stubbed).

    Returns:
        StratumReport with all required metrics.
    """
    if raw_records is None:
        # Production path: load from cached digest (stubbed)
        all_records: list[dict[str, Any]] = []
        for arm in ["noop", "largest_component", "temporal_continuity"]:
            all_records.extend(_load_shadow_arm(arm))
    else:
        all_records = raw_records

    # Filter to CMR stratum only
    cmr_records = _filter_cmr_stratum(all_records)
    records = _parse_records(cmr_records)

    total_cmr_cases = len(records)
    outlier_persistence_noop = _compute_outlier_persistence(records, "noop")
    outlier_persistence_largest = _compute_outlier_persistence(records, "largest_component")
    outlier_persistence_temporal = _compute_outlier_persistence(records, "temporal_continuity")
    cooccurrence_rates = _compute_cooccurrence_rates(records)
    temporal_cv_pre, temporal_cv_post = _compute_temporal_centroid_cv(records)
    cross_arm_agreement = _compute_cross_arm_agreement(records)
    attribution = _compute_attribution_fractions(records)

    # Extract phenotype fractions and residual
    phenotype_keys = [
        "cmr_myo_small_area",
        "cmr_myo_excess_components",
        "cmr_myo_touches_image_border",
        "myo_topology",
        "cmr_centroid_drift",
        "cmr_layer_area_jump",
        "temporal_instability",
    ]
    attribution_fractions = {k: attribution[k] for k in phenotype_keys}
    unexplained_residual = attribution.get("unexplained_residual", 0.0)

    return StratumReport(
        total_cmr_cases=total_cmr_cases,
        outlier_persistence_noop=outlier_persistence_noop,
        outlier_persistence_largest=outlier_persistence_largest,
        outlier_persistence_temporal=outlier_persistence_temporal,
        cooccurrence_rates=cooccurrence_rates,
        temporal_centroid_cv_pre=temporal_cv_pre,
        temporal_centroid_cv_post=temporal_cv_post,
        cross_arm_agreement=cross_arm_agreement,
        attribution_fractions=attribution_fractions,
        unexplained_residual=unexplained_residual,
    )


def report_to_dict(report: StratumReport) -> dict[str, Any]:
    """Serialize report to dict for JSON output."""
    return {
        "total_cmr_cases": report.total_cmr_cases,
        "outlier_persistence": {
            "noop": report.outlier_persistence_noop,
            "largest_component": report.outlier_persistence_largest,
            "temporal_continuity": report.outlier_persistence_temporal,
        },
        "cooccurrence_rates": report.cooccurrence_rates,
        "temporal_centroid_cv": {
            "pre": report.temporal_centroid_cv_pre,
            "post": report.temporal_centroid_cv_post,
        },
        "cross_arm_agreement": report.cross_arm_agreement,
        "attribution_fractions": report.attribution_fractions,
        "unexplained_residual": report.unexplained_residual,
    }


if __name__ == "__main__":
    report = run_s1_cmr_stratum_analysis()
    print(json.dumps(report_to_dict(report), indent=2))
