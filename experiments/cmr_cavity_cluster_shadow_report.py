"""Read-only shadow triage report for stratified real held-out pending CMR volumes.

Groups per-frame flags into three cavity-context clusters, records suppression
reasons (empty, border-touch, temporal-hole, centroid-drift) with counts and
denominators, and emits per-case counts plus a 95% Wilson CI for per-cluster
flag concordance. Never writes labels or code; concordance/CI are reported as
unavailable when no independent reference labels are supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CLUSTERS: Tuple[str, ...] = ("cavity_base", "cavity_mid", "cavity_apex")
SUPPRESSION_REASONS: Tuple[str, ...] = (
    "empty",
    "border_touch",
    "temporal_hole",
    "centroid_drift",
)


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Optional[Tuple[float, float]]:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _cluster_of(frame_index: int, n_frames: int) -> str:
    if n_frames <= 0:
        return CLUSTERS[0]
    third = max(1, n_frames // 3)
    if frame_index < third:
        return CLUSTERS[0]
    if frame_index < 2 * third:
        return CLUSTERS[1]
    return CLUSTERS[2]


@dataclass
class FrameFlag:
    frame_index: int
    flag: bool
    suppression_reason: Optional[str] = None


@dataclass
class VolumeInput:
    case_id: str
    n_frames: int
    frames: List[FrameFlag] = field(default_factory=list)
    reference_flags: Optional[List[bool]] = None


def _validate_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    if reason not in SUPPRESSION_REASONS:
        raise ValueError(f"unknown suppression reason: {reason!r}")
    return reason


def triage_volume(volume: VolumeInput) -> Dict[str, Any]:
    """Read-only per-volume triage. Returns per-frame flags and cluster counts."""
    n = volume.n_frames
    per_frame: List[Dict[str, Any]] = []
    cluster_counts: Dict[str, Dict[str, int]] = {
        c: {"flagged": 0, "total": 0} for c in CLUSTERS
    }
    suppression_counts: Dict[str, int] = {r: 0 for r in SUPPRESSION_REASONS}
    suppression_denom = 0

    for ff in volume.frames:
        if ff.frame_index < 0 or ff.frame_index >= n:
            raise ValueError(f"frame_index {ff.frame_index} out of range for {volume.case_id}")
        reason = _validate_reason(ff.suppression_reason)
        cluster = _cluster_of(ff.frame_index, n)
        cluster_counts[cluster]["total"] += 1
        if ff.flag:
            cluster_counts[cluster]["flagged"] += 1
        if reason is not None:
            suppression_counts[reason] += 1
            suppression_denom += 1
        per_frame.append(
            {
                "frame_index": ff.frame_index,
                "cluster": cluster,
                "flag": bool(ff.flag),
                "suppression_reason": reason,
            }
        )

    return {
        "case_id": volume.case_id,
        "n_frames": n,
        "per_frame": per_frame,
        "cluster_counts": cluster_counts,
        "suppression_counts": suppression_counts,
        "suppression_denominator": suppression_denom,
    }


def _concordance_for_volume(volume: VolumeInput) -> Optional[Dict[str, Any]]:
    if volume.reference_flags is None:
        return None
    if len(volume.reference_flags) != volume.n_frames:
        raise ValueError(f"reference_flags length mismatch for {volume.case_id}")
    by_cluster: Dict[str, Dict[str, int]] = {c: {"agree": 0, "total": 0} for c in CLUSTERS}
    for ff in volume.frames:
        cluster = _cluster_of(ff.frame_index, volume.n_frames)
        by_cluster[cluster]["total"] += 1
        if bool(ff.flag) == bool(volume.reference_flags[ff.frame_index]):
            by_cluster[cluster]["agree"] += 1
    out: Dict[str, Any] = {}
    for c in CLUSTERS:
        a = by_cluster[c]["agree"]
        t = by_cluster[c]["total"]
        out[c] = {"agree": a, "total": t, "ci95": _wilson_ci(a, t)}
    return out


def build_report(volumes: Sequence[VolumeInput]) -> Dict[str, Any]:
    """Build the read-only shadow report over the supplied held-out volumes."""
    per_case: List[Dict[str, Any]] = []
    agg_cluster: Dict[str, Dict[str, int]] = {c: {"flagged": 0, "total": 0} for c in CLUSTERS}
    agg_suppression: Dict[str, int] = {r: 0 for r in SUPPRESSION_REASONS}
    agg_suppression_denom = 0
    concordance_available = True
    agg_concordance: Dict[str, Dict[str, int]] = {c: {"agree": 0, "total": 0} for c in CLUSTERS}

    for v in volumes:
        tri = triage_volume(v)
        conc = _concordance_for_volume(v)
        if conc is None:
            concordance_available = False
        else:
            for c in CLUSTERS:
                agg_concordance[c]["agree"] += conc[c]["agree"]
                agg_concordance[c]["total"] += conc[c]["total"]
        for c in CLUSTERS:
            agg_cluster[c]["flagged"] += tri["cluster_counts"][c]["flagged"]
            agg_cluster[c]["total"] += tri["cluster_counts"][c]["total"]
        for r in SUPPRESSION_REASONS:
            agg_suppression[r] += tri["suppression_counts"][r]
        agg_suppression_denom += tri["suppression_denominator"]
        per_case.append(
            {
                "case_id": tri["case_id"],
                "n_frames": tri["n_frames"],
                "cluster_counts": tri["cluster_counts"],
                "suppression_counts": tri["suppression_counts"],
                "suppression_denominator": tri["suppression_denominator"],
                "concordance": conc,
            }
        )

    concordance_block: Dict[str, Any]
    if concordance_available and all(agg_concordance[c]["total"] > 0 for c in CLUSTERS):
        concordance_block = {
            "available": True,
            "per_cluster": {
                c: {
                    "agree": agg_concordance[c]["agree"],
                    "total": agg_concordance[c]["total"],
                    "ci95": _wilson_ci(agg_concordance[c]["agree"], agg_concordance[c]["total"]),
                }
                for c in CLUSTERS
            },
        }
    else:
        concordance_block = {
            "available": False,
            "reason": "no independent reference labels supplied; concordance/CI unavailable",
        }

    return {
        "mode": "read_only_shadow",
        "n_volumes": len(volumes),
        "clusters": list(CLUSTERS),
        "per_case": per_case,
        "aggregate_cluster_counts": agg_cluster,
        "suppression": {
            "counts": agg_suppression,
            "denominator": agg_suppression_denom,
        },
        "concordance": concordance_block,
        "writes": {"labels": 0, "code": 0},
    }


def module_hash() -> str:
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def audit_diff() -> str:
    return "read-only shadow report: no production label or code writes performed"


def rollback_command() -> str:
    return "git checkout -- experiments/cmr_cavity_cluster_shadow_report.py tests/test_cmr_cavity_cluster_shadow_report.py"


def report_with_provenance(volumes: Sequence[VolumeInput]) -> Dict[str, Any]:
    rep = build_report(volumes)
    rep["provenance"] = {
        "module_hash": module_hash(),
        "audit_diff": audit_diff(),
        "rollback_command": rollback_command(),
    }
    return rep


def _main(argv: Optional[Iterable[str]] = None) -> int:
    # CLI is intentionally inert: no dataset paths, no writes.
    print(json.dumps({"mode": "read_only_shadow", "note": "import and call build_report"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
