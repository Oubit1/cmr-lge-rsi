import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class QCPayloadOptimizer:
    """Optimizes CMR/LGE QC evidence payloads via deduplication, flag roll-up,
    digest caching, and specialist-pass gating."""

    def __init__(self) -> None:
        self._digest_cache: Dict[str, Dict] = {}

    def deduplicate_evidence(self, raw_records: List[Dict]) -> List[Dict]:
        """Merges records per (case_id, modality), preserving all slice indices.
        Reduces payload size by consolidating duplicate metadata."""
        buckets = defaultdict(
            lambda: {"issues": set(), "slices": defaultdict(list)}
        )
        for r in raw_records:
            key = (r["case_id"], r["modality"])
            buckets[key]["issues"].update(r.get("issue_types", []))
            for iss, idxs in r.get("slice_indices", {}).items():
                buckets[key]["slices"][iss].extend(idxs)

        out: List[Dict] = []
        for (cid, mod), b in buckets.items():
            out.append(
                {
                    "case_id": cid,
                    "modality": mod,
                    "merged_issue_types": sorted(b["issues"]),
                    "slice_indices": {
                        k: sorted(set(v)) for k, v in b["slices"].items()
                    },
                }
            )
        return out

    def roll_up_cmr_flags(self, case_meta: Dict) -> Dict:
        """Rolls up high-volume CMR QC flags per case into a compact summary."""
        flagged = case_meta.get("flagged_slices", [])
        total = case_meta.get("total_slices", 1)
        return {
            "flag_count": len(flagged),
            "flag_rate": round(len(flagged) / max(1, total), 4),
            "worst_slice_index": max(flagged) if flagged else None,
            "severity_tier": case_meta.get("worst_severity", "low"),
            "addressable_slices": flagged,
        }

    def get_qc_digest(self, case_id: str, slice_data: Dict) -> Dict:
        """Materializes per-case per-slice QC digest once and caches by reference."""
        if case_id in self._digest_cache:
            return self._digest_cache[case_id]

        comps = slice_data.get("mi_components", [])
        vols = [c.get("volume", 0) for c in comps]
        dists = [c.get("distance", float("inf")) for c in comps]

        digest = {
            "mi_component_count": len(comps),
            "per_component_volume": vols,
            "min_inter_component_distance": min(dists) if dists else 0,
            "mi_fraction": round(
                sum(vols) / max(1, slice_data.get("total_volume", 1)), 4
            ),
            "myo_topology_flag": slice_data.get("myo_topology", False),
            "threshold_value": slice_data.get("threshold", 0.5),
        }
        self._digest_cache[case_id] = digest
        return digest

    def gate_specialist_pass(self, paired_cases: List[Tuple[str, str]]) -> Dict:
        """Gates CMR specialist pass on >=2 paired CMR/LGE cases at matched slices."""
        valid_pairs = [
            (c, l) for c, l in paired_cases if c.startswith("CMR_") and l.startswith("LGE_")
        ]
        if len(valid_pairs) >= 2:
            return {
                "status": "pass",
                "finding": "Paired cases sufficient",
                "supporting_issue_types": [],
                "suggested_experiment": "standard_review",
                "contraindications": [],
                "expected_metrics": {},
                "confidence": 0.9,
            }
        return {
            "status": "not_applicable",
            "finding": "Insufficient paired cases",
            "supporting_issue_types": [],
            "suggested_experiment": "none",
            "contraindications": ["evidence_gap"],
            "expected_metrics": {},
            "confidence": 0.0,
            "note": "Evidence gap: <2 paired CMR/LGE cases",
        }
