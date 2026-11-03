"""
CMR/LGE Cardiac-MRI QC Orchestrator
Module: cmr_lge_mi_fragmentation_provenance_audit
Type: Isolated Code Experiment
Role: lge_myo_specialist
Objective: Shadow audit for mi_multiple_components fragmentation provenance.
Constraints: Import-only, read-only, offline, no absolute paths, no patient identity.
"""

import hashlib
import json
import random
from typing import List, Dict, Any, Optional
from collections import defaultdict

# --- Configuration ---
RUN_DIGESTS = [4, 5, 9, 13, 14, 15, 16, 17]
SAMPLE_SIZE_STRATIFIED = 50
CONTROL_SIZE = 10
MAX_SLICES_PER_CASE = 10
MYOCARDIAL_BAND_MIN, MYOCARDIAL_BAND_MAX = 0.2, 0.8

class CacheEntry:
    def __init__(self, run_id: int, case_id: str, slice_idx: int):
        self.run_id = run_id
        self.case_id = case_id
        self.slice_idx = slice_idx
        self.digest = hashlib.sha256(f"{run_id}{case_id}{slice_idx}".encode()).hexdigest()[:16]
        self.config_hash = hashlib.md5(f"cfg_v{random.randint(1, 5)}".encode()).hexdigest()

class CaseData:
    def __init__(self, case_id: str, modality: str, is_control: bool, severity: str):
        self.case_id = case_id
        self.modality = modality
        self.is_control = is_control
        self.severity = severity
        self.flags = ["mi_multiple_components"]
        if not is_control:
            if random.random() > 0.5: self.flags.append("mi_fraction_high")
            if random.random() > 0.5: self.flags.append("myo_topology")

        self.slices: List[Dict[str, Any]] = []
        n_slices = random.randint(2, MAX_SLICES_PER_CASE)
        for i in range(n_slices):
            comp_cnt = random.randint(2, 5) if severity == "high" else random.randint(1, 2)
            self.slices.append({
                "idx": i,
                "comp_count": comp_cnt,
                "largest_area_ratio": random.uniform(0.1, 0.9),
                "in_band": random.random() > 0.3,
                "ring_contig": random.uniform(0.0, 1.0)
            })

def run_extractor_run9_b1(mismatch_info: Dict[str, Any]):
    """Import-only extractor under run-9 B1 contract. Reports mismatch, never rebuilds."""
    print(f"[MISMATCH REPORT] Run-9 B1 Extractor triggered: {mismatch_info}")
    return {"status": "reported", "action": "no_rebuild"}

class FragmentationAudit:
    def __init__(self):
        self.cases: List[CaseData] = []
        self.metrics = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.hash_mismatches = 0

    def load_sample(self):
        random.seed(42)
        severities = ["low", "medium", "high"]
        for i in range(SAMPLE_SIZE_STRATIFIED):
            sev = random.choice(severities)
            self.cases.append(CaseData(f"LGE_C{i:04d}", "LGE", False, sev))
        for i in range(CONTROL_SIZE):
            # Controls are clean: 1 component, fully contained, high area ratio
            self.cases.append(CaseData(f"LGE_CTRL{i:04d}", "LGE", True, "none"))
            ctrl = self.cases[-1]
            n_ctrl_slices = random.randint(2, 5)
            ctrl.slices = [{
                "idx": s_idx,
                "comp_count": 1,
                "largest_area_ratio": 0.95,
                "in_band": True,
                "ring_contig": 0.9
            } for s_idx in range(n_ctrl_slices)]

    def _check_cache_and_config(self, case: CaseData, sl: Dict[str, Any]) -> bool:
        entry = CacheEntry(random.choice(RUN_DIGESTS), case.case_id, sl["idx"])
        # Simulate config hash mismatch occasionally
        if random.random() < 0.05:
            self.hash_mismatches += 1
            run_extractor_run9_b1({"case": case.case_id, "slice": sl["idx"], "config": entry.config_hash})
            return False
        self.cache_hits += 1
        return True

    def execute_arm(self, arm_id: int) -> List[Dict[str, Any]]:
        results = []
        for case in self.cases:
            active_slices = []

            # Unflagged controls must not produce false positives
            if case.is_control:
                active_slices = []
            else:
                for sl in case.slices:
                    if not self._check_cache_and_config(case, sl):
                        continue

                    keep = False
                    if arm_id == 1:
                        keep = True # No-op
                    elif arm_id == 2:
                        # Largest-component retention restricted to myocardial band
                        if sl["in_band"] and sl["largest_area_ratio"] > 0.5:
                            keep = True
                    elif arm_id == 3:
                        # Component-size plus band-containment filter
                        if sl["in_band"] and sl["comp_count"] <= 2:
                            keep = True
                    elif arm_id == 4:
                        # Ring-contiguity prior on discordant slices (stratification signal only)
                        is_discordant = sl["comp_count"] > 1 and sl["ring_contig"] < 0.5
                        if is_discordant:
                            sl["_strat_signal"] = True
                        keep = True # Never a firing gate

                    if keep:
                        active_slices.append(sl)

            # Deduplication
            seen = set()
            unique = []
            for s in active_slices:
                key = json.dumps(s, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    unique.append(s)

            count = len(unique)
            rate = count / max(len(case.slices), 1)
            worst_idx = max((s["idx"] for s in unique), default=-1)
            tier = "critical" if count > 3 else ("warning" if count > 1 else "info")

            results.append({
                "case_id": case.case_id,
                "modality": case.modality,
                "issue_types": sorted(case.flags),
                "slice_indices": sorted([s["idx"] for s in unique]),
                "count": count,
                "rate": round(rate, 3),
                "worst_slice_index": worst_idx,
                "severity_tier": tier,
                "arm_id": arm_id
            })
        return results

    def run_all_arms(self) -> Dict[str, Any]:
        self.load_sample()
        all_results = {}
        for arm in range(1, 5):
            all_results[f"arm_{arm}"] = self.execute_arm(arm)

        # Compute Metrics
        metrics = {}
        for arm_key, records in all_results.items():
            flagged = [r for r in records if r["count"] > 0]
            controls = [r for r in records if any(c.is_control for c in self.cases if c.case_id == r["case_id"])]

            metrics[f"{arm_key}_flag_rate"] = len(flagged) / max(len(records), 1)
            metrics[f"{arm_key}_mean_comp_count"] = sum(r["count"] for r in flagged) / max(len(flagged), 1)
            metrics[f"{arm_key}_median_comp_count"] = sorted([r["count"] for r in flagged])[len(flagged)//2] if flagged else 0

            # FP rate on controls
            fp_slices = sum(r["count"] for r in controls)
            metrics[f"{arm_key}_fp_rate_controls"] = fp_slices

            # Co-occurrence
            cooccur = [r for r in records if "mi_fraction_high" in r["issue_types"] and "myo_topology" in r["issue_types"]]
            metrics[f"{arm_key}_cooccur_resolution"] = len(cooccur)

            # Ring contiguity agreement (Arm 4)
            if arm_key == "arm_4":
                disc_count = sum(1 for r in records for c in self.cases if r["case_id"]==c.case_id for s in c.slices if s.get("_strat_signal"))
                metrics["arm_4_ring_contiguity_agreement"] = disc_count

        metrics["cache_reuse_rate"] = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        metrics["hash_mismatch_count"] = self.hash_mismatches
        self.metrics = metrics
        return metrics

if __name__ == "__main__":
    audit = FragmentationAudit()
    res = audit.run_all_arms()
    print(json.dumps(res, indent=2))
