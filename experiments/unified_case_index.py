import logging
from typing import Dict, List, Set, Any

logger = logging.getLogger(__name__)


class UnifiedCaseIndex:
    """
    Replaces iterative per-stratum resampling with a single unified case index.
    Pre-maps co-occurrence flags so the same slice is not resampled for different strata.
    """

    def __init__(self):
        self.index: Dict[str, Dict[str, Any]] = {}

    def build_index(self, slices_data: List[Dict[str, Any]]):
        """
        Builds the unified index from raw slice data.
        """
        for item in slices_data:
            case_id = item["case_id"]
            slice_idx = item["slice_index"]
            stratum = item.get("stratum", "default")
            flags = item.get("flags", [])

            if case_id not in self.index:
                self.index[case_id] = {"slices": {}, "co_occurrence": {}}

            if slice_idx not in self.index[case_id]["slices"]:
                self.index[case_id]["slices"][slice_idx] = {
                    "strata": set(),
                    "flags": set(),
                }

            self.index[case_id]["slices"][slice_idx]["strata"].add(stratum)
            self.index[case_id]["slices"][slice_idx]["flags"].update(flags)

            # Track co-occurrence
            for flag in flags:
                if flag not in self.index[case_id]["co_occurrence"]:
                    self.index[case_id]["co_occurrence"][flag] = set()
                self.index[case_id]["co_occurrence"][flag].add(slice_idx)

    def get_sampling_plan(self, case_id: str) -> Dict[str, Any]:
        """
        Returns a sampling plan ensuring no slice is resampled for different strata.
        """
        if case_id not in self.index:
            return {"plan": [], "skipped": []}

        case_data = self.index[case_id]
        plan = []
        skipped = []
        assigned_slices: Set[int] = set()
        stratum_slices: Dict[str, List[int]] = {}

        for slice_idx, info in case_data["slices"].items():
            for stratum in info["strata"]:
                if stratum not in stratum_slices:
                    stratum_slices[stratum] = []

                if slice_idx not in assigned_slices:
                    stratum_slices[stratum].append(slice_idx)
                    assigned_slices.add(slice_idx)
                else:
                    skipped.append(
                        {
                            "slice": slice_idx,
                            "stratum": stratum,
                            "reason": "Already assigned to another stratum",
                        }
                    )

        plan = [{"stratum": s, "slices": sl} for s, sl in stratum_slices.items()]
        return {"plan": plan, "skipped": skipped}
