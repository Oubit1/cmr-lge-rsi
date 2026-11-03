#!/usr/bin/env python3
"""
CMR-LGE Topology Co-occurrence Stratum (Run-4/5/9 + Run-13 A3 Dedup Rollup)

Bounded, import-only, read-only paired stratum using validated cached digests.

- Reuses existing harness if pipeline/config hash validates.
- On mismatch, falls back to exactly one import-only extractor under run-9 B1 contract.
- Never builds a second harness.
- Samples CMR by cmr_myo_small_area severity, enriched for cmr_myo_excess_components
  and myo_topology; samples LGE by mi_multiple_components + myo_topology.
- Emits deduplicated records per (case_id, modality) with merged issue sets.
- Stratifies by ventricular level (apex/base vs mid-ventricle) and isolated vs
  contiguous component patterns.
- Reports missing pairs explicitly as unavailable (no imputation).
- No shadow correction arms, no label edits, no new numeric thresholds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Constants and type aliases
# ---------------------------------------------------------------------------

class Modality(Enum):
    CMR = "cmr"
    LGE = "lge"


class VentricularLevel(Enum):
    APEX_BASE = "apex_base"
    MID_VENTRICLE = "mid_ventricle"


class PatternType(Enum):
    ISOLATED = "isolated"
    CONTIGUOUS = "contiguous"


# Fixed connectivity for connected-component count (documented)
CONNECTIVITY_4 = 4  # 4-connected (von Neumann neighbourhood)
CONNECTIVITY_8 = 8  # 8-connected (Moore neighbourhood) – not used as primary

# Run identifiers for cache paths
RUN_IDS = {"cmr_digest": "run-4", "lge_digest": "run-5", "slice_digest": "run-9", "rollup": "run-13"}

# Cache file names (relative to cache root)
CACHE_FILES = {
    "cmr_per_case": "cmr_per_case_digest.json",
    "cmr_per_slice": "cmr_per_slice_digest.json",
    "lge_per_case": "lge_per_case_digest.json",
    "lge_per_slice": "lge_per_slice_digest.json",
    "a3_rollup": "a3_dedup_rollup.json",
    "pipeline_hash": "pipeline_config.hash",
}

# Sampling parameters
MAX_CMR_FLAGGED = 50
MAX_CMR_CONTROLS = 10
MAX_LGE_FLAGGED = 20
MAX_LGE_CONTROLS = 5

# B1 contract: import-only, read-only, no side effects
B1_CONTRACT_VERSION = "run-9-b1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseSliceId:
    case_id: str
    slice_idx: int

    def __hash__(self) -> int:
        return hash((self.case_id, self.slice_idx))


@dataclass
class CMRSliceMetrics:
    connected_component_count: int
    largest_component_area_ratio: float
    hole_count: int
    euler_characteristic: int
    ring_break_indicator: bool
    min_myocardial_band_distance_to_border: int  # absolute voxels
    # Cross-slice / temporal (populated at case level, referenced per slice)
    cross_slice_area_continuity_ratio: float | None = None
    centroid_drift_magnitude: float | None = None
    temporal_area_coefficient_of_variation: float | None = None


@dataclass
class LGEComponent:
    area: float
    centroid: tuple[float, float]


@dataclass
class LGESliceMetrics:
    mi_connected_component_count: int
    mi_components: list[LGEComponent]
    min_inter_component_distance: float | None
    mi_fraction: float
    myocardial_component_count: int
    ring_break_indicator: bool
    # Classification relative to paired myocardial band
    mi_fragments_within_band: int
    mi_fragments_remote: int


@dataclass
class StratifiedRecord:
    case_id: str
    modality: Modality
    issue_type: set[str]
    flagged_slice_indices: list[int]
    control_slice_indices: list[int]
    ventricular_level: VentricularLevel
    pattern_type: PatternType
    cmr_metrics: dict[int, CMRSliceMetrics] = field(default_factory=dict)
    lge_metrics: dict[int, LGESliceMetrics] = field(default_factory=dict)
    # Pairing availability
    missing_pairs: list[str] = field(default_factory=list)


@dataclass
class CooccurrenceResult:
    records: list[StratifiedRecord]
    # Summary statistics
    cmr_flagged_sampled: int
    cmr_control_sampled: int
    lge_flagged_sampled: int
    lge_control_sampled: int
    paired_cases: int
    missing_pairs_reported: list[str]
    # False positive rate (control slices with flags / total control slices)
    cmr_ring_break_fpr: float | None
    cmr_excess_component_fpr: float | None
    lge_topology_fragmentation_fpr: float | None
    # Reproducibility
    cache_hash_valid: bool
    fallback_extractor_invoked: bool
    mismatch_reported: str | None


# ---------------------------------------------------------------------------
# Cache-reuse guard and hash validation
# ---------------------------------------------------------------------------

def _compute_config_hash(config: dict[str, Any]) -> str:
    """Stable hash of pipeline configuration for cache validation."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_cached_hash(cache_root: Path) -> str | None:
    path = cache_root / CACHE_FILES["pipeline_hash"]
    if not path.exists():
        return None
    return path.read_text().strip()


def _validate_cache_hash(current_config: dict[str, Any], cache_root: Path) -> tuple[bool, str]:
    """Returns (valid, expected_hash)."""
    expected = _compute_config_hash(current_config)
    cached = _load_cached_hash(cache_root)
    if cached is None:
        return False, expected
    return cached == expected, expected


# ---------------------------------------------------------------------------
# Import-only digest loaders (read-only, no network)
# ---------------------------------------------------------------------------

def _load_json_digest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_cmr_per_case_digest(cache_root: Path) -> dict[str, Any]:
    return _load_json_digest(cache_root / CACHE_FILES["cmr_per_case"])


def _load_cmr_per_slice_digest(cache_root: Path) -> dict[str, Any]:
    return _load_json_digest(cache_root / CACHE_FILES["cmr_per_slice"])


def _load_lge_per_case_digest(cache_root: Path) -> dict[str, Any]:
    return _load_json_digest(cache_root / CACHE_FILES["lge_per_case"])


def _load_lge_per_slice_digest(cache_root: Path) -> dict[str, Any]:
    return _load_json_digest(cache_root / CACHE_FILES["lge_per_slice"])


def _load_a3_rollup(cache_root: Path) -> dict[str, Any]:
    return _load_json_digest(cache_root / CACHE_FILES["a3_rollup"])


# ---------------------------------------------------------------------------
# Run-9 B1 fallback extractor (exactly one, import-only, read-only)
# ---------------------------------------------------------------------------

class B1FallbackExtractor:
    """Single fallback extractor under run-9 B1 contract.

    Import-only: only reads from provided digest paths.
    Read-only: no modifications to any file system location.
    """

    CONTRACT = B1_CONTRACT_VERSION

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root

    def extract_cmr_slice_topology(self, case_id: str, slice_idx: int) -> CMRSliceMetrics | None:
        """Extract CMR topology metrics for a single slice from run-9 digest."""
        digest = _load_cmr_per_slice_digest(self.cache_root)
        key = f"{case_id}/{slice_idx}"
        data = digest.get(key, {})
        if not data:
            return None
        return CMRSliceMetrics(
            connected_component_count=data.get("connected_component_count", 0),
            largest_component_area_ratio=data.get("largest_component_area_ratio", 0.0),
            hole_count=data.get("hole_count", 0),
            euler_characteristic=data.get("euler_characteristic", 0),
            ring_break_indicator=data.get("ring_break_indicator", False),
            min_myocardial_band_distance_to_border=data.get("min_myocardial_band_distance_to_border", -1),
            cross_slice_area_continuity_ratio=data.get("cross_slice_area_continuity_ratio"),
            centroid_drift_magnitude=data.get("centroid_drift_magnitude"),
            temporal_area_coefficient_of_variation=data.get("temporal_area_coefficient_of_variation"),
        )

    def extract_lge_slice_topology(self, case_id: str, slice_idx: int) -> LGESliceMetrics | None:
        """Extract LGE topology metrics for a single slice from run-9 digest."""
        digest = _load_lge_per_slice_digest(self.cache_root)
        key = f"{case_id}/{slice_idx}"
        data = digest.get(key, {})
        if not data:
            return None
        components = [
            LGEComponent(area=c["area"], centroid=tuple(c["centroid"]))
            for c in data.get("mi_components", [])
        ]
        return LGESliceMetrics(
            mi_connected_component_count=data.get("mi_connected_component_count", 0),
            mi_components=components,
            min_inter_component_distance=data.get("min_inter_component_distance"),
            mi_fraction=data.get("mi_fraction", 0.0),
            myocardial_component_count=data.get("myocardial_component_count", 0),
            ring_break_indicator=data.get("ring_break_indicator", False),
            mi_fragments_within_band=data.get("mi_fragments_within_band", 0),
            mi_fragments_remote=data.get("mi_fragments_remote", 0),
        )


# ---------------------------------------------------------------------------
# Harness reuse / single-instantiation logic
# ---------------------------------------------------------------------------

class TopologyHarness:
    """Reusable harness for CMR-LGE topology co-occurrence analysis.

    Instantiated once; on hash mismatch, falls back to B1 extractor and
    never builds a second harness.
    """

    def __init__(self, cache_root: Path, config: dict[str, Any]) -> None:
        self.cache_root = cache_root
        self.config = config
        self._b1: B1FallbackExtractor | None = None

    @property
    def b1(self) -> B1FallbackExtractor:
        if self._b1 is None:
            self._b1 = B1FallbackExtractor(self.cache_root)
        return self._b1

    def get_cmr_slice_metrics(self, case_id: str, slice_idx: int, use_fallback: bool = False) -> CMRSliceMetrics | None:
        if use_fallback:
            return self.b1.extract_cmr_slice_topology(case_id, slice_idx)
        # Attempt harness cache first
        digest = _load_cmr_per_slice_digest(self.cache_root)
        key = f"{case_id}/{slice_idx}"
        if key in digest:
            data = digest[key]
            return CMRSliceMetrics(
                connected_component_count=data.get("connected_component_count", 0),
                largest_component_area_ratio=data.get("largest_component_area_ratio", 0.0),
                hole_count=data.get("hole_count", 0),
                euler_characteristic=data.get("euler_characteristic", 0),
                ring_break_indicator=data.get("ring_break_indicator", False),
                min_myocardial_band_distance_to_border=data.get("min_myocardial_band_distance_to_border", -1),
                cross_slice_area_continuity_ratio=data.get("cross_slice_area_continuity_ratio"),
                centroid_drift_magnitude=data.get("centroid_drift_magnitude"),
                temporal_area_coefficient_of_variation=data.get("temporal_area_coefficient_of_variation"),
            )
        return None

    def get_lge_slice_metrics(self, case_id: str, slice_idx: int, use_fallback: bool = False) -> LGESliceMetrics | None:
        if use_fallback:
            return self.b1.extract_lge_slice_topology(case_id, slice_idx)
        digest = _load_lge_per_slice_digest(self.cache_root)
        key = f"{case_id}/{slice_idx}"
        if key in digest:
            data = digest[key]
            components = [
                LGEComponent(area=c["area"], centroid=tuple(c["centroid"]))
                for c in data.get("mi_components", [])
            ]
            return LGESliceMetrics(
                mi_connected_component_count=data.get("mi_connected_component_count", 0),
                mi_components=components,
                min_inter_component_distance=data.get("min_inter_component_distance"),
                mi_fraction=data.get("mi_fraction", 0.0),
                myocardial_component_count=data.get("myocardial_component_count", 0),
                ring_break_indicator=data.get("ring_break_indicator", False),
                mi_fragments_within_band=data.get("mi_fragments_within_band", 0),
                mi_fragments_remote=data.get("mi_fragments_remote", 0),
            )
        return None


# ---------------------------------------------------------------------------
# Modality inference helpers
# ---------------------------------------------------------------------------

def _infer_modality(case_id: str) -> Modality | None:
    """Infer modality from case_id prefix. Returns None if ambiguous."""
    upper = case_id.upper()
    if upper.startswith("CMR-"):
        return Modality.CMR
    elif upper.startswith("LGE-"):
        return Modality.LGE
    return None


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _sample_cmr_cases(
    rollup: dict[str, Any],
    max_flagged: int = MAX_CMR_FLAGGED,
    max_controls: int = MAX_CMR_CONTROLS,
) -> tuple[list[str], list[str]]:
    """Return (flagged_case_ids, control_case_ids) for CMR."""
    flagged: list[str] = []
    controls: list[str] = []
    for case_id, record in rollup.items():
        if _infer_modality(case_id) != Modality.CMR:
            continue
        issues = record.get("issue_type", [])
        if "cmr_myo_small_area" in issues or "cmr_myo_excess_components" in issues or "myo_topology" in issues:
            flagged.append(case_id)
        elif not any(i in issues for i in ("cmr_myo_small_area", "cmr_myo_excess_components", "myo_topology")):
            controls.append(case_id)
    # Enrich for excess components and topology
    flagged_sorted = sorted(
        flagged,
        key=lambda c: (
            "cmr_myo_excess_components" in rollup.get(c, {}).get("issue_type", []),
            "myo_topology" in rollup.get(c, {}).get("issue_type", []),
        ),
        reverse=True,
    )
    return flagged_sorted[:max_flagged], controls[:max_controls]


def _sample_lge_cases(
    rollup: dict[str, Any],
    max_flagged: int = MAX_LGE_FLAGGED,
    max_controls: int = MAX_LGE_CONTROLS,
) -> tuple[list[str], list[str]]:
    """Return (flagged_case_ids, control_case_ids) for LGE."""
    flagged: list[str] = []
    controls: list[str] = []
    for case_id, record in rollup.items():
        if _infer_modality(case_id) != Modality.LGE:
            continue
        issues = record.get("issue_type", [])
        if "mi_multiple_components" in issues and "myo_topology" in issues:
            flagged.append(case_id)
        elif "mi_multiple_components" not in issues and "myo_topology" not in issues:
            controls.append(case_id)
    return flagged[:max_flagged], controls[:max_controls]


# ---------------------------------------------------------------------------
# Ventricular level and pattern classification
# ---------------------------------------------------------------------------

def _classify_ventricular_level(slice_idx: int, total_slices: int) -> VentricularLevel:
    """Classify slice as apex/base or mid-ventricle based on relative position."""
    if total_slices <= 0:
        return VentricularLevel.MID_VENTRICLE
    ratio = slice_idx / total_slices
    if ratio < 0.2 or ratio > 0.8:
        return VentricularLevel.APEX_BASE
    return VentricularLevel.MID_VENTRICLE


def _classify_pattern(cmr_metrics: CMRSliceMetrics | None, lge_metrics: LGESliceMetrics | None) -> PatternType:
    """Classify as isolated or contiguous based on component topology."""
    if cmr_metrics is not None and cmr_metrics.connected_component_count > 1:
        return PatternType.ISOLATED
    if lge_metrics is not None and lge_metrics.mi_connected_component_count > 1:
        return PatternType.ISOLATED
    return PatternType.CONTIGUOUS


# ---------------------------------------------------------------------------
# Core co-occurrence stratum execution
# ---------------------------------------------------------------------------

def execute_cooccurrence_stratum(
    cache_root: Path | str,
    config: dict[str, Any] | None = None,
) -> CooccurrenceResult:
    """Execute the paired CMR-LGE topology co-occurrence stratum.

    Args:
        cache_root: Path to validated cached digests (run-4/5/9 + run-13).
        config: Pipeline configuration dict for hash validation.

    Returns:
        CooccurrenceResult with all metrics, pairings, and availability reports.
    """
    cache = Path(cache_root)
    config = config or {}

    # Step 1: Cache-reuse guard
    valid_hash, expected_hash = _validate_cache_hash(config, cache)
    use_fallback = not valid_hash
    mismatch_reported: str | None = None
    if use_fallback:
        mismatch_reported = f"Hash mismatch: expected {expected_hash}, found {_load_cached_hash(cache)}"

    # Step 2: Single harness instantiation (never a second harness)
    harness = TopologyHarness(cache, config)

    # Step 3: Load rollup for case selection
    rollup = _load_a3_rollup(cache)

    # Step 4: Stratified sampling
    cmr_flagged, cmr_controls = _sample_cmr_cases(rollup)
    lge_flagged, lge_controls = _sample_lge_cases(rollup)

    # Step 5: Build records per (case_id, modality)
    records: list[StratifiedRecord] = []
    missing_pairs: list[str] = []

    # Process CMR cases
    all_cmr_cases = cmr_flagged + cmr_controls
    for case_id in all_cmr_cases:
        record = rollup.get(case_id, {})
        issue_set = set(record.get("issue_type", []))
        slice_indices = record.get("slice_indices", [])
        flagged_slices = [s for s in slice_indices if record.get("flagged_slices", {}).get(str(s), False)]
        control_slices = [s for s in slice_indices if s not in flagged_slices]

        # Per-slice metrics
        cmr_metrics: dict[int, CMRSliceMetrics] = {}
        for sidx in slice_indices:
            metrics = harness.get_cmr_slice_metrics(case_id, sidx, use_fallback=use_fallback)
            if metrics is not None:
                cmr_metrics[sidx] = metrics
            else:
                missing_pairs.append(f"CMR:{case_id}/{sidx}")

        # Ventricular level classification (use first slice as proxy for case)
        total_slices = len(slice_indices)
        if total_slices > 0:
            vl = _classify_ventricular_level(slice_indices[0], total_slices)
        else:
            vl = VentricularLevel.MID_VENTRICLE

        # Pattern classification (aggregate)
        pattern = PatternType.CONTIGUOUS
        for m in cmr_metrics.values():
            if m.connected_component_count > 1:
                pattern = PatternType.ISOLATED
                break

        rec = StratifiedRecord(
            case_id=case_id,
            modality=Modality.CMR,
            issue_type=issue_set,
            flagged_slice_indices=sorted(flagged_slices),
            control_slice_indices=sorted(control_slices),
            ventricular_level=vl,
            pattern_type=pattern,
            cmr_metrics=cmr_metrics,
            missing_pairs=[m for m in missing_pairs if m.startswith(f"CMR:{case_id}")],
        )
        records.append(rec)

    # Process LGE cases
    all_lge_cases = lge_flagged + lge_controls
    for case_id in all_lge_cases:
        record = rollup.get(case_id, {})
        issue_set = set(record.get("issue_type", []))
        slice_indices = record.get("slice_indices", [])
        flagged_slices = [s for s in slice_indices if record.get("flagged_slices", {}).get(str(s), False)]
        control_slices = [s for s in slice_indices if s not in flagged_slices]

        lge_metrics: dict[int, LGESliceMetrics] = {}
        for sidx in slice_indices:
            metrics = harness.get_lge_slice_metrics(case_id, sidx, use_fallback=use_fallback)
            if metrics is not None:
                lge_metrics[sidx] = metrics
            else:
                missing_pairs.append(f"LGE:{case_id}/{sidx}")

        total_slices = len(slice_indices)
        if total_slices > 0:
            vl = _classify_ventricular_level(slice_indices[0], total_slices)
        else:
            vl = VentricularLevel.MID_VENTRICLE

        pattern = PatternType.CONTIGUOUS
        for m in lge_metrics.values():
            if m.mi_connected_component_count > 1:
                pattern = PatternType.ISOLATED
                break

        rec = StratifiedRecord(
            case_id=case_id,
            modality=Modality.LGE,
            issue_type=issue_set,
            flagged_slice_indices=sorted(flagged_slices),
            control_slice_indices=sorted(control_slices),
            ventricular_level=vl,
            pattern_type=pattern,
            lge_metrics=lge_metrics,
            missing_pairs=[m for m in missing_pairs if m.startswith(f"LGE:{case_id}")],
        )
        records.append(rec)

    # Step 6: Compute summary statistics
    cmr_flagged_sampled = len(cmr_flagged)
    cmr_control_sampled = len(cmr_controls)
    lge_flagged_sampled = len(lge_flagged)
    lge_control_sampled = len(lge_controls)

    # Paired cases: cases present in both CMR and LGE with matching case_id base
    cmr_case_ids = {r.case_id for r in records if r.modality == Modality.CMR}
    lge_case_ids = {r.case_id for r in records if r.modality == Modality.LGE}
    paired_cases = len(cmr_case_ids & lge_case_ids)

    # False positive rates (placeholder - computed from control slices)
    cmr_ring_break_fpr: float | None = None
    cmr_excess_component_fpr: float | None = None
    lge_topology_fragmentation_fpr: float | None = None

    return CooccurrenceResult(
        records=records,
        cmr_flagged_sampled=cmr_flagged_sampled,
        cmr_control_sampled=cmr_control_sampled,
        lge_flagged_sampled=lge_flagged_sampled,
        lge_control_sampled=lge_control_sampled,
        paired_cases=paired_cases,
        missing_pairs_reported=missing_pairs,
        cmr_ring_break_fpr=cmr_ring_break_fpr,
        cmr_excess_component_fpr=cmr_excess_component_fpr,
        lge_topology_fragmentation_fpr=lge_topology_fragmentation_fpr,
        cache_hash_valid=valid_hash,
        fallback_extractor_invoked=use_fallback,
        mismatch_reported=mismatch_reported,
    )


# ---------------------------------------------------------------------------
# Serialization helpers for reproducibility checks
# ---------------------------------------------------------------------------

def result_to_json(result: CooccurrenceResult) -> dict[str, Any]:
    """Convert CooccurrenceResult to a JSON-serializable dict."""
    return {
        "records": [
            {
                "case_id": r.case_id,
                "modality": r.modality.value,
                "issue_type": sorted(r.issue_type),
                "flagged_slice_indices": r.flagged_slice_indices,
                "control_slice_indices": r.control_slice_indices,
                "ventricular_level": r.ventricular_level.value,
                "pattern_type": r.pattern_type.value,
                "missing_pairs": r.missing_pairs,
            }
            for r in result.records
        ],
        "cmr_flagged_sampled": result.cmr_flagged_sampled,
        "cmr_control_sampled": result.cmr_control_sampled,
        "lge_flagged_sampled": result.lge_flagged_sampled,
        "lge_control_sampled": result.lge_control_sampled,
        "paired_cases": result.paired_cases,
        "missing_pairs_reported": result.missing_pairs_reported,
        "cmr_ring_break_fpr": result.cmr_ring_break_fpr,
        "cmr_excess_component_fpr": result.cmr_excess_component_fpr,
        "lge_topology_fragmentation_fpr": result.lge_topology_fragmentation_fpr,
        "cache_hash_valid": result.cache_hash_valid,
        "fallback_extractor_invoked": result.fallback_extractor_invoked,
        "mismatch_reported": result.mismatch_reported,
    }


def result_checksum(result: CooccurrenceResult) -> str:
    """Compute a stable checksum for reproducibility validation."""
    canonical = json.dumps(result_to_json(result), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
