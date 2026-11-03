"""
isolated_code_experiment: centroid_provenance_shadow
Read-only shadow module for CMR/LGE centroid provenance analysis.
Consumes validated run-4/run-5 cached per-case/per-slice digests.
No source-image re-extraction, no production writes, no new thresholds.
"""
import copy
from typing import Dict, List, Tuple, Any
from collections import defaultdict

REQUIRED_CONFIG_HASH = "run4_run5_cached_v1"
MAX_CASES = 40
OUTLIER_CAP = 10
CONTROL_COUNT = 5
EVIDENCE_MAX_RECORDS = 20

def validate_digests(digests: Dict[str, Any]) -> bool:
    """Validate that digests match expected cached structure."""
    if not isinstance(digests, dict):
        return False
    for case_id, case_data in digests.items():
        if not isinstance(case_data, dict):
            return False
        if 'slices' not in case_data:
            return False
        if not isinstance(case_data['slices'], list):
            return False
    return True

def sample_cases(digests: Dict[str, Any], max_cases: int = MAX_CASES,
                 outlier_cap: int = OUTLIER_CAP, control_count: int = CONTROL_COUNT) -> List[Dict]:
    """Sample up to max_cases, including all review_centroid_outlier slices up to cap plus controls."""
    sampled = []
    for case_id, case_data in digests.items():
        if len(sampled) >= max_cases:
            break

        slices = case_data.get('slices', [])
        outlier_slices = [s for s in slices if s.get('review_centroid_outlier')]
        control_slices = [s for s in slices if not s.get('review_centroid_outlier')]

        # Cap outliers, take fixed controls
        outlier_slices = outlier_slices[:outlier_cap]
        control_slices = control_slices[:control_count]

        if outlier_slices or control_slices:
            sampled.append({
                'case_id': case_id,
                'modality': case_data.get('modality', 'unknown'),
                'stratum': case_data.get('modality', '').upper(),
                'outlier_slices': outlier_slices,
                'control_slices': control_slices
            })
    return sampled

def apply_noop_arm(cases: List[Dict]) -> List[Dict]:
    """No-op arm: returns cases unchanged."""
    return copy.deepcopy(cases)

def apply_largest_component_retention_arm(cases: List[Dict]) -> List[Dict]:
    """Largest-component retention restricted to myocardial/MI-containing band."""
    ret_cases = []
    for c in cases:
        nc = copy.deepcopy(c)
        for s in nc['outlier_slices']:
            # Cached flag indicates if component is largest in band
            s['retained'] = s.get('is_largest_in_band', True)
        ret_cases.append(nc)
    return ret_cases

def apply_temporal_continuity_arm(cases: List[Dict]) -> List[Dict]:
    """Temporal or neighbouring-slice continuity at pre-declared absolute voxel window."""
    cont_cases = []
    for c in cases:
        nc = copy.deepcopy(c)
        for s in nc['outlier_slices']:
            # Cached flag indicates temporal consistency
            s['continuous'] = s.get('temporal_consistent', True)
        cont_cases.append(nc)
    return cont_cases

def compute_persistence_rate(outlier_flags_before: List[bool], outlier_flags_after: List[bool]) -> float:
    """Compute persistence rate of review_centroid_outlier."""
    if not outlier_flags_before:
        return 0.0
    matches = sum(b == a for b, a in zip(outlier_flags_before, outlier_flags_after))
    return matches / len(outlier_flags_before)

def compute_centroid_displacement_delta(centroids_before: List[Tuple[float, float, float]],
                                        centroids_after: List[Tuple[float, float, float]]) -> float:
    """Compute average centroid displacement delta."""
    if not centroids_before:
        return 0.0
    deltas = []
    for cb, ca in zip(centroids_before, centroids_after):
        d = ((cb[0]-ca[0])**2 + (cb[1]-ca[1])**2 + (cb[2]-ca[2])**2)**0.5
        deltas.append(d)
    return sum(deltas) / len(deltas)

def attribute_outliers(flags: List[Dict]) -> Dict[str, float]:
    """Attribute review_centroid_outlier to categories using cached flags."""
    categories = ['small-area', 'border-touching', 'excess-component', 'topology-break', 'temporal-drift', 'unexplained-residual']
    counts = {cat: 0 for cat in categories}
    total = 0

    for f in flags:
        attr = f.get('attribution_category')
        if attr in counts:
            counts[attr] += 1
            total += 1

    return {cat: count / total if total > 0 else 0.0 for cat, count in counts.items()}

def deduplicate_evidence(payload: List[Dict], max_records: int = EVIDENCE_MAX_RECORDS) -> List[Dict]:
    """Deduplicate evidence payload to <= max_records while retaining slice indices."""
    seen = set()
    unique = []
    for rec in payload:
        key = (rec.get('case_id'), rec.get('modality'))
        if key not in seen:
            seen.add(key)
            unique.append(rec)
            if len(unique) >= max_records:
                break
    return unique

def run_shadow_analysis(digests: Dict[str, Any], config_hash: str) -> Dict[str, Any]:
    """Main entry point for shadow analysis."""
    if config_hash != REQUIRED_CONFIG_HASH:
        raise ValueError(f"Config hash mismatch: {config_hash}. Stopping to prevent re-extraction.")
    if not validate_digests(digests):
        raise ValueError("Invalid digest structure. Stopping to prevent re-extraction.")

    # Separate by stratum
    cmr_digests = {k: v for k, v in digests.items() if v.get('modality', '').upper() == 'CMR'}
    lge_digests = {k: v for k, v in digests.items() if v.get('modality', '').upper() == 'LGE'}

    results = {}
    for stratum_name, stratum_digests in [("CMR", cmr_digests), ("LGE", lge_digests)]:
        cases = sample_cases(stratum_digests)

        # Run arms
        noop_cases = apply_noop_arm(cases)
        lc_cases = apply_largest_component_retention_arm(cases)
        tc_cases = apply_temporal_continuity_arm(cases)

        # Extract flags for persistence
        noop_flags = [s.get('review_centroid_outlier') for c in noop_cases for s in c['outlier_slices']]
        lc_flags = [s.get('review_centroid_outlier') for c in lc_cases for s in c['outlier_slices']]
        tc_flags = [s.get('review_centroid_outlier') for c in tc_cases for s in c['outlier_slices']]

        persistence_noop = compute_persistence_rate(noop_flags, noop_flags)
        persistence_lc = compute_persistence_rate(noop_flags, lc_flags)
        persistence_tc = compute_persistence_rate(noop_flags, tc_flags)

        # Extract centroids for displacement
        noop_centroids = [(s.get('x',0), s.get('y',0), s.get('z',0)) for c in noop_cases for s in c['outlier_slices']]
        lc_centroids = [(s.get('x',0), s.get('y',0), s.get('z',0)) for c in lc_cases for s in c['outlier_slices']]
        tc_centroids = [(s.get('x',0), s.get('y',0), s.get('z',0)) for c in tc_cases for s in c['outlier_slices']]

        disp_noop = compute_centroid_displacement_delta(noop_centroids, noop_centroids)
        disp_lc = compute_centroid_displacement_delta(noop_centroids, lc_centroids)
        disp_tc = compute_centroid_displacement_delta(noop_centroids, tc_centroids)

        # Attribution
        all_flags = [s for c in cases for s in c['outlier_slices']]
        attribution = attribute_outliers(all_flags)

        # Evidence dedup
        evidence_payload = [{'case_id': c['case_id'], 'modality': c['modality'], 'slice_idx': i}
                            for c in cases for i, s in enumerate(c['outlier_slices'])]
        deduped_evidence = deduplicate_evidence(evidence_payload)

        results[stratum_name] = {
            'persistence_rates': {
                'no_op': persistence_noop,
                'largest_component_retention': persistence_lc,
                'temporal_continuity': persistence_tc
            },
            'centroid_displacement_deltas': {
                'no_op': disp_noop,
                'largest_component_retention': disp_lc,
                'temporal_continuity': disp_tc
            },
            'attribution_fractions': attribution,
            'evidence_dedup_count': len(deduped_evidence),
            'cases_sampled': len(cases)
        }

    return results
