"""
Bounded LGE-side co-occurrence stratum for mi_fraction_high, mi_multiple_components, myo_topology.
Run-16 evidence collection. Read-only, zero production writes.
"""
import numpy as np
from typing import List, Dict, Any

def _morph_close_1voxel(mask: np.ndarray) -> np.ndarray:
    """Fixed 1-voxel morphological closing."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    h, w = mask.shape
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    out = np.zeros_like(padded)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            out[i+1, j+1] = 1 if np.all(region & kernel) else 0
    return out[1:-1, 1:-1].astype(np.uint8)

def _distance_to_myocardium(mask: np.ndarray, cmr_myo: np.ndarray) -> float:
    """Approximate mean distance from mask boundary to CMR myocardial boundary."""
    if not np.any(mask):
        return 0.0
    dist_map = np.full_like(cmr_myo, 999.0, dtype=float)
    dist_map[cmr_myo > 0] = 0.0
    for _ in range(10):
        new_dist = dist_map.copy()
        for i in range(1, dist_map.shape[0]-1):
            for j in range(1, dist_map.shape[1]-1):
                neighbors = [dist_map[i-1,j], dist_map[i+1,j], dist_map[i,j-1], dist_map[i,j+1]]
                new_dist[i,j] = min(new_dist[i,j], min(neighbors) + 1)
        dist_map = new_dist
    return float(np.mean(dist_map[mask > 0]))

def _largest_component_area_ratio(mask: np.ndarray) -> float:
    """Compute ratio of largest connected component to total foreground area."""
    if not np.any(mask):
        return 0.0
    visited = np.zeros_like(mask, dtype=bool)
    max_area = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] and not visited[i,j]:
                stack = [(i,j)]
                area = 0
                while stack:
                    ci, cj = stack.pop()
                    if 0 <= ci < mask.shape[0] and 0 <= cj < mask.shape[1] and mask[ci,cj] and not visited[ci,cj]:
                        visited[ci,cj] = True
                        area += 1
                        stack.extend([(ci+1,cj), (ci-1,cj), (ci,cj+1), (ci,cj-1)])
                if area > max_area:
                    max_area = area
    return max_area / np.sum(mask)

def _ring_break_aware_largest_component(mask: np.ndarray, cmr_myo: np.ndarray) -> np.ndarray:
    """Retain largest component constrained to paired CMR myocardial band."""
    if not np.any(mask):
        return mask
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] and not visited[i,j]:
                stack = [(i,j)]
                comp = []
                while stack:
                    ci, cj = stack.pop()
                    if 0 <= ci < mask.shape[0] and 0 <= cj < mask.shape[1] and mask[ci,cj] and not visited[ci,cj]:
                        visited[ci,cj] = True
                        comp.append((ci,cj))
                        stack.extend([(ci+1,cj), (ci-1,cj), (ci,cj+1), (ci,cj-1)])
                components.append(comp)
    components.sort(key=lambda c: len(c), reverse=True)
    result = np.zeros_like(mask)
    for comp in components:
        overlap = sum(1 for r, c in comp if cmr_myo[r,c] > 0)
        if overlap > 0:
            for r, c in comp:
                result[r,c] = 1
            break
    return result

def apply_shadow_arms(lge_mask: np.ndarray, cmr_myo_mask: np.ndarray,
                      prev_slice_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """Apply all shadow arms and return metrics."""
    results = {}
    results['noop'] = lge_mask.copy()
    closed = _morph_close_1voxel(lge_mask)
    dist = _distance_to_myocardium(closed, cmr_myo_mask)
    results['close_merge'] = closed
    results['close_merge_dist'] = dist
    ring_constrained = _ring_break_aware_largest_component(lge_mask, cmr_myo_mask)
    results['ring_break'] = ring_constrained
    if prev_slice_result and 'ring_break' in prev_slice_result:
        prev_comp = prev_slice_result['ring_break']
        if np.any(prev_comp):
            overlap = np.sum(ring_constrained & prev_comp)
            union = np.sum(ring_constrained | prev_comp)
            iou = overlap / union if union > 0 else 0
            results['temporal_iou'] = iou
            results['temporal_flag'] = iou < 0.5
        else:
            results['temporal_flag'] = False
    else:
        results['temporal_flag'] = False
    return results

def compute_stratum_metrics(sample_data: List[Dict]) -> Dict[str, Any]:
    """Process stratified sample and emit shadow metrics + review queue."""
    metrics = {
        'total_cases': len(sample_data),
        'stratified_counts': {'with_topo': 0, 'without_topo': 0, 'controls': 0},
        'flag_rates': {'original': 0.0, 'noop': 0.0, 'close_merge': 0.0, 'ring_break': 0.0},
        'component_counts': [],
        'area_ratios': [],
        'distances': [],
        'topology_resolution_rate': 0.0,
        'ring_break_resolution_rate': 0.0,
        'cross_arm_agreement': 0.0,
        'review_queue': []
    }
    total_with_topo = 0
    total_without_topo = 0
    total_controls = 0
    flag_counts = {'original': 0, 'noop': 0, 'close_merge': 0, 'ring_break': 0}
    component_counts_all = []
    area_ratios_all = []
    distances_all = []
    resolutions = {'topo': 0, 'ring_break': 0}
    agreements = 0
    total_slices = 0
    for case in sample_data:
        lge_mask = np.array(case['lge_mask'], dtype=np.uint8)
        cmr_mask = np.array(case['cmr_mask'], dtype=np.uint8)
        has_topo = case.get('has_myocardial_topology', False)
        is_control = case.get('is_control', False)
        if is_control:
            total_controls += 1
        elif has_topo:
            total_with_topo += 1
        else:
            total_without_topo += 1
        orig_flag = case.get('mi_fraction_high', False)
        if orig_flag: flag_counts['original'] += 1
        shadows = apply_shadow_arms(lge_mask, cmr_mask)
        noop_flag = np.any(shadows['noop'])
        close_flag = np.any(shadows['close_merge'])
        ring_flag = np.any(shadows['ring_break'])
        if noop_flag: flag_counts['noop'] += 1
        if close_flag: flag_counts['close_merge'] += 1
        if ring_flag: flag_counts['ring_break'] += 1
        cc_orig = len(np.unique(shadows['noop'])) - 1
        cc_close = len(np.unique(shadows['close_merge'])) - 1
        cc_ring = len(np.unique(shadows['ring_break'])) - 1
        ar_orig = _largest_component_area_ratio(shadows['noop'])
        ar_close = _largest_component_area_ratio(shadows['close_merge'])
        ar_ring = _largest_component_area_ratio(shadows['ring_break'])
        dist = _distance_to_myocardium(shadows['ring_break'], cmr_mask)
        component_counts_all.append({'orig': cc_orig, 'close': cc_close, 'ring': cc_ring})
        area_ratios_all.append({'orig': ar_orig, 'close': ar_close, 'ring': ar_ring})
        distances_all.append(dist)
        if has_topo and not ring_flag:
            resolutions['ring_break'] += 1
        if has_topo and cc_ring <= 1:
            resolutions['topo'] += 1
        if noop_flag == close_flag == ring_flag:
            agreements += 1
        total_slices += 1
        discordance = ""
        if cc_orig != cc_ring:
            discordance += f"CC changed {cc_orig}->{cc_ring}; "
        if ar_orig < 0.5 and ar_ring > 0.8:
            discordance += f"Area ratio shifted {ar_orig:.2f}->{ar_ring:.2f}; "
        if dist > 2.0 and has_topo:
            discordance += f"Distance to CMR high ({dist:.1f}); "
        if discordance:
            metrics['review_queue'].append({
                'case_id': case.get('id', 'unknown'),
                'reason': discordance.strip('; '),
                'arm': 'ring_break'
            })
    metrics['stratified_counts'] = {'with_topo': total_with_topo, 'without_topo': total_without_topo, 'controls': total_controls}
    metrics['flag_rates'] = {k: v/len(sample_data) for k, v in flag_counts.items()}
    metrics['component_counts'] = component_counts_all
    metrics['area_ratios'] = area_ratios_all
    metrics['distances'] = distances_all
    metrics['topology_resolution_rate'] = resolutions['topo'] / max(1, total_with_topo)
    metrics['ring_break_resolution_rate'] = resolutions['ring_break'] / max(1, total_with_topo)
    metrics['cross_arm_agreement'] = agreements / max(1, total_slices)
    return metrics
