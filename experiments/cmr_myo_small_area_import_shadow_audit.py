'''Read-only import shadow audit for cmr_myo_small_area.

This module only aggregates records that were produced by the approved
run-4/5/9/13-18 cache summaries. It does not extract, relabel, or write
production artifacts. On cache hash mismatch, the only permitted extractor
fallback is the run-9 B1 import extractor; callers should record that call
without creating a second extraction path.
'''
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

TARGET_LABEL = 'cmr_myo_small_area'
SUPPORT_FLAGS = (
    'excess_components',
    'touches_image_border',
    'missing_temporal_hole',
    'centroid_drift',
    'layer_area_jump',
    'myo_empty',
)
ALLOWED_CACHE_RUNS = (
    'run-4',
    'run-5',
    'run-9',
    'run-13',
    'run-14',
    'run-15',
    'run-16',
    'run-17',
    'run-18',
)
B1_EXTRACTOR_RUN = 'run-9'
B1_EXTRACTOR_NAME = 'B1_import_extractor'
POSITIVE_SAMPLE_SIZE = 50
CONTROL_SAMPLE_SIZE = 10
SEVERITY_ORDER = ('none', 'low', 'medium', 'high', 'critical')
_SEVERITY_RANK = {name: idx for idx, name in enumerate(SEVERITY_ORDER)}


@dataclass(frozen=True)
class SliceRecord:
    case_id: str
    modality: str
    slice_index: int
    label: str
    flags: Mapping[str, bool] = field(default_factory=dict)
    cached_hash: str | None = None
    live_hash: str | None = None
    source_run: str | None = None
    extractor: str | None = None

    @property
    def is_target_positive(self) -> bool:
        return self.label == TARGET_LABEL

    @property
    def is_hash_mismatch(self) -> bool:
        if self.cached_hash is None or self.live_hash is None:
            return False
        return self.cached_hash != self.live_hash

    @property
    def used_b1_on_mismatch(self) -> bool:
        return self.is_hash_mismatch and self.extractor == B1_EXTRACTOR_NAME


def _flag(record: SliceRecord, name: str) -> bool:
    return bool(record.flags.get(name, False))


def has_any_same_slice_support(record: SliceRecord) -> bool:
    return any(_flag(record, flag) for flag in SUPPORT_FLAGS)


def positive_records(records: Iterable[SliceRecord]) -> list[SliceRecord]:
    return [record for record in records if record.is_target_positive]


def positive_slice_support_summary(records: Iterable[SliceRecord]) -> dict:
    positives = positive_records(records)
    total = len(positives)
    unsupported = [record for record in positives if not has_any_same_slice_support(record)]
    co_counts = {flag: 0 for flag in SUPPORT_FLAGS}
    for record in positives:
        for flag in SUPPORT_FLAGS:
            if _flag(record, flag):
                co_counts[flag] += 1
    co_ratios = {
        flag: (co_counts[flag] / total if total else 0.0)
        for flag in SUPPORT_FLAGS
    }
    return {
        'positive_slice_count': total,
        'unsupported_positive_slice_count': len(unsupported),
        'unsupported_positive_slice_ratio': (len(unsupported) / total if total else 0.0),
        'same_slice_cooccurrence_counts': co_counts,
        'same_slice_cooccurrence_ratios': co_ratios,
    }


def severity_tier(record: SliceRecord) -> str:
    if _flag(record, 'myo_empty'):
        return 'critical'
    count = sum(1 for flag in SUPPORT_FLAGS if _flag(record, flag))
    if count <= 0:
        return 'none'
    if count == 1:
        return 'low'
    if count == 2:
        return 'medium'
    return 'high'


def _severity_rank(tier: str) -> int:
    return _SEVERITY_RANK.get(tier, 0)


def worst_slice_per_case(records: Iterable[SliceRecord]) -> dict:
    grouped: dict[str, list[SliceRecord]] = defaultdict(list)
    for record in records:
        if record.is_target_positive:
            grouped[record.case_id].append(record)
    result = {}
    for case_id, case_records in grouped.items():
        def sort_key(record: SliceRecord):
            return (
                _severity_rank(severity_tier(record)),
                sum(1 for flag in SUPPORT_FLAGS if _flag(record, flag)),
                record.slice_index,
            )
        worst = max(case_records, key=sort_key)
        result[case_id] = {
            'worst_slice_index': worst.slice_index,
            'worst_slice_severity': severity_tier(worst),
            'flagged_slice_indices': sorted({record.slice_index for record in case_records}),
            'slice_index_distribution': dict(sorted(Counter(record.slice_index for record in case_records).items())),
        }
    return result


def flagged_slice_indices_addressable(records: Iterable[SliceRecord]) -> dict:
    all_by_case: dict[str, set[int]] = defaultdict(set)
    flagged_by_case: dict[str, set[int]] = defaultdict(set)
    for record in records:
        all_by_case[record.case_id].add(record.slice_index)
        if record.is_target_positive:
            flagged_by_case[record.case_id].add(record.slice_index)
    failures = {}
    for case_id, flagged in flagged_by_case.items():
        missing = sorted(flagged - all_by_case.get(case_id, set()))
        if missing:
            failures[case_id] = missing
    return {
        'addressable': not failures,
        'failures': failures,
        'flagged_case_count': len(flagged_by_case),
    }


def cache_hash_and_b1_summary(records: Iterable[SliceRecord]) -> dict:
    mismatches = []
    b1_calls = 0
    non_b1_on_mismatch = []
    disallowed_runs = []
    for record in records:
        if record.source_run is not None and record.source_run not in ALLOWED_CACHE_RUNS:
            disallowed_runs.append((record.case_id, record.modality, record.slice_index, record.source_run))
        if record.is_hash_mismatch:
            mismatches.append(record)
            if record.used_b1_on_mismatch:
                b1_calls += 1
            else:
                non_b1_on_mismatch.append((record.case_id, record.modality, record.slice_index))
    return {
        'cache_hash_mismatch_count': len(mismatches),
        'b1_extractor_call_count': b1_calls,
        'non_b1_on_mismatch_count': len(non_b1_on_mismatch),
        'non_b1_on_mismatch': non_b1_on_mismatch,
        'disallowed_cache_run_count': len(disallowed_runs),
        'disallowed_cache_runs': disallowed_runs,
        'allowed_cache_runs': list(ALLOWED_CACHE_RUNS),
        'b1_extractor_run': B1_EXTRACTOR_RUN,
        'b1_extractor_name': B1_EXTRACTOR_NAME,
    }


def dedup_gate_summary(records: Iterable[SliceRecord]) -> dict:
    groups: dict[tuple[str, str], list[SliceRecord]] = defaultdict(list)
    for record in records:
        groups[(record.case_id, record.modality)].append(record)
    duplicate_groups = {}
    passed = 0
    for key, group in groups.items():
        if len(group) == 1:
            passed += 1
        else:
            duplicate_groups['|'.join(key)] = len(group)
    total = len(groups)
    return {
        'group_count': total,
        'passed_group_count': passed,
        'dedup_gate_pass_rate': (passed / total if total else 1.0),
        'duplicate_groups': duplicate_groups,
        'exactly_one_per_case_modality': passed == total,
    }


def records_from_cache_summaries(summaries: Iterable[Mapping]) -> list[SliceRecord]:
    records = []
    for summary in summaries:
        source_run = summary.get('source_run')
        if source_run is not None and source_run not in ALLOWED_CACHE_RUNS:
            continue
        records.append(SliceRecord(
            case_id=str(summary['case_id']),
            modality=str(summary['modality']),
            slice_index=int(summary['slice_index']),
            label=str(summary.get('label', TARGET_LABEL)),
            flags=dict(summary.get('flags') or {}),
            cached_hash=summary.get('cached_hash'),
            live_hash=summary.get('live_hash'),
            source_run=source_run,
            extractor=summary.get('extractor'),
        ))
    return records


def build_import_shadow_audit(records: Iterable[SliceRecord]) -> dict:
    record_list = list(records)
    return {
        'target_label': TARGET_LABEL,
        'support_flags': list(SUPPORT_FLAGS),
        'sample_plan': {'positive': POSITIVE_SAMPLE_SIZE, 'control': CONTROL_SAMPLE_SIZE},
        'positive_slice_support': positive_slice_support_summary(record_list),
        'worst_slice_per_case': worst_slice_per_case(record_list),
        'slice_index_addressability': flagged_slice_indices_addressable(record_list),
        'cache_hash_and_b1': cache_hash_and_b1_summary(record_list),
        'dedup_gate': dedup_gate_summary(record_list),
    }
