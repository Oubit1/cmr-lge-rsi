'''Bounded read-only shadow audit for the cmr_myo_missing_temporal_hole stratum.

The module is fully offline and self-contained. It consumes only
already-materialised shadow-audit records from the cached run-4/5/9/13-18
artifacts and computes the acceptance metrics required for the bounded
read-only audit:

* positive predictive value (PPV) of the flag among audit positives,
* false-positive rate (FPR) in the cmr control cohort,
* slice-level flag vs shadow temporal-continuity support agreement,
* cache hit rate,
* hash-mismatch routing (run-9 B1 extractor only) plus registration.

No labels, no review rules and no thresholds are written or redefined here;
the module is a pure read-only consumer of caller-supplied records.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = [
    'READ_ONLY',
    'RUN9_B1_EXTRACTOR',
    'ALLOWED_HASH_MISMATCH_EXTRACTORS',
    'CACHED_RUN_IDS',
    'PPV_MIN',
    'FPR_MAX',
    'CONSISTENCY_MIN',
    'CACHE_HIT_MIN',
    'AUDIT_POSITIVE_TARGET',
    'CONTROL_TARGET',
    'is_reused_run',
    'validate_reused_run',
    'CaseRecord',
    'SliceRecord',
    'CacheStats',
    'HashMismatchRecord',
    'HashMismatchRegistry',
    'AuditReport',
    'compute_ppv',
    'compute_control_fpr',
    'compute_consistency',
    'reextract_on_hash_mismatch',
    'run_shadow_audit',
]

# The audit is read-only: no writes, no label/rule/threshold mutation.
READ_ONLY = True

# Hash mismatches may only be re-extracted with the run-9 B1 extractor.
RUN9_B1_EXTRACTOR = 'run-9:B1'
ALLOWED_HASH_MISMATCH_EXTRACTORS = frozenset({RUN9_B1_EXTRACTOR})

# Runs whose cached artifacts may be reused by this audit.
CACHED_RUN_IDS = (
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

# Acceptance thresholds (read-only mirrors of the approved action).
PPV_MIN = 0.7
FPR_MAX = 0.3
CONSISTENCY_MIN = 0.8
CACHE_HIT_MIN = 0.9

# Cohort sizes requested by the approved action.
AUDIT_POSITIVE_TARGET = 30
CONTROL_TARGET = 10


def is_reused_run(run_id):
    '''Return True when run_id belongs to the cached run-4/5/9/13-18 set.'''
    return run_id in CACHED_RUN_IDS


def validate_reused_run(run_id):
    '''Raise ValueError when a run is outside the allowed cached set.'''
    if not is_reused_run(run_id):
        raise ValueError(
            'run {0!r} is not in the cached run-4/5/9/13-18 set'.format(run_id)
        )


@dataclass(frozen=True)
class CaseRecord:
    '''One case of the audit positive or cmr control cohort.'''

    case_id: str
    flag: bool
    shadow_reference: bool
    cohort: str = 'positive'

    def __post_init__(self):
        if self.cohort not in ('positive', 'control'):
            raise ValueError(
                'cohort must be positive or control, got {0!r}'.format(self.cohort)
            )


@dataclass(frozen=True)
class SliceRecord:
    '''One slice-level observation used for the continuity agreement metric.'''

    case_id: str
    slice_id: str
    flag: bool
    shadow_continuity_support: bool


@dataclass
class CacheStats:
    '''Read-only cache accounting for the reused run-4/5/9/13-18 artifacts.'''

    hits: int = 0
    misses: int = 0

    @property
    def accesses(self):
        return self.hits + self.misses

    @property
    def hit_rate(self):
        if self.accesses == 0:
            return 0.0
        return self.hits / self.accesses

    def register_access(self, hit):
        '''Record one read-only cache access.'''
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def merge(self, other):
        '''Merge another CacheStats into this one (counts only).'''
        self.hits += other.hits
        self.misses += other.misses


@dataclass(frozen=True)
class HashMismatchRecord:
    '''A logged hash mismatch re-extracted via the run-9 B1 extractor.'''

    case_id: str
    slice_id: str
    expected_hash: str
    observed_hash: str
    extractor_id: str = RUN9_B1_EXTRACTOR


class HashMismatchRegistry:
    '''Append-only registry for hash mismatches re-extracted via run-9 B1.'''

    def __init__(self):
        self._records = []

    @property
    def records(self):
        return tuple(self._records)

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def register(
        self,
        case_id,
        slice_id,
        expected_hash,
        observed_hash,
        extractor_id=RUN9_B1_EXTRACTOR,
    ):
        '''Register one hash mismatch; only the run-9 B1 extractor is allowed.'''
        if extractor_id not in ALLOWED_HASH_MISMATCH_EXTRACTORS:
            raise ValueError(
                'hash mismatch re-extraction must use the run-9 B1 extractor; '
                'got {0!r}'.format(extractor_id)
            )
        record = HashMismatchRecord(
            case_id=case_id,
            slice_id=slice_id,
            expected_hash=expected_hash,
            observed_hash=observed_hash,
            extractor_id=extractor_id,
        )
        self._records.append(record)
        return record

    def case_ids(self):
        '''Registered case ids, in registration order.'''
        return tuple(record.case_id for record in self._records)


def reextract_on_hash_mismatch(
    registry,
    case_id,
    slice_id,
    expected_hash,
    observed_hash,
    extractor_id=RUN9_B1_EXTRACTOR,
):
    '''Register a hash mismatch re-extracted with the run-9 B1 extractor.

    Returns None when the hashes match (nothing to register), otherwise the
    newly registered HashMismatchRecord. Any other extractor is rejected.
    '''
    if expected_hash == observed_hash:
        return None
    return registry.register(
        case_id=case_id,
        slice_id=slice_id,
        expected_hash=expected_hash,
        observed_hash=observed_hash,
        extractor_id=extractor_id,
    )


def compute_ppv(positives):
    '''PPV of the flag among audit positives (cases where flag is True).'''
    flagged = [record for record in positives if record.flag]
    if not flagged:
        return 0.0
    true_positives = sum(1 for record in flagged if record.shadow_reference)
    return true_positives / len(flagged)


def compute_control_fpr(controls):
    '''Fraction of cmr control cases incorrectly flagged.'''
    if not controls:
        return 0.0
    false_positives = sum(1 for record in controls if record.flag)
    return false_positives / len(controls)


def compute_consistency(slices):
    '''Agreement between slice-level flag and shadow continuity support.'''
    if not slices:
        return 0.0
    agree = sum(
        1 for record in slices if record.flag == record.shadow_continuity_support
    )
    return agree / len(slices)


@dataclass(frozen=True)
class AuditReport:
    '''Immutable result of the bounded read-only shadow audit.'''

    ppv: float
    control_fpr: float
    consistency: float
    cache_hit_rate: float
    positive_count: int
    control_count: int
    slice_count: int
    hash_mismatch_records: Tuple[HashMismatchRecord, ...] = ()

    @property
    def hash_mismatch_count(self):
        return len(self.hash_mismatch_records)

    @property
    def hash_mismatch_extractors(self):
        return tuple(sorted({r.extractor_id for r in self.hash_mismatch_records}))

    def hash_mismatch_routing_ok(self):
        '''True when every mismatch was re-extracted via the run-9 B1 extractor.'''
        return all(
            record.extractor_id in ALLOWED_HASH_MISMATCH_EXTRACTORS
            for record in self.hash_mismatch_records
        )

    def failures(self):
        '''Names of the acceptance metrics that are not satisfied.'''
        problems = []
        if self.ppv < PPV_MIN:
            problems.append('ppv')
        if self.control_fpr > FPR_MAX:
            problems.append('control_fpr')
        if self.consistency < CONSISTENCY_MIN:
            problems.append('consistency')
        if self.cache_hit_rate < CACHE_HIT_MIN:
            problems.append('cache_hit_rate')
        if not self.hash_mismatch_routing_ok():
            problems.append('hash_mismatch_routing')
        return tuple(problems)

    def passes(self):
        '''True only when every acceptance metric is satisfied.'''
        return not self.failures()


def run_shadow_audit(positives, controls, slices, cache_stats=None, registry=None):
    '''Compute the read-only shadow-audit report from caller-supplied records.'''
    stats = CacheStats() if cache_stats is None else cache_stats
    reg = HashMismatchRegistry() if registry is None else registry
    return AuditReport(
        ppv=compute_ppv(positives),
        control_fpr=compute_control_fpr(controls),
        consistency=compute_consistency(slices),
        cache_hit_rate=stats.hit_rate,
        positive_count=len(positives),
        control_count=len(controls),
        slice_count=len(slices),
        hash_mismatch_records=reg.records,
    )
