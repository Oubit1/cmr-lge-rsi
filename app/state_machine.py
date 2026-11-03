from __future__ import annotations

from enum import Enum


class RunState(str, Enum):
    REGISTERED = "REGISTERED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    EVIDENCE_READY = "EVIDENCE_READY"
    QC_PASSED = "QC_PASSED"
    QC_FAILED = "QC_FAILED"
    FAILURE_ANALYSIS = "FAILURE_ANALYSIS"
    EXPERIMENT_PROPOSED = "EXPERIMENT_PROPOSED"
    VALIDATING = "VALIDATING"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"
    ERROR = "ERROR"


ALLOWED_TRANSITIONS = {
    RunState.REGISTERED: {RunState.QUEUED, RunState.EVIDENCE_READY},
    RunState.QUEUED: {RunState.RUNNING, RunState.CANCELLED, RunState.SUPERSEDED, RunState.ERROR},
    RunState.RUNNING: {RunState.QC_PASSED, RunState.QC_FAILED, RunState.CANCELLED,
                       RunState.SUPERSEDED, RunState.ERROR},
    RunState.QC_FAILED: {RunState.FAILURE_ANALYSIS, RunState.QUEUED, RunState.NEEDS_REVIEW},
    RunState.EVIDENCE_READY: {RunState.FAILURE_ANALYSIS, RunState.NEEDS_REVIEW},
    RunState.FAILURE_ANALYSIS: {RunState.EXPERIMENT_PROPOSED, RunState.NEEDS_REVIEW, RunState.ERROR},
    RunState.EXPERIMENT_PROPOSED: {RunState.VALIDATING, RunState.NEEDS_REVIEW},
    RunState.VALIDATING: {RunState.PROMOTED, RunState.REJECTED, RunState.NEEDS_REVIEW, RunState.ERROR},
    RunState.ERROR: {RunState.QUEUED, RunState.NEEDS_REVIEW},
    RunState.QC_PASSED: set(),
    RunState.PROMOTED: set(),
    RunState.REJECTED: set(),
    RunState.NEEDS_REVIEW: {RunState.QUEUED, RunState.FAILURE_ANALYSIS},
    RunState.CANCELLED: set(),
    RunState.SUPERSEDED: set(),
}


def validate_transition(current: str, target: str) -> None:
    source = RunState(current)
    destination = RunState(target)
    if destination not in ALLOWED_TRANSITIONS[source]:
        raise ValueError(f"invalid workflow transition: {source.value} -> {destination.value}")
