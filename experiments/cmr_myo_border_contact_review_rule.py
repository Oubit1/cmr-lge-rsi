"""Draft, default-OFF, review-only rule ``R-CMR-MYO-BORDER-CONTACT-REVIEW-01``.

Slice positive
    ``modality == "cmr"`` **and** ``cmr_myo_touches_image_border`` is true
    **and** at least one allowed support flag is true on the same slice.  The
    allowed support flags are exactly ``small_area``, ``excess_components`` and
    ``empty``; any other flag name is ignored.

Case escalation
    at least ``CASE_MIN_POSITIVE_SLICES`` positive slice(s) **and** at least
    ``CASE_MIN_SUPPORT_FLAGS`` distinct case-level support flags.  Case-level
    flags are the union of the supported flags observed on the case's slices
    plus any supported flag declared directly on the case record.

Guarantees of this module
-------------------------
* The rule is review-only and disabled by default (:func:`default_spec`);
  nothing here blocks a payload, mutates a payload or writes a production rule
  (``PRODUCTION_WRITE_ENABLED`` is ``False`` and is reported verbatim in every
  verification record).
* No new numeric QC threshold is defined.  ``CASE_MIN_*`` are integer parts of
  the approved rule definition; every activation threshold
  (``min_slice_precision``, ``max_case_escalation_rate``) must be injected by
  the caller from the existing promotion-gate configuration.
  :func:`evaluate_activation` refuses to return an eligible decision when the
  thresholds are absent, so a metric that does not clear the gate can never
  enable the rule.
* :func:`verification_record` returns a deterministic SHA-256 stamped record so
  the A1 evidence review (slice precision and case escalation rate) can be
  reproduced and audited.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ALLOWED_SUPPORT_FLAGS",
    "BORDER_CONTACT_KEY",
    "CASE_MIN_POSITIVE_SLICES",
    "CASE_MIN_SUPPORT_FLAGS",
    "CMR_MODALITY",
    "GATE_THRESHOLD_KEYS",
    "PRODUCTION_WRITE_ENABLED",
    "REVIEW_ONLY",
    "RULE_ID",
    "ActivationDecision",
    "CaseDecision",
    "ReviewRuleSpec",
    "RuleMetrics",
    "SliceDecision",
    "case_slices",
    "default_spec",
    "evaluate_activation",
    "evaluate_case",
    "evaluate_slice",
    "record_modality",
    "record_reference",
    "score_evidence",
    "support_flags",
    "verification_record",
]

RULE_ID = "R-CMR-MYO-BORDER-CONTACT-REVIEW-01"

CMR_MODALITY = "cmr"
BORDER_CONTACT_KEY = "cmr_myo_touches_image_border"

#: Support flags the draft rule is allowed to use (rule definition, not a tuned
#: numeric set).  Anything outside this tuple is ignored on purpose so the draft
#: cannot silently widen its own support set.
ALLOWED_SUPPORT_FLAGS: tuple[str, ...] = ("small_area", "excess_components", "empty")

#: Rule-definition counts taken from the approved spec.  These are structural
#: parts of the rule, not tuned numeric thresholds.
CASE_MIN_POSITIVE_SLICES = 1
CASE_MIN_SUPPORT_FLAGS = 2

#: The draft never writes a production rule and never blocks a payload.
PRODUCTION_WRITE_ENABLED = False
REVIEW_ONLY = True

#: Threshold *key names* consumed from the existing promotion-gate configuration.
#: No numeric value for these keys is defined in this module.
GATE_THRESHOLD_KEYS: tuple[str, ...] = ("min_slice_precision", "max_case_escalation_rate")

MODALITY_KEYS: tuple[str, ...] = ("modality", "mod", "case_modality", "image_modality")
CASE_ID_KEYS: tuple[str, ...] = ("case_id", "case_uid", "case", "patient_id", "study_id")
SLICE_ID_KEYS: tuple[str, ...] = ("slice_id", "slice_uid", "slice", "id", "uid")
SLICE_CONTAINER_KEYS: tuple[str, ...] = ("slices", "slice_records")
FLAG_CONTAINER_KEYS: tuple[str, ...] = (
    "support_flags",
    "flags",
    "review_flags",
    "support_flag_names",
)
REFERENCE_KEYS: tuple[str, ...] = (
    "reference_positive",
    "ground_truth_positive",
    "ground_truth",
    "label",
    "human_positive",
    "reviewer_positive",
    "is_positive",
)
INHERIT_KEYS: tuple[str, ...] = MODALITY_KEYS + CASE_ID_KEYS

_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "on", "positive", "pos"})
_UNKNOWN_TOKENS = frozenset(
    {"", "unknown", "na", "n/a", "none", "null", "unlabelled", "unlabeled", "skip"}
)


# ---------------------------------------------------------------------------
# small value helpers
# ---------------------------------------------------------------------------
def _is_true(value: Any) -> bool:
    """Truthiness for QC payload values (bool, 0/1, "true"/"yes", non-empty)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_TOKENS
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) > 0
    if isinstance(value, Mapping):
        return any(_is_true(item) for item in value.values())
    return bool(value)


def _identifier(record: Any, keys: Sequence[str], default: str) -> str:
    if isinstance(record, Mapping):
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return default


def _container_flags(container: Any) -> set[str]:
    found: set[str] = set()
    if container is None:
        return found
    if isinstance(container, str):
        text = container.strip().lower()
        if text:
            found.add(text)
    elif isinstance(container, Mapping):
        for name, value in container.items():
            if name is None:
                continue
            if _is_true(value):
                found.add(str(name).strip().lower())
    elif isinstance(container, (list, tuple, set, frozenset)):
        for item in container:
            if item is None:
                continue
            if isinstance(item, Mapping):
                for name, value in item.items():
                    if _is_true(value):
                        found.add(str(name).strip().lower())
            else:
                text = str(item).strip().lower()
                if text:
                    found.add(text)
    return found


def support_flags(record: Any) -> tuple[str, ...]:
    """Return the *allowed* support flags that are true on ``record``.

    Only names in :data:`ALLOWED_SUPPORT_FLAGS` are returned; any other flag
    (for example ``temporal_hole`` or ``centroid_drift``) is ignored.
    """
    if not isinstance(record, Mapping):
        return ()
    found: set[str] = set()
    for key in FLAG_CONTAINER_KEYS:
        if key in record:
            found |= _container_flags(record.get(key))
    for flag in ALLOWED_SUPPORT_FLAGS:
        if flag in record and _is_true(record.get(flag)):
            found.add(flag)
    allowed = frozenset(ALLOWED_SUPPORT_FLAGS)
    return tuple(sorted(found & allowed))


def record_modality(record: Any) -> str:
    if not isinstance(record, Mapping):
        return ""
    for key in MODALITY_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def record_reference(record: Any) -> bool | None:
    """Reference label of a record, or ``None`` when it is not reviewable."""
    if not isinstance(record, Mapping):
        return None
    for key in REFERENCE_KEYS:
        if key not in record:
            continue
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in _UNKNOWN_TOKENS:
            continue
        return _is_true(value)
    return None


# ---------------------------------------------------------------------------
# rule specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReviewRuleSpec:
    """Frozen description of the draft rule.  Defaults are the approved draft."""

    rule_id: str = RULE_ID
    enabled: bool = False
    review_only: bool = True
    blocks_payload: bool = False
    writes_production_rule: bool = False
    support_flags: tuple[str, ...] = ALLOWED_SUPPORT_FLAGS
    case_min_positive_slices: int = CASE_MIN_POSITIVE_SLICES
    case_min_support_flags: int = CASE_MIN_SUPPORT_FLAGS


def default_spec() -> ReviewRuleSpec:
    """Return the approved draft spec: review-only, disabled by default."""
    return ReviewRuleSpec()


# ---------------------------------------------------------------------------
# per-slice / per-case evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SliceDecision:
    slice_id: str
    case_id: str
    positive: bool
    support_flags: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CaseDecision:
    case_id: str
    escalated: bool
    positive_slice_ids: tuple[str, ...]
    case_support_flags: tuple[str, ...]
    slice_decisions: tuple[SliceDecision, ...]
    reasons: tuple[str, ...]


def case_slices(case_record: Any) -> tuple[Mapping[str, Any], ...]:
    """Return the slice mappings of a case record (falls back to the record)."""
    if not isinstance(case_record, Mapping):
        return ()
    for key in SLICE_CONTAINER_KEYS:
        value = case_record.get(key)
        if isinstance(value, (list, tuple)):
            return tuple(item for item in value if isinstance(item, Mapping))
        if isinstance(value, Mapping):
            return tuple(item for item in value.values() if isinstance(item, Mapping))
    return (case_record,)


def _inherit(case_record: Mapping[str, Any], slice_record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy case-level identifiers/modality down; never inherit slice evidence.

    Only identification and modality keys are inherited.  ``cmr_myo_touches_image_border``
    and support flags stay slice-local, so a case-level flag cannot create a
    positive slice.
    """
    merged: dict[str, Any] = dict(slice_record)
    for key in INHERIT_KEYS:
        if key in merged:
            continue
        if key in case_record:
            merged[key] = case_record[key]
    return merged


def evaluate_slice(slice_record: Any, spec: ReviewRuleSpec | None = None) -> SliceDecision:
    """Apply the slice-positive definition to one slice record."""
    spec = spec or default_spec()
    if not isinstance(slice_record, Mapping):
        return SliceDecision(
            slice_id="unknown-slice",
            case_id="unknown-case",
            positive=False,
            support_flags=(),
            reasons=("slice_record_not_mapping",),
        )
    allowed = frozenset(spec.support_flags)
    flags = tuple(flag for flag in support_flags(slice_record) if flag in allowed)
    reasons: list[str] = []
    if record_modality(slice_record) != CMR_MODALITY:
        reasons.append("modality_not_cmr")
    if not _is_true(slice_record.get(BORDER_CONTACT_KEY)):
        reasons.append("myo_does_not_touch_image_border")
    if not flags:
        reasons.append("no_allowed_support_flag")
    return SliceDecision(
        slice_id=_identifier(slice_record, SLICE_ID_KEYS, "unknown-slice"),
        case_id=_identifier(slice_record, CASE_ID_KEYS, "unknown-case"),
        positive=not reasons,
        support_flags=flags,
        reasons=tuple(reasons),
    )


def evaluate_case(case_record: Any, spec: ReviewRuleSpec | None = None) -> CaseDecision:
    """Apply the slice rule to every slice and the case escalation definition."""
    spec = spec or default_spec()
    if not isinstance(case_record, Mapping):
        return CaseDecision(
            case_id="unknown-case",
            escalated=False,
            positive_slice_ids=(),
            case_support_flags=(),
            slice_decisions=(),
            reasons=("case_record_not_mapping",),
        )
    allowed = frozenset(spec.support_flags)
    case_id = _identifier(case_record, CASE_ID_KEYS, "unknown-case")
    case_flags = {flag for flag in support_flags(case_record) if flag in allowed}
    decisions: list[SliceDecision] = []
    positive_ids: list[str] = []
    for raw_slice in case_slices(case_record):
        decision = evaluate_slice(_inherit(case_record, raw_slice), spec)
        decisions.append(decision)
        case_flags.update(decision.support_flags)
        if decision.positive:
            positive_ids.append(decision.slice_id)
    reasons: list[str] = []
    if len(positive_ids) < spec.case_min_positive_slices:
        reasons.append("no_positive_slice")
    if len(case_flags) < spec.case_min_support_flags:
        reasons.append("case_support_flags_below_minimum")
    return CaseDecision(
        case_id=case_id,
        escalated=not reasons,
        positive_slice_ids=tuple(positive_ids),
        case_support_flags=tuple(sorted(case_flags)),
        slice_decisions=tuple(decisions),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# A1 evidence metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuleMetrics:
    rule_id: str
    slices_scored: int
    slices_positive: int
    positive_slices_with_reference: int
    true_positive_slices: int
    false_positive_slices: int
    slice_precision: float | None
    cases_scored: int
    cases_escalated: int
    case_escalation_rate: float | None
    escalated_case_ids: tuple[str, ...]
    positive_slice_ids: tuple[str, ...]
    reasons: tuple[str, ...]


def score_evidence(
    cases: Iterable[Any] | None, spec: ReviewRuleSpec | None = None
) -> RuleMetrics:
    """Score A1 evidence cases with the draft rule (offline, read-only).

    ``slice_precision`` is computed over predicted-positive slices that carry a
    reviewable reference label; unlabelled positives are reported through
    ``reasons`` instead of being counted as errors.
    """
    spec = spec or default_spec()
    allowed = frozenset(spec.support_flags)
    slices_scored = 0
    slices_positive = 0
    labelled = 0
    true_positive = 0
    false_positive = 0
    cases_scored = 0
    positive_slice_ids: list[str] = []
    escalated_case_ids: list[str] = []
    reasons: set[str] = set()

    for case_record in cases or ():
        if not isinstance(case_record, Mapping):
            reasons.add("case_record_not_mapping")
            continue
        cases_scored += 1
        case_id = _identifier(case_record, CASE_ID_KEYS, "case-%d" % cases_scored)
        case_flags = {flag for flag in support_flags(case_record) if flag in allowed}
        case_positive = False
        raw_slices = case_slices(case_record)
        if not raw_slices:
            reasons.add("case_without_slices")
        for raw_slice in raw_slices:
            slice_record = _inherit(case_record, raw_slice)
            decision = evaluate_slice(slice_record, spec)
            slices_scored += 1
            case_flags.update(decision.support_flags)
            if not decision.positive:
                continue
            slices_positive += 1
            case_positive = True
            positive_slice_ids.append(decision.slice_id)
            reference = record_reference(slice_record)
            if reference is None:
                reference = record_reference(case_record)
            if reference is None:
                reasons.add("reference_label_missing")
                continue
            labelled += 1
            if reference:
                true_positive += 1
            else:
                false_positive += 1
        if case_positive and len(case_flags) >= spec.case_min_support_flags:
            escalated_case_ids.append(case_id)

    if slices_positive == 0:
        reasons.add("no_positive_slices_observed")
    if labelled == 0:
        reasons.add("slice_precision_unavailable")
    if cases_scored == 0:
        reasons.add("no_cases_scored")

    slice_precision = (true_positive / labelled) if labelled else None
    case_escalation_rate = (len(escalated_case_ids) / cases_scored) if cases_scored else None
    return RuleMetrics(
        rule_id=spec.rule_id,
        slices_scored=slices_scored,
        slices_positive=slices_positive,
        positive_slices_with_reference=labelled,
        true_positive_slices=true_positive,
        false_positive_slices=false_positive,
        slice_precision=slice_precision,
        cases_scored=cases_scored,
        cases_escalated=len(escalated_case_ids),
        case_escalation_rate=case_escalation_rate,
        escalated_case_ids=tuple(escalated_case_ids),
        positive_slice_ids=tuple(positive_slice_ids),
        reasons=tuple(sorted(reasons)),
    )


# ---------------------------------------------------------------------------
# activation gate (thresholds injected by the caller, never defined here)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActivationDecision:
    rule_id: str
    eligible: bool
    enabled: bool
    reasons: tuple[str, ...]
    thresholds: dict[str, float] | None
    metrics: RuleMetrics


def _normalise_thresholds(
    thresholds: Mapping[str, Any] | None, reasons: list[str]
) -> dict[str, float] | None:
    if not isinstance(thresholds, Mapping):
        reasons.append("missing_thresholds")
        return None
    normalised: dict[str, float] = {}
    ok = True
    for key in GATE_THRESHOLD_KEYS:
        if key not in thresholds:
            reasons.append("missing_threshold_key:" + key)
            ok = False
            continue
        value = thresholds[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reasons.append("non_numeric_threshold:" + key)
            ok = False
            continue
        normalised[key] = float(value)
    return normalised if ok else None


def evaluate_activation(
    metrics: RuleMetrics, thresholds: Mapping[str, Any] | None = None
) -> ActivationDecision:
    """Report whether the draft may be considered for activation.

    ``eligible`` never enables the rule: this module keeps ``enabled=False`` in
    every decision, and it returns ``eligible=False`` whenever the A1 evidence
    metrics are unavailable or the injected gate thresholds are absent or not
    met.  The numeric thresholds themselves live in the existing promotion-gate
    configuration, not here.
    """
    reasons: list[str] = []
    normalised = _normalise_thresholds(thresholds, reasons)
    if metrics.slice_precision is None:
        reasons.append("slice_precision_unavailable")
    elif (
        normalised is not None
        and metrics.slice_precision < normalised["min_slice_precision"]
    ):
        reasons.append("slice_precision_below_threshold")
    if metrics.case_escalation_rate is None:
        reasons.append("case_escalation_rate_unavailable")
    elif (
        normalised is not None
        and metrics.case_escalation_rate > normalised["max_case_escalation_rate"]
    ):
        reasons.append("case_escalation_rate_above_threshold")
    return ActivationDecision(
        rule_id=metrics.rule_id,
        eligible=not reasons,
        enabled=False,
        reasons=tuple(reasons),
        thresholds=normalised,
        metrics=metrics,
    )


def verification_record(
    metrics: RuleMetrics,
    thresholds: Mapping[str, Any] | None = None,
    spec: ReviewRuleSpec | None = None,
) -> dict[str, Any]:
    """Deterministic, hash-stamped review record for the A1 evidence run.

    The record carries the injected gate thresholds (or ``None`` when they are
    absent), the A1 slice-precision / case-escalation metrics and the explicit
    statement that no production rule was written and that the draft stays
    disabled.  ``digest`` is a SHA-256 over the canonical JSON payload, so two
    runs over the same evidence and thresholds produce the same record and the
    review is reproducible.
    """
    spec = spec or default_spec()
    decision = evaluate_activation(metrics, thresholds)
    payload: dict[str, Any] = {
        "rule_id": spec.rule_id,
        "spec": asdict(spec),
        "metrics": asdict(metrics),
        "thresholds": (
            dict(sorted(decision.thresholds.items())) if decision.thresholds else None
        ),
        "activation_eligible": decision.eligible,
        "activation_enabled": decision.enabled,
        "activation_reasons": list(decision.reasons),
        "review_only": spec.review_only,
        "blocks_payload": spec.blocks_payload,
        "production_rule_written": bool(
            spec.writes_production_rule or PRODUCTION_WRITE_ENABLED
        ),
        "gate_threshold_keys": list(GATE_THRESHOLD_KEYS),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"digest": digest, "payload": payload}
