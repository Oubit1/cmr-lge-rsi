"""Class-level coverage-ledger registration for ``cmr_myo_small_area``.

Isolated-side helper: registers the class-level coverage entry for the small
myo-area rule, delegates the duplicate hard gate to an existing isolated module
when one is importable, and otherwise applies the local
(rule_version, cache_hash) fingerprint rule.  It also back-fills the still-open
``lge_mi_fraction_high`` items from run-47 without emitting duplicate proposals.

Pure data in / data out: no I/O, no network, no dataset or label access.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CLASS_ENTRY_ID = "cmr_myo_small_area"
CLASS_ID = "cmr_myo_small_area"
BACKFILL_SOURCE_RUN = "run-47"
BACKFILL_ENTRY_ID = "lge_mi_fraction_high"
BASELINE_RUN = "run-50"

SKIP = "skip_unchanged"
REGISTER = "register_new"
INVALIDATE = "invalidate_reregister"

_IDENTITY_FIELDS = ("entry_id", "class_id", "rule_version", "cache_hash")

#: existing isolated modules that may already own the duplicate hard gate
_SHARED_GATE_MODULES = (
    "coverage_ledger_dedup",
    "audit_coverage_ledger",
    "coverage_ledger_closure",
    "evidence_dedup",
)
_SHARED_GATE_FUNCS = (
    "dedup_decision",
    "decide_dedup",
    "hard_gate_decision",
    "is_duplicate",
)

_GATE_CALLABLE: Optional[Any] = None
_GATE_RESOLVED = False


# --------------------------------------------------------------------------- identity


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(entry: Mapping[str, Any]) -> str:
    """Stable sha256 over the identity fields used by the dedup hard gate."""
    identity = {field: entry.get(field) for field in _IDENTITY_FIELDS}
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def entry_key(entry: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(entry.get("entry_id")), str(entry.get("class_id")))


def make_entry(
    entry_id: str,
    class_id: str,
    rule_version: str,
    cache_hash: str,
    source_run: str,
) -> Dict[str, Any]:
    return {
        "entry_id": str(entry_id),
        "class_id": str(class_id),
        "rule_version": str(rule_version),
        "cache_hash": str(cache_hash),
        "source_run": str(source_run),
    }


def small_area_entry(
    rule_version: str, cache_hash: str, source_run: str = BASELINE_RUN
) -> Dict[str, Any]:
    return make_entry(CLASS_ENTRY_ID, CLASS_ID, rule_version, cache_hash, source_run)


def mi_fraction_high_entry(
    rule_version: str,
    cache_hash: str,
    source_run: str = BACKFILL_SOURCE_RUN,
) -> Dict[str, Any]:
    return make_entry(
        BACKFILL_ENTRY_ID, BACKFILL_ENTRY_ID, rule_version, cache_hash, source_run
    )


# --------------------------------------------------------------------- shared gate


def _accepts_two_positional_args(fn: Any) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    required = [
        p
        for p in params
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(required) > 2:
        return False
    capacity = sum(
        1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    return has_varargs or capacity >= 2


def load_shared_gate() -> Optional[Any]:
    """Return an existing dedup hard gate callable, or None when unavailable."""
    for module_name in _SHARED_GATE_MODULES:
        try:
            module = importlib.import_module("experiments." + module_name)
        except Exception:
            continue
        for func_name in _SHARED_GATE_FUNCS:
            candidate = getattr(module, func_name, None)
            if callable(candidate) and _accepts_two_positional_args(candidate):
                return candidate
    return None


def reset_shared_gate_cache(gate: Optional[Any] = None, resolved: bool = False) -> None:
    """Test seam: pin or clear the memoised shared-gate lookup."""
    global _GATE_CALLABLE, _GATE_RESOLVED
    _GATE_CALLABLE = gate
    _GATE_RESOLVED = resolved


def _shared_gate_callable() -> Optional[Any]:
    global _GATE_RESOLVED
    if not _GATE_RESOLVED:
        reset_shared_gate_cache(load_shared_gate(), True)
    return _GATE_CALLABLE


def _interpret_verdict(verdict: Any) -> Optional[bool]:
    if verdict is None:
        return None
    if isinstance(verdict, bool):
        return verdict
    if isinstance(verdict, str):
        token = verdict.strip().lower()
        if token in ("skip", "skip_unchanged", "duplicate", "dup", "hit", "covered", "true"):
            return True
        if token in (
            "register",
            "register_new",
            "new",
            "miss",
            "false",
            "invalidate",
            "invalidate_reregister",
        ):
            return False
        return None
    if isinstance(verdict, Mapping):
        for key in ("duplicate", "is_duplicate", "hit", "skip", "covered"):
            if key in verdict:
                return bool(verdict[key])
    return None


def _shared_gate_says_covered(
    entry: Mapping[str, Any], previous: Optional[Mapping[str, Any]]
) -> Optional[bool]:
    gate = _shared_gate_callable()
    if gate is None:
        return None
    try:
        return _interpret_verdict(gate(entry, previous))
    except Exception:
        return None


# ------------------------------------------------------------------------- ledger


def decide(
    entry: Mapping[str, Any], previous: Optional[Mapping[str, Any]] = None
) -> str:
    """Hard-gate verdict for one ledger entry against its previous row."""
    if previous is None:
        return REGISTER
    if fingerprint(entry) == fingerprint(previous):
        covered = _shared_gate_says_covered(entry, previous)
        if covered is None or covered:
            return SKIP
    return INVALIDATE


def register_entry(
    ledger: Mapping[Tuple[str, str], Mapping[str, Any]],
    entry: Mapping[str, Any],
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], str]:
    """Insert ``entry`` unless the unchanged class should be skipped."""
    updated: Dict[Tuple[str, str], Dict[str, Any]] = {
        key: dict(row) for key, row in ledger.items()
    }
    key = entry_key(entry)
    decision = decide(entry, updated.get(key))
    if decision != SKIP:
        updated[key] = dict(entry)
    return updated, decision


def backfill_run47(
    ledger: Mapping[Tuple[str, str], Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[Tuple[Tuple[str, str], str]]]:
    """Re-emit the open run-47 entries, never proposing a duplicate row twice."""
    updated: Dict[Tuple[str, str], Dict[str, Any]] = {
        key: dict(row) for key, row in ledger.items()
    }
    decisions: List[Tuple[Tuple[str, str], str]] = []
    for entry in entries:
        updated, decision = register_entry(updated, entry)
        decisions.append((entry_key(entry), decision))
    return updated, decisions


def rows(ledger: Mapping[Tuple[str, str], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in ledger.values()]


def duplicate_reduction_ratio(
    baseline_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> float:
    baseline_count = len(list(baseline_rows))
    if baseline_count == 0:
        return 0.0
    return max(0.0, 1.0 - (len(list(current_rows)) / baseline_count))
