"""Deterministic ownership for agent work.

The orchestrator, not an interactive coding assistant, assigns every medical or
maintenance task to one configured role. Keeping this mapping in executable
code prevents a generic model from silently absorbing specialist work.
"""
from __future__ import annotations

from typing import Any


ROLE_RESPONSIBILITIES: dict[str, tuple[str, ...]] = {
    "workflow_maintainer": ("workflow_protocol_repair", "control_plane_diagnostics"),
    "supervisor": ("final_decision", "action_prioritization"),
    "failure_triage": ("failure_triage",),
    "cmr_specialist": ("cmr_label_analysis", "cmr_prompt_repair", "cmr_edit_learning"),
    "lge_myo_specialist": (
        "lge_myo_label_analysis", "lge_myo_prompt_repair", "lge_myo_edit_learning",
    ),
    "lge_mi_specialist": (
        "lge_mi_label_analysis", "lge_mi_prompt_repair", "lge_mi_edit_learning",
    ),
    "independent_reviewer": ("independent_review",),
    "qc_code_optimizer": ("qc_code_optimization", "isolated_code_experiment"),
    "rule_curator": ("rule_curation", "casebook_compaction"),
    "skill_maintainer": ("skill_maintenance", "handoff_maintenance"),
    "efficiency_auditor": ("efficiency_audit", "token_audit"),
}


def role_for_label_task(request: dict[str, Any]) -> str:
    """Route a de-identified label task by modality and target."""
    modality = str(request.get("modality") or "").lower()
    target = str(request.get("target") or "").lower()
    if modality == "cmr" and target == "myo":
        return "cmr_specialist"
    if modality == "lge" and target == "myo":
        return "lge_myo_specialist"
    if modality == "lge" and target == "mi":
        return "lge_mi_specialist"
    raise ValueError(f"unsupported label task: modality={modality!r}, target={target!r}")


def responsibility_for_label_task(request: dict[str, Any], *, prompt: bool) -> str:
    role = role_for_label_task(request)
    suffix = "prompt_repair" if prompt else "edit_learning"
    prefix = "cmr" if role == "cmr_specialist" else (
        "lge_myo" if role == "lge_myo_specialist" else "lge_mi"
    )
    return f"{prefix}_{suffix}"


def assert_role_owns(role: str, responsibility: str) -> None:
    if responsibility not in ROLE_RESPONSIBILITIES.get(role, ()):
        raise RuntimeError(
            f"agent routing violation: {role!r} does not own {responsibility!r}"
        )


def validate_agent_config(configs: dict[str, Any], *, required_roles: set[str] | None = None) -> None:
    """Fail early if a production profile is missing an owner or grants writes."""
    required = set(ROLE_RESPONSIBILITIES) - {"workflow_maintainer"} if required_roles is None else set(required_roles)
    missing = sorted(required - set(configs))
    if missing:
        raise RuntimeError(f"agent profile is missing required roles: {', '.join(missing)}")
    writable = sorted(role for role in (required | ({"workflow_maintainer"} & set(configs)))
                      if bool(configs[role].get("can_write")))
    if writable:
        raise RuntimeError(
            "agent roles must propose isolated work, not write governed data directly: "
            + ", ".join(writable)
        )
