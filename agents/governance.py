from __future__ import annotations

from typing import Any

from app.database import Database


ACTION_OWNERS = {
    "framework_repair": "workflow_maintainer",
    "isolated_code_experiment": "qc_code_optimizer",
    "rule_draft": "rule_curator",
    "skill_draft": "skill_maintainer",
}
EXECUTABLE_ACTION_STATUSES = {"APPROVED", "RUNNING", "SUCCEEDED"}


def enforce_action_ownership(actions: list) -> tuple[list, list[dict[str, str]]]:
    """Reject executable work assigned to a role that does not own it."""
    kept, blocked = [], []
    for action in actions:
        expected = ACTION_OWNERS.get(action.action_class)
        if expected and action.owner_role != expected:
            blocked.append({
                "action_id": action.action_id,
                "owner_role": action.owner_role,
                "expected_owner": expected,
                "reason": "role_ownership_mismatch",
            })
        else:
            kept.append(action)
    return kept, blocked


def enforce_autonomous_executor_contract(actions: list) -> tuple[list, list[dict[str, str]]]:
    """Drop actions that claim autonomous execution without a configured owner."""
    kept, blocked = [], []
    for action in actions:
        if action.auto_execute and action.action_class not in ACTION_OWNERS:
            blocked.append({
                "action_id": action.action_id,
                "owner_role": action.owner_role,
                "action_class": action.action_class,
                "reason": "no_autonomous_executor",
                "instruction": (
                    "Choose an action_class from action_ownership for autonomous work, "
                    "or set auto_execute false only for a genuine human dependency."
                ),
            })
        else:
            kept.append(action)
    return kept, blocked


def agent_action_authorized(db: Database, run_id: int, row: dict[str, Any],
                            payload: dict[str, Any]) -> bool:
    """Require supervisor assignment plus independent-review handoff."""
    return row["status"] in EXECUTABLE_ACTION_STATUSES and original_action_approved(db,run_id,row,payload)


def original_action_approved(db: Database, run_id: int, row: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Read original role handoffs; does not make a rejected action executable."""
    if not payload.get("review_approved"):
        return False
    expected = ACTION_OWNERS.get(payload.get("action_class"))
    if expected and row["owner_role"] != expected:
        return False
    thread = f"run-{run_id}:{row['action_id']}"
    inbox = db.agent_inbox(run_id, row["owner_role"], thread_id=thread)
    assigned = any(item["sender_role"] == "supervisor"
                   and item["message_type"] == "assigned_action" for item in inbox)
    reviews = [item for item in inbox
               if item["sender_role"] == "independent_reviewer"
               and item["message_type"] == "action_review"]
    reviewed = bool(reviews and reviews[-1]["payload"].get("decision") == "accept_experiment")
    return assigned and reviewed


def implementation_review_payload(db: Database, run_id: int, action_id: str,
                                  owner_role: str) -> dict[str, Any] | None:
    thread = f"run-{run_id}:{action_id}"
    reviews = [item for item in db.agent_inbox(run_id, owner_role, thread_id=thread)
               if item["sender_role"] == "independent_reviewer"
               and item["message_type"] == "implementation_review"]
    return reviews[-1]["payload"] if reviews else None


def implementation_review_approved(db: Database, run_id: int, action_id: str,
                                   owner_role: str) -> bool:
    review = implementation_review_payload(db, run_id, action_id, owner_role)
    return bool(review and review.get("decision") == "accept_experiment"
                and not review.get("required_validation_metrics"))
