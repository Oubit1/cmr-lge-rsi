#!/usr/bin/env python3
"""Local dashboard and audited human-review API for the CMR/LGE loop.

    python scripts/board_server.py --port 8765

Serves an HTML page at /, a JSON snapshot at /api/state, and accepts bounded
human dispositions at /api/reviews.  Review operations can change workflow
metadata but never read or write medical images, masks, or labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import urllib.request
import subprocess
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_card import build_score_card  # noqa: E402
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import Database
from app.agent_continuity import build_agent_continuity
from app.rsi_quality_benchmark import benchmark_status
from app.rsi_postpromotion import consumer_gate
from app.lge_execution import status as lge_execution_status
from app.check_inventory import build_check_inventory
from app.check_validation import review_state, submit_decision
from app.artifact_registry import csv_issue_slices
from app.existing_qc import MANIFESTS
from app.manual_annotation import (
    _manifest,
    _row_identity,
    apply_edit,
    boundary_review_counts,
    boundary_review_status,
    edit_status,
    frame_metadata,
    list_candidates,
    png_layer,
    prompt_candidate_png,
    prompt_repair_status,
    queue_prompt_repair,
    save_edit,
    set_item_excluded,
    set_boundary_review_status,
)
from agents.evidence import (approved_manual_edit_records,
                             approved_manual_review_records,
                             dependent_workflow_status,
                             enrich_workflow_dependency,
                             safe_review_feedback)

DB = ROOT / "runtime/workflow.sqlite"
STATE = ROOT / "runtime/autopilot_state.json"
import sys as _path_sys
_path_sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import paths as _public_paths
CHECK = (_public_paths.DATASET_ROOT / "labels/check")
DATASET = CHECK.parents[1]
MANUAL_WORK = ROOT / "runtime/manual_annotations"
EDITOR_PAGE = ROOT / "static/annotation_editor.html"
CHECK_INVENTORY_PAGE = ROOT / "static/check_inventory.html"
CHECK_VALIDATION_PAGE = ROOT / "static/check_validation.html"
LABEL_APPLY_AUDIT = MANUAL_WORK / "board_apply_audit.jsonl"
LABEL_APPLY_LOCK = threading.Lock()
CHECK_VALIDATION_LOCK = threading.Lock()
MAINTENANCE_CHAT_LOCK = threading.Lock()
MAINTENANCE_CHAT_AUDIT = ROOT / "runtime/maintenance_chat/audit.jsonl"
AUTOPILOT_CONTROL_LOCK = threading.RLock()
AUTOPILOT_CONTROL_AUDIT = ROOT / "runtime/control/autopilot_control.jsonl"
from app.autonomy_control import autonomy_status
from app.rsi_generator_benchmark import status as generator_benchmark_status

OPERATOR_CONTROL_STATE = ROOT / "runtime/control/operator_state.json"
WORKFLOW_SERVICES = (
    "cmr-lge-edit-agent.service",
    "cmr-lge-check-loop.service",
    "cmr-lge-autopilot.service",
)
BOARD_CONTROL_TOKEN = os.environ.get("CMR_LGE_BOARD_CONTROL_TOKEN", "").strip()
ALIYUN_BILLING_SNAPSHOT = Path(os.environ.get(
    "CMR_LGE_ALIYUN_BILLING_SNAPSHOT",
    str(ROOT / "runtime/private/aliyun_billing.json"),
))


def aliyun_billing_status(path: Path = ALIYUN_BILLING_SNAPSHOT) -> dict:
    """Return a credential-free billing snapshot for the local dashboard.

    The board never receives or exposes AccessKey material.  A separate
    read-only collector may write this small JSON file after calling
    QueryAccountBalance and QueryCashCoupons.
    """
    if not path.is_file():
        return {
            "status": "unconfigured",
            "message": "未配置阿里云只读账单快照",
            "required_permissions": ["bss:DescribeAcccount", "bss:DescribeCoupons"],
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": "ok",
            "available_amount": str(raw.get("available_amount", "-")),
            "cash_amount": str(raw.get("cash_amount", "-")),
            "coupon_balance": str(raw.get("coupon_balance", "0")),
            "currency": str(raw.get("currency", "CNY")),
            "updated_at": str(raw.get("updated_at", "")),
        }
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "error", "message": f"账单快照不可读：{exc}"}


MAINTENANCE_CHAT_BOUNDARY = {
    "allowed": ["服务与进程", "多智能体调度", "角色交接", "配置与接口",
                "测试与部署", "看板与交接文档"],
    "forbidden": ["医学影像判读", "标签修改", "样本分桶或路由",
                  "绕过复核、备份或审计"],
    "execution": "当前模型负责诊断与受控修复规划；生产代码变更仍须隔离快照、测试和独立复核。",
}


class AutopilotControlError(RuntimeError):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def board_control_authorized(host: str, supplied_token: str,
                             configured_token: str = BOARD_CONTROL_TOKEN) -> bool:
    """Allow local UI control, or require a secret for a non-local hostname."""
    if configured_token:
        return bool(supplied_token) and hmac.compare_digest(
            supplied_token, configured_token)
    name = (urlparse("//" + (host or "")).hostname or "").lower()
    return name in {"127.0.0.1", "localhost", "::1"}


def _append_control_audit(record: dict, path: Path = AUTOPILOT_CONTROL_AUDIT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def resume_autopilot(*, state_path: Path = STATE,
                     audit_path: Path = AUTOPILOT_CONTROL_AUDIT,
                     control_path: Path = OPERATOR_CONTROL_STATE,
                     runner=subprocess.run, fingerprint_fn=None) -> dict:
    """Audit a fuse resume, then use the same all-service Start transition."""
    with AUTOPILOT_CONTROL_LOCK:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise AutopilotControlError(f"Autopilot 状态不可读：{exc}",503) from exc
        paused = state.get("paused_issue_fingerprint")
        if not paused:
            raise AutopilotControlError("闭环当前没有处于可恢复的熔断状态")
        record = {"timestamp":datetime.now(timezone.utc).isoformat(),
                  "event":"BOARD_AUTOPILOT_RESUME_REQUESTED","source_run_id":state.get("escalated_for")}
        _append_control_audit(record,audit_path)
        if fingerprint_fn is None:
            from scripts.autopilot import issue_fingerprint
            fingerprint_fn = issue_fingerprint
        current, _ = fingerprint_fn(DATASET)
        try:
            stop = runner(["systemctl","--user","stop","cmr-lge-autopilot.service"],
                          cwd=ROOT,text=True,capture_output=True,timeout=45,check=False)
            if stop.returncode:
                raise AutopilotControlError("停止旧控制器失败",503)
            if current == paused:
                resume = runner([str(Path(sys.executable)),str(ROOT/"scripts/autopilot.py"),
                                 "--dataset",str(DATASET),"--resume-after-fix"],
                                cwd=ROOT,text=True,capture_output=True,timeout=120,check=False)
                if resume.returncode:
                    raise AutopilotControlError("恢复审计未通过："+(resume.stderr or resume.stdout or "resume failed")[-1000:])
            else:
                _append_control_audit({**record,"event":"BOARD_RESUME_WITH_NEW_KNOWLEDGE"},audit_path)
            started = control_workflow("start",state_path=state_path,control_path=control_path,
                                       audit_path=audit_path,runner=runner,fingerprint_fn=fingerprint_fn)
        except (OSError,subprocess.SubprocessError,AutopilotControlError) as exc:
            _append_control_audit({**record,"event":"BOARD_AUTOPILOT_RESUME_FAILED",
                                  "error":str(exc)[-1000:],"service_restarted":False},audit_path)
            if isinstance(exc,AutopilotControlError):raise
            raise AutopilotControlError("恢复未完成，处理服务没有被重新启动："+str(exc)[-500:],503) from exc
        _append_control_audit({**record,"event":"BOARD_AUTOPILOT_RESUMED"},audit_path)
        return {**started,"status":"resumed","message":"恢复审计完成，处理服务已统一启动",
                "source_run_id":state.get("escalated_for")}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def operator_control_state(path: Path = OPERATOR_CONTROL_STATE) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"paused": False}
    return value if isinstance(value, dict) else {"paused": False}


def framework_fix_available(state: dict, *, runner=subprocess.run) -> bool:
    """Return true only when a fused controller has a newer committed fix.

    The board uses this to make its normal Start button choose the existing
    audited ``resume_after_fix`` path.  An unchanged checkout must keep the
    fuse closed so an operator cannot accidentally start a paid retry loop.
    """
    if not state.get("paused_issue_fingerprint"):
        return False
    try:
        result = runner(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    commit = (result.stdout or "").strip() if result.returncode == 0 else ""
    return bool(commit and commit != state.get("last_code_fingerprint"))


def control_workflow(action: str, *, state_path: Path = STATE,
                     control_path: Path = OPERATOR_CONTROL_STATE,
                     audit_path: Path = AUTOPILOT_CONTROL_AUDIT,
                     runner=subprocess.run, fingerprint_fn=None) -> dict:
    """Start or pause all processing services without weakening domain gates."""
    if action not in {"start", "pause"}:
        raise AutopilotControlError("unsupported workflow control action", 400)
    requested_at = datetime.now(timezone.utc).isoformat()
    verb = "start" if action == "start" else "stop"
    with AUTOPILOT_CONTROL_LOCK:
        record = {
            "timestamp": requested_at,
            "event": f"BOARD_WORKFLOW_{action.upper()}_REQUESTED",
            "services": list(WORKFLOW_SERVICES),
        }
        _append_control_audit(record, audit_path)
        if action == "start":
            from app.workflow_budget import begin_operator_session
            try:
                session=begin_operator_session(control_path.parent.parent.parent)
                if session:_append_control_audit({"timestamp":requested_at,"event":"BOARD_MODEL_ALLOWANCE","limits":session["limits"],"used":session["used"],"deadline":session["deadline"]},audit_path)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                raise AutopilotControlError("启动预算配置不可用："+str(exc),503) from exc
        operator = {"paused":action == "pause", "updated_at":requested_at, "source":"board"}
        _atomic_json(control_path,operator)
        commands, errors = [], []
        services = WORKFLOW_SERVICES if action == "start" else tuple(reversed(WORKFLOW_SERVICES))
        for service in services:
            # Do not hold the HTTP request open while a role finishes its
            # current bounded call.  systemd keeps supervising the transition
            # and the board reports the intermediate stopping/starting state.
            command = ["systemctl", "--user", "--no-block", verb, service]
            commands.append(command)
            try:
                result = runner(command, cwd=ROOT, text=True, capture_output=True,
                                timeout=60, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{service}: {exc}")
                continue
            if result.returncode:
                errors.append(
                    f"{service}: {(result.stderr or result.stdout or 'systemctl failed').strip()[:500]}"
                )
        if errors:
            if action == "start":
                _atomic_json(control_path,{"paused":True,"updated_at":requested_at,"source":"board_start_failed"})
                for service in reversed(WORKFLOW_SERVICES):
                    try:
                        runner(["systemctl","--user","--no-block","stop",service],cwd=ROOT,text=True,capture_output=True,timeout=20,check=False)
                    except (OSError,subprocess.SubprocessError):
                        pass
            _append_control_audit({
                **record,
                "event": f"BOARD_WORKFLOW_{action.upper()}_FAILED",
                "errors": errors,
            }, audit_path)
            raise AutopilotControlError("；".join(errors), 503)

        operator = {
            "paused": action == "pause",
            "updated_at": requested_at,
            "source": "board",
        }
        _atomic_json(control_path, operator)
        if action == "pause":
            result = {
                "status": "paused",
                "message": "数据处理已暂停；看板保持在线，可继续查看和人工复核",
            }
        else:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                state = {}
            paused_fingerprint = state.get("paused_issue_fingerprint")
            knowledge_changed = False
            if paused_fingerprint:
                # Import lazily to keep the board startup lightweight and avoid
                # duplicating the exact knowledge/queue fingerprint contract.
                if fingerprint_fn is None:
                    from scripts.autopilot import issue_fingerprint
                    fingerprint_fn = issue_fingerprint
                current_fingerprint, _ = fingerprint_fn(DATASET)
                knowledge_changed = current_fingerprint != paused_fingerprint
            if paused_fingerprint and not knowledge_changed:
                result = {
                    "status": "waiting_for_human",
                    "message": (
                        "服务已启动，但现有证据已无新增收益。请复核并应用 1–2 个代表样本；"
                        "示例分析完成后 Autopilot 会自动继续。"
                    ),
                }
            else:
                result = {
                    "status": "started",
                    "message": (
                        "闭环服务已启动；检测到新知识后会自动进入下一轮"
                        if paused_fingerprint else "闭环服务已启动"
                    ),
                }
        _append_control_audit({
            **record,
            "event": "BOARD_WORKFLOW_STARTED" if action == "start" else "BOARD_WORKFLOW_PAUSED",
            "result": result["status"],
        }, audit_path)
        return {**result, "commands": commands}


def manual_learning_status(project_root: Path = ROOT) -> dict:
    approved = approved_manual_edit_records(project_root)
    reviewed = approved_manual_review_records(DATASET)
    statuses: dict[str, int] = {}
    review_decisions: dict[str, int] = {}
    for row in reviewed:
        decision = str(row.get("decision") or "unknown")
        review_decisions[decision] = review_decisions.get(decision, 0) + 1
    edits = project_root / "runtime/manual_annotations/edits"
    if edits.is_dir():
        for provenance in edits.glob("*/provenance.json"):
            try:
                row = json.loads(provenance.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            status = str(row.get("status") or "UNKNOWN")
            statuses[status] = statuses.get(status, 0) + 1
    return {
        "approved_examples": len(approved),
        "approved_no_edit_reviews": len(reviewed),
        "review_decisions": review_decisions,
        "edit_statuses": statuses,
        "contract": "APPLIED + operator + applied_hash",
    }


def maintenance_chat_context(snapshot: dict | None = None) -> dict:
    """Build a bounded, de-identified control-plane context for the chat model."""
    data = snapshot or collect()
    state = data.get("state") or {}
    branches = {}
    for role in ("failure_triage", "cmr_specialist", "lge_myo_specialist",
                 "lge_mi_specialist", "supervisor", "independent_reviewer"):
        row = next((item for item in data.get("agents", [])
                    if item.get("who") == role or str(item.get("who", "")).startswith(role + ":")), None)
        branches[role] = ({key: row.get(key) for key in
                           ("time", "run_id", "model", "status", "detail")} if row else None)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            timeout=5).strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, text=True,
            capture_output=True, timeout=5, check=False).stdout.splitlines()
        dirty = [line[3:] for line in status if len(line) > 3][:20]
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", []
    return {
        "generated_at": data.get("generated_at"),
        "autopilot": {
            "status": state.get("status"), "paused": bool(state.get("paused")),
            "last_event": state.get("last_event"), "last_event_at": state.get("last_event_at"),
            "cycles": state.get("cycles"), "stagnant": state.get("stagnant"),
        },
        "services": {name: (pid != "-") for name, pid in (data.get("controllers") or {}).items()},
        "check_buckets": data.get("boundary_review") or {},
        "latest_runs": [{key: row.get(key) for key in ("run_id", "status", "current_step")}
                        for row in (data.get("runs") or [])[:3]],
        "recent_events": (data.get("events") or [])[:8],
        "agent_branches": branches,
        "repository": {"commit": commit, "dirty_paths": dirty,
                       "dirty_count_visible": len(dirty)},
        "responsibility_boundary": MAINTENANCE_CHAT_BOUNDARY,
    }


def maintenance_chat(messages: list[dict], snapshot: dict | None = None) -> dict:
    """Ask the configured Qwen maintainer without granting shell or write access."""
    cleaned = []
    for item in messages[-12:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:5000]})
    if not cleaned or cleaned[-1]["role"] != "user":
        raise ValueError("最后一条消息必须来自用户")
    from app.board_provider import generate_reply, subscription_active
    subscription = subscription_active("workflow_maintainer", ROOT)
    context = maintenance_chat_context(snapshot)
    model = (os.environ.get("ALIYUN_MAINTENANCE_CHAT_MODEL")
             or os.environ.get("ALIYUN_TRIAGE_MODEL")
             or os.environ.get("ALIYUN_MODEL") or "qwen3.6-flash")
    system = (
        "你是 CMR/LGE orchestrator 的项目维护助手，只负责控制面。"
        "你可以诊断服务、调度、角色交接、配置、API、测试、部署、UI 和文档问题；"
        "不得查看影像作医学判断，不得决定、修改、删除或路由任何标签，不得绕过复核和审计。"
        "你当前没有 shell 和文件写权限；需要修改时给出最小修复计划，明确目标文件、验收测试、风险和回滚点，"
        "并注明必须走隔离快照、测试及独立复核。只根据随附状态作答，不要编造运行结果。"
        "回答使用简体中文，先给结论，再给证据与下一步；避免堆砌细节。"
    )
    started = time.monotonic()
    request_messages = [
            {"role": "system", "content": system},
            {"role": "system", "content": "当前去标识化控制面状态：" + json.dumps(
                context, ensure_ascii=False, separators=(",", ":"))},
            *cleaned,
        ]
    if subscription:
        result = generate_reply("workflow_maintainer", request_messages, root=ROOT, timeout=150,
                                budget_callback=lambda: None, task_prompt=system)
        # This explicit UI request has its own single-invocation limit; the
        # automatic workflow may remain paused while the user diagnoses it.
        reply = str(result["choices"][0]["message"].get("content") or "").strip()
        elapsed = round(time.monotonic() - started, 1)
        usage = result.get("usage") or {}
        model = result["model"]
        with MAINTENANCE_CHAT_LOCK:
            MAINTENANCE_CHAT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
            with MAINTENANCE_CHAT_AUDIT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model, "request_sha256": hashlib.sha256(cleaned[-1]["content"].encode()).hexdigest(),
                    "elapsed_seconds": elapsed, "total_tokens": usage.get("total_tokens"),
                    "boundary": "control_plane_only", "scope": "interactive_maintenance"}) + "\n")
        return {"reply": reply, "model": model, "worked_for": elapsed,
                "usage": usage, "boundary": MAINTENANCE_CHAT_BOUNDARY}
    key = os.environ.get("DASHSCOPE_API_KEY")
    base = os.environ.get("DASHSCOPE_BASE_URL")
    if not (key and base):
        raise ValueError("Qwen 模型凭据未配置")
    body = json.dumps({
        "model": model,
        "messages": request_messages,
        "max_tokens": 2200,
    }).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=150) as resp:
            result = json.loads(resp.read())
        reply = str(result["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        raise ValueError(f"维护模型调用失败：{type(exc).__name__}: {exc}") from exc
    elapsed = round(time.monotonic() - started, 1)
    usage = result.get("usage") or {}
    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "model": model,
        "request_sha256": hashlib.sha256(cleaned[-1]["content"].encode()).hexdigest(),
        "elapsed_seconds": elapsed, "total_tokens": usage.get("total_tokens"),
        "boundary": "control_plane_only",
    }
    with MAINTENANCE_CHAT_LOCK:
        MAINTENANCE_CHAT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with MAINTENANCE_CHAT_AUDIT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return {"reply": reply, "model": model, "worked_for": elapsed,
            "usage": usage, "boundary": MAINTENANCE_CHAT_BOUNDARY}


def apply_staged_edit_local(edit_id: str, reviewer: str, confirmation: str,
                            *, dataset_root: Path = DATASET,
                            work_root: Path = MANUAL_WORK,
                            audit_path: Path = LABEL_APPLY_AUDIT) -> dict:
    """Apply an operator-confirmed staged edit in the board process, with audit."""
    with LABEL_APPLY_LOCK:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        def audit(event: str, **details: object) -> None:
            record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                      "event": event, "edit_id": edit_id, "reviewer": reviewer,
                      **details}
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        audit("requested")  # Refuse mutation if the audit destination is unavailable.
        try:
            result = apply_edit(dataset_root, work_root, edit_id,
                                reviewer=reviewer, confirmation=confirmation)
        except Exception as exc:
            audit("rejected", error=f"{type(exc).__name__}: {exc}"[:1000])
            raise
        try:
            feedback = release_linked_manual_example(result, reviewer)
            if feedback:
                result["maintenance_feedback"] = feedback
        except (KeyError, TypeError, ValueError, OSError) as exc:
            # The production edit has already committed.  Preserve that fact
            # and expose a retryable control-plane warning instead of claiming
            # the label application failed.
            result["maintenance_feedback_warning"] = str(exc)
        try:
            audit("applied", backup=result.get("backup"),
                  boundary_review_status=result.get("boundary_review_status"),
                  source_run_id=result.get("source_run_id"),
                  source_action_id=result.get("source_action_id"),
                  maintenance_feedback=result.get("maintenance_feedback"),
                  maintenance_feedback_warning=result.get("maintenance_feedback_warning"))
        except OSError as exc:
            # The edit already committed; never report it as an apply failure.
            result["audit_warning"] = f"post-apply audit failed: {exc}"
        return result


def release_linked_manual_example(result: dict, reviewer: str,
                                  *, db_path: Path = DB) -> dict | None:
    """Turn an applied, action-linked example into one audited bounded retry."""
    run_raw = result.get("source_run_id")
    action_id = str(result.get("source_action_id") or "").strip()
    if run_raw in (None, "") and not action_id:
        return None
    if run_raw in (None, "") or not action_id:
        raise ValueError("incomplete maintenance action link on applied edit")
    run_id = int(run_raw)
    database = Database(db_path)
    database.initialize()
    action = database.get_maintenance_action(run_id, action_id)
    expected_pool = pool_for(action_id, action["payload"].get("objective", ""))
    actual_pool = str(result.get("pool") or "")
    if expected_pool and actual_pool != expected_pool:
        raise ValueError(
            f"manual example pool mismatch: expected {expected_pool}, got {actual_pool or '-'}"
        )
    if action["status"] != "HUMAN_REQUIRED":
        return {
            "run_id": run_id, "action_id": action_id,
            "resulting_status": action["status"], "already_released": True,
            "operator_evidence_retry": action["payload"].get("operator_evidence_retry"),
        }
    review = database.review_maintenance_action(
        run_id, action_id,
        decision="retry_once",
        reviewer=str(reviewer or "local_operator")[:80],
        note=(f"已应用人工示例 {result.get('edit_id')}; "
              "将该示例交给既有 Agent、评分与验证链做一次有界再优化。"),
        confirmed=True,
    )
    review["operator_evidence_retry"] = database.attach_operator_evidence_retry(
        run_id, action_id,
        edit_id=str(result.get("edit_id") or ""),
        pool=actual_pool,
    )
    return review

POOLS = {
    "cmr_boundary_manifest": "manifest_backup/cmr_boundary_manifest.csv",
    "cmr_middle_manifest": "manifest_backup/cmr_middle_manifest.csv",
    "lge_boundary_manifest": "manifest_backup/lge_boundary_manifest.csv",
    "lge_issue_manifest": "manifest_backup/lge_issue_manifest.csv",
    "lge_middle_manifest": "manifest_backup/lge_middle_manifest.csv",
    "trajectory_manifest": "manifest_backup/trajectory_manifest.csv",
}
POOL_META = {
    "cmr_boundary_manifest": {
        "label": "CMR 边界/非标准 SAX",
        "target": "CMR cine 心肌 Mask",
        "unit": "空间层（待处理内保留具体问题帧）",
        "task": "判定心尖、心底、多腔或采集失败层是否仍属于可用 SAX",
        "next": "可修复者转入中间层修复；不可用者须与配对 LGE 成套移入 quarantine",
        "queue": "boundary", "aggregate": False,
    },
    "cmr_middle_manifest": {
        "label": "CMR 中间 SAX 待修复",
        "target": "CMR cine 心肌 Mask",
        "unit": "时间帧（同时按患者-层级归组）",
        "task": "修复环缺口、错误连通域、质心跳变和时间不连续",
        "next": "先看相邻时间帧、上下层及配对 LGE；候选必须过 topology gate",
        "queue": "repair", "aggregate": False,
    },
    "lge_boundary_manifest": {
        "label": "LGE 边界/非标准 SAX",
        "target": "lge_checked 心肌 + MI Mask",
        "unit": "空间层",
        "task": "判定心尖、心底、多腔或采集失败层是否仍可用于配对数据集",
        "next": "不可用时连同配对 CMR、图像和全部标签成套隔离，禁止单边删除",
        "queue": "boundary", "aggregate": False,
    },
    "lge_issue_manifest": {
        "label": "LGE 全部问题（汇总索引）",
        "target": "lge_checked 心肌 + MI Mask",
        "unit": "空间层",
        "task": "LGE boundary 与 middle 的全集，不是第三个独立待办池",
        "next": "实际处理请进入 LGE 边界池或 LGE 中间层池",
        "queue": "aggregate", "aggregate": True,
    },
    "lge_middle_manifest": {
        "label": "LGE 中间 SAX 待修复",
        "target": "lge_checked 心肌 + MI Mask",
        "unit": "空间层",
        "task": "校正心肌拓扑、MI 多连通域、MI/心肌包含与层间连续性",
        "next": "维护 checked；结合上下层及图像对比度，保留歧义病例人工复核",
        "queue": "repair", "aggregate": False,
    },
    "trajectory_manifest": {
        "label": "跨层左室位置异常",
        "target": "CMR/LGE 心肌 Mask",
        "unit": "异常帧/层",
        "task": "用患者多数层的左室轨迹校准潜在错位层",
        "next": "只在唯一高置信匹配时修复；位置仍歧义则进入人工编辑器",
        "queue": "trajectory", "aggregate": False,
    },
}
STAGE_RANK = {name: i for i, name in enumerate(
    ["PROPOSED", "APPROVED", "RUNNING", "NEEDS_REVIEW", "SUCCEEDED", "VALIDATED", "PROMOTED",
     "HUMAN_REQUIRED", "FAILED", "REJECTED"])}
GAPS = {
    "无动作": "还没有提案", "PROPOSED": "已提案，等待独立评审",
    "APPROVED": "智能体交接与独立评审已通过，待所属角色执行", "RUNNING": "所属角色正在隔离实现/验证",
    "NEEDS_REVIEW": "待 gate 与晋级", "SUCCEEDED": "待 gate 与晋级",
    "PROMOTED": "评审通过，待应用到生产并重跑 QC",
    "HUMAN_REQUIRED": "重试耗尽，需人工判断", "FAILED": "失败，待重试", "REJECTED": "已否决",
}

_counts_cache: dict[str, object] = {"at": 0.0, "value": {}}
_inventory_cache: dict[str, object] = {"at": 0.0, "value": {}}


def local_time(value, with_date: bool = False) -> str:
    """The DB stores UTC; the operator reads Beijing time."""
    if not value:
        return "-"
    text = str(value)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return text[11:19] if len(text) > 19 else text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone()
    return dt.strftime("%m-%d %H:%M:%S") if with_date else dt.strftime("%H:%M:%S")


def pid_of(pattern: str) -> str:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=4).stdout.split()
        return out[0] if out else "-"
    except Exception:
        return "-"


def pool_for(action_id: str, objective: str) -> str:
    key = action_id.lower()
    if "lge" in key:
        if any(t in key for t in ("mi", "infarct", "scar", "frag")):
            return "lge_issue_manifest"
        if "paired" in key or "myo" in key:
            return "lge_boundary_manifest"
        return "lge_issue_manifest"
    if "traj" in key or "centroid" in key:
        return "trajectory_manifest"
    if "cmr" in key:
        if any(t in key for t in ("small", "area", "middle", "excess")):
            return "cmr_middle_manifest"
        return "cmr_boundary_manifest"
    if "evidence" in key or "dedup" in key:
        return "cmr_boundary_manifest"
    return ""


def pool_counts(ttl: float = 60.0) -> dict[str, int]:
    now = time.time()
    if now - float(_counts_cache["at"]) < ttl and _counts_cache["value"]:
        return _counts_cache["value"]  # type: ignore[return-value]
    counts = {}
    for key in POOLS:
        path = DATASET / MANIFESTS[key]
        try:
            counts[key] = csv_issue_slices(path)
        except OSError:
            counts[key] = 0
    _counts_cache.update(at=now, value=counts)
    return counts


def pool_inventory(ttl: float = 60.0) -> dict[str, dict]:
    """Describe the actual review unit behind each manifest, not just row counts."""
    now = time.time()
    if now - float(_inventory_cache["at"]) < ttl and _inventory_cache["value"]:
        return _inventory_cache["value"]  # type: ignore[return-value]
    output: dict[str, dict] = {}
    for key in POOLS:
        path = DATASET / MANIFESTS[key]
        patients: set[str] = set()
        levels: set[tuple[str, str]] = set()
        checks: set[str] = set()
        issue_counts: dict[str, int] = {}
        rows = 0
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows += 1
                    patient = str(row.get("patient") or row.get("patient_id") or "")
                    level = str(row.get("volume") or row.get("slice") or "")
                    check = str(row.get("check_view") or row.get("check") or "")
                    if patient:
                        patients.add(patient)
                    if patient or level:
                        levels.add((patient, level))
                    if check:
                        checks.add(check)
                    raw_issues = str(row.get("issues") or row.get("decision") or "")
                    for issue in filter(None, (part.strip() for part in raw_issues.split(";"))):
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
        except OSError:
            pass
        output[key] = {
            "rows": rows,
            "review_units": len(checks) or rows,
            "patients": len(patients),
            "levels": len(levels),
            "top_issues": [
                {"issue": issue, "count": count}
                for issue, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
            ],
        }
    _inventory_cache.update(at=now, value=output)
    return output


def refresh_board_caches() -> None:
    _counts_cache.update(at=0.0, value={})
    _inventory_cache.update(at=0.0, value={})


TRANS_CACHE = ROOT / "runtime/translation_cache.json"
_trans_cache: dict = {}
_trans_loaded = False


def _load_trans_cache() -> None:
    global _trans_loaded
    if _trans_loaded:
        return
    _trans_loaded = True
    if TRANS_CACHE.exists():
        try:
            _trans_cache.update(json.loads(TRANS_CACHE.read_text(encoding="utf-8")))
        except Exception:
            pass


def translate_texts(texts: list, brief: bool = False) -> list:
    """Batch-translate to Simplified Chinese via the configured DashScope model.

    Falls back to the original text when no key is configured or the call fails,
    so the board never breaks because of translation.
    """
    _load_trans_cache()
    translate_texts.last_error = None
    out = [None] * len(texts)
    todo = []
    for i, text in enumerate(texts):
        digest = hashlib.sha256((("brief:" if brief else "full:") + text).encode("utf-8")).hexdigest()
        if digest in _trans_cache:
            out[i] = _trans_cache[digest]
        else:
            todo.append((i, digest, text))
    from app.board_provider import subscription_active
    if todo and subscription_active("efficiency_auditor", ROOT):
        for i, _, text in todo:
            out[i] = text
        return out
    key = os.environ.get("DASHSCOPE_API_KEY")
    base = os.environ.get("DASHSCOPE_BASE_URL")
    if todo and key and base:
        try:
            items = [t for _, _, t in todo]
            body = json.dumps({
                "model": os.environ.get("ALIYUN_TRANSLATE_MODEL")
                         or os.environ.get("ALIYUN_MODEL") or "qwen3.6-flash",
                "messages": [
                    {"role": "system", "content": (
                        "你是医学影像 QC 系统的中文技术编辑。把用户给的 JSON 字符串数组逐条改写成简体中文。"
                        "硬性长度要求：长段落必须压缩到 60 字以内，格式为『做什么 + 关键约束』一句话；"
                        "单条指标压缩到 25 字以内的短语。只保留目的、对象、阈值数字，删掉实现细节和枚举。"
                        "不要遗漏条目。"
                        if brief else
                        "你是医学影像术语翻译。把用户给的 JSON 字符串数组逐条翻译成简体中文，"
                        "术语保持专业准确。")
                        + '只返回 JSON 对象 {"t": [结果...]}，长度与输入一致。'},
                    {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
                ],
            }).encode("utf-8")
            req = urllib.request.Request(
                base.rstrip("/") + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=150) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            start, end = content.find("{"), content.rfind("}")
            parsed = json.loads(content[start:end + 1]) if start >= 0 < end else {}
            arr = parsed.get("t") if isinstance(parsed, dict) else parsed
            if isinstance(arr, list) and len(arr) == len(todo):
                for (i, digest, _), tr in zip(todo, arr):
                    if isinstance(tr, str) and tr.strip():
                        _trans_cache[digest] = tr
                        out[i] = tr
                TRANS_CACHE.write_text(json.dumps(_trans_cache, ensure_ascii=False),
                                       encoding="utf-8")
        except Exception as exc:
            translate_texts.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[translate] {translate_texts.last_error}", flush=True)
    for i, text in enumerate(texts):
        if out[i] is None:
            out[i] = text
    return out


def artifact_summary(path_str):
    """Structured summary of an isolated-sandbox artifact, for the review modal."""
    if not path_str:
        return None
    manifest = Path(path_str) / "maintenance_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        r = data.get("result") or {}
        return {
            "ready": bool(r.get("ready_for_validation")),
            "summary": str(r.get("summary") or "")[:500],
            "changed_files": [str(x) for x in (r.get("changed_files") or [])][:8],
            "tests": [str(x) for x in (r.get("tests_run") or [])][:6],
        }
    except Exception:
        return None


BRIEF_CACHE = ROOT / "runtime/escalation_cache.json"
_brief_cache: dict = {}
_brief_loaded = False


def _load_brief_cache() -> None:
    global _brief_loaded
    if _brief_loaded:
        return
    _brief_loaded = True
    if BRIEF_CACHE.exists():
        try:
            _brief_cache.update(json.loads(BRIEF_CACHE.read_text(encoding="utf-8")))
        except Exception:
            pass


def _action_facts(row, payload: dict) -> dict:
    """Deterministic escalation facts: attempts, sandbox verdict, reviewer flags."""
    root = ROOT / "runtime/maintenance" / f"run_{int(row['run_id']):06d}"
    workspaces = sorted(root.glob(f"{row['action_id']}_*")) if root.exists() else []
    facts = {"status": row["status"], "action_class": row["action_class"],
             "owner_role": row["owner_role"],
             "attempts": sum(1 for w in workspaces if (w / "maintenance_manifest.json").exists()),
             "max_attempts": payload.get("max_attempts")}
    if workspaces:
        man = workspaces[-1] / "maintenance_manifest.json"
        if man.exists():
            try:
                r = json.loads(man.read_text(encoding="utf-8")).get("result") or {}
                facts["sandbox_self_test_passed"] = bool(r.get("ready_for_validation"))
                facts["remaining_risks"] = [str(x)[:110] for x in (r.get("remaining_risks") or [])][:3]
                log = workspaces[-1] / "test_output.log"
                if log.exists():
                    bad = [l.strip()[:110] for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                           if ("Error" in l or "FAIL" in l) and l.strip()]
                    facts["test_errors"] = bad[-3:]
            except Exception:
                pass
    try:
        db2 = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        db2.row_factory = sqlite3.Row
        rev = db2.execute(
            "select output_json from agent_outputs where run_id=? and role='independent_reviewer' "
            "order by output_id desc limit 1", (int(row["run_id"]),)).fetchone()
        db2.close()
        if rev:
            rd = json.loads(rev["output_json"])
            facts["reviewer_decision"] = rd.get("decision")
            facts["reviewer_risk_flags"] = [str(x)[:110] for x in (rd.get("risk_flags") or [])][:3]
    except Exception:
        pass
    return facts


def _fallback_brief(f: dict):
    verdict = "通过" if f.get("sandbox_self_test_passed") else "未通过"
    why = (f"隔离实验重试 {f.get('attempts', 0)} 次后沙箱自测仍{verdict}，"
           f"超出自动重试上限，需人工裁定")
    blockers = ([x for x in f.get("test_errors", [])][:1]
                + [x for x in f.get("remaining_risks", [])][:2]
                + [x for x in f.get("reviewer_risk_flags", [])][:1])
    return why, (blockers or ["无结构化失败信息，见沙箱产物"])[:3]


def _llm_brief(facts: dict):
    from app.board_provider import subscription_active
    if subscription_active("efficiency_auditor", ROOT):
        return None, None
    key = os.environ.get("DASHSCOPE_API_KEY")
    base = os.environ.get("DASHSCOPE_BASE_URL")
    if not (key and base):
        return None, None
    try:
        body = json.dumps({
            "model": os.environ.get("ALIYUN_TRANSLATE_MODEL") or os.environ.get("ALIYUN_MODEL") or "qwen3.6-flash",
            "messages": [
                {"role": "system", "content":
                    "你是心脏MRI QC系统的人工干预简报助手。根据升级事实 JSON 输出 "
                    "{\"why\":\"...\",\"blockers\":[...]}。why=该动作升级为人工处理的唯一核心原因，"
                    "40字以内；blockers=过不了或无法自动判定的核心点，每条18字以内，最多3条。"
                    "只准使用给定事实，禁止编造，不写背景。"},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
            ]}).encode("utf-8")
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        a, b = content.find("{"), content.rfind("}")
        parsed = json.loads(content[a:b + 1])
        why = str(parsed.get("why", ""))[:80]
        blockers = [str(x)[:40] for x in (parsed.get("blockers") or [])][:3]
        if why:
            return why, blockers
    except Exception as exc:
        print(f"[brief] {type(exc).__name__}: {exc}", flush=True)
    return None, None


def escalation_brief(run_id_raw, action_id):
    try:
        run_id = int(run_id_raw)
    except (TypeError, ValueError):
        return {"error": "bad run_id"}
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    row = db.execute(
        "select run_id,action_id,owner_role,action_class,status,payload_json "
        "from maintenance_actions where run_id=? and action_id=?", (run_id, action_id)).fetchone()
    db.close()
    if not row:
        return {"error": "not found"}
    payload = json.loads(row["payload_json"] or "{}")
    facts = _action_facts(row, payload)
    digest = hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    _load_brief_cache()
    entry = _brief_cache.get(digest)
    if entry is None:
        why, blockers = _llm_brief(facts)
        if why is not None:
            entry = {"why": why, "blockers": blockers}
            _brief_cache[digest] = entry
            try:
                BRIEF_CACHE.write_text(json.dumps(_brief_cache, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        else:
            # 不缓存兜底结果：下次请求还有机会拿到 LLM 精简版
            entry = {"why": _fallback_brief(facts)[0], "blockers": _fallback_brief(facts)[1]}
    out = dict(entry)
    out.update({"objective": payload.get("objective") or "",
                "acceptance_metrics": payload.get("acceptance_metrics") or [],
                "owner_role": row["owner_role"], "action_class": row["action_class"],
                "status": row["status"], "run_id": run_id, "action_id": row["action_id"],
                "facts": facts})
    return out


def editor_action(run_id_raw, action_id):
    try:
        run_id = int(run_id_raw)
    except (TypeError, ValueError):
        return {"error": "bad run_id"}
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    row = db.execute(
        "select run_id,action_id,owner_role,action_class,status,payload_json,artifact_path "
        "from maintenance_actions where run_id=? and action_id=?", (run_id, action_id)).fetchone()
    db.close()
    if not row:
        return {"error": "not found"}
    payload = json.loads(row["payload_json"] or "{}")
    return {"run_id": row["run_id"], "action_id": row["action_id"],
            "owner_role": row["owner_role"], "action_class": row["action_class"],
            "status": row["status"], "objective": payload.get("objective") or "",
            "acceptance_metrics": payload.get("acceptance_metrics") or [],
            "artifact_summary": artifact_summary(row["artifact_path"])}


def _db_epoch(value: object) -> float:
    """Convert a DB timestamp to epoch; DB-naive timestamps are UTC."""
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _read_json(path: Path) -> dict | None:
    """Read a complete JSON object, tolerating concurrently written files."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def agent_activity(
    db: sqlite3.Connection,
    limit: int = 25,
    manual_root: Path = MANUAL_WORK,
    maintenance_root: Path = ROOT / "runtime/maintenance",
) -> list[dict]:
    """Merge analysis, prompt-edit and maintenance activity into one timeline."""
    rows: list[dict] = []
    for record in db.execute(
        "select run_id,role,provider,model,status,created_at from agent_outputs "
        "order by output_id desc limit ?", (max(limit * 2, 50),)
    ):
        item = dict(record)
        rows.append({
            "epoch": _db_epoch(item.get("created_at")),
            "time": local_time(item.get("created_at")),
            "src": "分析", "run_id": item.get("run_id", "-"),
            "who": item.get("role") or "agent",
            "model": item.get("model") or item.get("provider") or "-",
            "status": item.get("status") or "UNKNOWN",
            "detail": item.get("provider") or "",
        })

    prompt_files = sorted(
        (manual_root / "prompt_jobs").glob("*/agent_plan.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:max(limit * 2, 50)]
    for path in prompt_files:
        data = _read_json(path)
        if data is None:
            continue
        plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}
        mtime = path.stat().st_mtime
        rows.append({
            "epoch": mtime, "time": datetime.fromtimestamp(mtime).strftime("%H:%M:%S"),
            "src": "编辑", "run_id": "-",
            "who": data.get("role") or "edit_agent",
            "model": data.get("model") or data.get("provider") or "-",
            "status": str(plan.get("decision") or "PLANNED").upper(),
            "detail": str(plan.get("rationale") or "")[:120],
        })

    maintenance_files = sorted(
        maintenance_root.glob("run_*/*/maintenance_manifest.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:max(limit * 2, 50)]
    for path in maintenance_files:
        data = _read_json(path)
        if data is None:
            continue
        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        run_name = next((part for part in path.parts if part.startswith("run_")), "")
        run_id = run_name.removeprefix("run_").lstrip("0") or "0"
        exit_code = data.get("exit_code")
        ready = bool(result.get("ready_for_validation"))
        status = "FAILED" if exit_code not in (None, 0) else ("SUCCEEDED" if ready else "NEEDS_REVIEW")
        changed = result.get("changed_files") if isinstance(result.get("changed_files"), list) else []
        mtime = path.stat().st_mtime
        rows.append({
            "epoch": mtime, "time": datetime.fromtimestamp(mtime).strftime("%H:%M:%S"),
            "src": "维护", "run_id": int(run_id) if run_id.isdigit() else run_id,
            "who": "maintenance_executor",
            "model": result.get("executor") or "-", "status": status,
            "detail": f"变更 {len(changed)} 个文件；{str(result.get('summary') or '')[:90]}",
        })

    rows.sort(key=lambda item: float(item.get("epoch") or 0), reverse=True)
    for item in rows:
        item.pop("epoch", None)
    return rows[:limit]


def batch_progress(root: Path = ROOT / "runtime/batch_candidates") -> dict:
    """Return the freshest complete/partial batch report for the header pill."""
    candidates = sorted(
        root.glob("*_report*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ) if root.exists() else []
    for path in candidates:
        data = _read_json(path)
        if data is None:
            continue
        partial = ".partial." in path.name
        pool = str(data.get("pool") or path.name.split("_report", 1)[0])
        return {
            "pool": pool, "scanned": int(data.get("scanned") or 0),
            "flagged": int(data.get("flagged") or 0),
            "auto_fixed": int(data.get("auto_fixed") or 0),
            "changed_pixels": int(data.get("changed_pixels_total") or 0),
            "status": "RUNNING" if partial else "COMPLETED",
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S"),
        }
    return {"status": "IDLE", "pool": "", "scanned": 0, "flagged": 0,
            "auto_fixed": 0, "changed_pixels": 0, "updated_at": "-"}


def batch_summary(root: Path = ROOT / "runtime/batch_candidates") -> dict:
    """Aggregate completed staging reports without counting partial reruns."""
    rows = []
    paths = sorted(root.glob("*_report.json")) if root.exists() else []
    for path in paths:
        data = _read_json(path)
        if data is None:
            continue
        rows.append({
            "pool": str(data.get("pool") or path.name.removesuffix("_report.json")),
            "scanned": int(data.get("scanned") or 0),
            "flagged": int(data.get("flagged") or 0),
            "auto_fixed": int(data.get("auto_fixed") or 0),
            "changed_pixels": int(data.get("changed_pixels_total") or 0),
        })
    validation = _read_json(root / "candidate_validation.json") or {}
    return {
        "reports": rows, "report_count": len(rows),
        "scanned": sum(row["scanned"] for row in rows),
        "flagged": sum(row["flagged"] for row in rows),
        "auto_fixed": sum(row["auto_fixed"] for row in rows),
        "changed_pixels": sum(row["changed_pixels"] for row in rows),
        "gate_total": int(validation.get("total") or 0),
        "gate_passed": int(validation.get("passed") or 0),
        "gate_rejected": int(validation.get("rejected") or 0),
    }


def rejected_batch_candidates(root: Path = ROOT / "runtime/batch_candidates",
                              limit: int = 50) -> list[dict]:
    validation = _read_json(root / "candidate_validation.json") or {}
    output = []
    for row in validation.get("records", []):
        if row.get("passed"):
            continue
        output.append({
            "pool": row.get("pool"), "index": row.get("index"),
            "operation": row.get("operation") or [],
            "reasons": row.get("reasons") or [], "metrics": row.get("metrics") or {},
        })
        if len(output) >= limit:
            break
    return output


def candidate_gate_detail(root: Path = ROOT / "runtime/batch_candidates",
                          rejected_limit: int = 50,
                          dataset_root: Path = DATASET) -> dict:
    """Turn the deterministic gate output into an operator-facing task list."""
    validation = _read_json(root / "candidate_validation.json") or {}
    passed, rejected = [], []
    reason_counts: dict[str, int] = {}
    manifest_cache: dict[str, list[dict[str, str]]] = {}
    for row in validation.get("records", []):
        pool = str(row.get("pool") or "")
        item = {
            "pool": pool, "index": row.get("index"),
            "operation": row.get("operation") or [],
            "reasons": row.get("reasons") or [], "metrics": row.get("metrics") or {},
        }
        try:
            if pool not in manifest_cache:
                manifest_cache[pool] = _manifest(dataset_root, pool)
            rows = manifest_cache[pool]
            ident = _row_identity(pool, rows[int(row["index"])])
            item.update({
                "patient": ident["patient"], "volume": ident["volume"],
                "frame": ident["frame"], "modality": ident["modality"],
                "sample_name": (
                    f'{ident["patient"]}/{ident["volume"]}#frame={ident["frame"]}'
                ),
            })
        except (KeyError, ValueError, IndexError, FileNotFoundError, OSError):
            item["sample_name"] = "候选样本"
        if row.get("passed"):
            item["task"] = "核对候选与原标签、相邻帧和配对 LGE；确认后才可写回"
            item["review_status"] = "work"
            passed.append(item)
        else:
            item["review_status"] = "rejected"
            rejected.append(item)
            for reason in item["reasons"]:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "total": int(validation.get("total") or len(passed) + len(rejected)),
        "passed": len(passed), "rejected": len(rejected),
        "passed_candidates": passed,
        "rejected_candidates": rejected[:rejected_limit],
        "rejected_visible": min(len(rejected), rejected_limit),
        "rejection_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "policy": "gate 通过仅表示拓扑约束通过，仍需人工确认；拒绝项不得写回正式标签",
    }


def active_label_review_tasks(rows: list[dict], dataset_root: Path = DATASET) -> list[dict]:
    """Show only gate-passed candidates that are actually in the work bucket.

    A passed candidate is an automated gate result, not by itself a human task.
    Resolve the current four-bucket state for every CMR pool so pending middle
    candidates do not inflate the operator queue.
    """
    manifest_cache: dict[str, list[dict[str, str]]] = {}
    output = []
    for row in rows:
        pool = str(row.get("pool") or "")
        try:
            if row.get("patient") and row.get("volume"):
                ident = {"patient": row["patient"], "volume": row["volume"]}
            else:
                if pool not in manifest_cache:
                    manifest_cache[pool] = _manifest(dataset_root, pool)
                source = manifest_cache[pool][int(row["index"])]
                resolved = _row_identity(pool, source)
                ident = {"patient": resolved["patient"], "volume": resolved["volume"]}
            if boundary_review_status(dataset_root, ident) != "work":
                continue
        except (KeyError, ValueError, IndexError, FileNotFoundError, OSError):
            continue
        output.append(row)
    return output


def call_summary() -> dict:
    """Summarize real provider HTTP calls, not inferred agent-output rows."""
    log = ROOT / "runtime/agent_calls.jsonl"
    rows = []
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-5000:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    cutoff = time.time() - 24 * 3600
    recent = [row for row in rows if _db_epoch(row.get("timestamp")) >= cutoff]
    models: dict[str, dict] = {}
    for row in recent:
        item = models.setdefault(str(row.get("model") or "unknown"),
                                 {"calls": 0, "ok": 0, "errors": 0, "tokens": 0,
                                  "last_at": "-", "roles": set()})
        item["calls"] += 1
        item["ok"] += int(row.get("status") == "SUCCEEDED")
        item["errors"] += int(row.get("status") == "ERROR")
        item["tokens"] += int(row.get("total_tokens") or 0)
        item["last_at"] = local_time(row.get("timestamp"))
        item["roles"].add(str(row.get("role") or "unspecified"))
    model_rows = [{**value, "model": model, "roles": sorted(value["roles"])}
                  for model, value in models.items()]
    model_rows.sort(key=lambda row: row["model"])
    health = _read_json(ROOT / "runtime/agent_model_health.json") or {}
    return {"calls": len(recent), "tokens": sum(int(x.get("total_tokens") or 0) for x in recent),
            "errors": sum(x.get("status") == "ERROR" for x in recent),
            "models": model_rows, "health": health}


def evaluation_summary() -> dict:
    rules = _read_json(ROOT / "runtime/eval_set/v1/rules_v2_report.json") or {}
    quality = _read_json(ROOT / "artifacts/image_label_quality_v1/defect_validation.json") or {}
    score = _read_json(ROOT / "artifacts/image_label_quality_v1/summary.json") or {}
    return {"rules": rules, "quality": quality, "score_groups": score.get("groups", {})}


def check_iteration_summary() -> dict:
    """Show measured check-loop results, never infer acceptance from a pixel gate."""
    path = ROOT / "runtime/check_iterations/latest.json"
    report = _read_json(path) or {}
    return {key: report.get(key) for key in (
        "created_at", "status", "pending", "work", "checked", "trash",
        "scanned_frames", "candidate_frames", "gate_passed_frames",
        "volumes_with_candidates", "counterfactual_salvage", "residual_pending",
        "residual_clusters", "promotion", "agent_decision", "reviewer_decision",
        "holdout_count", "holdout_sha256", "control_gate_status",
        "spatial_boundary_candidates", "checked_false_routes",
        "candidate_quality", "next_step", "report_path")}


def workflow_outcome_summary(actions: list[dict], state: dict, boundary: dict) -> dict:
    """Keep maintenance-code promotion separate from label acceptance."""
    promoted = [a for a in actions if a.get("status") == "PROMOTED"]
    latest = max(promoted, key=lambda a: int(a.get("run_id") or 0), default=None)
    return {
        "code_promoted_actions": len(promoted),
        "latest_code_promotion": ({
            "run_id": latest["run_id"], "action_id": latest["action_id"],
        } if latest else None),
        "rule_shadow_validation": "NOT_ESTABLISHED",
        "label_resolution": "NOT_ESTABLISHED",
        "pending_labels": int(boundary.get("pending") or 0),
        "checked_labels": int(boundary.get("checked") or 0),
        "autopilot_status": state.get("status") or "UNKNOWN",
    }


def reviewer_validation_backlog(db: sqlite3.Connection) -> list[dict]:
    """Expose only de-identified, still-unmet independent-review requirements."""
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                      "AND name='agent_messages'").fetchone():
        return []
    rows = db.execute(
        "SELECT m.run_id,a.action_id,a.status,m.payload_json FROM agent_messages m "
        "JOIN maintenance_actions a ON a.run_id=m.run_id "
        "AND m.thread_id=('run-' || a.run_id || ':' || a.action_id) "
        "WHERE m.sender_role='independent_reviewer' "
        "AND m.message_type IN ('action_review','implementation_review') "
        "ORDER BY m.message_id DESC LIMIT 100"
    )
    backlog, seen = [], set()
    for row in rows:
        key = (row["run_id"], row["action_id"])
        if key in seen:
            continue
        seen.add(key)
        if row["status"] not in {"PROMOTED", "HUMAN_REQUIRED"}:
            continue
        try:
            feedback = safe_review_feedback(json.loads(row["payload_json"]))
        except json.JSONDecodeError:
            continue
        metrics = feedback.get("required_validation_metrics") or []
        if metrics:
            backlog.append({"run_id": row["run_id"], "action_id": row["action_id"],
                            "status": row["status"], "review_decision": feedback["decision"],
                            "required_validation_metrics": metrics})
        if len(backlog) >= 6:
            break
    return backlog


def collect(*, refresh: bool = False) -> dict:
    if refresh:
        refresh_board_caches()
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    runs = [dict(r) for r in db.execute(
        "select run_id,status,execute_mode,current_step,created_at from workflow_runs "
        "order by run_id desc limit 8")]

    agents = agent_activity(db)

    review_rows = []
    if db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='human_action_reviews'"
    ).fetchone():
        review_rows = [dict(r) for r in db.execute(
            "SELECT * FROM human_action_reviews ORDER BY review_id DESC LIMIT 500"
        )]
    reviews_by_action: dict[tuple[int, str], list[dict]] = {}
    for review in review_rows:
        review["time"] = local_time(review.get("created_at"), with_date=True)
        reviews_by_action.setdefault(
            (int(review["run_id"]), str(review["action_id"])), []
        ).append(review)

    handoffs: list[dict] = []
    if db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_messages'"
    ).fetchone():
        for message in db.execute(
            "SELECT message_id,run_id,thread_id,sender_role,recipient_role,"
            "message_type,reply_to,created_at FROM agent_messages "
            "ORDER BY message_id DESC LIMIT 100"
        ):
            item = dict(message)
            item["time"] = local_time(item.get("created_at"), with_date=True)
            handoffs.append(item)

    invocation_traces: list[dict] = []
    if db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_invocations'"
    ).fetchone():
        for trace in db.execute(
            "SELECT i.invocation_id,i.run_id,i.thread_id,i.role,i.provider,"
            "i.requested_model,i.input_hash,i.status,i.started_at,i.finished_at,"
            "e.actual_model,e.status AS call_status,e.fallback_reason,e.request_id,"
            "e.latency_ms,e.total_tokens FROM agent_invocations i "
            "LEFT JOIN (SELECT invocation_id,MAX(event_id) AS event_id "
            "FROM agent_invocation_events GROUP BY invocation_id) latest "
            "ON latest.invocation_id=i.invocation_id "
            "LEFT JOIN agent_invocation_events e ON e.event_id=latest.event_id "
            "ORDER BY i.started_at DESC LIMIT 100"
        ):
            item = dict(trace)
            item["time"] = local_time(item.get("started_at"), with_date=True)
            invocation_traces.append(item)

    actions, covered = [], {}
    for r in db.execute("select run_id,action_id,owner_role,action_class,status,payload_json "
                        ",artifact_path,created_at,finished_at "
                        "from maintenance_actions order by run_id desc, action_id"):
        row = dict(r)
        payload = json.loads(row.pop("payload_json") or "{}")
        pool = pool_for(row["action_id"], payload.get("objective", ""))
        row["pool"] = pool
        row["pool_path"] = str(CHECK / POOLS[pool]) if pool in POOLS else ""
        row["task_label"] = POOL_META.get(pool, {}).get("label", "")
        row["objective"] = payload.get("objective") or ""
        row["acceptance_metrics"] = payload.get("acceptance_metrics") or []
        row["priority"] = payload.get("priority")
        row["max_attempts"] = payload.get("max_attempts")
        row["auto_execute"] = bool(payload.get("auto_execute", False))
        row["reviews"] = reviews_by_action.get(
            (int(row["run_id"]), str(row["action_id"])), []
        )
        actions.append(row)
        if pool:
            rank = STAGE_RANK.get(row["status"], 0)
            if pool not in covered or rank > covered[pool][0]:
                covered[pool] = (rank, row["status"])

    counts = pool_counts()
    boundary_state = boundary_review_counts(DATASET)
    inventory = pool_inventory()
    pools = []
    for key in POOLS:
        meta = POOL_META[key]
        # Historical maintenance actions are diagnostics, not the medical-label
        # workflow stage.  The operator-facing stage comes from the pool's task.
        if meta["queue"] == "aggregate":
            stage, gap = "汇总索引", meta["next"]
        elif meta["queue"] == "boundary":
            stage, gap = "待可用性判定", meta["next"]
        elif meta["queue"] == "trajectory":
            stage, gap = "待位置校准", meta["next"]
        else:
            stage, gap = "待标签修复", meta["next"]
        visible_count = counts.get(key, 0)
        if key == "cmr_boundary_manifest":
            visible_count = boundary_state["pending"] + boundary_state["work"]
            stage = f'待筛 {boundary_state["pending"]} / 待处理 {boundary_state["work"]}'
        pools.append({
            "key": key, "label": meta["label"], "count": visible_count,
            "stage": stage, "gap": gap, "target": meta["target"],
            "unit": meta["unit"], "task": meta["task"],
            "aggregate": meta["aggregate"], **inventory.get(key, {}),
        })

    events = state.get("events", [])[-40:]
    last = events[-1] if events else {}
    maintenance_human_required = [row for row in actions if row["status"] == "HUMAN_REQUIRED"]
    gate = candidate_gate_detail()
    label_review_tasks = active_label_review_tasks(gate["passed_candidates"])
    operator = operator_control_state()
    learning = manual_learning_status()
    workflow_dependency = enrich_workflow_dependency(
        dependent_workflow_status(DATASET, ROOT), [{"actions": actions[:50]}]
    )
    agent_continuity = build_agent_continuity(Database(DB))
    try:
        validation_report = review_state(DATASET, ROOT / "runtime").get("report", {})
    except (OSError, ValueError, TypeError, KeyError):
        validation_report = {}
    validation_selected = int(validation_report.get("selected") or 0)
    validation_reviewed = int(validation_report.get("reviewed") or 0)
    validation_remaining = max(0, validation_selected - validation_reviewed)
    fuse_paused = bool(state.get("paused_issue_fingerprint"))
    fix_available = framework_fix_available(state)
    attention = {
        "required": bool(not operator.get("paused") and fuse_paused
                         and (boundary_state["pending"] or boundary_state["work"])),
        "title": "需要少量人工示例以继续自我优化",
        "message": (
            "当前证据已完成一轮有界优化。请在编辑器确认并应用 1–2 个代表样本；"
            "示例学习 Agent 会提取已确认差分，Autopilot 检测到新知识后自动续跑。"
        ),
        "href": "/editor?pool=cmr_boundary_manifest&review_status=pending&mode=manual",
        "action": "复核代表样本",
        "validation_remaining": validation_remaining,
        "validation_href": "/check-validation",
    }
    autonomy = autonomy_status(ROOT)
    for credit in autonomy.get("feedback", {}).get("role_credit", []):
        role = (agent_continuity.get("roles") or {}).get(credit.get("role"))
        if role is not None:
            role.setdefault("actual_work",[]).append(credit)
    lge_execution = lge_execution_status(ROOT)
    if not operator.get("paused"):
        review_items = [item for stage in lge_execution.get("stages", {}).values()
                        for item in stage.get("manual_review_items", [])]
        review_items = [item for item in review_items if item.get("reason") != "冻结抽样复核"]
        active_stage = (lge_execution.get("stages") or {}).get(workflow_dependency.get("stage"), {})
        if review_items and active_stage.get(str(workflow_dependency.get("stage")) + "_auto_pending") == 0:
            attention.update(required=True, title="LGE 有样本需要人工复核",
                message="自动流程会保留无法可靠处理的样本。可先确认 1–2 个代表样本，已应用的修改会进入知识链。",
                href=review_items[0]["href"], action="复核 LGE 样本")
    if not operator.get("paused") and (workflow_dependency.get("protocol_repair_required") or autonomy.get("consumer", {}).get("status") in {"FAILED","STALE"}):
        attention.update(required=True,title="执行故障正在等待流程维护",
                         message="这是接口或服务故障，不需要医生逐例修改标签。请查看流程维护模块和执行状态。",
                         href="#agent-flow",action="查看流程维护")
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "UTC" + datetime.now().astimezone().strftime("%z")[:3] + ":" + datetime.now().astimezone().strftime("%z")[3:],
        "controllers": {
            "autopilot": pid_of("scripts/autopilot.py"),
            "board": pid_of("scripts/board_server.py"),
            "worker": pid_of("cli.py worker"),
            "manual_edit_agent": pid_of("analyze_manual_edit_queue.py --watch"),
            "check_loop": pid_of("scripts/check_feedback_loop.py"),
        },
        "state": {
            "cycles": state.get("cycles"), "stagnant": state.get("stagnant_cycles"),
            "last_event": last.get("event"), "last_event_at": local_time(last.get("timestamp")),
            "cooldown": state.get("codex_cooldown_until"), "escalated_until": state.get("escalated_until"),
            "status": state.get("status"), "fingerprint": (state.get("last_issue_fingerprint") or "")[:12],
            "paused": bool(operator.get("paused") or fuse_paused),
            "operator_paused": bool(operator.get("paused")),
            "pause_reason": ("operator" if operator.get("paused") else
                             "no_progress" if fuse_paused else None),
            "framework_fix_available": fix_available,
            "start_action": "resume_after_fix" if fix_available else "start",
        },
        "runs": runs, "agents": agents, "actions": actions,
        "agent_handoffs": handoffs, "agent_handoff_count": len(handoffs),
        "agent_invocations": invocation_traces,
        "agent_invocation_count": len(invocation_traces),
        "calls": call_summary(), "evaluation": evaluation_summary(),
        "aliyun_billing": aliyun_billing_status(),
        "check_iteration": check_iteration_summary(),
        "workflow_outcome": workflow_outcome_summary(actions, state, boundary_state),
        "validation_backlog": reviewer_validation_backlog(db),
        "batch": batch_progress(), "batch_summary": batch_summary(),
        "candidate_gate": gate, "batch_rejected": gate["rejected_candidates"],
        "boundary_review": boundary_state,
        "operator_control": operator,
        "manual_learning": learning,
        "agent_continuity": agent_continuity,
        "lge_execution": lge_execution,
        "autonomy": autonomy,
        "rsi": {**agent_continuity.get("evolution", {}),
                "quality_benchmarks": benchmark_status(ROOT), "generator_benchmark": generator_benchmark_status(ROOT), "installation": consumer_gate(ROOT)},
        "workflow_dependency": workflow_dependency,
        "segmentation_lifecycle": __import__("app.segmentation_status", fromlist=["status"]).status(ROOT),
        "human_attention": attention,
        "check_validation_progress": {
            "selected": validation_selected,
            "reviewed": validation_reviewed,
            "remaining": validation_remaining,
        },
        "label_review_tasks": label_review_tasks,
        "label_review_count": len(label_review_tasks),
        # Compatibility aliases: old clients still receive maintenance actions.
        "human_required": maintenance_human_required,
        "human_required_count": len(maintenance_human_required),
        "maintenance_human_required": maintenance_human_required,
        "reviews": review_rows[:100], "pools": pools,
        "events": [{"t": local_time(e.get("timestamp")), "event": e.get("event"),
                    "detail": e.get("action_id") or (f"run {e.get('run_id')}" if e.get("run_id") else "")}
                   for e in reversed(events)],
    }


def submit_review(payload: dict, db_path: Path = DB) -> dict:
    """Validate and persist one UI review; intentionally label-blind."""
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    try:
        run_id = int(payload["run_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("valid run_id is required") from exc
    action_id = str(payload.get("action_id") or "").strip()
    if not action_id or len(action_id) > 240:
        raise ValueError("valid action_id is required")
    database = Database(db_path)
    database.initialize()
    return database.review_maintenance_action(
        run_id,
        action_id,
        decision=str(payload.get("decision") or ""),
        reviewer=str(payload.get("reviewer") or ""),
        note=str(payload.get("note") or ""),
        confirmed=payload.get("confirmed") is True,
    )


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMR/LGE 闭环看板</title>
<style>
:root{--bg:#0c1118;--card:#151d27;--card2:#101720;--line:#2a394b;--fg:#e6edf3;--dim:#91a2b5;--ok:#3fb950;--warn:#d7a22a;--bad:#f85149;--run:#4b91f1;--focus:#79c0ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 Inter,"Microsoft YaHei","PingFang SC",Arial,sans-serif}
button,input,textarea,select{font:inherit}header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 18px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#0f1419f2;z-index:5}
h1{font-size:18px;margin:0;font-weight:700;letter-spacing:.01em}.pill{border:1px solid var(--line);border-radius:8px;padding:5px 10px;color:var(--dim)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px}a.pill{text-decoration:none;color:var(--focus)}main{display:grid;grid-template-columns:1.15fr 1fr;gap:14px;padding:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;min-width:0}.wide{grid-column:1/-1}
.card h2{font-size:14px;letter-spacing:.02em;color:var(--fg);margin:0 0 10px}.heading{display:flex;align-items:center;justify-content:space-between;gap:12px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid #222e3b;vertical-align:top}th{color:var(--dim);font-weight:500;font-size:11px}td.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:5px;background:#1d2733;border-radius:3px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--run)}
.tag{font-size:11px;padding:1px 6px;border-radius:4px;border:1px solid;white-space:nowrap}.t-PROPOSED{color:var(--dim);border-color:var(--line)}.t-APPROVED,.t-RUNNING{color:var(--run);border-color:#1f3b5c}.t-NEEDS_REVIEW,.t-SUCCEEDED{color:var(--warn);border-color:#4a3a12}.t-PROMOTED,.t-VALIDATED{color:var(--ok);border-color:#1f4429}.t-HUMAN_REQUIRED,.t-FAILED{color:var(--bad);border-color:#5c1f1f}.t-REJECTED{color:#aaa;border-color:#555}
.scroll{max-height:340px;overflow:auto}small,.muted{color:var(--dim)}#offline{display:none!important;color:var(--warn);font-size:0}#offline.confirmed{display:inline!important}#offline:after{content:'连续 15 秒未收到状态，正在自动重试';font-size:12px}
.review-table tr[data-key]{cursor:pointer}.review-table tr[data-key]:hover{background:#1c2733}.review-table .objective{max-width:620px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.count{background:#5c1f1f;color:#ffb4ae;border-radius:999px;padding:1px 8px}.empty{padding:24px;text-align:center;color:var(--dim);border:1px dashed var(--line);border-radius:6px}
.review-list{max-height:360px;overflow-y:auto;overflow-x:hidden;display:grid;gap:8px;padding-right:4px}.review-item{min-width:0;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:start;padding:10px 12px;background:#111820;border:1px solid var(--line);border-radius:7px}.review-item-main{min-width:0}.review-top,.review-facts{display:flex;align-items:center;flex-wrap:wrap;gap:6px 12px}.review-sample{overflow-wrap:anywhere}.review-facts{margin-top:5px;color:var(--dim)}.review-task{margin-top:6px;overflow-wrap:break-word;line-height:1.45}.review-item .btn{white-space:nowrap}
.btn{border:1px solid var(--line);background:#202b37;color:var(--fg);border-radius:5px;padding:5px 9px;cursor:pointer}.btn:hover{border-color:var(--focus)}.btn.primary{background:#173d5c;border-color:#2f6f9f}.btn.danger{background:#4a1f22;border-color:#7b3035}.btn.warn{background:#463715;border-color:#745d21}.btn:disabled{opacity:.45;cursor:not-allowed}
dialog{width:min(900px,94vw);max-height:90vh;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:0;box-shadow:0 24px 80px #000a}dialog::backdrop{background:#000a}.modal-head{position:sticky;top:0;background:var(--card);display:flex;justify-content:space-between;gap:12px;padding:14px 18px;border-bottom:1px solid var(--line);z-index:2}.modal-body{padding:16px 18px;overflow:auto}.grid{display:grid;grid-template-columns:140px 1fr;gap:7px 14px}.label{color:var(--dim)}.textblock{white-space:pre-wrap;overflow-wrap:anywhere;background:#111820;border:1px solid var(--line);border-radius:6px;padding:9px}.metrics{margin:5px 0 0;padding-left:20px}.metrics li{margin:4px 0}.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}.field{display:flex;flex-direction:column;gap:5px}.field.full{grid-column:1/-1}.field input,.field textarea,.field select{width:100%;background:#10171e;color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:8px}.field textarea{min-height:88px;resize:vertical}.confirm{display:flex;gap:8px;align-items:flex-start;color:var(--warn)}.modal-actions{display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap;padding-top:12px}.notice{margin-top:10px;padding:8px;border-radius:5px;display:none}.notice.ok{display:block;background:#153a23;color:#8ee6a1}.notice.err{display:block;background:#4a1f22;color:#ffb4ae}.history{margin-top:14px}.history-item{border-left:2px solid var(--line);padding:4px 8px;margin:6px 0}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.kpi{background:#111820;border:1px solid var(--line);border-radius:7px;padding:10px}.kpi b{display:block;font-size:21px;color:#fff}.kpi span{color:var(--dim)}details.card>summary{cursor:pointer;color:var(--dim);font-weight:600;letter-spacing:.05em}details.card[open]>summary{margin-bottom:10px}
.topbar{justify-content:space-between}.brand,.nav,.live-chip,.summary-head,.status-line,.node-title{display:flex;align-items:center}.brand,.nav{gap:9px}.nav{margin-left:auto}.nav .pill{color:var(--focus);background:#121b26}.icon{width:17px;height:17px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}.live-chip{gap:7px;border:1px solid var(--line);border-radius:9px;padding:6px 10px;background:var(--card2);font-weight:600}.overview{display:grid;grid-template-columns:1.25fr 1fr .82fr;gap:14px}.hero{padding:17px 18px}.hero-state{font-size:27px;font-weight:750;margin:7px 0 4px}.hero-reason{color:var(--dim);max-width:70ch}.status-line{gap:9px;color:var(--dim)}.status-line strong{color:var(--fg)}.status-badge{display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:4px 9px;border:1px solid var(--line);font-size:12px}.status-badge.ok{color:#74d987;background:#10271a;border-color:#285c36}.status-badge.warn{color:#f0c75e;background:#2b2410;border-color:#66531e}.status-badge.bad{color:#ff8c86;background:#301619;border-color:#713035}.summary-head{justify-content:space-between;color:var(--dim)}.metric-main{font-size:31px;font-weight:760;color:#fff;line-height:1.15;margin:8px 0 5px;font-variant-numeric:tabular-nums}.bucket-row{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:12px}.bucket{background:var(--card2);border:1px solid var(--line);border-radius:7px;padding:8px;text-align:center}.bucket b{display:block;font-size:18px;font-variant-numeric:tabular-nums}.bucket small{font-size:11px}.progress-track{height:8px;background:#263140;border-radius:999px;overflow:hidden;margin-top:12px}.progress-track i{display:block;height:100%;background:linear-gradient(90deg,var(--run),var(--ok))}.wallet-link{display:inline-flex;align-items:center;gap:6px;color:var(--focus);text-decoration:none;margin-top:9px}.workflow-card{padding:16px 18px}.workflow-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:15px}.workflow-head p{margin:2px 0 0;color:var(--dim)}.flow{display:grid;grid-template-columns:minmax(170px,.8fr) 28px minmax(190px,1fr) 28px minmax(190px,1fr) 28px minmax(210px,1.05fr);align-items:stretch}.flow-arrow{display:grid;place-items:center;color:#53677d;font-size:21px}.flow-stage{display:flex;flex-direction:column;gap:8px}.flow-label{font-size:11px;color:var(--dim);letter-spacing:.06em}.agent-node{position:relative;background:var(--card2);border:1px solid var(--line);border-radius:9px;padding:10px 11px;min-height:62px}.agent-node[data-state="active"]{border-color:var(--run);box-shadow:0 0 0 1px #2d64a0 inset}.agent-node[data-state="ok"]{border-color:#2b5d39}.agent-node[data-state="warn"]{border-color:#735b22;background:#211d12}.agent-node[data-state="bad"]{border-color:#783039;background:#271519}.node-title{justify-content:space-between;gap:8px;font-weight:700}.node-title>span:first-child{white-space:nowrap}.node-state{font-size:11px;color:var(--dim);white-space:nowrap}.node-meta{margin-top:6px;color:var(--dim);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.parallel{display:grid;grid-template-columns:1fr;gap:8px}.diagnostics>summary{list-style:none}.diagnostics>summary::-webkit-details-marker{display:none}.diagnostics>summary:after{content:'展开';float:right;color:var(--focus);font-weight:500}.diagnostics[open]>summary:after{content:'收起'}.service-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.service{background:var(--card2);border:1px solid var(--line);border-radius:7px;padding:8px}.service b{display:block}.service small{color:var(--dim)}
.control-row{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-top:12px}.control-row .btn{display:inline-flex;align-items:center;gap:7px}.control-note{color:var(--dim);font-size:12px}.control-note.ok{color:var(--ok)}.control-note.err{color:var(--bad)}
.attention-card{display:none;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:16px;border-color:#745d21;background:linear-gradient(110deg,#171b20,#211d12)}.attention-card.visible{display:grid}.attention-card h2{margin:0 0 5px;font-size:17px}.attention-card p{margin:0;color:var(--dim);line-height:1.55}.attention-actions{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.learning-chip{display:inline-flex;margin-top:8px;padding:3px 8px;border:1px solid var(--line);border-radius:999px;color:var(--focus);font-size:12px}
.bucket{display:block;color:inherit;text-decoration:none;cursor:pointer}.bucket:hover{border-color:var(--focus);background:#172332}.runtime-strip{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin-top:12px;padding:9px 11px;background:var(--card2);border:1px solid var(--line);border-radius:7px}.runtime-strip strong{color:var(--focus)}.runtime-strip small{grid-column:2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@keyframes nodeFlow{0%{background-position:130% 0}100%{background-position:-130% 0}}@keyframes wireFlow{0%{background-position:130% 0}100%{background-position:-30% 0}}.agent-node{overflow:hidden;isolation:isolate;background:#10171f;border-color:#344557;box-shadow:0 7px 18px #0004,inset 0 1px #ffffff08;transition:border-color .35s,box-shadow .35s}.agent-node:before{content:'';position:absolute;z-index:-1;inset:0;pointer-events:none;background:none;opacity:0}.agent-node>*{position:relative;z-index:1}.agent-node[data-state="active"]{background:#10171f;border-color:#55e889;box-shadow:0 0 0 1px #39d97933 inset,0 0 22px #26c86636,0 8px 22px #0007}.agent-node[data-state="active"]:before{opacity:1;background:linear-gradient(90deg,transparent 0%,transparent 34%,#1a6d3a66 45%,#58f493aa 50%,#1a6d3a66 55%,transparent 66%,transparent 100%);background-size:230% 100%;animation:nodeFlow 2.8s linear infinite}.agent-node[data-state="ok"]{background:#10171f;border-color:#3a4958;box-shadow:0 7px 18px #0004}.agent-node[data-state="warn"]{background:#10171f;border-color:#94742d}.agent-node[data-state="bad"]{background:#10171f;border-color:#a8414c}.flow-arrow{position:relative;min-width:28px;font-size:0;color:transparent;overflow:visible}.flow-arrow:before{content:'';display:block;width:calc(100% + 12px);height:2px;border-radius:999px;background:#405166;box-shadow:0 0 8px #40516688}.flow-arrow:after{content:'';position:absolute;right:-7px;width:8px;height:8px;border-top:2px solid #52677e;border-right:2px solid #52677e;transform:rotate(45deg)}#agent-flow[data-state="active"] .flow-arrow:before{background:linear-gradient(90deg,#1c5b35 0%,#35ef7b 28%,#86ffb3 44%,#1c5b35 62%,#1c5b35 100%);background-size:240% 100%;box-shadow:0 0 9px #35ef7baa;animation:wireFlow 2.2s linear infinite}#agent-flow[data-state="active"] .flow-arrow:after{border-color:#62f596;filter:drop-shadow(0 0 4px #48ef87)}#agent-flow[data-state="warn"] .flow-arrow:before{background:#6e5927;box-shadow:0 0 6px #d7a22a44}#agent-flow[data-state="warn"] .flow-arrow:after{border-color:#d7a22a}
@keyframes wireDash{to{stroke-dashoffset:-28}}.flow-shell{position:relative;overflow-x:auto;overflow-y:hidden;padding:10px 8px 16px;scrollbar-color:#41566d #101720}.flow-shell .flow{min-width:1420px;grid-template-columns:minmax(185px,.8fr) 54px minmax(365px,1.35fr) 54px minmax(390px,1.45fr) 54px minmax(220px,.9fr);gap:0}.flow-shell .parallel{grid-template-columns:repeat(2,minmax(165px,1fr));gap:18px}.flow-shell .flow-stage{gap:18px}.flow-shell .flow-label{position:relative;z-index:4}.flow-shell .agent-node{position:relative;z-index:3;min-height:78px;padding:13px 14px;background-color:#0d131a}.flow-shell .flow-arrow{visibility:hidden}.flow-wires{position:absolute;z-index:1;pointer-events:none;overflow:visible}.flow-wire{fill:none;stroke:#4b6075;stroke-width:1.5;opacity:.66;marker-end:url(#arrow-idle);filter:drop-shadow(0 0 3px #344a60)}.flow-wire.ok{stroke:#39cf70;opacity:.76;marker-end:url(#arrow-ok);filter:drop-shadow(0 0 4px #2cc86588)}.flow-wire.active{stroke:#67ff9e;stroke-width:2.1;stroke-dasharray:8 6;opacity:1;marker-end:url(#arrow-ok);filter:drop-shadow(0 0 6px #51f08dcc);animation:wireDash 1.25s linear infinite}.flow-wire.warn{stroke:#d7a22a;marker-end:url(#arrow-warn);filter:drop-shadow(0 0 4px #d7a22a88)}.flow-wire.bad{stroke:#f15b65;marker-end:url(#arrow-bad);filter:drop-shadow(0 0 4px #ef4d5988)}
.maint-nav{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}.maint-panel{position:fixed;right:18px;bottom:18px;z-index:30;width:min(470px,calc(100vw - 24px));height:min(720px,calc(100vh - 36px));display:none;grid-template-rows:auto auto 1fr auto;background:#101720;border:1px solid var(--line);border-radius:13px;box-shadow:0 24px 90px #000c;overflow:hidden}.maint-panel.open{display:grid}.maint-head{display:flex;justify-content:space-between;align-items:center;padding:13px 15px;border-bottom:1px solid var(--line)}.maint-head-title{display:flex;align-items:center;gap:9px;font-weight:750}.maint-boundary{padding:9px 14px;background:#121d29;border-bottom:1px solid var(--line);color:var(--dim);font-size:12px}.maint-boundary b{color:var(--ok)}.maint-boundary .forbidden{color:#f0b06a}.maint-log{padding:14px;overflow:auto;display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth}.chat-bubble{max-width:90%;padding:9px 11px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere}.chat-bubble.user{align-self:flex-end;background:#174b7b;color:#fff}.chat-bubble.assistant{align-self:flex-start;background:#192431;border:1px solid var(--line)}.chat-meta{display:block;margin-top:5px;color:var(--dim);font-size:11px}.maint-compose{padding:11px;border-top:1px solid var(--line);background:#0e151d}.quick-prompts{display:flex;gap:6px;overflow-x:auto;margin-bottom:8px}.quick-prompts button{white-space:nowrap;border:1px solid var(--line);background:#172230;color:var(--dim);border-radius:999px;padding:4px 8px;cursor:pointer}.compose-row{display:grid;grid-template-columns:1fr auto;gap:8px}.compose-row textarea{min-height:64px;max-height:150px;resize:vertical;background:#0b1118;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:9px}.compose-row button{align-self:stretch;padding:0 14px}.maint-note{margin-top:6px;color:var(--dim);font-size:11px}
/* Four stages run left-to-right; parallel roles branch vertically inside a stage. */
.flow-shell{overflow:hidden;padding:8px 2px 12px}.flow-shell .flow{display:grid;width:100%;min-width:0;grid-template-columns:minmax(105px,.58fr) minmax(145px,.78fr) minmax(430px,2.5fr) minmax(155px,.86fr);align-items:start;gap:clamp(16px,2vw,34px)}.flow-shell .flow-stage{display:grid;grid-template-rows:16px auto;width:100%;min-width:0;gap:7px}.flow-shell .parallel{display:grid;align-items:stretch;gap:8px;min-width:0}.flow-shell .flow-stage[data-stage="domain"] .parallel{grid-template-columns:1fr}.flow-shell .flow-stage[data-stage="optimization"] .parallel{grid-template-columns:repeat(3,minmax(0,1fr));grid-template-areas:'qc rules skill' 'maintenance audit .'}.flow-shell .flow-stage[data-stage="optimization"] [data-role="qc_code_optimizer"]{grid-area:qc}.flow-shell .flow-stage[data-stage="optimization"] [data-role="rule_curator"]{grid-area:rules}.flow-shell .flow-stage[data-stage="optimization"] [data-role="skill_maintainer"]{grid-area:skill}.flow-shell .flow-stage[data-stage="optimization"] [data-role="workflow_maintainer"]{grid-area:maintenance}.flow-shell .flow-stage[data-stage="optimization"] [data-role="efficiency_auditor"]{grid-area:audit}.flow-shell .flow-stage[data-stage="decision"] .parallel{grid-template-columns:1fr}.flow-shell .flow-arrow{display:none}.flow-shell .agent-node{min-width:0;min-height:58px;padding:8px 8px}.node-head{display:grid;grid-template-columns:25px minmax(0,1fr);align-items:center;gap:7px}.node-icon{display:grid;place-items:center;width:25px;height:25px;border:1px solid #3b5369;border-radius:7px;background:#0b1219;color:#aebdcb}.node-icon svg{width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}.agent-node[data-state="active"] .node-icon{color:#8affb1;border-color:#55d980}.agent-node[data-state="ok"] .node-icon{color:#aebdcb;border-color:#3b5369}.agent-node[data-state="warn"] .node-icon{color:#f2ca63;border-color:#8f712b}.agent-node[data-state="bad"] .node-icon{color:#ff8a91;border-color:#a44049}.node-name{min-width:0;font-size:clamp(10px,.78vw,12px);font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.node-state{position:absolute;right:6px;bottom:4px;max-width:44px;overflow:hidden;text-overflow:ellipsis;font-size:9px;color:var(--dim);white-space:nowrap}.node-meta{display:block;margin:4px 46px 0 32px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:9px;color:#7f91a3}.flow-label{margin:0;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
@media(max-width:900px){main{grid-template-columns:1fr}.wide{grid-column:auto}.form-grid{grid-template-columns:1fr}.field.full{grid-column:auto}.grid{grid-template-columns:1fr}.label{margin-top:5px}}
@media(max-width:1100px){.overview{grid-template-columns:1fr 1fr}.hero{grid-column:1/-1}.flow{grid-template-columns:1fr}.flow-arrow{height:24px;transform:rotate(90deg)}}
@media(max-width:620px){.review-item{grid-template-columns:1fr}.review-item .btn{justify-self:start}}
@media(max-width:680px){header{padding:10px 12px}.nav a:nth-of-type(n+3){display:none}.overview{grid-template-columns:1fr}.bucket-row,.parallel,.service-grid{grid-template-columns:1fr 1fr}}
.node-proof{display:flex;gap:4px;flex-wrap:wrap;margin:6px 0 0 32px}.node-proof span{border:1px solid #2d3a49;border-radius:999px;padding:0 4px;color:#9eb1c5;font-size:9px;line-height:14px;white-space:nowrap}.node-proof .up{color:#76dd91;border-color:#285b38}.node-proof .down{color:#ff8e88;border-color:#70323a}
</style></head><body>
<header class="topbar"><div class="brand"><svg class="icon" viewBox="0 0 24 24"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg><h1>CMR/LGE 闭环控制台</h1></div><nav class="nav"><a class="pill" href="/editor">标签编辑器</a><a class="pill" href="/check-inventory">数据现状</a><a class="pill" href="/check-validation">抽样验收</a><a class="pill" href="/score">评估</a><button class="btn primary maint-nav" onclick="toggleMaint(true)"><svg class="icon" viewBox="0 0 24 24"><path d="M4 5h16v11H8l-4 3z"/></svg><span>维护助手</span></button><button class="btn" onclick="tick(true)">刷新</button><span class="live-chip"><span class="dot" id="dot" style="background:#555"></span><span id="ctl">连接中</span></span><span id="offline">后端无响应</span></nav></header>
<main>
 <section class="wide overview"><article class="card hero"><div class="status-line"><span id="system-badge" class="status-badge">读取中</span><span>当前运行</span><strong id="current-run">-</strong><span id="ts">-</span></div><div class="hero-state" id="hero-state">正在读取工作流状态</div><div class="hero-reason" id="hero-reason">控制器、执行器和最近事件会在这里汇总。</div><div class="control-row"><button class="btn primary" id="workflow-start" onclick="controlWorkflow('start')" disabled><svg class="icon" viewBox="0 0 24 24"><path d="m8 5 11 7-11 7z"/></svg><span>启动闭环</span></button><button class="btn danger" id="workflow-pause" onclick="controlWorkflow('pause')" disabled><svg class="icon" viewBox="0 0 24 24"><path d="M7 5v14M17 5v14"/></svg><span>暂停</span></button><span class="control-note" id="control-note">正在确认是否可控制</span></div><div class="service-grid" id="services" style="margin-top:14px"></div></article><article class="card"><div class="summary-head"><span>数据处理进度</span><a href="/check-inventory" class="wallet-link">查看明细</a></div><div class="metric-main"><span id="issue-total">0</span> <small style="font-size:13px;color:var(--dim)">尚未解决</small></div><div id="issue-detail" class="muted">待筛 · 待修改</div><div id="manual-review-progress" class="runtime-strip"><strong>人工复核</strong><span>读取中</span><small>确认结果会自动进入知识链</small></div><div class="progress-track"><i id="resolution-bar" style="width:0%"></i></div><div class="bucket-row" id="bucket-summary"></div></article><article class="card"><div class="summary-head"><span>阿里云可用余额</span><svg class="icon" viewBox="0 0 24 24"><path d="M3 6h18v13H3zM3 9h18M16 14h2"/></svg></div><div class="metric-main" id="aliyun-balance">未接入</div><div class="muted" id="aliyun-detail">服务器尚无只读账单凭据</div><a class="wallet-link" target="_blank" rel="noopener" href="https://billing-cost.console.aliyun.com/coupons/coupon">打开阿里云费用中心 →</a></article></section>
 <section class="card wide" id="lge-execution-panel"><h2>LGE 自动处理</h2><p class="muted" id="lge-execution-note">读取实际应用记录…</p><div id="lge-execution-counts" style="display:flex;gap:24px;flex-wrap:wrap"></div></section>
 <section class="card wide attention-card" id="human-attention"><div><h2 id="attention-title">需要少量人工示例</h2><p id="attention-message"></p><span class="learning-chip" id="learning-chip">已确认示例 0</span></div><div class="attention-actions"><a class="btn primary" id="attention-action" href="/editor">复核代表样本</a><a class="btn" id="validation-action" href="/check-validation">抽样验收</a></div></section>
 <section class="card wide workflow-card"><div class="workflow-head"><div><h2>多智能体工作流</h2><p>节点与端到端连线直接映射运行状态；并行角色采用分散布局，一对多关系分别连线。</p></div><span id="flow-updated" class="muted"></span></div><div class="flow-shell" id="flow-shell"><svg class="flow-wires" id="flow-wires" aria-hidden="true"></svg><div class="flow" id="agent-flow"></div></div></section>
 <section class="card wide"><div class="heading"><h2>成套数据与分割模型</h2><span id="dataset-release-time" class="muted"></span></div><div id="segmentation-lifecycle" class="kpis"></div><div class="control-row"><button class="btn dataset-action" onclick="controlDataset('audit')">↻ 核验配对</button><button class="btn dataset-action" onclick="controlDataset('validate')">✓ 验收下一批</button><button class="btn dataset-action" onclick="controlDataset('freeze')">▣ 导出通过集</button><span id="dataset-job-note" class="control-note" role="status" aria-live="polite"></span></div><p class="muted">按现有评分体系，每次最多验收 100 组；保留待复核和剔除项。仅生成验收记录与独立数据副本。</p><p id="dataset-release-note" class="muted">等待验收汇总</p></section>
 <details class="card wide diagnostics"><summary>人工复核与候选（<span id="hr-count">0</span>）</summary><div class="heading"><h2>标签人工复核队列</h2></div><div id="label-review"></div><div class="heading" style="margin-top:16px"><h2>候选 gate</h2><span id="gate-summary" class="muted">读取中…</span></div><div id="gate-reasons"></div><details style="margin-top:8px"><summary>被拒绝候选</summary><div id="batch-rejected" style="margin-top:10px"></div></details></details>
 <details class="card wide diagnostics"><summary>数据池与迭代结果</summary><div class="heading"><h2>四桶现状</h2><span class="count" id="boundary-total">0</span></div><div id="boundary-progress" class="kpis"></div><h2 style="margin-top:16px">check 自迭代实测</h2><div id="check-loop" class="muted">读取中…</div><div id="workflow-outcome" class="muted">读取中…</div><h2 style="margin-top:16px">问题池</h2><div class="scroll"><table id="pools"></table></div></details>
 <details class="card wide diagnostics"><summary>模型调用与评估</summary><div class="kpis"><div class="kpi"><span>真实模型调用（24h）</span><b id="call-count">0</b><small id="call-detail">等待遥测</small></div><div class="kpi"><span>固定评估集</span><b id="eval-result">-</b><small id="eval-detail">rules_v2</small></div><div class="kpi"><span>候选 gate 通过</span><b id="staged-total">0</b><small id="staged-detail">尚无候选</small></div><div class="kpi"><span>标签人工复核</span><b id="human-kpi">0</b><small>可直接打开的候选</small></div></div><div class="scroll" style="margin-top:12px"><table id="models"></table></div></details>
 <details class="card wide"><summary>维护代码实验的人工裁定（诊断，不是标签队列） <span id="maintenance-count"></span></summary><div id="human" style="margin-top:10px"></div></details>
 <details class="card"><summary>Agent 活动（诊断）</summary><div class="scroll"><table id="agents"></table></div></details>
 <details class="card"><summary>维护动作（诊断）</summary><div class="scroll"><table id="actions"></table></div></details>
 <details class="card"><summary>运行记录（诊断）</summary><div class="scroll"><table id="runs"></table></div></details>
 <details class="card"><summary>事件流（诊断）</summary><div class="scroll"><table id="events"></table></div></details>
</main>
<dialog id="review-dialog"><div class="modal-head"><div><b id="modal-title">人工处理</b><div class="muted" id="modal-subtitle"></div></div><button class="btn" onclick="closeReview()">关闭</button></div><div class="modal-body">
 <div class="grid"><div class="label">状态 / 负责人</div><div id="modal-meta"></div><div class="label">为什么需要你人工介入</div><div class="textblock" id="modal-why">简报生成中…</div><div class="label">过不了的核心原因</div><div><ul class="metrics" id="modal-blockers"></ul></div><div class="label">关联标签任务</div><div class="textblock" id="modal-pool"></div><div class="label">隔离实验产物</div><div class="textblock" id="modal-artifact"></div></div>
<details style="margin-top:8px"><summary class="muted" style="cursor:pointer">动作目标与验收指标原文（存档，点击展开）</summary><div class="textblock" id="modal-objective" style="margin-top:6px"></div><ol class="metrics" id="modal-metrics"></ol></details>
 <div class="history"><b>人工处置历史</b><div id="modal-history"></div></div>
 <div class="form-grid"><label class="field"><span>审阅人</span><input id="reviewer" value="local_operator" maxlength="80"></label><label class="field"><span>处置</span><select id="decision" onchange="syncConfirm()"><option value="defer">保留待审（仅记录意见）</option><option value="retry_once">追加一次隔离重试</option><option value="resolve_external">确认已在系统外解决</option><option value="reject">拒绝该维护动作</option></select></label><label class="field full"><span>判断依据 / 后续要求（必填）</span><textarea id="note" maxlength="4000" placeholder="写明证据、判断和下一步；至少 3 个字符"></textarea></label><label class="field full confirm"><input type="checkbox" id="confirmed"><span id="confirm-text">保留待审只记录意见，不改变动作状态。</span></label></div>
 <div id="review-notice" class="notice"></div><div class="modal-actions"><button class="btn primary" id="prompt-editor" onclick="openEditor('prompt')">进入标签编辑器（Agent 候选 → 手工兜底）</button><button class="btn" onclick="closeReview()">取消</button><button class="btn primary" id="submit-review" onclick="submitCurrentReview()">提交处置</button></div>
</div></dialog>
<aside class="maint-panel" id="maint-panel" aria-label="项目维护助手"><div class="maint-head"><div class="maint-head-title"><svg class="icon" viewBox="0 0 24 24"><path d="M12 3a7 7 0 0 0-7 7v4l-2 3h18l-2-3v-4a7 7 0 0 0-7-7zM9 21h6"/></svg><span>项目维护助手</span><span class="tag t-RUNNING" id="maint-model">Qwen</span></div><div><button class="btn" onclick="clearMaint()">清空</button><button class="btn" onclick="toggleMaint(false)">关闭</button></div></div><div class="maint-boundary"><b>负责：</b>服务、调度、Agent 交接、配置、接口、测试、看板与文档。 <span class="forbidden">不负责：医学判定、标签修改、删除或分桶。</span><br>当前 Qwen 可诊断并生成最小修复方案；任何代码修改必须进入隔离快照、测试和独立复核。</div><div class="maint-log" id="maint-log"></div><div class="maint-compose"><div class="quick-prompts"><button onclick="setMaintPrompt('为什么自动闭环暂停？请定位控制面原因。')">分析暂停</button><button onclick="setMaintPrompt('检查三个领域 Agent 是否都获得了并行调度。')">检查并行分支</button><button onclick="setMaintPrompt('分析最近的失败事件，并给出最小控制面修复方案。')">最近故障</button><button onclick="setMaintPrompt('概括当前配置、服务状态和下一项维护风险。')">配置与风险</button></div><div class="compose-row"><textarea id="maint-input" maxlength="5000" placeholder="描述项目维护或 Bug；Enter 发送，Shift+Enter 换行"></textarea><button class="btn primary" id="maint-send" onclick="sendMaint()">发送</button></div><div class="maint-note" id="maint-worked">对话仅接收去标识化控制面状态，不向模型发送影像、标签或密钥。</div></div></aside>
<script>
let snapshot=null,current=null;
function renderLgeExecution(d){const e=d.lge_execution||{},stages=e.stages||{},release=e.generator||{};document.getElementById('lge-execution-note').textContent=release.ready?'只统计独立验证后的实际验收记录；待复核项单独保留。':'应用链已接入；等待领域智能体提交并通过评审的候选生成工具。';document.getElementById('lge-execution-counts').innerHTML=[['lge_myo','心肌'],['lge_mi','梗死']].map(([key,title])=>{const s=stages[key];return s?`<div><strong>${title}</strong><p>待处理 ${Number(s[key+'_pending']||0)} · 待复核 ${Number(s[key+'_work']||0)} · 已通过 ${Number(s[key+'_resolved']||0)}</p></div>`:`<div><strong>${title}</strong><p class="muted">尚无实际应用记录</p></div>`}).join('');}
let tickBusy=false,failedTicks=0,lastTickAt=0,lastGoodAt=Date.now();
function zhMetric(text){
  const map = [["target_issue_recall","目标问题召回"],["false_repair_rate","误修率"],
    ["new_critical_qc_issues","新增严重问题"],["per case","按病例"],["per slice","按切片"],
    ["flag rate","触发率"],["side-effect","副作用"],["precision","精确率"],["recall","召回率"],
    ["distribution","分布"],["threshold","阈值"],["component","连通域"],["Euler","欧拉数"],
    ["hole","血池洞"],["border","边界"],["centroid","质心"],["continuity","连续性"],
    ["proportion","比例"],["magnitude","幅度"],["rate","比率"],["vs","对比"]];
  let out = String(text);
  for (const pair of map) out = out.split(pair[0]).join(pair[1]);
  return out;
}
const esc=v=>String(v??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
const t=(s,c)=>`<span class="tag t-${esc(c)}">${esc(s)}</span>`;
const taskLabel=k=>({cmr_boundary_manifest:'CMR 边界筛选',cmr_middle_manifest:'CMR 标签修复',lge_boundary_manifest:'LGE 边界筛选',lge_issue_manifest:'LGE 问题汇总',lge_middle_manifest:'LGE 标签修复',trajectory_manifest:'跨层位置校准'}[k]||'标签复核任务');
async function loadBrief(a){
  const why=document.getElementById('modal-why'),bl=document.getElementById('modal-blockers');
  why.textContent='简报生成中…';bl.innerHTML='';
  try{
    const r=await fetch(`/api/brief?run_id=${encodeURIComponent(a.run_id)}&action_id=${encodeURIComponent(a.action_id)}`,{cache:'no-store'});
    const d=await r.json();
    if(d.error){why.textContent='（无简报：'+d.error+'）';return}
    why.textContent=d.why||'（无简报）';
    bl.innerHTML=(d.blockers||[]).map(x=>`<li>${esc(x)}</li>`).join('')||'<li class="muted">无</li>';
  }catch(e){why.textContent='简报生成失败：'+e.message}
}
function gotoEditor(i){const a=snapshot.human_required[i];if(!a)return;if(!a.pool){alert('该动作尚未关联具体标签任务；请打开弹窗走处置流程');return}location.href=`/editor?pool=${encodeURIComponent(a.pool)}&mode=prompt&run_id=${encodeURIComponent(a.run_id)}&action_id=${encodeURIComponent(a.action_id)}`}
function artifactHtml(a){const s=a.artifact_summary;if(!s)return '<span class="muted">暂无产物：该动作只有提案，还没在隔离沙箱里产出代码/测试结果。产物=沙箱里生成的实验代码、测试与验证清单，未应用到生产。</span>';let h=s.ready?'<span class="tag t-PROMOTED">沙箱自测通过</span>':'<span class="tag t-FAILED">沙箱自测未过</span>';(s.changed_files||[]).forEach(f=>{h+=`<div>📄 ${esc(f)}</div>`});(s.tests||[]).forEach(x=>{h+=`<div class="muted">✓ ${esc(x)}</div>`});if(s.summary)h+=`<div class="muted" id="artifact-summary" style="margin-top:4px">${esc(s.summary)}</div>`;return h}
async function translateModal(a){const texts=[];const refs=[];if(a.objective){texts.push(a.objective);refs.push(['o',0])}(a.acceptance_metrics||[]).forEach((m,i)=>{texts.push(m);refs.push(['m',i])});const sm=a.artifact_summary;if(sm&&sm.summary){texts.push(sm.summary);refs.push(['s',0])}if(!texts.length)return;try{const r=await fetch('/api/translate',{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'cmr-lge-board'},body:JSON.stringify({texts:texts.slice(0,12),brief:true})});const d=await r.json();const tr=d.translations||[];refs.forEach(([kind,i],idx)=>{const v=tr[idx];if(!v)return;if(kind==='o'){const el=document.getElementById('modal-objective');el.textContent=v;el.title=a.objective}if(kind==='m'){const lis=document.getElementById('modal-metrics').querySelectorAll('li');if(lis[i]){lis[i].textContent=v;lis[i].title=a.acceptance_metrics[i]}}if(kind==='s'){const el=document.getElementById('artifact-summary');if(el){el.textContent=v;el.title=sm.summary}}})}catch(e){}}
function openReview(index){current=snapshot.human_required[index];if(!current)return;document.getElementById('modal-title').textContent=current.action_id;document.getElementById('modal-subtitle').textContent=`run ${current.run_id} · ${current.action_class}`;document.getElementById('modal-meta').innerHTML=t(current.status,current.status)+'　'+esc(current.owner_role)+`　priority ${esc(current.priority??'-')}　max attempts ${esc(current.max_attempts??'-')}`;document.getElementById('modal-objective').textContent=current.objective||'-';document.getElementById('modal-metrics').innerHTML=(current.acceptance_metrics||[]).map(x=>`<li>${esc(typeof zhMetric==='function'?zhMetric(x):x)}</li>`).join('')||'<li class="muted">无</li>';document.getElementById('modal-pool').textContent=current.task_label||'尚未关联具体标签任务';document.getElementById('modal-artifact').innerHTML=artifactHtml(current);document.getElementById('modal-history').innerHTML=(current.reviews||[]).map(r=>`<div class="history-item"><b>${esc(r.decision)}</b> · ${esc(r.reviewer)} · ${esc(r.time)}<br><span class="muted">${esc(r.previous_status)} → ${esc(r.resulting_status)}</span><br>${esc(r.note)}</div>`).join('')||'<div class="muted">暂无人工处置</div>';document.getElementById('note').value='';document.getElementById('decision').value='defer';document.getElementById('confirmed').checked=false;document.getElementById('review-notice').className='notice';const routed=Boolean(current.pool);document.getElementById('prompt-editor').disabled=!routed;syncConfirm();document.getElementById('review-dialog').showModal();translateModal(current);loadBrief(current)}
function closeReview(){document.getElementById('review-dialog').close();current=null;}
function openEditor(mode){if(!current||!current.pool)return;location.href=`/editor?pool=${encodeURIComponent(current.pool)}&mode=${encodeURIComponent(mode)}&run_id=${encodeURIComponent(current.run_id)}&action_id=${encodeURIComponent(current.action_id)}`;}
function openBatchCandidate(pool,index){location.href=`/editor?pool=${encodeURIComponent(pool)}&index=${encodeURIComponent(index)}&mode=manual`;}
function openLabelReview(index){const row=snapshot?.label_review_tasks?.[index];if(!row||!Number.isInteger(Number(row.index))||Number(row.index)<0)return;openBatchCandidate(row.pool,Number(row.index))}
function reviewOperation(operations){const labels={remove_small_components:'移除孤立小区域',fill_internal_hole:'填补内部空洞',expand_myo:'增补心肌区域',shrink_myo:'收缩心肌区域'};return (Array.isArray(operations)?operations:[]).map(x=>labels[x]||x).join('、')||'查看候选修改'}
function renderLabelReview(review){const box=document.getElementById('label-review');if(!review.length){box.innerHTML='<div class="empty">当前没有通过 gate、等待人工确认的像素候选</div>';return}box.innerHTML='<div class="review-list">'+review.map((x,i)=>`<article class="review-item"><div class="review-item-main"><div class="review-top"><b class="review-sample">${esc(x.sample_name||'候选样本')}</b><span class="tag">待处理</span></div><div class="review-facts"><span title="${esc((x.operation||[]).join(', '))}">修改：${esc(reviewOperation(x.operation))}</span><span>面积：${esc(x.metrics?.area_before??'-')} → ${esc(x.metrics?.area_after??'-')} 像素</span><span>变化：${esc(x.metrics?.changed_pixels??'-')} 像素</span></div><div class="review-task">${esc(x.task||'请核对候选与原标签。')}</div></div><button class="btn warn" onclick="openLabelReview(${i})">打开复核</button></article>`).join('')+'</div>'}
function syncConfirm(){const d=document.getElementById('decision').value,box=document.getElementById('confirmed');box.disabled=d==='defer';if(d==='defer')box.checked=false;const msg={defer:'保留待审只记录意见，不改变动作状态。',retry_once:'确认后会增加一次 max_attempts，并把动作重新置为 PROPOSED。',resolve_external:'确认外部处理和证据已经核查；动作将置为 VALIDATED。',reject:'确认不再执行该动作；动作将置为 REJECTED。'};document.getElementById('confirm-text').textContent=msg[d];}
async function submitCurrentReview(){if(!current)return;const btn=document.getElementById('submit-review'),notice=document.getElementById('review-notice');btn.disabled=true;notice.className='notice';try{const body={run_id:current.run_id,action_id:current.action_id,decision:document.getElementById('decision').value,reviewer:document.getElementById('reviewer').value,note:document.getElementById('note').value,confirmed:document.getElementById('confirmed').checked};const r=await fetch('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'cmr-lge-board'},body:JSON.stringify(body)});const out=await r.json();if(!r.ok)throw new Error(out.error||`HTTP ${r.status}`);notice.textContent=`已记录：${out.previous_status} → ${out.resulting_status}`;notice.className='notice ok';await tick();setTimeout(closeReview,700);}catch(e){notice.textContent=e.message;notice.className='notice err';}finally{btn.disabled=false;}}
const roleNames={failure_triage:'问题分诊',cmr_specialist:'配对 CMR 心肌',lge_myo_specialist:'LGE 心肌',lge_mi_specialist:'LGE MI',pairing_gate:'配对层筛选',workflow_maintainer:'流程维护',qc_code_optimizer:'QC 优化',rule_curator:'规则整理',skill_maintainer:'知识 / Skill 同步',efficiency_auditor:'效率审计',supervisor:'Supervisor',duplicate_guard:'重复提案守卫',independent_reviewer:'独立 Reviewer','independent_reviewer:implementation':'实现复核',maintenance_executor:'隔离执行器'};
function latestFor(d,role){return (d.agents||[]).find(a=>a.who===role||a.who.startsWith(role+':'))}
function runningFor(d,role){return (d.agent_invocations||[]).find(x=>x.status==='RUNNING'&&(x.role===role||String(x.role).startsWith(role+':')))}
function elapsedSeconds(started){const value=Date.parse(started);return Number.isFinite(value)?Math.max(0,Math.floor((Date.now()-value)/1000)):0}
const roleIconPaths={failure_triage:'M4 5h16v11H9l-5 4zm4 4h8M8 12h5',cmr_specialist:'M12 20s-7-4.4-7-10a4 4 0 0 1 7-2 4 4 0 0 1 7 2c0 5.6-7 10-7 10z',lge_myo_specialist:'M12 20s-7-4.4-7-10a4 4 0 0 1 7-2 4 4 0 0 1 7 2c0 5.6-7 10-7 10zM9 12h6',lge_mi_specialist:'M12 20s-7-4.4-7-10a4 4 0 0 1 7-2 4 4 0 0 1 7 2c0 5.6-7 10-7 10zM12 8v5m0 3v.1',qc_code_optimizer:'M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1m0-12.8-2.1 2.1M7.7 16.3l-2.1 2.1M9 12a3 3 0 1 0 6 0 3 3 0 0 0-6 0z',rule_curator:'M5 4h14v16H5zM8 8h8M8 12h8M8 16h5',skill_maintainer:'M4 6h6l2 2h8v11H4zM12 11v5m-2.5-2.5h5',efficiency_auditor:'M4 19h16M7 16v-4m5 4V7m5 9V4',supervisor:'M12 3l8 4v5c0 4.7-3.3 7.7-8 9-4.7-1.3-8-4.3-8-9V7zM9 12l2 2 4-5',independent_reviewer:'M3 12s3.3-6 9-6 9 6 9 6-3.3 6-9 6-9-6-9-6zm9 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',executor:'M8 5l11 7-11 7z'};
function roleIcon(role){return `<span class="node-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="${roleIconPaths[role]||roleIconPaths.executor}"/></svg></span>`}
function shortModel(value){return String(value||'').replace(/^llm:/,'').split('/').pop()||'确定性工具'}
function roleProof(d,role){const c=d.agent_continuity||{},m=c.roles?.[role];if(!m)return '';const score=m.score||{},quality=score.recent_mean==null?'—':`${Math.round(score.recent_mean*100)}%`,rawDelta=score.delta==null?null:Math.round(score.delta*100),trend=rawDelta==null?'':rawDelta>0?` ↑${rawDelta}`:rawDelta<0?` ↓${Math.abs(rawDelta)}`:' →',trendClass=rawDelta>0?'up':rawDelta<0?'down':'',promoted=Number(m.owned_actions?.PROMOTED||0),history=Array.isArray(m.model_history)?m.model_history:[],inheritance=c.inheritance_ready?'记忆已接续':'记忆初始化';const title=`角色记忆 ${c.revision||'-'} · 历史模型 ${history.join(' → ')||'无'} · 交接 收${m.handoffs_received||0}/发${m.handoffs_sent||0}`;return `<div class="node-proof" title="${esc(title)}"><span class="${trendClass}">输出质检 ${quality}${trend}</span><span>成功 ${m.successful_invocations||0}/${m.invocations||0}</span>${promoted?`<span class="up">晋级 ${promoted}</span>`:''}<span>${inheritance}</span></div>`}
function flowNode(d,role,label){const paused=!!d.state.paused,controllerAlive=d.controllers.autopilot!=='-',running=(!paused&&controllerAlive)?runningFor(d,role):null,a=latestFor(d,role);const task=d.autonomy?.consumer_task,toolRole={lge_myo_specialist:'lge_myo',lge_mi_specialist:'lge_mi'},toolRunning=!d.operator_control?.paused&&!d.autonomy?.operator_paused&&d.autonomy?.consumer?.status==='RUNNING'&&task?.status==='RUNNING'&&task.workflow_stage===toolRole[role];const toolReady=!d.operator_control?.paused&&!d.autonomy?.operator_paused&&!!toolRole[role]&&task?.status==='READY'&&task.workflow_stage===toolRole[role];let state='idle',stamp='-',meta=paused?'等待恢复':'待调度',detail='';if(running){state='active';stamp=`${elapsedSeconds(running.started_at)}s`;meta=shortModel(running.requested_model||running.model||running.provider);const action=String(running.thread_id||'当前动作').split(':').slice(1).join(':')||'当前动作';detail=`${action} · ${meta} · 已运行 ${stamp}`}else if(toolRunning){state='active';stamp=`${elapsedSeconds(task.started_at)}s`;meta='已晋级工具执行';detail=`处理已批准候选 · 每批最多 ${task.limit||10} 项`}else if(toolReady){state='ok';stamp=(task.finished_at||'').split('T')[1]?.slice(0,8)||'-';meta=`本批通过 ${task.accepted||0} · 写入 ${task.label_writes||0}`;detail='已晋级工具最近批次执行结果'}else if(a){state=a.status==='FAILED'?'bad':(paused?'warn':'ok');stamp=a.time||'-';meta=`${a.status==='FAILED'?'失败 · ':paused?'暂停 · ':''}${shortModel(a.model)}`;detail=`${stamp} · ${a.model||'确定性工具'} · ${a.detail||''}`}return `<div class="agent-node" data-role="${esc(role)}" data-state="${state}" title="${esc(detail)}"><div class="node-head">${roleIcon(role)}<span class="node-name">${esc(label||roleNames[role]||role)}</span><span class="node-state">${esc(stamp)}</span></div><div class="node-meta">${esc(meta)}</div>${roleProof(d,role)}</div>`}
function dependencyNode(d){const w=d.workflow_dependency||{},active=w.stage==='paired_selection',done=['cmr_paired'].includes(w.stage),state=active?'active':done?'ok':'idle',meta=w.pairing_ready?`${w.paired_slice_count||0} 层已锁定`:'等待 LGE 心肌与 MI 完成';return `<div class="agent-node" data-role="pairing_gate" data-state="${state}" title="${esc(w.contract||'')}\"><div class="node-head">${roleIcon('pairing_gate')}<span class="node-name">配对层筛选</span><span class="node-state">${active?'运行':'闸门'}</span></div><div class="node-meta">${esc(meta)}</div></div>`}
function drawFlowWires(){
 const shell=document.getElementById('flow-shell'),flow=document.getElementById('agent-flow'),svg=document.getElementById('flow-wires');if(!shell||!flow||!svg)return;
 const executor=flow.querySelector('.flow-stage:last-child .agent-node:last-child');if(executor)executor.dataset.role='executor';
 const node=role=>flow.querySelector(`.agent-node[data-role="${role}"]`),links=[
  ['failure_triage','lge_myo_specialist'],
  ['lge_myo_specialist','lge_mi_specialist','down'],['lge_mi_specialist','pairing_gate','down'],['pairing_gate','cmr_specialist','down'],
  ['cmr_specialist','qc_code_optimizer','curve',0],
  ['failure_triage','workflow_maintainer','curve',0],['workflow_maintainer','supervisor','curve',0],['qc_code_optimizer','rule_curator'],['rule_curator','skill_maintainer'],['skill_maintainer','qc_code_optimizer','loop'],
  ['skill_maintainer','efficiency_auditor','down'],['efficiency_auditor','supervisor'],
  ['supervisor','independent_reviewer','down'],['independent_reviewer','executor','down']
 ];
 const fr=flow.getBoundingClientRect(),width=Math.max(flow.scrollWidth,Math.ceil(fr.width)),height=Math.max(flow.scrollHeight,Math.ceil(fr.height));
 svg.style.left=flow.offsetLeft+'px';svg.style.top=flow.offsetTop+'px';svg.style.width=width+'px';svg.style.height=height+'px';svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
 const defs=`<defs><marker id="arrow-idle" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="#52677e"/></marker><marker id="arrow-ok" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="#67ff9e"/></marker><marker id="arrow-warn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="#d7a22a"/></marker><marker id="arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="#f15b65"/></marker></defs>`;
 const paths=[];
 for(const [a,b,kind='curve',slot=0] of links){
  const from=node(a),to=node(b);if(!from||!to)continue;const ar=from.getBoundingClientRect(),br=to.getBoundingClientRect(),states=[from.dataset.state,to.dataset.state];
  let state=states.includes('active')?'active':states.includes('bad')?'bad':states.includes('warn')?'warn':states.every(x=>x==='ok')?'ok':'idle',path='';
  if(kind==='loop'){
   const x1=ar.left+ar.width/2-fr.left,y1=ar.top-fr.top,x2=br.left+br.width/2-fr.left,y2=br.top-fr.top,yLoop=Math.max(2,Math.min(y1,y2)-13);
   path=`M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${x1.toFixed(1)} ${yLoop.toFixed(1)}, ${x2.toFixed(1)} ${yLoop.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`;
  }else if(kind==='down'){
   const x1=ar.left+ar.width/2-fr.left,y1=ar.bottom-fr.top,x2=br.left+br.width/2-fr.left,y2=br.top-fr.top,mid=y1+(y2-y1)*.52;
   path=`M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${x1.toFixed(1)} ${mid.toFixed(1)}, ${x2.toFixed(1)} ${mid.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`;
  }else{
   const x1=ar.right-fr.left,y1=ar.top+ar.height/2-fr.top,x2=br.left-fr.left,y2=br.top+br.height/2-fr.top+slot*10,bend=Math.max(20,(x2-x1)*.42);
   path=`M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${(x1+bend).toFixed(1)} ${y1.toFixed(1)}, ${(x2-bend).toFixed(1)} ${y2.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`;
  }
  paths.push(`<path class="flow-wire ${state}" d="${path}"/>`);
 }
 svg.innerHTML=defs+paths.join('');
}
function renderFlow(d){const paused=!!d.state.paused,alive=d.controllers.autopilot!=='-',checkAlive=d.controllers.check_loop!=='-',event=d.state.last_event||'-',dependency=d.workflow_dependency||{},stageNames={lge_myo:'LGE 心肌',lge_mi:'LGE MI',paired_selection:'配对层筛选',cmr_paired:'配对 CMR 心肌'};const execState=paused?'warn':(!alive?'bad':'active'),execStamp=d.state.last_event_at||'-',execMeta=paused?(checkAlive?'已暂停 · QC 仅监听变化':'已暂停'):`cycle ${d.state.cycles??'-'}`;document.getElementById('agent-flow').innerHTML=`<div class="flow-stage" data-stage="triage"><div class="flow-label">① 分诊</div>${flowNode(d,'failure_triage')}</div><div class="flow-stage" data-stage="domain"><div class="flow-label">② 数据依赖链</div><div class="parallel">${flowNode(d,'lge_myo_specialist')}${flowNode(d,'lge_mi_specialist')}${dependencyNode(d)}${flowNode(d,'cmr_specialist')}</div></div><div class="flow-stage" data-stage="optimization"><div class="flow-label">③ 当前阶段 QC 优化</div><div class="parallel">${flowNode(d,'qc_code_optimizer')}${flowNode(d,'rule_curator')}${flowNode(d,'skill_maintainer')}${flowNode(d,'workflow_maintainer')}${flowNode(d,'efficiency_auditor')}</div></div><div class="flow-stage" data-stage="decision"><div class="flow-label">④ 审批与回流</div><div class="parallel">${flowNode(d,'supervisor')}${flowNode(d,'independent_reviewer')}<div class="agent-node" data-role="executor" data-state="${execState}" title="${esc(event)} · cycles ${esc(d.state.cycles??'-')} · stagnant ${esc(d.state.stagnant??'-')}"><div class="node-head">${roleIcon('executor')}<span class="node-name">Autopilot / Check Loop</span><span class="node-state">${esc(execStamp)}</span></div><div class="node-meta">${esc(execMeta)}</div></div></div></div>`;document.getElementById('flow-updated').textContent=`当前阶段：${stageNames[dependency.stage]||dependency.stage||'-'} · 最近事件 ${event} · ${d.state.last_event_at||''}`;if(d.autonomy){const credit=(d.autonomy.feedback?.role_credit||[]),accepted=credit.reduce((n,r)=>n+(r.accepted||0),0),writes=credit.reduce((n,r)=>n+(r.label_writes||0),0);document.getElementById('flow-updated').textContent+=` · 实际通过 ${accepted} · 标签更新 ${writes} · 执行器 ${d.autonomy.consumer?.status||'-'}`;const task=d.autonomy.consumer_task;if(task){const target={lge_myo:'LGE 心肌',lge_mi:'LGE 梗死'}[task.workflow_stage]||task.workflow_stage;document.getElementById('flow-updated').textContent+=` · ${target}：${task.status==='RUNNING'?'工具执行中':`最近一批通过 ${task.accepted||0}、写入 ${task.label_writes||0}`}`;}}}
function renderReviewBuckets(boundary){const rows=[['pending','待筛',boundary.pending||0],['work','待修改',boundary.work||0],['checked','已通过',boundary.checked||0],['trash','已剔除',boundary.trash||0]];document.getElementById('bucket-summary').innerHTML=rows.map(([state,name,value])=>`<a class="bucket" href="/editor?pool=cmr_boundary_manifest&review_status=${state}&mode=manual" title="打开${esc(name)}队列，逐例快速复核"><b>${esc(value)}</b><small>${esc(name)} · 点击复核</small></a>`).join('')}
function updateControl(d){const paused=!!d.state.paused,operatorPaused=!!d.state.operator_paused,alive=d.controllers.autopilot!=='-',hasFix=!!d.state.framework_fix_available,start=document.getElementById('workflow-start'),pause=document.getElementById('workflow-pause'),note=document.getElementById('control-note');start.disabled=!paused&&alive;pause.disabled=operatorPaused||(!alive&&!d.controllers.check_loop);start.querySelector('span').textContent=hasFix?'应用修复并恢复':operatorPaused||!alive?'启动闭环':(paused?'检查新知识并继续':'运行中');note.className='control-note';note.textContent=operatorPaused?'已由你从控制台暂停。':hasFix?'检测到新的框架提交；启动将走审计恢复并保留全部 Reviewer 与写入闸。':paused?'等待新的人工示例；没有新知识或框架修复时不会重复烧 Token。':alive?'系统正在自主运行，可随时暂停。':'处理服务离线，可从这里启动。'}
async function controlWorkflow(action){const start=document.getElementById('workflow-start'),pause=document.getElementById('workflow-pause'),note=document.getElementById('control-note'),clicked=action==='pause'?pause:start;if(clicked.disabled)return;const actualAction=action==='start'?(snapshot?.state?.start_action||'start'):action,isFixResume=actualAction==='resume_after_fix',question=action==='pause'?'暂停后不会继续调用领域模型或自动处理数据，确认暂停？':isFixResume?'检测到新的框架修复。将通过审计恢复并继续调用已配置模型，确认恢复？':'启动后会调用已配置的付费模型，并按现有 Reviewer 与写入闸推进，确认启动？';if(!confirm(question))return;start.disabled=true;pause.disabled=true;note.className='control-note';note.textContent=action==='pause'?'正在暂停处理服务…':isFixResume?'正在验证框架提交并审计恢复…':'正在启动处理服务并检查新知识…';let token=sessionStorage.getItem('cmr-lge-control-token')||'';async function call(){return fetch('/api/autopilot/control',{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'cmr-lge-board',...(token?{'X-CMR-LGE-Control-Token':token}:{})},body:JSON.stringify({action:actualAction})})}try{let r=await call();if(r.status===403){token=prompt('公网控制需要管理令牌。令牌只保存在当前浏览器会话：')||'';if(!token)throw new Error('未提供控制令牌');sessionStorage.setItem('cmr-lge-control-token',token);r=await call()}const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);note.className=d.status==='waiting_for_human'?'control-note':'control-note ok';note.textContent=d.message||'控制命令已执行';setTimeout(()=>tick(true),1200)}catch(e){note.className='control-note err';note.textContent=e.message;start.disabled=false;pause.disabled=false}}
function renderHumanAttention(d){const a=d.human_attention||{},learning=d.manual_learning||{},box=document.getElementById('human-attention'),reviewed=Number(learning.approved_no_edit_reviews||0),edited=Number(learning.approved_examples||0),decisions=learning.review_decisions||{},trashReviewed=Number(decisions.confirmed_trash||0),checkedReviewed=Number(decisions.confirmed_checked||0),trashTotal=Number(d.boundary_review?.trash||0);box.classList.toggle('visible',!!a.required);document.getElementById('attention-title').textContent=a.title||'需要少量人工示例';document.getElementById('attention-message').textContent=a.message||'';document.getElementById('learning-chip').textContent=`已确认并进入知识链：复核 ${reviewed} · 修改 ${edited}`;document.getElementById('manual-review-progress').innerHTML=`<strong>人工复核</strong><span>已确认 ${reviewed} 例</span><small>确认剔除 ${trashReviewed} · 确认通过 ${checkedReviewed} · 垃圾桶剩余待复核 ${Math.max(0,trashTotal-trashReviewed)}</small>`;const primary=document.getElementById('attention-action');primary.href=a.href||'/editor';primary.textContent=a.action||'复核代表样本';const validation=document.getElementById('validation-action'),remaining=Number(a.validation_remaining||0);validation.href=a.validation_href||'/check-validation';validation.textContent=remaining?`抽样验收（剩余 ${remaining}）`:'查看抽样验收';}
function renderDatasetJob(job={}){
 const active=['STARTING','RUNNING'].includes(job.status),note=document.getElementById('dataset-job-note');
 document.querySelectorAll('.dataset-action').forEach(b=>b.disabled=active);
 const names={IDLE:'尚未启动',STARTING:'准备中',RUNNING:'执行中',SUCCEEDED:'已完成',FAILED:'执行失败',TIMED_OUT:'已超时'};
 const phases={QUEUED:'已排队',STARTING:'准备中',AUDITING:'核验文件与验收记录',VALIDATING:'按现有评分体系验收',REFRESHING_AUDIT:'更新成套数量',FREEZING:'生成独立数据版本',COMPLETE:'完成'};
 const errors={AUDIT_REQUIRED:'请先核验配对',OPERATION_FAILED:'执行失败，详情已保存',TIME_LIMIT:'超时停止，可查看记录',WORKER_EXITED:'工作进程已退出',SUPERVISOR_EXITED:'控制进程已退出',LAUNCH_FAILED:'启动失败'};
 note.className='control-note'+(['FAILED','TIMED_OUT'].includes(job.status)?' err':'');
 note.textContent=job.error_code?(errors[job.error_code]||'任务状态异常'):active?`${phases[job.progress?.phase]||'执行中'} · ${Math.floor(job.elapsed_seconds||0)} 秒`:names[job.status]||'尚未启动';
 const release=job.result?.release,validation=job.result?.validation;
 if(!active&&release)note.textContent+=` · 已导出 ${release.pair_count} 组`;
 else if(!active&&validation)note.textContent+=` · 本批 ${validation.attempted} 组，通过 ${validation.counts?.accepted_pairs||0} 组`;
}
async function controlDataset(action){
 const buttons=[...document.querySelectorAll('.dataset-action')],note=document.getElementById('dataset-job-note');
 if(buttons.some(b=>b.disabled))return;
 buttons.forEach(b=>b.disabled=true);note.textContent='正在提交…';
 let token=sessionStorage.getItem('cmr-lge-control-token')||'';
 async function call(){return fetch('/api/dataset/control',{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'cmr-lge-board',...(token?{'X-CMR-LGE-Control-Token':token}:{})},body:JSON.stringify({action})})}
 try{let r=await call();if(r.status===403){token=prompt('请输入控制台管理令牌')||'';if(!token)throw new Error('未提供管理令牌');sessionStorage.setItem('cmr-lge-control-token',token);r=await call()}const result=await r.json();if(!r.ok)throw new Error(result.error||`HTTP ${r.status}`);renderDatasetJob(result);setTimeout(()=>tick(true),1000)}catch(e){note.className='control-note err';note.textContent=e.message;buttons.forEach(b=>b.disabled=false)}
}
async function tick(force=false){const now=Date.now();if(tickBusy&&now-lastTickAt<9000)return;tickBusy=true;lastTickAt=now;const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),force?20000:8000);try{
 const r=await fetch(force?'/api/state?refresh=1':'/api/state',{cache:'no-store',signal:controller.signal});if(!r.ok)throw new Error(r.status);
 const d=await r.json();snapshot=d;failedTicks=0;lastGoodAt=Date.now();tickBusy=false;clearTimeout(timeout);document.getElementById('offline').classList.remove('confirmed');
const alive=d.controllers.autopilot!=='-',paused=!!d.state.paused,last=d.state.last_event||'-',operatorPaused=!!d.state.operator_paused;
document.getElementById('dot').style.background=paused?'var(--warn)':(alive?'var(--ok)':'var(--bad)');
document.getElementById('ctl').textContent=paused?'已暂停':(alive?'运行中':'离线');
const badge=document.getElementById('system-badge');badge.className='status-badge '+(paused?'warn':(alive?'ok':'bad'));badge.textContent=paused?'自动闭环已暂停':(alive?'自动闭环运行中':'自动闭环离线');
document.getElementById('hero-state').textContent=paused?'工作流已停止推进':(alive?'多智能体正在自主运行':'控制器没有运行');
const heroReason=operatorPaused?'已由你从控制台暂停；看板保持可用，启动后会恢复示例学习、Check Loop 与 Autopilot。':paused?`当前证据已达到本轮能力边界（${last}）。请确认 1–2 个代表样本，系统吸收新知识后会自动续跑。`:alive?`最近事件：${last}，控制器将继续调度下一阶段。`:'处理服务未运行，可直接使用上方按钮启动。';
document.getElementById('hero-reason').textContent=heroReason;
document.getElementById('current-run').textContent=d.runs?.[0]?.run_id?`run ${d.runs[0].run_id}`:'-';document.getElementById('ts').textContent='更新 '+d.generated_at;
renderDatasetJob(d.segmentation_lifecycle?.dataset_job||{});
const serviceRows=[['Autopilot',d.controllers.autopilot],['Check Loop',d.controllers.check_loop],['Worker',d.controllers.worker],['示例学习 Agent',d.controllers.manual_edit_agent]];
document.getElementById('services').innerHTML=serviceRows.map(function(row){const name=row[0],pid=row[1];const state=pid&&pid!=='-'?'在线 · PID '+esc(pid):'离线';return '<div class="service"><b>'+esc(name)+'</b><small>'+state+'</small></div>';}).join('');
 renderLgeExecution(d);renderFlow(d);document.getElementById('agent-flow').dataset.state=paused?'warn':(alive?'active':'idle');requestAnimationFrame(drawFlowWires);updateControl(d);renderHumanAttention(d);document.getElementById('hr-count').textContent=d.label_review_count||0;
const calls=d.calls||{},evl=d.evaluation||{},rules=evl.rules||{},rates=rules.detection_rate||{};document.getElementById('call-count').textContent=calls.calls||0;document.getElementById('call-detail').textContent=`${calls.tokens||0} tokens · ${calls.errors||0} error`;const recalls=Object.values(rates);const gate=(rules.clean_false_positive_rate<=.10&&rules.true_small_false_repair<=.05&&recalls.length===4&&recalls.every(x=>x>=.60));document.getElementById('eval-result').textContent=gate?'PASS':'未达标/缺失';document.getElementById('eval-result').style.color=gate?'var(--ok)':'var(--warn)';const groups=Object.entries(evl.score_groups||{}),mean=t=>{const v=groups.filter(([k])=>k.endsWith(':'+t)).map(([,x])=>Number(x.mean_score)).filter(Number.isFinite);return v.length?(v.reduce((a,b)=>a+b,0)/v.length).toFixed(1):'-'};document.getElementById('eval-detail').textContent=`CMR ${mean('cmr_myo')} · LGE心肌 ${mean('lge_myo')} · MI ${mean('lge_mi')} · FPR ${rules.clean_false_positive_rate??'-'}`;const issueTotal=d.pools.filter(p=>!p.aggregate).reduce((n,p)=>n+p.count,0),boundary=d.boundary_review||{},checkUnresolved=Number(boundary.pending||0)+Number(boundary.work||0),allCheck=Object.values(boundary).reduce((n,v)=>n+Number(v||0),0),resolved=Number(boundary.checked||0)+Number(boundary.trash||0),pct=allCheck?Math.round(100*resolved/allCheck):0;document.getElementById('issue-total').textContent=checkUnresolved;document.getElementById('issue-detail').textContent=`待筛 ${boundary.pending||0} · 待修改 ${boundary.work||0} · 已处理 ${resolved}（${pct}%）`;document.getElementById('resolution-bar').style.width=pct+'%';renderReviewBuckets(boundary);const billing=d.aliyun_billing||{};if(billing.status==='ok'){document.getElementById('aliyun-balance').textContent=`${billing.currency||'CNY'} ${billing.available_amount||'-'}`;document.getElementById('aliyun-detail').textContent=`现金 ${billing.cash_amount||'-'} · 可用代金券 ${billing.coupon_balance||'0'} · ${billing.updated_at||''}`;}else{document.getElementById('aliyun-balance').textContent='未接入';document.getElementById('aliyun-detail').textContent=billing.message||'服务器尚无只读账单凭据';}const staged=d.batch_summary||{};document.getElementById('staged-total').textContent=staged.gate_passed||0;document.getElementById('staged-detail').textContent=`共 ${staged.gate_total||staged.auto_fixed||0} · 拒绝 ${staged.gate_rejected||0} · ${staged.changed_pixels||0} px · 未写正式标签`;document.getElementById('human-kpi').textContent=d.label_review_count||0;document.getElementById('models').innerHTML='<tr><th>模型</th><th>职责</th><th>调用</th><th>Token</th><th>错误</th><th>最近</th></tr>'+((calls.models||[]).map(m=>`<tr><td>${esc(m.model)}</td><td><small>${esc((m.roles||[]).join(', '))}</small></td><td>${m.calls}</td><td>${m.tokens}</td><td>${m.errors}</td><td>${esc(m.last_at)}</td></tr>`).join('')||'<tr><td colspan="6" class="muted">尚无真实 HTTP 调用遥测</td></tr>');
 const sl=d.segmentation_lifecycle||{},ss=sl.summary||{},st=sl.training||{};
 const trainingNames={NOT_STARTED:'尚未启动',PREFLIGHT:'准备检查',AWAITING_APPROVAL:'等待方案评审',PAUSED:'已暂停',TRAINING:'训练中',TRAINED:'训练完成',REGISTERED:'候选已归档',PREDICTING:'对照预测中',AWAITING_EVALUATION:'等待独立验证',ASSESSED:'已完成比较',AWAITING_ACTIVATION_APPROVAL:'等待版本审批',ACTIVATING:'版本切换中',ACTIVATED:'新版本已启用',RETAIN_PARENT:'保留原模型',FAILED:'执行失败',CANCELLED:'已取消'};
 document.getElementById('segmentation-lifecycle').innerHTML=`<div class="kpi"><span>文件配齐</span><b>${ss.complete_files??'—'}</b><small>两幅图像 + 三类标签</small></div><div class="kpi"><span>成套验收通过</span><b>${ss.releasable_pairs??'—'}</b><small>当前文件与验收记录一致</small></div><div class="kpi"><span>尚未纳入通过集</span><b>${ss.blocked_pairs??'—'}</b><small>含 ${ss.excluded_pairs??0} 组固定留出</small></div><div class="kpi"><span>分割模型训练</span><b>${esc(trainingNames[st.status]||'状态待核查')}</b><small>与 QC 和对话模型调用分开计量</small></div>`;
 document.getElementById('dataset-release-time').textContent=sl.generated_at?`验收快照 ${new Date(sl.generated_at).toLocaleString('zh-CN')}`:'尚未核验';
 const registered=Object.values(sl.models||{}).some(m=>m.status==='SELECTED');
const dr=sl.dispatch_readiness||{};const dispatchNote=dr.enabled?'训练调度已配置；仍按方案审批、独立评估和启用审批逐步推进。':'训练调度未启用；缺少可信的训练配置/审批，当前不能开始模型训练。';
document.getElementById('dataset-release-note').textContent=`${sl.release?`最近导出 ${sl.release.pair_count} 组成套数据；上方计数来自当前验收快照。`:'尚未导出成套数据。'}自动 QC 通过与医生确认分别保留来源。${registered?'已配置模型版本；效果以独立比较为准。':'当前仍使用原分割权重，尚未启用新训练版本。'} ${dispatchNote}`;
 const review=d.label_review_tasks||[];renderLabelReview(review);
const cg=d.candidate_gate||{};document.getElementById('gate-summary').textContent=`${cg.total||0} 个候选：通过 ${cg.passed||0}，拒绝 ${cg.rejected||0}。${cg.policy||''}`;document.getElementById('gate-reasons').innerHTML=(cg.rejection_reasons||[]).length?'<div class="kpis">'+cg.rejection_reasons.slice(0,6).map(x=>`<div class="kpi"><span>${esc(x.reason)}</span><b>${esc(x.count)}</b></div>`).join('')+'</div>':'<div class="empty">暂无 gate 拒绝原因</div>';
const br=d.boundary_review||{};document.getElementById('boundary-total').textContent=Object.values(br).reduce((a,b)=>a+Number(b||0),0);document.getElementById('boundary-progress').innerHTML=`<div class="kpi"><span>待筛</span><b>${br.pending||0}</b></div><div class="kpi"><span>待处理</span><b>${br.work||0}</b></div><div class="kpi"><span>垃圾桶</span><b>${br.trash||0}</b></div><div class="kpi"><span>checked</span><b>${br.checked||0}</b></div>`;
const outcome=d.workflow_outcome||{};document.getElementById('workflow-outcome').textContent=`历史维护代码晋升 ${outcome.code_promoted_actions??'-'} 项（最近 run ${outcome.latest_code_promotion?.run_id??'-'}）；影子规则验收：未建立；本轮标签解决：未建立；当前待筛 ${outcome.pending_labels??'-'}。独立复核待补指标 ${(d.validation_backlog||[]).length} 项。代码晋升不等于标签验收。Autopilot ${outcome.autopilot_status||'-'}`;
const ci=d.check_iteration||{};document.getElementById('check-loop').innerHTML=ci.created_at?`<div class="kpis"><div class="kpi"><span>待筛时间帧</span><b>${esc(ci.scanned_frames??'-')}</b></div><div class="kpi"><span>生成 / gate 通过</span><b>${esc(ci.candidate_frames??'-')} / ${esc(ci.gate_passed_frames??'-')}</b></div><div class="kpi"><span>反事实可晋升空间层</span><b>${esc(ci.counterfactual_salvage??'-')}</b></div><div class="kpi"><span>剩余待筛空间层</span><b>${esc(ci.residual_pending??'-')}</b></div></div><p>多智能体 ${esc(ci.agent_decision||'未完成')} / 独立复核 ${esc(ci.reviewer_decision||'未完成')} · 冻结保留集 ${esc(ci.holdout_count??'-')} · SAX 核心控制闸 ${esc(ci.control_gate_status||'未运行')}（候选 ${esc(ci.spatial_boundary_candidates??'-')} / 与已晋升冲突 ${esc(ci.checked_false_routes??'-')}）</p><p>${esc(ci.next_step||'')} · ${esc(ci.created_at||'')}</p>`:'尚无独立 check 迭代实测';
document.getElementById('batch-rejected').innerHTML=(d.batch_rejected||[]).length?'<div class="scroll"><table><tr><th>样本</th><th>候选操作</th><th>拒绝原因</th><th></th></tr>'+d.batch_rejected.map(x=>`<tr><td><b>${esc(x.sample_name||'候选样本')}</b></td><td>${esc((x.operation||[]).join(', '))}</td><td><small>${esc((x.reasons||[]).join(', '))}</small></td><td><button class="btn" onclick="openBatchCandidate('${esc(x.pool)}',${Number(x.index)})">仅查看</button></td></tr>`).join('')+'</table><div class="muted">这里只显示前 ${esc(cg.rejected_visible||0)} 条；完整拒绝数 ${esc(cg.rejected||0)}。拒绝项不能直接应用。</div></div>':'<div class="empty">没有被 gate 拒绝的候选</div>';
document.getElementById('maintenance-count').textContent=`(${d.human_required_count||0})`;document.getElementById('human').innerHTML=d.human_required.length?'<div class="scroll"><table class="review-table"><tr><th>run</th><th>代码/规则动作</th><th>负责人</th><th>关联标签任务</th><th></th></tr>'+d.human_required.map((a,i)=>`<tr data-key="${esc(a.run_id)}:${esc(a.action_id)}" onclick="openReview(${i})"><td>${a.run_id}</td><td><b>${esc(a.action_id)}</b><div class="objective muted">${esc(a.objective)}</div></td><td>${esc(a.owner_role)}</td><td>${esc(a.task_label||'未关联')}</td><td><button class="btn" onclick="event.stopPropagation();openReview(${i})">裁定</button></td></tr>`).join('')+'</table></div>':'<div class="empty">当前没有维护实验需要人工裁定</div>';
document.getElementById('pools').innerHTML='<tr><th>任务类型 / 标签目标</th><th>规模</th><th>当前任务</th><th>下一步</th><th></th></tr>'+d.pools.map(p=>`<tr><td><b>${esc(p.label)}</b><br><small>${esc(p.target)} · ${esc(p.unit)}</small></td><td><b>${esc(p.count)}</b> 待审记录<br><small>${esc(p.patients)} 患者 · ${esc(p.levels)} 层级</small></td><td>${esc(p.task)}<br><small>${(p.top_issues||[]).map(x=>`${esc(x.issue)} ${esc(x.count)}`).join(' · ')}</small></td><td><span class="tag">${esc(p.stage)}</span><br><small>${esc(p.gap)}</small></td><td>${p.aggregate?'':`<button class="btn" onclick="location.href='/editor?pool=${encodeURIComponent(p.key)}&mode=prompt'">打开任务</button>`}</td></tr>`).join('');document.getElementById('agents').innerHTML='<tr><th>时间</th><th>来源</th><th>run</th><th>角色 / 执行器</th><th>模型</th><th>状态</th></tr>'+d.agents.map(a=>`<tr title="${esc(a.detail||'')}"><td>${esc(a.time)}</td><td>${esc(a.src)}</td><td>${esc(a.run_id)}</td><td>${esc(a.who)}</td><td><small>${esc(a.model)}</small></td><td>${t(a.status,a.status)}</td></tr>`).join('');document.getElementById('actions').innerHTML='<tr><th>run</th><th>动作</th><th>关联标签任务</th><th>状态</th></tr>'+d.actions.map(a=>`<tr><td>${a.run_id}</td><td><small>${esc(a.action_id)}</small></td><td><small>${esc(a.task_label||'未关联')}</small></td><td>${t(a.status,a.status)}</td></tr>`).join('');document.getElementById('runs').innerHTML='<tr><th>run</th><th>状态</th><th>步骤</th><th>创建</th></tr>'+d.runs.map(x=>`<tr><td>${x.run_id}</td><td>${t(x.status,x.status)}</td><td><small>${esc(x.current_step)}</small></td><td><small>${esc(String(x.created_at).slice(5,19))}</small></td></tr>`).join('');document.getElementById('events').innerHTML='<tr><th>时间</th><th>事件</th><th>详情</th></tr>'+d.events.map(e=>`<tr><td>${esc(e.t)}</td><td>${esc(e.event)}</td><td><small>${esc(e.detail||'')}</small></td></tr>`).join('');}catch(err){document.getElementById('offline').style.display='inline';}}
let maintMessages=[];
try{maintMessages=JSON.parse(localStorage.getItem('cmr-lge-maint-chat')||'[]').slice(-12)}catch(e){maintMessages=[]}
function toggleMaint(open){document.getElementById('maint-panel').classList.toggle('open',open);if(open){renderMaint();document.getElementById('maint-input').focus()}}
function renderMaint(){const log=document.getElementById('maint-log');log.innerHTML='';if(!maintMessages.length){const welcome=document.createElement('div');welcome.className='chat-bubble assistant';welcome.textContent='我是项目维护助手。可以结合当前服务、Agent、配置和事件状态定位控制面问题，并给出受控修复方案；不会替领域智能体判断或修改标签。';log.appendChild(welcome)}for(const m of maintMessages){const box=document.createElement('div');box.className='chat-bubble '+m.role;box.textContent=m.content;if(m.meta){const meta=document.createElement('span');meta.className='chat-meta';meta.textContent=m.meta;box.appendChild(meta)}log.appendChild(box)}log.scrollTop=log.scrollHeight}
function saveMaint(){localStorage.setItem('cmr-lge-maint-chat',JSON.stringify(maintMessages.slice(-12)))}
function clearMaint(){maintMessages=[];saveMaint();renderMaint()}
function setMaintPrompt(text){document.getElementById('maint-input').value=text;document.getElementById('maint-input').focus()}
async function sendMaint(){const input=document.getElementById('maint-input'),btn=document.getElementById('maint-send'),worked=document.getElementById('maint-worked'),content=input.value.trim();if(!content||btn.disabled)return;maintMessages.push({role:'user',content});input.value='';saveMaint();renderMaint();btn.disabled=true;let seconds=0;worked.textContent='正在读取控制面状态并调用维护模型 · Worked for 0s';const timer=setInterval(()=>{seconds++;worked.textContent=`正在分析 · Worked for ${seconds}s`},1000);try{const r=await fetch('/api/maintenance-chat',{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'cmr-lge-board'},body:JSON.stringify({messages:maintMessages.slice(-12)})});const d=await r.json();if(!r.ok)throw new Error(d.error||`HTTP ${r.status}`);maintMessages.push({role:'assistant',content:d.reply,meta:`${d.model} · Worked for ${d.worked_for}s`});document.getElementById('maint-model').textContent=d.model;worked.textContent='控制面专用 · 不包含影像、标签和密钥';}catch(e){maintMessages.push({role:'assistant',content:'调用失败：'+e.message,meta:'未执行任何修改'});worked.textContent='维护助手调用失败';}finally{clearInterval(timer);btn.disabled=false;saveMaint();renderMaint()}}
document.getElementById('maint-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMaint()}});
function updateConnectionWarning(){document.getElementById('offline').classList.toggle('confirmed',Date.now()-lastGoodAt>=15000)}
tick();setInterval(tick,3000);setInterval(()=>{updateConnectionWarning();if(snapshot){renderFlow(snapshot);requestAnimationFrame(drawFlowWires)}},1000);window.addEventListener('resize',()=>requestAnimationFrame(drawFlowWires));
</script></body></html>"""


SCORE_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>评估记分卡</title>
<style>
body{margin:0;background:#0f1419;color:#d7e0ea;font:13px/1.6 ui-monospace,Consolas,"Microsoft YaHei",monospace;padding:18px}
h1{font-size:17px}h2{font-size:13px;color:#58a6ff;margin:18px 0 6px}
.card{background:#171e26;border:1px solid #243040;border-radius:8px;padding:12px 14px;margin-bottom:12px}
table{border-collapse:collapse}th,td{text-align:left;padding:3px 10px 3px 0;border-bottom:1px solid #1d2733}
th{color:#7d8ea1;font-weight:500}.muted{color:#7d8ea1}a{color:#58a6ff}
.ok{color:#3fb950}.bad{color:#f85149}.warn{color:#d29922}
</style></head><body>
<h1>评估记分卡 <span class="muted" id="ts"></span></h1>
<p class="muted">测试与评分系统统一视图 · 10 秒自动刷新 · <a href="/">返回看板</a> · <a href="/api/score">原始 JSON</a></p>
<div id="root">加载中…</div>
<script>
const esc=v=>String(v??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const pct=v=>v==null?'-':(v*100).toFixed(1)+'%';
const taskLabel=k=>({cmr_boundary_manifest:'CMR 边界筛选',cmr_middle_manifest:'CMR 标签修复',lge_boundary_manifest:'LGE 边界筛选',lge_issue_manifest:'LGE 问题汇总',lge_middle_manifest:'LGE 标签修复',trajectory_manifest:'跨层位置校准'}[k]||'标签复核任务');
function card(t,body){return `<div class="card"><h2>${t}</h2>${body}</div>`}
function kv(obj){return Object.entries(obj||{}).map(([k,v])=>`<tr><th>${esc(k)}</th><td>${typeof v==='object'?esc(JSON.stringify(v)):esc(v)}</td></tr>`).join('')}
async function tick(){
 try{
  const d=await (await fetch('/api/score',{cache:'no-store'})).json();
  document.getElementById('ts').textContent=d.generated_at;
  let h='';
  const r=d.rules_v2||{};
  const dr=r.detection_rate||{};
  h+=card('① 规则 v2 · 冻结评估集（'+((d.eval_set||{}).total||'?')+' 条：金标准/准/外部）',
   `<table><tr><th>干净集误报</th><td class="${r.clean_false_positive_rate<=0.05?'ok':'bad'}">${pct(r.clean_false_positive_rate)}</td><th>真小面积误伤</th><td class="${(r.true_small_false_repair||0)<=0.03?'ok':'bad'}">${pct(r.true_small_false_repair)}</td></tr>
    <tr><th>碎片化检出</th><td>${pct(dr.fragmentation)}</td><th>洞缺检出</th><td>${pct(dr.hole_fill)}</td><th>边界裁剪</th><td>${pct(dr.border_clip)}</td><th>单帧跳变</th><td>${pct(dr.layer_jump)}</td></tr></table>`);
  const g=d.gate;
  if(g)h+=card('② 确定性解剖 gate · 批量候选',
   `<table><tr><th>候选</th><td>${g.total}</td><th>通过</th><td class="ok">${g.passed}</td><th>拒绝</th><td class="bad">${g.rejected}</td><th>通过率</th><td class="${g.pass_rate>=0.5?'ok':'warn'}">${pct(g.pass_rate)}</td></tr></table>
    <div class="muted" style="margin-top:6px">拒绝原因（用于反哺规则迭代）：</div><table>${kv(g.reject_reasons)}</table>`);
  const bs=d.batch_screening||{};
  let bt='<table><tr><th>任务类型</th><th>已扫</th><th>命中</th><th>自动候选</th><th>改像素</th><th>状态</th></tr>';
  for(const[k,v]of Object.entries(bs))bt+=`<tr><td>${esc(taskLabel(k))}</td><td>${v.scanned}</td><td>${v.flagged}</td><td>${v.auto_fixed}</td><td>${v.changed_pixels_total||0}</td><td>${v.running?'<span class="warn">运行中</span>':'<span class="ok">完成</span>'}</td></tr>`;
  h+=card('③ 批量筛查（全样本自动 check）',bt+'</table>');
  const e=d.edits||{},pj=d.prompt_agent_jobs||{};
  h+=card('④ 修复来源对比（自动 vs Agent vs 手工）',
   `<table><tr><th>人工/编辑器补丁</th><td>${e.total||0} 个</td><th>${esc(JSON.stringify(e.by_origin||{}))}</th><td>${esc(JSON.stringify(e.by_status||{}))}</td></tr>
    <tr><th>提示词 Agent 任务</th><td>${Object.values(pj).reduce((a,b)=>a+b,0)}</td><td>${esc(JSON.stringify(pj))}</td><td></td></tr></table>`);
  const tri=d.boundary_triage;
  if(tri&&tri.buckets)h+=card('⑥ Boundary 分诊（心尖/心底三桶 · '+(tri.rows||0)+' 行'+(tri.volumes_done&&tri.volumes_done<tri.volumes?' · 运行中':'')+'）',
   `<table><tr><th>丢弃 DISCARD</th><td>${tri.buckets.DISCARD} (${pct(tri.discard_rate)})</td><th>捞回 SALVAGE</th><td class="ok">${tri.buckets.SALVAGE} (${pct(tri.salvage_rate)})</td><th>人工 REVIEW</th><td class="warn">${tri.buckets.REVIEW} (${pct(tri.review_rate)})</td></tr></table><div class="muted">丢弃=无血池腔或面积极小（心尖帽/心底实心）；捞回=环形态完整、面积≥45%本例参考；仅 REVIEW 桶需逐层人工。</div>`);
  if(d.quality_scoring)h+=card('⑤ 图像-标签质量评分 v1','<pre class="muted">'+esc(JSON.stringify(d.quality_scoring,null,1).slice(0,700))+'</pre>');
  const ci=d.check_iteration||{};
  if(ci.created_at)h+=card('⑦ check 闭环验收（空间层级）',
   `<table><tr><th>待筛</th><td>${esc(ci.pending)}</td><th>已晋升</th><td>${esc(ci.checked)}</td><th>隔离</th><td>${esc(ci.trash)}</td></tr>
    <tr><th>直通</th><td>${esc(ci.promotion?.direct_promoted??'-')}</td><th>修正后晋升</th><td>${esc(ci.promotion?.fix_applied??'-')}</td><th>修正帧</th><td>${esc(ci.promotion?.fixed_frames??'-')}</td></tr>
    <tr><th>帧候选 gate</th><td>${esc(ci.gate_passed_frames)} / ${esc(ci.candidate_frames)}</td><th>整层可晋升</th><td>${esc(ci.counterfactual_salvage)}</td><th>剩余</th><td class="warn">${esc(ci.residual_pending)}</td></tr>
    <tr><th>SAX 候选</th><td>${esc(ci.spatial_boundary_candidates)}</td><th>已晋升冲突</th><td>${esc(ci.checked_false_routes)}</td><th>控制闸</th><td class="${ci.control_gate_status==='PASS'?'ok':'bad'}">${esc(ci.control_gate_status)}</td></tr>
    <tr><th>候选质量评分</th><td>${esc(ci.candidate_quality?.scored_volumes??'-')} 层</td><th>升 / 降 / 不变</th><td>${esc(ci.candidate_quality?.improved??'-')} / ${esc(ci.candidate_quality?.degraded??'-')} / ${esc(ci.candidate_quality?.unchanged??'-')}</td><th>中位变化</th><td>${esc(ci.candidate_quality?.median_delta??'-')}</td></tr></table>
    <div class="muted">${esc(ci.next_step||'')} · 保留集 ${esc(ci.holdout_count??'-')} · ${esc(ci.created_at)}</div>`);
  const wo=d.workflow_outcome||{};
  const backlog=d.validation_backlog||[];
  h+=card('⑧ 维护代码与标签验收边界',`<table><tr><th>历史实验代码晋升</th><td>${esc(wo.code_promoted_actions??'-')} 项</td><th>最近</th><td>run ${esc(wo.latest_code_promotion?.run_id??'-')}</td></tr><tr><th>影子规则验收</th><td class="warn">未建立</td><th>本轮标签解决</th><td class="warn">未建立</td></tr><tr><th>当前待筛</th><td>${esc(wo.pending_labels??'-')}</td><th>Autopilot</th><td>${esc(wo.autopilot_status||'-')}</td></tr></table><div class="muted">PROMOTED 仅表示维护代码晋升，不代表独立保留集验证或标签解决。</div><div class="muted">独立复核待补指标（最近 ${backlog.length} 项）：</div><ul>${backlog.map(x=>`<li>run ${esc(x.run_id)} · ${esc(x.status)}：${esc((x.required_validation_metrics||[]).join('；'))}</li>`).join('')||'<li>无待补指标记录</li>'}</ul>`);
  document.getElementById('root').innerHTML=h;
 }catch(err){document.getElementById('root').textContent='加载失败: '+err.message}
}
tick();setInterval(tick,10000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def respond_json(self, status: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/state":
            body = json.dumps(
                collect(refresh=query.get("refresh", ["0"])[0] == "1"),
                ensure_ascii=False,
            ).encode()
            ctype = "application/json; charset=utf-8"
        elif path == "/api/check-inventory":
            self.respond_json(200, build_check_inventory(DATASET, ROOT / "runtime"))
            return
        elif path == "/api/check-validation":
            try:
                self.respond_json(200, review_state(DATASET, ROOT / "runtime"))
            except (ValueError, OSError, KeyError) as exc:
                self.respond_json(409, {"error": str(exc)})
            return
        elif path == "/check-inventory":
            body = CHECK_INVENTORY_PAGE.read_bytes()
            ctype = "text/html; charset=utf-8"
        elif path == "/check-validation":
            body = CHECK_VALIDATION_PAGE.read_bytes()
            ctype = "text/html; charset=utf-8"
        elif path == "/api/editor/cases":
            try:
                result = list_candidates(
                    DATASET,
                    query.get("pool", [""])[0],
                    offset=int(query.get("offset", ["0"])[0]),
                    limit=int(query.get("limit", ["100"])[0]),
                    review_status=query.get("status", [None])[0],
                    group_spatial=query.get("group_spatial", ["0"])[0] == "1",
                    include_excluded=query.get("include_excluded", ["0"])[0] == "1",
                    hide_reviewed=query.get("hide_reviewed", ["0"])[0] == "1",
                )
            except (ValueError, OSError) as exc:
                self.respond_json(400, {"error": str(exc)})
                return
            body = json.dumps(result, ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        elif path == "/api/score":
            score = build_score_card()
            score["check_iteration"] = check_iteration_summary()
            state = _read_json(STATE) or {}
            with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as db:
                db.row_factory = sqlite3.Row
                actions = [dict(zip(("run_id", "action_id", "status"), row)) for row in
                           db.execute("SELECT run_id,action_id,status FROM maintenance_actions")]
                score["validation_backlog"] = reviewer_validation_backlog(db)
            score["workflow_outcome"] = workflow_outcome_summary(
                actions, state, boundary_review_counts(DATASET))
            self.respond_json(200, score)
            return
        elif path == "/score":
            body = SCORE_PAGE.encode()
            ctype = "text/html; charset=utf-8"
        elif path == "/api/brief":
            self.respond_json(200, escalation_brief(
                query.get("run_id", ["0"])[0], query.get("action_id", [""])[0]))
            return
        elif path == "/api/editor/action":
            self.respond_json(200, editor_action(
                query.get("run_id", ["0"])[0], query.get("action_id", [""])[0]))
            return
        elif path == "/api/editor/meta":
            try:
                result = frame_metadata(
                    DATASET,
                    query.get("pool", [""])[0],
                    int(query.get("index", ["-1"])[0]),
                )
            except (ValueError, IndexError, FileNotFoundError, OSError) as exc:
                self.respond_json(400, {"error": str(exc)})
                return
            body = json.dumps(result, ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        elif path == "/api/editor/edit":
            try:
                result = edit_status(MANUAL_WORK, query.get("edit_id", [""])[0])
            except (ValueError, FileNotFoundError, OSError) as exc:
                self.respond_json(400, {"error": str(exc)})
                return
            body = json.dumps(result, ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        elif path == "/api/editor/prompt-repair":
            try:
                result = prompt_repair_status(
                    MANUAL_WORK, query.get("job_id", [""])[0]
                )
            except (ValueError, FileNotFoundError, OSError) as exc:
                self.respond_json(400, {"error": str(exc)})
                return
            body = json.dumps(result, ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        elif path == "/api/editor/prompt-mask":
            try:
                body = prompt_candidate_png(
                    MANUAL_WORK, query.get("job_id", [""])[0]
                )
            except (ValueError, FileNotFoundError, OSError) as exc:
                self.respond_json(400, {"error": str(exc)})
                return
            ctype = "image/png"
        elif path == "/api/editor/layer":
            try:
                body = png_layer(
                    DATASET,
                    query.get("pool", [""])[0],
                    int(query.get("index", ["-1"])[0]),
                    query.get("layer", [""])[0],
                    int(query.get("neighbor", ["0"])[0]),
                    layer_neighbor=int(query.get("layer_neighbor", ["0"])[0]),
                )
            except (ValueError, IndexError, FileNotFoundError, OSError) as exc:
                self.respond_json(400, {"error": str(exc)})
                return
            ctype = "image/png"
        elif path in ("/editor", "/editor/"):
            body = EDITOR_PAGE.read_bytes()
            ctype = "text/html; charset=utf-8"
        elif path in ("/", "/index.html"):
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/reviews", "/api/editor/edits", "/api/editor/apply",
                        "/api/check-validation",
                        "/api/editor/prompt-repairs", "/api/editor/boundary-status",
                        "/api/editor/exclusion",
                        "/api/translate", "/api/maintenance-chat",
                        "/api/autopilot/control", "/api/dataset/control"}:
            self.respond_json(404, {"error": "not found"})
            return
        if self.headers.get("X-Requested-With") != "cmr-lge-board":
            self.respond_json(403, {"error": "missing local-board request marker"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.respond_json(400, {"error": "invalid Content-Length"})
            return
        max_length = 8 * 1024 * 1024 if path == "/api/editor/edits" else 65536
        if length <= 0 or length > max_length:
            self.respond_json(413, {"error": f"payload must be 1..{max_length} bytes"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if path == "/api/dataset/control":
                if not board_control_authorized(
                        self.headers.get("Host", ""),
                        self.headers.get("X-CMR-LGE-Control-Token", "")):
                    self.respond_json(403, {"error": "请提供控制台管理令牌"})
                    return
                from app.dataset_control import launch
                result = launch(ROOT, DATASET, str(payload.get("action") or ""))
            elif path == "/api/autopilot/control":
                if not board_control_authorized(
                        self.headers.get("Host", ""),
                        self.headers.get("X-CMR-LGE-Control-Token", "")):
                    self.respond_json(403, {
                        "error": "公网控制未授权：请配置并输入 CMR_LGE_BOARD_CONTROL_TOKEN"
                    })
                    return
                action = str(payload.get("action") or "")
                if action in {"start", "pause"}:
                    result = control_workflow(action)
                elif action == "resume_after_fix":
                    result = resume_autopilot()
                else:
                    raise ValueError("unsupported workflow control action")
            elif path == "/api/maintenance-chat":
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                result = maintenance_chat(messages)
            elif path == "/api/translate":
                texts = payload.get("texts")
                if not isinstance(texts, list):
                    raise ValueError("texts must be a list")
                translations = translate_texts(
                    [str(x)[:4000] for x in texts[:12]], brief=bool(payload.get("brief")))
                result = {"translations": translations,
                          "translate_error": getattr(translate_texts, "last_error", None)}
            elif path == "/api/reviews":
                result = submit_review(payload)
            elif path == "/api/check-validation":
                with CHECK_VALIDATION_LOCK:
                    result = submit_decision(DATASET, ROOT / "runtime", payload)
            elif path == "/api/editor/edits":
                MANUAL_WORK.mkdir(parents=True, exist_ok=True)
                result = save_edit(DATASET, MANUAL_WORK, payload)
            elif path == "/api/editor/prompt-repairs":
                MANUAL_WORK.mkdir(parents=True, exist_ok=True)
                result = queue_prompt_repair(DATASET, MANUAL_WORK, payload)
                from app.prompt_dispatch import launch
                launch(ROOT,MANUAL_WORK,result["job_id"])
            elif path == "/api/editor/boundary-status":
                result = set_boundary_review_status(
                    DATASET, str(payload.get("pool") or ""), int(payload.get("index", -1)),
                    str(payload.get("status") or ""),
                    reviewer=str(payload.get("reviewer") or "local_operator"),
                )
            elif path == "/api/editor/exclusion":
                if not isinstance(payload.get("excluded"), bool):
                    raise ValueError("excluded must be a boolean")
                result = set_item_excluded(
                    DATASET, str(payload.get("pool") or ""), int(payload.get("index", -1)),
                    excluded=payload.get("excluded") is True,
                    reason=str(payload.get("reason") or ""),
                    detail=str(payload.get("detail") or ""),
                    reviewer=str(payload.get("reviewer") or "local_operator"),
                )
            else:
                result = apply_staged_edit_local(
                    dataset_root=DATASET,
                    work_root=MANUAL_WORK,
                    audit_path=LABEL_APPLY_AUDIT,
                    edit_id=str(payload.get("edit_id") or ""),
                    reviewer=str(payload.get("reviewer") or ""),
                    confirmation=str(payload.get("confirmation") or ""),
                )
        except json.JSONDecodeError:
            self.respond_json(400, {"error": "invalid JSON"})
            return
        except KeyError as exc:
            self.respond_json(404, {"error": str(exc)})
            return
        except AutopilotControlError as exc:
            self.respond_json(exc.status, {"error": str(exc)})
            return
        except (TypeError, ValueError, IndexError, FileNotFoundError, OSError) as exc:
            self.respond_json(400, {"error": str(exc)})
            return
        except sqlite3.Error as exc:
            self.respond_json(409, {"error": f"database conflict: {exc}"})
            return
        self.respond_json(201, result)

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        ap.error("direct label apply requires a loopback-only board listener")
    Database(DB).initialize()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"board on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
