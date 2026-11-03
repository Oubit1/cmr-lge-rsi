from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.state_machine import RunState, validate_transition


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  raw_path TEXT,
  processed_path TEXT NOT NULL UNIQUE,
  data_hash TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workflow_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id),
  status TEXT NOT NULL,
  execute_mode TEXT NOT NULL,
  code_commit TEXT,
  config_json TEXT NOT NULL,
  current_step TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
  task_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES workflow_runs(run_id),
  task_type TEXT NOT NULL,
  status TEXT NOT NULL,
  command_json TEXT NOT NULL,
  gpu_id TEXT,
  attempt INTEGER NOT NULL DEFAULT 1,
  exit_code INTEGER,
  pid INTEGER,
  heartbeat_at TEXT,
  log_path TEXT,
  started_at TEXT,
  finished_at TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES workflow_runs(run_id),
  task_id INTEGER REFERENCES tasks(task_id),
  artifact_type TEXT NOT NULL,
  path TEXT NOT NULL,
  checksum TEXT,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, path)
);
CREATE TABLE IF NOT EXISTS agent_outputs (
  output_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES workflow_runs(run_id),
  role TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  status TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  output_json TEXT NOT NULL,
  error_message TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_messages (
  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES workflow_runs(run_id),
  thread_id TEXT NOT NULL,
  sender_role TEXT NOT NULL,
  recipient_role TEXT NOT NULL,
  message_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  reply_to INTEGER REFERENCES agent_messages(message_id),
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_invocations (
  invocation_id TEXT PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES workflow_runs(run_id),
  thread_id TEXT NOT NULL,
  role TEXT NOT NULL,
  provider TEXT NOT NULL,
  requested_model TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  input_json TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  error_message TEXT
);
CREATE TABLE IF NOT EXISTS agent_invocation_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  invocation_id TEXT NOT NULL REFERENCES agent_invocations(invocation_id),
  attempt INTEGER NOT NULL,
  actual_model TEXT NOT NULL,
  status TEXT NOT NULL,
  fallback_reason TEXT,
  request_id TEXT,
  latency_ms INTEGER,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  total_tokens INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_run_id INTEGER NOT NULL REFERENCES workflow_runs(run_id),
  status TEXT NOT NULL,
  hypothesis_json TEXT NOT NULL,
  worktree_path TEXT,
  baseline_metrics_json TEXT,
  candidate_metrics_json TEXT,
  decision_json TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
);
CREATE TABLE IF NOT EXISTS agent_scorecards (
  scorecard_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES workflow_runs(run_id),
  role TEXT NOT NULL,
  score_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS maintenance_actions (
  action_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES workflow_runs(run_id),
  action_id TEXT NOT NULL,
  owner_role TEXT NOT NULL,
  action_class TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  artifact_path TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT,
  UNIQUE(run_id, action_id)
);
CREATE TABLE IF NOT EXISTS human_action_reviews (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  action_id TEXT NOT NULL,
  decision TEXT NOT NULL,
  reviewer TEXT NOT NULL,
  note TEXT NOT NULL,
  previous_status TEXT NOT NULL,
  resulting_status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id, action_id) REFERENCES maintenance_actions(run_id, action_id)
);
CREATE TABLE IF NOT EXISTS evolution_variants (
  variant_id TEXT PRIMARY KEY,
  parent_variant_id TEXT REFERENCES evolution_variants(variant_id),
  source_run_id INTEGER REFERENCES workflow_runs(run_id),
  action_id TEXT,
  workflow_stage TEXT,
  code_commit TEXT,
  continuity_revision TEXT,
  artifact_path TEXT,
  rollback_manifest_path TEXT,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evolution_evaluations (
  evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  variant_id TEXT NOT NULL REFERENCES evolution_variants(variant_id),
  parent_variant_id TEXT REFERENCES evolution_variants(variant_id),
  benchmark_id TEXT NOT NULL,
  benchmark_hash TEXT NOT NULL,
  baseline_metrics_json TEXT NOT NULL,
  candidate_metrics_json TEXT NOT NULL,
  constraints_json TEXT NOT NULL,
  decision TEXT NOT NULL,
  reasons_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evolution_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  variant_id TEXT REFERENCES evolution_variants(variant_id),
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON workflow_runs(status);
CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_scorecards_run ON agent_scorecards(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_messages_inbox
  ON agent_messages(run_id, recipient_role, thread_id, message_id);
CREATE INDEX IF NOT EXISTS idx_agent_invocations_run
  ON agent_invocations(run_id, started_at);
-- Keep dashboard scans off the large stored invocation input/output payloads.
CREATE INDEX IF NOT EXISTS idx_agent_invocations_dashboard
  ON agent_invocations(started_at,invocation_id,run_id,thread_id,role,provider,
                       requested_model,input_hash,status,finished_at);
CREATE INDEX IF NOT EXISTS idx_agent_invocation_events_invocation
  ON agent_invocation_events(invocation_id, event_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_actions_run ON maintenance_actions(run_id);
CREATE INDEX IF NOT EXISTS idx_human_action_reviews_action
  ON human_action_reviews(run_id, action_id, review_id);
CREATE INDEX IF NOT EXISTS idx_evolution_variants_parent
  ON evolution_variants(parent_variant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_evolution_variants_status
  ON evolution_variants(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_evolution_evaluations_variant
  ON evolution_evaluations(variant_id, evaluation_id);
CREATE INDEX IF NOT EXISTS idx_evolution_events_variant
  ON evolution_events(variant_id, event_id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            task_columns = {row["name"] for row in connection.execute("PRAGMA table_info(tasks)")}
            if "pid" not in task_columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN pid INTEGER")
            if "heartbeat_at" not in task_columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN heartbeat_at TEXT")

    def register_dataset(self, name: str, processed_path: Path, raw_path: Path | None = None) -> int:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO datasets(name, raw_path, processed_path, created_at) VALUES(?,?,?,?)",
                (name, str(raw_path.resolve()) if raw_path else None, str(processed_path.resolve()), utcnow()),
            )
            row = connection.execute("SELECT dataset_id FROM datasets WHERE processed_path=?", (str(processed_path.resolve()),)).fetchone()
            return int(row["dataset_id"])

    def create_run(self, dataset_id: int, execute_mode: str, config: dict[str, Any]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO workflow_runs(dataset_id,status,execute_mode,config_json,created_at) VALUES(?,?,?,?,?)",
                (dataset_id, RunState.REGISTERED.value, execute_mode, json.dumps(config, ensure_ascii=False), utcnow()),
            )
            return int(cursor.lastrowid)

    def transition(self, run_id: int, target: RunState, *, current_step: str | None = None,
                   error_message: str | None = None) -> None:
        with self.connect() as connection:
            row = connection.execute("SELECT status FROM workflow_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            validate_transition(row["status"], target.value)
            terminal = {RunState.QC_PASSED, RunState.QC_FAILED, RunState.ERROR,
                        RunState.PROMOTED, RunState.REJECTED, RunState.CANCELLED,
                        RunState.SUPERSEDED}
            if target == RunState.QUEUED:
                connection.execute(
                    "UPDATE workflow_runs SET status=?,current_step=COALESCE(?,current_step),"
                    "error_message=?,started_at=NULL,finished_at=NULL WHERE run_id=?",
                    (target.value, current_step, error_message, run_id),
                )
            elif target == RunState.RUNNING:
                connection.execute(
                    "UPDATE workflow_runs SET status=?,current_step=COALESCE(?,current_step),"
                    "error_message=?,started_at=?,finished_at=NULL WHERE run_id=?",
                    (target.value, current_step, error_message, utcnow(), run_id),
                )
            elif target in terminal:
                connection.execute(
                    "UPDATE workflow_runs SET status=?,current_step=COALESCE(?,current_step),"
                    "error_message=?,finished_at=? WHERE run_id=?",
                    (target.value, current_step, error_message, utcnow(), run_id),
                )
            else:
                connection.execute(
                    "UPDATE workflow_runs SET status=?,current_step=COALESCE(?,current_step),"
                    "error_message=? WHERE run_id=?",
                    (target.value, current_step, error_message, run_id),
                )

    def get_run(self, run_id: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT r.*,d.name dataset_name,d.raw_path,d.processed_path FROM workflow_runs r "
                "JOIN datasets d ON d.dataset_id=r.dataset_id WHERE r.run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            result = dict(row)
            result["config"] = json.loads(result.pop("config_json"))
            return result

    def claim_run(self, run_id: int) -> bool:
        """Atomically claim one queued run when its dataset is not active."""
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE workflow_runs SET status=?,current_step=?,started_at=?,finished_at=NULL "
                "WHERE run_id=? AND status=? AND NOT EXISTS ("
                "SELECT 1 FROM workflow_runs active "
                "WHERE active.dataset_id=workflow_runs.dataset_id AND active.status=? "
                "AND active.run_id<>workflow_runs.run_id)",
                (RunState.RUNNING.value, "claimed_by_worker", utcnow(), run_id,
                 RunState.QUEUED.value, RunState.RUNNING.value),
            )
            return cursor.rowcount == 1

    def next_queued_run_id(self) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT queued.run_id FROM workflow_runs queued WHERE queued.status=? "
                "AND NOT EXISTS (SELECT 1 FROM workflow_runs active "
                "WHERE active.dataset_id=queued.dataset_id AND active.status=?) "
                "ORDER BY queued.run_id LIMIT 1",
                (RunState.QUEUED.value, RunState.RUNNING.value),
            ).fetchone()
            return int(row["run_id"]) if row else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT r.run_id,r.status,r.execute_mode,r.current_step,r.created_at,d.name dataset_name,"
                "d.processed_path FROM workflow_runs r JOIN datasets d ON d.dataset_id=r.dataset_id "
                "ORDER BY r.run_id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def create_task(self, run_id: int, task_type: str, command: list[str], log_path: Path,
                    gpu_id: str | None = None) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO tasks(run_id,task_type,status,command_json,gpu_id,log_path) VALUES(?,?,?,?,?,?)",
                (run_id, task_type, "QUEUED", json.dumps(command), gpu_id, str(log_path)),
            )
            return int(cursor.lastrowid)

    def update_task(self, task_id: int, status: str, *, exit_code: int | None = None) -> None:
        with self.connect() as connection:
            started = utcnow() if status == "RUNNING" else None
            finished = utcnow() if status in {"SUCCEEDED", "FAILED"} else None
            connection.execute(
                "UPDATE tasks SET status=?,exit_code=COALESCE(?,exit_code),started_at=COALESCE(?,started_at),"
                "finished_at=COALESCE(?,finished_at) WHERE task_id=?",
                (status, exit_code, started, finished, task_id),
            )

    def mark_task_process(self, task_id: int, pid: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE tasks SET pid=?,heartbeat_at=? WHERE task_id=?",
                (pid, utcnow(), task_id),
            )

    def touch_task(self, task_id: int) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE tasks SET heartbeat_at=? WHERE task_id=?", (utcnow(), task_id))

    def fail_open_tasks(self, run_id: int, *, exit_code: int = -15) -> int:
        """Close dangling task rows when a worker has been confirmed dead."""
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status='FAILED',exit_code=?,finished_at=? "
                "WHERE run_id=? AND status IN ('QUEUED','RUNNING')",
                (exit_code, utcnow(), run_id),
            )
            return cursor.rowcount

    def register_artifact(self, run_id: int, task_id: int | None, artifact_type: str, path: Path,
                          metadata: dict[str, Any], checksum: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO artifacts(run_id,task_id,artifact_type,path,checksum,metadata_json,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (run_id, task_id, artifact_type, str(path.resolve()), checksum,
                 json.dumps(metadata, ensure_ascii=False), utcnow()),
            )

    def save_agent_output(self, run_id: int, role: str, provider: str, model: str, status: str,
                          input_hash: str, output: dict[str, Any], error_message: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO agent_outputs(run_id,role,provider,model,status,input_hash,output_json,error_message,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, role, provider, model, status, input_hash,
                 json.dumps(output, ensure_ascii=False), error_message, utcnow()),
            )

    def begin_agent_invocation(self, invocation_id: str, *, run_id: int, thread_id: str,
                               role: str, provider: str, requested_model: str,
                               input_hash: str, input_data: dict[str, Any]) -> None:
        """Persist the exact de-identified payload delivered to one role."""
        public_input = {
            key: value for key, value in input_data.items()
            if not str(key).startswith("_")
        }
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO agent_invocations(invocation_id,run_id,thread_id,role,provider,"
                "requested_model,input_hash,input_json,status,started_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (invocation_id, run_id, thread_id, role, provider, requested_model,
                 input_hash, json.dumps(public_input, ensure_ascii=False), "RUNNING", utcnow()),
            )

    def record_agent_invocation_event(self, invocation_id: str,
                                      event: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO agent_invocation_events(invocation_id,attempt,actual_model,status,"
                "fallback_reason,request_id,latency_ms,prompt_tokens,completion_tokens,"
                "total_tokens,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (invocation_id, int(event.get("attempt") or 1),
                 str(event.get("model") or "unknown"), str(event.get("status") or "UNKNOWN"),
                 event.get("fallback_reason"), event.get("request_id"),
                 int(event.get("latency_ms") or 0), int(event.get("prompt_tokens") or 0),
                 int(event.get("completion_tokens") or 0), int(event.get("total_tokens") or 0),
                 str(event.get("timestamp") or utcnow())),
            )

    def finish_agent_invocation(self, invocation_id: str, *, status: str,
                                error_message: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE agent_invocations SET status=?,finished_at=?,error_message=? "
                "WHERE invocation_id=?",
                (status, utcnow(), error_message, invocation_id),
            )

    def agent_invocation_event_count(self, invocation_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM agent_invocation_events WHERE invocation_id=?",
                (invocation_id,),
            ).fetchone()
            return int(row["count"])

    def post_agent_message(self, run_id: int, *, thread_id: str, sender_role: str,
                           recipient_role: str, message_type: str,
                           payload: dict[str, Any], reply_to: int | None = None) -> int:
        """Persist one validated role-to-role handoff.

        The orchestrator transports JSON but does not reinterpret the medical
        content.  Sender and recipient are explicit so downstream execution can
        prove which agents supplied and reviewed an action.
        """
        if not all(str(value).strip() for value in (
            thread_id, sender_role, recipient_role, message_type
        )):
            raise ValueError("agent message routing fields must be non-empty")
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO agent_messages(run_id,thread_id,sender_role,recipient_role,"
                "message_type,payload_json,reply_to,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (run_id, thread_id, sender_role, recipient_role, message_type,
                 json.dumps(payload, ensure_ascii=False), reply_to, utcnow()),
            )
            return int(cursor.lastrowid)

    def agent_inbox(self, run_id: int, recipient_role: str, *,
                    thread_id: str | None = None) -> list[dict[str, Any]]:
        clauses = ["run_id=?", "recipient_role=?"]
        params: list[Any] = [run_id, recipient_role]
        if thread_id is not None:
            clauses.append("thread_id=?")
            params.append(thread_id)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_messages WHERE " + " AND ".join(clauses)
                + " ORDER BY message_id", params,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def save_agent_scorecard(self, run_id: int, role: str, score: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO agent_scorecards(run_id,role,score_json,created_at) VALUES(?,?,?,?)",
                (run_id, role, json.dumps(score, ensure_ascii=False), utcnow()),
            )

    def save_maintenance_action(self, run_id: int, action: dict[str, Any],
                                status: str = "PROPOSED") -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO maintenance_actions("
                "run_id,action_id,owner_role,action_class,status,payload_json,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (run_id, action["action_id"], action["owner_role"], action["action_class"],
                 status, json.dumps(action, ensure_ascii=False), utcnow()),
            )

    def update_maintenance_action(self, run_id: int, action_id: str, *, status: str,
                                  artifact_path: Path | None = None,
                                  evolution_variant_id: str | None = None,
                                  rollback_manifest_path: Path | None = None) -> None:
        finished = utcnow() if status in {"SUCCEEDED", "FAILED", "REJECTED", "NEEDS_REVIEW"} else None
        with self.connect() as connection:
            connection.execute(
                "UPDATE maintenance_actions SET status=?,artifact_path=COALESCE(?,artifact_path),"
                "finished_at=? WHERE run_id=? AND action_id=?",
                (status, str(artifact_path) if artifact_path else None, finished, run_id, action_id),
            )
            if evolution_variant_id:
                cursor = connection.execute(
                    "UPDATE evolution_variants SET status=?,"
                    "rollback_manifest_path=COALESCE(?,rollback_manifest_path),updated_at=? "
                    "WHERE variant_id=? AND source_run_id=? AND action_id=?",
                    (status, str(rollback_manifest_path) if rollback_manifest_path else None,
                     utcnow(), evolution_variant_id, run_id, action_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError("evolution/action identity mismatch")

    def get_maintenance_action(self, run_id: int, action_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM maintenance_actions WHERE run_id=? AND action_id=?",
                (run_id, action_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown maintenance action: run={run_id} action={action_id}")
            result = dict(row)
            result["payload"] = json.loads(result.pop("payload_json"))
            return result

    def attach_operator_evidence_retry(self, run_id: int, action_id: str, *,
                                       edit_id: str, pool: str) -> dict[str, Any]:
        """Attach one exact, auditable manual-example retry permit.

        The permit is deliberately narrower than ``human_retry_history``: it
        names the committed edit and pool and can be consumed only once.  It
        authorizes the configured owner to reconsider its existing action; it
        does not authorize a label write or a medical routing decision.
        """
        edit_id = str(edit_id or "").strip()
        pool = str(pool or "").strip()
        if not edit_id or not pool:
            raise ValueError("edit_id and pool are required for operator evidence")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status,payload_json FROM maintenance_actions "
                "WHERE run_id=? AND action_id=?", (run_id, action_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown maintenance action: run={run_id} action={action_id}")
            if str(row["status"]) != "APPROVED":
                raise ValueError(
                    "operator evidence retry requires an APPROVED action, got "
                    f"{row['status']}"
                )
            payload = json.loads(row["payload_json"] or "{}")
            existing = payload.get("operator_evidence_retry")
            if isinstance(existing, dict):
                if (existing.get("edit_id") == edit_id
                        and existing.get("pool") == pool):
                    return dict(existing)
                raise ValueError("action already has a different operator evidence permit")
            marker = {
                "edit_id": edit_id,
                "pool": pool,
                "granted_at": utcnow(),
                "consumed": False,
            }
            payload["operator_evidence_retry"] = marker
            connection.execute(
                "UPDATE maintenance_actions SET payload_json=? "
                "WHERE run_id=? AND action_id=? AND status='APPROVED'",
                (json.dumps(payload, ensure_ascii=False), run_id, action_id),
            )
            return dict(marker)

    def consume_operator_evidence_retry(self, run_id: int, action_id: str) -> dict[str, Any]:
        """Atomically consume the action's single operator-evidence permit."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status,payload_json FROM maintenance_actions "
                "WHERE run_id=? AND action_id=?", (run_id, action_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown maintenance action: run={run_id} action={action_id}")
            if str(row["status"]) != "APPROVED":
                raise ValueError(
                    "operator evidence retry requires an APPROVED action, got "
                    f"{row['status']}"
                )
            payload = json.loads(row["payload_json"] or "{}")
            marker = payload.get("operator_evidence_retry")
            if not isinstance(marker, dict) or marker.get("consumed") is not False:
                raise ValueError("no unconsumed operator evidence retry")
            marker = dict(marker)
            marker["consumed"] = True
            marker["consumed_at"] = utcnow()
            payload["operator_evidence_retry"] = marker
            cursor = connection.execute(
                "UPDATE maintenance_actions SET payload_json=? "
                "WHERE run_id=? AND action_id=? AND status='APPROVED'",
                (json.dumps(payload, ensure_ascii=False), run_id, action_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("operator evidence retry could not be consumed")
            return dict(marker)

    def review_maintenance_action(self, run_id: int, action_id: str, *, decision: str,
                                  reviewer: str, note: str,
                                  confirmed: bool = False) -> dict[str, Any]:
        """Record a bounded human disposition without touching medical labels.

        ``retry_once`` is the only decision that changes an action payload: it
        explicitly grants one additional implementation attempt and returns the
        action to PROPOSED.  This avoids a visually successful retry button that
        would immediately be exhausted again by the autopilot.
        """
        decision = decision.strip().lower()
        reviewer = reviewer.strip()
        note = note.strip()
        allowed = {"defer", "retry_once", "resolve_external", "reject"}
        if decision not in allowed:
            raise ValueError(f"unsupported review decision: {decision}")
        if not reviewer:
            raise ValueError("reviewer is required")
        if len(note) < 3:
            raise ValueError("review note must contain at least 3 characters")
        if decision != "defer" and not confirmed:
            raise ValueError("confirmed=true is required for a state-changing decision")

        with self.connect() as connection:
            row = connection.execute(
                "SELECT status,payload_json FROM maintenance_actions "
                "WHERE run_id=? AND action_id=?",
                (run_id, action_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown maintenance action: run={run_id} action={action_id}")
            previous = str(row["status"])
            if previous != "HUMAN_REQUIRED":
                raise ValueError(
                    f"manual disposition requires HUMAN_REQUIRED, got {previous}"
                )

            resulting = previous
            payload_json = row["payload_json"]
            if decision == "retry_once":
                payload = json.loads(payload_json or "{}")
                payload["max_attempts"] = max(1, int(payload.get("max_attempts", 1))) + 1
                payload.setdefault("human_retry_history", []).append({
                    "reviewer": reviewer,
                    "note": note,
                    "created_at": utcnow(),
                })
                payload_json = json.dumps(payload, ensure_ascii=False)
                resulting = "APPROVED"
            elif decision == "resolve_external":
                resulting = "VALIDATED"
            elif decision == "reject":
                resulting = "REJECTED"

            reviewed_at = utcnow()
            if resulting != previous:
                finished = reviewed_at if resulting in {"VALIDATED", "REJECTED"} else None
                connection.execute(
                    "UPDATE maintenance_actions SET status=?,payload_json=?,finished_at=? "
                    "WHERE run_id=? AND action_id=? AND status=?",
                    (resulting, payload_json, finished, run_id, action_id, previous),
                )
            cursor = connection.execute(
                "INSERT INTO human_action_reviews("
                "run_id,action_id,decision,reviewer,note,previous_status,resulting_status,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (run_id, action_id, decision, reviewer, note, previous, resulting, reviewed_at),
            )
            return {
                "review_id": int(cursor.lastrowid),
                "run_id": run_id,
                "action_id": action_id,
                "decision": decision,
                "reviewer": reviewer,
                "note": note,
                "previous_status": previous,
                "resulting_status": resulting,
                "created_at": reviewed_at,
            }

    def list_human_action_reviews(self, run_id: int | None = None,
                                  action_id: str | None = None,
                                  limit: int = 200) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id=?")
            params.append(run_id)
        if action_id is not None:
            clauses.append("action_id=?")
            params.append(action_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM human_action_reviews" + where
                + " ORDER BY review_id DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def register_evolution_variant(
        self, *, variant_id: str, parent_variant_id: str | None,
        source_run_id: int | None, action_id: str | None,
        workflow_stage: str | None, code_commit: str | None,
        continuity_revision: str | None, artifact_path: Path | None,
        status: str = "CANDIDATE", metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist one immutable node in the self-improvement lineage.

        Re-registering the same variant is intentionally idempotent.  The
        candidate's identity and ancestry cannot be rewritten after creation.
        """
        timestamp = utcnow()
        encoded = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO evolution_variants("
                "variant_id,parent_variant_id,source_run_id,action_id,workflow_stage,"
                "code_commit,continuity_revision,artifact_path,status,metadata_json,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (variant_id, parent_variant_id, source_run_id, action_id,
                 workflow_stage, code_commit, continuity_revision,
                 str(artifact_path) if artifact_path else None, status, encoded,
                 timestamp, timestamp),
            )
            row = connection.execute(
                "SELECT parent_variant_id,source_run_id,action_id,workflow_stage,"
                "code_commit,continuity_revision,artifact_path,metadata_json "
                "FROM evolution_variants WHERE variant_id=?", (variant_id,),
            ).fetchone()
            expected = (
                parent_variant_id, source_run_id, action_id, workflow_stage,
                code_commit, continuity_revision,
                str(artifact_path) if artifact_path else None, encoded,
            )
            actual = tuple(row) if row is not None else None
            if actual != expected:
                raise ValueError(f"evolution variant identity collision: {variant_id}")

    def transition_evolution_variant(
        self, variant_id: str, *, status: str,
        rollback_manifest_path: Path | None = None,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE evolution_variants SET status=?,"
                "rollback_manifest_path=COALESCE(?,rollback_manifest_path),updated_at=? "
                "WHERE variant_id=?",
                (status,
                 str(rollback_manifest_path) if rollback_manifest_path else None,
                 utcnow(), variant_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown evolution variant: {variant_id}")

    def record_evolution_evaluation(
        self, *, variant_id: str, parent_variant_id: str | None,
        benchmark_id: str, benchmark_hash: str,
        baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any],
        constraints: dict[str, Any], decision: str, reasons: list[str],
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO evolution_evaluations("
                "variant_id,parent_variant_id,benchmark_id,benchmark_hash,"
                "baseline_metrics_json,candidate_metrics_json,constraints_json,"
                "decision,reasons_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (variant_id, parent_variant_id, benchmark_id, benchmark_hash,
                 json.dumps(baseline_metrics, ensure_ascii=False, sort_keys=True),
                 json.dumps(candidate_metrics, ensure_ascii=False, sort_keys=True),
                 json.dumps(constraints, ensure_ascii=False, sort_keys=True),
                 decision, json.dumps(reasons, ensure_ascii=False), utcnow()),
            )
            return int(cursor.lastrowid)

    def append_evolution_event(
        self, event_type: str, payload: dict[str, Any], *,
        variant_id: str | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO evolution_events(variant_id,event_type,payload_json,created_at) "
                "VALUES(?,?,?,?)",
                (variant_id, event_type,
                 json.dumps(payload, ensure_ascii=False, sort_keys=True), utcnow()),
            )
            return int(cursor.lastrowid)

    def list_evolution_variants(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evolution_variants ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            result.append(item)
        return result

    def evolution_evaluations(self, variant_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evolution_evaluations WHERE variant_id=? "
                "ORDER BY evaluation_id", (variant_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for source, target in (
                ("baseline_metrics_json", "baseline_metrics"),
                ("candidate_metrics_json", "candidate_metrics"),
                ("constraints_json", "constraints"),
                ("reasons_json", "reasons"),
            ):
                item[target] = json.loads(item.pop(source))
            result.append(item)
        return result

    def create_experiment(self, source_run_id: int, hypothesis: dict[str, Any]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO experiments(source_run_id,status,hypothesis_json,created_at) VALUES(?,?,?,?)",
                (source_run_id, "CREATED", json.dumps(hypothesis, ensure_ascii=False), utcnow()),
            )
            return int(cursor.lastrowid)

    def update_experiment(self, experiment_id: int, *, status: str,
                          worktree_path: Path | None = None,
                          baseline_metrics: dict[str, Any] | None = None,
                          candidate_metrics: dict[str, Any] | None = None,
                          decision: dict[str, Any] | None = None) -> None:
        finished = utcnow() if status in {"PROMOTED", "REJECTED", "NEEDS_REVIEW", "ERROR"} else None
        with self.connect() as connection:
            connection.execute(
                "UPDATE experiments SET status=?,worktree_path=COALESCE(?,worktree_path),"
                "baseline_metrics_json=COALESCE(?,baseline_metrics_json),"
                "candidate_metrics_json=COALESCE(?,candidate_metrics_json),"
                "decision_json=COALESCE(?,decision_json),finished_at=COALESCE(?,finished_at) "
                "WHERE experiment_id=?",
                (status, str(worktree_path) if worktree_path else None,
                 json.dumps(baseline_metrics) if baseline_metrics is not None else None,
                 json.dumps(candidate_metrics) if candidate_metrics is not None else None,
                 json.dumps(decision, ensure_ascii=False) if decision is not None else None,
                 finished, experiment_id),
            )

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown experiment: {experiment_id}")
            result = dict(row)
            result["hypothesis"] = json.loads(result.pop("hypothesis_json"))
            return result

    def run_details(self, run_id: int) -> dict[str, Any]:
        result = self.get_run(run_id)
        with self.connect() as connection:
            result["tasks"] = [dict(row) for row in connection.execute(
                "SELECT * FROM tasks WHERE run_id=? ORDER BY task_id", (run_id,)).fetchall()]
            result["artifacts"] = [dict(row) for row in connection.execute(
                "SELECT * FROM artifacts WHERE run_id=? ORDER BY artifact_id", (run_id,)).fetchall()]
            result["agent_outputs"] = [dict(row) for row in connection.execute(
                "SELECT role,provider,model,status,input_hash,output_json,error_message,created_at "
                "FROM agent_outputs WHERE run_id=? ORDER BY output_id", (run_id,)).fetchall()]
            result["agent_messages"] = [dict(row) for row in connection.execute(
                "SELECT message_id,thread_id,sender_role,recipient_role,message_type,"
                "payload_json,reply_to,created_at FROM agent_messages "
                "WHERE run_id=? ORDER BY message_id", (run_id,)).fetchall()]
            result["agent_invocations"] = [dict(row) for row in connection.execute(
                "SELECT invocation_id,thread_id,role,provider,requested_model,input_hash,"
                "input_json,status,started_at,finished_at,error_message "
                "FROM agent_invocations WHERE run_id=? ORDER BY started_at", (run_id,)).fetchall()]
            result["agent_invocation_events"] = [dict(row) for row in connection.execute(
                "SELECT e.* FROM agent_invocation_events e JOIN agent_invocations i "
                "ON i.invocation_id=e.invocation_id WHERE i.run_id=? ORDER BY e.event_id",
                (run_id,)).fetchall()]
            result["agent_scorecards"] = [dict(row) for row in connection.execute(
                "SELECT role,score_json,created_at FROM agent_scorecards "
                "WHERE run_id=? ORDER BY scorecard_id", (run_id,)).fetchall()]
            result["maintenance_actions"] = [dict(row) for row in connection.execute(
                "SELECT action_id,owner_role,action_class,status,payload_json,artifact_path,created_at,finished_at "
                "FROM maintenance_actions WHERE run_id=? ORDER BY action_row_id", (run_id,)).fetchall()]
        return result
