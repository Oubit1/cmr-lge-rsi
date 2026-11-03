#!/usr/bin/env bash
# One idempotent tick of the orchestration loop. Safe to run from cron every 5 min.
#   - keeps exactly one worker alive
#   - runs shadow analysis for QC_FAILED runs that have not been analysed yet
#   - prints runs waiting at EXPERIMENT_PROPOSED (needs an agent/CLI to execute)
set -euo pipefail

ROOT="${CMR_LGE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${CMR_LGE_PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"

"$ROOT/scripts/start_worker.sh" >/dev/null 2>&1 || true

needs_analysis=$("$PY" - <<'PY'
import sqlite3
db = sqlite3.connect("runtime/workflow.sqlite")
rows = db.execute("""
    select r.run_id from workflow_runs r
    where r.status = 'QC_FAILED'
      and not exists (select 1 from agent_outputs a where a.run_id = r.run_id and a.role = 'supervisor')
    order by r.run_id
""").fetchall()
print(" ".join(str(row[0]) for row in rows))
PY
)

for run_id in $needs_analysis; do
  echo "analyzing run $run_id"
  "$ROOT/scripts/analyze_run.sh" "$run_id" || true
done

"$PY" - <<'PY'
import sqlite3
db = sqlite3.connect("runtime/workflow.sqlite")
rows = db.execute("""
    select run_id, current_step from workflow_runs where status = 'EXPERIMENT_PROPOSED'
""").fetchall()
for run_id, step in rows:
    actions = db.execute(
        "select action_id, status from maintenance_actions where run_id = ?", (run_id,)
    ).fetchall()
    print(f"EXPERIMENT_PROPOSED run={run_id} step={step} actions={actions}")
PY
