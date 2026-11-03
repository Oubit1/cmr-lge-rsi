#!/usr/bin/env bash
set -euo pipefail
ROOT="${CMR_LGE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SECRETS=${CMR_LGE_SECRETS:-$HOME/.config/cmr-lge-workflow/secrets.env}
PY="${CMR_LGE_PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
mkdir -p runtime/control
exec 9>runtime/control/edit-agent.lock
if ! flock -n 9; then
  echo "another managed edit agent owns runtime/control/edit-agent.lock"
  exit 0
fi
[ -f "$SECRETS" ] && { set -a; . "$SECRETS"; set +a; }
export PYTHONUNBUFFERED=1 TMPDIR="${TMPDIR:-/tmp}"
exec "$PY" -u scripts/analyze_manual_edit_queue.py --watch \
  --poll-seconds "${CMR_LGE_EDIT_POLL_SECONDS:-1}" \
  --providers configs/providers.yaml --agents configs/agents.aliyun-roles.yaml
