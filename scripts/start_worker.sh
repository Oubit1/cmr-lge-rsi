#!/usr/bin/env bash
# Start the orchestrator worker with the AliCloud role profile.
# Safe to re-run: refuses to start a second worker.
set -euo pipefail

ROOT="${CMR_LGE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SECRETS=${CMR_LGE_SECRETS:-$HOME/.config/cmr-lge-workflow/secrets.env}
PY="${CMR_LGE_PYTHON:-$ROOT/.venv/bin/python}"

cd "$ROOT"

if pgrep -f "cli.py worker" >/dev/null 2>&1; then
  echo "worker already running: $(pgrep -f 'cli.py worker' | tr '\n' ' ')"
  exit 0
fi

if [ ! -f "$SECRETS" ]; then
  echo "missing secrets file: $SECRETS" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
set -a
# shellcheck disable=SC1090
. "$SECRETS"
set +a

nohup "$PY" -u cli.py worker \
  --auto-analyze --auto-maintain \
  --providers configs/providers.yaml \
  --agents configs/agents.aliyun-roles.yaml \
  >> artifacts/auto_worker.log 2>&1 &

echo "worker started pid=$!"
