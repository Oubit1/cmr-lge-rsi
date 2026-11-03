#!/usr/bin/env bash
set -euo pipefail

ROOT="${CMR_LGE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${CMR_LGE_PYTHON:-$ROOT/.venv/bin/python}"
SECRETS=${CMR_LGE_SECRETS:-$HOME/.config/cmr-lge-workflow/secrets.env}
DATASET=${CMR_LGE_DATASET:-$ROOT/data/dataset}

cd "$ROOT"
if pgrep -f "scripts/autopilot.py --dataset $DATASET" >/dev/null 2>&1; then
  echo "autopilot already running: $(pgrep -f "scripts/autopilot.py --dataset $DATASET" | tr '\n' ' ')"
  # Temporary failure: systemd should retry and take ownership after an
  # externally launched controller exits.  This preserves the single-writer
  # lock without leaving the persistent service silently inactive.
  exit 75
fi
if [ ! -f "$SECRETS" ]; then
  echo "missing secrets file: $SECRETS" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
. "$SECRETS"
set +a
export PYTHONUNBUFFERED=1
exec "$PY" -u scripts/autopilot.py \
  --dataset "$DATASET" \
  --providers configs/providers.yaml \
  --agents configs/agents.aliyun-roles.yaml \
  --interval 30 --max-stagnant 3 --action-timeout 240
