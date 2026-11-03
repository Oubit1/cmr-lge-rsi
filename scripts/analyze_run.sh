#!/usr/bin/env bash
# Run the multi-agent shadow analysis for one run that is already QC_FAILED.
set -euo pipefail

ROOT="${CMR_LGE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SECRETS=${CMR_LGE_SECRETS:-$HOME/.config/cmr-lge-workflow/secrets.env}
PY="${CMR_LGE_PYTHON:-$ROOT/.venv/bin/python}"
RUN_ID=${1:?usage: analyze_run.sh <run_id>}

cd "$ROOT"
set -a
# shellcheck disable=SC1090
. "$SECRETS"
set +a

"$PY" -u cli.py analyze \
  --run-id "$RUN_ID" \
  --providers configs/providers.yaml \
  --agents configs/agents.aliyun-roles.yaml
