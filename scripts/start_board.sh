#!/usr/bin/env bash
# 看板需要 DASHSCOPE_API_KEY 才能做中文翻译，所以统一从这里启动
set -euo pipefail
ROOT="${CMR_LGE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SECRETS=${CMR_LGE_SECRETS:-$HOME/.config/cmr-lge-workflow/secrets.env}
PY="${CMR_LGE_PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
if [ -f "$SECRETS" ]; then set -a; . "$SECRETS"; set +a; fi
export PYTHONUNBUFFERED=1 TMPDIR="${TMPDIR:-/tmp}"
exec "$PY" -u scripts/board_server.py --port "${BOARD_PORT:-8765}"
