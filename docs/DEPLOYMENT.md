# Deployment

Use Linux / WSL with Python 3.10–3.12 and Git. The dashboard and control services
must run under an account that owns its installation and authorized local data.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/create_demo.py
python cli.py init
export CMR_LGE_DATASET="$PWD/data/dataset"
export CMR_LGE_PYTHON="$PWD/.venv/bin/python"
python scripts/board_server.py --port 8765
```

The demo contains generated geometric images only. It does not initialize a live
model, clinical calibration or production approval. Do not infer a working paid
autonomous service merely from the dashboard's availability.

## Real local installation

- Point `CMR_LGE_DATASET` and both local execution JSON files at the authorized
  dataset. Keep these local changes uncommitted. Configure your directory layout
  using the image/label target paths and manifests expected by the QC tools.
- Copy the reviewed-pair YAML example and build calibration and holdouts using
  the scoring/RSI commands. Do not reuse absent private benchmark hashes. Use
  `python scripts/rsi_control.py --help` for the benchmark creation interface.
- Export a stable private `CMR_LGE_CASE_SALT`; configure provider credentials and
  model names through `configs/providers.yaml` and the role YAML examples.
- Keep API keys in an ignored `secrets.env`, optionally selected by
  `CMR_LGE_SECRETS`. The supplied scripts use `CMR_LGE_PROJECT_ROOT` and
  `CMR_LGE_PYTHON`; shell defaults use the repository's `.venv`.
- The files under `deploy/` are templates. Replace `/srv/rsi/project` and service
  executable paths with your install paths. Set the same dataset/Python/secret
  environment for the board, edit agent, check loop and autopilot. Install them
  as user services; do not require root label-executor identity separation unless
  you choose that optional architecture.
- Set `enabled` explicitly in the local execution configurations only after
  successful local verification. UI Start/Pause coordinates installed services;
  it does not install missing services, provide API credit or create gold data.

Never use `--allow-label-writes` on a dataset until its independent gates,
calibration, backups and permissions have been validated. The standalone label
executor is an optional older deployment boundary; the audited board-local
manual apply path is also present.

The repository does not include original neural-network inference implementations
or checkpoints. Supply these under the Skill's expected `vendor/` and `models/`
paths when authorized, or use existing local image/label pairs for QC-only work.
