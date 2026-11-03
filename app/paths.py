"""Portable installation paths; dataset values stay local to the operator."""
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("CMR_LGE_DATASET", str(PROJECT_ROOT / "data/dataset"))).expanduser().resolve()
SKILL_ROOT = Path(os.environ.get("CMR_LGE_SKILL_ROOT", str(PROJECT_ROOT / "skills/cmr-lge-label-qc-workflow"))).expanduser().resolve()
PYTHON = Path(os.environ.get("CMR_LGE_PYTHON", sys.executable)).expanduser().resolve()
