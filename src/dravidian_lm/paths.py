from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("DRAVIDIAN_LM_BASE")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = ROOT / "artifacts"
TOKENIZERS_DIR = ARTIFACTS_DIR / "tokenizers"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
