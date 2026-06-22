"""Centralized configuration: constants and filesystem paths.

Paths that the original code anchored to a module file (the Iconclass data and
ChromaDB store) are anchored here to the project root so they keep resolving to
the same `data/` directory regardless of where individual modules now live.
Paths that the original code resolved relative to the current working directory
(model weights, eval data, outputs) are kept as plain relative strings so they
keep resolving against the cwd exactly as before.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_DB_PATH = str(DATA_DIR / "iconclass_db")
DEFAULT_EMBEDDINGS_PATH = str(DATA_DIR / "iconclass_embeddings.jsonl")
DEFAULT_CLEAN_TXT_PATH = str(DATA_DIR / "iconclass_clean.txt")

DEFAULT_TRAINED_MODEL = "models/yolo26n.pt"
DEFAULT_GEMMA_MODEL = "gemma4:12b"

EMBEDDING_MODEL = "mxbai-embed-large"
BATCH_SIZE = 128

EVAL_DIR = "eval_data"
COCO_DIR = "datasets/coco"
OUTPUT_DIR = "output"
