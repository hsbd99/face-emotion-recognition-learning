"""Project-wide paths and label definitions."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("EMOTION_DATA_DIR", PROJECT_ROOT / "dataset")).expanduser()
OUTPUT_DIR = Path(os.getenv("EMOTION_OUTPUT_DIR", PROJECT_ROOT / "exp_result")).expanduser()
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUALIZATION_DIR = OUTPUT_DIR / "visualization"

FER2013_RAW_DIR = DATA_DIR / "FER2013"
FER2013_DIR = DATA_DIR / "FER2013_aligned"
CKPLUS_DIR = DATA_DIR / "CK+"

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_TO_INDEX = {name: index for index, name in enumerate(EMOTIONS)}
INDEX_TO_EMOTION = {index: name for name, index in EMOTION_TO_INDEX.items()}

# CK+ uses different directory and label names. "contempt" is mapped to the
# closest available FER2013 class for compatibility with the 7-class model.
CKPLUS_ALIASES = {
    "anger": "angry",
    "contempt": "disgust",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

DEFAULT_ARCH = "mobilenet_v3_small"
DEFAULT_IMAGE_SIZE = 96
DEFAULT_MODEL_PATH = CHECKPOINT_DIR / "emotion_best.pth"
