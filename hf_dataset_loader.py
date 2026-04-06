"""Robust HuggingFace datasets loader.

Avoids importing the local `datasets/` folder in this repo by temporarily
removing repo root from `sys.path` before importing HuggingFace `datasets`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def load_dataset(*args, **kwargs):
    repo_root = str(Path(__file__).resolve().parent)
    removed = False
    if repo_root in sys.path:
        sys.path.remove(repo_root)
        removed = True
    try:
        from datasets import load_dataset as hf_load_dataset
    finally:
        if removed:
            sys.path.insert(0, repo_root)
    return hf_load_dataset(*args, **kwargs)

