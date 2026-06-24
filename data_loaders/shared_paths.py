from __future__ import annotations

import os
from pathlib import Path

# 多用户共享资源根目录；可通过环境变量覆盖。
_SHARED_ROOT = Path(os.environ.get("FINETUNE_SHARED_ROOT", "/userhome/cs3/duanty"))

SHARED_DATASET_ROOT = Path(
    os.environ.get("FINETUNE_SHARED_DATASET_ROOT", str(_SHARED_ROOT / "dataset"))
)
SHARED_HF_DATASETS_CACHE = Path(
    os.environ.get(
        "FINETUNE_SHARED_HF_DATASETS_CACHE",
        str(_SHARED_ROOT / ".cache" / "huggingface" / "datasets"),
    )
)
