from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_root_main():
    root = Path(__file__).resolve().parents[1]
    src = root / "train_vsibench.py"
    spec = importlib.util.spec_from_file_location("root_train_vsibench", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    _load_root_main()()
