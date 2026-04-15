"""
download_egoschema.py - 仅下载 EgoSchema 数据集到本地缓存
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# 允许从 download/ 子目录直接运行脚本时导入项目内模块
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.base import load_dataset


def download_dataset(
    dataset_name: str,
    split: str,
    cache_dir: str,
    dataset_config: str | None,
    no_dataset_config: bool,
):
    kwargs: dict[str, Any] = {"cache_dir": cache_dir}
    if dataset_config and not no_dataset_config:
        kwargs["name"] = dataset_config
    return load_dataset(dataset_name, split=split, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="仅下载 EgoSchema 数据集到本地缓存")
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/EgoSchema")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--dataset_config", type=str, default="Subset")
    parser.add_argument("--no_dataset_config", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./egoschema_cache")
    args = parser.parse_args()

    cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("EgoSchema 数据下载工具（仅下载数据集）")
    print("=" * 60)
    print(f"dataset_name: {args.dataset_name}")
    print(f"dataset_split: {args.dataset_split}")
    print(f"dataset_config: {args.dataset_config if not args.no_dataset_config else '<disabled>'}")
    print(f"cache_dir: {cache_dir}")

    dataset_split = download_dataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=cache_dir,
        dataset_config=args.dataset_config,
        no_dataset_config=args.no_dataset_config,
    )
    print(f"\n✓ 下载完成: {args.dataset_name} [{args.dataset_split}]")
    print(f"样本数: {len(dataset_split)}")
    if len(dataset_split) > 0:
        print(f"字段: {list(dataset_split.features.keys())}")


if __name__ == "__main__":
    main()
