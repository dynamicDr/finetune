"""
download_mlvu.py - 下载并准备 MLVU/MVLU 数据集

数据集: https://huggingface.co/datasets/MLVU/MVLU
许可: CC-BY-NC-SA-4.0（仅限学术研究，需先在 HF 页面同意条款）

用法:
    # 下载全部（约 430GB，需先 hf auth login 并同意许可）
    python download/download_mlvu.py

    # 仅下载标注 JSON（约 1MB）
    python download/download_mlvu.py --json_only

    # 指定保存目录
    python download/download_mlvu.py --cache_dir ~/dataset/mlvu

    # 只下载部分视频任务目录
    python download/download_mlvu.py --video_categories 1_plotQA 2_needle
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from download.hf_log_progress import LogFileTqdm, log_print, setup_line_buffered_stdout

REPO_ID = "MLVU/MVLU"

JSON_FILES = [
    "MLVU/json/1_plotQA.json",
    "MLVU/json/2_needle.json",
    "MLVU/json/3_ego.json",
    "MLVU/json/4_count.json",
    "MLVU/json/5_order.json",
    "MLVU/json/6_anomaly_reco.json",
    "MLVU/json/7_topic_reasoning.json",
    "MLVU/json/8_sub_scene.json",
    "MLVU/json/9_summary.json",
]

VIDEO_CATEGORIES = [
    "1_plotQA",
    "2_needle",
    "3_ego",
    "4_count",
    "5_order",
    "6_anomaly_reco",
    "7_topic_reasoning",
    "8_sub_scene",
    "9_summary",
]


def check_hf_auth() -> None:
    """检查 HuggingFace 登录状态，gated 数据集需要有效 token。"""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()
        log_print(f"✓ 已登录 HuggingFace: {user.get('name', user)}")
    except Exception:
        log_print("✗ 未检测到 HuggingFace 登录状态。")
        log_print("  请先执行: hf auth login")
        log_print("  (旧版 CLI 可用: huggingface-cli login)")
        log_print(f"  并在页面同意许可: https://huggingface.co/datasets/{REPO_ID}")
        sys.exit(1)


def build_allow_patterns(
    json_only: bool,
    video_categories: list[str] | None,
) -> list[str] | None:
    if json_only:
        return ["MLVU/json/*"]

    patterns = ["MLVU/json/*"]
    if video_categories:
        for cat in video_categories:
            patterns.append(f"MLVU/video/{cat}/*")
        return patterns

    return None


def download_mlvu(
    cache_dir: str,
    json_only: bool = False,
    video_categories: list[str] | None = None,
) -> str:
    from huggingface_hub import snapshot_download

    allow_patterns = build_allow_patterns(json_only, video_categories)

    log_print("=" * 60)
    log_print("MLVU/MVLU 数据集下载")
    log_print("=" * 60)
    log_print(f"repo_id: {REPO_ID}")
    log_print(f"cache_dir: {cache_dir}")
    if json_only:
        log_print("模式: 仅下载 JSON 标注")
    elif video_categories:
        log_print(f"模式: JSON + 指定视频目录 ({', '.join(video_categories)})")
    else:
        log_print("模式: 完整下载（JSON + 全部视频，约 430GB）")
    if allow_patterns:
        log_print(f"allow_patterns: {allow_patterns}")
    log_print("开始从 HuggingFace 拉取文件（进度将定期写入日志）...")

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns=allow_patterns,
        tqdm_class=LogFileTqdm,
    )
    log_print(f"HF 下载阶段完成: {local_dir}")
    return local_dir


def verify_download(cache_dir: str) -> None:
    log_print("\n" + "=" * 60)
    log_print("验证下载结果...")
    log_print("=" * 60)

    json_dir = Path(cache_dir) / "MLVU" / "json"
    video_dir = Path(cache_dir) / "MLVU" / "video"

    json_found = 0
    if json_dir.exists():
        json_found = len(list(json_dir.glob("*.json")))
    log_print(f"\nJSON 标注: {json_found}/{len(JSON_FILES)}")
    if json_found < len(JSON_FILES):
        missing = [p for p in JSON_FILES if not (json_dir / Path(p).name).exists()]
        if missing:
            log_print("  缺失:")
            for p in missing:
                log_print(f"    - {p}")

    if video_dir.exists():
        total_mp4 = 0
        log_print("\n视频目录:")
        for cat in VIDEO_CATEGORIES:
            cat_dir = video_dir / cat
            if not cat_dir.exists():
                log_print(f"  - {cat}: (未下载)")
                continue
            mp4_count = len(list(cat_dir.rglob("*.mp4")))
            total_mp4 += mp4_count
            log_print(f"  - {cat}: {mp4_count} 个 mp4")
        log_print(f"\n视频总计: {total_mp4} 个 mp4")
    else:
        log_print("\n视频目录: (未下载)")


def create_config_file(cache_dir: str) -> None:
    config_path = Path(cache_dir) / "mlvu_config.txt"
    json_dir = Path(cache_dir) / "MLVU" / "json"
    video_dir = Path(cache_dir) / "MLVU" / "video"

    with open(config_path, "w") as f:
        f.write(f"cache_dir={cache_dir}\n")
        f.write(f"json_dir={json_dir}\n")
        f.write(f"video_dir={video_dir}\n")

    log_print(f"\n配置文件已保存: {config_path}")


def main() -> None:
    setup_line_buffered_stdout()
    parser = argparse.ArgumentParser(description="下载并准备 MLVU/MVLU 数据集")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/dataset/mlvu",
        help="数据集保存目录 (默认: ~/dataset/mlvu)",
    )
    parser.add_argument(
        "--json_only",
        action="store_true",
        help="仅下载 JSON 标注文件（约 1MB）",
    )
    parser.add_argument(
        "--video_categories",
        nargs="+",
        choices=VIDEO_CATEGORIES,
        default=None,
        help="仅下载指定视频任务目录（仍会下载全部 JSON）",
    )
    parser.add_argument(
        "--skip_auth_check",
        action="store_true",
        help="跳过 HuggingFace 登录检查",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    log_print("\n" + "=" * 60)
    log_print("MLVU/MVLU 数据集下载工具")
    log_print("=" * 60)
    log_print(f"保存目录: {cache_dir}")
    log_print(f"数据集页面: https://huggingface.co/datasets/{REPO_ID}")
    log_print("注意: 该数据集为 gated dataset，需登录 HF 并在页面同意学术许可条款。")

    if not args.skip_auth_check:
        check_hf_auth()

    local_dir = download_mlvu(
        cache_dir=cache_dir,
        json_only=args.json_only,
        video_categories=args.video_categories,
    )

    verify_download(local_dir)
    create_config_file(local_dir)

    log_print("\n" + "=" * 60)
    log_print("✓ 下载完成!")
    log_print("=" * 60)
    log_print(f"\n目录结构:")
    log_print(f"  {local_dir}/")
    log_print(f"  ├── MLVU/json/     # 9 个任务标注 JSON")
    log_print(f"  └── MLVU/video/    # 9 个任务视频目录")
    log_print("")


if __name__ == "__main__":
    main()
