"""
download_longvideobench.py - 下载并准备 LongVideoBench validation 集

数据集: https://huggingface.co/datasets/longvideobench/LongVideoBench
许可: CC-BY-NC-SA 4.0（仅限学术研究，需先在 HF 页面同意条款）

默认仅下载 validation 标注（约 1.34k 条）与字幕；视频可选。

用法:
    # 下载 validation 标注 + 字幕
    python download/download_longvideobench.py

    # 同时下载视频分卷
    python download/download_longvideobench.py --include_videos

    # 下载视频并自动合并解压
    python download/download_longvideobench.py --include_videos --extract_videos

    # 指定保存目录
    python download/download_longvideobench.py --cache_dir ~/dataset/LongVideoBench
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_ID = "longvideobench/LongVideoBench"

ANNOTATION_FILES = [
    "lvb_val.json",
    "validation-00000-of-00001.parquet",
]

SUBTITLE_FILES = [
    "subtitles.tar",
]

VIDEO_PARTS = [f"videos.tar.part.{suffix}" for suffix in "abcdefghijklmnopqrstuvwxyz"]


class PctTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self._last_pct = -1

    def refresh(self, nolock=False, lock_args=None):
        pass

    def update(self, n=1):
        result = super().update(n)
        if self.total and self.total > 0:
            pct = min(100, int(100 * self.n / self.total))
            while self._last_pct < pct:
                self._last_pct += 1
                label = self.desc or "进度"
                print(f"{label}: {self._last_pct}%", flush=True)
        return result

    def close(self):
        if self.total and self.total > 0 and self._last_pct < 100:
            label = self.desc or "进度"
            print(f"{label}: 100%", flush=True)
        self.disable = True
        super().close()


def check_hf_auth() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()
        print(f"✓ 已登录 HuggingFace: {user.get('name', user)}", flush=True)
    except Exception:
        print("✗ 未检测到 HuggingFace 登录状态。", flush=True)
        print("  请先执行: hf auth login", flush=True)
        print(f"  并在页面同意许可: https://huggingface.co/datasets/{REPO_ID}", flush=True)
        sys.exit(1)


def build_allow_patterns(include_videos: bool) -> list[str]:
    patterns = list(ANNOTATION_FILES + SUBTITLE_FILES)
    if include_videos:
        patterns.append("videos.tar.part.*")
    return patterns


def download_longvideobench(cache_dir: str, include_videos: bool = False) -> str:
    from huggingface_hub import snapshot_download

    allow_patterns = build_allow_patterns(include_videos)
    print("=" * 60, flush=True)
    print("LongVideoBench validation 下载", flush=True)
    print("=" * 60, flush=True)
    print(f"repo_id: {REPO_ID}", flush=True)
    print(f"cache_dir: {cache_dir}", flush=True)
    print(f"allow_patterns: {allow_patterns}", flush=True)

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns=allow_patterns,
        tqdm_class=PctTqdm,
        max_workers=1,
    )
    return local_dir


def _copy_with_pct(src: Path, dst, *, label: str) -> None:
    total = src.stat().st_size
    written = 0
    last_pct = -1
    with open(src, "rb") as in_f, open(dst, "ab") as out_f:
        while chunk := in_f.read(64 * 1024 * 1024):
            out_f.write(chunk)
            written += len(chunk)
            if total > 0:
                pct = min(100, int(100 * written / total))
                while last_pct < pct:
                    last_pct += 1
                    print(f"{label}: {last_pct}%", flush=True)


def extract_subtitles(cache_dir: str) -> None:
    root = Path(cache_dir)
    tar_path = root / "subtitles.tar"
    if not tar_path.is_file():
        print(f"跳过字幕解压: 未找到 {tar_path}", flush=True)
        return
    subtitle_dir = root / "subtitles"
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    print(f"解压字幕到 {subtitle_dir} ...", flush=True)
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(subtitle_dir)], check=True)
    json_count = len(list(subtitle_dir.rglob("*.json")))
    print(f"✓ 字幕解压完成: {json_count} 个 json", flush=True)


def extract_videos(cache_dir: str) -> None:
    root = Path(cache_dir)
    parts = [root / name for name in VIDEO_PARTS if (root / name).is_file()]
    if not parts:
        print("✗ 未找到 videos.tar.part.* 分卷，无法解压视频。", flush=True)
        sys.exit(1)

    combined = root / "videos.tar"
    video_dir = root / "videos"
    print("\n" + "=" * 60, flush=True)
    print("合并并解压视频...", flush=True)
    print("=" * 60, flush=True)

    if not combined.exists():
        print(f"合并分卷 -> {combined}", flush=True)
        combined.write_bytes(b"")
        for i, part in enumerate(parts, start=1):
            print(f"合并分卷 [{i}/{len(parts)}]: {part.name}", flush=True)
            _copy_with_pct(part, combined, label=f"合并 {part.name}")
    else:
        print(f"已存在合并文件，跳过合并: {combined}", flush=True)

    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"解压到 {video_dir}（可能需较长时间）...", flush=True)
    subprocess.run(["tar", "-xf", str(combined), "-C", str(video_dir)], check=True)
    mp4_count = len(list(video_dir.rglob("*.mp4")))
    print(f"✓ 解压完成: {mp4_count} 个 mp4", flush=True)


def verify_download(cache_dir: str, include_videos: bool) -> None:
    print("\n" + "=" * 60, flush=True)
    print("验证下载结果...", flush=True)
    print("=" * 60, flush=True)

    root = Path(cache_dir)
    for rel in ANNOTATION_FILES:
        path = root / rel
        status = "✓" if path.is_file() else "✗"
        extra = ""
        if path.suffix == ".json" and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                extra = f" ({len(data)} 条)" if isinstance(data, list) else ""
            except json.JSONDecodeError:
                extra = " (JSON 解析失败)"
        print(f"  {status} {rel}{extra}", flush=True)

    subtitle_dir = root / "subtitles"
    subtitle_count = len(list(subtitle_dir.rglob("*.json"))) if subtitle_dir.is_dir() else 0
    print(f"  {'✓' if subtitle_count else '✗'} subtitles/*.json ({subtitle_count} 个)", flush=True)

    if include_videos:
        video_dir = root / "videos"
        mp4_count = len(list(video_dir.rglob("*.mp4"))) if video_dir.is_dir() else 0
        parts_found = sum(1 for name in VIDEO_PARTS if (root / name).is_file())
        print(f"  视频分卷: {parts_found} 个", flush=True)
        print(f"  {'✓' if mp4_count else '✗'} videos/*.mp4 ({mp4_count} 个)", flush=True)


def create_config_file(cache_dir: str) -> None:
    root = Path(cache_dir)
    config_path = root / "longvideobench_config.txt"
    config_path.write_text(
        "\n".join(
            [
                f"cache_dir={cache_dir}",
                f"annotations={root / 'lvb_val.json'}",
                f"parquet={root / 'validation-00000-of-00001.parquet'}",
                f"video_dir={root / 'videos'}",
                f"subtitle_dir={root / 'subtitles'}",
                "split=validation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n配置文件已保存: {config_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 LongVideoBench validation 集")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/dataset/LongVideoBench",
        help="数据集保存目录 (默认: ~/dataset/LongVideoBench)",
    )
    parser.add_argument(
        "--include_videos",
        action="store_true",
        help="同时下载视频分卷",
    )
    parser.add_argument(
        "--extract_videos",
        action="store_true",
        help="下载后合并 tar 分卷并解压视频（需配合 --include_videos）",
    )
    parser.add_argument(
        "--skip_auth_check",
        action="store_true",
        help="跳过 HuggingFace 登录检查",
    )
    args = parser.parse_args()

    if args.extract_videos and not args.include_videos:
        parser.error("--extract_videos 需要同时指定 --include_videos")

    cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    print("\n" + "=" * 60, flush=True)
    print("LongVideoBench validation 下载工具", flush=True)
    print("=" * 60, flush=True)
    print(f"保存目录: {cache_dir}", flush=True)
    print(f"数据集页面: https://huggingface.co/datasets/{REPO_ID}", flush=True)
    print("注意: 该数据集为 gated dataset，需登录 HF 并在页面同意学术许可条款。", flush=True)

    if not args.skip_auth_check:
        check_hf_auth()

    local_dir = download_longvideobench(cache_dir=cache_dir, include_videos=args.include_videos)
    extract_subtitles(local_dir)
    if args.extract_videos:
        extract_videos(local_dir)

    verify_download(local_dir, include_videos=args.include_videos)
    create_config_file(local_dir)

    print("\n" + "=" * 60, flush=True)
    print("✓ 下载完成!", flush=True)
    print("=" * 60, flush=True)
    print("评测示例:", flush=True)
    print(
        "  python vqa_eval.py --dataset lvb --num_samples 10 --num_frames 16 --use_subtitles",
        flush=True,
    )
    print(
        "  python vqa_eval_ours.py --dataset lvb --num_samples 10 --num_frames 16 --use_subtitles",
        flush=True,
    )


if __name__ == "__main__":
    main()
