"""
download_mlvu_test.py - 下载并准备 MLVU Test 集（含 ground truth 答案）

数据集: https://huggingface.co/datasets/MLVU/MLVU_Test
许可: CC-BY-NC-SA-4.0（仅限学术研究，需先在 HF 页面同意条款）

用法:
    # 下载题目 + 答案标注（约 0.6MB，推荐）
    python download/download_mlvu_test.py

    # 同时下载视频（约 75GB，分 8 个 tar 分卷）
    python download/download_mlvu_test.py --include_videos

    # 下载视频并自动合并解压
    python download/download_mlvu_test.py --include_videos --extract_videos

    # 指定保存目录
    python download/download_mlvu_test.py --cache_dir ~/dataset/mlvu_test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_ID = "MLVU/MLVU_Test"

# 题目文件（不含答案）
QUESTION_FILES = [
    "test_multi_choice_tasks.json",
    "test_generation_tasks.json",
    "MLVU_Test/test_question.json",
]

# Ground truth（含答案）
GROUND_TRUTH_FILES = [
    "test-ground-truth/test_mcq_gt.json",
    "test-ground-truth/test_ssc_gt.json",
    "test-ground-truth/test_vs_gt.json",
]

VIDEO_PARTS = [f"MLVU_Test/test_video.tar.gz.part-{suffix}" for suffix in "abcdefgh"]


class PctTqdm(tqdm):
    """每完成 1% 打印一行，nohup 日志里也能看到进度。"""

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
    """检查 HuggingFace 登录状态，gated 数据集需要有效 token。"""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()
        print(f"✓ 已登录 HuggingFace: {user.get('name', user)}", flush=True)
    except Exception:
        print("✗ 未检测到 HuggingFace 登录状态。", flush=True)
        print("  请先执行: hf auth login", flush=True)
        print("  (旧版 CLI 可用: huggingface-cli login)", flush=True)
        print(f"  并在页面同意许可: https://huggingface.co/datasets/{REPO_ID}", flush=True)
        sys.exit(1)


def build_allow_patterns(include_videos: bool) -> list[str]:
    patterns = [f"{p}" for p in QUESTION_FILES + GROUND_TRUTH_FILES]
    if include_videos:
        patterns.append("MLVU_Test/test_video.tar.gz.part-*")
    return patterns


def download_mlvu_test(cache_dir: str, include_videos: bool = False) -> str:
    from huggingface_hub import snapshot_download

    allow_patterns = build_allow_patterns(include_videos)

    print("=" * 60, flush=True)
    print("MLVU Test 集下载", flush=True)
    print("=" * 60, flush=True)
    print(f"repo_id: {REPO_ID}", flush=True)
    print(f"cache_dir: {cache_dir}", flush=True)
    if include_videos:
        print("模式: 题目 + 答案 + 视频（约 75GB）", flush=True)
    else:
        print("模式: 题目 + 答案标注（约 0.6MB）", flush=True)
    print(f"allow_patterns: {allow_patterns}", flush=True)

    # max_workers=1：经代理下载大文件时，并行容易断连后僵死（CLOSE-WAIT）
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
    """按 1% 打印复制进度。"""
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


def extract_videos(cache_dir: str) -> None:
    """合并 tar 分卷并解压到 MLVU_Test/video/。"""
    mlvu_test_dir = Path(cache_dir) / "MLVU_Test"
    parts = [mlvu_test_dir / Path(p).name for p in VIDEO_PARTS]
    missing = [p for p in parts if not p.exists()]
    if missing:
        print("✗ 视频分卷不完整，无法解压:", flush=True)
        for p in missing:
            print(f"    - {p.name}", flush=True)
        sys.exit(1)

    combined = mlvu_test_dir / "test_video.tar.gz"
    video_dir = mlvu_test_dir / "video"

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
    print(f"解压到 {video_dir}（可能需数十分钟）...", flush=True)
    subprocess.run(
        ["tar", "-xzf", str(combined), "-C", str(video_dir)],
        check=True,
    )
    mp4_count = len(list(video_dir.rglob("*.mp4")))
    print(f"✓ 解压完成: {mp4_count} 个 mp4", flush=True)


def verify_download(cache_dir: str, include_videos: bool) -> None:
    import json

    print("\n" + "=" * 60, flush=True)
    print("验证下载结果...", flush=True)
    print("=" * 60, flush=True)

    root = Path(cache_dir)

    print("\n题目文件:", flush=True)
    for rel in QUESTION_FILES:
        path = root / rel
        status = "✓" if path.exists() else "✗"
        extra = ""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                extra = f" ({len(data)} 条)" if isinstance(data, list) else ""
            except json.JSONDecodeError:
                extra = " (JSON 解析失败)"
        print(f"  {status} {rel}{extra}", flush=True)

    print("\nGround truth（答案）:", flush=True)
    for rel in GROUND_TRUTH_FILES:
        path = root / rel
        status = "✓" if path.exists() else "✗"
        extra = ""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    extra = f" ({len(data)} 条)"
                elif isinstance(data, dict):
                    extra = f" ({len(data)} 条)"
            except json.JSONDecodeError:
                extra = " (JSON 解析失败)"
        print(f"  {status} {rel}{extra}", flush=True)

    if include_videos:
        mlvu_test_dir = root / "MLVU_Test"
        parts_found = sum(1 for p in VIDEO_PARTS if (mlvu_test_dir / Path(p).name).exists())
        print(f"\n视频分卷: {parts_found}/{len(VIDEO_PARTS)}", flush=True)
        video_dir = mlvu_test_dir / "video"
        if video_dir.exists():
            mp4_count = len(list(video_dir.rglob("*.mp4")))
            print(f"解压后视频: {mp4_count} 个 mp4", flush=True)
        else:
            print("解压后视频: (未解压，可加 --extract_videos)", flush=True)


def create_config_file(cache_dir: str) -> None:
    root = Path(cache_dir)
    config_path = root / "mlvu_test_config.txt"

    with open(config_path, "w") as f:
        f.write(f"cache_dir={cache_dir}\n")
        f.write(f"mcq_questions={root / 'test_multi_choice_tasks.json'}\n")
        f.write(f"generation_questions={root / 'test_generation_tasks.json'}\n")
        f.write(f"test_question={root / 'MLVU_Test' / 'test_question.json'}\n")
        f.write(f"mcq_ground_truth={root / 'test-ground-truth' / 'test_mcq_gt.json'}\n")
        f.write(f"ssc_ground_truth={root / 'test-ground-truth' / 'test_ssc_gt.json'}\n")
        f.write(f"vs_ground_truth={root / 'test-ground-truth' / 'test_vs_gt.json'}\n")
        f.write(f"video_dir={root / 'MLVU_Test' / 'video'}\n")

    print(f"\n配置文件已保存: {config_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 MLVU Test 集（含 ground truth 答案）")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/dataset/mlvu_test",
        help="数据集保存目录 (默认: ~/dataset/mlvu_test)",
    )
    parser.add_argument(
        "--include_videos",
        action="store_true",
        help="同时下载视频分卷（约 75GB）",
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
    print("MLVU Test 集下载工具", flush=True)
    print("=" * 60, flush=True)
    print(f"保存目录: {cache_dir}", flush=True)
    print(f"数据集页面: https://huggingface.co/datasets/{REPO_ID}", flush=True)
    print("注意: 该数据集为 gated dataset，需登录 HF 并在页面同意学术许可条款。", flush=True)

    if not args.skip_auth_check:
        check_hf_auth()

    local_dir = download_mlvu_test(
        cache_dir=cache_dir,
        include_videos=args.include_videos,
    )

    if args.extract_videos:
        extract_videos(local_dir)

    verify_download(local_dir, include_videos=args.include_videos)
    create_config_file(local_dir)

    print("\n" + "=" * 60, flush=True)
    print("✓ 下载完成!", flush=True)
    print("=" * 60, flush=True)
    print(f"\n目录结构:", flush=True)
    print(f"  {local_dir}/", flush=True)
    print(f"  ├── test_multi_choice_tasks.json      # 多选题（6 选项，无答案）", flush=True)
    print(f"  ├── test_generation_tasks.json        # 生成式任务题目", flush=True)
    print(f"  ├── test-ground-truth/", flush=True)
    print(f"  │   ├── test_mcq_gt.json              # 多选题答案", flush=True)
    print(f"  │   ├── test_ssc_gt.json              # 子场景描述答案", flush=True)
    print(f"  │   └── test_vs_gt.json               # 视频摘要答案", flush=True)
    print(f"  └── MLVU_Test/", flush=True)
    print(f"      ├── test_question.json            # 评测用题目汇总", flush=True)
    if args.include_videos:
        print(f"      ├── test_video.tar.gz.part-*      # 视频分卷", flush=True)
        if args.extract_videos:
            print(f"      └── video/                        # 解压后的 mp4", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
