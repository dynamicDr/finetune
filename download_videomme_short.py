#!/usr/bin/env python3
"""
download_videomme_short.py - 打包 / 下载 Video-MME short 子集

在服务器上直接打包（推荐）:

    python download_videomme_short.py --pack

默认在 ~/videomme_short_pack 生成合集，包含:
  - videos/              全部 short 视频
  - questions/           每个视频对应的问题与选项（.json + .txt）
  - short_video_questions.json
  - questions_readable.txt

从 Windows 本地拉取合集（无需 rsync，用 scp）:

    python download_videomme_short.py --pull-pack --local_dir ./videomme_short_pack
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REMOTE = "duanty@gpu3gate1:/userhome/cs3/duanty/dataset/Video-MME"
DEFAULT_REMOTE_HOST = "duanty@gpu3gate1"
DEFAULT_LOCAL_DIR = "~/dataset/Video-MME"
DEFAULT_VIDEO_DIR = "~/dataset/Video-MME"
DEFAULT_PACK_DIR = "~/videomme_short_pack"
DEFAULT_REMOTE_REPO = "/userhome/cs3/duanty/finetune"
REMOTE_SCRIPT_NAME = "download_videomme_short.py"
QUESTIONS_MAP_NAME = "short_video_questions.json"

def _expand_local(path: str) -> str:
    """展开本地路径（Windows/macOS/Linux 本机）。"""
    return os.path.abspath(os.path.expanduser(path))


def _normalize_posix_path(path: str) -> str:
    """保留远程 Linux POSIX 路径，避免在 Windows 上被 abspath 污染。"""
    return path.strip().replace("\\", "/").rstrip("/")


def _bash_single_quote(value: str) -> str:
    """生成适用于远程 bash 的单引号字符串。"""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _parse_remote(remote: str) -> tuple[str | None, str]:
    if ":" not in remote:
        return None, _expand_local(remote)
    host, remote_path = remote.rsplit(":", 1)
    host = host.strip() or None
    return host, _normalize_posix_path(remote_path)


def _load_short_samples(video_dir: str):
    from data_loaders import get_data_loader

    loader = get_data_loader(
        "videomme",
        video_dir=video_dir,
        seed=42,
        task_filter="short",
        no_dataset_config=True,
    )
    samples = loader.get_split_samples(
        split="test",
        use_train_split=False,
        sample_count=None,
    )
    if not samples:
        raise RuntimeError(f"未找到 short 样本，请检查 video_dir={video_dir}")
    return samples


def _video_key(sample) -> str:
    meta = sample.metadata or {}
    for key in ("videoID", "video_id"):
        value = str(meta.get(key, "")).strip()
        if value:
            return value
    return Path(sample.video_path).stem


def _build_questions_payload(samples, *, video_file_map: dict[str, str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for sample in samples:
        video_id = _video_key(sample)
        entry = grouped.setdefault(
            video_id,
            {
                "video_id": video_id,
                "video_file": video_file_map.get(video_id, f"videos/{video_id}.mp4"),
                "questions": [],
            },
        )
        meta = sample.metadata or {}
        entry["questions"].append(
            {
                "question_id": sample.sample_id,
                "question": sample.question,
                "options": sample.options or [],
                "answer": sample.answer,
                "domain": str(meta.get("domain", "")).strip(),
                "sub_category": str(meta.get("sub_category", "")).strip(),
                "task_type": str(meta.get("task_type", "")).strip(),
                "duration": str(meta.get("duration", "")).strip(),
            }
        )

    for entry in grouped.values():
        entry["questions"].sort(key=lambda item: item["question_id"])

    return {
        "dataset": "videomme",
        "task_filter": "short",
        "video_count": len(grouped),
        "question_count": len(samples),
        "videos": dict(sorted(grouped.items(), key=lambda item: item[0])),
    }


def _format_video_questions_text(entry: dict[str, Any]) -> str:
    lines = [
        f"Video ID: {entry['video_id']}",
        f"Video File: {entry['video_file']}",
        f"Question Count: {len(entry['questions'])}",
        "",
    ]
    for idx, q in enumerate(entry["questions"], start=1):
        lines.append(f"[Q{idx}] {q['question_id']}")
        lines.append(f"Question: {q['question']}")
        for opt in q["options"]:
            lines.append(f"  {opt}")
        lines.append(f"Answer: {q['answer']}")
        if q.get("domain"):
            lines.append(f"Domain: {q['domain']}")
        if q.get("sub_category"):
            lines.append(f"Sub Category: {q['sub_category']}")
        if q.get("task_type"):
            lines.append(f"Task Type: {q['task_type']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_short_pack(
    video_dir: str,
    output_dir: str,
    *,
    hardlink: bool = False,
    overwrite: bool = False,
) -> tuple[Path, dict[str, int]]:
    video_dir = _expand_local(video_dir)
    output_dir = Path(_expand_local(output_dir))
    videos_dir = output_dir / "videos"
    questions_dir = output_dir / "questions"

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    videos_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_short_samples(video_dir)
    video_src_map: dict[str, Path] = {}
    missing = 0

    for sample in samples:
        video_id = _video_key(sample)
        src = Path(sample.video_path)
        if not src.is_file():
            missing += 1
            continue
        video_src_map.setdefault(video_id, src)

    video_file_map: dict[str, str] = {}
    copied_videos = 0
    total_bytes = 0

    for video_id, src in sorted(video_src_map.items()):
        dst = videos_dir / f"{video_id}{src.suffix.lower()}"
        video_file_map[video_id] = f"videos/{dst.name}"
        if dst.exists():
            total_bytes += dst.stat().st_size
            continue
        if hardlink:
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
        copied_videos += 1
        total_bytes += dst.stat().st_size

    payload = _build_questions_payload(samples, video_file_map=video_file_map)
    readable_blocks: list[str] = []

    for video_id, entry in payload["videos"].items():
        per_video_json = questions_dir / f"{video_id}.json"
        per_video_txt = questions_dir / f"{video_id}.txt"
        per_video_json.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        per_video_txt.write_text(_format_video_questions_text(entry), encoding="utf-8")
        readable_blocks.append(_format_video_questions_text(entry))

    (output_dir / QUESTIONS_MAP_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "questions_readable.txt").write_text(
        "\n".join(block.rstrip() for block in readable_blocks) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                "Video-MME short pack",
                f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Source: {video_dir}",
                f"Videos: {payload['video_count']}",
                f"Questions: {payload['question_count']}",
                "",
                "Structure:",
                "  videos/                   short 视频文件",
                "  questions/                每个视频的问题与选项",
                "  short_video_questions.json  汇总 JSON",
                "  questions_readable.txt      汇总可读文本",
                "",
                "下载整个目录到本地示例:",
                "  Windows (PowerShell, 推荐):",
                f"    scp -r {DEFAULT_REMOTE_HOST}:{DEFAULT_PACK_DIR} ./videomme_short_pack",
                "  Linux/macOS:",
                f"    rsync -avh --progress {output_dir}/ ./videomme_short_pack/",
                f"    scp -r {output_dir} ./videomme_short_pack",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = {
        "questions": payload["question_count"],
        "videos": payload["video_count"],
        "files": 2 + len(video_file_map) * 2 + len(video_file_map),  # json+txt per video + readme + map + readable
        "missing_videos": missing,
        "total_bytes": total_bytes,
        "copied_videos": copied_videos,
    }
    return output_dir, stats


def _write_video_questions_map(video_dir: str, samples) -> str:
    base = Path(_expand_local(video_dir))
    output_path = base / "videomme" / QUESTIONS_MAP_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_file_map: dict[str, str] = {}
    for sample in samples:
        video_id = _video_key(sample)
        video_path = Path(sample.video_path)
        if video_path.is_file():
            try:
                rel = video_path.relative_to(base).as_posix()
            except ValueError:
                rel = video_path.name
            video_file_map[video_id] = rel

    payload = _build_questions_payload(samples, video_file_map=video_file_map)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return f"videomme/{QUESTIONS_MAP_NAME}"


def collect_short_relative_paths(video_dir: str) -> tuple[list[str], dict[str, int]]:
    if os.name == "nt" and video_dir.replace("\\", "/").startswith("/"):
        video_dir = _normalize_posix_path(video_dir)
    else:
        video_dir = _expand_local(video_dir)
    samples = _load_short_samples(video_dir)

    base = Path(video_dir)
    rel_paths: set[str] = set()
    missing = 0
    total_bytes = 0

    for sample in samples:
        video_path = Path(sample.video_path)
        if not video_path.is_file():
            missing += 1
            continue
        try:
            rel = video_path.relative_to(base).as_posix()
        except ValueError:
            rel = video_path.name
        rel_paths.add(rel)
        total_bytes += video_path.stat().st_size

    parquet_candidates = [
        base / "videomme" / "test-00000-of-00001.parquet",
        base / "test-00000-of-00001.parquet",
    ]
    for parquet_path in parquet_candidates:
        if parquet_path.is_file():
            rel_paths.add(parquet_path.relative_to(base).as_posix())

    rel_paths.add(_write_video_questions_map(video_dir, samples))

    video_count = sum(1 for p in rel_paths if p.endswith(".mp4"))
    stats = {
        "questions": len(samples),
        "videos": video_count,
        "files": len(rel_paths),
        "missing_videos": missing,
        "total_bytes": total_bytes,
    }
    return sorted(rel_paths), stats


def _remote_list_command(video_dir: str, remote_repo: str) -> str:
    remote_repo = _normalize_posix_path(remote_repo)
    video_dir = _normalize_posix_path(video_dir)
    remote_script = f"{remote_repo}/{REMOTE_SCRIPT_NAME}"
    parts = [
        "python3",
        remote_script,
        "--remote_manifest",
        "--video_dir",
        video_dir,
    ]
    return " ".join(_bash_single_quote(part) for part in parts)


def fetch_remote_file_list(
    remote_host: str,
    remote_video_dir: str,
    remote_repo: str,
) -> tuple[list[str], dict[str, int]]:
    remote_video_dir = _normalize_posix_path(remote_video_dir)
    remote_repo = _normalize_posix_path(remote_repo)
    cmd = ["ssh", remote_host, _remote_list_command(remote_video_dir, remote_repo)]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "SSH 远程命令失败。\n"
            f"命令: {' '.join(cmd)}\n"
            f"exit code: {exc.returncode}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    stats_line = next((ln for ln in lines if ln.startswith("__STATS__")), None)
    if stats_line is None:
        raise RuntimeError(f"远程列表生成失败，输出:\n{proc.stdout}\n{proc.stderr}")

    _, q, v, files, missing, total_bytes = stats_line.split()
    rel_paths = [ln for ln in lines if not ln.startswith("__STATS__")]
    stats = {
        "questions": int(q),
        "videos": int(v),
        "files": int(files),
        "missing_videos": int(missing),
        "total_bytes": int(total_bytes),
    }
    return rel_paths, stats


def run_rsync(
    *,
    remote: str,
    local_dir: str,
    rel_paths: list[str],
    dry_run: bool,
) -> None:
    local_dir = _expand_local(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    remote_host, remote_video_dir = _parse_remote(remote)
    if remote_host is None:
        source = remote_video_dir.rstrip("/") + "/"
        rsync_prefix: list[str] = ["rsync", "-avh", "--progress"]
    else:
        source = f"{remote_host}:{remote_video_dir.rstrip('/')}/"
        rsync_prefix = ["rsync", "-avh", "--progress", "-e", "ssh"]

    local_target = Path(local_dir).as_posix().rstrip("/") + "/"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, newline="\n") as tf:
        tf.write("\n".join(rel_paths))
        tf.write("\n")
        files_from = tf.name

    cmd = rsync_prefix + [
        "--files-from",
        files_from,
        source,
        local_target,
    ]
    if dry_run:
        cmd.insert(1, "--dry-run")

    print("执行:", " ".join(shlex.quote(x) for x in cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(files_from)


def _format_bytes(num: int) -> str:
    return f"{num / 1024 ** 3:.2f} GB"


def _command_exists(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def pull_pack_via_scp(
    *,
    remote_host: str,
    remote_pack: str,
    local_dir: str,
) -> Path:
    local_path = Path(_expand_local(local_dir))
    remote_pack = _normalize_posix_path(remote_pack)
    remote_name = Path(remote_pack).name
    parent = local_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    staging = parent / remote_name
    if staging.exists():
        print(f"删除已存在的目录: {staging}")
        shutil.rmtree(staging)

    remote_spec = f"{remote_host}:{remote_pack}"
    cmd = ["scp", "-r", remote_spec, str(parent)]
    print("执行:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)

    if staging != local_path:
        if local_path.exists():
            shutil.rmtree(local_path)
        staging.rename(local_path)

    return local_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="打包或下载 Video-MME short 子集",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="在服务器生成 short 合集目录（默认输出 ~/videomme_short_pack）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_PACK_DIR,
        help=f"合集输出目录 (默认: {DEFAULT_PACK_DIR})",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"Video-MME 数据根目录 (默认: {DEFAULT_VIDEO_DIR})",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="视频使用硬链接而非复制（省空间，但目录不可跨盘）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出目录已存在则先删除再重建",
    )
    parser.add_argument(
        "--pull-pack",
        action="store_true",
        help="用 scp 拉取服务器上的 short 合集（适合 Windows，无需 rsync）",
    )
    parser.add_argument(
        "--remote_host",
        type=str,
        default=DEFAULT_REMOTE_HOST,
        help=f"远程 SSH 地址 (默认: {DEFAULT_REMOTE_HOST})",
    )
    parser.add_argument(
        "--remote_pack",
        type=str,
        default=DEFAULT_PACK_DIR,
        help=f"服务器上合集目录 (默认: {DEFAULT_PACK_DIR})",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="从远程 rsync 拉取原始 Video-MME 子集（需 rsync）",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=DEFAULT_REMOTE,
        help=f"远程源，格式 user@host:/path/to/Video-MME (默认: {DEFAULT_REMOTE})",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=DEFAULT_LOCAL_DIR,
        help=f"本地保存目录，配合 --pull 使用 (默认: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--remote_repo",
        type=str,
        default=DEFAULT_REMOTE_REPO,
        help=f"服务器上 finetune 仓库路径 (默认: {DEFAULT_REMOTE_REPO})",
    )
    parser.add_argument(
        "--remote_manifest",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--print_files",
        action="store_true",
        help="仅打印相对路径列表（不执行 rsync）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="rsync 预览模式，不实际传输",
    )
    parser.add_argument(
        "--write_questions_only",
        action="store_true",
        help="仅生成 videomme/short_video_questions.json，不执行 rsync",
    )
    parser.add_argument(
        "--skip_rsync",
        action="store_true",
        help="只生成/打印文件列表，不执行 rsync",
    )
    args = parser.parse_args()

    remote_host, remote_video_dir = _parse_remote(args.remote)
    local_dir = _expand_local(args.local_dir)
    remote_repo = _normalize_posix_path(args.remote_repo)
    video_dir = _expand_local(args.video_dir)

    if args.remote_manifest:
        manifest_video_dir = _normalize_posix_path(args.video_dir or remote_video_dir)
        rel_paths, stats = collect_short_relative_paths(manifest_video_dir)
        print(
            "__STATS__",
            stats["questions"],
            stats["videos"],
            stats["files"],
            stats["missing_videos"],
            stats["total_bytes"],
        )
        for rel in rel_paths:
            print(rel)
        return

    if args.pull_pack:
        print("=" * 60)
        print("Video-MME short 合集下载 (scp)")
        print("=" * 60)
        local_target = (
            args.local_dir
            if args.local_dir != DEFAULT_LOCAL_DIR
            else "./videomme_short_pack"
        )
        print(f"remote_host: {args.remote_host}")
        print(f"remote_pack: {args.remote_pack}")
        print(f"local_dir: {_expand_local(local_target)}")
        if not _command_exists("scp"):
            raise RuntimeError("未找到 scp 命令。请在 Windows 设置中启用 OpenSSH 客户端。")
        pack_dir = pull_pack_via_scp(
            remote_host=args.remote_host,
            remote_pack=args.remote_pack,
            local_dir=local_target,
        )
        print("\n" + "=" * 60)
        print("✓ 下载完成")
        print("=" * 60)
        print(f"本地目录: {pack_dir}")
        return

    if not args.pull:
        print("=" * 60)
        print("Video-MME short 合集打包")
        print("=" * 60)
        print(f"video_dir: {video_dir}")
        print(f"output_dir: {_expand_local(args.output_dir)}")
        pack_dir, stats = build_short_pack(
            video_dir=video_dir,
            output_dir=args.output_dir,
            hardlink=args.hardlink,
            overwrite=args.overwrite,
        )
        print(
            f"short 问题数: {stats['questions']}, "
            f"视频数: {stats['videos']}, "
            f"新复制视频: {stats['copied_videos']}, "
            f"缺失视频: {stats['missing_videos']}, "
            f"视频总大小: {_format_bytes(stats['total_bytes'])}"
        )
        if stats["missing_videos"]:
            print("警告: 部分 short 视频在源目录中不存在。")
        print("\n" + "=" * 60)
        print("✓ 合集已生成")
        print("=" * 60)
        print(f"目录: {pack_dir}")
        print("结构:")
        print(f"  {pack_dir}/")
        print("  ├── videos/")
        print("  ├── questions/")
        print(f"  ├── {QUESTIONS_MAP_NAME}")
        print("  ├── questions_readable.txt")
        print("  └── README.txt")
        print("\n下载到本地示例:")
        print(f"  scp -r {DEFAULT_REMOTE_HOST}:{DEFAULT_PACK_DIR} ./videomme_short_pack")
        print("  python download_videomme_short.py --pull-pack --local_dir ./videomme_short_pack")
        return

    print("=" * 60)
    print("Video-MME short 远程拉取")
    print("=" * 60)
    print(f"remote: {args.remote}")
    print(f"local_dir: {local_dir}")

    use_local_list = (
        args.print_files
        or args.skip_rsync
        or args.write_questions_only
    )

    if use_local_list:
        print(f"video_dir: {video_dir}")
        rel_paths, stats = collect_short_relative_paths(video_dir)
    elif remote_host:
        print(f"remote_repo: {remote_repo}")
        rel_paths, stats = fetch_remote_file_list(
            remote_host,
            remote_video_dir,
            remote_repo,
        )
    else:
        print(f"video_dir: {video_dir}")
        rel_paths, stats = collect_short_relative_paths(video_dir)

    print(
        f"short 问题数: {stats['questions']}, "
        f"视频数: {stats['videos']}, "
        f"待同步文件数: {stats['files']}, "
        f"缺失视频: {stats['missing_videos']}, "
        f"视频总大小(不含 parquet): {_format_bytes(stats['total_bytes'])}"
    )

    if stats["missing_videos"]:
        print("警告: 部分 short 视频在源目录中不存在，请检查服务器数据是否完整。")

    if args.write_questions_only:
        rel_path = _write_video_questions_map(video_dir, _load_short_samples(video_dir))
        print(f"已生成: {Path(video_dir) / rel_path}")
        return

    if args.print_files or args.skip_rsync:
        for rel in rel_paths:
            print(rel)
        return

    if not rel_paths:
        raise RuntimeError("文件列表为空，终止同步。")

    run_rsync(
        remote=args.remote,
        local_dir=local_dir,
        rel_paths=rel_paths,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("✓ 同步完成")
    print("=" * 60)
    print(f"本地目录: {local_dir}")
    print("目录结构示例:")
    print(f"  {local_dir}/")
    print(f"  ├── videomme/test-00000-of-00001.parquet")
    print(f"  ├── videomme/{QUESTIONS_MAP_NAME}")
    print("  └── videos/data/*.mp4")
    print("\n本地评测示例:")
    print(
        "  python vqa_eval_ours.py "
        "--dataset videomme --no_dataset_config --task_filter short "
        f"--video_dir {local_dir}"
    )


if __name__ == "__main__":
    main()
