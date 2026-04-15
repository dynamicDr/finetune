"""
download_visbench.py - 下载并准备 VSI-Bench 数据集
用法:
    python download/download_visbench.py
    python download/download_visbench.py --cache_dir ./my_cache
    python download/download_visbench.py --num_samples 100  # 只验证前100个样本
"""

import os
import argparse
import zipfile
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

# 允许从 download/ 子目录直接运行脚本时导入项目内模块
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.base import load_dataset


def download_dataset_metadata(cache_dir: str):
    """下载数据集元数据 (parquet 文件)"""
    print("=" * 60)
    print("步骤 1/3: 下载数据集元数据...")
    print("=" * 60)

    dataset = load_dataset(
        "nyu-visionx/VSI-Bench",
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    print(f"数据集加载成功!")
    print(f"数据集结构: {dataset}")

    # 显示数据集信息
    for split in dataset:
        print(f"  - {split}: {len(dataset[split])} 个样本")
        if len(dataset[split]) > 0:
            
            print(f"    字段: {list(dataset[split].features.keys())}")

    return dataset


def download_videos(cache_dir: str, tmp_dir: str):
    """下载视频 zip 文件"""
    from huggingface_hub import hf_hub_download

    print("\n" + "=" * 60)
    print("步骤 2/3: 下载视频文件...")
    print("=" * 60)

    os.makedirs(tmp_dir, exist_ok=True)

    zip_files = [
        "arkitscenes.zip",
        "scannet.zip",
        "scannetpp.zip"
    ]

    downloaded_zips = []

    for zip_file in zip_files:
        print(f"\n正在下载 {zip_file}...")
        try:
            zip_path = hf_hub_download(
                repo_id="nyu-visionx/VSI-Bench",
                filename=zip_file,
                repo_type="dataset",
                cache_dir=cache_dir,
                resume_download=True,  # 支持断点续传
            )
            # 复制到临时目录，后续统一从 tmp 解压并在最后删除 tmp
            local_zip_path = os.path.join(tmp_dir, zip_file)
            if not os.path.exists(local_zip_path):
                shutil.copy2(zip_path, local_zip_path)
            downloaded_zips.append((zip_file, local_zip_path))
            print(f"✓ 下载完成: {zip_file}")

            # 显示文件大小
            size_gb = os.path.getsize(zip_path) / (1024 ** 3)
            print(f"  文件大小: {size_gb:.2f} GB")

        except Exception as e:
            print(f"✗ 下载失败 {zip_file}: {e}")
            continue

    return downloaded_zips


def extract_videos(downloaded_zips: list, video_dir: str):
    """从 zip 中提取所有 mp4 到目标目录（扁平化存放）"""
    print("\n" + "=" * 60)
    print("步骤 3/3: 解压视频文件...")
    print("=" * 60)
    os.makedirs(video_dir, exist_ok=True)

    for zip_file, zip_path in downloaded_zips:
        print(f"\n正在解压 {zip_file}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = [f for f in zip_ref.namelist() if f.lower().endswith(".mp4")]
                print(f"  共 {len(file_list)} 个 mp4 文件")

                extracted = 0
                skipped = 0
                for file in tqdm(file_list, desc=f"  提取 {zip_file}"):
                    target_name = os.path.basename(file)
                    if not target_name:
                        continue
                    target_path = os.path.join(video_dir, target_name)
                    if os.path.exists(target_path):
                        skipped += 1
                        continue
                    with zip_ref.open(file) as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted += 1

            print(f"✓ 解压完成: {zip_file}")
            print(f"  新增: {extracted}，跳过(已存在): {skipped}")

        except Exception as e:
            print(f"✗ 解压失败 {zip_file}: {e}")
            continue


def verify_dataset(dataset, video_dir: str, num_samples: int = None):
    """验证数据集完整性"""
    print("\n" + "=" * 60)
    print("验证数据集完整性...")
    print("=" * 60)

    # 统计视频文件（扁平目录）
    if os.path.exists(video_dir):
        files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
        total_videos = len(files)
    else:
        total_videos = 0

    print(f"\n视频文件统计:")
    print(f"  目录: {video_dir}")
    print(f"  总计: {total_videos} 个 mp4 视频")

    # 验证样本的视频路径
    if "test" in dataset:
        test_data = dataset["test"]
        samples_to_check = num_samples if num_samples else len(test_data)
        samples_to_check = min(samples_to_check, len(test_data))

        print(f"\n检查 {samples_to_check} 个样本的视频文件...")

        found = 0
        missing = 0
        missing_examples = []

        for i in tqdm(range(samples_to_check), desc="  验证中"):
            sample = test_data[i]

            # 尝试获取视频路径字段
            video_path = None
            for field in ["video_path", "video", "video_id", "scene_id"]:
                if field in sample and sample[field]:
                    video_path = sample[field]
                    break

            if video_path is None:
                continue

            # 查找视频文件
            video_found = False

            # 直接路径
            full_path = os.path.join(video_dir, video_path)
            if os.path.exists(full_path):
                video_found = True
            else:
                # 尝试添加扩展名或搜索子目录
                video_name = os.path.basename(video_path)
                if not video_name.endswith(('.mp4', '.avi', '.mov')):
                    video_name += ".mp4"

                candidate = os.path.join(video_dir, video_name)
                if os.path.exists(candidate):
                    video_found = True

            if video_found:
                found += 1
            else:
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(video_path)

        print(f"\n验证结果:")
        print(f"  ✓ 找到视频: {found}")
        print(f"  ✗ 缺失视频: {missing}")

        if missing_examples:
            print(f"\n缺失视频示例 (前5个):")
            for ex in missing_examples:
                print(f"    - {ex}")


def print_sample_info(dataset):
    """打印样本信息示例"""
    print("\n" + "=" * 60)
    print("样本信息示例:")
    print("=" * 60)

    if "test" in dataset and len(dataset["test"]) > 0:
        sample = dataset["test"][0]
        print("\n第一个样本的字段:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")


def create_config_file(cache_dir: str, video_dir: str):
    """创建配置文件，方便后续使用"""
    config_path = os.path.join(cache_dir, "vsibench_config.txt")

    with open(config_path, "w") as f:
        f.write(f"cache_dir={cache_dir}\n")
        f.write(f"video_dir={video_dir}\n")

    print(f"\n配置文件已保存: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="下载并准备 VSI-Bench 数据集")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./vsi_bench_cache",
        help="缓存目录路径 (默认: ./vsi_bench_cache)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="验证的样本数量 (默认: 100, 设为 -1 验证全部)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="跳过下载，只进行验证"
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    video_dir = os.path.expanduser("~/dataset/vsi_bench")
    tmp_dir = os.path.join(cache_dir, "tmp_download")

    print("\n" + "=" * 60)
    print("VSI-Bench 数据集下载工具")
    print("=" * 60)
    print(f"缓存目录: {cache_dir}")
    print(f"视频目录: {video_dir}")
    print(f"临时目录: {tmp_dir}")

    # 创建目录
    os.makedirs(cache_dir, exist_ok=True)

    if not args.skip_download:
        # 步骤 1: 下载元数据
        dataset = download_dataset_metadata(cache_dir)

        # 步骤 2: 下载视频到 tmp
        downloaded_zips = download_videos(cache_dir, tmp_dir)

        # 步骤 3: 解压视频
        if downloaded_zips:
            extract_videos(downloaded_zips, video_dir)
            # 清理临时下载目录
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                print(f"\n已清理临时目录: {tmp_dir}")
    else:
        print("\n跳过下载，加载现有数据集...")
        dataset = load_dataset(
            "nyu-visionx/VSI-Bench",
            cache_dir=cache_dir,
            trust_remote_code=True
        )

    # 验证数据集
    num_samples = None if args.num_samples == -1 else args.num_samples
    verify_dataset(dataset, video_dir, num_samples)

    # 打印样本信息
    print_sample_info(dataset)

    # 创建配置文件
    create_config_file(cache_dir, video_dir)

    # 完成
    print("\n" + "=" * 60)
    print("✓ 数据集准备完成!")
    print("=" * 60)
    print(f"\n后续使用方法:")
    print(f"  from data_loaders.base import load_dataset")
    print(f"  dataset = load_dataset('nyu-visionx/VSI-Bench', cache_dir='{cache_dir}')")
    print(f"  视频文件路径: {video_dir}")
    print("\n")


if __name__ == "__main__":
    main()