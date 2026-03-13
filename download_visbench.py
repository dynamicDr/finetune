"""
download_vsibench.py - 下载并准备 VSI-Bench 数据集
用法:
    python download_vsibench.py
    python download_vsibench.py --cache_dir ./my_cache
    python download_vsibench.py --num_samples 100  # 只验证前100个样本
"""

import os
import argparse
import zipfile
from tqdm import tqdm


def download_dataset_metadata(cache_dir: str):
    """下载数据集元数据 (parquet 文件)"""
    from datasets import load_dataset

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


def download_videos(cache_dir: str):
    """下载视频 zip 文件"""
    from huggingface_hub import hf_hub_download

    print("\n" + "=" * 60)
    print("步骤 2/3: 下载视频文件...")
    print("=" * 60)

    video_dir = os.path.join(cache_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

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
            downloaded_zips.append((zip_file, zip_path))
            print(f"✓ 下载完成: {zip_file}")

            # 显示文件大小
            size_gb = os.path.getsize(zip_path) / (1024 ** 3)
            print(f"  文件大小: {size_gb:.2f} GB")

        except Exception as e:
            print(f"✗ 下载失败 {zip_file}: {e}")
            continue

    return downloaded_zips, video_dir


def extract_videos(downloaded_zips: list, video_dir: str):
    """解压视频文件"""
    print("\n" + "=" * 60)
    print("步骤 3/3: 解压视频文件...")
    print("=" * 60)

    for zip_file, zip_path in downloaded_zips:
        # 检查是否已解压
        folder_name = zip_file.replace(".zip", "")
        extract_path = os.path.join(video_dir, folder_name)

        if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
            print(f"✓ {folder_name} 已解压，跳过 ({len(os.listdir(extract_path))} 个文件)")
            continue

        print(f"\n正在解压 {zip_file}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 获取文件列表
                file_list = zip_ref.namelist()
                print(f"  共 {len(file_list)} 个文件")

                # 带进度条解压
                for file in tqdm(file_list, desc=f"  解压 {folder_name}"):
                    zip_ref.extract(file, video_dir)

            print(f"✓ 解压完成: {zip_file}")

        except Exception as e:
            print(f"✗ 解压失败 {zip_file}: {e}")
            continue


def verify_dataset(dataset, video_dir: str, num_samples: int = None):
    """验证数据集完整性"""
    print("\n" + "=" * 60)
    print("验证数据集完整性...")
    print("=" * 60)

    # 统计视频文件
    total_videos = 0
    video_stats = {}

    for subdir in ["arkitscenes", "scannet", "scannetpp"]:
        subdir_path = os.path.join(video_dir, subdir)
        if os.path.exists(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            video_stats[subdir] = len(files)
            total_videos += len(files)
        else:
            video_stats[subdir] = 0

    print(f"\n视频文件统计:")
    for subdir, count in video_stats.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {subdir}: {count} 个视频")
    print(f"  总计: {total_videos} 个视频")

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

                for subdir in ["arkitscenes", "scannet", "scannetpp"]:
                    candidate = os.path.join(video_dir, subdir, video_name)
                    if os.path.exists(candidate):
                        video_found = True
                        break

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
    video_dir = os.path.join(cache_dir, "videos")

    print("\n" + "=" * 60)
    print("VSI-Bench 数据集下载工具")
    print("=" * 60)
    print(f"缓存目录: {cache_dir}")
    print(f"视频目录: {video_dir}")

    # 创建目录
    os.makedirs(cache_dir, exist_ok=True)

    if not args.skip_download:
        # 步骤 1: 下载元数据
        dataset = download_dataset_metadata(cache_dir)

        # 步骤 2: 下载视频
        downloaded_zips, video_dir = download_videos(cache_dir)

        # 步骤 3: 解压视频
        if downloaded_zips:
            extract_videos(downloaded_zips, video_dir)
    else:
        print("\n跳过下载，加载现有数据集...")
        from datasets import load_dataset
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
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('nyu-visionx/VSI-Bench', cache_dir='{cache_dir}')")
    print(f"  视频文件路径: {video_dir}")
    print("\n")


if __name__ == "__main__":
    main()