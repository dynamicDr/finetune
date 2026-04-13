"""
test_partition.py - 验证 train/test 脚本的数据划分是否一致

运行方式：
  python test_partition.py --seed 42 --train_ratio 0.8

这个脚本会模拟 train 和 test 脚本的划分逻辑，验证：
1. 同一个 seed + train_ratio 下，训练集和测试集是否一致
2. 训练集和测试集是否有重叠（应该没有）
3. 训练集 + 测试集是否等于全集
"""

import argparse
import random
from data_loaders.base import load_dataset


def should_include_sample(sample, task_filter: str) -> bool:
    """与 train/test 脚本一致"""
    return task_filter == "all" or (sample.get("options") is not None) == (task_filter == "mcq")


def split_indices(
    indices: list[int],
    seed: int,
    train_ratio: float,
    use_train_split: bool,
) -> list[int]:
    """与 train/test 脚本一致"""
    indices_copy = indices.copy()
    random.seed(seed)
    random.shuffle(indices_copy)
    
    split_point = int(len(indices_copy) * train_ratio)
    if use_train_split:
        return indices_copy[:split_point]
    else:
        return indices_copy[split_point:]


def get_split_indices(seed: int, train_ratio: float, task_filter: str, use_train_split: bool) -> list[int]:
    """模拟 train/test 脚本获取索引的完整流程"""
    dataset = load_dataset("nyu-visionx/VSI-Bench", split="test")
    
    filtered_indices = [
        i for i in range(len(dataset))
        if should_include_sample(dataset[i], task_filter)
    ]
    
    return split_indices(filtered_indices, seed, train_ratio, use_train_split)


def main():
    parser = argparse.ArgumentParser(description="验证数据划分一致性")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--task_filter", type=str, default="all", choices=["all", "mcq", "numeric"])
    args = parser.parse_args()

    print(f"参数: seed={args.seed}, train_ratio={args.train_ratio}, task_filter={args.task_filter}")
    print("=" * 60)

    # 模拟调用两次，验证结果一致
    print("\n[测试1] 多次调用划分函数，结果应完全一致")
    
    train_indices_1 = get_split_indices(args.seed, args.train_ratio, args.task_filter, use_train_split=True)
    train_indices_2 = get_split_indices(args.seed, args.train_ratio, args.task_filter, use_train_split=True)
    test_indices_1 = get_split_indices(args.seed, args.train_ratio, args.task_filter, use_train_split=False)
    test_indices_2 = get_split_indices(args.seed, args.train_ratio, args.task_filter, use_train_split=False)

    train_match = train_indices_1 == train_indices_2
    test_match = test_indices_1 == test_indices_2
    
    print(f"  训练集两次调用一致: {'✓ PASS' if train_match else '✗ FAIL'}")
    print(f"  测试集两次调用一致: {'✓ PASS' if test_match else '✗ FAIL'}")

    # 验证无重叠
    print("\n[测试2] 训练集和测试集应无重叠")
    
    train_set = set(train_indices_1)
    test_set = set(test_indices_1)
    overlap = train_set & test_set
    
    no_overlap = len(overlap) == 0
    print(f"  训练集大小: {len(train_set)}")
    print(f"  测试集大小: {len(test_set)}")
    print(f"  重叠样本数: {len(overlap)}")
    print(f"  无重叠: {'✓ PASS' if no_overlap else '✗ FAIL'}")
    
    if overlap:
        print(f"  重叠的索引（前10个）: {list(overlap)[:10]}")

    # 验证覆盖全集
    print("\n[测试3] 训练集 + 测试集应覆盖全部筛选后的样本")
    
    dataset = load_dataset("nyu-visionx/VSI-Bench", split="test")
    all_filtered = [
        i for i in range(len(dataset))
        if should_include_sample(dataset[i], args.task_filter)
    ]
    all_set = set(all_filtered)
    combined_set = train_set | test_set
    
    covers_all = combined_set == all_set
    print(f"  筛选后总样本数: {len(all_set)}")
    print(f"  训练+测试合并后: {len(combined_set)}")
    print(f"  完全覆盖: {'✓ PASS' if covers_all else '✗ FAIL'}")

    # 验证比例
    print("\n[测试4] 划分比例应接近设定值")
    
    actual_train_ratio = len(train_set) / len(all_set)
    ratio_diff = abs(actual_train_ratio - args.train_ratio)
    ratio_ok = ratio_diff < 0.01  # 允许1%误差（因为整数划分）
    
    print(f"  设定比例: {args.train_ratio}")
    print(f"  实际比例: {actual_train_ratio:.4f}")
    print(f"  比例误差: {ratio_diff:.4f}")
    print(f"  比例正确: {'✓ PASS' if ratio_ok else '✗ FAIL'}")

    # 打印样本示例
    print("\n[信息] 划分后的样本索引示例")
    print(f"  训练集前10个索引: {train_indices_1[:10]}")
    print(f"  测试集前10个索引: {test_indices_1[:10]}")

    # 总结
    print("\n" + "=" * 60)
    all_pass = train_match and test_match and no_overlap and covers_all and ratio_ok
    if all_pass:
        print("✓ 所有测试通过！数据划分逻辑正确。")
    else:
        print("✗ 存在测试失败，请检查划分逻辑。")

    # 额外：测试不同 seed 产生不同划分
    print("\n[测试5] 不同 seed 应产生不同划分")
    
    train_indices_other_seed = get_split_indices(args.seed + 1, args.train_ratio, args.task_filter, use_train_split=True)
    different_seed_different_split = train_indices_1 != train_indices_other_seed
    
    print(f"  seed={args.seed} 训练集前5: {train_indices_1[:5]}")
    print(f"  seed={args.seed + 1} 训练集前5: {train_indices_other_seed[:5]}")
    print(f"  不同seed产生不同划分: {'✓ PASS' if different_seed_different_split else '✗ FAIL'}")


if __name__ == "__main__":
    main()