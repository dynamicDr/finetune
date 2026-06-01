"""
download_models.py

用途：
- 批量下载并缓存指定的模型列表到 $HOME/.cache/huggingface/hub
- 之后其它脚本直接用 from_pretrained("模型名") 即可，无需重复下载
"""

from huggingface_hub import snapshot_download


MODEL_LIST = [
    # "Qwen/Qwen3-VL-30B-A3B-Instruct",
    # "Qwen/Qwen3-VL-8B-Instruct",
    # "Qwen/Qwen3.5-27B",
    # "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    # "Qwen/Qwen3-VL-8B-Thinking",
    # "Qwen/Qwen3-VL-30B-A3B-Thinking",
    # "Qwen/Qwen3-VL-32B-Thinking",
]

def download_one(model_id: str) -> None:
    print(f"\n[{MODEL_LIST.index(model_id) + 1}/{len(MODEL_LIST)}] 下载模型: {model_id}")
    # 仅做仓库快照下载，不在下载阶段实例化模型，避免触发本地环境依赖冲突。
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_files_only=False,
    )
    print(f"✓ 完成: {model_id}")


def main() -> None:
    print(f"批量下载 {len(MODEL_LIST)} 个模型到缓存目录...")
    print(f"缓存路径: $HOME/.cache/huggingface/hub\n")
    
    for model_id in MODEL_LIST:
        download_one(model_id)
    
    print("\n✓ 所有模型已下载并缓存完成。")


if __name__ == "__main__":
    main()