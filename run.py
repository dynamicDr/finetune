import logging
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


log = logging.getLogger(__name__)


def resolve_model_path(base_path: str) -> str:
    base = Path(base_path)
    snapshots = base / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"找不到 snapshots 目录: {snapshots}")
    snapshot = next(snapshots.iterdir(), None)
    if snapshot is None:
        raise FileNotFoundError(f"snapshots 目录为空: {snapshots}")
    return str(snapshot)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log.info(f"当前配置:\n{OmegaConf.to_yaml(cfg)}")

    model_path = resolve_model_path(cfg.model.path)

    cmd = [
        "python",
        cfg.script,
        "--model_path",
        model_path,
        "--video_dir",
        cfg.video_dir,
        "--num_samples",
        str(cfg.num_samples),
        "--num_frames",
        str(cfg.num_frames),
        "--seed",
        str(cfg.seed),
        "--train_ratio",
        str(cfg.train_ratio),
        "--task_filter",
        cfg.task_filter,
        "--log_file",
        "results.csv",
    ]

    log.info(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"执行失败，返回码: {result.returncode}")


if __name__ == "__main__":
    main()

