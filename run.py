import logging
import subprocess
from pathlib import Path
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
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


def build_batch_timestamp() -> str:
    try:
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode.name == "MULTIRUN" and hydra_cfg.sweep.dir:
            name = Path(hydra_cfg.sweep.dir).name
            dt = datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
            return dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        pass
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log.info(f"当前配置:\n{OmegaConf.to_yaml(cfg)}")

    name = str(cfg.script)
    root = Path(get_original_cwd())
    sp = root / name if (root / name).is_file() else Path(__file__).resolve().parent / name
    if not sp.is_file():
        raise FileNotFoundError(name)

    shared = [
        "--video_dir",
        cfg.video_dir,
        "--num_frames",
        str(cfg.num_frames),
        "--seed",
        str(cfg.seed),
        "--train_ratio",
        str(cfg.train_ratio),
        "--task_filter",
        cfg.task_filter,
    ]
    model_snapshot = resolve_model_path(cfg.model.path)

    if "train_vsibench" in name:
        if OmegaConf.select(cfg, "lora.enabled", default=False):
            raise ValueError("训练时不要设置 lora.enabled=true")
        out = OmegaConf.select(cfg, "train.output_dir", default=None)
        if not out or str(out).strip() in ("null", "~", ""):
            out = f"outputs/vsibench_train/{cfg.model.name}/{cfg.task_filter}_frames{cfg.num_frames}"
        cmd = ["python", str(sp), "--model_path", model_snapshot, "--output_dir", str(out), *shared]
        ms = OmegaConf.select(cfg, "train.max_samples", default=None)
        if ms is not None and str(ms).strip() not in ("", "null", "None"):
            cmd += ["--max_samples", str(int(ms))]
    else:
        cmd = [
            "python",
            str(sp),
            *shared,
            "--num_samples",
            str(cfg.num_samples),
            "--log_file",
            f"result_{build_batch_timestamp()}.csv",
        ]
        if OmegaConf.select(cfg, "lora.enabled", default=False):
            ap = str(OmegaConf.select(cfg, "lora.adapter_path", default="") or "").strip()
            if not ap:
                raise ValueError("lora.enabled=true 需要 lora.adapter_path")
            bm = OmegaConf.select(cfg, "lora.base_model", default=None)
            bm = str(bm).strip() if bm and str(bm) not in ("null", "") else str(cfg.model.name)
            cmd += ["--use_lora", "--model_path", str(Path(ap).expanduser()), "--base_model", bm]
            if OmegaConf.select(cfg, "lora.merge_lora", default=False):
                cmd.append("--merge_lora")
        else:
            cmd += ["--model_path", model_snapshot]

    log.info(f"执行命令: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        raise RuntimeError(f"执行失败，返回码: {r.returncode}")


if __name__ == "__main__":
    main()
