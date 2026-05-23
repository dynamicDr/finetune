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


def append_model_args(cfg: DictConfig, cmd: list[str], model_snapshot: str) -> None:
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


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log.info(f"当前配置:\n{OmegaConf.to_yaml(cfg)}")

    name = str(cfg.script)
    script_key = Path(name).name
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
    dataset_shared: list[str] = []
    if script_key in ("vqa_train.py", "vqa_eval.py", "vqa_eval_emb.py", "vqa_eval_ours.py"):
        dataset_shared += ["--model_name", str(cfg.model.name)]
        dataset_shared += ["--dataset", str(OmegaConf.select(cfg, "dataset", default="vsibench"))]
        frame_sampling_method = OmegaConf.select(cfg, "frame_sampling_method", default=None)
        if frame_sampling_method:
            dataset_shared += ["--frame_sampling_method", str(frame_sampling_method)]
        if script_key != "vqa_eval_ours.py":
            focus_blip_model_name = OmegaConf.select(cfg, "focus_blip_model_name", default=None)
            if focus_blip_model_name:
                dataset_shared += ["--focus_blip_model_name", str(focus_blip_model_name)]
            focus_blip_device = OmegaConf.select(cfg, "focus_blip_device", default=None)
            if focus_blip_device is not None and str(focus_blip_device).strip() not in ("", "null", "None"):
                dataset_shared += ["--focus_blip_device", str(focus_blip_device)]
            focus_blip_batch_size = OmegaConf.select(cfg, "focus_blip_batch_size", default=None)
            if focus_blip_batch_size is not None and str(focus_blip_batch_size).strip() not in ("", "null", "None"):
                dataset_shared += ["--focus_blip_batch_size", str(int(focus_blip_batch_size))]
        dataset_split = OmegaConf.select(cfg, "dataset_split", default=None)
        if dataset_split:
            dataset_shared += ["--dataset_split", str(dataset_split)]
        dataset_name = OmegaConf.select(cfg, "dataset_name", default=None)
        if dataset_name:
            dataset_shared += ["--dataset_name", str(dataset_name)]
        dataset_config = OmegaConf.select(cfg, "dataset_config", default=None)
        if dataset_config is not None and str(dataset_config).strip() not in ("", "null", "None"):
            dataset_shared += ["--dataset_config", str(dataset_config)]
        if OmegaConf.select(cfg, "dataset_no_config", default=False):
            dataset_shared.append("--no_dataset_config")
        if script_key in ("vqa_eval.py", "vqa_eval_ours.py"):
            use_preprocessed_clip_frames = OmegaConf.select(cfg, "use_preprocessed_clip_frames", default=False)
            if bool(use_preprocessed_clip_frames):
                dataset_shared.append("--use_preprocessed_clip_frames")
            preprocessed_clip_fps = OmegaConf.select(cfg, "preprocessed_clip_fps", default=None)
            if preprocessed_clip_fps is not None and str(preprocessed_clip_fps).strip() not in ("", "null", "None"):
                dataset_shared += ["--preprocessed_clip_fps", str(float(preprocessed_clip_fps))]
            preprocessed_clip_dir = OmegaConf.select(cfg, "preprocessed_clip_dir", default=None)
            if preprocessed_clip_dir is not None and str(preprocessed_clip_dir).strip() not in ("", "null", "None"):
                dataset_shared += ["--preprocessed_clip_dir", str(preprocessed_clip_dir)]
        if script_key == "vqa_eval_ours.py":
            quota_prescreen_alpha = OmegaConf.select(cfg, "quota_prescreen_alpha", default=None)
            if quota_prescreen_alpha is not None and str(quota_prescreen_alpha).strip() not in ("", "null", "None"):
                dataset_shared += ["--quota_prescreen_alpha", str(int(quota_prescreen_alpha))]
            max_keywords = OmegaConf.select(cfg, "max_keywords", default=None)
            if max_keywords is not None and str(max_keywords).strip() not in ("", "null", "None"):
                dataset_shared += ["--max_keywords", str(int(max_keywords))]
            ours_clip_model_id = OmegaConf.select(cfg, "ours_clip_model_id", default=None)
            if ours_clip_model_id is not None and str(ours_clip_model_id).strip() not in ("", "null", "None"):
                dataset_shared += ["--ours_clip_model_id", str(ours_clip_model_id)]
            ours_clip_device = OmegaConf.select(cfg, "ours_clip_device", default=None)
            if ours_clip_device is not None and str(ours_clip_device).strip() not in ("", "null", "None"):
                dataset_shared += ["--ours_clip_device", str(ours_clip_device)]
            ours_clip_batch_size = OmegaConf.select(cfg, "ours_clip_batch_size", default=None)
            if ours_clip_batch_size is not None and str(ours_clip_batch_size).strip() not in ("", "null", "None"):
                dataset_shared += ["--ours_clip_batch_size", str(int(ours_clip_batch_size))]
            candidate_pool_size = OmegaConf.select(cfg, "candidate_pool_size", default=None)
            if candidate_pool_size is not None and str(candidate_pool_size).strip() not in ("", "null", "None"):
                dataset_shared += ["--candidate_pool_size", str(int(candidate_pool_size))]
            keyword_sim_merge_threshold = OmegaConf.select(cfg, "keyword_sim_merge_threshold", default=None)
            if keyword_sim_merge_threshold is not None and str(keyword_sim_merge_threshold).strip() not in ("", "null", "None"):
                dataset_shared += ["--keyword_sim_merge_threshold", str(float(keyword_sim_merge_threshold))]
            info_peak_floor = OmegaConf.select(cfg, "info_peak_floor", default=None)
            if info_peak_floor is not None and str(info_peak_floor).strip() not in ("", "null", "None"):
                dataset_shared += ["--info_peak_floor", str(float(info_peak_floor))]
            info_peak_ceiling = OmegaConf.select(cfg, "info_peak_ceiling", default=None)
            if info_peak_ceiling is not None and str(info_peak_ceiling).strip() not in ("", "null", "None"):
                dataset_shared += ["--info_peak_ceiling", str(float(info_peak_ceiling))]
            info_prominence_center = OmegaConf.select(cfg, "info_prominence_center", default=None)
            if info_prominence_center is not None and str(info_prominence_center).strip() not in ("", "null", "None"):
                dataset_shared += ["--info_prominence_center", str(float(info_prominence_center))]
            info_prominence_scale = OmegaConf.select(cfg, "info_prominence_scale", default=None)
            if info_prominence_scale is not None and str(info_prominence_scale).strip() not in ("", "null", "None"):
                dataset_shared += ["--info_prominence_scale", str(float(info_prominence_scale))]
            info_entropy_temperature = OmegaConf.select(cfg, "info_entropy_temperature", default=None)
            if info_entropy_temperature is not None and str(info_entropy_temperature).strip() not in ("", "null", "None"):
                dataset_shared += ["--info_entropy_temperature", str(float(info_entropy_temperature))]
            info_mix_prominence = OmegaConf.select(cfg, "info_mix_prominence", default=None)
            if info_mix_prominence is not None and str(info_mix_prominence).strip() not in ("", "null", "None"):
                dataset_shared += ["--info_mix_prominence", str(float(info_mix_prominence))]
            keyword_keep_peak_min = OmegaConf.select(cfg, "keyword_keep_peak_min", default=None)
            if keyword_keep_peak_min is not None and str(keyword_keep_peak_min).strip() not in ("", "null", "None"):
                dataset_shared += ["--keyword_keep_peak_min", str(float(keyword_keep_peak_min))]
            keyword_keep_info_min = OmegaConf.select(cfg, "keyword_keep_info_min", default=None)
            if keyword_keep_info_min is not None and str(keyword_keep_info_min).strip() not in ("", "null", "None"):
                dataset_shared += ["--keyword_keep_info_min", str(float(keyword_keep_info_min))]
            keyword_keep_info_quantile = OmegaConf.select(cfg, "keyword_keep_info_quantile", default=None)
            if keyword_keep_info_quantile is not None and str(keyword_keep_info_quantile).strip() not in ("", "null", "None"):
                dataset_shared += ["--keyword_keep_info_quantile", str(float(keyword_keep_info_quantile))]
            keyword_keep_min_keywords = OmegaConf.select(cfg, "keyword_keep_min_keywords", default=None)
            if keyword_keep_min_keywords is not None and str(keyword_keep_min_keywords).strip() not in ("", "null", "None"):
                dataset_shared += ["--keyword_keep_min_keywords", str(int(keyword_keep_min_keywords))]
            coarse_uniform_ratio = OmegaConf.select(cfg, "coarse_uniform_ratio", default=None)
            if coarse_uniform_ratio is not None and str(coarse_uniform_ratio).strip() not in ("", "null", "None"):
                dataset_shared += ["--coarse_uniform_ratio", str(float(coarse_uniform_ratio))]
            lowres_size = OmegaConf.select(cfg, "lowres_size", default=None)
            if lowres_size is not None and str(lowres_size).strip() not in ("", "null", "None"):
                dataset_shared += ["--lowres_size", str(int(lowres_size))]
    model_snapshot = resolve_model_path(cfg.model.path)

    if script_key in ("train_vsibench.py", "vqa_train.py"):
        if OmegaConf.select(cfg, "lora.enabled", default=False):
            raise ValueError("训练时不要设置 lora.enabled=true")
        out = OmegaConf.select(cfg, "train.output_dir", default=None)
        if not out or str(out).strip() in ("null", "~", ""):
            out = f"outputs/vsibench_train/{cfg.model.name}/{cfg.task_filter}_frames{cfg.num_frames}"
        cmd = ["python", str(sp), "--model_path", model_snapshot, "--output_dir", str(out), *shared, *dataset_shared]
        ms = OmegaConf.select(cfg, "train.max_samples", default=None)
        if ms is not None and str(ms).strip() not in ("", "null", "None"):
            cmd += ["--max_samples", str(int(ms))]
    elif script_key in ("train_vsibench_confidence.py",):
        out = OmegaConf.select(cfg, "train.output_dir", default=None)
        if not out or str(out).strip() in ("null", "~", ""):
            out = f"outputs/vsibench_confidence/{cfg.model.name}/{cfg.task_filter}_frames{cfg.num_frames}"
        cmd = ["python", str(sp), "--output_dir", str(out), *shared]
        append_model_args(cfg, cmd, model_snapshot)
        ms = OmegaConf.select(cfg, "train.max_samples", default=None)
        if ms is not None and str(ms).strip() not in ("", "null", "None"):
            cmd += ["--max_samples", str(int(ms))]
        ep = OmegaConf.select(cfg, "confidence.num_train_epochs", default=None)
        if ep is not None and str(ep).strip() not in ("", "null", "None"):
            cmd += ["--num_train_epochs", str(int(ep))]
        lr = OmegaConf.select(cfg, "confidence.learning_rate", default=None)
        if lr is not None and str(lr).strip() not in ("", "null", "None"):
            cmd += ["--learning_rate", str(float(lr))]
    else:
        cmd = [
            "python",
            str(sp),
            *shared,
            *dataset_shared,
            "--num_samples",
            str(cfg.num_samples),
            "--log_file",
            f"result_{build_batch_timestamp()}.csv",
        ]
        if script_key in ("eval_vsibench_confidence.py", "test_vsibench_confidence.py"):
            head_path = str(OmegaConf.select(cfg, "confidence.head_path", default="") or "").strip()
            if not head_path:
                raise ValueError("eval_vsibench_confidence.py 需要 confidence.head_path")
            cmd += [
                "--confidence_head_path",
                str(Path(head_path).expanduser()),
                "--detail_file",
                f"detail_{build_batch_timestamp()}.csv",
            ]
            th = OmegaConf.select(cfg, "confidence.threshold", default=None)
            if th is not None and str(th).strip() not in ("", "null", "None"):
                cmd += ["--threshold", str(float(th))]
        append_model_args(cfg, cmd, model_snapshot)

    log.info(f"执行命令: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        raise RuntimeError(f"执行失败，返回码: {r.returncode}")


if __name__ == "__main__":
    main()
