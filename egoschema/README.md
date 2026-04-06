# EgoSchema 训练与评测

本目录放置 EgoSchema 数据集专有脚本；公共组件位于仓库根目录（例如 `vl_common.py`）。

## 目录约定

- 本地视频目录默认：`~/dataset/egoschema/videos`
- 训练输出目录示例：`outputs/egoschema_train/<model>/<tag>`
- 评测 CSV 输出目录：`eval_csv/`

## 1) 训练

```bash
python egoschema/train_egoschema.py \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --video_dir ~/dataset/egoschema/videos \
  --output_dir outputs/egoschema_train/Qwen3-VL-4B-Instruct/base \
  --num_frames 4 \
  --train_ratio 0.8 \
  --seed 42
```

## 2) 评测（基座模型）

```bash
python egoschema/eval_egoschema.py \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --video_dir ~/dataset/egoschema/videos \
  --num_frames 4 \
  --num_samples all \
  --train_ratio 0.8 \
  --seed 42
```

## 3) 评测（LoRA）

```bash
python egoschema/eval_egoschema.py \
  --use_lora \
  --base_model Qwen/Qwen3-VL-4B-Instruct \
  --model_path outputs/egoschema_train/Qwen3-VL-4B-Instruct/base \
  --video_dir ~/dataset/egoschema/videos \
  --num_frames 4 \
  --num_samples all \
  --train_ratio 0.8 \
  --seed 42
```

> 训练/评测请保持相同的 `train_ratio` 与 `seed`，确保划分一致。
