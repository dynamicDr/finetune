from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("outputs/videomme_short_random_confidence/sample_level_results.csv").resolve()
OUT_DIR = INPUT_CSV.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_CSV.is_file():
    raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
if df.empty:
    raise RuntimeError(f"输入 CSV 为空: {INPUT_CSV}")

required_cols = {"is_correct", "pred_option_prob", "gt_option_prob", "max_option_prob"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"输入 CSV 缺少必要列: {sorted(missing)}")

df["is_correct_float"] = df["is_correct"].astype(float)

corr_pred_conf = float(df["pred_option_prob"].corr(df["is_correct_float"]))
corr_max_conf = float(df["max_option_prob"].corr(df["is_correct_float"]))
overall_acc = float(df["is_correct_float"].mean())

bins = np.linspace(0.0, 1.0, 11)
df["pred_conf_bin"] = pd.cut(df["pred_option_prob"], bins=bins, include_lowest=True)
bin_df = (
    df.groupby("pred_conf_bin", observed=False)
    .agg(
        n=("is_correct", "count"),
        accuracy=("is_correct_float", "mean"),
        pred_option_prob_mean=("pred_option_prob", "mean"),
        gt_option_prob_mean=("gt_option_prob", "mean"),
        max_option_prob_mean=("max_option_prob", "mean"),
    )
    .reset_index()
)

bin_csv = OUT_DIR / "confidence_bin_summary_replot.csv"
summary_txt = OUT_DIR / "summary_replot.txt"

bin_df.to_csv(bin_csv, index=False, encoding="utf-8")
with summary_txt.open("w", encoding="utf-8") as f:
    f.write(f"num_samples={len(df)}\n")
    f.write(f"overall_accuracy={overall_acc:.6f}\n")
    f.write(f"corr(is_correct, pred_option_prob)={corr_pred_conf:.6f}\n")
    f.write(f"corr(is_correct, max_option_prob)={corr_max_conf:.6f}\n")

plot_bin = bin_df.copy()
plot_bin["bin_mid"] = (
    plot_bin["pred_conf_bin"]
    .astype(str)
    .str.extract(r"\((.*), (.*)\]")
    .astype(float)
    .mean(axis=1)
)
plt.figure(figsize=(8, 5))
plt.plot(plot_bin["bin_mid"], plot_bin["accuracy"], marker="o")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, alpha=0.5)
plt.xlabel("Predicted Option Probability (bin midpoint)")
plt.ylabel("Accuracy in Bin")
plt.title("Binned Accuracy vs Predicted Probability (Replot)")
plt.grid(alpha=0.2)
plt.tight_layout()
binned_png = OUT_DIR / "binned_accuracy_vs_pred_prob_replot.png"
plt.savefig(binned_png, dpi=160)
plt.close()

# 图2：Hexbin 密度图（替代散点，避免点重叠）
rng = np.random.default_rng(42)
jitter = rng.uniform(-0.03, 0.03, size=len(df))
plt.figure(figsize=(8, 5))
hb = plt.hexbin(
    df["pred_option_prob"].values,
    df["is_correct_float"].values + jitter,
    gridsize=35,
    cmap="viridis",
    mincnt=1,
)
plt.yticks([0, 1], ["Wrong", "Correct"])
plt.ylim(-0.15, 1.15)
plt.xlabel("Predicted Option Probability")
plt.ylabel("Correctness (jittered)")
plt.title("Hexbin Density: Correctness vs Predicted Probability")
cb = plt.colorbar(hb)
cb.set_label("Sample Count")
plt.tight_layout()
hexbin_png = OUT_DIR / "hexbin_correctness_vs_pred_prob_replot.png"
plt.savefig(hexbin_png, dpi=160)
plt.close()

# 图3：按正确/错误分组的预测置信度箱线图
correct_probs = df.loc[df["is_correct"] == 1, "pred_option_prob"].values
wrong_probs = df.loc[df["is_correct"] == 0, "pred_option_prob"].values
plt.figure(figsize=(7, 5))
plt.boxplot(
    [wrong_probs, correct_probs],
    labels=["Wrong", "Correct"],
    showfliers=False,
)
plt.ylabel("Predicted Option Probability")
plt.title("Predicted Probability by Correctness")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
boxplot_png = OUT_DIR / "boxplot_pred_prob_by_correctness_replot.png"
plt.savefig(boxplot_png, dpi=160)
plt.close()

print("重绘完成，输出文件：")
print(bin_csv)
print(summary_txt)
print(binned_png)
print(hexbin_png)
print(boxplot_png)
