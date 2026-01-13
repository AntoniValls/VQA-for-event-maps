#!/usr/bin/env python3
"""
Compare VQA model metrics across result folders.

Assumed structure:
  data/<Region>/<City>/results/<model_name>/
      ├── metrics_summary.csv
      └── metrics.json

Outputs:
  - Aggregated CSV
  - Comparison plots (Accuracy, F1, Precision, Recall, Specificity)
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================= CONFIG =================
BASE_RESULTS_DIR = "../data"   # change if needed
OUTPUT_DIR = "../model_comparison"
METRICS_TO_PLOT = ["Accuracy", "F1", "Precision", "Recall", "Specificity"]
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def collect_results(base_dir):
    records = []

    for root, dirs, files in os.walk(base_dir):
        if "metrics_summary.csv" in files:
            root = Path(root)
            model_name = root.name

            # Try to infer region / city from path
            parts = root.parts
            try:
                region = parts[parts.index("data") + 1]
                city = parts[parts.index("data") + 2]
            except Exception:
                region, city = "Unknown", "Unknown"

            csv_path = root / "metrics_summary.csv"
            json_path = root / "metrics.json"

            df = pd.read_csv(csv_path)

            # Only keep overall row
            overall = df[df["Category"] == "Overall"].iloc[0]

            record = {
                "model": model_name,
                "region": region,
                "city": city,
            }
            print(record)

            for m in METRICS_TO_PLOT:
                record[m] = overall[m]

            # Load extra info from JSON if needed
            if json_path.exists():
                with open(json_path, "r") as f:
                    metrics_json = json.load(f)
                record["Total"] = metrics_json["overall_metrics"]["Total"]

            records.append(record)

    return pd.DataFrame(records)


# ---------------- Collect ----------------
df_all = collect_results(BASE_RESULTS_DIR)

if df_all.empty:
    raise RuntimeError("No metrics_summary.csv files found")

# Save aggregated table
csv_out = Path(OUTPUT_DIR) / "aggregated_model_metrics.csv"
df_all.to_csv(csv_out, index=False)
print(f"Saved aggregated metrics → {csv_out}")


# ---------------- Plotting ----------------
import numpy as np

# Aggregate: mean per model
df_mean = df_all.groupby("model")[METRICS_TO_PLOT].mean()

models = df_mean.index.tolist()
n_models = len(models)
n_metrics = len(METRICS_TO_PLOT)

# Color map: consistent color per model
cmap = plt.get_cmap("tab10")
model_colors = {model: cmap(i % 10) for i, model in enumerate(models)}

fig, axes = plt.subplots(
    nrows=1,
    ncols=n_metrics,
    figsize=(4 * n_metrics, 5),
    sharey=True
)

if n_metrics == 1:
    axes = [axes]

for ax, metric in zip(axes, METRICS_TO_PLOT):
    values = df_mean[metric]

    bars = ax.bar(
        models,
        values,
        color=[model_colors[m] for m in models],
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

axes[0].set_ylabel("Score")

# Build legend once
legend_handles = [
    plt.Line2D([0], [0], color=model_colors[m], lw=6, label=m)
    for m in models
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=min(5, n_models),
    bbox_to_anchor=(0.5, 1.05)
)

fig.suptitle("VQA Model Comparison (Mean Across Cities)", fontsize=14)

out_path = Path(OUTPUT_DIR) / "model_comparison_all_metrics.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved combined plot → {out_path}")

print("\n✅ Model comparison complete")
