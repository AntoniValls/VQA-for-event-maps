#!/usr/bin/env python3
"""
Performance comparison: Global summary of all models + 
Targeted Topic and Continent analysis for Qwen-VL.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ================= CONFIG =================
BASE_RESULTS_DIR = "../data"   
OUTPUT_DIR = "../data/model_comparison"
METRICS = ["Accuracy", "F1", "Precision", "Recall", "Specificity"]
FOCUS_MODEL = "qwen-vl" 
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_results(base_dir):
    overall_records = []
    topic_records = []
    root_path = Path(base_dir)

    for csv_path in root_path.rglob("metrics_summary.csv"):
        root = csv_path.parent
        model_name = root.name.lower()
        parts = root.parts
        try:
            data_idx = parts.index("data")
            region, city = parts[data_idx + 1], parts[data_idx + 2]
        except (ValueError, IndexError):
            region, city = "Unknown", "Unknown"

        df = pd.read_csv(csv_path)

        # 1. Global/Overall data
        overall = df[df["Category"] == "Overall"].copy()
        if not overall.empty:
            row = overall.iloc[0].to_dict()
            row.update({"model": model_name, "region": region, "city": city})
            overall_records.append(row)

        # 2. Topic-based data
        topics = df[df["Category"] == "Topic"].copy()
        for _, t_row in topics.iterrows():
            t_dict = t_row.to_dict()
            t_dict.update({"model": model_name, "region": region, "city": city})
            topic_records.append(t_dict)

    return pd.DataFrame(overall_records), pd.DataFrame(topic_records)

# ---------------- Execution ----------------
df_all, df_topics = collect_results(BASE_RESULTS_DIR)

# Filter Qwen for specialized analysis
df_qwen_all = df_all[df_all["model"] == FOCUS_MODEL]
df_qwen_topics = df_topics[df_topics["model"] == FOCUS_MODEL]

# ---------------- Terminal Reporting ----------------

# 1. GLOBAL SUMMARY
print("\n" + "="*85)
print("📊 GLOBAL MODEL SUMMARY (Mean across all cities)")
print("="*85)
summary_table = df_all.groupby("model")[METRICS].mean()
print(summary_table.to_string(formatters={m: '{:.3f}'.format for m in METRICS}))

# 2. QWEN CONTINENT DEEP DIVE
print("\n" + "="*85)
print(f"🌍 CONTINENT ANALYSIS (Focus: {FOCUS_MODEL})")
print("="*85)
continent_qwen = df_qwen_all.groupby("region")[METRICS].mean()
print(continent_qwen.to_string(formatters={m: '{:.3f}'.format for m in METRICS}))

# 3. QWEN TOPIC DEEP DIVE
print("\n" + "="*85)
print(f"🧩 TOPIC ANALYSIS (Focus: {FOCUS_MODEL})")
print("="*85)
topic_qwen = df_qwen_topics.groupby("Subcategory")[METRICS].mean()
print(topic_qwen.to_string(formatters={m: '{:.3f}'.format for m in METRICS}))


# ---------------- Plotting ----------------

# Plot 1: Global Model Comparison (F1 Score)
plt.figure(figsize=(10, 6))
summary_table['F1'].sort_values().plot(kind='bar', color='gray', edgecolor='black')
plt.title("Global Model Comparison (Overall F1 Score)", fontsize=14)
plt.ylabel("F1 Score")
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "global_comparison_f1.png")

# Plot 2: Qwen performance by Continent
plt.figure(figsize=(12, 6))
continent_qwen[METRICS].plot(kind='bar', figsize=(14, 7), edgecolor='black')
plt.title(f"Qwen-VL: Performance Disaggregation by Continent", fontsize=16)
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.legend(loc='upper right', ncol=len(METRICS))
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "qwen_continent_deepdive.png")

# Plot 3: Qwen performance by Topic (F1 vs Recall)
plt.figure(figsize=(14, 8))
topic_qwen[['F1', 'Recall']].sort_values(by='F1').plot(kind='barh', figsize=(14, 8), edgecolor='black')
plt.title(f"Qwen-VL: Safety Metrics (F1 & Recall) per Navigation Topic", fontsize=16)
plt.xlabel("Score")
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "qwen_topic_safety_analysis.png")

print(f"\n✅ All reports and focused plots saved in: {OUTPUT_DIR}")