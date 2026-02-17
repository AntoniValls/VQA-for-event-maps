#!/usr/bin/env python3
"""
Compare VQA model metrics across result folders, including Topic and Risk Score analysis.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ================= CONFIG =================
BASE_RESULTS_DIR = "../data"   
OUTPUT_DIR = "../data/model_comparison"
# Added Risk_MAE to the main metrics list
METRICS = ["Accuracy", "F1", "Precision", "Recall", "Specificity", "Risk_MAE"]
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_results(base_dir):
    overall_records = []
    topic_records = []
    
    root_path = Path(base_dir)
    csv_files = list(root_path.rglob("metrics_summary.csv"))
    print(f"Found {len(csv_files)} metrics files. Processing...")

    for csv_path in csv_files:
        root = csv_path.parent
        model_name = root.name
        
        parts = root.parts
        try:
            data_idx = parts.index("data")
            region = parts[data_idx + 1]
            city = parts[data_idx + 2]
        except (ValueError, IndexError):
            region, city = "Unknown", "Unknown"

        df = pd.read_csv(csv_path)

        # 1. Process Overall Metrics (Risk_MAE is valid here)
        overall = df[df["Category"] == "Overall"].copy()
        if not overall.empty:
            row = overall.iloc[0].to_dict()
            row.update({"model": model_name, "region": region, "city": city})
            overall_records.append(row)

        # 2. Process Topic Metrics 
        # Note: In your CSV, Topic Risk_MAE is 0.0 because risk is an image-level metric,
        # but we keep the logic consistent for future updates.
        topics = df[df["Category"] == "Topic"].copy()
        for _, t_row in topics.iterrows():
            t_dict = t_row.to_dict()
            t_dict.update({"model": model_name, "region": region, "city": city})
            topic_records.append(t_dict)

    return pd.DataFrame(overall_records), pd.DataFrame(topic_records)

# ---------------- Execution ----------------
df_all, df_topics = collect_results(BASE_RESULTS_DIR)

if df_all.empty:
    raise RuntimeError("No data collected. Check your directory structure.")

# Ensure numeric types for aggregation
for m in METRICS:
    df_all[m] = pd.to_numeric(df_all[m])
    df_topics[m] = pd.to_numeric(df_topics[m])

# 1. GLOBAL MODEL SUMMARY
summary_table = df_all.groupby("model")[METRICS].mean()

# 2. CONTINENT ANALYSIS
continent_table = df_all.groupby(["region", "model"])[METRICS].mean()

# 3. TOPIC ANALYSIS
topic_table = df_topics.groupby(["Subcategory", "model"])[METRICS].mean()

# ---------------- Printing ----------------
def print_header(title):
    print("\n" + "="*95 + f"\n📊 {title}\n" + "="*95)

# Update formatters to include Risk_MAE
format_map = {m: '{:.3f}'.format for m in METRICS}

print_header("GLOBAL MODEL PERFORMANCE (Mean across all cities)")
print(summary_table.to_string(formatters=format_map))

print_header("CONTINENT-BASED PERFORMANCE")
print(continent_table.to_string(formatters=format_map))

print_header("TOPIC-BASED PERFORMANCE")
print(topic_table.to_string(formatters=format_map))

# ---------------- Plotting ----------------
def save_comparison_plot(df, group_col, filename, title):
    plot_df = df.groupby(group_col)[METRICS].mean()
    groups = plot_df.index.tolist()
    
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 6), sharey=False)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(groups)))

    for i, metric in enumerate(METRICS):
        values = plot_df[metric]
        axes[i].bar(groups, values, color=colors, edgecolor="black", alpha=0.8)
        axes[i].set_title(f"{metric}", fontsize=14, fontweight='bold')
        
        # Risk_MAE should not have a fixed 1.05 limit if error exceeds it
        if metric == "Risk_MAE":
            axes[i].set_ylabel("Error (Lower is Better)")
            axes[i].set_ylim(0, max(values) * 1.2 if not values.empty else 1.0)
        else:
            axes[i].set_ylim(0, 1.05)
            
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(title, fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / filename, bbox_inches="tight", dpi=200)
    plt.close()

# Save Plots
save_comparison_plot(df_all, "model", "model_overall_comparison.png", "Overall Model Comparison")
save_comparison_plot(df_all, "region", "continent_comparison.png", "Performance by Continent")

# --- Risk Score Specific Plot ---
def save_risk_distribution_plot(df, output_path):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    # 1. Map the raw names to pretty names
    # Example: 'qwen-vl' -> 'Qwen-VL'
    def format_label(name):
        pretty_name = name.replace("-", " ").title()
        # Specific overrides for acronyms if needed
        pretty_name = pretty_name.replace("Vilt", "ViLT").replace("Qwen Vl", "Qwen-VL").replace("Instructblip", "InstructBLIP").replace("Llava", "LLaVA")
        return f"{pretty_name}"

    # 2. Create the violin plot
    ax = sns.violinplot(
        data=df, 
        x="model", 
        y="Risk_MAE", 
        hue="model",
        palette="muted",
        inner="quartile", # Useful to see the distribution medians
        legend=False
    )
    
    # 3. Apply the improved x-labels
    new_labels = [format_label(l.get_text()) for l in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels, fontweight='bold')

    # 4. Styling
    plt.ylabel(r"MAE $\mathcal{R}_{img}$", fontsize=14)
    plt.xlabel("VQA Model", fontsize=12, fontweight='bold')
    
    # Add a horizontal line at 0 for reference
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

# Execute the new violin plot
save_risk_distribution_plot(df_all, Path(OUTPUT_DIR) / "model_risk_mae_distribution.png")

# Topic Plot: F1 Score
plt.figure(figsize=(12, 6))
topic_f1 = df_topics.pivot_table(index="Subcategory", columns="model", values="F1", aggfunc="mean")
topic_f1.plot(kind="bar", figsize=(14, 7), edgecolor="black")
plt.title("F1 Score by Topic and Model", fontsize=16)
plt.ylabel("F1 Score")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "topic_f1_comparison.png")
plt.close()

print(f"\n✅ All reports and plots (including Risk MAE) saved in: {OUTPUT_DIR}")