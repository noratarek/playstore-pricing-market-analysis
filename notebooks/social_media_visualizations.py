import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(PROCESSED_DATA_DIR / "processed_playstore_data.csv")
print(f"Loaded dataset with shape: {df.shape}")
# Create bubble chart for market saturation analysis
plt.figure(figsize=(14, 10))

# Use all categories, not just top opportunities
category_stats = (
    df.groupby("Category")
    .agg(
        {
            "App": "count",
            "Installs": "mean",
            "Market Saturation Index": "mean",  # From your data preparation
        }
    )
    .reset_index()
)

category_stats.columns = ["Category", "App_Count", "Avg_Installs", "Saturation_Index"]

# Create scatter plot
scatter = plt.scatter(
    category_stats["App_Count"],  # X-axis: Number of competing apps
    category_stats["Avg_Installs"],  # Y-axis: Average success (installs)
    s=category_stats["Saturation_Index"] * 50,  # Bubble size = saturation level
    alpha=0.6,
    c=category_stats["Saturation_Index"],  # Color also represents saturation
    cmap="RdYlBu_r",  # Red = high saturation, Blue = low saturation
)

# Add category labels to bubbles
for i, row in category_stats.iterrows():
    plt.annotate(row["Category"], (row["App_Count"], row["Avg_Installs"]), fontsize=9, ha="center")

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Market Saturation Index", fontsize=12)

# Add quadrant lines (optional)
plt.axhline(y=category_stats["Avg_Installs"].median(), color="gray", linestyle="--", alpha=0.5)
plt.axvline(x=category_stats["App_Count"].median(), color="gray", linestyle="--", alpha=0.5)

# Labels and title
plt.xlabel("Number of Apps (Competition Level)", fontsize=12)
plt.ylabel("Average Installs per App (Market Success)", fontsize=12)
plt.title("Market Saturation Analysis by Category", fontsize=15)

# Add quadrant labels
plt.text(
    category_stats["App_Count"].max() * 0.1,
    category_stats["Avg_Installs"].max() * 0.9,
    "HIGH OPPORTUNITY\n(Low Competition,\nHigh Success)",
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
)

plt.text(
    category_stats["App_Count"].max() * 0.9,
    category_stats["Avg_Installs"].max() * 0.9,
    "COMPETITIVE MARKET\n(High Competition,\nHigh Success)",
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
)

plt.text(
    category_stats["App_Count"].max() * 0.9,
    category_stats["Avg_Installs"].max() * 0.1,
    "OVERSATURATED\n(High Competition,\nLow Success)",
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
)

plt.text(
    category_stats["App_Count"].max() * 0.1,
    category_stats["Avg_Installs"].max() * 0.1,
    "EMERGING MARKET\n(Low Competition,\nLow Success)",
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
)

plt.tight_layout()
plt.show()
