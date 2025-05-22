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

# Price order and labels
order = ["Free", "Low-cost", "Mid-price", "High-price", "Premium"]
label_map = {
    "Free": "Free ($0)",
    "Low-cost": "Low-cost ($0.01–0.99)",
    "Mid-price": "Mid-price ($1–2.99)",
    "High-price": "High-price ($3–6.99)",
    "Premium": "Premium ($7+)",
}
color_palette = ["#4878d0", "#a1d99b", "#74c476", "#31a354", "#006d2c"]

# Start plot
plt.figure(figsize=(12, 8))
plt.rcParams["axes.facecolor"] = "#f8f9fa"
plt.rcParams["figure.facecolor"] = "white"
plt.grid(True, linestyle="--", alpha=0.3, axis="x")

# Horizontal boxplot
ax = sns.boxplot(
    y="Price Category",
    x="Rating",
    data=df,
    order=order,
    palette=color_palette,
    width=0.6,
    linewidth=1.5,
    fliersize=3,
    showfliers=True,
    orient="h",
)

# Add medians and
grouped = df.groupby("Price Category")["Rating"]
medians = grouped.median()


for i, category in enumerate(order):
    y_coord = i
    if category in medians:
        ax.text(
            medians[category] + 0.1,
            y_coord + 0.15,
            f"Median: {medians[category]:.2f}",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )


# Update y-ticks to show detailed price category
ax.set_yticklabels([label_map[cat] for cat in order], fontsize=11)

# Titles and labels
plt.title("Users Give Higher Ratings to Paid Mobile Apps", fontsize=18, pad=20, fontweight="bold")
plt.ylabel("App Price Category", fontsize=14, labelpad=10)
plt.xlabel("Average User Rating (1–5 scale)", fontsize=14, labelpad=10)

# Explanatory subtitle
plt.figtext(
    0.5,
    0.01,
    "Analysis of 9,745 Google Play apps.",
    ha="right",
    fontsize=12,
    style="italic",
)

# Branding
plt.figtext(0.95, 0.01, "Norhan Kelany • KU Leuven", ha="right", fontsize=8, color="gray")

# Set x-axis limits
plt.xlim(0.8, 5.2)

# Final layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save in multiple formats
basename = FIGURES_DIR / "rating_by_price_horizontal"
for ext in ["png", "svg", "pdf"]:
    path = f"{basename}.{ext}"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.show()
