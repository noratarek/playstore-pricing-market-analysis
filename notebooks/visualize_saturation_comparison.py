# visualize_saturation_comparison.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "processed_playstore_data.csv")

# Calculate different saturation metrics
category_stats = (
    df.groupby("Category").agg({"App": "count", "Installs": ["mean", "median", "sum"]}).round(0)
)

category_stats.columns = ["App_Count", "Avg_Installs", "Median_Installs", "Total_Installs"]

# Calculate different metrics
# Old formula (might produce infinity)
with np.errstate(divide="ignore"):
    category_stats["Old_Saturation"] = (
        category_stats["App_Count"] / category_stats["Avg_Installs"] * 10000
    )
    category_stats["Old_Saturation"] = category_stats["Old_Saturation"].replace([np.inf], np.nan)

# New log-based formula
category_stats["Log_Saturation"] = np.log1p(category_stats["App_Count"]) / np.log1p(
    category_stats["Avg_Installs"]
)

# Scaled formula
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
category_stats["App_Scaled"] = scaler.fit_transform(category_stats[["App_Count"]])
category_stats["Installs_Scaled"] = scaler.fit_transform(category_stats[["Avg_Installs"]])
category_stats["Scaled_Saturation"] = category_stats["App_Scaled"] * (
    1 - category_stats["Installs_Scaled"]
)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Old vs New Saturation
ax1 = axes[0, 0]
valid_old = category_stats.dropna(subset=["Old_Saturation"])
ax1.scatter(valid_old["Old_Saturation"], valid_old["Log_Saturation"])
ax1.set_xlabel("Old Saturation (with infinities removed)")
ax1.set_ylabel("Log-based Saturation")
ax1.set_title("Old vs New Saturation Metrics")

# Plot 2: App Count vs Installs with saturation colors
ax2 = axes[0, 1]
scatter = ax2.scatter(
    category_stats["App_Count"],
    category_stats["Avg_Installs"],
    c=category_stats["Log_Saturation"],
    cmap="RdYlBu_r",
    s=100,
)
ax2.set_xlabel("Number of Apps")
ax2.set_ylabel("Average Installs")
ax2.set_yscale("log")
ax2.set_title("Market Saturation Visualization")
plt.colorbar(scatter, ax=ax2, label="Log Saturation")

# Plot 3: Saturation rankings comparison
ax3 = axes[1, 0]
top_10_old = category_stats.nlargest(10, "Old_Saturation", keep="all").index
top_10_new = category_stats.nlargest(10, "Log_Saturation").index
rankings_df = pd.DataFrame(
    {
        "Old_Rank": [
            list(top_10_old).index(cat) + 1 if cat in top_10_old else 11
            for cat in category_stats.index
        ],
        "New_Rank": [
            list(top_10_new).index(cat) + 1 if cat in top_10_new else 11
            for cat in category_stats.index
        ],
    }
)
rankings_df = rankings_df[rankings_df["Old_Rank"] <= 10]
ax3.scatter(rankings_df["Old_Rank"], rankings_df["New_Rank"])
ax3.set_xlabel("Old Formula Rank")
ax3.set_ylabel("New Formula Rank")
ax3.set_title("Top 10 Ranking Comparison")

# Plot 4: Distribution of saturation values
ax4 = axes[1, 1]
ax4.hist(category_stats["Log_Saturation"].dropna(), bins=20, alpha=0.7, label="Log-based")
ax4.hist(category_stats["Scaled_Saturation"].dropna(), bins=20, alpha=0.7, label="Scaled")
ax4.set_xlabel("Saturation Value")
ax4.set_ylabel("Frequency")
ax4.set_title("Distribution of Saturation Values")
ax4.legend()

plt.tight_layout()
plt.savefig(PROJECT_DIR / "reports" / "figures" / "saturation_comparison.png")
plt.show()

# Print summary
print("=== SATURATION METRICS SUMMARY ===")
print(
    f"Categories with infinite old saturation: {np.isinf(category_stats['Old_Saturation']).sum()}"
)
print(
    f"Categories with infinite log saturation: {np.isinf(category_stats['Log_Saturation']).sum()}"
)

print("\nMost saturated categories (Log-based):")
print(
    category_stats.nlargest(10, "Log_Saturation")[["App_Count", "Avg_Installs", "Log_Saturation"]]
)
