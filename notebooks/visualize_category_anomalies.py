# visualize_category_anomalies.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "processed_playstore_data.csv")

# Calculate category statistics
category_stats = (
    df.groupby("Category")
    .agg(
        {
            "App": "count",
            "Installs": ["sum", "mean", "median", "std"],
            "Rating": "mean",
            "Price": lambda x: (x > 0).mean() * 100,
        }
    )
    .round(2)
)

category_stats.columns = [
    "App_Count",
    "Total_Installs",
    "Avg_Installs",
    "Median_Installs",
    "Std_Installs",
    "Avg_Rating",
    "Paid_Pct",
]

# Calculate coefficient of variation (std/mean) for installs
category_stats["Install_CV"] = category_stats["Std_Installs"] / category_stats["Avg_Installs"]

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. App Count vs Average Installs
ax1 = axes[0, 0]
scatter = ax1.scatter(
    category_stats["App_Count"], category_stats["Avg_Installs"], s=100, alpha=0.6
)
# Highlight MEDICAL and FAMILY
for category in ["MEDICAL", "FAMILY"]:
    if category in category_stats.index:
        row = category_stats.loc[category]
        ax1.scatter(
            row["App_Count"],
            row["Avg_Installs"],
            s=200,
            color="red",
            edgecolors="black",
            linewidth=2,
        )
        ax1.annotate(
            category, (row["App_Count"], row["Avg_Installs"]), fontsize=12, fontweight="bold"
        )
ax1.set_xlabel("Number of Apps")
ax1.set_ylabel("Average Installs per App")
ax1.set_yscale("log")
ax1.set_title("App Count vs Average Installs")
ax1.grid(True, alpha=0.3)

# 2. Distribution of installs for MEDICAL vs others
ax2 = axes[0, 1]
medical_installs = df[df["Category"] == "MEDICAL"]["Installs"]
other_installs = df[df["Category"] != "MEDICAL"]["Installs"].sample(n=len(medical_installs))

ax2.hist(np.log10(medical_installs + 1), bins=30, alpha=0.5, label="MEDICAL", density=True)
ax2.hist(np.log10(other_installs + 1), bins=30, alpha=0.5, label="Others (sample)", density=True)
ax2.set_xlabel("Log10(Installs + 1)")
ax2.set_ylabel("Density")
ax2.set_title("Install Distribution: MEDICAL vs Others")
ax2.legend()

# 3. Distribution of installs for FAMILY vs others
ax3 = axes[0, 2]
family_installs = df[df["Category"] == "FAMILY"]["Installs"]
other_installs2 = df[df["Category"] != "FAMILY"]["Installs"].sample(
    n=min(len(family_installs), 1000)
)

ax3.hist(np.log10(family_installs + 1), bins=30, alpha=0.5, label="FAMILY", density=True)
ax3.hist(np.log10(other_installs2 + 1), bins=30, alpha=0.5, label="Others (sample)", density=True)
ax3.set_xlabel("Log10(Installs + 1)")
ax3.set_ylabel("Density")
ax3.set_title("Install Distribution: FAMILY vs Others")
ax3.legend()

# 4. Paid percentage vs Average installs
ax4 = axes[1, 0]
ax4.scatter(
    category_stats["Paid_Pct"],
    category_stats["Avg_Installs"],
    s=category_stats["App_Count"] / 5,
    alpha=0.6,
)
for category in ["MEDICAL", "FAMILY"]:
    if category in category_stats.index:
        row = category_stats.loc[category]
        ax4.scatter(
            row["Paid_Pct"],
            row["Avg_Installs"],
            s=row["App_Count"] / 5,
            color="red",
            edgecolors="black",
            linewidth=2,
        )
        ax4.annotate(
            category, (row["Paid_Pct"], row["Avg_Installs"]), fontsize=10, fontweight="bold"
        )
ax4.set_xlabel("Percentage of Paid Apps")
ax4.set_ylabel("Average Installs")
ax4.set_yscale("log")
ax4.set_title("Paid Apps % vs Average Installs (size = app count)")

# 5. Box plot of installs by category (top 10 + MEDICAL)
ax5 = axes[1, 1]
top_categories = category_stats.nlargest(10, "App_Count").index.tolist()
if "MEDICAL" not in top_categories:
    top_categories.append("MEDICAL")

plot_data = df[df["Category"].isin(top_categories)][["Category", "Installs"]]
plot_data["Log_Installs"] = np.log10(plot_data["Installs"] + 1)

sns.boxplot(data=plot_data, x="Category", y="Log_Installs", ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha="right")
ax5.set_ylabel("Log10(Installs + 1)")
ax5.set_title("Install Distribution by Category")

# 6. Median vs Mean installs (to detect skewness)
ax6 = axes[1, 2]
ax6.scatter(category_stats["Median_Installs"], category_stats["Avg_Installs"], s=100, alpha=0.6)
# Add diagonal line
max_val = max(category_stats["Median_Installs"].max(), category_stats["Avg_Installs"].max())
ax6.plot([0, max_val], [0, max_val], "k--", alpha=0.3)

for category in ["MEDICAL", "FAMILY"]:
    if category in category_stats.index:
        row = category_stats.loc[category]
        ax6.scatter(
            row["Median_Installs"],
            row["Avg_Installs"],
            s=200,
            color="red",
            edgecolors="black",
            linewidth=2,
        )
        ax6.annotate(
            category, (row["Median_Installs"], row["Avg_Installs"]), fontsize=10, fontweight="bold"
        )

ax6.set_xlabel("Median Installs")
ax6.set_ylabel("Average Installs")
ax6.set_xscale("log")
ax6.set_yscale("log")
ax6.set_title("Median vs Mean Installs (skewness indicator)")

plt.tight_layout()
plt.savefig(PROJECT_DIR / "reports" / "figures" / "category_anomaly_investigation.png", dpi=300)
plt.show()
