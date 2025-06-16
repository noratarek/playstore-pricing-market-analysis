# saturation_metrics.py
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_saturation_metrics(df):
    """Calculate various saturation metrics for comparison."""

    # Group by category
    category_stats = df.groupby("Category").agg(
        {"App": "count", "Installs": ["sum", "mean", "median"], "Rating": "mean"}
    )

    # Flatten column names
    category_stats.columns = [
        "_".join(col).strip() if col[1] else col[0] for col in category_stats.columns
    ]
    category_stats.rename(
        columns={
            "App_count": "App_Count",
            "Installs_sum": "Total_Installs",
            "Installs_mean": "Avg_Installs",
            "Installs_median": "Median_Installs",
            "Rating_mean": "Avg_Rating",
        },
        inplace=True,
    )

    # Method 1: Log-normalized saturation
    # Use log to handle scale differences
    category_stats["Saturation_Log"] = np.log1p(category_stats["App_Count"]) / np.log1p(
        category_stats["Avg_Installs"]
    )

    # Method 2: Market share approach
    # What percentage of total installs does each app get on average
    category_stats["Avg_Market_Share"] = (
        category_stats["Avg_Installs"] / category_stats["Total_Installs"] * 100
    )
    category_stats["Saturation_Market_Share"] = (
        1 / category_stats["Avg_Market_Share"]  # Inverse - lower share = more saturated
    )

    # Method 3: Percentile-based saturation
    # Normalize both metrics to 0-100 scale
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 100))

    category_stats["App_Count_Scaled"] = scaler.fit_transform(category_stats[["App_Count"]])
    category_stats["Installs_Scaled"] = scaler.fit_transform(category_stats[["Avg_Installs"]])

    # High app count + low installs = high saturation
    category_stats["Saturation_Scaled"] = (
        category_stats["App_Count_Scaled"] * (100 - category_stats["Installs_Scaled"]) / 100
    )

    # Method 4: Competition intensity
    # Apps per million installs
    category_stats["Competition_Intensity"] = category_stats["App_Count"] / (
        category_stats["Total_Installs"] / 1_000_000
    )

    # Method 5: Install concentration (Herfindahl-like)
    # Lower median/mean ratio suggests few apps dominate
    category_stats["Install_Concentration"] = (
        category_stats["Median_Installs"] / category_stats["Avg_Installs"]
    )
    category_stats["Saturation_Concentration"] = category_stats["App_Count"] * (
        1 - category_stats["Install_Concentration"]
    )

    return category_stats


# Test the metrics
PROJECT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "processed_playstore_data.csv")

results = calculate_saturation_metrics(df)

# Compare different metrics
print("=== COMPARING SATURATION METRICS ===")
for metric in [
    "Saturation_Log",
    "Saturation_Market_Share",
    "Saturation_Scaled",
    "Competition_Intensity",
    "Saturation_Concentration",
]:
    print(f"\n{metric}:")
    print(f"Range: {results[metric].min():.2f} to {results[metric].max():.2f}")
    print(f"No infinities: {not np.isinf(results[metric]).any()}")
    print("Top 5 most saturated:")
    print(results.sort_values(metric, ascending=False).head(5)[[metric]].round(2))
