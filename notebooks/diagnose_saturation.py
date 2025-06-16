# diagnose_saturation.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

# Load the data
df = pd.read_csv(PROCESSED_DATA_DIR / "processed_playstore_data.csv")

# Check current saturation values
print("=== CURRENT SATURATION INDEX VALUES ===")
print(f"Column exists: {'Market Saturation Index' in df.columns}")

if "Market Saturation Index" in df.columns:
    print(f"Min: {df['Market Saturation Index'].min()}")
    print(f"Max: {df['Market Saturation Index'].max()}")
    print(f"Mean: {df['Market Saturation Index'].mean()}")
    print(f"Median: {df['Market Saturation Index'].median()}")
    print(f"Contains inf: {np.isinf(df['Market Saturation Index']).any()}")
    print(f"Contains nan: {df['Market Saturation Index'].isna().any()}")

    # Check distribution
    print("\nValue distribution:")
    print(df["Market Saturation Index"].describe())

    # Check which categories have extreme values
    category_saturation = df.groupby("Category")["Market Saturation Index"].agg(
        ["mean", "min", "max"]
    )
    print("\n=== SATURATION BY CATEGORY ===")
    print("Top 10 highest saturation:")
    print(category_saturation.sort_values("mean", ascending=False).head(10))

    print("\nCategories with infinite values:")
    inf_mask = np.isinf(df["Market Saturation Index"])
    if inf_mask.any():
        inf_categories = df[inf_mask]["Category"].value_counts()
        print(inf_categories)

# Let's recalculate with the current formula to understand the issue
print("\n=== RECALCULATING TO DIAGNOSE ===")
category_stats = (
    df.groupby("Category")
    .agg({"App": "count", "Installs": "mean"})
    .rename(columns={"App": "App_Count", "Installs": "Avg_Installs"})
)

# Current formula
category_stats["Saturation_Current"] = (
    category_stats["App_Count"] / category_stats["Avg_Installs"]
) * 10000

print("\nCategories with lowest average installs:")
print(category_stats.sort_values("Avg_Installs").head(10))

print("\nWhich causes these saturation values:")
print(category_stats.sort_values("Saturation_Current", ascending=False).head(10))
