# investigate_medical_family.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures" / "investigation"

# Create directory if it doesn't exist
import os

os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
df = pd.read_csv(PROCESSED_DATA_DIR / "processed_playstore_data.csv")

print("=== INVESTIGATING MEDICAL AND FAMILY CATEGORIES ===\n")

# Get detailed statistics for MEDICAL and FAMILY
categories_to_investigate = ["MEDICAL", "FAMILY"]

for category in categories_to_investigate:
    cat_data = df[df["Category"] == category]

    print(f"\n{'=' * 50}")
    print(f"CATEGORY: {category}")
    print(f"{'=' * 50}")

    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Total apps: {len(cat_data)}")
    print(
        f"Free apps: {(cat_data['Price'] == 0).sum()} ({(cat_data['Price'] == 0).sum() / len(cat_data) * 100:.1f}%)"
    )
    print(
        f"Paid apps: {(cat_data['Price'] > 0).sum()} ({(cat_data['Price'] > 0).sum() / len(cat_data) * 100:.1f}%)"
    )

    # Installation statistics
    print(f"\nInstallation Statistics:")
    print(f"Total installs: {cat_data['Installs'].sum():,.0f}")
    print(f"Average installs: {cat_data['Installs'].mean():,.0f}")
    print(f"Median installs: {cat_data['Installs'].median():,.0f}")
    print(f"Min installs: {cat_data['Installs'].min():,.0f}")
    print(f"Max installs: {cat_data['Installs'].max():,.0f}")

    # Check for outliers
    Q1 = cat_data["Installs"].quantile(0.25)
    Q3 = cat_data["Installs"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = cat_data[
        (cat_data["Installs"] < (Q1 - 1.5 * IQR)) | (cat_data["Installs"] > (Q3 + 1.5 * IQR))
    ]
    print(f"Number of install outliers: {len(outliers)}")

    # Distribution of installs
    print(f"\nInstall Distribution:")
    install_ranges = ["0-1K", "1K-10K", "10K-100K", "100K-1M", "1M-10M", "10M+"]
    bins = [0, 1000, 10000, 100000, 1000000, 10000000, float("inf")]
    install_dist = pd.cut(cat_data["Installs"], bins=bins, labels=install_ranges)
    print(install_dist.value_counts().sort_index())

    # Top apps by installs
    print(f"\nTop 5 apps by installs:")
    top_apps = cat_data.nlargest(5, "Installs")[["App", "Installs", "Rating", "Price"]]
    for idx, row in top_apps.iterrows():
        print(
            f"  {row['App']}: {row['Installs']:,.0f} installs, {row['Rating']} rating, ${row['Price']}"
        )

    # Bottom apps by installs
    print(f"\nBottom 5 apps by installs:")
    bottom_apps = cat_data.nsmallest(5, "Installs")[["App", "Installs", "Rating", "Price"]]
    for idx, row in bottom_apps.iterrows():
        print(
            f"  {row['App']}: {row['Installs']:,.0f} installs, {row['Rating']} rating, ${row['Price']}"
        )

# Compare all categories
print("\n\n=== CATEGORY COMPARISON ===")
category_stats = (
    df.groupby("Category")
    .agg(
        {
            "App": "count",
            "Installs": ["sum", "mean", "median"],
            "Price": lambda x: (x > 0).mean() * 100,  # Percentage of paid apps
            "Rating": "mean",
        }
    )
    .round(2)
)

category_stats.columns = [
    "App_Count",
    "Total_Installs",
    "Avg_Installs",
    "Median_Installs",
    "Paid_Percentage",
    "Avg_Rating",
]
category_stats = category_stats.sort_values("App_Count", ascending=False)

print("\nTop 10 categories by app count:")
print(category_stats.head(10))

# Highlight our categories of interest
print(f"\nMEDICAL rank by app count: {list(category_stats.index).index('MEDICAL') + 1}")
print(f"FAMILY rank by app count: {list(category_stats.index).index('FAMILY') + 1}")
