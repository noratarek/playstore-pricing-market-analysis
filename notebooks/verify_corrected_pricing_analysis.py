"""
Quick verification script to check the corrected pricing analysis
Run this to verify your numbers are now realistic
"""

import pandas as pd
import numpy as np
from pathlib import Path


def verify_corrected_analysis():
    """Verify the corrected pricing analysis results."""

    # Load your processed data
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

    # Load the main dataset
    df = pd.read_csv(PROCESSED_DATA_DIR / "processed_playstore_data.csv")

    print("=== VERIFICATION OF CORRECTED ANALYSIS ===\n")

    # 1. Verify novelty app filtering
    if "is_novelty_app" in df.columns:
        novelty_count = df["is_novelty_app"].sum()
        print(f"✓ Novelty apps identified: {novelty_count}")
    else:
        # Create novelty flag
        import re

        rich_pattern = re.compile(r"rich|wealth|millionaire|billionaire|money", re.IGNORECASE)
        df["is_novelty_app"] = df.apply(
            lambda row: bool(rich_pattern.search(str(row["App"]))) and row["Price"] > 20, axis=1
        )
        novelty_count = df["is_novelty_app"].sum()
        print(f"✓ Novelty apps identified: {novelty_count}")

    # 2. Create legitimate apps dataset
    legitimate_df = df[~df["is_novelty_app"]].copy()
    legitimate_paid_df = legitimate_df[legitimate_df["Price"] > 0].copy()

    print(f"✓ Legitimate apps: {len(legitimate_df):,}")
    print(f"✓ Legitimate paid apps: {len(legitimate_paid_df):,}")
    print(
        f"✓ Free app percentage: {len(legitimate_df[legitimate_df['Price'] == 0]) / len(legitimate_df) * 100:.1f}%\n"
    )

    # 3. VERIFY CORRECTED CATEGORY AVERAGES
    print("=== CORRECTED CATEGORY AVERAGES (Including Free Apps) ===")

    # Correct method: include all apps
    correct_averages = (
        legitimate_df.groupby("Category").agg({"Price": "mean", "App": "count"}).round(2)
    )
    correct_averages.columns = ["Avg_Price_Correct", "App_Count"]

    # Filter to categories with at least 10 apps
    reliable_categories = correct_averages[correct_averages["App_Count"] >= 10]
    reliable_categories = reliable_categories.sort_values("Avg_Price_Correct", ascending=False)

    print("Top 10 categories by average price (corrected method):")
    print(reliable_categories.head(10))

    # 4. SHOW THE DIFFERENCE FROM WRONG METHOD
    print(f"\n=== COMPARISON: Wrong vs Correct Method ===")

    # Wrong method: paid apps only
    wrong_averages = legitimate_paid_df.groupby("Category")["Price"].mean().round(2)

    # Compare for top categories
    comparison_categories = ["EVENTS", "FINANCE", "BUSINESS", "MEDICAL"]

    for category in comparison_categories:
        if category in wrong_averages.index and category in correct_averages.index:
            wrong_avg = wrong_averages[category]
            correct_avg = correct_averages.loc[category, "Avg_Price_Correct"]
            sample_size = correct_averages.loc[category, "App_Count"]

            print(f"{category}:")
            print(f"  Wrong method (paid only): ${wrong_avg:.2f}")
            print(f"  Correct method (all apps): ${correct_avg:.2f}")
            print(f"  Sample size: {sample_size} apps")
            print(f"  Difference: {wrong_avg / correct_avg:.1f}x overestimated\n")

    # 5. VERIFY INSTALLATION RATIOS
    print("=== INSTALLATION RATIOS VERIFICATION ===")

    order = ["Free", "Low-cost", "Mid-price", "High-price", "Premium"]
    medians = legitimate_df.groupby("Price Category")["Installs"].median()

    print("Median installations by price category:")
    for category in order:
        if category in medians:
            print(f"  {category}: {medians[category]:,.0f}")

    if "Free" in medians:
        free_median = medians["Free"]
        print(f"\nInstallation ratios (Free vs Others):")
        for category in ["Low-cost", "Mid-price", "High-price", "Premium"]:
            if category in medians and medians[category] > 0:
                ratio = free_median / medians[category]
                print(f"  Free vs {category}: {ratio:.1f}:1")

    # 6. CHECK FOR SMALL SAMPLE SIZES
    print(f"\n=== SMALL SAMPLE SIZE WARNINGS ===")
    small_samples = correct_averages[correct_averages["App_Count"] < 10]
    if len(small_samples) > 0:
        print("Categories with < 10 apps (unreliable averages):")
        print(small_samples.sort_values("App_Count"))
    else:
        print("✓ All categories have adequate sample sizes")

    # 7. VERIFY PRICE-RATING CORRELATION
    print(f"\n=== PRICE-RATING CORRELATION VERIFICATION ===")
    correlation = legitimate_paid_df["Price"].corr(legitimate_paid_df["Rating"])
    print(f"Price-Rating correlation (legitimate paid apps): {correlation:.4f}")

    if abs(correlation) < 0.01:
        print("✓ Correlation is essentially zero (as expected)")
    else:
        print("⚠ Correlation is still significant - check data filtering")

    print(f"\n=== VERIFICATION COMPLETE ===")
    print("If all checks pass, your corrected analysis is ready!")

    return correct_averages, medians


if __name__ == "__main__":
    verify_corrected_analysis()
