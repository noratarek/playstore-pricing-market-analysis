# -*- coding: utf-8 -*-
"""
Initial exploration of Google Play Store data - Day 1
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths - using relative paths from the notebook directory
PROJECT_DIR = Path(__file__).resolve().parents[2]  # Adjust if needed
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_DIR / "data" / "interim"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

# Ensure directories exist
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load the raw Google Play Store dataset."""
    logger.info("Loading raw Google Play Store data")

    # Adjust filename to match your actual dataset file
    dataset_path = RAW_DATA_DIR / "googleplaystore.csv"

    # Check if file exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load the dataset
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset loaded with shape: {df.shape}")

    return df


# [Rest of your explore_data.py code goes here...]


# Main function
def main():
    """Main function to run exploratory data analysis."""
    logger.info("Starting exploratory data analysis")

    # Load dataset
    df = load_data()

    # Print dataset info
    print("\n=== DATASET STRUCTURE ===")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)

    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    print("\n=== MISSING VALUES ===")
    missing_df = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage})
    print(
        missing_df[missing_df["Missing Values"] > 0].sort_values("Missing Values", ascending=False)
    )

    # Basic exploration of key columns
    print("\n=== CATEGORICAL FEATURES ===")
    for col in ["Category", "Type", "Content Rating"]:
        if col in df.columns:
            print(f"\n{col} value counts:")
            print(df[col].value_counts().head(10))

    # For simplicity, just do a quick check of the pricing information
    if "Price" in df.columns:
        print("\n=== PRICING INFORMATION ===")
        # Convert price to numeric if needed
        if not pd.api.types.is_numeric_dtype(df["Price"]):
            df["Price"] = df["Price"].str.replace("$", "").astype(float)

        # Basic stats on price
        print("Price statistics:")
        print(df["Price"].describe())

        # Count of free vs paid apps
        free_count = (df["Price"] == 0).sum()
        paid_count = (df["Price"] > 0).sum()
        print(f"Free apps: {free_count} ({free_count / len(df) * 100:.2f}%)")
        print(f"Paid apps: {paid_count} ({paid_count / len(df) * 100:.2f}%)")

        # Create a simple price distribution plot
        plt.figure(figsize=(10, 6))
        # Only plot prices > 0 to see the distribution better
        paid_prices = df[df["Price"] > 0]["Price"]
        sns.histplot(paid_prices, bins=30, kde=True)
        plt.title("Distribution of Paid App Prices")
        plt.xlabel("Price ($)")
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(FIGURES_DIR / "price_distribution.png")
        plt.close()
        print("Price distribution plot saved")

    # Simple market saturation check
    if "Category" in df.columns:
        print("\n=== CATEGORY DISTRIBUTION ===")
        category_counts = df["Category"].value_counts()
        print("Top 10 categories by number of apps:")
        print(category_counts.head(10))

        # Simple bar chart of top categories
        plt.figure(figsize=(12, 6))
        category_counts.head(10).plot(kind="bar")
        plt.title("Top 10 Categories by Number of Apps")
        plt.xlabel("Category")
        plt.ylabel("Number of Apps")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "top_categories.png")
        plt.close()
        print("Category distribution plot saved")

    logger.info("Basic exploratory data analysis completed")
    print("\nExploration complete. Check the reports/figures directory for generated plots.")


if __name__ == "__main__":
    main()
