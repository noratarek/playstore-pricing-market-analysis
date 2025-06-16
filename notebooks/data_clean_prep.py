# -*- coding: utf-8 -*-
"""
Data Cleaning and Preparation - Google Play Store Analysis
Phase 1: Finalize Data Preparation
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
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]  # Adjust if needed based on your script location
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

# Ensure directories exist
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load the raw Google Play Store dataset."""
    logger.info("Loading raw Google Play Store data")

    dataset_path = RAW_DATA_DIR / "googleplaystore.csv"

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset loaded with shape: {df.shape}")

    return df


def clean_data(df):
    """
    Comprehensive data cleaning for the Google Play Store dataset.
    """
    logger.info("Starting comprehensive data cleaning process")

    # Make a copy to avoid modifying the original
    df_clean = df.copy()

    # remove problematic rows
    df_clean = remove_problematic_rows(df_clean)

    # 1. Handle duplicate records
    logger.info("Checking for duplicate records")
    duplicates = df_clean.duplicated(subset=["App", "Category"])
    if duplicates.sum() > 0:
        logger.info(f"Found {duplicates.sum()} duplicate records. Removing duplicates.")
        df_clean = df_clean.drop_duplicates(subset=["App", "Category"], keep="first")

    # 2. Handle missing values
    logger.info("Handling missing values")

    # 2.1 Rating (13.6% missing) - impute with category median
    if df_clean["Rating"].isnull().sum() > 0:
        # Convert to numeric first (handle any non-numeric values)
        df_clean["Rating"] = pd.to_numeric(df_clean["Rating"], errors="coerce")

        # Calculate category medians for imputation
        category_medians = df_clean.groupby("Category")["Rating"].median()

        # For each category, fill missing ratings with category median
        for category, median_rating in category_medians.items():
            category_mask = (df_clean["Category"] == category) & (df_clean["Rating"].isnull())
            df_clean.loc[category_mask, "Rating"] = median_rating

        # If any ratings are still null (rare categories), use global median
        if df_clean["Rating"].isnull().sum() > 0:
            global_median = df_clean["Rating"].median()
            df_clean["Rating"].fillna(global_median, inplace=True)

        logger.info("Missing Rating values imputed with category medians")

    # 2.2 Other columns with minor missing values (< 0.1%)
    for col in ["Type", "Content Rating", "Current Ver", "Android Ver"]:
        if df_clean[col].isnull().sum() > 0:
            # For categorical columns, fill with mode (most common value)
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value, inplace=True)
            logger.info(f"Missing {col} values filled with mode: {mode_value}")

    # 3. Clean and convert numeric columns
    logger.info("Cleaning and converting numeric columns")

    # 3.1 Clean Reviews column
    logger.info("Cleaning Reviews column")
    df_clean["Reviews"] = pd.to_numeric(df_clean["Reviews"], errors="coerce")
    # Fill any NaN values with 0 (assume no reviews)
    df_clean["Reviews"].fillna(0, inplace=True)

    # 3.2 Clean Installs column
    logger.info("Cleaning Installs column")

    def clean_installs(value):
        if pd.isna(value):
            return 0
        try:
            # Remove '+', ',' and convert to integer
            return int(str(value).replace("+", "").replace(",", ""))
        except:
            return 0

    df_clean["Installs"] = df_clean["Installs"].apply(clean_installs)

    # 3.3 Clean Price column
    logger.info("Cleaning Price column")

    def clean_price(value):
        if pd.isna(value):
            return 0.0
        try:
            # Remove '$' and convert to float
            return float(str(value).replace("$", ""))
        except:
            return 0.0

    df_clean["Price"] = df_clean["Price"].apply(clean_price)

    # 3.4 Clean Size column
    logger.info("Cleaning Size column")

    def clean_size(value):
        if pd.isna(value) or value == "Varies with device":
            return np.nan

        try:
            if "M" in str(value):
                return float(str(value).replace("M", ""))
            elif "k" in str(value) or "K" in str(value):
                # Convert k to MB
                return float(str(value).lower().replace("k", "")) / 1024
            else:
                return float(value)
        except:
            return np.nan

    df_clean["Size"] = df_clean["Size"].apply(clean_size)

    # 4. Clean categorical columns
    logger.info("Cleaning categorical columns")

    # 4.1 Standardize content rating categories
    content_rating_map = {
        "Everyone": "Everyone",
        "Teen": "Teen",
        "Mature 17+": "Mature 17+",
        "Everyone 10+": "Everyone 10+",
        "Adults only 18+": "Adults only 18+",
        "Unrated": "Unrated",
    }
    df_clean["Content Rating"] = df_clean["Content Rating"].map(
        lambda x: content_rating_map.get(x, x)
    )

    # 4.2 Extract primary genre from Genres column (if multiple)
    df_clean["Primary Genre"] = df_clean["Genres"].apply(
        lambda x: str(x).split(";")[0] if pd.notnull(x) and ";" in str(x) else x
    )

    # 4.3 Convert last updated to datetime
    logger.info("Converting Last Updated to datetime")
    df_clean["Last Updated"] = pd.to_datetime(df_clean["Last Updated"], errors="coerce")

    # 5. Handle outliers
    logger.info("Handling outliers")

    # 5.1 Cap Rating at 5.0 (Google Play Store maximum)
    if (df_clean["Rating"] > 5).any():
        logger.info(f"Capping {(df_clean['Rating'] > 5).sum()} Rating values to 5.0")
        df_clean.loc[df_clean["Rating"] > 5, "Rating"] = 5.0

    # 5.2 Cap minimum Rating at 1.0 (Google Play Store minimum)
    if (df_clean["Rating"] < 1).any() & (df_clean["Rating"] > 0).any():
        logger.info(
            f"Capping {((df_clean['Rating'] < 1) & (df_clean['Rating'] > 0)).sum()} Rating values to 1.0"
        )
        df_clean.loc[(df_clean["Rating"] < 1) & (df_clean["Rating"] > 0), "Rating"] = 1.0

    # 5.3 Create flags for extreme values (potential outliers) for analysis
    # Define outlier thresholds (specific to this dataset)
    PRICE_THRESHOLD = 30  # $30+ considered high for most apps
    INSTALLS_THRESHOLD = 10000000  # 10M+ installs is extraordinary
    REVIEWS_THRESHOLD = 1000000  # 1M+ reviews is extraordinary

    df_clean["is_price_outlier"] = df_clean["Price"] > PRICE_THRESHOLD
    df_clean["is_installs_outlier"] = df_clean["Installs"] > INSTALLS_THRESHOLD
    df_clean["is_reviews_outlier"] = df_clean["Reviews"] > REVIEWS_THRESHOLD

    # 5.4 Check for 'I am rich' type apps which are novelty high-priced apps
    logger.info('Identifying novelty high-priced apps ("I am rich" type)')
    rich_pattern = re.compile(r"rich|wealth|millionaire|billionaire|money", re.IGNORECASE)
    df_clean["is_novelty_app"] = df_clean.apply(
        lambda row: bool(rich_pattern.search(str(row["App"]))) and row["Price"] > 20, axis=1
    )

    if df_clean["is_novelty_app"].sum() > 0:
        logger.info(
            f"Identified {df_clean['is_novelty_app'].sum()} potential novelty high-priced apps"
        )

    logger.info("Data cleaning completed")
    return df_clean


def create_derived_features(df_clean):
    """
    Create derived features for analysis.
    """
    logger.info("Creating derived features")

    df_features = df_clean.copy()

    # 1. Price categories
    logger.info("Creating price categories")
    price_bins = [-0.001, 0.001, 1, 5, 10, float("inf")]
    price_labels = ["Free", "Low-cost", "Mid-price", "High-price", "Premium"]
    df_features["Price Category"] = pd.cut(
        df_features["Price"], bins=price_bins, labels=price_labels
    )

    # 2. App age (in days from last update to current date)
    logger.info("Creating app age feature")
    current_date = pd.to_datetime("2018-08-08")  # Dataset appears to be from Aug 2018

    # Calculate app age in days
    df_features["App Age (days)"] = (current_date - df_features["Last Updated"]).dt.days

    # Create age categories
    age_bins = [0, 30, 90, 180, 365, float("inf")]
    age_labels = ["Very Recent", "Recent", "Moderate", "Older", "Very Old"]
    df_features["App Age Category"] = pd.cut(
        df_features["App Age (days)"], bins=age_bins, labels=age_labels
    )

    # 3. Market saturation metrics
    logger.info("Creating market saturation metrics")

    # Calculate app count by category
    category_counts = df_features.groupby("Category")["App"].count()
    df_features["Category App Count"] = df_features["Category"].map(category_counts)

    # Calculate average installs by category
    category_avg_installs = df_features.groupby("Category")["Installs"].mean()
    df_features["Category Avg Installs"] = df_features["Category"].map(category_avg_installs)

    # NEW: Use log-based saturation index to handle scale differences
    # This prevents infinity when average installs is very low
    df_features["Market Saturation Index"] = df_features.apply(
        lambda row: np.log1p(row["Category App Count"])
        / np.log1p(row["Category Avg Installs"] + 1),
        axis=1,
    )

    # Calculate market saturation index
    # Higher values indicate more apps competing for same install base (more saturated)
    df_features["Market Saturation Index"] = (
        df_features["Category App Count"] / df_features["Category Avg Installs"] * 10000
    )

    # 4. Competitive position metrics
    logger.info("Creating competitive position metrics")

    # Calculate app installs relative to category average (>1 means better than average)
    df_features["Relative Popularity"] = (
        df_features["Installs"] / df_features["Category Avg Installs"]
    )

    # Calculate app rating relative to category average
    category_avg_rating = df_features.groupby("Category")["Rating"].mean()
    df_features["Category Avg Rating"] = df_features["Category"].map(category_avg_rating)
    df_features["Relative Rating"] = df_features["Rating"] / df_features["Category Avg Rating"]

    # 5. Binary flags
    logger.info("Creating binary flags")
    df_features["is_free"] = df_features["Price"] == 0
    df_features["has_high_rating"] = df_features["Rating"] >= 4.5
    df_features["is_recently_updated"] = df_features["App Age (days)"] <= 90

    # 6. Create a competitiveness tier feature
    logger.info("Creating competitiveness tier feature")

    def get_competitiveness_tier(row):
        # If market is highly saturated
        if row["Market Saturation Index"] > df_features["Market Saturation Index"].median():
            if row["Relative Popularity"] > 1:
                return "Competitive Leader"
            else:
                return "Highly Competitive"
        else:
            if row["Relative Popularity"] > 1:
                return "Market Leader"
            else:
                return "Low Competition"

    df_features["Competitiveness Tier"] = df_features.apply(get_competitiveness_tier, axis=1)

    # 7. Create price efficiency metric (rating per dollar for paid apps)
    logger.info("Creating price efficiency metric")
    df_features["Price Efficiency"] = df_features.apply(
        lambda row: row["Rating"] / row["Price"] if row["Price"] > 0 else row["Rating"], axis=1
    )

    logger.info("Derived features creation completed")
    return df_features


def split_data_for_analysis(df_features):
    """
    Split the dataset into different subsets for specialized analysis.
    """
    logger.info("Splitting data for analysis")

    # Create dictionary to hold the different data splits
    data_splits = {}

    # 1. Free vs paid apps
    data_splits["free_apps"] = df_features[df_features["is_free"]]
    data_splits["paid_apps"] = df_features[~df_features["is_free"]]
    logger.info(
        f"Split into {len(data_splits['free_apps'])} free apps and {len(data_splits['paid_apps'])} paid apps"
    )

    # 2. By category (top 10 categories by app count)
    top_categories = df_features["Category"].value_counts().head(10).index.tolist()
    for category in top_categories:
        key = f"category_{category.lower().replace(' & ', '_').replace(' ', '_')}"
        data_splits[key] = df_features[df_features["Category"] == category]
        logger.info(f"Created split for {category} with {len(data_splits[key])} apps")

    # 3. By competitiveness tier
    for tier in df_features["Competitiveness Tier"].unique():
        key = f"competition_{tier.lower().replace(' ', '_')}"
        data_splits[key] = df_features[df_features["Competitiveness Tier"] == tier]
        logger.info(f"Created split for {tier} with {len(data_splits[key])} apps")

    # 4. Novelty apps split
    if "is_novelty_app" in df_features.columns and df_features["is_novelty_app"].sum() > 0:
        data_splits["novelty_apps"] = df_features[df_features["is_novelty_app"]]
        logger.info(f"Created split for novelty apps with {len(data_splits['novelty_apps'])} apps")

    # 5. Premium-priced apps (non-novelty)
    premium_mask = (df_features["Price"] > 10) & (~df_features["is_novelty_app"])
    data_splits["premium_apps"] = df_features[premium_mask]
    logger.info(f"Created split for premium apps with {len(data_splits['premium_apps'])} apps")

    logger.info("Data splitting completed")
    return data_splits


def generate_summary_visualizations(df_features):
    """
    Generate summary visualizations of the cleaned and feature-enhanced dataset.
    """
    logger.info("Generating summary visualizations")

    # 1. Price Category Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(
        y="Price Category",
        data=df_features,
        order=df_features["Price Category"].value_counts().index,
    )
    plt.title("Distribution of App Price Categories (After Cleaning)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clean_price_category_distribution.png")
    plt.close()

    # 2. Market Saturation by Category
    # Prepare data for market saturation plot
    category_stats = (
        df_features.groupby("Category")
        .agg({"App": "count", "Installs": "mean", "Market Saturation Index": "mean"})
        .reset_index()
    )
    category_stats.columns = ["Category", "App Count", "Avg Installs", "Saturation Index"]

    # Create bubble chart
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        category_stats["App Count"],
        category_stats["Avg Installs"],
        s=category_stats["Saturation Index"] * 10,  # Size represents saturation
        alpha=0.6,
        c=category_stats["Saturation Index"],
        cmap="coolwarm",
    )

    # Add a colorbar legend
    cbar = plt.colorbar(scatter)
    cbar.set_label("Market Saturation Index")

    # Add category labels to points
    for i, row in category_stats.iterrows():
        plt.annotate(row["Category"], (row["App Count"], row["Avg Installs"]), fontsize=9)

    plt.title("Market Saturation Analysis by Category (After Cleaning)")
    plt.xlabel("Number of Apps (Competition)")
    plt.ylabel("Average Installs per App (Success)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clean_market_saturation.png")
    plt.close()

    # 3. Rating Distribution by Price Category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Price Category", y="Rating", data=df_features)
    plt.title("Rating Distribution by Price Category (After Cleaning)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clean_rating_by_price.png")
    plt.close()

    # 4. Competitiveness Tier Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(
        y="Competitiveness Tier",
        data=df_features,
        order=df_features["Competitiveness Tier"].value_counts().index,
    )
    plt.title("Distribution of Competitiveness Tiers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "competitiveness_tiers.png")
    plt.close()

    # 5. Price Efficiency by Category (for Paid Apps)
    plt.figure(figsize=(12, 8))
    paid_apps = df_features[df_features["Price"] > 0]
    category_price_efficiency = (
        paid_apps.groupby("Category")["Price Efficiency"].mean().sort_values(ascending=False)
    )

    # Plot top 15 categories by price efficiency
    category_price_efficiency.head(15).plot(kind="bar")
    plt.title("Average Price Efficiency by Category (Rating per Dollar, Paid Apps Only)")
    plt.ylabel("Price Efficiency (Rating/Dollar)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_efficiency_by_category.png")
    plt.close()

    logger.info("Summary visualizations generated")


def save_data_quality_report(df_features):
    """
    Generate a simple data quality report after cleaning.
    """
    report = {
        "dataset_size": {"rows": len(df_features), "columns": len(df_features.columns)},
        "missing_values": {
            col: int(count) for col, count in df_features.isnull().sum().items() if count > 0
        },
        "column_types": {col: str(dtype) for col, dtype in df_features.dtypes.items()},
        "derived_features": [
            col
            for col in df_features.columns
            if col
            not in [
                "App",
                "Category",
                "Rating",
                "Reviews",
                "Size",
                "Installs",
                "Type",
                "Price",
                "Content Rating",
                "Genres",
                "Last Updated",
                "Current Ver",
                "Android Ver",
            ]
        ],
    }

    # Save as JSON
    with open(PROJECT_DIR / "reports" / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=4)

    logger.info("Data quality report saved")
    return report


def remove_problematic_rows(df):
    """Remove known problematic rows from the dataset."""
    logger.info("Removing known problematic rows")

    df_clean = df.copy()

    # Remove rows with numeric categories
    numeric_category_mask = df_clean["Category"].str.match(r"^\d+\.?\d*$", na=False)
    if numeric_category_mask.any():
        logger.info(f"Removing {numeric_category_mask.sum()} rows with numeric categories")
        df_clean = df_clean[~numeric_category_mask]

    # Specifically remove the problematic app if it still exists
    problematic_app = "Life Made WI-Fi Touchscreen Photo Frame"
    if problematic_app in df_clean["App"].values:
        logger.info(f"Removing '{problematic_app}'")
        df_clean = df_clean[df_clean["App"] != problematic_app]

    return df_clean


def main():
    """Main function to execute the data preparation process."""
    logger.info("Starting comprehensive data preparation process")

    # 1. Load the raw data
    df_raw = load_data()

    # 2. Clean the data
    df_clean = clean_data(df_raw)

    # 3. Create derived features
    df_features = create_derived_features(df_clean)

    # 4. Split data for analysis
    data_splits = split_data_for_analysis(df_features)

    # 5. Generate summary visualizations
    generate_summary_visualizations(df_features)

    # 6. Save data quality report
    save_data_quality_report(df_features)

    # 7. Save processed data
    df_features.to_csv(PROCESSED_DATA_DIR / "processed_playstore_data.csv", index=False)
    logger.info(
        f"Processed dataset saved to {PROCESSED_DATA_DIR / 'processed_playstore_data.csv'}"
    )

    # 8. Save data splits
    for name, df in data_splits.items():
        df.to_csv(PROCESSED_DATA_DIR / f"{name}.csv", index=False)
        logger.info(f"Data split {name} saved with {len(df)} records")

    logger.info("Data preparation process completed successfully")


if __name__ == "__main__":
    main()
