# -*- coding: utf-8 -*-
"""
Initial exploration of Google Play Store data - Day 1 (Fixed)
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
PROJECT_DIR = Path(__file__).resolve().parents[2]  # Adjust if needed based on your script location
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_DIR / "data" / "interim"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

# Ensure directories exist
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
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


def preprocess_numeric_columns(df):
    """Convert string columns to numeric where appropriate."""
    logger.info("Converting string columns to numeric")

    # Make a copy to avoid modifying the original
    df_processed = df.copy()

    # Handle Reviews column - remove commas and convert to numeric
    if "Reviews" in df_processed.columns:
        try:
            # First try simple conversion
            df_processed["Reviews"] = pd.to_numeric(df_processed["Reviews"], errors="coerce")
        except:
            # If that fails, handle more complex formatting
            logger.info("Using regex to clean Reviews column")

            def clean_reviews(value):
                if pd.isna(value):
                    return np.nan

                try:
                    # Handle values like '3.0M'
                    if isinstance(value, str) and "M" in value:
                        return float(value.replace("M", "")) * 1000000
                    elif isinstance(value, str) and "K" in value:
                        return float(value.replace("K", "")) * 1000
                    else:
                        return float(str(value).replace(",", ""))
                except:
                    return np.nan

            df_processed["Reviews"] = df_processed["Reviews"].apply(clean_reviews)

    # Handle Installs column - remove '+', ',' and convert to numeric
    if "Installs" in df_processed.columns:

        def clean_installs(value):
            if pd.isna(value):
                return np.nan

            try:
                # Remove '+', ',' and convert to float
                return float(str(value).replace("+", "").replace(",", ""))
            except:
                return np.nan

        df_processed["Installs"] = df_processed["Installs"].apply(clean_installs)

    # Handle Price column - remove '$' and convert to numeric
    if "Price" in df_processed.columns:

        def clean_price(value):
            if pd.isna(value):
                return np.nan

            try:
                # Remove '$' and convert to float
                return float(str(value).replace("$", ""))
            except:
                return np.nan

        df_processed["Price"] = df_processed["Price"].apply(clean_price)

    # Handle Size column - convert to MB
    if "Size" in df_processed.columns:

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

        df_processed["Size"] = df_processed["Size"].apply(clean_size)

    logger.info("Numeric columns conversion completed")
    return df_processed


def explore_data_structure(df):
    """Explore basic dataset structure and missing values."""
    logger.info("Exploring data structure")

    # Basic dataset info
    print("\n=== DATASET STRUCTURE ===")
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)

    # Check first few rows
    print("\nFirst 5 rows:")
    print(df.head())

    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    print("\n=== MISSING VALUES ===")
    missing_df = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage})
    print(
        missing_df[missing_df["Missing Values"] > 0].sort_values("Missing Values", ascending=False)
    )

    # Save missing values report
    missing_report_path = INTERIM_DATA_DIR / "missing_values_report.csv"
    missing_df.to_csv(missing_report_path)
    logger.info(f"Missing values report saved to {missing_report_path}")

    return missing_df


def explore_categorical_features(df):
    """Explore categorical features in the dataset."""
    logger.info("Analyzing categorical features")

    print("\n=== CATEGORICAL FEATURES ===")
    categorical_columns = ["Category", "Type", "Content Rating", "Genres"]

    for col in categorical_columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            print(f"\n{col}: {unique_count} unique values")

            # Show value counts
            print(df[col].value_counts().head(10))

            # Create visualizations for important categorical variables
            plt.figure(figsize=(12, 6))
            value_counts = df[col].value_counts().head(15).sort_values(ascending=False)

            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f"Top 15 {col} Values")
            plt.xticks(rotation=90)
            plt.tight_layout()

            figure_path = FIGURES_DIR / f"{col}_distribution.png"
            plt.savefig(figure_path)
            plt.close()
            logger.info(f"Figure saved to {figure_path}")


def explore_numerical_features(df_processed):
    """Explore numerical features in the preprocessed dataset."""
    logger.info("Analyzing numerical features")

    print("\n=== NUMERICAL FEATURES ===")

    # Define numerical columns after preprocessing
    numeric_columns = ["Rating", "Reviews", "Size", "Installs", "Price"]

    # Filter to only include columns that exist and have numeric data
    available_numeric = []
    for col in numeric_columns:
        if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
            available_numeric.append(col)

    if not available_numeric:
        logger.warning("No numeric columns available for analysis")
        return

    # Generate summary statistics
    print("\nSummary statistics for numerical features:")
    print(df_processed[available_numeric].describe())

    # Create distribution plots for numeric features
    for col in available_numeric:
        plt.figure(figsize=(12, 5))

        # Main distribution plot
        plt.subplot(1, 2, 1)
        sns.histplot(df_processed[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")

        # Boxplot to identify outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df_processed[col].dropna())
        plt.title(f"Boxplot of {col}")

        plt.tight_layout()
        figure_path = FIGURES_DIR / f"{col}_distribution.png"
        plt.savefig(figure_path)
        plt.close()
        logger.info(f"Figure saved to {figure_path}")

    # Explore relationships between numeric features if we have at least 2
    if len(available_numeric) >= 2:
        try:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df_processed[available_numeric].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Matrix of Numerical Features")
            corr_path = FIGURES_DIR / "correlation_matrix.png"
            plt.savefig(corr_path)
            plt.close()
            logger.info(f"Correlation matrix saved to {corr_path}")

            # Save correlation matrix to CSV
            correlation_matrix.to_csv(INTERIM_DATA_DIR / "correlation_matrix.csv")
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")


def analyze_pricing_patterns(df_processed):
    """Analyze pricing patterns in relation to other features."""
    logger.info("Analyzing pricing patterns")

    if "Price" not in df_processed.columns or not pd.api.types.is_numeric_dtype(
        df_processed["Price"]
    ):
        logger.warning("Price column not available as numeric")
        return

    print("\n=== PRICING ANALYSIS ===")

    # Basic price statistics
    print("\nPrice statistics:")
    print(df_processed["Price"].describe())

    # Count of free vs paid apps
    free_count = (df_processed["Price"] == 0).sum()
    paid_count = (df_processed["Price"] > 0).sum()
    print(f"Free apps: {free_count} ({free_count / len(df_processed) * 100:.2f}%)")
    print(f"Paid apps: {paid_count} ({paid_count / len(df_processed) * 100:.2f}%)")

    # Create price categories
    price_bins = [-0.001, 0.001, 1, 5, 10, float("inf")]
    price_labels = ["Free", "Low-cost", "Mid-price", "High-price", "Premium"]
    df_processed["Price Category"] = pd.cut(
        df_processed["Price"], bins=price_bins, labels=price_labels
    )

    # Show price distribution by category
    price_category_counts = df_processed["Price Category"].value_counts()
    print("\nPrice Category Distribution:")
    print(price_category_counts)

    # Visualize price category distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(y="Price Category", data=df_processed, order=price_category_counts.index)
    plt.title("Distribution of App Price Categories")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_category_distribution.png")
    plt.close()
    logger.info("Price category distribution saved")

    # Analyze price by category
    if "Category" in df_processed.columns:
        # Average price by category
        category_price = (
            df_processed.groupby("Category")["Price"].mean().sort_values(ascending=False)
        )
        print("\nAverage Price by Category (Top 10):")
        print(category_price.head(10))

        # Visualize top 10 categories by average price
        plt.figure(figsize=(12, 6))
        category_price.head(10).plot(kind="bar")
        plt.title("Top 10 Categories by Average Price")
        plt.ylabel("Average Price ($)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "top_categories_by_price.png")
        plt.close()
        logger.info("Top categories by price saved")

    # Price vs. Rating analysis
    if "Rating" in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed["Rating"]):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Price Category", y="Rating", data=df_processed)
        plt.title("Rating Distribution by Price Category")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "rating_by_price.png")
        plt.close()
        logger.info("Rating by price category saved")

    # Save price analysis data
    if "Category" in df_processed.columns:
        price_analysis = pd.DataFrame(
            {"Category": category_price.index, "Average Price": category_price.values}
        )
        price_analysis.to_csv(INTERIM_DATA_DIR / "price_by_category.csv", index=False)
        logger.info("Price analysis data saved")


def analyze_market_saturation(df_processed):
    """Analyze market saturation across app categories."""
    logger.info("Analyzing market saturation")

    if "Category" not in df_processed.columns:
        logger.warning("Category column not found in dataset")
        return

    print("\n=== MARKET SATURATION ANALYSIS ===")

    # Count apps per category
    category_counts = df_processed["Category"].value_counts()
    print("\nNumber of Apps per Category (Top 10):")
    print(category_counts.head(10))

    # Create a simple visualization of app counts by category
    plt.figure(figsize=(12, 6))
    category_counts.head(15).plot(kind="bar")
    plt.title("Number of Apps by Category (Top 15)")
    plt.ylabel("Number of Apps")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "app_counts_by_category.png")
    plt.close()
    logger.info("App counts by category saved")

    # Check if we have Installs as numeric for further analysis
    if "Installs" in df_processed.columns and pd.api.types.is_numeric_dtype(
        df_processed["Installs"]
    ):
        # Sum installs by category
        category_installs = (
            df_processed.groupby("Category")["Installs"].sum().sort_values(ascending=False)
        )
        print("\nTotal Installs per Category (Top 10):")
        print(category_installs.head(10))

        # Calculate average installs per app in each category
        category_avg_installs = (
            df_processed.groupby("Category")["Installs"].mean().sort_values(ascending=False)
        )
        print("\nAverage Installs per App by Category (Top 10):")
        print(category_avg_installs.head(10))

        # Create a market saturation dataframe
        market_df = pd.DataFrame(
            {
                "App Count": category_counts,
                "Total Installs": category_installs,
                "Avg Installs Per App": category_avg_installs,
            }
        ).sort_values("App Count", ascending=False)

        # Calculate saturation metric: ratio of app count to average installs
        # Higher values indicate more apps competing for the same install base
        market_df["Saturation Index"] = (
            market_df["App Count"] / market_df["Avg Installs Per App"] * 10000
        )
        market_df = market_df.sort_values("Saturation Index", ascending=False)

        print("\nCategory Saturation Index (Top 10 most saturated):")
        print(market_df[["App Count", "Avg Installs Per App", "Saturation Index"]].head(10))

        # Visualize market saturation
        plt.figure(figsize=(12, 8))
        plt.scatter(
            market_df["App Count"],
            market_df["Avg Installs Per App"],
            s=market_df["Total Installs"] / 1e6,
            alpha=0.7,
        )

        # Annotate points with category names for top categories
        for i, row in market_df.head(15).iterrows():
            plt.annotate(i, (row["App Count"], row["Avg Installs Per App"]), fontsize=8, alpha=0.8)

        plt.title("Market Saturation Analysis by Category")
        plt.xlabel("Number of Apps (Competition)")
        plt.ylabel("Average Installs per App (Success)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "market_saturation.png")
        plt.close()
        logger.info("Market saturation plot saved")

        # Save market analysis data
        market_df.to_csv(INTERIM_DATA_DIR / "market_saturation.csv")
        logger.info("Market saturation analysis saved")


def generate_data_quality_report(df, missing_df):
    """Generate comprehensive data quality report."""
    logger.info("Generating data quality report")

    quality_issues = {
        "dataset_info": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "missing_data": {
            column: f"{percentage:.2f}%"
            for column, percentage in missing_df["Percentage"].items()
            if missing_df["Missing Values"][column] > 0
        },
        "data_type_issues": {},
        "inconsistent_formats": {},
        "outliers": {},
    }

    # Check for data type issues
    for col in df.columns:
        if col in ["Rating", "Reviews", "Size", "Installs", "Price"]:
            non_numeric_count = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if non_numeric_count > 0:
                quality_issues["data_type_issues"][col] = f"{non_numeric_count} non-numeric values"

    # Check for inconsistent formats
    if "Size" in df.columns:
        if df["Size"].astype(str).str.contains("k|M|Varies").any():
            quality_issues["inconsistent_formats"]["Size"] = (
                "Mixed units (k, M) or 'Varies with device'"
            )

    if "Installs" in df.columns:
        if df["Installs"].astype(str).str.contains("\+|\,").any():
            quality_issues["inconsistent_formats"]["Installs"] = "Contains '+' or ',' characters"

    if "Price" in df.columns:
        if df["Price"].astype(str).str.contains("\$").any():
            quality_issues["inconsistent_formats"]["Price"] = "Contains '$' characters"

    # Save quality report
    with open(PROJECT_DIR / "reports" / "data_quality_report.json", "w") as f:
        json.dump(quality_issues, f, indent=4)

    # Print summary
    print("\n=== DATA QUALITY SUMMARY ===")
    print(f"Total rows: {quality_issues['dataset_info']['total_rows']}")
    print(f"Total columns: {quality_issues['dataset_info']['total_columns']}")
    print(f"Columns with missing data: {len(quality_issues['missing_data'])}")
    print(f"Columns with data type issues: {len(quality_issues['data_type_issues'])}")
    print(f"Columns with inconsistent formats: {len(quality_issues['inconsistent_formats'])}")

    logger.info("Data quality report generated")
    return quality_issues


def main():
    """Main function to run exploratory data analysis."""
    logger.info("Starting exploratory data analysis")

    # Load dataset
    df = load_data()

    # Explore data structure and missing values
    missing_df = explore_data_structure(df)

    # Explore categorical features
    explore_categorical_features(df)

    # Preprocess numeric columns
    df_processed = preprocess_numeric_columns(df)

    # Explore numerical features
    explore_numerical_features(df_processed)

    # Analyze pricing patterns
    analyze_pricing_patterns(df_processed)

    # Analyze market saturation
    analyze_market_saturation(df_processed)

    # Generate comprehensive data quality report
    generate_data_quality_report(df, missing_df)

    # Save processed dataset for future use
    df_processed.to_csv(INTERIM_DATA_DIR / "processed_playstore_data.csv", index=False)
    logger.info("Processed dataset saved for future use")

    logger.info("Exploratory data analysis completed")
    print("\nAnalysis complete! Check the reports/figures directory for visualizations.")


if __name__ == "__main__":
    main()
