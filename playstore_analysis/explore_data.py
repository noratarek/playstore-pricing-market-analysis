"""
Initial exploration of Google Play Store data.
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

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
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
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for col in categorical_columns:
        unique_count = df[col].nunique()
        print(f"\n{col}: {unique_count} unique values")

        # Show value counts for columns with manageable number of categories
        if unique_count < 20:
            print(df[col].value_counts().head(10))
        else:
            print(f"Top 5 most common values:")
            print(df[col].value_counts().head(5))

        # Create visualizations for important categorical variables
        if col in ["Category", "Type", "Content Rating"]:
            plt.figure(figsize=(12, 6))
            value_counts = df[col].value_counts().sort_values(ascending=False)

            # Limit to top 15 for readability
            if len(value_counts) > 15:
                value_counts = value_counts.head(15)

            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=90)
            plt.tight_layout()

            figure_path = FIGURES_DIR / f"{col}_distribution.png"
            plt.savefig(figure_path)
            plt.close()
            logger.info(f"Figure saved to {figure_path}")


def explore_numerical_features(df):
    """Explore numerical features in the dataset."""
    logger.info("Analyzing numerical features")

    print("\n=== NUMERICAL FEATURES ===")

    # First attempt numeric conversion for semi-numeric columns
    numeric_columns = []

    # Try to identify and convert numeric columns that might be stored as strings
    possible_numeric = ["Installs", "Price", "Rating", "Size", "Reviews"]

    for col in possible_numeric:
        if col in df.columns:
            # Handle specific pre-processing for known columns
            try:
                if col == "Installs":
                    df[col] = df[col].str.replace(",", "").str.replace("+", "").astype(float)
                elif col == "Price":
                    df[col] = df[col].str.replace("$", "").astype(float)
                elif col == "Size":
                    # Convert size to numeric (MB)
                    df[col] = df[col].replace("Varies with device", np.nan)
                    # Handle M and k suffixes
                    size_numeric = []
                    for size in df[col]:
                        if pd.isna(size):
                            size_numeric.append(np.nan)
                        elif "M" in str(size):
                            size_numeric.append(float(str(size).replace("M", "")))
                        elif "k" in str(size):
                            size_numeric.append(float(str(size).replace("k", "")) / 1000)
                        else:
                            size_numeric.append(float(size))
                    df[col] = size_numeric

                numeric_columns.append(col)
            except Exception as e:
                logger.warning(f"Could not convert {col} to numeric: {e}")

    # Generate summary statistics
    print("\nSummary statistics for numerical features:")
    print(df[numeric_columns].describe())

    # Create distribution plots for numeric features
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))

        # Main distribution plot
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")

        # Boxplot to identify outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col].dropna())
        plt.title(f"Boxplot of {col}")

        plt.tight_layout()
        figure_path = FIGURES_DIR / f"{col}_distribution.png"
        plt.savefig(figure_path)
        plt.close()
        logger.info(f"Figure saved to {figure_path}")

    # Explore relationships between numeric features
    if len(numeric_columns) >= 2:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Matrix of Numerical Features")
        corr_path = FIGURES_DIR / "correlation_matrix.png"
        plt.savefig(corr_path)
        plt.close()
        logger.info(f"Correlation matrix saved to {corr_path}")

        # Save correlation matrix to CSV
        correlation_matrix.to_csv(INTERIM_DATA_DIR / "correlation_matrix.csv")

    return df[numeric_columns].describe()


def analyze_pricing_patterns(df):
    """Analyze pricing patterns in relation to other features."""
    logger.info("Analyzing pricing patterns")

    if "Price" not in df.columns:
        logger.warning("Price column not found in dataset")
        return

    # Ensure Price is numeric
    if not pd.api.types.is_numeric_dtype(df["Price"]):
        df["Price"] = df["Price"].str.replace("$", "").astype(float)

    print("\n=== PRICING ANALYSIS ===")

    # Create price categories
    price_bins = [-0.001, 0.001, 1, 5, 10, float("inf")]
    price_labels = ["Free", "Low-cost", "Mid-price", "High-price", "Premium"]
    df["Price Category"] = pd.cut(df["Price"], bins=price_bins, labels=price_labels)

    # Show price distribution by category
    price_category_counts = df["Price Category"].value_counts()
    print("\nPrice Category Distribution:")
    print(price_category_counts)

    # Visualize price category distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(y="Price Category", data=df, order=price_category_counts.index)
    plt.title("Distribution of App Price Categories")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_category_distribution.png")
    plt.close()

    # Analyze price by category
    if "Category" in df.columns:
        # Average price by category
        category_price = df.groupby("Category")["Price"].mean().sort_values(ascending=False)
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

    # Price vs. Rating analysis
    if "Rating" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Price Category", y="Rating", data=df)
        plt.title("Rating Distribution by Price Category")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "rating_by_price.png")
        plt.close()

    # Save price analysis data
    price_analysis = pd.DataFrame(
        {"Category": category_price.index, "Average Price": category_price.values}
    )
    price_analysis.to_csv(INTERIM_DATA_DIR / "price_by_category.csv", index=False)
    logger.info("Price analysis data saved")


def analyze_market_saturation(df):
    """Analyze market saturation across app categories."""
    logger.info("Analyzing market saturation")

    if "Category" not in df.columns:
        logger.warning("Category column not found in dataset")
        return

    print("\n=== MARKET SATURATION ANALYSIS ===")

    # Count apps per category
    category_counts = df["Category"].value_counts()
    print("\nNumber of Apps per Category:")
    print(category_counts.head(10))

    # Count installs per category (if available)
    if "Installs" in df.columns:
        # Ensure Installs is numeric
        if not pd.api.types.is_numeric_dtype(df["Installs"]):
            try:
                df["Installs"] = (
                    df["Installs"].str.replace(",", "").str.replace("+", "").astype(float)
                )
            except Exception as e:
                logger.warning(f"Could not convert Installs to numeric: {e}")
                return

        # Sum installs by category
        category_installs = df.groupby("Category")["Installs"].sum().sort_values(ascending=False)
        print("\nTotal Installs per Category (Top 10):")
        print(category_installs.head(10))

        # Calculate average installs per app in each category
        category_avg_installs = (
            df.groupby("Category")["Installs"].mean().sort_values(ascending=False)
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
        sns.scatterplot(
            x="App Count",
            y="Avg Installs Per App",
            size="Total Installs",
            hue="Saturation Index",
            sizes=(20, 500),
            data=market_df,
        )

        # Annotate points with category names for top categories
        for i, row in market_df.head(15).iterrows():
            plt.annotate(i, (row["App Count"], row["Avg Installs Per App"]), fontsize=8, alpha=0.7)

        plt.title("Market Saturation Analysis by Category")
        plt.xlabel("Number of Apps (Competition)")
        plt.ylabel("Average Installs per App (Success)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "market_saturation.png")
        plt.close()

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

    # Check for outliers
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
        if outliers > 0:
            quality_issues["outliers"][col] = (
                f"{outliers} potential outliers ({outliers / len(df) * 100:.2f}%)"
            )

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
    print(f"Columns with outliers: {len(quality_issues['outliers'])}")

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

    # Explore numerical features
    explore_numerical_features(df)

    # Analyze pricing patterns
    analyze_pricing_patterns(df)

    # Analyze market saturation
    analyze_market_saturation(df)

    # Generate comprehensive data quality report
    generate_data_quality_report(df, missing_df)

    logger.info("Exploratory data analysis completed")


if __name__ == "__main__":
    main()
