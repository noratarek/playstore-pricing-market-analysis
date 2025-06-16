"""
Market Opportunity Analysis - Google Play Store (Updated)
Identifying market gaps and saturation patterns across app categories
Updated to align with data cleaning and novelty app removal fixes
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Add parent directory to path to import from playstore_analysis
sys.path.append(str(Path(__file__).resolve().parents[1]))

from playstore_analysis.data_loader import PlayStoreDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures" / "market"
RESULTS_DIR = PROJECT_DIR / "reports" / "market_analysis"

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_clean_market_data():
    """Load data for market opportunity analysis with proper cleaning."""
    logger.info("Loading and cleaning market data")

    # Load the main processed dataset
    df_path = PROCESSED_DATA_DIR / "processed_playstore_data.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {df_path}")

    df = pd.read_csv(df_path)
    logger.info(f"Loaded dataset with {len(df)} apps")

    # CRITICAL: Filter out novelty apps (as per your earlier fixes)
    # Remove apps flagged as novelty apps
    if "is_novelty_app" in df.columns:
        business_apps = df[~df["is_novelty_app"]].copy()
        logger.info(f"Removed {(df['is_novelty_app']).sum()} novelty apps")
    else:
        # Alternative: Remove high-priced apps that might be novelty apps
        business_apps = df[df["Price"] <= 30].copy()  # Reasonable business app price limit
        logger.info(f"Filtered apps with price <= $30: {len(business_apps)} apps remaining")

    # Ensure we have required columns from data preparation
    required_columns = [
        "Category",
        "Rating",
        "Installs",
        "Price",
        "Market Saturation Index",
        "Competitiveness Tier",
        "App Age (days)",
        "Price Category",
    ]

    missing_columns = [col for col in required_columns if col not in business_apps.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")

    return business_apps


def calculate_market_summary(df):
    """Calculate market summary statistics per category (business apps only)."""
    logger.info("Calculating market summary for business apps")

    market_summary = (
        df.groupby("Category")
        .agg(
            {
                "App": "count",
                "Installs": ["mean", "median", "sum"],
                "Rating": "mean",
                "Price": lambda x: (x > 0).mean() * 100,  # Percent paid
                "Market Saturation Index": "mean",
                "App Age (days)": "mean",
            }
        )
        .round(2)
    )

    # Flatten column names
    market_summary.columns = [
        "App_Count",
        "Avg_Installs",
        "Median_Installs",
        "Total_Installs",
        "Avg_Rating",
        "Percent_Paid",
        "Avg_Saturation_Index",
        "Avg_App_Age",
    ]

    # Calculate additional metrics
    market_summary["Market_Share"] = (
        market_summary["Total_Installs"] / market_summary["Total_Installs"].sum() * 100
    )

    return market_summary


def create_main_market_saturation_visualization(df, market_summary):
    """Create the main market saturation bubble chart (corrected version)."""
    logger.info("Creating main market saturation visualization")

    plt.figure(figsize=(14, 10))
    plt.rcParams["axes.facecolor"] = "#f8f9fa"
    plt.rcParams["figure.facecolor"] = "white"
    plt.grid(True, linestyle="--", alpha=0.3)

    # Create the bubble chart
    scatter = plt.scatter(
        market_summary["App_Count"],
        market_summary["Avg_Installs"],
        s=market_summary["Avg_Saturation_Index"] * 50,  # Bubble size = saturation
        alpha=0.6,
        c=market_summary["Avg_Saturation_Index"],
        cmap="RdYlBu_r",  # Red = high saturation, Blue = low saturation
    )

    # Add category labels
    for idx, row in market_summary.iterrows():
        plt.annotate(
            idx, (row["App_Count"], row["Avg_Installs"]), fontsize=9, ha="center", alpha=0.8
        )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Market Saturation Index", fontsize=12)

    # Add quadrant lines at medians
    plt.axhline(y=market_summary["Avg_Installs"].median(), color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=market_summary["App_Count"].median(), color="gray", linestyle="--", alpha=0.5)

    # Add quadrant labels
    plt.text(
        market_summary["App_Count"].max() * 0.1,
        market_summary["Avg_Installs"].max() * 0.9,
        "HIGH OPPORTUNITY\n(Low Competition,\nHigh Success)",
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    plt.text(
        market_summary["App_Count"].max() * 0.9,
        market_summary["Avg_Installs"].max() * 0.1,
        "OVERSATURATED\n(High Competition,\nLow Success)",
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    plt.xlabel("Number of Apps (Competition Level)", fontsize=12)
    plt.ylabel("Average Installs per App (Market Success)", fontsize=12)
    plt.title("Market Saturation Analysis by Category", fontsize=15)
    plt.yscale("log")  # Use log scale for better visualization
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "main_market_saturation_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_saturation_rankings(market_summary):
    """Create rankings of most/least saturated categories."""
    logger.info("Creating saturation rankings")

    # Sort by saturation index
    sorted_by_saturation = market_summary.sort_values("Avg_Saturation_Index", ascending=False)

    plt.figure(figsize=(12, 10))

    # Top 10 most saturated
    plt.subplot(2, 1, 1)
    most_saturated = sorted_by_saturation.head(10)
    plt.barh(most_saturated.index, most_saturated["Avg_Saturation_Index"], color="red", alpha=0.7)
    plt.title("Top 10 Most Saturated Categories", fontsize=14)
    plt.xlabel("Saturation Index")

    # Top 10 least saturated
    plt.subplot(2, 1, 2)
    least_saturated = sorted_by_saturation.tail(10)
    plt.barh(
        least_saturated.index, least_saturated["Avg_Saturation_Index"], color="green", alpha=0.7
    )
    plt.title("Top 10 Least Saturated Categories", fontsize=14)
    plt.xlabel("Saturation Index")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "saturation_rankings.png", dpi=300, bbox_inches="tight")
    plt.close()

    return sorted_by_saturation


def calculate_business_opportunity_score(market_summary):
    """Calculate opportunity scores based on business app data only."""
    logger.info("Calculating business opportunity scores")

    opportunity_df = market_summary.copy()

    # 1. Demand Score: Higher installs = higher demand
    opportunity_df["Demand_Score"] = (
        opportunity_df["Avg_Installs"] / opportunity_df["Avg_Installs"].max()
    )

    # 2. Competition Score: Lower app count = less competition
    opportunity_df["Competition_Score"] = 1 - (
        opportunity_df["App_Count"] / opportunity_df["App_Count"].max()
    )

    # 3. Saturation Score: Lower saturation = better opportunity
    opportunity_df["Saturation_Score"] = 1 - (
        opportunity_df["Avg_Saturation_Index"] / opportunity_df["Avg_Saturation_Index"].max()
    )

    # 4. Quality Score: Higher ratings = better market
    opportunity_df["Quality_Score"] = (
        opportunity_df["Avg_Rating"] / 5.0  # Normalize to 0-1
    )

    # 5. Combined Opportunity Score
    opportunity_df["Opportunity_Score"] = (
        0.3 * opportunity_df["Demand_Score"]
        + 0.3 * opportunity_df["Competition_Score"]
        + 0.2 * opportunity_df["Saturation_Score"]
        + 0.2 * opportunity_df["Quality_Score"]
    )

    # Sort by opportunity score
    opportunity_df = opportunity_df.sort_values("Opportunity_Score", ascending=False)

    return opportunity_df


def create_opportunity_analysis_visualizations(opportunity_df):
    """Create opportunity analysis visualizations."""
    logger.info("Creating opportunity analysis visualizations")

    # 1. Opportunity Score Ranking
    plt.figure(figsize=(12, 8))
    top_15 = opportunity_df.head(15)
    bars = plt.bar(range(len(top_15)), top_15["Opportunity_Score"], color="darkgreen")
    plt.xticks(range(len(top_15)), top_15.index, rotation=45, ha="right")
    plt.title("Market Opportunity Score by Category", fontsize=15)
    plt.xlabel("Category")
    plt.ylabel("Opportunity Score")
    plt.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{top_15.iloc[i]['Opportunity_Score']:.3f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "opportunity_score_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Corrected Opportunity Matrix (Competition vs Demand)
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        opportunity_df["App_Count"],
        opportunity_df["Avg_Installs"],
        s=opportunity_df["Opportunity_Score"] * 1000,
        alpha=0.6,
        c=opportunity_df["Opportunity_Score"],
        cmap="viridis",
    )

    # Add category labels for top opportunities
    for idx, row in opportunity_df.head(15).iterrows():
        plt.annotate(idx, (row["App_Count"], row["Avg_Installs"]), fontsize=9, ha="center")

    plt.colorbar(scatter, label="Opportunity Score")
    plt.xlabel("Number of Apps (Competition Level)")
    plt.ylabel("Average Installs (Market Demand)")
    plt.title("Market Opportunity Analysis - Top 15 Categories")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "corrected_opportunity_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def analyze_competitiveness_tiers(df):
    """Analyze competitiveness tiers distribution."""
    logger.info("Analyzing competitiveness tiers")

    if "Competitiveness Tier" not in df.columns:
        logger.warning("Competitiveness Tier column not found")
        return None

    # Calculate tier distributions and metrics
    tier_analysis = (
        df.groupby("Competitiveness Tier")
        .agg(
            {
                "App": "count",
                "Rating": "mean",
                "Installs": ["mean", "median"],
                "Price": lambda x: (x > 0).mean() * 100,
            }
        )
        .round(2)
    )

    tier_analysis.columns = [
        "App_Count",
        "Avg_Rating",
        "Avg_Installs",
        "Median_Installs",
        "Percent_Paid",
    ]

    # Create the 4-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Distribution pie chart
    tier_counts = df["Competitiveness Tier"].value_counts()
    tier_percentages = (tier_counts / len(df) * 100).round(1)

    axes[0, 0].pie(
        tier_percentages.values, labels=tier_percentages.index, autopct="%1.1f%%", startangle=90
    )
    axes[0, 0].set_title("App Distribution Across Competition Tiers")

    # 2. Average installs by tier (log scale)
    axes[0, 1].bar(tier_analysis.index, tier_analysis["Avg_Installs"], color="skyblue")
    axes[0, 1].set_title("Average Installs by Competition Tier")
    axes[0, 1].set_ylabel("Average Installs")
    axes[0, 1].set_yscale("log")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 3. Rating comparison
    axes[1, 0].bar(tier_analysis.index, tier_analysis["Avg_Rating"], color="lightcoral")
    axes[1, 0].set_title("Average Rating by Competition Tier")
    axes[1, 0].set_ylabel("Average Rating")
    axes[1, 0].set_ylim(0, 5)
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. Monetization by tier
    axes[1, 1].bar(tier_analysis.index, tier_analysis["Percent_Paid"], color="lightgreen")
    axes[1, 1].set_title("Percentage of Paid Apps by Competition Tier")
    axes[1, 1].set_ylabel("% Paid Apps")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "competitiveness_tiers_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    return tier_analysis


def generate_market_insights_report(market_summary, opportunity_df, tier_analysis):
    """Generate comprehensive market insights report."""
    logger.info("Generating market insights report")

    with open(RESULTS_DIR / "corrected_market_insights.txt", "w") as f:
        f.write("# CORRECTED MARKET OPPORTUNITY ANALYSIS SUMMARY\n")
        f.write("# Based on Business Apps Only (Novelty Apps Removed)\n\n")

        f.write("## 1. Market Saturation Insights\n")
        sorted_saturation = market_summary.sort_values("Avg_Saturation_Index", ascending=False)
        f.write(
            f"- Most saturated categories: {', '.join(sorted_saturation.head(3).index.tolist())}\n"
        )
        f.write(
            f"- Least saturated categories: {', '.join(sorted_saturation.tail(3).index.tolist())}\n"
        )
        f.write(
            f"- Average saturation index: {sorted_saturation['Avg_Saturation_Index'].mean():.2f}\n\n"
        )

        f.write("## 2. TOP OPPORTUNITY CATEGORIES (Based on Corrected Analysis)\n")
        f.write("Categories with highest opportunity scores:\n")
        for i, (idx, row) in enumerate(opportunity_df.head(5).iterrows(), 1):
            f.write(f"{i}. {idx}: Score = {row['Opportunity_Score']:.3f}, ")
            f.write(f"Apps = {row['App_Count']:.0f}, Avg Installs = {row['Avg_Installs']:,.0f}\n")
        f.write("\n")

        f.write("## 3. AVOID CATEGORIES (Oversaturated)\n")
        oversaturated = sorted_saturation.head(5)
        for i, (idx, row) in enumerate(oversaturated.iterrows(), 1):
            f.write(f"{i}. {idx}: Saturation = {row['Avg_Saturation_Index']:.2f}, ")
            f.write(f"Apps = {row['App_Count']:.0f}, Avg Installs = {row['Avg_Installs']:,.0f}\n")
        f.write("\n")

        if tier_analysis is not None:
            f.write("## 4. Competition Tier Analysis\n")
            for tier, metrics in tier_analysis.iterrows():
                f.write(f"- {tier}: {metrics['App_Count']:.0f} apps, ")
                f.write(f"Avg installs: {metrics['Avg_Installs']:,.0f}\n")
            f.write("\n")

        f.write("## 5. KEY INSIGHTS FROM CORRECTED ANALYSIS\n")
        f.write("- Analysis excludes novelty/status symbol apps for realistic business insights\n")
        f.write("- Focus on sustainable business models rather than extreme outliers\n")
        f.write("- Opportunity scores based on realistic market dynamics\n")
        f.write("- Categories recommended align with actual business potential\n")


def main():
    """Execute corrected market opportunity analysis."""
    logger.info("Starting corrected market opportunity analysis")

    # Load cleaned data (business apps only)
    df = load_clean_market_data()

    # Calculate market summary
    market_summary = calculate_market_summary(df)

    # Create main visualizations
    create_main_market_saturation_visualization(df, market_summary)

    # Create saturation rankings
    sorted_saturation = create_saturation_rankings(market_summary)

    # Calculate business opportunity scores
    opportunity_df = calculate_business_opportunity_score(market_summary)

    # Create opportunity visualizations
    create_opportunity_analysis_visualizations(opportunity_df)

    # Analyze competitiveness tiers
    tier_analysis = analyze_competitiveness_tiers(df)

    # Generate insights report
    generate_market_insights_report(market_summary, opportunity_df, tier_analysis)

    # Save all data
    market_summary.to_csv(RESULTS_DIR / "corrected_market_summary.csv")
    opportunity_df.to_csv(RESULTS_DIR / "corrected_opportunity_analysis.csv")
    sorted_saturation.to_csv(RESULTS_DIR / "corrected_saturation_rankings.csv")

    logger.info("Corrected market opportunity analysis completed")
    print("\nCorrected Analysis Complete! Check the following directories:")
    print(f"- Figures: {FIGURES_DIR}")
    print(f"- Results: {RESULTS_DIR}")
    print(f"- Main insights: {RESULTS_DIR}/corrected_market_insights.txt")


if __name__ == "__main__":
    main()
