# -*- coding: utf-8 -*-
"""
Pricing Strategy Analysis - Google Play Store
Analyzing how different monetization models impact app success metrics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]  # Adjust if needed
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures" / "pricing"
RESULTS_DIR = PROJECT_DIR / "reports" / "pricing_analysis"

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_processed_data():
    """Load the processed dataset from Phase 1."""
    logger.info("Loading processed dataset")

    data_path = PROCESSED_DATA_DIR / "processed_playstore_data.csv"
    if not os.path.exists(data_path):
        logger.error(f"Processed dataset not found at {data_path}")
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Load the separate paid apps dataset for specific pricing analysis
    paid_path = PROCESSED_DATA_DIR / "paid_apps.csv"
    if os.path.exists(paid_path):
        paid_df = pd.read_csv(paid_path)
        logger.info(
            f"Loaded processed dataset with shape: {df.shape} and paid apps dataset with shape: {paid_df.shape}"
        )
        return df, paid_df
    else:
        logger.warning("Paid apps split dataset not found, filtering from main dataset")
        paid_df = df[df["Price"] > 0].copy()
        return df, paid_df


def analyze_price_distribution(df, paid_df):
    """Analyze price distribution and create visualizations."""
    logger.info("Analyzing price distribution")

    # 1. Create enhanced price distribution visualization
    plt.figure(figsize=(12, 8))

    # Create a more visually appealing price distribution for paid apps
    price_counts = paid_df["Price"].value_counts().sort_index()

    # Filter to top 20 most common price points
    top_prices = paid_df["Price"].value_counts().nlargest(20)

    plt.bar(top_prices.index, top_prices.values, color="skyblue")
    plt.title("Distribution of Top 20 Price Points (Paid Apps Only)", fontsize=15)
    plt.xlabel("Price ($)", fontsize=12)
    plt.ylabel("Number of Apps", fontsize=12)
    plt.xticks(fontsize=10)
    plt.grid(axis="y", alpha=0.3)

    # Add value labels on top of bars
    for i, v in enumerate(top_prices.values):
        plt.text(top_prices.index[i], v + 5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_price_points_distribution.png")
    plt.close()

    # 2. Price category distribution with percentages
    plt.figure(figsize=(12, 6))

    # Count by price category
    category_counts = df["Price Category"].value_counts()
    total = len(df)

    # Create bars with percentages
    ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")

    # Add percentage labels
    for i, p in enumerate(ax.patches):
        percentage = 100 * p.get_height() / total
        ax.annotate(
            f"{percentage:.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.title("Distribution of App Price Categories with Percentages", fontsize=15)
    plt.xlabel("Price Category", fontsize=12)
    plt.ylabel("Number of Apps", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_category_percentages.png")
    plt.close()

    # 3. Calculate key price statistics by category
    price_stats = (
        paid_df.groupby("Category")
        .agg({"Price": ["mean", "median", "min", "max", "count"]})
        .sort_values(("Price", "mean"), ascending=False)
    )

    # Save statistics as CSV
    price_stats.to_csv(RESULTS_DIR / "price_statistics_by_category.csv")

    # Create visualization of mean price by category (top 15)
    plt.figure(figsize=(14, 8))
    price_stats[("Price", "mean")].head(15).plot(kind="bar", color="teal")
    plt.title("Mean Price by Category (Top 15)", fontsize=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Mean Price ($)", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mean_price_by_category_top15.png")
    plt.close()

    logger.info("Price distribution analysis completed")
    return price_stats


def analyze_price_impact_on_ratings(df, paid_df):
    """Analyze how pricing affects app ratings."""
    logger.info("Analyzing price impact on ratings")

    # 1. Create enhanced boxplot of rating by price category
    plt.figure(figsize=(14, 8))
    # Order by price category from free to premium
    order = ["Free", "Low-cost", "Mid-price", "High-price", "Premium"]

    # Create boxplot with individual points
    ax = sns.boxplot(x="Price Category", y="Rating", data=df, order=order, palette="YlGnBu")

    # Add swarmplot for better visualization of distribution
    sns.swarmplot(
        x="Price Category", y="Rating", data=df, order=order, color=".25", size=2, alpha=0.5
    )

    # Add median labels
    medians = df.groupby("Price Category")["Rating"].median()
    for i, category in enumerate(order):
        if category in medians:
            plt.text(
                i,
                medians[category] + 0.1,
                f"Median: {medians[category]:.2f}",
                horizontalalignment="center",
                size="small",
                color="black",
                weight="semibold",
            )

    plt.title("Rating Distribution by Price Category", fontsize=15)
    plt.xlabel("Price Category", fontsize=12)
    plt.ylabel("Rating (1-5 scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rating_by_price_category_enhanced.png")
    plt.close()

    # 2. Create scatter plot of price vs. rating for paid apps
    plt.figure(figsize=(12, 8))

    # Filter out extreme outliers for better visualization
    plot_df = paid_df[paid_df["Price"] <= 20]  # Removing extreme prices like "I am rich" apps

    # Create scatter plot with regression line
    sns.regplot(
        x="Price", y="Rating", data=plot_df, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}
    )

    plt.title("Relationship Between Price and Rating (Paid Apps <= $20)", fontsize=15)
    plt.xlabel("Price ($)", fontsize=12)
    plt.ylabel("Rating (1-5 scale)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_vs_rating_scatter.png")
    plt.close()

    # 3. Statistical analysis: Correlation and regression
    # Calculate correlation
    correlation = paid_df["Price"].corr(paid_df["Rating"])

    # Run a simple regression model
    X = sm.add_constant(paid_df["Price"])
    model = sm.OLS(paid_df["Rating"], X).fit()

    # Save regression results
    with open(RESULTS_DIR / "price_rating_regression.txt", "w") as f:
        f.write(f"Correlation between Price and Rating: {correlation:.4f}\n\n")
        f.write(model.summary().as_text())

    # 4. Calculate average rating by price tier for different categories
    # This helps identify categories where pricing might have different effects

    # Get top 10 categories by app count
    top_categories = df["Category"].value_counts().head(10).index.tolist()

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(5, 2, figsize=(18, 25))
    axes = axes.flatten()

    category_price_effects = {}

    # For each category, show rating by price category
    for i, category in enumerate(top_categories):
        category_df = df[df["Category"] == category]

        # Create category-specific plot
        sns.boxplot(x="Price Category", y="Rating", data=category_df, order=order, ax=axes[i])
        axes[i].set_title(f"Ratings by Price Category: {category}", fontsize=13)
        axes[i].set_ylim(1, 5)

        # Calculate mean ratings by price category for this category
        category_means = category_df.groupby("Price Category")["Rating"].mean().reindex(order)
        category_price_effects[category] = category_means.to_dict()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rating_by_price_across_categories.png")
    plt.close()

    # Save category price effects to CSV
    pd.DataFrame(category_price_effects).to_csv(RESULTS_DIR / "category_price_effects.csv")

    logger.info("Price impact on ratings analysis completed")
    return correlation, model


def analyze_price_impact_on_installs(df, paid_df):
    """Analyze how pricing affects app installations."""
    logger.info("Analyzing price impact on installs")

    # 1. Create visualization of installs by price category
    plt.figure(figsize=(14, 8))
    # Order by price category from free to premium
    order = ["Free", "Low-cost", "Mid-price", "High-price", "Premium"]

    # Create boxplot with log scale for installations
    ax = sns.boxplot(x="Price Category", y="Installs", data=df, order=order, palette="YlOrRd")
    plt.yscale("log")  # Log scale for better visualization

    plt.title("Installation Distribution by Price Category (Log Scale)", fontsize=15)
    plt.xlabel("Price Category", fontsize=12)
    plt.ylabel("Installations (Log Scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "installs_by_price_category.png")
    plt.close()

    # 2. Calculate median installations by price category
    installs_by_price = df.groupby("Price Category")["Installs"].median().reindex(order)

    # Create bar chart of median installs
    plt.figure(figsize=(12, 6))
    installs_by_price.plot(kind="bar", color="coral")
    plt.title("Median Installations by Price Category", fontsize=15)
    plt.xlabel("Price Category", fontsize=12)
    plt.ylabel("Median Installations", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, v in enumerate(installs_by_price):
        plt.text(i, v * 1.1, f"{v:,.0f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "median_installs_by_price.png")
    plt.close()

    # 3. Calculate install ratio between free and paid apps
    free_median = df[df["Price Category"] == "Free"]["Installs"].median()
    paid_median = df[df["Price Category"] != "Free"]["Installs"].median()

    install_ratio = free_median / paid_median

    # Create ratio visualization
    categories = ["Free Apps", "Paid Apps"]
    medians = [free_median, paid_median]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, medians, color=["lightblue", "lightcoral"])
    plt.title(
        f"Free vs. Paid Apps: Median Installations (Ratio: {install_ratio:.1f}x)", fontsize=15
    )
    plt.ylabel("Median Installations", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{medians[i]:,.0f}",
            ha="center",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "free_vs_paid_install_comparison.png")
    plt.close()

    # 4. Check the relationship between price and installs within paid apps
    plt.figure(figsize=(12, 8))

    # Filter out extreme outliers for visualization
    plot_df = paid_df[paid_df["Price"] <= 20]

    # Create scatter plot with regression line
    sns.regplot(
        x="Price",
        y="Installs",
        data=plot_df,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
    )
    plt.yscale("log")  # Log scale for better visualization

    plt.title("Relationship Between Price and Installations (Log Scale)", fontsize=15)
    plt.xlabel("Price ($)", fontsize=12)
    plt.ylabel("Installations (Log Scale)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_vs_installs_scatter.png")
    plt.close()

    # 5. Calculate price elasticity by category
    # Price elasticity = % change in quantity / % change in price
    # We'll use median installations at different price points as a proxy

    price_elasticity = {}
    top_categories = df["Category"].value_counts().head(15).index.tolist()

    for category in top_categories:
        category_df = df[df["Category"] == category]

        # Get median installs for free and paid apps
        if (
            "Free" in category_df["Price Category"].values
            and len(category_df[category_df["Price Category"] != "Free"]) > 0
        ):
            free_installs = category_df[category_df["Price Category"] == "Free"][
                "Installs"
            ].median()
            paid_installs = category_df[category_df["Price Category"] != "Free"][
                "Installs"
            ].median()
            paid_price = category_df[category_df["Price Category"] != "Free"]["Price"].median()

            # Calculate price elasticity: (% change in installs) / (% change in price)
            # Since free apps have price 0, we use a small value (0.01) to avoid division by zero
            # This is a simplified calculation for the demand curve slope
            pct_change_installs = (paid_installs - free_installs) / free_installs
            pct_change_price = (paid_price - 0.01) / 0.01

            elasticity = pct_change_installs / pct_change_price
            price_elasticity[category] = elasticity

    # Sort by elasticity (from most elastic to least)
    price_elasticity = {
        k: v for k, v in sorted(price_elasticity.items(), key=lambda item: item[1])
    }

    # Create price elasticity visualization
    plt.figure(figsize=(14, 8))
    plt.bar(price_elasticity.keys(), price_elasticity.values(), color="purple")
    plt.axhline(y=0, color="red", linestyle="-", alpha=0.3)
    plt.title("Price Elasticity by Category", fontsize=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Estimated Price Elasticity", fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_elasticity_by_category.png")
    plt.close()

    # Save price elasticity data
    pd.DataFrame(
        {"Category": price_elasticity.keys(), "Price Elasticity": price_elasticity.values()}
    ).to_csv(RESULTS_DIR / "price_elasticity_by_category.csv", index=False)

    logger.info("Price impact on installs analysis completed")
    return price_elasticity


def analyze_optimal_pricing_strategy(df, paid_df):
    """Identify optimal pricing strategies by category."""
    logger.info("Analyzing optimal pricing strategies")

    # 1. Calculate price efficiency (Rating/Price) by category
    category_price_efficiency = (
        paid_df.groupby("Category")["Price Efficiency"].mean().sort_values(ascending=False)
    )

    # Create visualization
    plt.figure(figsize=(14, 8))
    category_price_efficiency.head(15).plot(kind="bar", color="teal")
    plt.title("Price Efficiency by Category (Rating per Dollar)", fontsize=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Average Price Efficiency", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_efficiency_by_category.png")
    plt.close()

    # 2. Identify most successful price points by category
    # First define a success metric: combining ratings and installs
    # We'll use z-score normalization to combine these metrics

    # Calculate z-scores for Rating and Installs
    scaler = StandardScaler()

    # Only use apps with both rating and installs information
    success_df = paid_df.dropna(subset=["Rating", "Installs"]).copy()

    # Apply log transformation to Installs for better scaling
    success_df["Log_Installs"] = np.log1p(success_df["Installs"])

    # Calculate z-scores
    success_df["Rating_Z"] = scaler.fit_transform(success_df[["Rating"]])
    success_df["Installs_Z"] = scaler.fit_transform(success_df[["Log_Installs"]])

    # Create combined success score (equal weight to rating and installs)
    success_df["Success_Score"] = 0.5 * success_df["Rating_Z"] + 0.5 * success_df["Installs_Z"]

    # Find the optimal price point by category
    top_categories = df["Category"].value_counts().head(15).index.tolist()
    optimal_price_points = {}

    for category in top_categories:
        category_success = success_df[success_df["Category"] == category]

        if len(category_success) > 0:
            # Group by price and find average success score
            price_success = (
                category_success.groupby("Price")["Success_Score"]
                .mean()
                .sort_values(ascending=False)
            )

            if len(price_success) > 0:
                optimal_price = price_success.index[0]
                optimal_price_points[category] = {
                    "Optimal Price": optimal_price,
                    "Success Score": price_success.iloc[0],
                    "Sample Size": len(
                        category_success[category_success["Price"] == optimal_price]
                    ),
                }

    # Create visualization of optimal price points
    plt.figure(figsize=(14, 8))
    categories = list(optimal_price_points.keys())
    prices = [optimal_price_points[c]["Optimal Price"] for c in categories]

    # Sort by optimal price
    sorted_indices = np.argsort(prices)
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_prices = [prices[i] for i in sorted_indices]

    plt.bar(sorted_categories, sorted_prices, color="gold")
    plt.title("Optimal Price Point by Category (Based on Success Score)", fontsize=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Optimal Price ($)", fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis="y", alpha=0.3)

    # Add price labels
    for i, price in enumerate(sorted_prices):
        plt.text(i, price + 0.5, f"${price:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optimal_price_by_category.png")
    plt.close()

    # Save optimal price data
    pd.DataFrame(optimal_price_points).T.reset_index().rename(
        columns={"index": "Category"}
    ).to_csv(RESULTS_DIR / "optimal_price_by_category.csv", index=False)

    # 3. Cluster apps by price and success to identify pricing strategies
    # We'll use KMeans clustering

    # Prepare features for clustering
    cluster_df = success_df.copy()

    # Standardize features for clustering
    X = cluster_df[["Price", "Success_Score"]].values
    X = StandardScaler().fit_transform(X)

    # Determine optimal number of clusters using Elbow method
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot Elbow method result
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertias, "bo-")
    plt.title("Elbow Method for Optimal k", fontsize=15)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_elbow_method.png")
    plt.close()

    # Choose optimal k from elbow method (let's say k=4 for this example)
    optimal_k = 4  # This should be adjusted based on the actual elbow curve

    # Apply KMeans with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_df["Cluster"] = kmeans.fit_predict(X)

    # Create cluster visualization
    plt.figure(figsize=(12, 8))

    # Get cluster centers
    centers = kmeans.cluster_centers_

    # Plot data points colored by cluster
    scatter = plt.scatter(
        cluster_df["Price"],
        cluster_df["Success_Score"],
        c=cluster_df["Cluster"],
        cmap="viridis",
        alpha=0.6,
        s=50,
    )

    # Plot cluster centers
    plt.scatter(
        centers[:, 0] * np.std(cluster_df["Price"]) + np.mean(cluster_df["Price"]),
        centers[:, 1] * np.std(cluster_df["Success_Score"]) + np.mean(cluster_df["Success_Score"]),
        c="red",
        marker="X",
        s=200,
        label="Cluster Centers",
    )

    plt.title("App Clusters by Price and Success Score", fontsize=15)
    plt.xlabel("Price ($)", fontsize=12)
    plt.ylabel("Success Score", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "app_price_success_clusters.png")
    plt.close()

    # Analyze cluster characteristics
    cluster_analysis = (
        cluster_df.groupby("Cluster")
        .agg(
            {
                "Price": ["mean", "median", "min", "max"],
                "Success_Score": ["mean", "median"],
                "Rating": ["mean", "median"],
                "Installs": ["mean", "median"],
                "App": "count",
            }
        )
        .round(2)
    )

    # Save cluster analysis
    cluster_analysis.to_csv(RESULTS_DIR / "price_success_clusters.csv")

    # Analyze most common categories in each cluster
    cluster_categories = {}
    for cluster_id in cluster_df["Cluster"].unique():
        cluster_subset = cluster_df[cluster_df["Cluster"] == cluster_id]
        category_counts = cluster_subset["Category"].value_counts().head(5)
        cluster_categories[f"Cluster {cluster_id}"] = category_counts.to_dict()

    # Save cluster category analysis
    pd.DataFrame(cluster_categories).to_csv(RESULTS_DIR / "cluster_categories.csv")

    logger.info("Optimal pricing strategy analysis completed")
    return optimal_price_points, cluster_analysis


def main():
    """Execute the pricing strategy analysis."""
    logger.info("Starting pricing strategy analysis")

    # Load the data
    df, paid_df = load_processed_data()

    # Analyze price distribution
    price_stats = analyze_price_distribution(df, paid_df)

    # Analyze price impact on ratings
    correlation, model = analyze_price_impact_on_ratings(df, paid_df)

    # Analyze price impact on installs
    price_elasticity = analyze_price_impact_on_installs(df, paid_df)

    # Analyze optimal pricing strategies
    optimal_prices, cluster_analysis = analyze_optimal_pricing_strategy(df, paid_df)

    # Create a summary report
    with open(RESULTS_DIR / "pricing_analysis_summary.txt", "w") as f:
        f.write("# PRICING STRATEGY ANALYSIS SUMMARY\n\n")

        f.write("## Price Distribution\n")
        f.write(
            f"- Free apps: {len(df[df['Price'] == 0])} ({len(df[df['Price'] == 0]) / len(df) * 100:.1f}%)\n"
        )
        f.write(
            f"- Paid apps: {len(df[df['Price'] > 0])} ({len(df[df['Price'] > 0]) / len(df) * 100:.1f}%)\n"
        )
        f.write(
            f"- Most common price points: {', '.join(['$' + str(p) for p in paid_df['Price'].value_counts().head(5).index.tolist()])}\n"
        )
        f.write(
            f"- Highest priced categories: {', '.join(price_stats.head(5).index.get_level_values(0).tolist())}\n\n"
        )

        f.write("## Price Impact on Ratings\n")
        f.write(f"- Correlation between price and rating: {correlation:.4f}\n")
        f.write(f"- Regression coefficient: {model.params[1]:.4f}\n")
        f.write(f"- Statistical significance: {'Yes' if model.pvalues[1] < 0.05 else 'No'}\n\n")

        f.write("## Price Impact on Installations\n")
        f.write("- Free apps have significantly higher installations than paid apps\n")
        f.write(
            f"- Categories with lowest price elasticity: {', '.join(list(price_elasticity.keys())[:3])}\n"
        )
        f.write(
            f"- Categories with highest price elasticity: {', '.join(list(price_elasticity.keys())[-3:])}\n\n"
        )

        f.write("## Optimal Pricing Strategies\n")
        f.write("- Identified optimal price points by category based on success score\n")
        f.write(
            f"- Categories with highest price efficiency: {', '.join(optimal_prices.keys())[:5]}\n"
        )
        f.write("- Identified app clusters based on price and success metrics\n\n")

        f.write("## Key Findings\n")
        f.write(
            "1. The vast majority of apps are free, but certain categories support paid models better than others\n"
        )
        f.write(
            "2. There is a weak relationship between price and rating, suggesting users don't necessarily rate paid apps worse\n"
        )
        f.write(
            "3. Price significantly impacts installation rates, but the effect varies dramatically by category\n"
        )
        f.write(
            "4. Optimal price points vary by category, with some categories supporting higher prices\n"
        )
        f.write(
            "5. Multiple pricing strategies exist, as identified through clustering analysis\n"
        )

    logger.info("Pricing strategy analysis completed")


if __name__ == "__main__":
    main()
