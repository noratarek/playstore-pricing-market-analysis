# playstore_analysis/data_loader.py
import pandas as pd
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


class PlayStoreDataLoader:
    """Centralized data loading for all analysis scripts."""

    def __init__(self, processed_data_dir):
        self.processed_data_dir = Path(processed_data_dir)
        self._cache = {}

    def get_main_dataset(self):
        """Load and cache the main processed dataset."""
        if "main_df" not in self._cache:
            logger.info("Loading main processed dataset")
            self._cache["main_df"] = pd.read_csv(
                self.processed_data_dir / "processed_playstore_data.csv"
            )
        return self._cache["main_df"].copy()

    def get_business_datasets(self):
        """Get business-focused datasets (excluding novelty apps)."""
        if "business_df" not in self._cache:
            df = self.get_main_dataset()

            # Filter out novelty apps
            if "is_novelty_app" in df.columns:
                self._cache["business_df"] = df[~df["is_novelty_app"]].copy()
                self._cache["business_paid_df"] = self._cache["business_df"][
                    self._cache["business_df"]["Price"] > 0
                ].copy()
            else:
                # Create novelty flag if it doesn't exist
                rich_pattern = re.compile(
                    r"rich|wealth|millionaire|billionaire|money", re.IGNORECASE
                )
                df["is_novelty_app"] = df.apply(
                    lambda row: bool(rich_pattern.search(str(row["App"]))) and row["Price"] > 20,
                    axis=1,
                )
                self._cache["business_df"] = df[~df["is_novelty_app"]].copy()
                self._cache["business_paid_df"] = self._cache["business_df"][
                    self._cache["business_df"]["Price"] > 0
                ].copy()

        return (self._cache["business_df"].copy(), self._cache["business_paid_df"].copy())

    def get_paid_dataset(self):
        """Get all paid apps (including novelty)."""
        if "paid_df" not in self._cache:
            df = self.get_main_dataset()
            self._cache["paid_df"] = df[df["Price"] > 0].copy()
        return self._cache["paid_df"].copy()

    def get_competition_splits(self):
        """Load pre-calculated competition tier datasets."""
        if "competition_splits" not in self._cache:
            splits = {}
            tier_files = {
                "competitive_leader": "competitive_leader.csv",
                "market_leader": "market_leader.csv",
                "highly_competitive": "highly_competitive.csv",
                "low_competition": "low_competition.csv",
            }

            for tier, filename in tier_files.items():
                path = self.processed_data_dir / filename
                if path.exists():
                    splits[tier] = pd.read_csv(path)
                else:
                    # Calculate from main dataset if split doesn't exist
                    df = self.get_main_dataset()
                    if "Competitiveness Tier" in df.columns:
                        tier_map = {
                            "competitive_leader": "Competitive Leader",
                            "market_leader": "Market Leader",
                            "highly_competitive": "Highly Competitive",
                            "low_competition": "Low Competition",
                        }
                        splits[tier] = df[df["Competitiveness Tier"] == tier_map[tier]].copy()

            self._cache["competition_splits"] = splits

        return self._cache["competition_splits"].copy()

    def get_category_splits(self):
        """Get datasets split by category."""
        if "category_splits" not in self._cache:
            df = self.get_main_dataset()
            top_categories = df["Category"].value_counts().head(10).index

            splits = {}
            for category in top_categories:
                key = f"category_{category.lower().replace(' & ', '_').replace(' ', '_')}"
                splits[key] = df[df["Category"] == category].copy()

            self._cache["category_splits"] = splits

        return self._cache["category_splits"].copy()

    def get_market_summary(self):
        """Get or calculate market summary statistics."""
        if "market_summary" not in self._cache:
            df = self.get_main_dataset()

            market_summary = (
                df.groupby("Category")
                .agg(
                    {
                        "App": "count",
                        "Installs": ["sum", "mean", "median"],
                        "Rating": ["mean", "median"],
                        "Price": lambda x: (x > 0).mean() * 100,  # % paid apps
                        "Market Saturation Index": "mean",
                    }
                )
                .round(2)
            )

            # Flatten column names
            market_summary.columns = [
                "_".join(col).strip() for col in market_summary.columns.values
            ]
            market_summary.rename(
                columns={
                    "App_count": "App_Count",
                    "Installs_sum": "Total_Installs",
                    "Installs_mean": "Avg_Installs",
                    "Installs_median": "Median_Installs",
                    "Rating_mean": "Avg_Rating",
                    "Rating_median": "Median_Rating",
                    "Price_<lambda>": "Percent_Paid",
                    "Market Saturation Index_mean": "Avg_Saturation_Index",
                },
                inplace=True,
            )

            self._cache["market_summary"] = market_summary

        return self._cache["market_summary"].copy()
