# diagnostic_script.py
import pandas as pd
from pathlib import Path

# Load the data
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "reports" / "market_analysis"  # Correct path

# Load the main dataset
df = pd.read_csv(PROCESSED_DATA_DIR / "processed_playstore_data.csv")

# Check unique categories
print("=== UNIQUE CATEGORIES ===")
print(f"Total unique categories: {df['Category'].nunique()}")
print("\nAll categories:")
for cat in sorted(df["Category"].unique()):
    print(f"  '{cat}' - {len(df[df['Category'] == cat])} apps")

# Check for numeric or strange category names
print("\n=== SUSPICIOUS CATEGORIES ===")
for cat in df["Category"].unique():
    # Check if category is numeric or very short
    if str(cat).replace(".", "").isdigit() or len(str(cat)) < 3:
        print(f"Suspicious category: '{cat}'")
        # Show some apps in this category
        print(f"Sample apps in '{cat}':")
        print(df[df["Category"] == cat][["App", "Category"]].head())
        print()

# Check for inconsistent naming (underscores vs spaces)
print("\n=== NAMING INCONSISTENCIES ===")
categories_with_underscores = [cat for cat in df["Category"].unique() if "_" in str(cat)]
categories_with_spaces = [cat for cat in df["Category"].unique() if " " in str(cat)]

print(f"Categories with underscores: {categories_with_underscores}")
print(f"Categories with spaces: {categories_with_spaces}")

# Load the market summary from the correct location
market_summary_path = RESULTS_DIR / "market_saturation_analysis.csv"
if market_summary_path.exists():
    market_summary = pd.read_csv(market_summary_path, index_col=0)
    print("\n=== MARKET SUMMARY INDEX ===")
    print("Index values (categories):")
    for idx in market_summary.index:
        print(f"  '{idx}'")
else:
    print(f"\nMarket summary file not found at: {market_summary_path}")
