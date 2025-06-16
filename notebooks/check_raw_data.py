# check_raw_data.py
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
raw_df = pd.read_csv(PROJECT_DIR / "data" / "raw" / "googleplaystore.csv")

print("=== CHECKING RAW DATA ===")
# Find the problematic app in raw data
problematic_app = "Life Made WI-Fi Touchscreen Photo Frame"
mask = raw_df["App"].str.contains(problematic_app, na=False)
if mask.any():
    print(f"\nFound '{problematic_app}' in raw data:")
    print(raw_df[mask])
    print("\nAll columns for this app:")
    for col in raw_df.columns:
        print(f"{col}: {raw_df[mask][col].values[0]}")

# Check if there's a pattern around this row
idx = raw_df[mask].index[0]
print(f"\n=== Context around row {idx} ===")
print("Previous rows:")
print(raw_df.iloc[max(0, idx - 2) : idx])
print("\nProblematic row:")
print(raw_df.iloc[idx : idx + 1])
print("\nNext rows:")
print(raw_df.iloc[idx + 1 : min(len(raw_df), idx + 3)])

# Check if 1.9 appears in other columns
print("\n=== Checking where '1.9' appears in the dataset ===")
for col in raw_df.columns:
    mask = raw_df[col].astype(str) == "1.9"
    if mask.any():
        print(f"Column '{col}' contains '1.9' in {mask.sum()} rows")
        if col != "Category":
            print("Sample rows where it appears:")
            print(raw_df[mask][["App", col]].head())
