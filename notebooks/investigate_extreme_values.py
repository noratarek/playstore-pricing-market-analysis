# investigate_extreme_values.py
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "processed_playstore_data.csv")

print("=== INVESTIGATING EXTREME VALUES ===\n")

# 1. Check FAMILY category for extreme apps
print("FAMILY CATEGORY - EXTREME APPS:")
family_df = df[df["Category"] == "FAMILY"]

# Find apps with extremely high installs
extreme_threshold = family_df["Installs"].quantile(0.99)
extreme_apps = family_df[family_df["Installs"] > extreme_threshold]

print(f"\nApps with installs > 99th percentile ({extreme_threshold:,.0f}):")
print(
    extreme_apps[["App", "Installs", "Rating", "Price", "Genres"]].sort_values(
        "Installs", ascending=False
    )
)

# Check if there are any data quality issues
print(f"\nChecking for potential data issues in FAMILY:")
print(f"Apps with installs > 1 billion: {(family_df['Installs'] > 1e9).sum()}")
print(
    f"Apps with suspiciously round install numbers: {(family_df['Installs'] % 1000000 == 0).sum()}"
)

# 2. Check MEDICAL category for why installs are low
print("\n\nMEDICAL CATEGORY - LOW INSTALL ANALYSIS:")
medical_df = df[df["Category"] == "MEDICAL"]

print(f"\nInstall distribution in MEDICAL:")
bins = [0, 100, 1000, 10000, 100000, 1000000, float("inf")]
labels = ["0-100", "100-1K", "1K-10K", "10K-100K", "100K-1M", "1M+"]
medical_df["Install_Range"] = pd.cut(medical_df["Installs"], bins=bins, labels=labels)
print(medical_df["Install_Range"].value_counts().sort_index())

print(f"\nPaid vs Free in MEDICAL:")
print(
    f"Free apps: {(medical_df['Price'] == 0).sum()} ({(medical_df['Price'] == 0).mean() * 100:.1f}%)"
)
print(
    f"Paid apps: {(medical_df['Price'] > 0).sum()} ({(medical_df['Price'] > 0).mean() * 100:.1f}%)"
)
print(
    f"Average price of paid medical apps: ${medical_df[medical_df['Price'] > 0]['Price'].mean():.2f}"
)

# 3. Compare with similar professional categories
print("\n\nCOMPARING PROFESSIONAL CATEGORIES:")
professional_categories = ["MEDICAL", "BUSINESS", "FINANCE", "EDUCATION"]

comparison_df = []
for cat in professional_categories:
    cat_data = df[df["Category"] == cat]
    comparison_df.append(
        {
            "Category": cat,
            "App_Count": len(cat_data),
            "Avg_Installs": cat_data["Installs"].mean(),
            "Median_Installs": cat_data["Installs"].median(),
            "Paid_Percentage": (cat_data["Price"] > 0).mean() * 100,
            "Avg_Price_Paid": cat_data[cat_data["Price"] > 0]["Price"].mean()
            if (cat_data["Price"] > 0).any()
            else 0,
            "Avg_Rating": cat_data["Rating"].mean(),
        }
    )

comparison_df = pd.DataFrame(comparison_df)
print(comparison_df.round(2))

# 4. Check for category misclassification
print("\n\nCHECKING FOR POTENTIAL MISCLASSIFICATION:")

# Look for medical-related apps in FAMILY
family_medical_keywords = ["health", "medical", "doctor", "pregnancy", "baby health"]
potential_medical_in_family = family_df[
    family_df["App"].str.lower().str.contains("|".join(family_medical_keywords), na=False)
]
print(f"\nPotential medical apps in FAMILY category: {len(potential_medical_in_family)}")
if len(potential_medical_in_family) > 0:
    print("Examples:")
    print(potential_medical_in_family[["App", "Installs"]].head())

# Look for family/kids apps in other categories
other_categories = df[df["Category"] != "FAMILY"]
family_keywords = ["kids", "children", "family", "parental", "baby", "toddler"]
potential_family_elsewhere = other_categories[
    other_categories["App"].str.lower().str.contains("|".join(family_keywords), na=False)
]
print(f"\nPotential family apps in other categories: {len(potential_family_elsewhere)}")
print(potential_family_elsewhere["Category"].value_counts().head())
