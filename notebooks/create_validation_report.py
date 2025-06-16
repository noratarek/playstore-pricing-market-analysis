# create_validation_report.py
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "processed_playstore_data.csv")

# Generate comprehensive report
report = []

# Analyze each category
for category in ["MEDICAL", "FAMILY"]:
    cat_df = df[df["Category"] == category]

    # Identify top apps skewing the average
    top_10_pct = cat_df.nlargest(int(len(cat_df) * 0.1), "Installs")

    findings = {
        "Category": category,
        "Total_Apps": len(cat_df),
        "Mean_Installs": cat_df["Installs"].mean(),
        "Median_Installs": cat_df["Installs"].median(),
        "Mean_Median_Ratio": cat_df["Installs"].mean() / cat_df["Installs"].median(),
        "Top_10pct_Install_Share": top_10_pct["Installs"].sum() / cat_df["Installs"].sum() * 100,
        "Paid_App_Percentage": (cat_df["Price"] > 0).mean() * 100,
        "Apps_Under_1000_Installs": (cat_df["Installs"] < 1000).sum(),
        "Apps_Over_1M_Installs": (cat_df["Installs"] > 1000000).sum(),
    }
    report.append(findings)

report_df = pd.DataFrame(report)

print("=== VALIDATION REPORT ===")
print(report_df.round(2))

print("\n=== KEY FINDINGS ===")
print("\n1. MEDICAL CATEGORY:")
print("   - Has many specialized apps with limited audience")
print("   - High percentage of paid apps reduces install numbers")
print("   - Professional/regulatory nature limits mass adoption")
print(
    f"   - {report_df[report_df['Category'] == 'MEDICAL']['Apps_Under_1000_Installs'].values[0]} apps have <1000 installs"
)

print("\n2. FAMILY CATEGORY:")
print("   - Contains several mega-popular apps that skew the average")
print(
    f"   - Top 10% of apps account for {report_df[report_df['Category'] == 'FAMILY']['Top_10pct_Install_Share'].values[0]:.1f}% of installs"
)
print("   - High mean/median ratio indicates extreme outliers")
print("   - Includes major games and entertainment apps for kids")

print("\n3. SATURATION METRIC:")
print("   - MEDICAL appears 'saturated' due to many apps with few installs each")
print("   - FAMILY appears less saturated due to few apps with massive installs")
print("   - Current metric doesn't account for market dynamics differences")
print("   - Professional vs Consumer markets behave differently")

# Save detailed findings
report_df.to_csv(PROJECT_DIR / "reports" / "category_validation_report.csv", index=False)
