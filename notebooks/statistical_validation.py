# statistical_validation.py
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "processed_playstore_data.csv")

print("=== STATISTICAL VALIDATION ===\n")

# 1. Test if MEDICAL installs are significantly different from overall
medical_installs = df[df["Category"] == "MEDICAL"]["Installs"]
other_installs = df[df["Category"] != "MEDICAL"]["Installs"]

print("MEDICAL vs Others - Statistical Test:")
print(f"Medical mean: {medical_installs.mean():,.0f}")
print(f"Others mean: {other_installs.mean():,.0f}")

# Use Mann-Whitney U test (non-parametric) due to skewed distribution
statistic, pvalue = stats.mannwhitneyu(medical_installs, other_installs, alternative="less")
print(f"Mann-Whitney U test p-value: {pvalue:.4f}")
print(f"Significantly lower: {'Yes' if pvalue < 0.05 else 'No'}")

# 2. Test if FAMILY has outliers affecting the mean
family_installs = df[df["Category"] == "FAMILY"]["Installs"]

print("\n\nFAMILY Outlier Analysis:")
Q1 = family_installs.quantile(0.25)
Q3 = family_installs.quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 3 * IQR  # Using 3*IQR for extreme outliers

extreme_outliers = family_installs[family_installs > outlier_threshold]
print(f"Number of extreme outliers (>Q3+3*IQR): {len(extreme_outliers)}")
print(f"Percentage of total: {len(extreme_outliers) / len(family_installs) * 100:.2f}%")

# Calculate impact on mean
mean_with_outliers = family_installs.mean()
mean_without_outliers = family_installs[family_installs <= outlier_threshold].mean()
print(f"\nMean with outliers: {mean_with_outliers:,.0f}")
print(f"Mean without extreme outliers: {mean_without_outliers:,.0f}")
print(
    f"Difference: {(mean_with_outliers - mean_without_outliers) / mean_without_outliers * 100:.1f}%"
)

# 3. Check data distribution normality
print("\n\nDistribution Tests:")
for category in ["MEDICAL", "FAMILY"]:
    cat_installs = df[df["Category"] == category]["Installs"]
    # Use log transformation for normality test
    log_installs = np.log1p(cat_installs)

    # Shapiro-Wilk test
    if len(log_installs) > 5000:
        # Sample for large datasets
        stat, p = stats.shapiro(log_installs.sample(5000))
    else:
        stat, p = stats.shapiro(log_installs)

    print(f"\n{category} - Shapiro-Wilk test on log(installs):")
    print(f"Statistic: {stat:.4f}, p-value: {p:.4f}")
    print(f"Normal distribution: {'No' if p < 0.05 else 'Yes'}")
