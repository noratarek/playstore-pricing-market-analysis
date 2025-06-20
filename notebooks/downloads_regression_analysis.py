import logging
import os
from pathlib import Path
import sys  # Add this

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

sys.path.append(str(Path(__file__).resolve().parents[1]))
# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]  # This should point to the root
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures" / "pricing"
RESULTS_DIR = PROJECT_DIR / "reports" / "pricing_analysis"

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


from playstore_analysis.data_loader import PlayStoreDataLoader

# Load processed dataset
"""Load data using the centralized data loader."""
loader = PlayStoreDataLoader(PROCESSED_DATA_DIR)
# Get all required datasets
df = loader.get_main_dataset()
legitimate_df = loader.get_legitimate_datasets()

# Log the dataset sizes
logger.info(f"Loaded datasets - Full: {len(df)}, Legitimate: {len(legitimate_df)}")

# Ensure installs are numeric and positive
df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")
df = df[df["Installs"].notna() & (df["Installs"] > 0)]

# Log-transform installs to reduce skewness
df["log_installs"] = np.log(df["Installs"])

# One-hot encode the Category feature
df = pd.get_dummies(df, columns=["Category"], drop_first=True)

# Define features and target
features = ["Price"] + [col for col in df.columns if col.startswith("Category_")]
# Convert features to numeric and drop any rows with NaNs
X_raw = sm.add_constant(df[features])
y_raw = df["log_installs"]

# Convert to float and drop any non-numeric rows
X = X_raw.apply(pd.to_numeric, errors="coerce").astype("float64")
y = pd.to_numeric(y_raw, errors="coerce").astype("float64")

# Drop rows with any NaNs in predictors or target
valid = X.notnull().all(axis=1) & y.notnull()
X = X[valid]
y = y[valid]

# Now fit the model
model = sm.OLS(y, X).fit()

# Fit OLS regression model
model = sm.OLS(y, X).fit()

# Print summary
print(model.summary())

# Plot coefficients
coef = model.params.sort_values()
plt.figure(figsize=(10, 8))
sns.barplot(x=coef.values, y=coef.index)
plt.title("Regression Coefficients for Log(Installs)")
plt.axvline(0, linestyle="--", color="gray")
plt.tight_layout()
import os

os.makedirs("reports/figures", exist_ok=True)

plt.savefig("reports/figures/regression_coefficients.png")
plt.show()
