import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Dataset Load cheyyuka
try:
    df = pd.read_csv("fact_ai_workplace_raw.csv")
except Exception:
    df = pd.read_excel("MainDatasetEXL.xlsx")

# Data Cleaning & Conversion
df["Wage_Num"] = (
    df["Avg_Monthly_Wage_INR"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.strip()
)
df["Wage_Num"] = pd.to_numeric(df["Wage_Num"], errors="coerce")

for col in ["AI_Exposure_Score", "Automation_Potential_Pct", "Augmentation_Potential_Pct", "Estimated_Workers_Thousands"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Longitudinal Summary Dataframe (49 Occupations)
occ_summary = df.pivot_table(
    index=["NCO_Code", "Occupation_Title", "Industry_Sector", "AI_Exposure_Score", 
           "Exposure_Tier", "Automation_Potential_Pct", "Augmentation_Potential_Pct"],
    columns="Year",
    values=["Estimated_Workers_Thousands", "Wage_Num"],
    aggfunc={"Estimated_Workers_Thousands": "sum", "Wage_Num": "mean"}
).reset_index()
occ_summary.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in occ_summary.columns]

# Target Variable (y): 4-Year Net Employment Growth %
occ_summary["Net_Growth_Pct"] = (
    (occ_summary["Estimated_Workers_Thousands_2023"] - occ_summary["Estimated_Workers_Thousands_2019"])
    / occ_summary["Estimated_Workers_Thousands_2019"]
) * 100

# 2. Features (X) & Target (y)
features = ["AI_Exposure_Score", "Automation_Potential_Pct", "Augmentation_Potential_Pct"]
X = occ_summary[features].fillna(0)
y = occ_summary["Net_Growth_Pct"].fillna(0)

# 3. 80:20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 6. Evaluation Metrics Calculation
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Results Display
print("=" * 60)
print(" MACHINE LEARNING MODEL EVALUATION & VALIDATION METRICS ")
print("=" * 60)
print(f"Total Occupations Analyzed : {len(occ_summary)}")
print(f"Training Samples (80%)     : {len(X_train)}")
print(f"Testing Samples (20%)      : {len(X_test)}")
print("-" * 60)
print(f"Train R² Score             : {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"Test R² Score              : {test_r2:.4f} ({test_r2*100:.2f}%)")
print("-" * 60)
print(f"Mean Absolute Error (MAE)  : {test_mae:.2f}%")
print(f"Root Mean Squared Error    : {test_rmse:.2f}%")
print("=" * 60)

# Regression Equation
coef = model.coef_
print("\nLearned Model Equation (Regression Coefficients):")
print(f"Growth % = {model.intercept_:.2f} + ({coef[0]:.2f} × AI_Exposure) + ({coef[1]:.2f} × Automation_Pct) + ({coef[2]:.2f} × Augmentation_Pct)")