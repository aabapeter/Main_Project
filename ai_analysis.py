import pandas as pd
import numpy as np
import scipy.stats as stats

# 1. Load CSV Dataset
file_name = "fact_ai_workplace_raw.csv"
df = pd.read_csv(file_name)
print("Dataset loaded successfully!")

# 2. Clean numeric columns FIRST before any calculations
numeric_cols = [
    "AI_Exposure_Score", "Automation_Potential_Pct", "Augmentation_Potential_Pct", 
    "Estimated_Workers_Thousands", "Avg_Monthly_Wage_INR", "Emerging_Skill_Importance_Score"
]

for col in numeric_cols:
    if col in df.columns:
        # Strip symbols, commas, and percentage signs, then convert to numeric
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("Numeric columns cleaned successfully!")

# 3. Longitudinal Pivot (2019 to 2023)
occ_summary = df.pivot_table(
    index=[
        "NCO_Code", "Occupation_Title", "Industry_Sector", "AI_Exposure_Score", 
        "Exposure_Tier", "Automation_Potential_Pct", "Augmentation_Potential_Pct"
    ],
    columns="Year",
    values=["Estimated_Workers_Thousands", "Avg_Monthly_Wage_INR"],
    aggfunc={"Estimated_Workers_Thousands": "sum", "Avg_Monthly_Wage_INR": "mean"}
).reset_index()

# Flatten column headers
occ_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in occ_summary.columns]

# 4. Calculate Net Growth and Wage Shift
occ_summary["Net_Employment_Growth_Pct"] = (
    (occ_summary["Estimated_Workers_Thousands_2023"] - occ_summary["Estimated_Workers_Thousands_2019"])
    / occ_summary["Estimated_Workers_Thousands_2019"]
) * 100

occ_summary["Wage_Growth_Pct"] = (
    (occ_summary["Avg_Monthly_Wage_INR_2023"] - occ_summary["Avg_Monthly_Wage_INR_2019"])
    / occ_summary["Avg_Monthly_Wage_INR_2019"]
) * 100

# 5. Statistical Correlation Analysis
corr_exposure, p_exp = stats.pearsonr(occ_summary["AI_Exposure_Score"], occ_summary["Net_Employment_Growth_Pct"])
corr_auto, p_auto = stats.pearsonr(occ_summary["Automation_Potential_Pct"], occ_summary["Net_Employment_Growth_Pct"])
corr_aug, p_aug = stats.pearsonr(occ_summary["Augmentation_Potential_Pct"], occ_summary["Net_Employment_Growth_Pct"])

print("\n" + "="*55)
print("STATISTICAL CORRELATION RESULTS:")
print(f"1. AI Exposure vs Net Growth:        r = {corr_exposure:.3f} (p-value: {p_exp:.4f})")
print(f"2. Automation Potential vs Growth:   r = {corr_auto:.3f} (p-value: {p_auto:.4f})")
print(f"3. Augmentation Potential vs Growth: r = {corr_aug:.3f} (p-value: {p_aug:.4f})")
print("="*55)

# 6. Save Aggregated Data for Power BI
occ_summary.to_csv("occupational_growth_analytics.csv", index=False)
print("\nExported: 'occupational_growth_analytics.csv' generated successfully for Power BI!")