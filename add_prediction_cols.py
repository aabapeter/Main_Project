import pandas as pd

# File load cheyyunnu
file_name = "occupational_growth_analytics.csv"
df = pd.read_csv(file_name)

# ML Regression formula vachulla calculations:
# Growth % = -63.85 + (95.48 * AI_Exposure) - (0.62 * Automation) + (0.62 * Augmentation)
pred_growth = (
    -63.85 
    + (95.48 * df["AI_Exposure_Score"]) 
    - (0.62 * df["Automation_Potential_Pct"]) 
    + (0.62 * df["Augmentation_Potential_Pct"])
)

# Puthiya 3 columns add cheyyunnu
df["Predicted_Growth_Pct_2027"] = pred_growth.round(2)
df["Projected_Workers_Thousands_2025"] = (df["Estimated_Workers_Thousands_2023"] * (1.0 + pred_growth / 200.0)).round(1)
df["Projected_Workers_Thousands_2027"] = (df["Estimated_Workers_Thousands_2023"] * (1.0 + pred_growth / 100.0)).round(1)

# Athe file-ilekku thanne update aakki save cheyyunnu
df.to_csv(file_name, index=False)

print("SUCCESS: occupational_growth_analytics.csv successfully updated with 2027 predictions!")
print("New columns:", df.columns.tolist()[-3:])