import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns  # <--- This is the library you asked for

# --- STEP 1: GENERATE PRO-LEVEL DATA ---
print("Generating 25 years of economic cycles...")
np.random.seed(42)
dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='MS')
n = len(dates)

# Structural Trends & Shocks
trend = np.linspace(0, 10, n)
crisis_2008 = -5 * np.exp(-0.01 * (np.arange(n) - 100)**2)
oil_crash_2014 = -3 * np.exp(-0.005 * (np.arange(n) - 170)**2)
covid_19 = -8 * np.exp(-0.1 * (np.arange(n) - 240)**2)

# Features (Macro Variables)
pcepi = 80 + trend * 4 + np.random.normal(0, 0.5, n)  # Inflation
# Note: GDP included to show correlation in heatmap, Ridge will handle the collinearity
gdp = 10000 + trend * 500 + (crisis_2008 + covid_19) * 200 + np.random.normal(0, 50, n)
unrate = 5 - (crisis_2008 + covid_19) * 1.5 + np.random.normal(0, 0.3, n)
fedfunds = 3 + 2 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 0.2, n)
mcoilwtico = 60 + 20 * np.sin(np.linspace(0, 6*np.pi, n)) + (oil_crash_2014 + covid_19) * 5 + np.random.normal(0, 5, n)

# Target: XLE (Energy Sector)
xle = 30 + (mcoilwtico * 0.8) - (fedfunds * 1.5) + (gdp * 0.0005) + np.random.normal(0, 3, n)

df = pd.DataFrame({
    'XLE': xle,
    'PCEPI': pcepi,
    'GDP': gdp,
    'UNRATE': unrate,
    'FEDFUNDS': fedfunds,
    'MCOILWTICO': mcoilwtico
}, index=dates)

# --- STEP 2: SCALING & VIF ---
print("Scaling data and checking Multicollinearity...")
features = ['PCEPI', 'GDP', 'UNRATE', 'FEDFUNDS', 'MCOILWTICO']
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# VIF Calculation
vif_data = pd.DataFrame()
vif_data["feature"] = features
vif_data["VIF"] = [variance_inflation_factor(df_scaled[features].values, i) for i in range(len(features))]

# --- STEP 3: RIDGE REGRESSION ---
print("Training Ridge Regression (L2 Regularization)...")
# Lag predictors by 1 month
df_lagged = df_scaled.copy()
df_lagged[features] = df_lagged[features].shift(1)
df_lagged = df_lagged.dropna()

X = df_lagged[features]
y = df_lagged['XLE']

model = Ridge(alpha=1.0)
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Model Accuracy (R2): {r2:.4f}")

# --- STEP 4: VISUALIZATION WITH SEABORN ---
print("Generating plots (saving to xle_seaborn_analysis.png)...")

# Set Seaborn Style
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot A: Actual vs Predicted
axes[0, 0].plot(df_lagged.index, y, label='Actual XLE', color='#1f77b4', lw=1.5)
axes[0, 0].plot(df_lagged.index, y_pred, label='Predicted', color='#ff7f0e', linestyle='--', lw=2)
axes[0, 0].set_title(f'Prediction Model (R2: {r2:.2f})', fontsize=12)
axes[0, 0].legend()

# Plot B: Feature Importance (Coefficients)
coefs = pd.Series(model.coef_, index=features).sort_values()
sns.barplot(x=coefs.values, y=coefs.index, ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title('Feature Importance (Impact on Price)', fontsize=12)

# Plot C: Correlation Heatmap (THE SEABORN PART)
corr_matrix = df_scaled[features + ['XLE']].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1, 0], cbar=True)
axes[1, 0].set_title('Correlation Heatmap', fontsize=12)

# Plot D: VIF Scores
sns.barplot(x='feature', y='VIF', data=vif_data, ax=axes[1, 1], palette="magma")
axes[1, 1].set_title('Multicollinearity Check (VIF Scores)', fontsize=12)
axes[1, 1].axhline(y=10, color='r', linestyle='--', label='Danger Zone (>10)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('xle_seaborn_analysis.png')
print("Done. Check 'xle_seaborn_analysis.png' for results.")