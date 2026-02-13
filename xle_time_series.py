import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- STEP 1: GENERATE DATA ---
print("Initializing model data...")
np.random.seed(42)
dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='MS')
n = len(dates)

# Economic Trends (Simulated)
trend = np.linspace(0, 10, n)
crisis_2008 = -5 * np.exp(-0.01 * (np.arange(n) - 100)**2)
covid_19 = -8 * np.exp(-0.1 * (np.arange(n) - 240)**2)

# Variables
pcepi = 80 + trend * 4 + np.random.normal(0, 0.5, n)
gdp = 10000 + trend * 500 + (crisis_2008 + covid_19) * 200 + np.random.normal(0, 50, n)
unrate = 5 - (crisis_2008 + covid_19) * 1.5 + np.random.normal(0, 0.3, n)
fedfunds = 3 + 2 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 0.2, n)
mcoilwtico = 60 + 20 * np.sin(np.linspace(0, 6*np.pi, n)) + (covid_19 * 5) + np.random.normal(0, 5, n)

# Target
xle = 30 + (mcoilwtico * 0.8) - (fedfunds * 1.5) + (gdp * 0.0005) + np.random.normal(0, 3, n)

df = pd.DataFrame({
    'XLE': xle, 'PCEPI': pcepi, 'GDP': gdp, 
    'UNRATE': unrate, 'FEDFUNDS': fedfunds, 'MCOILWTICO': mcoilwtico
}, index=dates)

# --- STEP 2: PCA TRANSFORMATION (The Clean-Up) ---
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Combine GDP & PCEPI into one 'Econ_Trend'
pca = PCA(n_components=1)
pc1 = pca.fit_transform(df_scaled[['GDP', 'PCEPI']])
df_scaled['ECON_TREND_PC1'] = pc1 

# --- STEP 3: TRAIN MODEL ---
features = ['ECON_TREND_PC1', 'UNRATE', 'FEDFUNDS', 'MCOILWTICO']
df_lagged = df_scaled.copy()
df_lagged[features] = df_lagged[features].shift(1) # 1-Month Lag
df_lagged = df_lagged.dropna()

X = df_lagged[features]
y = df_lagged['XLE'] # Target (Scaled)

model = Ridge(alpha=1.0)
model.fit(X, y)
y_pred = model.predict(X)

# Inverse transform y_pred to get Real Prices ($) for the graph
# We have to do a little trickery since scaler was fit on all cols
y_actual_dollars = df.loc[y.index, 'XLE']
# Approximate rescale (or just plot scaled to see the fit quality)
# For simplicity and visual clarity, we will plot the Scaled Trends which align perfectly

# --- STEP 4: GENERATE THE TIME SERIES PLOT ---
print("Plotting Final Time Series...")
sns.set_theme(style="whitegrid") # Pro styling

plt.figure(figsize=(14, 7))

# 1. Plot Actual
plt.plot(y.index, y, color='#333333', linewidth=2.5, alpha=0.8, label='Actual Price (Scaled)')

# 2. Plot Predicted
plt.plot(y.index, y_pred, color='#FF5733', linewidth=2, linestyle='--', label='PCA Model Prediction')

# 3. Styling
plt.title(f'XLE Price Prediction Model (PCA-Enhanced)\nAccuracy (R²): {r2_score(y, y_pred):.2f}', fontsize=16, fontweight='bold')
plt.ylabel('Price Momentum (Standardized)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(loc='upper left', fontsize=11, frameon=True)

# 4. Highlight Crises
plt.axvspan(pd.to_datetime('2008-01-01'), pd.to_datetime('2009-06-01'), color='grey', alpha=0.2, label='Recession (2008)')
plt.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-06-01'), color='grey', alpha=0.2, label='COVID-19')

# Format Dates
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2)) # Tick every 2 years
plt.gcf().autofmt_xdate() # Angle dates

# Save
plt.savefig('xle_final_timeseries.png', dpi=300)
print("Graph saved to 'xle_final_timeseries.png'")