import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def load_fred_csv(filepath, val_col_name):
    df = pd.read_csv(filepath)
    df.rename(columns={df.columns[0]: 'DATE', df.columns[1]: val_col_name}, inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])
    df[val_col_name] = pd.to_numeric(df[val_col_name], errors='coerce')
    return df.set_index('DATE')

# 1. Load the exact macros used in M2.6
pcepi = load_fred_csv('PCEPI.csv', 'PCEPI')
gdp = load_fred_csv('GDP.csv', 'GDP')
unrate = load_fred_csv('UNRATE.csv', 'UNRATE')
fedfunds = load_fred_csv('FEDFUNDS.csv', 'FEDFUNDS')
oil = load_fred_csv('MCOILWTICO.csv', 'MCOILWTICO') 

ticker_data = yf.download('XLE', start='2000-01-01', end='2025-01-01')

if isinstance(ticker_data.columns, pd.MultiIndex):
    ticker_data.columns = ticker_data.columns.get_level_values(0)

target = ticker_data[['Close']].copy()
target.columns = ['XLE'] 
target = target.resample('MS').last()

df = target.join([pcepi, gdp, unrate, fedfunds, oil], how='outer')
df = df.resample('MS').ffill().dropna()
df = df.loc['2000-01-01':'2025-01-01']
df.columns = df.columns.astype(str)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# 2. Replicate the M2.6 Preprocessing (Correlated Group: GDP & PCEPI)
pca = PCA(n_components=1)
df_scaled['GDP_PCEPI_PC1'] = pca.fit_transform(df_scaled[['GDP', 'PCEPI']])

# 3. Define the exact M2.6 Features
features = ['GDP_PCEPI_PC1', 'UNRATE', 'FEDFUNDS', 'MCOILWTICO'] 

df_lagged = df_scaled.copy()
df_lagged[features] = df_lagged[features].shift(1)
df_lagged = df_lagged.dropna()

df_model = pd.concat([df_lagged['XLE'], df_lagged[features]], axis=1)

split_idx = int(len(df_model) * 0.8)
df_train = df_model.iloc[:split_idx]
df_test = df_model.iloc[split_idx:]

# 4. Run M2.6 through ARMA(1, 0, 1)
model = ARIMA(endog=df_train['XLE'], exog=df_train[features], order=(1, 0, 1))
fitted_model = model.fit()

print("\n" + "="*80)
print("ARMA M2.6 REGRESSION SUMMARY (TRAIN)")
print("="*80)
print(fitted_model.summary())

y_train_pred = fitted_model.predict(start=df_train.index[0], end=df_train.index[-1], exog=df_train[features])
y_test_pred = fitted_model.predict(start=df_test.index[0], end=df_test.index[-1], exog=df_test[features])

train_r2 = r2_score(df_train['XLE'], y_train_pred)
test_r2 = r2_score(df_test['XLE'], y_test_pred)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 7))

plt.plot(df_model.index, df_model['XLE'], color='#333333', linewidth=2.5, alpha=0.8, label='Actual XLE (Scaled)')
plt.plot(df_train.index, y_train_pred, color='#FF5733', linewidth=2, linestyle='--', label=f'Train Prediction (R²: {train_r2:.2f})')
plt.plot(df_test.index, y_test_pred, color='#0078D7', linewidth=2, linestyle='--', label=f'Test Prediction (R²: {test_r2:.2f})')

split_date = df_model.index[split_idx]
plt.axvline(x=split_date, color='black', linestyle=':', linewidth=2, label='Train/Test Split')

plt.title('M2.6 Combination: XLE Price Prediction ARMA Model', fontsize=16, fontweight='bold')
plt.ylabel('Price Momentum (Standardized)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(loc='upper left', fontsize=11, frameon=True)
plt.axvspan(pd.to_datetime('2008-01-01'), pd.to_datetime('2009-06-01'), color='grey', alpha=0.2, label='Recession (2008)')
plt.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-06-01'), color='grey', alpha=0.2, label='COVID-19')

plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gcf().autofmt_xdate()
plt.savefig('xle_arma_m2_6.png', dpi=300)
print("Done. Graph saved to 'xle_arma_m2_6.png'")