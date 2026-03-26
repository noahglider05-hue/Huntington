import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

print("Loading local CSV files...")

def load_fred_csv(filepath, val_col_name):
    df = pd.read_csv(filepath)
    df.rename(columns={df.columns[0]: 'DATE', df.columns[1]: val_col_name}, inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])
    df[val_col_name] = pd.to_numeric(df[val_col_name], errors='coerce')
    return df.set_index('DATE')

pcepi = load_fred_csv('PCEPI.csv', 'PCEPI')
gdp = load_fred_csv('GDP.csv', 'GDP')
unrate = load_fred_csv('UNRATE.csv', 'UNRATE')
fedfunds = load_fred_csv('FEDFUNDS.csv', 'FEDFUNDS')
dgs10 = load_fred_csv('DGS10.csv', 'DGS10') 

print("Fetching VGT data via Yahoo Finance API...")
ticker_data = yf.download('VGT', start='2000-01-01', end='2025-01-01')

if isinstance(ticker_data.columns, pd.MultiIndex):
    ticker_data.columns = ticker_data.columns.get_level_values(0)

target = ticker_data[['Close']].copy()
target.columns = ['VGT'] 
target = target.resample('MS').last()

print("Merging data...")
df = target.join([pcepi, gdp, unrate, fedfunds, dgs10], how='outer')
df = df.resample('MS').ffill().dropna()
df = df.loc['2000-01-01':'2025-01-01']

df.columns = df.columns.astype(str)

print("Running PCA...")
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

pca = PCA(n_components=1)
df_scaled['ECON_TREND_PC1'] = pca.fit_transform(df_scaled[['GDP', 'PCEPI']])

features = ['ECON_TREND_PC1', 'UNRATE', 'FEDFUNDS', 'DGS10'] 

df_lagged = df_scaled.copy()
df_lagged[features] = df_lagged[features].shift(1)
df_lagged = df_lagged.dropna()

df_model = pd.concat([df_lagged['VGT'], df_lagged[features]], axis=1)

print("Performing Train/Test Split (Time Series: 80% Train, 20% Test)...")
split_idx = int(len(df_model) * 0.8)
df_train = df_model.iloc[:split_idx]
df_test = df_model.iloc[split_idx:]

print("Training OLS Regression Model on Training Data...")
model = smf.ols('VGT ~ ECON_TREND_PC1 + UNRATE + FEDFUNDS + DGS10', data=df_train).fit()

print("\n" + "="*80)
print("OLS REGRESSION SUMMARY (TRAIN)")
print("="*80)
print(model.summary())

print("\n" + "="*80)
print("ANOVA TABLE (TRAIN)")
print("="*80)
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
print("="*80 + "\n")

y_train_pred = model.predict(df_train[features])
y_test_pred = model.predict(df_test[features])

train_r2 = model.rsquared
test_r2 = r2_score(df_test['VGT'], y_test_pred)

print("Generating final graph...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 7))

plt.plot(df_model.index, df_model['VGT'], color='#333333', linewidth=2.5, alpha=0.8, label='Actual VGT (Scaled)')

plt.plot(df_train.index, y_train_pred, color='#0078D7', linewidth=2, linestyle='--', label=f'Train Prediction (R²: {train_r2:.2f})')
plt.plot(df_test.index, y_test_pred, color='#FF5733', linewidth=2, linestyle='--', label=f'Test Prediction (R²: {test_r2:.2f})')

split_date = df_model.index[split_idx]
plt.axvline(x=split_date, color='black', linestyle=':', linewidth=2, label='Train/Test Split')

plt.title('Real Data: VGT Price Prediction Model (Train/Test Split)', fontsize=16, fontweight='bold')
plt.ylabel('Price Momentum (Standardized)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(loc='upper left', fontsize=11, frameon=True)

plt.axvspan(pd.to_datetime('2008-01-01'), pd.to_datetime('2009-06-01'), color='grey', alpha=0.2, label='Recession (2008)')
plt.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-06-01'), color='grey', alpha=0.2, label='COVID-19')

plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gcf().autofmt_xdate()

plt.savefig('vgt_ols_traintest.png', dpi=300)
print("Done. Graph saved to 'vgt_ols_traintest.png'")