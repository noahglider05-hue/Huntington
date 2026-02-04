import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def fix_pd(csv_file):
    ETF = pd.read_csv(csv_file)
    ETF['observation_date'] = pd.to_datetime(ETF['observation_date'])
    ETF.set_index('observation_date', inplace=True)
    return ETF

# Get data
ETF = fix_pd('data/raw_data/XLE_monthly.csv')
# ETF = fix_pd('data/cleanedData/XLE_quarterly.csv')

ETF = ETF['Close']
MACRO = fix_pd('data/cleanedData/M_master_macro_table.csv')
# MACRO = fix_pd('data/cleanedData/Q_master_macro_table.csv')

master_table = MACRO.merge(ETF, on='observation_date', how='left')
X = master_table[["PCEPI", "GDP", "UNRATE", "FEDFUNDS", "MCOILWTICO"]]
X = X.shift(3)

y = master_table["Close"]

# Align after shift
data = pd.concat([X, y], axis=1).dropna()
X = data[X.columns]
y = data["Close"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False
)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# mean squared error and r2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# -------------------------
# Coefficients
# -------------------------
coefficients = pd.Series(model.coef_, index=X.columns)
print(f"Model Coefficients: {coefficients}")

# -------------------------
# Plot Actual vs Predicted
# -------------------------
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.index, y_test, label="Actual Close", linewidth=2)
# plt.plot(y_test.index, y_pred, label="Predicted Close", linestyle="--")
# plt.title("Actual vs Predicted Closing Price")
# plt.xlabel("Date")
# plt.ylabel("Close")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))

plt.plot(y_train.index, y_train, label="Train (Actual)", alpha=0.6)
plt.plot(y_test.index, y_test, label="Test (Actual)", linewidth=2)
plt.plot(y_test.index, y_pred, label="Test (Predicted)", linestyle="--")

plt.axvline(y_test.index[0], color="black", linestyle=":", label="Train/Test Split")

plt.title("Closing Price: Train vs Test")
plt.xlabel("Date")
plt.ylabel("Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()