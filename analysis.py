import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from stock import get_ticker


master_table = pd.read_csv('master_macro_table.csv')
master_table['observation_date'] = pd.to_datetime(master_table['observation_date'])


# Pick one column
ticker = 'XLE'
# stock = get_ticker(ticker) # use if the data set it not downloaded

stock = pd.read_csv(f'{ticker}_quarterly.csv')
stock['observation_date'] = pd.to_datetime(stock['observation_date'])

#Use the close column for analysis 
stock = stock[['observation_date', 'Close']]

# What you really want is adjusted close
master_table = master_table.merge(stock, on='observation_date', how='left')

# Gets messed up if it picks up date, only need the second column
corr_matrix = master_table.select_dtypes(include='number').corr()

print(corr_matrix)

plt.figure(figsize=(8,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Greens')
plt.title(f'Correlation Matrix of Macroeconomic Variables and {ticker}')
plt.savefig(f'plots/{ticker}.png')
plt.show()
