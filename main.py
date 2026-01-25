from data_cleanse import read_quarterly, master_table

# file paths
inflation = 'data/PCEPI.csv'
gdp = 'data/GDP.csv'
unemployment = 'data/UNRATE.csv'
intrest_rates = 'data/FEDFUNDS.csv'
oil_rates = 'data/MCOILWTICO.csv'

data = inflation, gdp, unemployment, intrest_rates, oil_rates

master_table(data)
