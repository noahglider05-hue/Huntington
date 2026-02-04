import pandas as pd
import yfinance as yf

'''
Clean up CSV files and put them into a master set
'''
def read_quarterly(csv_file):
    # Make monthly->quarterly adjustments
    df = pd.read_csv(csv_file)

    if 'observation_date' not in df.columns:
        return "no observation_date found"
    
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df['date_diff'] = df['observation_date'].diff()
    median_diff = df['date_diff'].median()

    if median_diff.days > 80 and median_diff.days < 95:
        df = df.drop(columns=['date_diff'])
        return df
    
    df = df.drop(columns=['date_diff'])
    df.set_index('observation_date', inplace=True)
    df_quarterly = df.resample('QE').mean()
    #QS puts 3 months into a group, mean aggregates them into a single number
    # Otherwise it is <class 'pandas.core.resample.DatetimeIndexResampler'>

    return df_quarterly 

def interpolate_monthly(csv_file):
    """
    Takes quarterly measured data and interpolates it to monthly frequency,
    keeping a complete monthly date index.
    """
    import pandas as pd

    df = pd.read_csv(csv_file)

    if 'observation_date' not in df.columns:
        return "no observation_date found"

    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values('observation_date')
    df.set_index('observation_date', inplace=True)

    # Create full monthly date range
    full_monthly_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='MS'
    )

    # Reindex to full monthly dates
    df = df.reindex(full_monthly_index)

    # Interpolate missing values
    df_monthly = df.interpolate(method='linear')

    # Restore index name
    df_monthly.index.name = 'observation_date'

    return df_monthly

def read_monthly(csv_file):
    '''
    For data already measured monthly, this makes it ready to use
    '''
    df = pd.read_csv(csv_file)

    if 'observation_date' not in df.columns:
        raise ValueError("no observation_date found")

    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values('observation_date')
    df.set_index('observation_date', inplace=True)

    return df

def master_table(data_paths):
    # data is a list of file paths to CSVs
    # putting them all togther into a combined csv with the same formatting
    data_csv = [None]*len(data_paths)
    for i, x in enumerate(data_paths):
        if x == 'data/raw_data/GDP.csv':
            data_csv[i] = interpolate_monthly(x)
        else:
            data_csv[i] = read_monthly(x)

        # data_csv[i] = read_quarterly(x)

    master_table = data_csv[0]

    for df in data_csv[1:]:
        master_table = master_table.merge(df, on='observation_date', how='inner')

    print(master_table.head)

    master_table.to_csv('monthly_master_macro_table.csv')
    # master_table.to_excel('master_macro_table.xlsx', index=False)

def get_ticker(ticker):
    '''
    Use yahoo finance API to get specific stock/ETF data
    '''
    # Cut off for master_table is 2025-07-01 
    df = yf.download(ticker, start="2000-01-01", end="2025-08-01",interval="1mo")

    df.reset_index(inplace=True)
    df.columns = ['observation_date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
    df = df.dropna(subset=['observation_date'])

    # Quarterly average
    df = df.set_index('observation_date')
    df_quarterly = df.resample('QS').last()  

    # reset index
    df_quarterly.reset_index(inplace=True)
    df_quarterly['observation_date'] = pd.to_datetime(df_quarterly['observation_date'])
    
    df_quarterly.to_csv(f'{ticker}_quarterly.csv')
    return df_quarterly


inflation = 'data/raw_data/PCEPI.csv'
gdp = 'data/raw_data/GDP.csv'
unemployment = 'data/raw_data/UNRATE.csv'
intrest_rates = 'data/raw_data/FEDFUNDS.csv'
oil_rates = 'data/raw_data/MCOILWTICO.csv'

data = inflation, gdp, unemployment, intrest_rates, oil_rates

master_table(data)

