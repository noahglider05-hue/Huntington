import yfinance as yf
import pandas as pd 
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