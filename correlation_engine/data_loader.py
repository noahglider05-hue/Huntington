import pandas as pd

def split_macro_and_etf(master_df: pd.DataFrame, macro_columns: list, etf_columns: list) -> pd.DataFrame: 
    """
    Splits the validated master DataFrame into two separate DataFrames:

    1) macro_df:
       - Contains only macroeconomic variables as columns
       - Uses observation_date as DatetimeIndex
       - Data is interpolated monthly

    2) etf_df:
       - Contains only ETF close price columns
       - Same DatetimeIndex as macro_df
    """
    macro_df = master_df[macro_columns].copy() # creates deep copy
    etf_df = master_df[etf_columns].copy() 

    return macro_df, etf_df