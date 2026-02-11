import pandas as pd
"""
Given a macro series, ETF series, and a lag range, compute the correlations.
"""
def create_time_windows(master_df: pd.DataFrame, window_years: int) -> list:
    """
    Generates the windows to be correlated based on the index.

    Example:
        If window_years = 5
        2000-2004
        2005-2009
        2010-2014
        etc.

    Returns list of (start_date, end_date) tuples or sliced DataFrames.

    Becomes more dynamic because we can test with multiple different windows (different years).
    """
    pass

def compute_lagged_correlations(macro_df: pd.DataFrame, etf_df: pd.DataFrame, windows: list, lags: int) -> dict:
    """
    Determine the optimal lag for each n-year chunk for every macro_variable across all etfs
    
    Returns something like this: 
    {
        'ETF_name1': {
            'macro_variable_name': [best_lag_for_window1, best_lag_for_window2, etc.]
        }
        
        'ETF_name2': {
            'macro_variable_name': [best_lag_for_window1, best_lag_for_window2, etc.]
        }
    }
    """
    pass

def aggregate_lags(lagged_correlations: dict) -> dict:
    """
    Takes in the output from the compute_lagged_correlations() function
        - for each etf, for each macro_variable, compute the mode ; mode = best/most_consistent lag for that macro variable
        
    Returns something like this (optimal_lags):
    {
        'ETF_name': {
            'macro_variable_name': optimal_lag
        }
    }

    """

    pass