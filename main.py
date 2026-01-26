from overlapGraph import overlap_graph
import pandas as pd

ETF = pd.read_csv('XLP_quarterly.csv')
ETF['observation_date'] = pd.to_datetime(ETF['observation_date'])
ETF.set_index('observation_date', inplace=True)
ETF = ETF['Close']

# Just overlap it with all the macro data... duh
MACRO = pd.read_csv('master_macro_table.csv')
MACRO['observation_date'] = pd.to_datetime(MACRO['observation_date'])
MACRO.set_index('observation_date', inplace = True)
MACRO = MACRO['FEDFUNDS']


overlap_graph(MACRO, ETF, "XLP", "FEDFUNDS")

