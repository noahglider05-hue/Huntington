import pandas as pd

'''
Clean up CSV files and put them into a master set
'''

def read_quarterly(csv_file):
    # Make monthly-quarterly adjustments
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
    df_quarterly = df.resample('QS').mean()

    return df_quarterly 


def master_table(data_paths):
    # data is a list of file paths to CSVs
    # putting them all togther into a combined csv with the same formatting
    data_csv = [None]*len(data_paths)
    for i, x in enumerate(data_paths):
        data_csv[i] = read_quarterly(x)

    master_table = data_csv[0]

    for df in data_csv[1:]:
        master_table = master_table.merge(df, on='observation_date', how='inner')

    master_table.to_csv('master_macro_table.csv', index=False)
    master_table.to_excel('master_macro_table.xlsx', index=False)
