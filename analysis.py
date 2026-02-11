import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

def correlation(master_df: pd.DataFrame):
    '''
    Finds correlation between specific ETF and macro data; 5 year chunks
    '''
    etf_col = master_df['Close']
    
    five_year_periods = [
        {"period": "2000-2004", "start": "2000-01-01", "end": "2004-12-31"},
        {"period": "2005-2009", "start": "2005-01-01", "end": "2009-12-31"},
        {"period": "2010-2014", "start": "2010-01-01", "end": "2014-12-31"},
        {"period": "2015-2019", "start": "2015-01-01", "end": "2019-12-31"},
        {"period": "2020-2025", "start": "2020-01-01", "end": "2025-12-31"},
    ]

    # All columns except ETF = macro variables
    macro_cols = [col for col in master_df.columns if col != etf_col]

    for period in five_year_periods:
        # Slice the dataframe for the period
        window = master_df.loc[period["start"]:period["end"]]

        if window.empty:
            continue

        # Compute correlation: macro vs ETF
        corr_series = window[macro_cols].corrwith(window[etf_col])
        corr_df = corr_series.to_frame(name="Correlation")

        # Plot heatmap
        plt.figure(figsize=(6, max(4, len(macro_cols) * 0.35)))
        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            cmap="Greens",
            center=0,
            linewidths=0.5
        )

        plt.title(f"{etf_col} vs Macros Correlation ({period['label']})")
        plt.tight_layout()
        plt.savefig(f"plots/{etf_col}_correlation_{period['label']}.png")
        plt.show()

# def graph(MACRO, ETF,  ETF_name, MACRO_name):
#     '''
#         MACRO- the macro df, typically from master_macro_table.csv
#         ETF- ETF df
#         ETF_name- string you want displayed, ticker will do
#         MACRO_name- string you want displayed for macro measurement

#         problems: units, not every macro is the same
#     '''
#     # Put into one table
#     data = pd.concat([ETF, MACRO], axis=1)
#     data.columns = [f'{ETF_name}', f'{MACRO_name}']

#     # visualize
#     fig, ax1 = plt.subplots(figsize=(10, 5))
#     ax1.plot(data.index, data[f'{ETF_name}'], color='tab:blue', label=f'{ETF_name}')
#     ax1.set_ylabel(f'{ETF_name} Price', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     ax2 = ax1.twinx()
#     ax2.plot(data.index, data[f'{MACRO_name}'], color='tab:red', label=f'{MACRO_name}')
#     ax2.set_ylabel(f'{MACRO_name} Price', color='tab:red')
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     plt.title(f'Quarterly Closing Prices: {ETF_name} vs {MACRO_name}')
#     fig.tight_layout()
#     plt.savefig(f'plots/{ETF_name}_vs_{MACRO_name}.png')
#     plt.show()

'''
ETF = fix_pd('data/cleanedData/XLE_quarterly.csv')
ETF = ETF['Close']
MACRO = fix_pd('monthly_master_macro_table.csv')

master_table = MACRO.merge(ETF, on='observation_date', how='left')

print(master_table.head)

correlation(master_table, 'XLE')

MACRO_specific = MACRO['PCEPI']
graph(MACRO_specific, ETF, "XLE", "PCEPI")
'''