import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
Graphs ETF to macro measurement
'''


def overlap_graph(MACRO, ETF,  ETF_name, MACRO_name):
    '''
    MACRO- the macro df, typically from master_macro_table.csv
    ETF- ETF df
    ETF_name- string you want displayed, ticker will do
    MACRO_name- string you want displayed for macro measurement

    problems: units, not every macro is the same
    '''
    # Put into one table
    data = pd.concat([ETF, MACRO], axis=1)
    data.columns = [f'{ETF_name}', f'{MACRO_name}']

    # visualize
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(data.index, data[f'{ETF_name}'], color='tab:blue', label=f'{ETF_name}')
    ax1.set_ylabel(f'{ETF_name} Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(data.index, data[f'{MACRO_name}'], color='tab:red', label=f'{MACRO_name}')
    ax2.set_ylabel(f'{MACRO_name} Price', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Quarterly Closing Prices: {ETF_name} vs {MACRO_name}')
    fig.tight_layout()
    plt.savefig(f'plots/{ETF_name}_vs_{MACRO_name}.png')
    plt.show()




    # Log returns
    # data_returns = np.log(data / data.shift(1)).dropna()
    # import statsmodels.api as sm
    # # Lag MCOILWTICO by 1 quarter, should probably change this to monthly
    # X = sm.add_constant(data_returns['MCOILWTICO'].shift(1).dropna())
    # y = data_returns['XLE'].loc[X.index]

    # model = sm.OLS(y, X).fit()
    # print(model.summary())
