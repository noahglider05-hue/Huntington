import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
import yfinance as yf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.stattools import durbin_watson


SECTOR_MODELS = {
    "XLB": {
        "name": "Materials",
        "data_start": "1999-01-01",
        "train_start": "2000-01-01",
        "train_end": "2016-12-31",
        "color": "forestgreen",
    },
    "VOX": {
        "name": "Communication Services",
        "data_start": "2004-01-01",
        "train_start": "2004-01-01",
        "train_end": "2016-12-31",
        "color": "purple",
    },
    "IYR": {
        "name": "Real Estate",
        "data_start": "1999-01-01",
        "train_start": "2000-01-01",
        "train_end": "2016-12-31",
        "color": "darkorange",
    },
    "XLF": {
        "name": "Financials",
        "data_start": "1999-01-01",
        "train_start": "2000-01-01",
        "train_end": "2015-12-31",
        "color": "red",
    },
    "XLE": {
        "name": "Energy",
        "data_start": "1999-01-01",
        "train_start": "2000-01-01",
        "train_end": "2016-12-31",
        "color": "red",
    },
}

MACRO_SERIES = {
    "DGS10": "Yield_10Y",
    "DTWEXBGS": "USD_Index",
    "VIXCLS": "Volatility",
}

FEATURE_COLUMNS = list(MACRO_SERIES.values())
TEST_START = "2016-01-01"
TEST_END = "2020-12-31"


st.set_page_config(
    page_title="Sector Signal Models",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_fred_data(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    data = pd.read_csv(url, index_col=0, parse_dates=True)
    data[series_id] = pd.to_numeric(data[series_id], errors="coerce")
    return data


@st.cache_data(show_spinner=False)
def get_macro_data():
    macro_frames = [get_fred_data(series_id) for series_id in MACRO_SERIES]
    df_macros = pd.concat(macro_frames, axis=1, sort=False).resample("MS").mean()
    df_macros.columns = FEATURE_COLUMNS
    return df_macros


@st.cache_data(show_spinner=False)
def get_monthly_returns(ticker, start_date):
    df_etf = yf.download(
        ticker,
        start=start_date,
        end="2021-01-01",
        auto_adjust=True,
        progress=False,
    )

    if df_etf.empty:
        raise ValueError(f"No price data returned for {ticker}.")

    if isinstance(df_etf.columns, pd.MultiIndex):
        if "Close" in df_etf.columns.get_level_values(0):
            prices = df_etf["Close"].iloc[:, 0]
        else:
            prices = df_etf.iloc[:, 0]
    elif "Close" in df_etf.columns:
        prices = df_etf["Close"]
    else:
        prices = df_etf.iloc[:, 0]

    return prices.resample("MS").last().pct_change().dropna() * 100


@st.cache_data(show_spinner=False)
def build_sector_model(ticker, data_start, train_start, train_end):
    y_raw = get_monthly_returns(ticker, data_start)
    df_macros = get_macro_data()

    combined = pd.concat([y_raw, df_macros], axis=1, sort=False).dropna()
    combined.columns = ["y", *FEATURE_COLUMNS]

    combined[FEATURE_COLUMNS] = (
        combined[FEATURE_COLUMNS] - combined[FEATURE_COLUMNS].mean()
    ) / combined[FEATURE_COLUMNS].std()

    train_df = combined.loc[train_start:train_end]
    static_model = smf.ols("y ~ Yield_10Y + USD_Index + Volatility", data=train_df).fit()
    anova_results = anova_lm(static_model)

    x_rolling = sm.add_constant(combined[FEATURE_COLUMNS])
    model = RollingOLS(combined["y"], x_rolling, window=60).fit()
    y_pred = (model.params.shift(1) * x_rolling).sum(axis=1)

    test_y = combined["y"].loc[TEST_START:TEST_END]
    test_pred = y_pred.loc[TEST_START:TEST_END]

    return combined, static_model, anova_results, test_y, test_pred


def build_coefficient_table(static_model):
    confidence_interval = static_model.conf_int()
    coefficient_table = pd.DataFrame(
        {
            "Coefficient": static_model.params,
            "Std Error": static_model.bse,
            "t-stat": static_model.tvalues,
            "p-value": static_model.pvalues,
            "CI Lower": confidence_interval[0],
            "CI Upper": confidence_interval[1],
        }
    )
    return coefficient_table


def build_diagnostics_table(static_model):
    return pd.DataFrame(
        {
            "Value": {
                "R-squared": static_model.rsquared,
                "Adjusted R-squared": static_model.rsquared_adj,
                "F-statistic": static_model.fvalue,
                "F p-value": static_model.f_pvalue,
                "Durbin-Watson": durbin_watson(static_model.resid),
                "AIC": static_model.aic,
                "BIC": static_model.bic,
                "Log-likelihood": static_model.llf,
                "Observations": static_model.nobs,
                "Residual degrees of freedom": static_model.df_resid,
            }
        }
    )


def plot_validation(ticker, sector_name, color, test_y, test_pred):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        test_y.index,
        test_y,
        label=f"Actual {ticker} Returns",
        color="black",
        alpha=0.35,
    )
    ax.plot(
        test_pred.index,
        test_pred,
        label="High-Signal Model",
        color=color,
        linewidth=2,
    )
    ax.set_title(f"{ticker} ({sector_name}): 2016-2020 High-Signal Validation")
    ax.set_ylabel("Monthly Return (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def render_model(ticker, config):
    with st.spinner(f"Loading {ticker} market and macro data..."):
        combined, static_model, anova_results, test_y, test_pred = build_sector_model(
            ticker,
            config["data_start"],
            config["train_start"],
            config["train_end"],
        )

    latest_date = combined.index.max().strftime("%B %Y")
    avg_actual = test_y.mean()
    avg_predicted = test_pred.dropna().mean()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Observations", f"{len(combined):,}")
    metric_cols[1].metric("Latest Data", latest_date)
    metric_cols[2].metric("Avg Actual Return", f"{avg_actual:.2f}%")
    metric_cols[3].metric("Avg Test Signal", f"{avg_predicted:.2f}%")

    st.subheader("Validation Plot")
    st.pyplot(
        plot_validation(
            ticker,
            config["name"],
            config["color"],
            test_y,
            test_pred,
        ),
        width="stretch",
    )

    st.subheader("Regression Statistics")
    stats_cols = st.columns(4)
    stats_cols[0].metric("R-squared", f"{static_model.rsquared:.3f}")
    stats_cols[1].metric("Adjusted R-squared", f"{static_model.rsquared_adj:.3f}")
    stats_cols[2].metric("F p-value", f"{static_model.f_pvalue:.4f}")
    stats_cols[3].metric("Durbin-Watson", f"{durbin_watson(static_model.resid):.3f}")

    st.subheader("Coefficients")
    st.dataframe(build_coefficient_table(static_model), width="stretch")

    st.subheader("Model Diagnostics")
    st.dataframe(build_diagnostics_table(static_model), width="stretch")

    st.subheader("ANOVA Results")
    st.dataframe(anova_results, width="stretch")

    st.subheader("Test Window Data")
    test_table = pd.DataFrame(
        {
            f"Actual {ticker} Returns": test_y,
            "High-Signal Model": test_pred,
        }
    )
    st.dataframe(test_table, width="stretch")


def main():
    st.title("Sector High-Signal Models")

    selected_label = st.sidebar.selectbox(
        "Sector model",
        [
            f"{ticker} - {config['name']}"
            for ticker, config in SECTOR_MODELS.items()
        ],
    )
    ticker = selected_label.split(" - ", maxsplit=1)[0]
    config = SECTOR_MODELS[ticker]

    st.caption(
        f"{ticker} {config['name']} | "
        f"Train {config['train_start'][:4]}-{config['train_end'][:4]} | "
        "Test 2016-2020"
    )

    try:
        render_model(ticker, config)
    except Exception as exc:
        st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
