import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
import yfinance as yf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.anova import anova_lm


st.set_page_config(
    page_title="XLB Materials Model",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_fred_data(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    data = pd.read_csv(url, index_col=0, parse_dates=True)
    data[series_id] = pd.to_numeric(data[series_id], errors="coerce")
    return data


@st.cache_data(show_spinner=False)
def build_xlb_model():
    df_etf = yf.download(
        "XLB",
        start="1999-01-01",
        end="2021-01-01",
        auto_adjust=True,
        progress=False,
    )

    if "Close" in df_etf.columns:
        xlb_price = df_etf["Close"]
    else:
        xlb_price = df_etf.iloc[:, 0]

    y_raw = xlb_price.resample("MS").last().pct_change().dropna() * 100

    yield_10y = get_fred_data("DGS10")
    dollar_idx = get_fred_data("DTWEXBGS")
    vix = get_fred_data("VIXCLS")

    df_macros = pd.concat([yield_10y, dollar_idx, vix], axis=1).resample("MS").mean()
    df_macros.columns = ["Yield_10Y", "USD_Index", "Volatility"]

    combined = pd.concat([y_raw, df_macros], axis=1).dropna()
    combined.columns = ["y", "Yield_10Y", "USD_Index", "Volatility"]

    cols_to_std = ["Yield_10Y", "USD_Index", "Volatility"]
    combined[cols_to_std] = (
        combined[cols_to_std] - combined[cols_to_std].mean()
    ) / combined[cols_to_std].std()

    train_df = combined.loc["2000-01-01":"2016-12-31"]
    static_model = smf.ols("y ~ Yield_10Y + USD_Index + Volatility", data=train_df).fit()
    anova_results = anova_lm(static_model)

    x_rolling = sm.add_constant(combined[cols_to_std])
    y_rolling = combined["y"]
    model = RollingOLS(y_rolling, x_rolling, window=60).fit()
    y_pred = (model.params.shift(1) * x_rolling).sum(axis=1)

    test_y = combined["y"].loc["2016-01-01":"2020-12-31"]
    test_pred = y_pred.loc["2016-01-01":"2020-12-31"]

    return combined, anova_results, test_y, test_pred


def plot_validation(test_y, test_pred):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        test_y.index,
        test_y,
        label="Actual XLB Returns",
        color="black",
        alpha=0.35,
    )
    ax.plot(
        test_pred.index,
        test_pred,
        label="High-Signal Model",
        color="forestgreen",
        linewidth=2,
    )
    ax.set_title("XLB (Materials): 2016-2020 High-Signal Validation")
    ax.set_ylabel("Monthly Return (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def main():
    st.title("XLB Materials High-Signal Model")
    st.caption("Train 2000-2016 | Test 2016-2020")

    try:
        with st.spinner("Loading market and macro data..."):
            combined, anova_results, test_y, test_pred = build_xlb_model()

        latest_date = combined.index.max().strftime("%B %Y")
        avg_predicted = test_pred.dropna().mean()

        metric_cols = st.columns(3)
        metric_cols[0].metric("Observations", f"{len(combined):,}")
        metric_cols[1].metric("Latest Data", latest_date)
        metric_cols[2].metric("Avg Test Signal", f"{avg_predicted:.2f}%")

        st.subheader("Validation Plot")
        st.pyplot(plot_validation(test_y, test_pred), width="stretch")

        st.subheader("ANOVA Results")
        st.dataframe(anova_results, width="stretch")

        st.subheader("Test Window Data")
        test_table = pd.DataFrame(
            {
                "Actual XLB Returns": test_y,
                "High-Signal Model": test_pred,
            }
        )
        st.dataframe(test_table, width="stretch")

    except Exception as exc:
        st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
