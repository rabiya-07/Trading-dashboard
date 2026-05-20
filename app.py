import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Trading Sentiment Dashboard", layout="wide")
st.title("Crypto Trading + Sentiment Dashboard")

TRADES_FILE = Path("trades.csv")
SENTIMENT_FILE = Path("sentiment.csv")


def _get_secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_data
def load_data(trades_source):
    trades = pd.read_csv(trades_source)
    sentiment = pd.read_csv(SENTIMENT_FILE)

    trades.columns = trades.columns.str.strip()
    sentiment.columns = sentiment.columns.str.strip().str.lower()

    trades["Date"] = pd.to_datetime(trades["time"], errors="coerce").dt.date
    sentiment["Date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.date

    return trades.merge(sentiment, on="Date", how="left")


st.sidebar.header("Data")
uploaded_trades = st.sidebar.file_uploader("Upload trades.csv", type="csv")
trades_url = _get_secret("TRADES_CSV_URL")

if uploaded_trades is not None:
    trades_source = uploaded_trades
elif trades_url:
    trades_source = trades_url
elif TRADES_FILE.exists():
    trades_source = TRADES_FILE
else:
    st.info(
        "Upload trades.csv in the sidebar, or add a TRADES_CSV_URL secret in "
        "Streamlit Cloud. The large trades.csv file should stay out of GitHub."
    )
    st.stop()

if not SENTIMENT_FILE.exists():
    st.error("Missing sentiment.csv. Keep sentiment.csv in GitHub with the app.")
    st.stop()

merged = load_data(trades_source)

required_columns = {"Coin", "Size USD", "Closed PnL", "Date"}
missing_columns = required_columns - set(merged.columns)
if missing_columns:
    st.error(f"Missing required column(s): {', '.join(sorted(missing_columns))}")
    st.stop()

st.sidebar.header("Filters")
coins = sorted(merged["Coin"].dropna().unique())

if len(coins) == 0:
    st.warning("No coins found in trades.csv.")
    st.stop()

selected_coin = st.sidebar.selectbox("Select Coin", coins)
filtered = merged[merged["Coin"] == selected_coin].copy()

filtered["abs_size"] = filtered["Size USD"].abs()
filtered["target"] = (filtered["Closed PnL"] > 0).astype(int)

if "classification" in filtered.columns:
    filtered["sentiment_score"] = filtered["classification"].map(
        {
            "Extreme Fear": 0,
            "Fear": 0,
            "Neutral": 0.5,
            "Greed": 1,
            "Extreme Greed": 1,
        }
    )

st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Total PnL", round(filtered["Closed PnL"].sum(), 2))
col2.metric("Avg Trade Size", round(filtered["Size USD"].mean(), 2))
col3.metric("Total Trades", len(filtered))

st.subheader("PnL Distribution")
fig1, ax1 = plt.subplots()
ax1.hist(filtered["Closed PnL"].dropna(), bins=40)
ax1.set_xlabel("Closed PnL")
ax1.set_ylabel("Trades")
st.pyplot(fig1)

st.subheader("Size vs PnL")
fig2, ax2 = plt.subplots()
ax2.scatter(filtered["Size USD"], filtered["Closed PnL"], alpha=0.4)
ax2.set_xlabel("Size USD")
ax2.set_ylabel("Closed PnL")
st.pyplot(fig2)

st.subheader("Daily PnL Trend")
daily = filtered.groupby("Date")["Closed PnL"].sum()
fig3, ax3 = plt.subplots()
daily.plot(ax=ax3)
ax3.set_xlabel("Date")
ax3.set_ylabel("Closed PnL")
st.pyplot(fig3)

if "sentiment_score" in filtered.columns:
    st.subheader("Sentiment Impact")
    st.write(filtered.groupby("sentiment_score")["Closed PnL"].mean())

st.subheader("Profit Prediction Model")
features = ["Size USD"]
if "sentiment_score" in filtered.columns:
    features.append("sentiment_score")

ml_data = filtered.dropna(subset=features + ["Closed PnL"]).copy()
ml_data["target"] = (ml_data["Closed PnL"] > 0).astype(int)

if len(ml_data) > 50 and ml_data["target"].nunique() > 1:
    X = ml_data[features]
    y = ml_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    st.metric("Model Accuracy", round(score, 3))

    st.text("Classification Report:")
    st.text(classification_report(y_test, model.predict(X_test)))
else:
    st.warning("Not enough data for ML model")

st.subheader("Insights")
st.write("- Analyze sentiment impact on profitability")
st.write("- Observe trade size vs returns")
st.write("- Identify best performing coins")
st.write("- Monitor daily trading performance")
