# ==========================================
# Streamlit Dashboard: Bitcoin Sentiment + Trading Analysis
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(page_title="Trading Sentiment Dashboard", layout="wide")

st.title("📊 Crypto Trading + Sentiment Dashboard")

# =========================
# 2. LOAD DATA
# =========================
@st.cache_data
def load_data():
    trades = pd.read_csv("trades.csv")
    sentiment = pd.read_csv("sentiment.csv")

    trades.columns = trades.columns.str.strip()
    sentiment.columns = sentiment.columns.str.strip()

    trades['Date'] = pd.to_datetime(trades['time'], errors='coerce')
    sentiment['Date'] = pd.to_datetime(sentiment['Date'], errors='coerce')

    merged = trades.merge(sentiment, on='Date', how='left')

    return merged

merged = load_data()

# =========================
# 3. SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

coins = merged['Coin'].dropna().unique()
selected_coin = st.sidebar.selectbox("Select Coin", coins)

filtered = merged[merged['Coin'] == selected_coin]

# =========================
# 4. FEATURE ENGINEERING
# =========================
filtered['abs_size'] = filtered['Size USD'].abs()
filtered['target'] = (filtered['Closed PnL'] > 0).astype(int)

if 'Classification' in filtered.columns:
    filtered['sentiment_score'] = filtered['Classification'].map({'Fear': 0, 'Greed': 1})

# =========================
# 5. KPI METRICS
# =========================
st.subheader("📌 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Total PnL", round(filtered['Closed PnL'].sum(), 2))
col2.metric("Avg Trade Size", round(filtered['Size USD'].mean(), 2))
col3.metric("Total Trades", len(filtered))

# =========================
# 6. PLOTS
# =========================
st.subheader("📈 PnL Distribution")
fig1, ax1 = plt.subplots()
ax1.hist(filtered['Closed PnL'].dropna(), bins=40)
st.pyplot(fig1)

st.subheader("📉 Size vs PnL")
fig2, ax2 = plt.subplots()
ax2.scatter(filtered['Size USD'], filtered['Closed PnL'], alpha=0.4)
st.pyplot(fig2)

# =========================
# 7. DAILY PNL TREND
# =========================
st.subheader("📊 Daily PnL Trend")

filtered['Date_only'] = filtered['Date'].dt.date
daily = filtered.groupby('Date_only')['Closed PnL'].sum()

fig3, ax3 = plt.subplots()
daily.plot(ax=ax3)
st.pyplot(fig3)

# =========================
# 8. SENTIMENT IMPACT
# =========================
if 'sentiment_score' in filtered.columns:
    st.subheader("🧠 Sentiment Impact")
    st.write(filtered.groupby('sentiment_score')['Closed PnL'].mean())

# =========================
# 9. SIMPLE ML MODEL
# =========================
st.subheader("🤖 Profit Prediction Model")

features = ['Size USD']
if 'sentiment_score' in filtered.columns:
    features.append('sentiment_score')

ml_data = filtered.dropna(subset=features + ['Closed PnL'])
ml_data['target'] = (ml_data['Closed PnL'] > 0).astype(int)

if len(ml_data) > 50:
    X = ml_data[features]
    y = ml_data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    st.metric("Model Accuracy", round(score, 3))

    st.text("Classification Report:")
    st.text(classification_report(y_test, model.predict(X_test)))
else:
    st.warning("Not enough data for ML model")

# =========================
# 10. TOP INSIGHTS
# =========================
st.subheader("💡 Insights")

st.write("- Analyze sentiment impact on profitability")
st.write("- Observe trade size vs returns")
st.write("- Identify best performing coins")
st.write("- Monitor daily trading performance")

# =========================
# END
# ========================="}
