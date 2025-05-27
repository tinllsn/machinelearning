import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Random Forest Stock Prediction", layout="centered")

st.title("📈 Random Forest Stock Market Prediction")

# 1. Load S&P 500 data
@st.cache_data
def load_sp500():
    sp500 = yf.Ticker("^GSPC")
    df = sp500.history(start="2022-01-01")
    return df

st.text('Loading data...')
sp500 = load_sp500()
sp500 = sp500.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
st.success('Data loaded successfully!')
st.write(sp500)

st.subheader('📊 Close Price Data')
st.line_chart(sp500['Close'])

# 2. Create moving averages and target
sp500["MA10"] = sp500["Close"].rolling(window=10).mean()
sp500["MA50"] = sp500["Close"].rolling(window=50).mean()
sp500["Target"] = (sp500["MA10"] > sp500["MA50"]).astype(int)
sp500 = sp500.dropna()

st.subheader('🎯 Target Based on MA10 vs MA50')
st.write(sp500)

# 3. Plot price with MA10 and MA50 (Plotly)
st.subheader('📉 Interactive Price Chart with MA10 and MA50')
fig = go.Figure()
fig.add_trace(go.Scatter(x=sp500.index, y=sp500['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=sp500.index, y=sp500['MA10'], mode='lines', name='MA10'))
fig.add_trace(go.Scatter(x=sp500.index, y=sp500['MA50'], mode='lines', name='MA50'))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    hovermode='x unified',
    template='plotly_white',
    legend=dict(x=0, y=1),
    height=600
)
st.plotly_chart(fig)

# 4. Simple matplotlib plot
st.subheader('🧾 Matplotlib: Price with MA10 and MA50')
fig_ma, ax = plt.subplots(figsize=(10,5))
sp500['Close'].plot(ax=ax, label='Close', alpha=0.5)
sp500['MA10'].plot(ax=ax, label='MA10', linestyle='--')
sp500['MA50'].plot(ax=ax, label='MA50', linestyle='--')
plt.legend()
st.pyplot(fig_ma)

# 5. Add rolling features
horizons = [2, 5, 10, 50, 60, 250]
new_predictors = []
for horizon in horizons:
    rolling_avg = sp500['Close'].rolling(horizon).mean()
    sp500[f'Close_Ratio_{horizon}'] = sp500['Close'] / rolling_avg
    sp500[f'Trend_{horizon}'] = sp500['Target'].shift(1).rolling(horizon).sum()
    new_predictors += [f'Close_Ratio_{horizon}', f'Trend_{horizon}']
print(new_predictors)

sp500 = sp500.dropna()
st.subheader('🧠 Technical Features')
st.write(sp500)

# 6. Train-test split
train_size = int(len(sp500)*0.8)
train = sp500.iloc[:train_size]
test = sp500.iloc[train_size:]
predictors = ['Close', 'Volume', 'Open', 'High', 'Low'] + new_predictors

# 7. Train model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
model.fit(train[predictors], train['Target'])

# 8. Predict on test set
preds_proba = model.predict_proba(test[predictors])[:,1]
threshold = 0.6
preds = (preds_proba >= threshold).astype(int)

# 9. Show performance
precision = precision_score(test['Target'], preds)
st.subheader("✅ Model Precision")
st.write(f"Precision Score (Threshold = {threshold}): {precision:.3f}")

# 10. Target vs Prediction table
results_df = pd.DataFrame({
    'Target': test['Target'],
    'Prediction': preds
}, index=test.index)

st.dataframe(results_df.tail(20))

# 11. Plot Target vs Prediction
fig, ax = plt.subplots(figsize=(10,5))
results_df['Target'].plot(ax=ax, label='Actual', alpha=0.7)
results_df['Prediction'].plot(ax=ax, label='Predicted', alpha=0.7)
plt.legend()
plt.title("Target vs Prediction")
st.pyplot(fig)

# 12. MA50 and MA200
st.subheader("📈 MA50 and MA200")
sp500["MA200"] = sp500["Close"].rolling(200).mean()
fig_ma2, ax_ma2 = plt.subplots(figsize=(10,5))
ax_ma2.plot(sp500["Close"], label="Close", color='black')
ax_ma2.plot(sp500["MA50"], label="MA50", color='blue')
ax_ma2.plot(sp500["MA200"], label="MA200", color='orange')
ax_ma2.set_title("Close with MA50 & MA200")
ax_ma2.legend()
st.pyplot(fig_ma2)

# 13. Trend prediction on chart
st.subheader("🔍 Trend Predictions on Price Chart")
fig_pred, ax_pred = plt.subplots(figsize=(10,5))
ax_pred.plot(test.index, test["Close"], label="Close", color="gray")
ax_pred.scatter(test.index[preds==1], test["Close"][preds==1], label="Up", color="green", marker="^")
ax_pred.scatter(test.index[preds==0], test["Close"][preds==0], label="Down", color="red", marker="v")
ax_pred.legend()
ax_pred.set_title("Trend Prediction on Close Price")
st.pyplot(fig_pred)

# 14. Bollinger Bands
st.subheader("📊 Bollinger Bands (20-day)")
sp500["20_SMA"] = sp500["Close"].rolling(window=20).mean()
sp500["20_STD"] = sp500["Close"].rolling(window=20).std()
sp500["Upper"] = sp500["20_SMA"] + (2 * sp500["20_STD"])
sp500["Lower"] = sp500["20_SMA"] - (2 * sp500["20_STD"])

fig_bb, ax_bb = plt.subplots(figsize=(10,5))
ax_bb.plot(sp500["Close"], label="Close", color='black')
ax_bb.plot(sp500["Upper"], label="Upper Band", linestyle='--', color='red')
ax_bb.plot(sp500["Lower"], label="Lower Band", linestyle='--', color='blue')
ax_bb.fill_between(sp500.index, sp500["Lower"], sp500["Upper"], color='lightgray', alpha=0.3)
ax_bb.set_title("Bollinger Bands")
ax_bb.legend()
st.pyplot(fig_bb)

# 15. Volume vs Close
st.subheader("📉 Volume vs Close")
fig_vol, ax_vol = plt.subplots(figsize=(10,5))
ax_vol2 = ax_vol.twinx()
ax_vol.bar(sp500.index, sp500["Volume"], width=1.0, color='lightblue', label='Volume')
ax_vol2.plot(sp500["Close"], color='black', label='Close')
ax_vol.set_ylabel("Volume", color='blue')
ax_vol2.set_ylabel("Close Price", color='black')
ax_vol.set_title("Volume and Close Price")
st.pyplot(fig_vol)
