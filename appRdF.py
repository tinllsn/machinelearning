import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def backtest_target_strategy(data, preds):
    returns = data["Close"].pct_change().shift(-1)  # lợi nhuận ngày hôm sau
    strategy_returns = returns * preds  # chỉ tính lợi nhuận khi dự đoán là 1 (long)
    cumulative_target_returns = (strategy_returns + 1).cumprod()
    buy_and_hold = (returns + 1).cumprod()
    return cumulative_target_returns, buy_and_hold

def backtest_ma_strategy(data, preds_ma):
    returns = data["Close"].pct_change().shift(-1)
    position = preds_ma  # giữ lệnh nếu mô hình dự đoán đang trong xu hướng tăng
    strategy_returns = returns * position
    cumulative_ma_returns = (strategy_returns + 1).cumprod()
    buy_and_hold = (returns + 1).cumprod()
    return cumulative_ma_returns, buy_and_hold

st.set_page_config(page_title="Random Forest Stock Prediction", layout="centered")

st.title("📈 Random Forest Stock Market Prediction")

# 1. Load S&P 500 data
@st.cache_data
def load_sp500():
    sp500 = yf.Ticker("^GSPC")
    df = sp500.history(period="max")
    
    return df

st.text('Loading data...')
sp500 = load_sp500()
sp500 = sp500.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
# sp500 = sp500.loc["2021-01-01":].copy()

# Convert index to timezone-naive datetime
sp500.index = sp500.index.tz_localize(None)

st.success('Data loaded successfully!')
st.write(sp500)

# Add date range selector
st.subheader('📅 Select Date Range')
# col1, col2 = st.columns(2)
# with col1:
#     start_date = st.date_input('Start Date', sp500.index.min().date())
# with col2:
#     end_date = st.date_input('End Date', sp500.index.max().date())

# # Filter data based on selected date range
# filtered_sp500 = sp500.loc[start_date:end_date]
# Lấy danh sách các ngày có sẵn trong dữ liệu
available_dates = sp500.index.date

# Chuyển sang list để dùng trong selectbox
date_options = list(available_dates)

col1, col2 = st.columns(2)

with col1:
    start_date = st.selectbox("Start Date", date_options, index=0)
with col2:
    end_date = st.selectbox("End Date", date_options, index=len(date_options)-1)

# Kiểm tra thứ tự ngày
if start_date > end_date:
    st.error("⚠️ Start Date must be before or equal to End Date.")
    st.stop()
else:
    filtered_sp500 = sp500.loc[str(start_date):str(end_date)]
    st.success(f"✅ Showing data from **{start_date}** to **{end_date}**")



st.subheader('📊 Close Price Data')
st.line_chart(filtered_sp500['Close'])

# 2. Create target
#previous close
filtered_sp500["Tomorrow"] = filtered_sp500["Close"].shift(-1)
filtered_sp500["Target"] = (filtered_sp500["Tomorrow"] > filtered_sp500["Close"]).astype(int)
#ma10 &ma50
filtered_sp500["MA10"] = filtered_sp500["Close"].rolling(window=10).mean()
filtered_sp500["MA50"] = filtered_sp500["Close"].rolling(window=50).mean()
filtered_sp500["TargetMA"] = (filtered_sp500["MA10"] > filtered_sp500["MA50"]).astype(int)

# filtered_sp500 = filtered_sp500.dropna()

# 5. Add rolling features
horizons = [2, 5, 60, 250]
new_predictors = []
for horizon in horizons:
    rolling_avg = filtered_sp500['Close'].rolling(horizon).mean()
    filtered_sp500[f'Close_Ratio_{horizon}'] = filtered_sp500['Close'] / rolling_avg
    filtered_sp500[f'Trend_{horizon}'] = filtered_sp500['Target'].shift(1).rolling(horizon).sum()
    new_predictors += [f'Close_Ratio_{horizon}', f'Trend_{horizon}']

filtered_sp500 = filtered_sp500.dropna()

#Train-test split
train_size = int(len(filtered_sp500)*0.8)
train = filtered_sp500.iloc[:train_size]
test = filtered_sp500.iloc[train_size:]
predictors = ['Close', 'Volume', 'Open', 'High', 'Low'] + new_predictors
predictors1 = ['Close', 'Volume', 'Open', 'High', 'Low']

# display compare 2 target: Tommorow-Close
st.subheader('🎯 Target Based on Tomorrow-Close')
sp500_Tomorrow = ['Open', 'High', 'Low', 'Close', 'Volume', 'Tomorrow', 'Target']
st.write(filtered_sp500[sp500_Tomorrow])
#no horizon
#Train model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
model.fit(train[predictors1], train['Target'])

#Predict on test set
preds_proba = model.predict_proba(test[predictors1])[:,1]
threshold = 0.6
preds = (preds_proba >= threshold).astype(int)

#Show performance
precision_Tomorrow = precision_score(test['Target'], preds)
st.subheader("✅ Model Precision")
st.write(f"Precision Score (Threshold = {threshold}): {precision_Tomorrow:.3f}")

# #Target vs Prediction table
# results_df_Tomorrow = pd.DataFrame({
#     'Target_Tomorrow': test['Target'],
#     'Prediction': preds
# }, index=test.index)

# st.dataframe(results_df_Tomorrow.tail(20))

# #Plot Target vs Prediction
# fig, ax = plt.subplots(figsize=(10,5))
# results_df_Tomorrow['Target_Tomorrow'].plot(ax=ax, label='Actual', alpha=0.7)
# results_df_Tomorrow['Prediction'].plot(ax=ax, label='Predicted', alpha=0.7)
# plt.legend()
# plt.title("Target_Tomorrow vs Prediction")
# st.pyplot(fig)

#use horizon
st.subheader('Add Features')
sp500_Tomorrow_ft = ['Open', 'High', 'Low', 'Close', 'Volume', 'Tomorrow', 'Target', 'Close_Ratio_2', 'Trend_2', 'Close_Ratio_5', 'Trend_5', 'Close_Ratio_60', 'Trend_60', 'Close_Ratio_250', 'Trend_250']
st.write(filtered_sp500[sp500_Tomorrow_ft])
#Train model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
model.fit(train[predictors], train['Target'])

#Predict on test set
preds_proba = model.predict_proba(test[predictors])[:,1]
threshold = 0.6
preds = (preds_proba >= threshold).astype(int)

#Show performance
precision_Tomorrow = precision_score(test['Target'], preds)
st.subheader("✅ Model Precision used Horizon")
st.write(f"Precision Score (Threshold = {threshold}): {precision_Tomorrow:.3f}")


#Target vs Prediction table
results_df_Tomorrow = pd.DataFrame({
    'Target_Tomorrow': test['Target'],
    'Prediction': preds
}, index=test.index)

st.dataframe(results_df_Tomorrow.tail(20))

#Plot Target vs Prediction
fig, ax = plt.subplots(figsize=(10,5))
results_df_Tomorrow['Target_Tomorrow'].plot(ax=ax, label='Actual', alpha=0.7)
results_df_Tomorrow['Prediction'].plot(ax=ax, label='Predicted', alpha=0.7)
plt.legend()
plt.title("Target_Tomorrow vs Prediction")
st.pyplot(fig)
#backtest
st.subheader("📈 Backtest Strategy: Predict Tomorrow's Price Direction")
cum_return, bh_return = backtest_target_strategy(test, preds)
fig_bt1, ax_bt1 = plt.subplots(figsize=(10,5))
ax_bt1.plot(cum_return.index, cum_return, label="Strategy Return")
ax_bt1.plot(bh_return.index, bh_return, label="Buy & Hold")
ax_bt1.set_title("Backtest: Target Strategy vs Buy & Hold")
ax_bt1.legend()
st.pyplot(fig_bt1)


# display compare 2 target: MA
st.subheader('🎯 Target Based on MA10 & MA50')
sp500_MA = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'TargetMA']
st.write(filtered_sp500[sp500_MA])

#no horizon
#trainModel
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
model.fit(train[predictors1], train['TargetMA'])
#precision
preds_proba_ma = model.predict_proba(test[predictors1])[:,1]
preds_ma = (preds_proba_ma >= threshold).astype(int)
precision_MA = precision_score(test['TargetMA'], preds_ma)
st.subheader("✅ Model Precision")
st.write(f"Precision Score (Threshold = {threshold}): {precision_MA:.3f}")

#use horizon
st.subheader('Add Features')
sp500_MA_ft = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'TargetMA', 'Close_Ratio_2', 'Trend_2', 'Close_Ratio_5', 'Trend_5', 'Close_Ratio_60', 'Trend_60', 'Close_Ratio_250', 'Trend_250']
st.write(filtered_sp500[sp500_MA_ft])

#trainModel
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
model.fit(train[predictors], train['TargetMA'])
#precision
preds_proba_ma = model.predict_proba(test[predictors])[:,1]
preds_ma = (preds_proba_ma >= threshold).astype(int)
precision_MA = precision_score(test['TargetMA'], preds_ma)
st.subheader("✅ Model Precision used Horizon")
st.write(f"Precision Score (Threshold = {threshold}): {precision_MA:.3f}")

# TargetMA vs Prediction table
results_df_MA = pd.DataFrame({
    'TargetMA': test['TargetMA'],
    'Prediction': preds_ma
}, index=test.index)

st.dataframe(results_df_MA.tail(20))

#Plot Target vs Prediction
fig, ax = plt.subplots(figsize=(10,5))
results_df_MA['TargetMA'].plot(ax=ax, label='Actual', alpha=0.7)
results_df_MA['Prediction'].plot(ax=ax, label='Predicted', alpha=0.7)
plt.legend()
plt.title("TargetMA vs Prediction")
st.pyplot(fig)

#backtest
st.subheader("📈 Backtest Strategy: Predict MA10 > MA50 Trend")
cum_return_ma, bh_return_ma = backtest_ma_strategy(test, preds_ma)
fig_bt2, ax_bt2 = plt.subplots(figsize=(10,5))
ax_bt2.plot(cum_return_ma.index, cum_return_ma, label="Strategy Return")
ax_bt2.plot(bh_return_ma.index, bh_return_ma, label="Buy & Hold")
ax_bt2.set_title("Backtest: MA Strategy vs Buy & Hold")
ax_bt2.legend()
st.pyplot(fig_bt2)


# 3. Plot price with MA10 and MA50 (Plotly)
# st.subheader('📉 Interactive Price Chart with MA10 and MA50')
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=filtered_sp500.index, y=filtered_sp500['Close'], mode='lines', name='Close'))
# fig.add_trace(go.Scatter(x=filtered_sp500.index, y=filtered_sp500['MA10'], mode='lines', name='MA10'))
# fig.add_trace(go.Scatter(x=filtered_sp500.index, y=filtered_sp500['MA50'], mode='lines', name='MA50'))

# fig.update_layout(
#     xaxis_title='Date',
#     yaxis_title='Price',
#     hovermode='x unified',
#     template='plotly_white',
#     legend=dict(x=0, y=1),
#     height=600,
#     xaxis=dict(
#         rangeslider=dict(visible=True),
#         type="date"
#     )
# )
# st.plotly_chart(fig, use_container_width=True)

# 4. Simple matplotlib plot
# st.subheader('🧾 Matplotlib: Price with MA10 and MA50')
# fig_ma, ax = plt.subplots(figsize=(12,6))
# filtered_sp500['Close'].plot(ax=ax, label='Close', alpha=0.5)
# filtered_sp500['MA10'].plot(ax=ax, label='MA10', linestyle='--')
# filtered_sp500['MA50'].plot(ax=ax, label='MA50', linestyle='--')
# plt.legend()
# plt.grid(True, alpha=0.3)
# st.pyplot(fig_ma)


#ss

st.subheader("📊 Backtest Comparison: All Strategies")

fig_compare, ax_compare = plt.subplots(figsize=(12, 6))
ax_compare.plot(cum_return.index, cum_return, label="Target Strategy (Next Day Up)")
ax_compare.plot(cum_return_ma.index, cum_return_ma, label="TargetMA Strategy (MA10 > MA50)")
ax_compare.plot(bh_return.index, bh_return, label="Buy & Hold Strategy")

ax_compare.set_title("Backtest Comparison: Target vs MA vs Buy & Hold")
ax_compare.set_xlabel("Date")
ax_compare.set_ylabel("Cumulative Return")
ax_compare.grid(True)
ax_compare.legend()
st.pyplot(fig_compare)






# 12. MA50 and MA200
st.subheader("📈 MA50 and MA200")
filtered_sp500["MA50"] = filtered_sp500["Close"].rolling(50).mean()
filtered_sp500["MA200"] = filtered_sp500["Close"].rolling(200).mean()
fig_ma2, ax_ma2 = plt.subplots(figsize=(10,5))
ax_ma2.plot(filtered_sp500["Close"], label="Close", color='black')
ax_ma2.plot(filtered_sp500["MA50"], label="MA50", color='blue')
ax_ma2.plot(filtered_sp500["MA200"], label="MA200", color='orange')
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
# st.subheader("📊 Bollinger Bands (20-day)")
# filtered_sp500["20_SMA"] = filtered_sp500["Close"].rolling(window=20).mean()
# filtered_sp500["20_STD"] = filtered_sp500["Close"].rolling(window=20).std()
# filtered_sp500["Upper"] = filtered_sp500["20_SMA"] + (2 * filtered_sp500["20_STD"])
# filtered_sp500["Lower"] = filtered_sp500["20_SMA"] - (2 * filtered_sp500["20_STD"])

# fig_bb, ax_bb = plt.subplots(figsize=(10,5))
# ax_bb.plot(filtered_sp500["Close"], label="Close", color='black')
# ax_bb.plot(filtered_sp500["Upper"], label="Upper Band", linestyle='--', color='red')
# ax_bb.plot(filtered_sp500["Lower"], label="Lower Band", linestyle='--', color='blue')
# ax_bb.fill_between(filtered_sp500.index, filtered_sp500["Lower"], filtered_sp500["Upper"], color='lightgray', alpha=0.3)
# ax_bb.set_title("Bollinger Bands")
# ax_bb.legend()
# st.pyplot(fig_bb)

# 15. Volume vs Close
# st.subheader("📉 Volume vs Close")
# fig_vol, ax_vol = plt.subplots(figsize=(10,5))
# ax_vol2 = ax_vol.twinx()
# ax_vol.bar(filtered_sp500.index, filtered_sp500["Volume"], width=1.0, color='lightblue', label='Volume')
# ax_vol2.plot(filtered_sp500["Close"], color='black', label='Close')
# ax_vol.set_ylabel("Volume", color='blue')
# ax_vol2.set_ylabel("Close Price", color='black')
# ax_vol.set_title("Volume and Close Price")
# st.pyplot(fig_vol)



