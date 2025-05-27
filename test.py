import streamlit as st
from sklearn.ensemble import RandomForestClassifier

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

# 2. Create moving averages and target
sp500["MA10"] = sp500["Close"].rolling(window=10).mean()
sp500["MA50"] = sp500["Close"].rolling(window=50).mean()
sp500["Target"] = (sp500["MA10"] > sp500["MA50"]).astype(int)
sp500 = sp500.dropna()

# 5. Add rolling features
horizons = [2, 5, 60, 250]
new_predictors = []
for horizon in horizons:
    rolling_avg = sp500['Close'].rolling(horizon).mean()
    sp500[f'Close_Ratio_{horizon}'] = sp500['Close'] / rolling_avg
    sp500[f'Trend_{horizon}'] = sp500['Target'].shift(1).rolling(horizon).sum()
    new_predictors += [f'Close_Ratio_{horizon}', f'Trend_{horizon}']

sp500 = sp500.dropna()

# 6. Train-test split
train_size = int(len(sp500)*0.8)
train = sp500.iloc[:train_size]
test = sp500.iloc[train_size:]
predictors = ['Close', 'Volume', 'Open', 'High', 'Low'] + new_predictors

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
model.fit(train[predictors], train['Target'])