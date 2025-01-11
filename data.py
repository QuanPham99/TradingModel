import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import util
import matplotlib.pyplot as plt

symbol = "AAPL"

# 1 minute interval data last 5 days
data = yf.download(symbol, period="5d", interval="1m")
data = data.ffill()
data.columns = ['_'.join(col) for col in data.columns]

print(data.head(10))

#### Moving Average
# 5 interval moving average
data["SMA_5"] = data[f"Close_{symbol}"].rolling(window=5).mean()
# 20 inteval moving average
data["SMA_20"] = data[f"Close_{symbol}"].rolling(window=20).mean()

#### Relative Strength Index (RSI)
data["RSI"] = util.calculate_rsi(data, symbol)

#### Exponential Moving Average (EMA)
data["EMA_9"] = data[f"Close_{symbol}"].ewm(span=9, adjust=False).mean()

#### Returns
data["Returns"] = data[f"Close_{symbol}"].pct_change()

# Resample to 5-minute intervals
resampled_data = data.resample('min').agg({
    f'Open_{symbol}': 'first',
    f'High_{symbol}': 'max',
    f'Low_{symbol}': 'min',
    f'Close_{symbol}': 'last',
    f'Volume_{symbol}': 'sum'
})

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[[f'Close_{symbol}', f'Volume_{symbol}']])
data[['Close_scaled', 'Volume_scaled']] = scaled_features
data.to_csv(f"{symbol}_processed.csv")

# Plot closing price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(data[f'Close_{symbol}'], label='Close Price')
plt.plot(data['SMA_5'], label='SMA 5', linestyle='--')
plt.plot(data['SMA_20'], label='SMA 20', linestyle='--')
plt.title(f"{symbol} Closing Price with Moving Averages")
plt.legend()
plt.show()
