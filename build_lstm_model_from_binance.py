import numpy as np
import pandas as pd
from binance.client import Client
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Bidirectional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import talib
import joblib

# --- PARAMETRELER ---
symbol = "BTCUSDT"
interval = "15m"
limit = 2000
window = 30
EPOCHS = 12
BATCH_SIZE = 32

# --- Binance API anahtarsız da public veri çeker ---
client = Client()

# 1. Binance'ten veri çek
klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
])
df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 2. Feature Engineering: TA, Pattern, Modern ML
df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
macd, macdsig, macdhist = talib.MACD(df['close'])
df['MACD'] = macd
df['MACD_signal'] = macdsig
df['MACD_hist'] = macdhist
upper, middle, lower = talib.BBANDS(df['close'])
df['BB_upper'], df['BB_middle'], df['BB_lower'] = upper, middle, lower
df['ADX_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)

# EMA/SMA crossover (binary)
df['ema_golden_cross'] = ((df['EMA_50'].shift(1) < df['EMA_200'].shift(1)) & (df['EMA_50'] > df['EMA_200'])).astype(int)
df['ema_death_cross'] = ((df['EMA_50'].shift(1) > df['EMA_200'].shift(1)) & (df['EMA_50'] < df['EMA_200'])).astype(int)
df['sma_golden_cross'] = ((df['SMA_20'].shift(1) < df['SMA_50'].shift(1)) & (df['SMA_20'] > df['SMA_50'])).astype(int)
df['sma_death_cross'] = ((df['SMA_20'].shift(1) > df['SMA_50'].shift(1)) & (df['SMA_20'] < df['SMA_50'])).astype(int)

# Bollinger breakout
df['bollinger_breakout_up'] = ((df['close'] > df['BB_upper']) & (df['close'].shift(1) <= df['BB_upper'].shift(1))).astype(int)
df['bollinger_breakout_down'] = ((df['close'] < df['BB_lower']) & (df['close'].shift(1) >= df['BB_lower'].shift(1))).astype(int)

df['rsi_overbought'] = (df['RSI_14'] > 70).astype(int)
df['rsi_oversold'] = (df['RSI_14'] < 30).astype(int)
df['adx_strong_trend'] = (df['ADX_14'] > 25).astype(int)
df['macd_hist_cross_up'] = ((df['MACD_hist'].shift(1) < 0) & (df['MACD_hist'] > 0)).astype(int)
df['macd_hist_cross_down'] = ((df['MACD_hist'].shift(1) > 0) & (df['MACD_hist'] < 0)).astype(int)
df['volume_mean_20'] = df['volume'].rolling(20).mean()
df['volume_spike'] = (df['volume'] > 2*df['volume_mean_20']).astype(int)

# --- Modern pattern/volatility/chop zone feature'ları ---
# Choppiness Index (trend/range ayrımı için)
def choppiness_index(close, high, low, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return 100 * np.log10(tr.rolling(period).sum() / (high.rolling(period).max() - low.rolling(period).min())) / np.log10(period)
df['Choppiness'] = choppiness_index(df['close'], df['high'], df['low'])

# Volatility (standart sapma)
df['Volatility'] = df['close'].rolling(20).std()

# Harmonik ve gelişmiş patternler için dummy feature (placeholder, gerçek patternler production entegrasyonunda eklenir)
df['harmonic_pattern_found'] = 0  # (pattern tarayıcıdan üretilebilir)

# --- NaN temizle ---
df = df.dropna().reset_index(drop=True)

# 3. Feature Scaling
feature_cols = [
    'open', 'high', 'low', 'close', 'volume',
    'RSI_14', 'EMA_50', 'EMA_200', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_middle', 'BB_lower', 'ADX_14', 'SMA_20', 'SMA_50',
    'ema_golden_cross', 'ema_death_cross', 'sma_golden_cross', 'sma_death_cross',
    'bollinger_breakout_up', 'bollinger_breakout_down',
    'rsi_overbought', 'rsi_oversold', 'adx_strong_trend',
    'macd_hist_cross_up', 'macd_hist_cross_down', 'volume_spike',
    'Choppiness', 'Volatility', 'harmonic_pattern_found'
]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[feature_cols])

# 4. Window'lu veri seti oluştur (ör: 30 barlık pencere ile "gelecek close artacak mı?")
X = []
y = []
for i in range(window, len(scaled)-1):
    X.append(scaled[i-window:i])
    y.append(int(df['close'].iloc[i+1] > df['close'].iloc[i]))
X = np.array(X)
y = np.array(y)

# 5. Model (Daha derin, normalize edilmiş, modern LSTM/GRU/ensemble altyapısına uygun)
model = Sequential([
    LayerNormalization(input_shape=(window, X.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    LSTM(32),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
model.save("lstm_model.h5")
print("Ekstra pattern ve modern indikatörlerle eğitilmiş model kaydedildi: lstm_model.h5")

# 6. Scaler kaydet
joblib.dump(scaler, "lstm_scaler.save")
print("Scaler da kaydedildi: lstm_scaler.save")