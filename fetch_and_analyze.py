import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import joblib
from price_action_engine import PriceActionPatterns

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="15m", limit=1000):
    """
    Binance'ten OHLCV veri çeker ve düzenler.
    """
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df

def analyze_patterns_and_ml(df, timeframe="15m"):
    """
    PriceActionPatterns ile pattern ve ML analizlerini döndürür.
    """
    # LSTM model ve scaler yükle
    try:
        from tensorflow.keras.models import load_model
        lstm_model = load_model("lstm_model.h5")
        scaler = joblib.load("lstm_scaler.save")
    except:
        lstm_model = None
        scaler = None

    pap = PriceActionPatterns(df, timeframe=timeframe, lstm_model=lstm_model)
    # Pattern raporu
    pattern_report = pap.summary_report()
    # ML/LSTM sinyali
    lstm_signal = pap.lstm_signal()
    # ML classifier ve anomaly score (eğer varsa)
    ml_cls = pap.ml_signal_classification() if hasattr(pap, "ml_signal_classification") else None
    anomaly = pap.anomaly_score() if hasattr(pap, "anomaly_score") else None

    return {
        "patterns": pap.get_all_patterns(),
        "pattern_report": pattern_report,
        "lstm_signal": lstm_signal,
        "ml_classification": ml_cls,
        "anomaly_score": anomaly
    }

def prepare_ml_features(df):
    """
    Modelin eğitimi/öngörüsü için feature engineering ve scaling.
    """
    # Feature engineering (modelde kullanılanlarla uyumlu)
    import talib
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
    # Binary pattern features (örnek)
    df['ema_golden_cross'] = ((df['EMA_50'].shift(1) < df['EMA_200'].shift(1)) & (df['EMA_50'] > df['EMA_200'])).astype(int)
    df['ema_death_cross'] = ((df['EMA_50'].shift(1) > df['EMA_200'].shift(1)) & (df['EMA_50'] < df['EMA_200'])).astype(int)
    df['sma_golden_cross'] = ((df['SMA_20'].shift(1) < df['SMA_50'].shift(1)) & (df['SMA_20'] > df['SMA_50'])).astype(int)
    df['sma_death_cross'] = ((df['SMA_20'].shift(1) > df['SMA_50'].shift(1)) & (df['SMA_20'] < df['SMA_50'])).astype(int)
    df['bollinger_breakout_up'] = ((df['close'] > df['BB_upper']) & (df['close'].shift(1) <= df['BB_upper'].shift(1))).astype(int)
    df['bollinger_breakout_down'] = ((df['close'] < df['BB_lower']) & (df['close'].shift(1) >= df['BB_lower'].shift(1))).astype(int)
    df['rsi_overbought'] = (df['RSI_14'] > 70).astype(int)
    df['rsi_oversold'] = (df['RSI_14'] < 30).astype(int)
    df['adx_strong_trend'] = (df['ADX_14'] > 25).astype(int)
    df['macd_hist_cross_up'] = ((df['MACD_hist'].shift(1) < 0) & (df['MACD_hist'] > 0)).astype(int)
    df['macd_hist_cross_down'] = ((df['MACD_hist'].shift(1) > 0) & (df['MACD_hist'] < 0)).astype(int)
    df['volume_mean_20'] = df['volume'].rolling(20).mean()
    df['volume_spike'] = (df['volume'] > 2*df['volume_mean_20']).astype(int)

    df = df.dropna().reset_index(drop=True)
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'RSI_14', 'EMA_50', 'EMA_200', 'MACD', 'MACD_signal', 'MACD_hist',
        'BB_upper', 'BB_middle', 'BB_lower', 'ADX_14', 'SMA_20', 'SMA_50',
        'ema_golden_cross', 'ema_death_cross', 'sma_golden_cross', 'sma_death_cross',
        'bollinger_breakout_up', 'bollinger_breakout_down',
        'rsi_overbought', 'rsi_oversold', 'adx_strong_trend',
        'macd_hist_cross_up', 'macd_hist_cross_down', 'volume_spike'
    ]
    try:
        scaler = joblib.load("lstm_scaler.save")
        scaled = scaler.transform(df[feature_cols])
    except:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler

if __name__ == "__main__":
    # Örnek analiz akışı (CLI/main değil, modül olarak kullanılacak şekilde sade)
    df = fetch_binance_ohlcv(symbol="BTCUSDT", interval="15m", limit=1000)
    analysis = analyze_patterns_and_ml(df)
    print("Pattern Report:\n", analysis["pattern_report"])
    print("Son LSTM Sinyali:", analysis["lstm_signal"])
    print("ML Sınıflandırıcı:", analysis["ml_classification"])
    print("Anomali Skoru:", analysis["anomaly_score"])