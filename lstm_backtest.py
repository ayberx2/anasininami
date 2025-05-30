import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import pattern_stats
from datetime import datetime

def run_lstm_backtest(
    symbol="BTCUSDT",
    interval="15m",
    limit=1000,
    window=30,
    tp_ratio=0.01,
    sl_ratio=0.01,
    conf_long=0.7,
    conf_short=0.3,
    model_path="lstm_model.h5",
    stats_pattern_prefix="lstm",
    pattern_stats_enabled=True,
    verbose=True
):
    """
    LSTM model ile otomatik sinyal üreten ve TP/SL ile backtest yapan fonksiyon.
    Sonuçlar pattern_stats'a kaydedilir.
    """
    # --- Binance'ten veri çek ---
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # --- Normalize ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    model = load_model(model_path)

    preds = []
    for i in range(window, len(df)-2):  # -2: TP/SL için bir bar ileride veri gerekiyor
        window_data = scaled[i-window:i].reshape(1, window, 5)
        conf = float(model.predict(window_data, verbose=0)[0,0])
        preds.append({
            "index": i,
            "confidence": conf,
            "future_close": df['close'].iloc[i+1],
            "future_high": df['high'].iloc[i+1],
            "future_low": df['low'].iloc[i+1],
            "current_close": df['close'].iloc[i],
            "timestamp": df['timestamp'].iloc[i]
        })

    # --- TP/SL ile backtest ---
    results = []
    for p in preds:
        action = None
        if p["confidence"] > conf_long:
            action = "long"
        elif p["confidence"] < conf_short:
            action = "short"

        if action:
            entry = p["current_close"]
            if action == "long":
                tp = entry * (1 + tp_ratio)
                sl = entry * (1 - sl_ratio)
                # Önce TP mi SL mi geliyor?
                if p["future_high"] >= tp:
                    win = True
                    close_type = "TP"
                elif p["future_low"] <= sl:
                    win = False
                    close_type = "SL"
                else:
                    win = p["future_close"] > entry
                    close_type = "CLOSE"
            elif action == "short":
                tp = entry * (1 - tp_ratio)
                sl = entry * (1 + sl_ratio)
                if p["future_low"] <= tp:
                    win = True
                    close_type = "TP"
                elif p["future_high"] >= sl:
                    win = False
                    close_type = "SL"
                else:
                    win = p["future_close"] < entry
                    close_type = "CLOSE"

            results.append(win)
            # Pattern key: lstm_long_15m veya lstm_short_15m
            if pattern_stats_enabled:
                pattern_key = f"{stats_pattern_prefix}_{action}_{interval}"
                pattern_stats.update_pattern_stats(
                    pattern_key=pattern_key,
                    is_win=win,
                    close_time=p["timestamp"].isoformat(),
                    entry=entry,
                    exit=p["future_close"],
                    tp=tp,
                    sl=sl,
                    pnl=(p["future_close"]-entry if action == "long" else entry-p["future_close"]),
                    close_type=close_type,
                    extra={"confidence": p["confidence"]}
                )

    if results:
        winrate = sum(results) / len(results)
        print(f"LSTM confidence + TP/SL backtest sonucu: {len(results)} trade, Winrate: %{winrate*100:.1f}")
    else:
        print("Hiç trade açılmadı (sınırları/conf sınırlarını düşürüp artırabilirsin).")

# Komut satırından veya ana dosya olarak çalıştırırsan:
if __name__ == "__main__":
    run_lstm_backtest()