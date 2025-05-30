"""
TÜM USDT paritelerinin 5m, 15m, 30m, 1h, 4h, 1d, 1w verisi 2018'den bugüne çekilir.
Her zaman diliminde CANLIDA ARANAN TÜM pattern ve teknik analiz fonksiyonları ile patternler aranır.
Her pattern için TP/SL, yön, time_frame ve teknik veriler ile birlikte dataset'e kaydedilir.
LabelEncoder ve MinMaxScaler ile 5 feature (entry, vol, direction_encoded, pattern_encoded, RSI_14) normalize edilir, kaydedilir.
pattern ve direction label'ları normalize_pattern_label ile normalize edilir!
"""

import pandas as pd
import numpy as np
import time
from binance.client import Client
from ta_engine import TechnicalAnalysisEngine
from price_action_engine import PriceActionPatterns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

# --- Ayarlar ---
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d", "1w"]
START_DATE = "1 Jan, 2018"

BINANCE_API_KEY = 'YOUR_API_KEY'
BINANCE_API_SECRET = 'YOUR_API_SECRET'
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def normalize_pattern_label(label):
    if not isinstance(label, str):
        return str(label)
    return label.strip().lower().replace(" ", "_")

def get_usdt_symbols():
    info = client.get_exchange_info()
    syms = []
    for s in info['symbols']:
        if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING' and s['isSpotTradingAllowed']:
            if not s['symbol'].endswith('BULLUSDT') and not s['symbol'].endswith('BEARUSDT'):
                syms.append(s['symbol'])
    return syms

def fetch_all_ohlcv(symbol, interval='1d', start_str=START_DATE):
    klines = []
    last = start_str
    while True:
        chunk = client.get_historical_klines(symbol, interval, last, limit=1000)
        if not chunk or len(chunk) == 0:
            break
        klines += chunk
        last = pd.to_datetime(chunk[-1][0], unit='ms') + pd.Timedelta(minutes=1)
        last = last.strftime('%d %b, %Y')
        if len(chunk) < 1000:
            break
        time.sleep(0.5)  # Rate limit
    if not klines:
        return None
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

def tp_sl_check(df, entry_idx, direction="long", tp_perc=0.05, sl_perc=0.03, lookahead=15):
    if entry_idx >= len(df) - 2:
        return "yetersiz veri"
    entry_price = df['close'].iloc[entry_idx]
    sl_price = entry_price * (1 - sl_perc) if direction == "long" else entry_price * (1 + sl_perc)
    tp_price = entry_price * (1 + tp_perc) if direction == "long" else entry_price * (1 - tp_perc)
    closes = df['close'].iloc[entry_idx+1:entry_idx+1+lookahead]
    for price in closes:
        if direction == "long":
            if price >= tp_price:
                return "TP"
            if price <= sl_price:
                return "SL"
        else:
            if price <= tp_price:
                return "TP"
            if price >= sl_price:
                return "SL"
    return "none"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi

def extract_patterns(df, symbol, time_frame):
    # --- CANLIDA ARANAN TÜM PATTERNLER VE TEKNİK ANALİZLER ---
    # 1. Fiyat aksiyonu patternleri (PriceActionPatterns)
    pa = PriceActionPatterns(df)
    pa_patterns = pa.get_all_patterns() if hasattr(pa, "get_all_patterns") else []
    # 2. Teknik analiz patternleri (TechnicalAnalysisEngine) -- eğer pattern dönüyorsa
    ta = TechnicalAnalysisEngine(df)
    ta_patterns = ta.get_all_patterns() if hasattr(ta, "get_all_patterns") else []

    all_patterns = []
    if pa_patterns:
        all_patterns.extend(pa_patterns)
    if ta_patterns:
        all_patterns.extend(ta_patterns)

    if not all_patterns:
        print(f"{symbol} {time_frame}: Hiç pattern bulunamadı!")
    else:
        found_types = set([normalize_pattern_label(p['type']) for p in all_patterns if 'type' in p])
        print(f"{symbol} {time_frame} için bulunan patternler: {found_types}")
    result = []
    rsi_14_all = compute_rsi(df['close'], period=14)
    for pat in all_patterns:
        pat_type = normalize_pattern_label(pat['type'])
        direction = "long"
        # Gelişmiş yön tespiti
        if "bear" in pat_type or "shooting" in pat_type or "down" in pat_type or "dark_cloud" in pat_type or "resistance_break" in pat_type or "triple_top" in pat_type or "double_top" in pat_type or "head_and_shoulders" in pat_type or "fractal_high" in pat_type:
            direction = "short"
        elif "bull" in pat_type or "morning_star" in pat_type or "bottom" in pat_type or "inverse" in pat_type or "support_break" in pat_type or "triple_bottom" in pat_type or "double_bottom" in pat_type or "falling_wedge" in pat_type or "fractal_low" in pat_type:
            direction = "long"
        if 'direction' in pat:
            direction = normalize_pattern_label(pat['direction'])
        else:
            direction = normalize_pattern_label(direction)
        # Index çıkarımı
        if 'idx' in pat:
            idx = pat['idx']
        elif 'indexes' in pat and isinstance(pat['indexes'], list) and pat['indexes']:
            idx = pat['indexes'][-1]
        elif 'peaks' in pat and isinstance(pat['peaks'], list) and pat['peaks']:
            idx = pat['peaks'][-1]
        elif 'troughs' in pat and isinstance(pat['troughs'], list) and pat['troughs']:
            idx = pat['troughs'][-1]
        elif 'bottom' in pat and isinstance(pat['bottom'], (int, float)):
            idx = df[df['close'] == pat['bottom']].index[-1]
        elif 'top' in pat and isinstance(pat['top'], (int, float)):
            idx = df[df['close'] == pat['top']].index[-1]
        else:
            idx = -2
        if idx < 0 or idx >= len(df):
            continue
        tp_sl = tp_sl_check(df, idx, direction)
        if tp_sl not in ["TP", "SL"]:
            continue
        result.append({
            "date": df['timestamp'].iloc[idx],
            "pattern": pat_type,
            "direction": direction,
            "entry": df['close'].iloc[idx],
            "tp_sl": tp_sl,
            "symbol": symbol,
            "idx": idx,
            "vol": df['volume'].iloc[idx],
            "RSI_14": float(rsi_14_all.iloc[idx]) if not pd.isna(rsi_14_all.iloc[idx]) else 50.0,
            "time_frame": time_frame
        })
    return result

def main():
    symbols = get_usdt_symbols()
    print(f"{len(symbols)} USDT parite bulundu.")
    all_examples = []
    for i, sym in enumerate(symbols):
        for tf in TIMEFRAMES:
            try:
                print(f"{i+1}/{len(symbols)}: {sym} {tf} veri çekiliyor...")
                df = fetch_all_ohlcv(sym, tf, START_DATE)
                if df is None or len(df) < 100:
                    print(f"{sym} {tf}: Veri yok/çok az.")
                    continue
                print(f"{sym} {tf}: Pattern taranıyor...")
                examples = extract_patterns(df, sym, tf)
                all_examples.extend(examples)
            except Exception as e:
                print(f"{sym} {tf} hata: {e}")
            time.sleep(0.5)

    outdf = pd.DataFrame(all_examples)
    outdf = outdf[outdf['tp_sl'].isin(['TP','SL'])]

    if len(outdf) == 0:
        print("Hiç örnek bulunamadı, dosya yazılmadı!")
        return

    outdf['direction'] = outdf['direction'].apply(normalize_pattern_label)
    outdf['pattern'] = outdf['pattern'].apply(normalize_pattern_label)
    le_dir = LabelEncoder()
    le_pat = LabelEncoder()
    outdf['direction_encoded'] = le_dir.fit_transform(outdf['direction'])
    outdf['pattern_encoded'] = le_pat.fit_transform(outdf['pattern'])

    # 5 feature: entry, vol, direction_encoded, pattern_encoded, RSI_14
    X = outdf[["entry", "vol", "direction_encoded", "pattern_encoded", "RSI_14"]].values
    y = (outdf['tp_sl'] == "TP").astype(int).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("le_dir.pkl", "wb") as f:
        pickle.dump(le_dir, f)
    with open("le_pat.pkl", "wb") as f:
        pickle.dump(le_pat, f)

    # Son dataset ve hedef
    dataset = pd.DataFrame(X, columns=["entry", "vol", "direction_encoded", "pattern_encoded", "RSI_14"])
    dataset["tp"] = y
    dataset["pattern"] = outdf["pattern"].values
    dataset["direction"] = outdf["direction"].values
    dataset["symbol"] = outdf["symbol"].values
    dataset["time_frame"] = outdf["time_frame"].values
    dataset.to_csv("pattern_dataset_ready.csv", index=False)

    print("Pattern classes:", le_pat.classes_)
    print("Direction classes:", le_dir.classes_)
    print(f"Toplam {len(dataset)} örnek pattern_dataset_ready.csv dosyasına yazıldı.")

if __name__ == "__main__":
    main()