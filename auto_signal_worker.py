import yaml
import asyncio
import pandas as pd
from binance.client import Client
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, CommandHandler
from telegram.error import BadRequest
from price_action_engine import PriceActionPatterns, normalize_pattern_label
import pattern_stats
from ta_engine import TechnicalAnalysisEngine
from datetime import datetime, timedelta

import numpy as np
import pickle

# --- Config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
BINANCE_API_KEY = config["binance"]["api_key"]
BINANCE_API_SECRET = config["binance"]["api_secret"]
TELEGRAM_TOKEN = config["telegram"]["bot_token"]
CHAT_ID = config["telegram"]["chat_id"]

# --- Binance ve Telegram client ---
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_TOKEN)

signal_log = {}

# --- Pattern başarı oranı eşik değeri ---
MIN_PATTERN_SUCCESS_RATE = 0.55

# --- LSTM, Scaler ve Encoder yükleme ---
try:
    from tensorflow.keras.models import load_model
    LSTM_MODEL_PATH = "lstm_model.h5"
    SCALER_PATH = "scaler.pkl"
    LE_DIR_PATH = "le_dir.pkl"
    LE_PAT_PATH = "le_pat.pkl"
    lstm_model = load_model(LSTM_MODEL_PATH) if LSTM_MODEL_PATH else None
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LE_DIR_PATH, "rb") as f:
        le_dir = pickle.load(f)
    with open(LE_PAT_PATH, "rb") as f:
        le_pat = pickle.load(f)
    ML_READY = True
    print("Pattern classes:", le_pat.classes_)
    print("Direction classes:", le_dir.classes_)
except Exception as e:
    print(f"LSTM/Scaler/Encoder yükleme hatası: {e}")
    lstm_model = None
    scaler = None
    le_dir = None
    le_pat = None
    ML_READY = False

# --- Feature listesi (her yerde aynı olacak!) ---
FEATURE_LIST = ["entry", "vol", "direction_encoded", "pattern_encoded", "RSI_14"]

def get_ml_probability(entry, vol, direction, pattern_name, rsi_14):
    """
    Model ve scaler/encoder ile TP olasılığı tahmini döner.
    """
    if not ML_READY or lstm_model is None or scaler is None or le_dir is None or le_pat is None:
        return None
    try:
        # LABEL NORMALİZASYONU!
        dir_label = normalize_pattern_label(direction)
        pat_label = normalize_pattern_label(pattern_name)
        # Encoder unseen label hatası önleme
        if dir_label not in le_dir.classes_ or pat_label not in le_pat.classes_:
            print(f"[UYARI] Görülmeyen label: {dir_label} veya {pat_label}")
            return None
        dir_enc = le_dir.transform([dir_label])[0]
        pat_enc = le_pat.transform([pat_label])[0]
        values = np.array([[entry, vol, dir_enc, pat_enc, rsi_14]])
        X_scaled = scaler.transform(values)
        X_scaled = X_scaled.reshape((1, 1, len(FEATURE_LIST)))
        prob = float(lstm_model.predict(X_scaled, verbose=0)[0][0])
        return prob
    except Exception as e:
        print(f"ML tahmin hatası: {e}")
        return None

def fetch_ohlcv_futures(symbol="BTCUSDT", interval="15m", limit=200):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

def parse_pattern_line(line):
    import re
    pattern = {
        "pattern_emoji": "🔥",
        "pattern_name": "Pattern",
        "direction": "",
        "score": "",
        "confidence": "",
        "success": "",
        "index": "",
    }
    m = re.search(r"🔥 LSTM Doğrulamalı ([\w\s]+) \(score=([\d]+)/10, conf=([\d\.]+), success=([\d\.]+)%\): index (\d+) \| (LONG|SHORT)", line)
    if m:
        pattern["pattern_name"] = m.group(1)
        pattern["score"] = m.group(2)
        pattern["confidence"] = m.group(3)
        pattern["success"] = m.group(4)
        pattern["index"] = m.group(5)
        pattern["direction"] = "long" if m.group(6) == "LONG" else "short"
    else:
        m2 = re.search(r"🔥 LSTM Doğrulamalı ([\w\s]+) \(score=([\d]+)/10, conf=([\d\.]+)\): index (\d+) \| (LONG|SHORT)", line)
        if m2:
            pattern["pattern_name"] = m2.group(1)
            pattern["score"] = m2.group(2)
            pattern["confidence"] = m2.group(3)
            pattern["index"] = m2.group(4)
            pattern["direction"] = "long" if m2.group(5) == "LONG" else "short"
            pattern["success"] = ""
        else:
            m3 = re.search(r"🔥 Trade Edilebilir ([\w\s]+) \(score=([\d]+)/10\): index (\d+) \| (LONG|SHORT)", line)
            if m3:
                pattern["pattern_name"] = m3.group(1)
                pattern["score"] = m3.group(2)
                pattern["index"] = m3.group(3)
                pattern["direction"] = "long" if m3.group(4) == "LONG" else "short"
                pattern["confidence"] = "0.5"
                pattern["success"] = ""
    # LABEL NORMALİZASYONU!
    pattern["pattern_name"] = normalize_pattern_label(pattern["pattern_name"])
    pattern["direction"] = normalize_pattern_label(pattern["direction"])
    return pattern

def build_signal_buttons():
    buttons = [
        [InlineKeyboardButton("📊 Pattern Başarıları", callback_data="show_stats")]
    ]
    return InlineKeyboardMarkup(buttons)

async def send_telegram_signal(symbol, interval, pattern_info, df, ind_snapshot, ta_report, tp=None, sl=None, comment=None, ml_prob=None):
    pattern_emoji = pattern_info.get("pattern_emoji", "🔥")
    pattern_name = pattern_info.get("pattern_name", "Pattern")
    direction = pattern_info.get("direction", "")
    score = pattern_info.get("score", "")
    confidence = pattern_info.get("confidence", "")
    success = pattern_info.get("success", "")
    idx = int(pattern_info.get("index", -1))
    entry_price = df['close'].iloc[idx] if (0 <= idx < len(df)) else df['close'].iloc[-1]
    ts = df['timestamp'].iloc[idx] if (0 <= idx < len(df)) else df['timestamp'].iloc[-1]
    signal_time = ts.strftime("%Y-%m-%d %H:%M")
    if not tp or not sl:
        tp, sl = PriceActionPatterns(df).dynamic_tp_sl(idx, entry_type=direction)
    if not comment:
        comment = f"{pattern_name}, hacim ve skor filtreli otomatik sinyal."
    ml_str = f"*ML TP Olasılığı:* `{ml_prob:.2f}`\n" if ml_prob is not None else ""
    msg = (
        f"🔔 *Yeni Trade Sinyali!* 🔔\n\n"
        f"*Sembol:* `{symbol}`\n"
        f"*Timeframe:* `{interval}`\n"
        f"*Pattern:* {pattern_emoji} *{pattern_name}*\n"
        f"*Yön:* {'🟩 LONG' if direction=='long' else '🟥 SHORT'}\n"
        f"*Skor:* `{score}/10`\n"
        f"*LSTM Güven:* `{confidence}`\n"
        f"{f'*Başarı Oranı:* `{success}%`\n' if success != '' else ''}"
        f"{ml_str}"
        f"*Tarih/Saat:* `{signal_time}`\n\n"
        f"*Entry:* `{entry_price}`\n"
        f"*TP:* `{tp}`\n"
        f"*SL:* `{sl}`\n"
        f"*Açıklama:* _{comment}_\n"
        f"\n"
        f"RSI: `{ind_snapshot.get('RSI_14', 'N/A')}` | EMA50: `{ind_snapshot.get('EMA_50', 'N/A')}` | EMA200: `{ind_snapshot.get('EMA_200', 'N/A')}`\n"
        f"MACD: `{ind_snapshot.get('MACD', 'N/A')}` | BB_UP: `{ind_snapshot.get('BB_upper', 'N/A')}` | BB_LOW: `{ind_snapshot.get('BB_lower', 'N/A')}`\n"
        f"\n"
        f"*Teknik Analiz Özeti:*\n{ta_report[:3000]}\n"
    )
    print(f"[TELEGRAM] Kart gönderiliyor: {symbol} ({interval}) {pattern_name} {direction.upper()}")
    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown", reply_markup=build_signal_buttons())

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query.data == "show_stats":
        stats = pattern_stats.get_top_patterns(10)
        msg = "*En Başarılı Patternler:*\n"
        for k, v in stats:
            total = v["win"] + v["lose"]
            winrate = v.get("winrate", 0.0) * 100  # zaman ağırlıklı winrate
            msg += f"`{k}`: {v['win']}/{total} (%{winrate:.1f})\n"
        await query.answer()
        try:
            await query.edit_message_text(msg, parse_mode="Markdown")
        except BadRequest as e:
            if "Message is not modified" in str(e):
                pass
            else:
                raise e

def is_pattern_tradeable(pattern_key, min_success=MIN_PATTERN_SUCCESS_RATE):
    """Pattern başarı oranı ve winrate kontrolü."""
    # LABEL NORMALİZASYONU!
    parts = pattern_key.split("_")
    if len(parts) >= 3:
        p_name = normalize_pattern_label("_".join(parts[:-2]))
        direction = normalize_pattern_label(parts[-2])
        interval = parts[-1].lower()
        pattern_key = f"{p_name}_{direction}_{interval}"
    rate = pattern_stats.get_pattern_success_rate(pattern_key)
    return rate >= min_success, rate

async def monitor_trade(symbol, interval, entry_price, tp, sl, direction, pattern_key, idx, monitor_minutes=180):
    start_time = datetime.utcnow()
    close_price = None
    close_type = None
    pnl = None

    while (datetime.utcnow() - start_time).total_seconds() < monitor_minutes*60:
        df = fetch_ohlcv_futures(symbol, interval, limit=2)
        price = df['close'].iloc[-1]
        if direction == "long":
            if price >= tp:
                close_price = tp
                close_type = "TP"
                pnl = (tp - entry_price) / entry_price
                break
            elif price <= sl:
                close_price = sl
                close_type = "SL"
                pnl = (sl - entry_price) / entry_price
                break
        else:  # short
            if price <= tp:
                close_price = tp
                close_type = "TP"
                pnl = (entry_price - tp) / entry_price
                break
            elif price >= sl:
                close_price = sl
                close_type = "SL"
                pnl = (entry_price - sl) / entry_price
                break
        await asyncio.sleep(30)

    # Otomatik kapanış (zaman aşımı)
    if close_price is None:
        close_price = price
        close_type = "timeout"
        pnl = (close_price - entry_price) / entry_price if direction == "long" else (entry_price - close_price) / entry_price

    is_win = close_type == "TP"
    pattern_stats.update_pattern_stats(
        pattern_key, is_win, close_time=datetime.utcnow().isoformat(),
        entry=entry_price, exit=close_price, tp=tp, sl=sl, pnl=pnl, close_type=close_type
    )

async def send_telegram_report(days=1):
    report = pattern_stats.get_periodic_report(days=days)
    msg = f"📊 *Son {days} gün Sinyal Özeti:*\n"
    for k, v in report:
        msg += f"`{k}`: {v['win']}/{v['total']} | PnL: {v['total_pnl']:.4f}\n"
    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

async def scan_and_notify():
    print("Binance Futures coinleri çekiliyor...")
    exchange_info = client.futures_exchange_info()
    symbols = [
        s['symbol'] for s in exchange_info['symbols']
        if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT'
    ]
    intervals = ["5m", "15m", "30m", "1h"]
    print(f"{len(symbols)} coin taranacak: {symbols[:5]}...")

    while True:
        for symbol in symbols:
            for interval in intervals:
                print(f"[INFO] Analiz başlıyor: {symbol} {interval}")
                try:
                    df = fetch_ohlcv_futures(symbol, interval)
                    pa = PriceActionPatterns(df, timeframe=interval)
                    summary = pa.summary_report()
                    print(f"[SUMMARY DEBUG] {symbol} {interval} summary: {summary[:200]}")
                    key = f"{symbol}_{interval}_{str(df['timestamp'].iloc[-1])}"

                    ta_engine = TechnicalAnalysisEngine(df, timeframe=interval)
                    ta_report = ta_engine.summary_report()

                    if "🔥" in summary and (key not in signal_log or signal_log[key] != summary):
                        signal_log[key] = summary
                        first_line = next((l for l in summary.splitlines() if "🔥" in l), None)
                        if first_line:
                            pattern_info = parse_pattern_line(first_line)
                            idx = int(pattern_info.get("index", -1))
                            ind_snapshot = pa.indicator_snapshot(idx)
                            tp, sl = pa.dynamic_tp_sl(idx, entry_type=pattern_info.get("direction", "long"))
                            entry_price = df['close'].iloc[idx] if (0 <= idx < len(df)) else df['close'].iloc[-1]

                            # LABEL NORMALİZASYONU pattern_key!
                            pattern_key = f"{pattern_info.get('pattern_name')}_{pattern_info.get('direction')}_{interval}".lower()
                            tradeable, pattern_success = is_pattern_tradeable(pattern_key)
                            pattern_info["success"] = f"{pattern_success*100:.1f}"

                            # ML tahmini
                            ml_prob = None
                            if ML_READY and lstm_model is not None:
                                rsi_14 = ind_snapshot.get("RSI_14", 50.0)
                                ml_prob = get_ml_probability(
                                    entry=entry_price,
                                    vol=df['volume'].iloc[idx] if (0 <= idx < len(df)) else df['volume'].iloc[-1],
                                    direction=pattern_info.get("direction", "long"),
                                    pattern_name=pattern_info.get("pattern_name", "pattern"),
                                    rsi_14=rsi_14
                                )
                                ml_score = ml_prob if ml_prob is not None else 0.0
                            else:
                                ml_score = float(pattern_info.get("confidence", "0.0"))

                            # Hem başarı oranı hem ML skoru kontrolü
                            if tradeable and ml_score >= 0.5:
                                print(f"[SİNYAL] Sinyal gönderiliyor: {symbol} {interval} {pattern_key} MLskor:{ml_score:.2f} Başarı:{pattern_success:.2f}")
                                await send_telegram_signal(symbol, interval, pattern_info, df, ind_snapshot, ta_report, tp, sl, ml_prob=ml_prob)
                                asyncio.create_task(monitor_trade(
                                    symbol, interval, entry_price=entry_price, tp=tp, sl=sl,
                                    direction=pattern_info.get("direction"), pattern_key=pattern_key, idx=idx
                                ))
                            else:
                                print(f"[FILTRE] {symbol} {interval} {pattern_key} --> Başarı oranı ({pattern_success:.2f}) veya ML skoru ({ml_score:.2f}) yetersiz.")
                        else:
                            msg = f"🔥 {symbol} ({interval})\n{summary[:3400]}\n\n*Teknik Analiz Özeti:*\n{ta_report[:3000]}"
                            print(f"[SİNYAL] {symbol} ({interval}): {summary.splitlines()[0]}")
                            await bot.send_message(chat_id=CHAT_ID, text=msg)
                except Exception as e:
                    print(f"[{symbol} {interval}] hata: {e}")
                await asyncio.sleep(1.3)

        print("Tüm coinler tarandı, 3 dk uyku.")
        await asyncio.sleep(180)

async def periodic_scan(app):
    last_daily = None
    last_weekly = None
    while True:
        now = datetime.utcnow()
        if (not last_daily or (now.date() != last_daily)):
            await send_telegram_report(days=1)
            last_daily = now.date()
        if (now.weekday() == 0) and (not last_weekly or (now.date() != last_weekly)):
            await send_telegram_report(days=7)
            last_weekly = now.date()
        await scan_and_notify()
        await asyncio.sleep(5)

# --------- MANUEL TRADE KAYDI VE DÜZELTME KOMUTLARI ---------
async def manualtrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Kullanım: /manualtrade <pattern_key> <win/lose> [entry] [exit] [tp] [sl] [pnl] [close_type]"
        )
        return
    pattern_key = normalize_pattern_label(args[0])
    is_win = True if args[1].lower() == "win" else False
    entry = float(args[2]) if len(args) > 2 else None
    exit_ = float(args[3]) if len(args) > 3 else None
    tp = float(args[4]) if len(args) > 4 else None
    sl = float(args[5]) if len(args) > 5 else None
    pnl = float(args[6]) if len(args) > 6 else None
    close_type = args[7] if len(args) > 7 else None

    pattern_stats.manual_add_trade(
        pattern_key=pattern_key,
        is_win=is_win,
        entry=entry,
        exit=exit_,
        tp=tp,
        sl=sl,
        pnl=pnl,
        close_type=close_type
    )
    await update.message.reply_text(f"Manuel trade kaydı başarıyla eklendi: {pattern_key} {args[1].upper()}")

async def manualedit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 4:
        await update.message.reply_text(
            "Kullanım: /manualedit <pattern_key> <idx> <field> <value>"
        )
        return
    pattern_key = normalize_pattern_label(args[0])
    idx = int(args[1])
    field = args[2]
    value = args[3]
    if field == "pnl":
        value = float(value)
    success = pattern_stats.manual_edit(pattern_key, idx, field, value)
    if success:
        await update.message.reply_text("Trade kaydı başarıyla güncellendi.")
    else:
        await update.message.reply_text("Güncelleme başarısız. Lütfen parametreleri kontrol et.")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(CommandHandler("manualtrade", manualtrade))
    application.add_handler(CommandHandler("manualedit", manualedit))

    async def start_scanner(app):
        asyncio.create_task(periodic_scan(app))

    application.post_init = start_scanner

    print("Otomatik Binance Futures tarama ve sinyal botu başlatılıyor (Telegram handler entegre)!")
    application.run_polling()

if __name__ == "__main__":
    main()