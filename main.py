import yaml
from binance.client import Client
import pandas as pd
from ta_engine import TechnicalAnalysisEngine
from price_action_engine import PriceActionPatterns
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from backtest import Backtester
from ccxt_live_trader import LiveTrader
import pattern_stats

import datetime
import asyncio
import joblib

# Config dosyasını yükle
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
BINANCE_API_KEY = config["binance"]["api_key"]
BINANCE_API_SECRET = config["binance"]["api_secret"]
TELEGRAM_TOKEN = config["telegram"]["bot_token"]

# Binance istemcisi
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Sinyal bildirim logu (tekrar sinyal spam'ini engellemek için)
signal_log = {}

def fetch_ohlcv(symbol="BTCUSDT", interval="15m", limit=200):
    interval_map = {
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR
    }
    klines = client.get_klines(
        symbol=symbol,
        interval=interval_map.get(interval, Client.KLINE_INTERVAL_15MINUTE),
        limit=limit
    )
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

def s_summary_limiter(summary, maxlen=3500):
    return summary if len(summary) < maxlen else summary[:maxlen-10] + "..."

# Otomatik sinyal bildirimi fonksiyonu (her 3 dakikada bir)
async def signal_notifier(application, chat_id, symbol="BTCUSDT", interval="15m"):
    df = fetch_ohlcv(symbol, interval)
    pa = PriceActionPatterns(df)
    summary = pa.summary_report()
    key = f"{symbol}_{interval}_{str(df['timestamp'].iloc[-1])}"
    if key not in signal_log or signal_log[key] != summary:
        signal_log[key] = summary
        # Sadece anlamlı sinyalde gönder (rapor, pattern veya ML skoru içeriyorsa)
        if (
            "🔥" in summary
            or "Trade Edilebilir" in summary
            or "DETECTED" in summary
            or "LSTM" in summary
            or "ML" in summary
            or "META-ENSEMBLE" in summary
            or "Market Regime" in summary
            or "overfit risk" in summary
        ):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            msg = f"[{now}] {symbol} ({interval})\n{s_summary_limiter(summary)}"
            await application.bot.send_message(chat_id=chat_id, text=msg)

# Telegram komutu: /analyse BTCUSDT 15m
async def analyse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) == 0:
        await update.message.reply_text("Sembol gir: /analyse BTCUSDT 15m")
        return
    symbol = args[0].upper()
    interval = args[1] if len(args) > 1 else "15m"
    df = fetch_ohlcv(symbol, interval)
    pa = PriceActionPatterns(df)
    result = pa.summary_report()
    await update.message.reply_text(f"{symbol} ({interval}) için Price Action Analizi:\n{s_summary_limiter(result)}")

# Telegram komutu: /ta BTCUSDT 15m
async def ta_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) == 0:
        await update.message.reply_text("Sembol gir: /ta BTCUSDT 15m")
        return
    symbol = args[0].upper()
    interval = args[1] if len(args) > 1 else "15m"
    df = fetch_ohlcv(symbol, interval)
    tae = TechnicalAnalysisEngine(df)
    result = tae.summary_report()
    await update.message.reply_text(f"{symbol} ({interval}) için Teknik Analiz Raporu:\n{s_summary_limiter(result)}")

# Telegram komutu: /backtest BTCUSDT 15m
async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) == 0:
        await update.message.reply_text("Sembol gir: /backtest BTCUSDT 15m")
        return
    symbol = args[0].upper()
    interval = args[1] if len(args) > 1 else "15m"
    df = fetch_ohlcv(symbol, interval)

    # Örnek ML ile pattern sinyali birleştiren strateji
    def strategy_func(df, idx):
        pap = PriceActionPatterns(df)
        patterns = pap.detect_pinbar()
        if patterns and patterns[-1].get('idx', patterns[-1].get('index', None)) == idx:
            p = patterns[-1]
            # ML sinyali kontrolü (LSTM veya ML classifier)
            ml = pap.lstm_signal(idx)
            if ml and 'score' in ml:
                if p['type'] in ['pinbar_bullish', 'bullish'] and ml['score'] > 0.52:
                    return "buy"
                elif p['type'] in ['pinbar_bearish', 'bearish'] and ml['score'] < 0.48:
                    return "sell"
            else:
                if p['type'] in ['pinbar_bullish', 'bullish']:
                    return "buy"
                elif p['type'] in ['pinbar_bearish', 'bearish']:
                    return "sell"
        return None

    def tp_sl_func(idx, entry_type):
        pap = PriceActionPatterns(df)
        return pap.dynamic_tp_sl(idx, entry_type)

    def pattern_key_func(df, idx, entry_type):
        pap = PriceActionPatterns(df)
        return pap.pattern_key_for_idx(idx, "pinbar", entry_type)

    backtester = Backtester(
        df,
        strategy_func,
        tp_sl_func,
        pattern_key_func=pattern_key_func,
        verbose=False,
        advanced_stats=True
    )
    backtester.run()
    summary = backtester.summary()
    msg = (
        f"Backtest Sonucu ({symbol} {interval}):\n"
        f"Son Bakiye: {summary['final_balance']:.2f}\n"
        f"Toplam İşlem: {summary['total_trades']}\n"
        f"Kazanç Oranı: {summary['win_rate']*100:.1f}%\n"
        f"Maksimum Drawdown: {summary['max_drawdown']*100:.1f}%"
    )
    # Overfit uyarılarını da ekle
    if "overfit_warnings" in summary and summary["overfit_warnings"]:
        msg += "\n⚠️ Overfit Uyarıları:\n" + "\n".join(summary["overfit_warnings"])
    await update.message.reply_text(msg)

# Telegram komutu: /trade BTCUSDT buy 0.001 [market/limit] [fiyat]
async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("Kullanım: /trade BTCUSDT buy 0.001 [market/limit] [fiyat]")
        return
    symbol = args[0].replace("USDT", "/USDT").upper()
    side = args[1].lower()
    amount = float(args[2])
    order_type = args[3] if len(args) > 3 else "market"
    price = float(args[4]) if len(args) > 4 and order_type == "limit" else None

    api_key = config["binance"]["api_key"]
    api_secret = config["binance"]["api_secret"]
    testnet = config["binance"].get("testnet", False)
    trader = LiveTrader("binance", api_key, api_secret, testnet=testnet)
    result = trader.send_order(symbol, side, amount, price, order_type)
    if result:
        await update.message.reply_text(f"Emir gönderildi: {result}")
    else:
        await update.message.reply_text("Emir gönderilemedi!")

# Telegram komutu: /startsignal BTCUSDT 15m
async def start_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Kullanım: /startsignal BTCUSDT 15m")
        return
    symbol = args[0].upper()
    interval = args[1] if len(args) > 1 else "15m"
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"{symbol} ({interval}) için otomatik sinyal bildirimi başlatıldı.")

    async def notifier_job(application):
        while True:
            try:
                await signal_notifier(application, chat_id, symbol, interval)
                await asyncio.sleep(180)
            except Exception as e:
                print("Sinyal bildirici hata:", e)
                await asyncio.sleep(180)

    application = update.application
    asyncio.create_task(notifier_job(application))

# --------- MANUEL TRADE KAYDI VE DÜZELTME KOMUTLARI ---------
async def manualtrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Kullanım:
    /manualtrade <pattern_key> <win/lose> [entry] [exit] [tp] [sl] [pnl] [close_type]
    """
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Kullanım: /manualtrade <pattern_key> <win/lose> [entry] [exit] [tp] [sl] [pnl] [close_type]"
        )
        return
    pattern_key = args[0]
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
    """
    Kullanım:
    /manualedit <pattern_key> <idx> <field> <value>
    idx: -1 en son trade
    field: örn. pnl, close_type, result
    value: yeni değer
    """
    args = context.args
    if len(args) < 4:
        await update.message.reply_text(
            "Kullanım: /manualedit <pattern_key> <idx> <field> <value>"
        )
        return
    pattern_key = args[0]
    idx = int(args[1])
    field = args[2]
    value = args[3]
    # Otomatik olarak tür dönüşümü: pnl float, result string, vs.
    if field == "pnl":
        value = float(value)
    success = pattern_stats.manual_edit(pattern_key, idx, field, value)
    if success:
        await update.message.reply_text("Trade kaydı başarıyla güncellendi.")
    else:
        await update.message.reply_text("Güncelleme başarısız. Lütfen parametreleri kontrol et.")

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("analyse", analyse))
    app.add_handler(CommandHandler("ta", ta_report))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("trade", trade))
    app.add_handler(CommandHandler("startsignal", start_signal))
    app.add_handler(CommandHandler("manualtrade", manualtrade))
    app.add_handler(CommandHandler("manualedit", manualedit))
    print("Bot başlatılıyor...")
    app.run_polling()