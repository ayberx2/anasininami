import ccxt
import pattern_stats
from datetime import datetime

class LiveTrader:
    def __init__(self, exchange_id, api_key, secret, password=None, testnet=False):
        exchange_class = getattr(ccxt, exchange_id)
        params = {'apiKey': api_key, 'secret': secret}
        if password:
            params['password'] = password
        if testnet and exchange_id == "binance":
            params['options'] = {'defaultType': 'future'}
            params['urls'] = {'api': {'private': 'https://testnet.binancefuture.com'}}
        self.exchange = exchange_class(params)

    def send_order(self, symbol, side, amount, price=None, order_type='market', pattern_key=None, ml_confirmed=None, ml_score=None, overfit_warning=None, extra=None):
        """
        Gelişmiş: pattern_key, ml_confirmed, ml_score, overfit_warning ve extra gibi özelliklerle entegre pattern/ML istatistik kaydı.
        """
        try:
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            else:
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            # Trade başarıyla açıldıysa, pattern/ML istatistiğini kaydet (ekstra breakdown ile)
            if pattern_key is not None:
                # Modern breakdown: ML/ensemble ve overfit bilgisiyle birlikte kayıt
                pattern_stats.update_pattern_stats(
                    pattern_key,
                    is_win=None,  # Pozisyon kapanınca güncellenecek!
                    close_time=datetime.utcnow().isoformat(),
                    entry=order.get('price', None),
                    extra={
                        "ml_confirmed": ml_confirmed,
                        "ml_score": ml_score,
                        "overfit_warning": overfit_warning,
                        **(extra if extra else {})
                    }
                )
            return order
        except Exception as e:
            print("Order Error:", str(e))
            return None

    def get_balance(self):
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            print("Balance Error:", str(e))
            return None

    def get_open_orders(self, symbol=None):
        try:
            return self.exchange.fetch_open_orders(symbol) if symbol else self.exchange.fetch_open_orders()
        except Exception as e:
            print("Open Orders Error:", str(e))
            return None

    def record_trade_result(self, pattern_key, is_win, close_time=None, entry=None, exit=None, tp=None, sl=None, pnl=None, close_type=None, ml_confirmed=None, ml_score=None, overfit_warning=None, extra=None):
        """
        Pozisyon kapanınca çağır: pattern_key (örn: 'pinbar_long_15m'), is_win (bool)
        ML/ensemble/overfit breakdown ile gelişmiş kayıt.
        """
        if not close_time:
            close_time = datetime.utcnow().isoformat()
        pattern_stats.update_pattern_stats(
            pattern_key,
            is_win,
            close_time=close_time,
            entry=entry,
            exit=exit,
            tp=tp,
            sl=sl,
            pnl=pnl,
            close_type=close_type,
            extra={
                "ml_confirmed": ml_confirmed,
                "ml_score": ml_score,
                "overfit_warning": overfit_warning,
                **(extra if extra else {})
            }
        )
        print(f"[STATS] Pattern '{pattern_key}' için {'WIN' if is_win else 'LOSE'} kaydedildi.")

# Örnek kullanım:
# trader = LiveTrader("binance", "...", "...")
# order = trader.send_order("BTC/USDT", "buy", 0.01, pattern_key="pinbar_long_15m", ml_confirmed=True, ml_score=0.83)
# trade kapanınca:
# trader.record_trade_result("pinbar_long_15m", is_win=True, ml_confirmed=True, ml_score=0.83)