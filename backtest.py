import pandas as pd
import numpy as np
import pattern_stats
from price_action_engine import PriceActionPatterns

class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        strategy_func,
        tp_sl_func,
        initial_balance=1000,
        fee_rate=0.0005,
        slippage=0.0,
        risk_pct=1.0,
        plot_equity_curve=False,
        verbose=False,
        pattern_key_func=None,
        ml_filter_func=None,        # ML sinyal onay fonksiyonu (ekstra)
        max_open_trades=1,         # Çoklu trade desteği için
        advanced_stats=False       # Ekstra pattern & ML analizleri için
    ):
        """
        df: OHLCV DataFrame
        strategy_func: (df, idx) -> "buy" or "sell" or None
        tp_sl_func: (idx, entry_type) -> (tp, sl)
        pattern_key_func: (df, idx, entry_type) -> str
        ml_filter_func: (df, idx, signal) -> bool [Opsiyonel ML/LSTM onayı]
        advanced_stats: Pattern ve ML bazlı ileri analiz ve breakdown
        """
        self.df = df.copy()
        self.strategy_func = strategy_func
        self.tp_sl_func = tp_sl_func
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.risk_pct = risk_pct
        self.plot_equity_curve = plot_equity_curve
        self.verbose = verbose
        self.pattern_key_func = pattern_key_func
        self.ml_filter_func = ml_filter_func
        self.max_open_trades = max_open_trades
        self.advanced_stats = advanced_stats

        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades = []
        self.active_trades = []
        self.max_drawdown = 0
        self.peak_balance = initial_balance

        # Ekstra pattern & ML breakdown
        self.pattern_breakdown = {}
        self.ml_breakdown = {}

        # Overfitting uyarılarını burada tut
        self.overfit_warnings = []

    def run(self):
        for idx in range(len(self.df)):
            price = self.df['close'].iloc[idx]

            # Overfitting detector: Son sinyallerde pattern spamı var mı?
            if self.advanced_stats and idx > 100:
                warnings = self._overfit_pattern_checks(idx)
                if warnings:
                    self.overfit_warnings.extend(warnings)

            # Check for open trades (çoklu trade)
            self._manage_active_trades(idx, price)

            # Yeni sinyal varsa işleme gir (maksimum açık trade kontrolü)
            if len(self.active_trades) < self.max_open_trades:
                signal = self.strategy_func(self.df, idx)
                # ML onayı (isteğe bağlı)
                if signal in ["buy", "sell"]:
                    if self.ml_filter_func:
                        if not self.ml_filter_func(self.df, idx, signal):
                            continue
                if signal in ["buy", "sell"]:
                    entry_type = "long" if signal == "buy" else "short"
                    tp, sl = self.tp_sl_func(idx, entry_type)
                    risk_per_unit = abs(self.df['close'].iloc[idx] - sl) if entry_type == "long" else abs(sl - self.df['close'].iloc[idx])
                    if risk_per_unit == 0:
                        continue
                    max_risk_amt = self.balance * self.risk_pct / 100
                    qty = max_risk_amt / risk_per_unit
                    qty = max(qty, 0)
                    if qty == 0:
                        continue

                    entry_price = price + self.slippage if entry_type == "long" else price - self.slippage
                    fee = entry_price * qty * self.fee_rate
                    pattern_key = self.pattern_key_func(self.df, idx, entry_type) if self.pattern_key_func else None

                    trade = dict(
                        entry_idx=idx,
                        entry_price=entry_price,
                        type=entry_type,
                        tp=tp,
                        sl=sl,
                        qty=qty,
                        fee=fee,
                        pattern_key=pattern_key,
                        ml_confirmed=None,
                        ml_score=None
                    )

                    # ML breakdown için skor kaydı (isteğe bağlı)
                    if self.ml_filter_func and hasattr(self.ml_filter_func, "last_score"):
                        trade["ml_score"] = self.ml_filter_func.last_score

                    self.active_trades.append(trade)
                    if self.verbose:
                        print(f"Giriş: {signal} idx={idx} Fiyat={entry_price:.2f} Miktar={qty:.4f} Fee={fee:.2f}")

    def _manage_active_trades(self, idx, price):
        # Aktif trade listesi üzerinde çıkış kontrolü
        still_open = []
        for t in self.active_trades:
            closed = False
            result = ""
            close_price = price - self.slippage if t["type"] == "long" else price + self.slippage

            # TP/SL kontrol
            if t["type"] == "long":
                if close_price >= t["tp"]:
                    closed = True
                    result = "tp"
                    exit_price = t["tp"]
                elif close_price <= t["sl"]:
                    closed = True
                    result = "sl"
                    exit_price = t["sl"]
            else:
                if close_price <= t["tp"]:
                    closed = True
                    result = "tp"
                    exit_price = t["tp"]
                elif close_price >= t["sl"]:
                    closed = True
                    result = "sl"
                    exit_price = t["sl"]

            if closed:
                fee_close = exit_price * t["qty"] * self.fee_rate
                if t["type"] == "long":
                    pnl = (exit_price - t["entry_price"]) * t["qty"] - t["fee"] - fee_close
                else:
                    pnl = (t["entry_price"] - exit_price) * t["qty"] - t["fee"] - fee_close

                if t.get("pattern_key"):
                    pattern_stats.update_pattern_stats(
                        t["pattern_key"],
                        is_win=(result == "tp"),
                        close_time=self.df['timestamp'].iloc[idx].isoformat() if 'timestamp' in self.df.columns else None,
                        entry=t["entry_price"],
                        exit=exit_price,
                        tp=t["tp"],
                        sl=t["sl"],
                        pnl=pnl,
                        close_type=result.upper()
                    )
                    # Advanced pattern breakdown
                    if self.advanced_stats:
                        self.pattern_breakdown.setdefault(t["pattern_key"], []).append(result)

                # ML breakdown/stat
                if self.advanced_stats and t.get("ml_score") is not None:
                    self.ml_breakdown.setdefault(t["pattern_key"], []).append((result, t["ml_score"]))

                self.trades.append(dict(
                    **t,
                    close_idx=idx,
                    close_price=exit_price,
                    result=result,
                    pnl=pnl,
                    fee_close=fee_close
                ))
                self.balance += pnl
                if self.verbose:
                    print(f"Çıkış: {result} idx={idx} Fiyat={exit_price:.2f} PnL={pnl:.2f} Yeni Bakiye={self.balance:.2f}")
            else:
                still_open.append(t)
        self.active_trades = still_open
        # Equity curve güncelle
        self.equity_curve.append(self.balance)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def _overfit_pattern_checks(self, idx, recent_window=100, max_allowed=10):
        """
        Son recent_window bar içinde belirli patternler çok fazla tekrar ettiyse uyarı döndür.
        """
        warnings = []
        if not hasattr(self, "pattern_key_func") or self.pattern_key_func is None:
            return warnings
        patterns_to_check = ["pinbar_bullish", "pinbar_bearish", "double_top", "double_bottom"]
        for pt in patterns_to_check:
            count = 0
            for t in self.trades[-recent_window:]:
                if t.get("pattern_key", "").startswith(pt):
                    count += 1
            if count > max_allowed:
                warnings.append(f"WARNING: {pt} overfit risk - {count} trades in last {recent_window} bars")
        return warnings

    def summary(self):
        results = pd.DataFrame(self.trades)
        returns = results['pnl'].sum() if not results.empty else 0
        win_trades = results[results['result'] == 'tp'] if not results.empty else []
        loss_trades = results[results['result'] == 'sl'] if not results.empty else []
        win_rate = len(win_trades) / len(results) if len(results) > 0 else 0
        avg_risk_reward = win_trades['pnl'].mean() / abs(loss_trades['pnl'].mean()) if len(win_trades) > 0 and len(loss_trades) > 0 else None

        summary = {
            "final_balance": self.balance,
            "total_trades": len(results),
            "win_rate": win_rate,
            "total_returns": returns,
            "max_drawdown": self.max_drawdown,
            "avg_risk_reward": avg_risk_reward,
            "results": results,
            "equity_curve": self.equity_curve
        }
        if self.advanced_stats:
            summary["pattern_breakdown"] = self.pattern_breakdown
            summary["ml_breakdown"] = self.ml_breakdown
            summary["overfit_warnings"] = self.overfit_warnings
        return summary

    def plot_equity(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(self.equity_curve)
        plt.title("Equity Curve")
        plt.xlabel("Bar")
        plt.ylabel("Balance")
        plt.show()

    def pattern_report(self):
        """
        Gelişmiş breakdown: Pattern bazlı istatistikler (sadece advanced_stats=True ile)
        """
        if not self.advanced_stats:
            print("Pattern breakdown sadece advanced_stats=True ile desteklenir.")
            return
        for key, results in self.pattern_breakdown.items():
            total = len(results)
            win = results.count("tp")
            loss = results.count("sl")
            print(f"{key}: Win={win} / Total={total} ({win/total:.2%})")

    def ml_report(self):
        """
        Gelişmiş breakdown: ML skorları ve sonuçları (sadece advanced_stats=True ile)
        """
        if not self.advanced_stats:
            print("ML breakdown sadece advanced_stats=True ile desteklenir.")
            return
        for key, tuples in self.ml_breakdown.items():
            if not tuples: continue
            win_scores = [score for res, score in tuples if res == "tp"]
            loss_scores = [score for res, score in tuples if res == "sl"]
            print(f"{key} - ML Win Avg: {np.mean(win_scores) if win_scores else '-'} | ML Loss Avg: {np.mean(loss_scores) if loss_scores else '-'}")

    def overfit_report(self):
        """
        Son backtestte pattern overfit uyarılarını gösterir (advanced_stats=True ise)
        """
        if not self.advanced_stats:
            print("Overfit breakdown sadece advanced_stats=True ile desteklenir.")
            return
        if not self.overfit_warnings:
            print("Hiç overfit uyarısı yok.")
            return
        for warn in self.overfit_warnings:
            print(warn)