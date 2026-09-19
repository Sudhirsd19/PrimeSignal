import math
import os
import time
import asyncio
from typing import Dict, Any, Optional
from config import Config

class RiskManager:
    def __init__(self):
        self.daily_starting_equity = None
        self.current_drawdown_pct = 0.0
        self.daily_profit_locked = False
        self.max_daily_profit_pct = float(getattr(Config, 'MAX_DAILY_PROFIT_PCT', 10.0))
        self.daily_profit_unlocked_manual = False
        self.reserved_risk_pct = 0.0
        self.reserved_open_count = 0
        self.reserved_longs_count = 0
        self.reserved_shorts_count = 0
        self.active_reservations: dict[str, dict[str, Any]] = {}
        self.max_correlated_risk_pct = Config.get_max_portfolio_risk_fraction()
        self._lock = asyncio.Lock()
        self.portfolio_lock = self._lock

    def _recalculate_reserved_totals(self):
        """Recalculate durable reservations without time-based auto-release.

        An exchange order can remain unresolved longer than any local timeout.
        ACTIVE risk therefore stays reserved until explicit release or broker
        reconciliation confirms that the reservation is no longer needed.
        """
        total_risk = 0.0
        open_count = 0
        longs = 0
        shorts = 0
        for res in self.active_reservations.values():
            if res.get('state') == 'ACTIVE':
                total_risk += float(res.get('risk_pct', 0.0))
                open_count += 1
                side = str(res.get('side', '')).upper()
                if side in ("BUY", "LONG"):
                    longs += 1
                elif side in ("SELL", "SHORT"):
                    shorts += 1
        self.reserved_risk_pct = max(0.0, total_risk)
        self.reserved_open_count = max(0, open_count)
        self.reserved_longs_count = max(0, longs)
        self.reserved_shorts_count = max(0, shorts)

    def check_and_reserve_risk_nolock(self, current_open_risk_pct: float, proposed_risk_pct: float, side: str = "BUY", reservation_id: str | None = None, symbol: str = "") -> bool:
        self._recalculate_reserved_totals()
        total_projected_risk = current_open_risk_pct + self.reserved_risk_pct + proposed_risk_pct
        if total_projected_risk > (self.max_correlated_risk_pct + 0.0001):
            try:
                print(f"[RISK] ⛔ Portfolio risk cap breach prevented! Projected: {total_projected_risk*100:.2f}%, Limit: {self.max_correlated_risk_pct*100:.2f}%")
            except Exception:
                print(f"[RISK] [BLOCK] Portfolio risk cap breach prevented! Projected: {total_projected_risk*100:.2f}%, Limit: {self.max_correlated_risk_pct*100:.2f}%")
            return False

        res_id = reservation_id or f"RES_{symbol.replace('/', '')}_{int(time.time()*1000)}"
        self.active_reservations[res_id] = {
            'reservation_id': res_id,
            'symbol': symbol,
            'side': side.upper(),
            'risk_pct': proposed_risk_pct,
            'timestamp': time.time(),
            'state': 'ACTIVE'
        }
        self._recalculate_reserved_totals()
        return True

    async def check_and_reserve_risk_atomic(self, current_open_risk_pct: float, proposed_risk_pct: float, side: str = "BUY", reservation_id: str | None = None, symbol: str = "") -> bool:
        """Atomically check and commit a durable portfolio-risk reservation."""
        async with self._lock:
            return self.check_and_reserve_risk_nolock(current_open_risk_pct, proposed_risk_pct, side, reservation_id, symbol)

    async def can_open_trade_atomic(self, current_open_risk_pct: float, proposed_risk_pct: float, side: str = "BUY", reservation_id: str | None = None, symbol: str = "") -> bool:
        return await self.check_and_reserve_risk_atomic(current_open_risk_pct, proposed_risk_pct, side=side, reservation_id=reservation_id, symbol=symbol)

    async def release_risk(self, risk_pct: float, side: str = "BUY", reservation_id: str | None = None):
        """Release risk idempotently by reservation ID or scalar fallback."""
        async with self._lock:
            if reservation_id and reservation_id in self.active_reservations:
                self.active_reservations[reservation_id]['state'] = 'RELEASED'
                del self.active_reservations[reservation_id]
            else:
                matched_id = None
                for r_id, r_data in list(self.active_reservations.items()):
                    if r_data.get('state') == 'ACTIVE' and abs(float(r_data.get('risk_pct', 0.0)) - risk_pct) < 1e-6:
                        matched_id = r_id
                        break
                if matched_id:
                    del self.active_reservations[matched_id]
            self._recalculate_reserved_totals()

    def serialize_reservations(self) -> dict[str, dict[str, Any]]:
        return {r_id: dict(data) for r_id, data in self.active_reservations.items() if data.get('state') == 'ACTIVE'}

    def serialize_daily_state(self) -> dict[str, Any]:
        return {
            'daily_starting_equity': self.daily_starting_equity,
            'current_drawdown_pct': self.current_drawdown_pct,
            'daily_profit_locked': self.daily_profit_locked,
            'daily_profit_unlocked_manual': self.daily_profit_unlocked_manual,
            'max_daily_profit_pct': self.max_daily_profit_pct,
        }

    def load_daily_state(self, data: dict[str, Any] | None, same_trading_day: bool = True):
        if not isinstance(data, dict) or not same_trading_day:
            return
        start_eq = data.get('daily_starting_equity')
        if start_eq is not None:
            try:
                start_eq = float(start_eq)
                self.daily_starting_equity = start_eq if start_eq > 0 else None
            except (TypeError, ValueError):
                self.daily_starting_equity = None
        try:
            self.current_drawdown_pct = float(data.get('current_drawdown_pct', 0.0) or 0.0)
        except (TypeError, ValueError):
            self.current_drawdown_pct = 0.0
        self.daily_profit_locked = bool(data.get('daily_profit_locked', False))
        self.daily_profit_unlocked_manual = bool(data.get('daily_profit_unlocked_manual', False))
        try:
            self.max_daily_profit_pct = float(data.get('max_daily_profit_pct', self.max_daily_profit_pct))
        except (TypeError, ValueError):
            pass

    def load_reservations(self, reservations_data: dict[str, Any] | None):
        self.active_reservations.clear()
        if isinstance(reservations_data, dict):
            for r_id, data in reservations_data.items():
                if isinstance(data, dict) and data.get('state', 'ACTIVE') == 'ACTIVE':
                    self.active_reservations[r_id] = {
                        'reservation_id': r_id,
                        'symbol': str(data.get('symbol', '')),
                        'side': str(data.get('side', 'BUY')).upper(),
                        'risk_pct': float(data.get('risk_pct', 0.0)),
                        'timestamp': float(data.get('timestamp', time.time())),
                        'state': 'ACTIVE'
                    }
        self._recalculate_reserved_totals()

    def calculate_position_size(self, account_equity: float, entry_price: float, stop_loss: float, quote_currency: str = "USDT", equity_currency: str | None = None, conversion_rate: float | None = None, is_inr: bool | None = None, risk_pct_override: float | None = None) -> float:
        if account_equity is None or entry_price is None or stop_loss is None:
            return 0.0
        if math.isnan(account_equity) or math.isnan(entry_price) or math.isnan(stop_loss):
            return 0.0
        if account_equity <= 0 or entry_price <= 0:
            return 0.0

        if is_inr is None:
            is_inr = (equity_currency == 'INR') or getattr(Config, 'PAPER_CURRENCY', 'INR') == 'INR' or getattr(Config, 'COINDCX_TRADE_INR', False)
        effective_equity_curr = "INR" if is_inr else (equity_currency or "USDT")
        rate = float(conversion_rate or getattr(Config, 'USDT_INR_RATE', 85.0))
        if rate <= 0:
            rate = 85.0

        if effective_equity_curr == "INR" and quote_currency.upper() in ("USDT", "USD"):
            entry_price_equity_curr = entry_price * rate
            stop_loss_equity_curr = stop_loss * rate
        elif effective_equity_curr == "USDT" and quote_currency.upper() == "INR":
            entry_price_equity_curr = entry_price / rate
            stop_loss_equity_curr = stop_loss / rate
        else:
            entry_price_equity_curr = entry_price
            stop_loss_equity_curr = stop_loss

        active_risk_pct = risk_pct_override if risk_pct_override is not None else (Config.RISK_PCT / 100.0)
        base_risk = account_equity * active_risk_pct
        curr_mult = rate if effective_equity_curr == "INR" else 1.0
        risk_scaler = active_risk_pct / max(Config.RISK_PCT / 100.0, 1e-9)

        # P1-01 Fix: Only apply single trade dollar risk cap if explicitly configured in Config / env
        raw_max_risk = getattr(Config, "MAX_SINGLE_TRADE_RISK_USDT", None)
        if raw_max_risk is not None and float(raw_max_risk) > 0:
            max_single_trade_risk = float(raw_max_risk) * curr_mult * risk_scaler
            threshold_equity = 1000.0 * curr_mult
            trade_risk = min(base_risk, max_single_trade_risk) if account_equity >= threshold_equity else base_risk
        else:
            trade_risk = base_risk

        stop_distance = abs(entry_price_equity_curr - stop_loss_equity_curr)
        if stop_distance <= 0 or math.isnan(stop_distance) or math.isinf(stop_distance):
            print(f"⚠️ RISK MANAGER WARNING: Stop distance is non-positive or invalid ({stop_distance})")
            print(f"   Entry Price: {entry_price}, Stop Loss: {stop_loss}")
            print("   Failing closed: returning 0.0 position size")
            return 0.0

        position_size = trade_risk / stop_distance
        is_futures = getattr(Config, 'EXCHANGE_TYPE', 'spot') == 'futures'
        leverage = getattr(Config, 'FUTURES_LEVERAGE', 1.0) if is_futures else 1.0
        max_alloc = getattr(Config, 'MAX_TRADE_ALLOCATION', getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35))
        if max_alloc > 1.0:
            max_alloc = max_alloc / 100.0
        if not (0.0 < max_alloc <= 1.0):
            print(f"[RISK] Invalid MAX_TRADE_ALLOCATION={max_alloc}; blocking trade.")
            return 0.0

        max_position_value = account_equity * max_alloc * leverage
        min_notional = 100.0 if effective_equity_curr == "INR" else 10.0
        if account_equity < min_notional or max_position_value < min_notional:
            print(f"[RISK] Minimum notional cannot be satisfied within configured allocation cap; blocking trade. Equity={account_equity:.2f} {effective_equity_curr}, cap={max_position_value:.2f}, minimum={min_notional:.2f}")
            return 0.0

        position_value_equity_curr = position_size * entry_price_equity_curr
        if position_value_equity_curr > max_position_value:
            if entry_price_equity_curr <= 0:
                return 0.0
            position_size = (max_position_value * 0.999) / entry_price_equity_curr
            print(f"[RISK] Position size capped at {max_alloc*100:.0f}% max capital allocation: {position_size:.6f} ({effective_equity_curr} {position_size*entry_price_equity_curr:.2f})")

        return round(position_size, 6)

    def check_circuit_breaker(self, current_equity: float) -> bool:
        if current_equity is None or math.isnan(current_equity):
            print("🚨 CIRCUIT BREAKER WARNING: Invalid equity value received. Halting trading.")
            return False
        if self.daily_starting_equity is None or math.isnan(self.daily_starting_equity):
            self.daily_starting_equity = current_equity
            return True
        pnl = current_equity - self.daily_starting_equity
        if self.daily_starting_equity <= 0:
            self.current_drawdown_pct = 0.0
        else:
            self.current_drawdown_pct = (pnl / self.daily_starting_equity) * 100.0
        if self.current_drawdown_pct <= -Config.MAX_DAILY_LOSS_PCT:
            msg = f"🚨 CIRCUIT BREAKER TRIGGERED: Daily loss limit hit ({self.current_drawdown_pct:.2f}%). Trading suspended."
            try: print(msg)
            except UnicodeEncodeError:
                import sys
                enc = sys.stdout.encoding or 'utf-8'
                print(msg.encode(enc, errors='replace').decode(enc))
            return False
        if getattr(Config, 'ENABLE_DAILY_PROFIT_LOCK', True):
            max_profit = getattr(self, 'max_daily_profit_pct', getattr(Config, 'MAX_DAILY_PROFIT_PCT', 10.0))
            if getattr(self, 'daily_profit_unlocked_manual', False):
                self.daily_profit_locked = False
            elif self.current_drawdown_pct >= max_profit:
                self.daily_profit_locked = True
                msg = f"🎯 PROFIT LOCK: Daily gain target hit (+{self.current_drawdown_pct:.2f}% >= +{max_profit:.1f}%). Locking profits, no new entries until midnight UTC."
                try: print(msg)
                except UnicodeEncodeError:
                    import sys
                    enc = sys.stdout.encoding or 'utf-8'
                    print(msg.encode(enc, errors='replace').decode(enc))
                return False
            else:
                self.daily_profit_locked = False
        return True

    def unlock_daily_profit(self) -> dict:
        self.daily_profit_locked = False
        self.daily_profit_unlocked_manual = True
        msg = "🔓 [RISK] Daily Profit Lock manually overridden and unlocked. Trading resumed."
        try: print(msg)
        except Exception: pass
        return {"status": "success", "locked": False, "message": msg}

    def set_daily_profit_target(self, target_pct: float) -> dict:
        self.max_daily_profit_pct = max(0.1, target_pct)
        Config.MAX_DAILY_PROFIT_PCT = self.max_daily_profit_pct
        if self.current_drawdown_pct < self.max_daily_profit_pct:
            self.daily_profit_locked = False
            self.daily_profit_unlocked_manual = False
        msg = f"🎯 [RISK] Daily profit target updated to {self.max_daily_profit_pct:.2f}%."
        try: print(msg)
        except Exception: pass
        return {"status": "success", "max_daily_profit_pct": self.max_daily_profit_pct, "locked": self.daily_profit_locked}

    def reset_daily_equity(self, current_equity: float):
        self.daily_starting_equity = current_equity
        self.current_drawdown_pct = 0.0
        self.daily_profit_locked = False
        self.daily_profit_unlocked_manual = False
        eq_val = current_equity if (current_equity is not None and not math.isnan(current_equity)) else 0.0
        print(f"[RISK] Daily equity checkpoint reset to {eq_val:.2f} USDT")

    def update_trailing_stop(self, entry_price: float, extreme_price: float, stop_loss: float, curr_atr: float, position_side: str = "LONG") -> float:
        if stop_loss is None or curr_atr is None or extreme_price is None:
            return stop_loss if stop_loss is not None else 0.0
        if math.isnan(stop_loss) or math.isnan(curr_atr) or math.isnan(extreme_price):
            return stop_loss
        trailing_offset = curr_atr * Config.TRAILING_ATR_MULT
        if position_side.upper() == "LONG":
            new_stop = extreme_price - trailing_offset
            return new_stop if new_stop > stop_loss else stop_loss
        if position_side.upper() == "SHORT":
            new_stop = extreme_price + trailing_offset
            return new_stop if new_stop < stop_loss else stop_loss
        return stop_loss

    def calculate_kelly_risk_pct(self, recent_trades: list[dict], base_risk: float | None = None) -> float:
        base = base_risk or Config.RISK_PCT
        lookback = getattr(Config, 'KELLY_LOOKBACK_TRADES', 20)
        trades = recent_trades[-lookback:] if len(recent_trades) > lookback else recent_trades
        if len(trades) < 10:
            return base
        wins = [t for t in trades if float(t.get('pnl_usdt', 0)) > 0]
        losses = [t for t in trades if float(t.get('pnl_usdt', 0)) < 0]
        if not losses:
            return min(base * 1.5, 2.0)
        if not wins:
            return max(0.2, base * 0.25)
        W = len(wins) / len(trades)
        avg_win = sum(float(t.get('pnl_usdt', 0)) for t in wins) / len(wins)
        avg_loss = abs(sum(float(t.get('pnl_usdt', 0)) for t in losses) / len(losses))
        R = avg_win / avg_loss if avg_loss > 0 else 1.0
        kelly = W - ((1 - W) / R)
        half_kelly = kelly * 0.5
        dynamic_risk = max(0.2, min(half_kelly * 100, 2.0))
        print(f"[KELLY] Win Rate: {W*100:.1f}%, Avg Win/Loss Ratio: {R:.2f}, Kelly%: {kelly*100:.2f}%, Half-Kelly Risk: {dynamic_risk:.2f}%")
        return round(dynamic_risk, 2)
