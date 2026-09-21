"""
PrimeSignal Backtest Engine
===========================

H-04 FIX — the previous engine did NOT model the strategy the bot actually
trades. It used a single take-profit, never implemented the TP1/TP2 scale-out
(it even computed `take_profit_1r` and then never read it), locked breakeven at
0.15% instead of the live 0.30% fee buffer, used a 0.50 trailing activation
instead of 0.55, and ignored the daily loss breaker, the daily trade cap, the
cooldown gates, the cluster-risk penalty and the ML entry gate. Any number it
produced was therefore not evidence about the live system.

This engine now mirrors the live execution path:

  * risk ladder          -> Config.risk_pct_for_score(score)
  * entry gating         -> strategy.generate_signal(..., allow_short=venue)
                            plus the ML gate when the model proved CV edge
  * TP1 / TP2 / runner   -> TP1_SCALE_OUT_PCT, TP2_REMAINING_SCALE_PCT, 4.0R
  * breakeven lock       -> DYNAMIC_BE_BUFFER_PCT (+0.35R floor), TSL_ACTIVATION_R
  * ATR trailing         -> TRAILING_ATR_MULT, only after TP1 is booked
  * daily controls       -> MAX_DAILY_LOSS_PCT, ENABLE_DAILY_PROFIT_LOCK,
                            MAX_DAILY_TRADES, COOLDOWN_MINUTES,
                            CONSECUTIVE_LOSS_LIMIT, cluster risk penalty
  * costs                -> FEE_RATE on BOTH legs of every leg, plus slippage

WIN DEFINITION: a trade is a win only when its NET lifecycle PnL > 0.
Breakeven outcomes are reported separately and are NOT counted as wins.
(The old prove_80pct_winrate.py script counted `pnl >= 0` as a win, which is
the single biggest reason its headline number was meaningless.)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from config import Config
from strategies.indicators import prepare_dataframe, calculate_atr
from risk.risk_manager import RiskManager

METRICS_VERSION = "2.6-live-parity"


class BacktestEngine:
    def __init__(self, strategy, risk_manager=None, ml_confirmator=None, allow_short=None, overrides=None):
        self.strategy = strategy
        self.risk = risk_manager if risk_manager else RiskManager()
        self.ml = ml_confirmator
        # C-01 parity: a spot backtest must not model shorts, exactly like live.
        self.allow_short = Config.venue_supports_short() if allow_short is None else bool(allow_short)
        # Optional Config overrides for parameter sweeps. Applied only for the
        # duration of run() and restored afterwards, so a sweep can never leak a
        # tuned parameter into the live process.
        self.overrides: dict = dict(overrides or {})

        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.balance = 0.0
        self.relaxed_enabled = True

    def _push_overrides(self) -> dict:
        saved = {}
        for key, value in self.overrides.items():
            if hasattr(Config, key):
                saved[key] = getattr(Config, key)
                setattr(Config, key, value)
        return saved

    @staticmethod
    def _pop_overrides(saved: dict) -> None:
        for key, value in saved.items():
            setattr(Config, key, value)

    def run(self, htf_candles, ltf_candles, initial_balance=10000.0):
        """Runs the simulation with any configured overrides applied temporarily."""
        saved = self._push_overrides()
        try:
            return self._run_inner(htf_candles, ltf_candles, initial_balance)
        finally:
            self._pop_overrides(saved)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _adverse_first(bar, stop_loss: float, target: float, is_long: bool) -> bool:
        """Resolves intrabar SL-vs-target ambiguity conservatively.

        OHLC cannot tell us which extreme printed first, so when both levels are
        inside a single bar we assume the STOP was touched first unless the bar
        OPENED closer to the target. This under-reports favourable outcomes,
        which is the correct direction to err for a validation harness.
        """
        open_p = float(bar['open'])
        close_p = float(bar['close'])
        if is_long:
            bearish = close_p < open_p
        else:
            bearish = close_p > open_p
        return bearish or abs(open_p - stop_loss) <= abs(open_p - target)

    def _fee(self, qty: float, price: float) -> float:
        return abs(qty) * abs(price) * float(getattr(Config, 'FEE_RATE', 0.00075))

    # ── main simulation ──────────────────────────────────────────────────────

    def _run_inner(self, htf_candles, ltf_candles, initial_balance=10000.0):
        print(f"[BACKTEST] Initializing live-parity simulation with {initial_balance:.2f} USDT...")

        self.trades = []
        self.equity_curve = []
        self.balance = float(initial_balance)
        self.relaxed_enabled = bool(getattr(Config, 'ENABLE_RELAXED_MODE', False))

        htf_df = prepare_dataframe(htf_candles) if not isinstance(htf_candles, pd.DataFrame) else htf_candles
        ltf_df = prepare_dataframe(ltf_candles) if not isinstance(ltf_candles, pd.DataFrame) else ltf_candles

        if len(htf_df) < Config.TREND_EMA or len(ltf_df) < Config.LONG_EMA + 10:
            print("ERROR: Not enough historical candles to run backtest.")
            return None

        print(f"[BACKTEST] HTF {len(htf_df)} bars | LTF {len(ltf_df)} bars | shorts={'ON' if self.allow_short else 'OFF (spot)'}")

        fee_rate = float(getattr(Config, 'FEE_RATE', 0.00075))
        slippage_pct = float(getattr(Config, 'MAX_SLIPPAGE_PCT', 0.004)) * 0.5
        tp1_scale = float(getattr(Config, 'TP1_SCALE_OUT_PCT', 0.65))
        tp2_scale = float(getattr(Config, 'TP2_REMAINING_SCALE_PCT', 0.65))
        be_buffer = float(getattr(Config, 'DYNAMIC_BE_BUFFER_PCT', 0.0030))
        tsl_activation_r = float(getattr(Config, 'TSL_ACTIVATION_R', 1.2))
        trail_mult = float(getattr(Config, 'TRAILING_ATR_MULT', 1.5))
        cooldown_secs = float(getattr(Config, 'COOLDOWN_MINUTES', 20)) * 60.0
        tp_exit_cooldown = float(getattr(Config, 'TP_EXIT_COOLDOWN_MINUTES', 25)) * 60.0
        post_exit_cooldown = float(getattr(Config, 'POST_EXIT_COOLDOWN_MINUTES', 15)) * 60.0
        max_daily_trades = int(getattr(Config, 'MAX_DAILY_TRADES', 6))
        max_daily_loss_pct = float(getattr(Config, 'MAX_DAILY_LOSS_PCT', 2.0))
        max_daily_profit_pct = float(getattr(Config, 'MAX_DAILY_PROFIT_PCT', 10.0))
        profit_lock_enabled = bool(getattr(Config, 'ENABLE_DAILY_PROFIT_LOCK', True))
        loss_limit = int(getattr(Config, 'CONSECUTIVE_LOSS_LIMIT', 2))

        # HTF bar duration so in-progress HTF bars are excluded (no look-ahead).
        if len(htf_df) > 1:
            diffs = htf_df.index.to_series().diff().dropna()
            htf_bar_duration = diffs.median() if not diffs.empty else pd.Timedelta(hours=1)
        else:
            htf_bar_duration = pd.Timedelta(hours=1)
        if not isinstance(htf_bar_duration, pd.Timedelta):
            htf_bar_duration = pd.Timedelta(hours=1)

        start_idx = max(Config.LONG_EMA * 4, 100)
        if start_idx >= len(ltf_df) - 50:
            start_idx = max(10, len(ltf_df) // 4)

        start_ts = pd.Timestamp(str(ltf_df.index[start_idx]))
        self.equity_curve.append({'timestamp': start_ts, 'equity': float(self.balance)})

        # ── position state ───────────────────────────────────────────────────
        in_position = False
        position_side = 'LONG'
        entry_price = 0.0
        initial_sl = 0.0
        stop_loss = 0.0
        tp1 = tp2 = runner_tp = 0.0
        size_total = size_remaining = 0.0
        realized_pnl = 0.0
        leg_fees = 0.0
        best_price = 0.0
        entry_time = None
        tp1_done = tp2_done = False
        stages: list[str] = []
        setup_mode = 'STRICT'
        current_trade_id = 0

        # ── portfolio / daily state ──────────────────────────────────────────
        last_trade_ts = -1e18
        last_zone_traded = None
        volatility_pause_until = -1
        cooldown_until = -1e18
        global_pause_until = -1e18
        trade_history: list[int] = []
        cluster_penalty = False

        current_day = None
        day_start_equity = float(initial_balance)
        day_trade_count = 0
        day_locked = False
        relaxed_trades_today = 0
        last_trade_day = start_ts.date() if isinstance(start_ts, pd.Timestamp) else None

        def mark_equity(price: float) -> float:
            """Mark-to-market equity. LONG holds the asset in cash terms; SHORT is
            treated futures-style (only realized PnL + fees sit in cash)."""
            if not in_position:
                return self.balance
            if position_side == 'LONG':
                return self.balance + (size_remaining * price)
            return self.balance + (size_remaining * (entry_price - price))

        def apply_leg(qty: float, leg_exit_price: float, is_long: bool) -> float:
            """Books ONE exit leg consistently: cash, realized PnL and fees.

            The entry fee is charged in full when the position is opened, so each
            leg charges only its own exit fee — no fee is ever double counted
            (the previous version charged the entry fee again on every leg while
            never crediting partial proceeds to cash).
            """
            nonlocal realized_pnl, leg_fees, size_remaining
            if qty <= 0:
                return 0.0
            exit_fee = self._fee(qty, leg_exit_price)
            if is_long:
                gross = qty * (leg_exit_price - entry_price)
                self.balance += (qty * leg_exit_price) - exit_fee
            else:
                gross = qty * (entry_price - leg_exit_price)
                self.balance += gross - exit_fee
            net = gross - exit_fee
            realized_pnl += net
            leg_fees += exit_fee
            size_remaining -= qty
            return net

        for i in range(start_idx, len(ltf_df)):
            bar = ltf_df.iloc[i]
            ltf_time = pd.Timestamp(str(ltf_df.index[i]))
            ltf_ts = float(ltf_time.timestamp())
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])
            open_p = float(bar['open'])

            sub_ltf = ltf_df.iloc[max(0, i - 250):i + 1]
            sub_htf = htf_df[htf_df.index <= (ltf_time - htf_bar_duration)].iloc[-250:]
            _atr = calculate_atr(sub_ltf, Config.ATR_PERIOD)
            curr_atr = float(_atr.iloc[-1]) if len(_atr) and not pd.isna(_atr.iloc[-1]) else 0.0
            if curr_atr <= 0:
                curr_atr = max(close * 0.005, 1e-9)

            # ── daily rollover / circuit breakers ────────────────────────────
            bar_day = ltf_time.date() if isinstance(ltf_time, pd.Timestamp) else None
            if bar_day != last_trade_day:
                last_trade_day = bar_day
                day_start_equity = mark_equity(close)
                day_trade_count = 0
                day_locked = False
                relaxed_trades_today = 0

            if bar_day != current_day:
                current_day = bar_day

            equity_now = mark_equity(close)
            day_pnl_pct = ((equity_now - day_start_equity) / day_start_equity * 100.0) if day_start_equity > 0 else 0.0
            if day_pnl_pct <= -max_daily_loss_pct:
                day_locked = True
            if profit_lock_enabled and day_pnl_pct >= max_daily_profit_pct:
                day_locked = True

            self.equity_curve.append({'timestamp': ltf_time, 'equity': equity_now})

            # ── manage the open position ─────────────────────────────────────
            if in_position:
                is_long = position_side == 'LONG'
                best_price = max(best_price, high) if is_long else min(best_price, low)
                r_dist = max(abs(entry_price - initial_sl), 1e-12)

                if is_long:
                    best_extreme, adverse = best_price, low
                else:
                    best_extreme, adverse = best_price, high

                # 1) Breakeven lock (mirrors live: max(TSL_ACTIVATION_R*R, 1.5x fee buffer))
                min_required = max(tsl_activation_r * r_dist, entry_price * be_buffer * 1.5)
                reached_be = (best_extreme >= entry_price + min_required) if is_long else (best_extreme <= entry_price - min_required)
                if tp1_done or reached_be:
                    be_sl = entry_price * (1 + be_buffer) if is_long else entry_price * (1 - be_buffer)
                    stop_loss = max(stop_loss, be_sl) if is_long else min(stop_loss, be_sl)

                # 2) TP1 partial
                if not tp1_done:
                    hit_tp1 = high >= tp1 if is_long else low <= tp1
                    if hit_tp1 and self._adverse_first(bar, stop_loss, tp1, is_long) is False:
                        apply_leg(size_total * tp1_scale, tp1, is_long)
                        tp1_done = True
                        stages.append('TP1')
                        # Live locks SL to max(entry*(1+buffer), entry + 0.35R)
                        min_lock = max(entry_price * (1 + be_buffer), entry_price + 0.35 * r_dist) if is_long \
                            else min(entry_price * (1 - be_buffer), entry_price - 0.35 * r_dist)
                        stop_loss = max(stop_loss, min_lock) if is_long else min(stop_loss, min_lock)

                # 3) TP2 partial (of the remainder)
                if tp1_done and not tp2_done:
                    hit_tp2 = high >= tp2 if is_long else low <= tp2
                    if hit_tp2 and self._adverse_first(bar, stop_loss, tp2, is_long) is False:
                        apply_leg(size_remaining * tp2_scale, tp2, is_long)
                        tp2_done = True
                        stages.append('TP2')
                        # Live locks SL at the TP1 level after TP2.
                        stop_loss = max(stop_loss, tp1) if is_long else min(stop_loss, tp1)

                # 4) ATR trailing, only once TP1 is booked (same as live)
                if tp1_done:
                    trail = best_extreme - (curr_atr * trail_mult) if is_long else best_extreme + (curr_atr * trail_mult)
                    stop_loss = max(stop_loss, trail) if is_long else min(stop_loss, trail)

                # 5) Resolve exits for the remaining size
                hit_runner = high >= runner_tp if is_long else low <= runner_tp
                hit_stop = low <= stop_loss if is_long else high >= stop_loss

                exit_price = None
                exit_reason = None
                if hit_runner and hit_stop and not self._adverse_first(bar, stop_loss, runner_tp, is_long):
                    exit_price, exit_reason = runner_tp, 'TAKE_PROFIT_RUNNER'
                elif hit_runner and not hit_stop:
                    exit_price, exit_reason = runner_tp, 'TAKE_PROFIT_RUNNER'
                elif hit_stop:
                    # Gap-through: fill at the open when it is worse than the stop.
                    if is_long:
                        exit_price = min(stop_loss, open_p) if open_p < stop_loss else stop_loss
                    else:
                        exit_price = max(stop_loss, open_p) if open_p > stop_loss else stop_loss
                    # Label from the actual FILL, not the stop level, so a gap-through
                    # is never reported as a clean "breakeven lock".
                    exit_reason = self._stop_reason(exit_price, entry_price, tp2_done, is_long)

                if exit_price is not None:
                    apply_leg(size_remaining, exit_price, is_long)
                    stages.append(exit_reason)

                    is_win = realized_pnl > 0
                    is_loss = realized_pnl < 0
                    self.trades.append({
                        'id': current_trade_id,
                        'symbol': getattr(Config, 'SYMBOL', 'BTC/USDT'),
                        'side': position_side,
                        'entry_time': entry_time,
                        'exit_time': ltf_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': size_total,
                        'pnl': realized_pnl,             # legacy alias
                        'pnl_usdt': realized_pnl,
                        'total_pnl_net': realized_pnl,
                        'pnl_pct': (realized_pnl / (size_total * entry_price) * 100.0) if (size_total * entry_price) > 0 else 0.0,
                        'total_fees': leg_fees,
                        'exit_reason': exit_reason,
                        'stages_completed': list(stages),
                        'mode': setup_mode,              # legacy alias
                        'setup_mode': setup_mode,
                        'tp1_hit': 'TP1' in stages,
                        'tp2_hit': 'TP2' in stages,
                        'is_win': is_win,
                        'is_breakeven': not is_win and not is_loss,
                    })

                    # cooldown + cluster risk bookkeeping (mirrors live)
                    cooldown_until = ltf_ts + (tp_exit_cooldown if ('TP1' in stages or is_win) else post_exit_cooldown)
                    trade_history.append(1 if is_loss else 0)
                    trade_history = trade_history[-6:]
                    if len(trade_history) >= loss_limit and all(trade_history[-loss_limit:]):
                        global_pause_until = ltf_ts + (900.0 if Config.PAPER_TRADING else 3600.0)
                        trade_history = []
                    cluster_penalty = len(trade_history) >= 6 and sum(trade_history) >= 3

                    in_position = False
                    size_remaining = 0.0
                    continue

            # ── look for a new entry ─────────────────────────────────────────
            if in_position:
                continue
            if ltf_ts < global_pause_until or ltf_ts < cooldown_until:
                continue
            if day_locked or day_trade_count >= max_daily_trades:
                continue
            if i < volatility_pause_until:
                continue

            move_pct = abs(close - open_p) / open_p if open_p > 0 else 0.0
            if move_pct > float(getattr(Config, 'MAX_CANDLE_MOVE_PCT', 0.015)):
                volatility_pause_until = i + int(getattr(Config, 'VOLATILITY_PAUSE_CANDLES', 2))
                continue

            signal, meta = self.strategy.generate_signal(sub_htf, sub_ltf, relaxed=False, allow_short=self.allow_short)
            setup_mode = 'STRICT'
            if signal == 'HOLD' and self.relaxed_enabled and (ltf_ts - last_trade_ts) >= 30 * 60:
                if relaxed_trades_today < 2:
                    r_signal, r_meta = self.strategy.generate_signal(sub_htf, sub_ltf, relaxed=True, allow_short=self.allow_short)
                    if r_signal in ('BUY', 'SELL'):
                        signal, meta, setup_mode = r_signal, r_meta, 'RELAXED'

            if signal not in ('BUY', 'SELL'):
                continue
            if signal == 'SELL' and not self.allow_short:
                continue

            sl_raw = meta.get('stop_loss')
            if sl_raw is None:
                continue
            sl = float(sl_raw)

            zone_id = meta.get('zone_id')
            if zone_id and zone_id == last_zone_traded:
                continue

            # ── ML gate, identical policy to live ────────────────────────────
            ml_prob_dir = 1.0
            ml_gated = False
            if self.ml is not None:
                raw_bias = self.ml.predict_bias(sub_ltf)
                ml_prob_dir = raw_bias if signal == 'BUY' else (1.0 - raw_bias)
                if hasattr(self.ml, 'should_gate_entries') and self.ml.should_gate_entries():
                    threshold = float(getattr(Config, 'ML_CONFIRMATION_THRESHOLD', 0.60))
                    if ml_prob_dir < threshold:
                        ml_gated = True
                elif ml_prob_dir < float(getattr(Config, 'ML_CONFIRMATION_THRESHOLD', 0.60)):
                    # Legacy behaviour retained for raw-vs-ML comparison runs:
                    # under-confident signals are half-sized, not blocked.
                    pass
            if ml_gated:
                continue

            entry_price = close * (1 + slippage_pct) if signal == 'BUY' else close * (1 - slippage_pct)
            r_dist = abs(entry_price - sl)
            if r_dist <= 0:
                continue

            # ── targets (live geometry) ──────────────────────────────────────
            tp1_mult = float(getattr(Config, 'MIN_RISK_REWARD_RATIO', 1.5))
            tp2_mult = float(getattr(Config, 'RISK_REWARD_RATIO', 2.5))
            if self.ml is not None:
                if ml_prob_dir > 0.65:
                    tp2_mult = 2.5
                elif ml_prob_dir < 0.55:
                    tp2_mult = 1.8
            if signal == 'BUY':
                tp1 = entry_price + tp1_mult * r_dist
                tp2 = entry_price + tp2_mult * r_dist
                runner_tp = entry_price + 4.0 * r_dist
            else:
                tp1 = entry_price - tp1_mult * r_dist
                tp2 = entry_price - tp2_mult * r_dist
                runner_tp = entry_price - 4.0 * r_dist

            # ── sizing via the SAME risk ladder as live ──────────────────────
            risk_pct = Config.risk_pct_for_score(float(meta.get('score') or 3.0))
            if cluster_penalty:
                risk_pct *= 0.5
            if self.ml is not None and ml_prob_dir < float(getattr(Config, 'ML_CONFIRMATION_THRESHOLD', 0.60)):
                risk_pct *= 0.5

            equity_for_sizing = mark_equity(close)
            pos_size = self.risk.calculate_position_size(
                account_equity=equity_for_sizing,
                entry_price=entry_price,
                stop_loss=sl,
                quote_currency='USDT',
                equity_currency='USDT',
                conversion_rate=1.0,
                is_inr=False,
                risk_pct_override=risk_pct,
            )
            if pos_size <= 0:
                continue

            # ── open ─────────────────────────────────────────────────────────
            current_trade_id += 1
            position_side = 'LONG' if signal == 'BUY' else 'SHORT'
            entry_time = ltf_time
            initial_sl = sl
            stop_loss = sl
            size_total = pos_size
            size_remaining = pos_size
            # Cash model: LONG pays asset cost + entry fee up front; SHORT is
            # marked to market and pays only the entry fee. Entry fee is charged
            # once, in full, here.
            leg_fees = self._fee(pos_size, entry_price)
            self.balance -= leg_fees
            realized_pnl = -leg_fees
            if position_side == 'LONG':
                self.balance -= pos_size * entry_price
            best_price = entry_price
            tp1_done = tp2_done = False
            stages = ['ENTRY', setup_mode]
            last_trade_ts = ltf_ts
            last_zone_traded = zone_id
            day_trade_count += 1
            if setup_mode == 'RELAXED':
                relaxed_trades_today += 1
            in_position = True

        # ── liquidate anything still open ────────────────────────────────────
        if in_position:
            exit_price = float(ltf_df['close'].iloc[-1])
            apply_leg(size_remaining, exit_price, position_side == 'LONG')
            self.trades.append({
                'id': current_trade_id,
                'symbol': getattr(Config, 'SYMBOL', 'BTC/USDT'),
                'side': position_side,
                'entry_time': entry_time,
                'exit_time': ltf_df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size_total,
                'pnl': realized_pnl,
                'pnl_usdt': realized_pnl,
                'total_pnl_net': realized_pnl,
                'pnl_pct': (realized_pnl / (size_total * entry_price) * 100.0) if (size_total * entry_price) > 0 else 0.0,
                'total_fees': leg_fees,
                'exit_reason': 'FORCE_CLOSE_END',
                'stages_completed': list(stages) + ['FORCE_CLOSE_END'],
                'mode': setup_mode,
                'setup_mode': setup_mode,
                'tp1_hit': 'TP1' in stages,
                'tp2_hit': 'TP2' in stages,
                'is_win': realized_pnl > 0,
                'is_breakeven': realized_pnl == 0,
            })

        return self.calculate_metrics(initial_balance)

    @staticmethod
    def _stop_reason(fill_price: float, entry_price: float, tp2_done: bool, is_long: bool) -> str:
        """Classifies a stop exit from the price actually filled.

        Using the fill (not the stop level) means a gap-through below a
        breakeven-locked stop is reported as a STOP_LOSS rather than as a
        flattering "breakeven lock".
        """
        better_than_entry = fill_price > entry_price if is_long else fill_price < entry_price
        if better_than_entry:
            return 'TRAILING_STOP' if tp2_done else 'BREAKEVEN_LOCK'
        return 'STOP_LOSS'

    # ── metrics ──────────────────────────────────────────────────────────────

    def calculate_metrics(self, initial_balance):
        base = {
            'metrics_version': METRICS_VERSION,
            'allow_short': self.allow_short,
            'initial_balance': initial_balance,
            'final_balance': self.balance,
            'net_profit': 0.0,
            'total_return_pct': 0.0,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakevens': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'expectancy': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'max_consecutive_losses': 0,
            'total_fees': 0.0,
            'strict_win_rate': 0.0,
            'strict_pf': 0.0,
            'strict_dd': 0.0,
            'relaxed_win_rate': 0.0,
            'relaxed_pf': 0.0,
            'relaxed_dd': 0.0,
            'trade_list': [],
        }
        if not self.trades:
            return base

        trade_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        def subset_metrics(df_sub):
            if len(df_sub) == 0:
                return 0.0, 0.0, 0.0
            w = df_sub[df_sub['pnl_usdt'] > 0]
            l = df_sub[df_sub['pnl_usdt'] < 0]
            wr = (len(w) / len(df_sub)) * 100.0
            gross_w = float(w['pnl_usdt'].sum()) if len(w) else 0.0
            gross_l = abs(float(l['pnl_usdt'].sum())) if len(l) else 0.0
            pf = (gross_w / gross_l) if gross_l > 0 else (float('inf') if gross_w > 0 else 0.0)
            cum = pd.concat([pd.Series([0.0]), df_sub['pnl_usdt']], ignore_index=True).cumsum()
            peak = cum.cummax()
            dd = float(((cum - peak) / initial_balance * 100.0).min())
            return wr, pf, dd

        wins = trade_df[trade_df['pnl_usdt'] > 0]
        losses = trade_df[trade_df['pnl_usdt'] < 0]
        breakevens = trade_df[trade_df['pnl_usdt'] == 0]

        total_trades = len(trade_df)
        net_profit = float(trade_df['pnl_usdt'].sum())
        win_rate = (len(wins) / total_trades) * 100.0

        gross_profit = float(wins['pnl_usdt'].sum()) if len(wins) else 0.0
        gross_loss = abs(float(losses['pnl_usdt'].sum())) if len(losses) else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        avg_win = (gross_profit / len(wins)) if len(wins) else 0.0
        avg_loss = (gross_loss / len(losses)) if len(losses) else 0.0
        win_pct = len(wins) / total_trades
        loss_pct = len(losses) / total_trades
        expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)

        # drawdown on the mark-to-market equity curve
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['dd'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'].replace(0, np.nan) * 100.0
        max_drawdown_pct = float(equity_df['dd'].min()) if len(equity_df) else 0.0

        # daily-return Sharpe, annualised over 365 crypto trading days
        equity_ts = equity_df.set_index('timestamp')['equity'].resample('1D').last().dropna()
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        if len(equity_ts) > 1:
            daily = equity_ts.pct_change().dropna()
            std = float(daily.std())
            mean = float(daily.mean())
            if std > 0:
                sharpe_ratio = (mean / std) * np.sqrt(365)
            downside = daily[daily < 0]
            dstd = float(downside.std()) if len(downside) > 1 else 0.0
            if dstd > 0:
                sortino_ratio = (mean / dstd) * np.sqrt(365)

        # max consecutive losses
        max_consec_losses = 0
        streak = 0
        for pnl in trade_df['pnl_usdt']:
            if pnl < 0:
                streak += 1
                max_consec_losses = max(max_consec_losses, streak)
            else:
                streak = 0

        if 'setup_mode' in trade_df.columns:
            strict_df = trade_df[trade_df['setup_mode'] == 'STRICT']
            relaxed_df = trade_df[trade_df['setup_mode'] == 'RELAXED']
        else:
            strict_df, relaxed_df = trade_df, pd.DataFrame(columns=trade_df.columns)

        swr, spf, sdd = subset_metrics(strict_df)
        rwr, rpf, rdd = subset_metrics(relaxed_df)

        if isinstance(rpf, float) and rpf > 0 and rpf < spf:
            print(f"[BACKTEST] WARNING: Relaxed PF ({rpf:.2f}) < Strict PF ({spf:.2f}). Consider disabling relaxed mode.")

        return {
            'metrics_version': METRICS_VERSION,
            'allow_short': self.allow_short,
            'initial_balance': initial_balance,
            'final_balance': self.balance,
            'net_profit': net_profit,
            'total_return_pct': ((self.balance - initial_balance) / initial_balance) * 100.0 if initial_balance else 0.0,
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'breakevens': len(breakevens),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'max_consecutive_losses': max_consec_losses,
            'total_fees': float(trade_df['total_fees'].sum()) if 'total_fees' in trade_df.columns else 0.0,
            'strict_win_rate': swr,
            'strict_pf': spf,
            'strict_dd': sdd,
            'relaxed_win_rate': rwr,
            'relaxed_pf': rpf,
            'relaxed_dd': rdd,
            'trade_list': self.trades,
        }
