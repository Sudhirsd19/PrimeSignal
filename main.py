"""PrimeSignal runtime entrypoint with deterministic hardening overlays.

The historical 2k+ line implementation is retained byte-for-byte in
``main_legacy.py``.  This thin entrypoint applies verified source-level safety
transformations before executing the application, which lets us harden the
large legacy file without manually rewriting unrelated code.
"""

import asyncio
import math
import sys
import types
from pathlib import Path

LEGACY_PATH = Path(__file__).with_name("main_legacy.py")


def _replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError("PrimeSignal hardening anchor mismatch for %s: expected 1, found %d" % (label, count))
    return text.replace(old, new, 1)


def _replace_exact_count(text, old, new, expected, label):
    count = text.count(old)
    if count != expected:
        raise RuntimeError("PrimeSignal hardening anchor mismatch for %s: expected %d, found %d" % (label, expected, count))
    return text.replace(old, new)


def _patch_source(text):
    """Apply only audited, deterministic transformations to the legacy source."""
    import_marker = "from core.macro_calendar import MacroNewsCalendar\n"
    import_add = import_marker + "from core.paper_accounting import simulate_paper_entry, simulate_paper_exit, net_leg_pnl_pct\n"
    if "from core.paper_accounting import " not in text:
        text = _replace_once(text, import_marker, import_add, "paper accounting import")
    if "import math\n" not in text:
        text = "import math\n" + text

    buy_old = '''                    min_paper_cost = 50.0 if is_inr else 1.0
                    cur_sym = "₹" if is_inr else "$"
                    max_alloc_pct = getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35)
                    max_allowed_cost = current_equity * max_alloc_pct
                    entry_cost_equity_curr = pos_size * entry_price * (conversion_rate if is_inr else 1.0)

                    if entry_cost_equity_curr > max_allowed_cost:
                        pos_size = max_allowed_cost / (entry_price * (conversion_rate if is_inr else 1.0))
                        entry_cost_equity_curr = max_allowed_cost

                    slippage_pct = getattr(Config, 'PAPER_SLIPPAGE_PCT', 0.0005)
                    sim_buy_fill = round(entry_price * (1.0 + slippage_pct), 4)
                    if entry_cost_equity_curr <= self._dry_run_balance_usdt:
                        self._dry_run_balance_usdt -= entry_cost_equity_curr
                        order = {'id': f'MOCK_BUY_{int(time.time()*1000)}', 'price': sim_buy_fill, 'average': sim_buy_fill, 'amount': pos_size, 'status': 'filled'}
                    elif self._dry_run_balance_usdt >= min_paper_cost:
                        usable_cash = min(self._dry_run_balance_usdt, self._dry_run_balance_usdt * max_alloc_pct)
                        if usable_cash < min_paper_cost:
                            usable_cash = self._dry_run_balance_usdt
                        pos_size = usable_cash / (entry_price * (conversion_rate if is_inr else 1.0))
                        self._dry_run_balance_usdt -= usable_cash
                        order = {'id': f'MOCK_BUY_{int(time.time()*1000)}', 'price': sim_buy_fill, 'average': sim_buy_fill, 'amount': pos_size, 'status': 'filled'}
                    else:
                        add_log_message(f"[{symbol}] ⚠️ Paper trading wallet balance depleted ({cur_sym}{self._dry_run_balance_usdt:.2f}). Reset wallet to continue.")
'''
    buy_new = '''                    min_paper_cost = 50.0 if is_inr else 1.0
                    cur_sym = "₹" if is_inr else "$"
                    paper_entry = simulate_paper_entry(
                        side="BUY",
                        requested_qty=pos_size,
                        signal_price=entry_price,
                        balance_cash=self._dry_run_balance_usdt,
                        current_equity_cash=current_equity,
                        max_alloc_pct=getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35),
                        slippage_pct=getattr(Config, 'PAPER_SLIPPAGE_PCT', 0.0005),
                        fee_rate=getattr(Config, 'FEE_RATE', 0.00075),
                        conversion_rate=(conversion_rate if is_inr else 1.0),
                        min_paper_cash=min_paper_cost,
                    )
                    if paper_entry is not None:
                        pos_size = paper_entry.quantity
                        self._dry_run_balance_usdt -= paper_entry.cash_debit
                        sim_buy_fill = paper_entry.fill_price
                        order = {'id': f'MOCK_BUY_{int(time.time()*1000)}', 'price': sim_buy_fill, 'average': sim_buy_fill, 'amount': pos_size, 'status': 'filled'}
                    else:
                        add_log_message(f"[{symbol}] ⚠️ Paper trading wallet cannot afford requested BUY allocation ({cur_sym}{self._dry_run_balance_usdt:.2f} available).")
'''
    text = _replace_once(text, buy_old, buy_new, "BUY paper entry")

    sell_old = '''                    min_paper_cost = 50.0 if is_inr else 1.0
                    cur_sym = "₹" if is_inr else "$"
                    max_alloc_pct = getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35)
                    max_allowed_cost = current_equity * max_alloc_pct
                    collateral_equity_curr = pos_size * entry_price * (conversion_rate if is_inr else 1.0)

                    if collateral_equity_curr > max_allowed_cost:
                        pos_size = max_allowed_cost / (entry_price * (conversion_rate if is_inr else 1.0))
                        collateral_equity_curr = max_allowed_cost

                    slippage_pct = getattr(Config, 'PAPER_SLIPPAGE_PCT', 0.0005)
                    sim_sell_fill = round(entry_price * (1.0 - slippage_pct), 4)
                    if collateral_equity_curr <= self._dry_run_balance_usdt:
                        self._dry_run_balance_usdt -= collateral_equity_curr
                        order = {'id': f'MOCK_SELL_{int(time.time()*1000)}', 'price': sim_sell_fill, 'average': sim_sell_fill, 'amount': pos_size, 'status': 'filled'}
                    elif self._dry_run_balance_usdt >= min_paper_cost:
                        usable_cash = min(self._dry_run_balance_usdt, self._dry_run_balance_usdt * max_alloc_pct)
                        if usable_cash < min_paper_cost:
                            usable_cash = self._dry_run_balance_usdt
                        pos_size = usable_cash / (entry_price * (conversion_rate if is_inr else 1.0))
                        self._dry_run_balance_usdt -= usable_cash
                        order = {'id': f'MOCK_SELL_{int(time.time()*1000)}', 'price': sim_sell_fill, 'average': sim_sell_fill, 'amount': pos_size, 'status': 'filled'}
                    else:
                        add_log_message(f"[{symbol}] ⚠️ Paper trading wallet balance depleted ({cur_sym}{self._dry_run_balance_usdt:.2f}). Reset wallet to continue.")
'''
    sell_new = '''                    min_paper_cost = 50.0 if is_inr else 1.0
                    cur_sym = "₹" if is_inr else "$"
                    paper_entry = simulate_paper_entry(
                        side="SELL",
                        requested_qty=pos_size,
                        signal_price=entry_price,
                        balance_cash=self._dry_run_balance_usdt,
                        current_equity_cash=current_equity,
                        max_alloc_pct=getattr(Config, 'MAX_TRADE_ALLOCATION_PCT', 0.35),
                        slippage_pct=getattr(Config, 'PAPER_SLIPPAGE_PCT', 0.0005),
                        fee_rate=getattr(Config, 'FEE_RATE', 0.00075),
                        conversion_rate=(conversion_rate if is_inr else 1.0),
                        min_paper_cash=min_paper_cost,
                    )
                    if paper_entry is not None:
                        pos_size = paper_entry.quantity
                        self._dry_run_balance_usdt -= paper_entry.cash_debit
                        sim_sell_fill = paper_entry.fill_price
                        order = {'id': f'MOCK_SELL_{int(time.time()*1000)}', 'price': sim_sell_fill, 'average': sim_sell_fill, 'amount': pos_size, 'status': 'filled'}
                    else:
                        add_log_message(f"[{symbol}] ⚠️ Paper trading wallet cannot afford requested SELL allocation ({cur_sym}{self._dry_run_balance_usdt:.2f} available).")
'''
    text = _replace_once(text, sell_old, sell_new, "SELL paper entry")

    long_tp1_old = '''                                else:
                                    self._dry_run_balance_usdt += tp1_size * curr_price * (rate if is_inr else 1.0)
                                    tp1_success = True
'''
    long_tp1_new = '''                                else:
                                    paper_tp1 = simulate_paper_exit(
                                        side="LONG",
                                        quantity=tp1_size,
                                        entry_price=self.entry_price[symbol],
                                        exit_price=curr_price,
                                        fee_rate=getattr(Config, 'FEE_RATE', 0.00075),
                                        conversion_rate=(rate if is_inr else 1.0),
                                    )
                                    self._dry_run_balance_usdt += paper_tp1.cash_credit
                                    tp1_success = True
'''
    text = _replace_once(text, long_tp1_old, long_tp1_new, "LONG TP1 paper exit")

    short_tp1_old = '''                                else:
                                    # Short TP cash return = entry_notional + (entry_notional - exit_notional) = profit + collateral
                                    tp1_pnl_usdt = tp1_size * (self.entry_price[symbol] - curr_price)
                                    tp1_proceeds_usdt = tp1_size * self.entry_price[symbol] + tp1_pnl_usdt
                                    self._dry_run_balance_usdt += tp1_proceeds_usdt * (rate if is_inr else 1.0)
                                    tp1_success = True
'''
    short_tp1_new = '''                                else:
                                    paper_tp1 = simulate_paper_exit(
                                        side="SHORT",
                                        quantity=tp1_size,
                                        entry_price=self.entry_price[symbol],
                                        exit_price=curr_price,
                                        fee_rate=getattr(Config, 'FEE_RATE', 0.00075),
                                        conversion_rate=(rate if is_inr else 1.0),
                                    )
                                    self._dry_run_balance_usdt += paper_tp1.cash_credit
                                    tp1_success = True
'''
    text = _replace_once(text, short_tp1_old, short_tp1_new, "SHORT TP1 paper exit")

    tp2_old = '''                                else:
                                    self._dry_run_balance_usdt += tp2_size * curr_price * (rate if is_inr else 1.0)
                                    tp2_success = True
'''
    tp2_new = '''                                else:
                                    paper_tp2 = simulate_paper_exit(
                                        side="LONG" if self.position_side[symbol] == "LONG" else "SHORT",
                                        quantity=tp2_size,
                                        entry_price=self.entry_price[symbol],
                                        exit_price=curr_price,
                                        fee_rate=getattr(Config, 'FEE_RATE', 0.00075),
                                        conversion_rate=(rate if is_inr else 1.0),
                                    )
                                    self._dry_run_balance_usdt += paper_tp2.cash_credit
                                    tp2_success = True
'''
    text = _replace_exact_count(text, tp2_old, tp2_new, 2, "TP2 paper exits")

    exit_long_old = '''                    if not self.has_keys or Config.PAPER_TRADING:
                        # Return cash proceeds from selling the asset at exit_price
                        self._dry_run_balance_usdt += actual_exit * exit_price * (rate if is_inr else 1.0)
'''
    exit_long_new = '''                    if not self.has_keys or Config.PAPER_TRADING:
                        paper_exit = simulate_paper_exit(
                            side="LONG",
                            quantity=actual_exit,
                            entry_price=self.entry_price[symbol],
                            exit_price=exit_price,
                            fee_rate=getattr(Config, 'FEE_RATE', 0.00075),
                            conversion_rate=(rate if is_inr else 1.0),
                        )
                        self._dry_run_balance_usdt += paper_exit.cash_credit
'''
    text = _replace_once(text, exit_long_old, exit_long_new, "LONG final paper exit")

    exit_short_old = '''                    if not self.has_keys or Config.PAPER_TRADING:
                        # C-02 FIX: Return collateral + profit = entry_notional + pnl_usdt
                        # But collateral was deducted at entry, so return the buy-back cost and the profit separately:
                        # Cash back = (collateral freed) + pnl = entry_notional + (entry_notional - exit_notional) = wrong
                        # Correct: collateral freed = entry_notional, cost to close = exit_notional
                        # Net cash returned = entry_notional - exit_notional + entry_notional (collateral) = entry_notional + pnl_usdt
                        # But entry_notional was already DEDUCTED, so re-add collateral + pnl:
                        self._dry_run_balance_usdt += (actual_exit * self.entry_price[symbol] + pnl_usdt) * (rate if is_inr else 1.0)
'''
    exit_short_new = '''                    if not self.has_keys or Config.PAPER_TRADING:
                        paper_exit = simulate_paper_exit(
                            side="SHORT",
                            quantity=actual_exit,
                            entry_price=self.entry_price[symbol],
                            exit_price=exit_price,
                            fee_rate=getattr(Config, 'FEE_RATE', 0.00075),
                            conversion_rate=(rate if is_inr else 1.0),
                        )
                        self._dry_run_balance_usdt += paper_exit.cash_credit
'''
    text = _replace_once(text, exit_short_old, exit_short_new, "SHORT final paper exit")

    text = _replace_exact_count(
        text,
        "                                    tp1_pnl_pct = (curr_price - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0",
        "                                    tp1_pnl_pct = net_leg_pnl_pct(tp1_pnl_usdt, tp1_size, self.entry_price[symbol])",
        1,
        "LONG TP1 pnl percentage",
    )
    text = _replace_exact_count(
        text,
        "                                    tp2_pnl_pct = (curr_price - self.entry_price[symbol]) / self.entry_price[symbol] * 100.0",
        "                                    tp2_pnl_pct = net_leg_pnl_pct(tp2_pnl_usdt, tp2_size, self.entry_price[symbol])",
        1,
        "LONG TP2 pnl percentage",
    )
    text = _replace_exact_count(
        text,
        "                                    tp1_pnl_pct = (self.entry_price[symbol] - curr_price) / self.entry_price[symbol] * 100.0",
        "                                    tp1_pnl_pct = net_leg_pnl_pct(tp1_pnl_usdt, tp1_size, self.entry_price[symbol])",
        1,
        "SHORT TP1 pnl percentage",
    )
    text = _replace_exact_count(
        text,
        "                                    tp2_pnl_pct = (self.entry_price[symbol] - curr_price) / self.entry_price[symbol] * 100.0",
        "                                    tp2_pnl_pct = net_leg_pnl_pct(tp2_pnl_usdt, tp2_size, self.entry_price[symbol])",
        1,
        "SHORT TP2 pnl percentage",
    )

    # Eliminate synthetic 2% stop fallback. Strategy protection is mandatory.
    sl_line = "            sl = float(metadata['stop_loss']) if metadata.get('stop_loss') is not None else (entry_price * 0.98)"
    sl_block = '''            raw_stop_loss = metadata.get('stop_loss')
            if raw_stop_loss is None:
                add_log_message(f"[{symbol}] Trade blocked: strategy did not provide a durable stop-loss.")
                return
            try:
                sl = float(raw_stop_loss)
            except (TypeError, ValueError):
                add_log_message(f"[{symbol}] Trade blocked: strategy stop-loss is invalid ({raw_stop_loss!r}).")
                return
            if not math.isfinite(sl) or sl <= 0:
                add_log_message(f"[{symbol}] Trade blocked: strategy stop-loss is non-finite/non-positive ({raw_stop_loss!r}).")
                return
            if signal == "BUY" and sl >= entry_price:
                add_log_message(f"[{symbol}] Trade blocked: LONG stop-loss {sl:.8f} is not below entry {entry_price:.8f}.")
                return
            if signal == "SELL" and sl <= entry_price:
                add_log_message(f"[{symbol}] Trade blocked: SHORT stop-loss {sl:.8f} is not above entry {entry_price:.8f}.")
                return'''
    text = _replace_once(text, sl_line, sl_block, "strategy stop-loss validation")
    text = text.replace("            # E-01 Fix: Provide safe fallback SL if None\n            sl_val = metadata.get('stop_loss')\n            if sl_val is None: sl_val = entry_price * 0.98 if signal == \"BUY\" else entry_price * 1.02", "            sl_val = sl")

    if "if sl_val is None: sl_val = entry_price * 0.98" in text or "if sl_val is None: sl_val = entry_price * 1.02" in text:
        raise RuntimeError("PrimeSignal hardening failed: synthetic 2% SL fallback still present")

    # Live ENTRY durable protection: the legacy source must pass the exact
    # strategy stop and position side into ExecutionEngine before journaling.
    buy_call = "order = await self.execution.place_order('buy', 'market', pos_size, price=entry_price, symbol=symbol)"
    buy_call_new = "order = await self.execution.place_order('buy', 'market', pos_size, price=entry_price, symbol=symbol, order_role='ENTRY', protection={\"stop_loss\": sl, \"position_side\": \"LONG\", \"signal_entry_price\": entry_price})"
    sell_call = "order = await self.execution.place_order('sell', 'market', pos_size, price=entry_price, symbol=symbol)"
    sell_call_new = "order = await self.execution.place_order('sell', 'market', pos_size, price=entry_price, symbol=symbol, order_role='ENTRY', protection={\"stop_loss\": sl, \"position_side\": \"SHORT\", \"signal_entry_price\": entry_price})"
    text = _replace_once(text, buy_call, buy_call_new, "live BUY protection")
    text = _replace_once(text, sell_call, sell_call_new, "live SELL protection")

    return text


def _load_patched_legacy():
    if not LEGACY_PATH.exists():
        raise RuntimeError("PrimeSignal legacy implementation is missing: %s" % LEGACY_PATH)
    source = LEGACY_PATH.read_text(encoding="utf-8")
    patched = _patch_source(source)
    module = types.ModuleType("main")
    module.__file__ = str(LEGACY_PATH)
    module.__package__ = None
    sys.modules["main"] = module
    exec(compile(patched, str(LEGACY_PATH), "exec"), module.__dict__)
    return module


if __name__ == "__main__":
    legacy = _load_patched_legacy()
    if sys.platform == "win32" and sys.version_info < (3, 12):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(legacy.start_all())
    except KeyboardInterrupt:
        print("\nStopping bot...")
