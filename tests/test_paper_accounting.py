from core.paper_accounting import net_leg_pnl_pct, simulate_paper_entry, simulate_paper_exit


def test_buy_entry_uses_actual_slipped_fill_and_debits_entry_fee():
    result = simulate_paper_entry("BUY", 1.0, 100.0, 1000.0, 1000.0, 0.50, 0.01, 0.001)
    assert result is not None
    assert result.fill_price == 101.0
    assert abs(result.fee_usdt - 0.101) < 1e-12
    assert abs(result.cash_debit - 101.101) < 1e-12


def test_short_entry_uses_actual_slipped_fill_and_debits_entry_fee():
    result = simulate_paper_entry("SELL", 1.0, 100.0, 1000.0, 1000.0, 0.50, 0.01, 0.001)
    assert result is not None
    assert result.fill_price == 99.0
    assert abs(result.fee_usdt - 0.099) < 1e-12
    assert abs(result.cash_debit - 99.099) < 1e-12


def test_long_exit_credits_proceeds_less_exit_fee():
    result = simulate_paper_exit("LONG", 1.0, 100.0, 110.0, 0.001)
    assert abs(result.gross_pnl_usdt - 10.0) < 1e-12
    assert abs(result.exit_fee_usdt - 0.11) < 1e-12
    assert abs(result.cash_credit - 109.89) < 1e-12


def test_short_exit_returns_collateral_and_pnl_less_exit_fee():
    result = simulate_paper_exit("SHORT", 1.0, 100.0, 90.0, 0.001)
    assert abs(result.gross_pnl_usdt - 10.0) < 1e-12
    assert abs(result.exit_fee_usdt - 0.09) < 1e-12
    assert abs(result.cash_credit - 109.91) < 1e-12


def test_net_leg_pnl_pct_matches_net_pnl():
    assert abs(net_leg_pnl_pct(8.789, 1.0, 100.0) - 8.789) < 1e-12


def test_minimum_ticket_never_bypasses_allocation_cap():
    result = simulate_paper_entry(
        "BUY", 10.0, 100.0, 100.0, 100.0,
        max_alloc_pct=0.35, slippage_pct=0.0, fee_rate=0.0,
        min_paper_cash=50.0,
    )
    assert result is None


def test_entry_cash_debit_never_exceeds_allocation_cap():
    result = simulate_paper_entry(
        "BUY", 10.0, 100.0, 1000.0, 1000.0,
        max_alloc_pct=0.35, slippage_pct=0.01, fee_rate=0.001,
        min_paper_cash=1.0,
    )
    assert result is not None
    assert result.cash_debit <= 350.0 + 1e-9


def test_futures_3x_leverage_entry_and_exit():
    # Buy 1 unit at 100 with 3x leverage -> notional 100, margin ~33.33 + fee
    entry = simulate_paper_entry("BUY", 1.0, 100.0, 100.0, 100.0, 0.45, 0.0, 0.001, leverage=3.0)
    assert entry is not None
    assert entry.quantity == 1.0
    # margin = 100/3 = 33.333333333333336, fee = 100 * 0.001 = 0.1
    expected_debit = (100.0 / 3.0) + 0.1
    assert abs(entry.cash_debit - expected_debit) < 1e-9

    # Exit at 110 (win of +10 PnL) -> margin released (33.33) + 10 PnL - exit fee (0.11)
    exit_win = simulate_paper_exit("LONG", 1.0, 100.0, 110.0, 0.001, leverage=3.0)
    expected_credit = (100.0 / 3.0) + 10.0 - 0.11
    assert abs(exit_win.cash_credit - expected_credit) < 1e-9
    assert abs(exit_win.gross_pnl_usdt - 10.0) < 1e-9

