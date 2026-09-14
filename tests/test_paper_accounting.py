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
