from core.order_state_machine import OrderState, PositionContext


def test_protected_state_rejects_missing_stop_loss():
    ctx = PositionContext("BTC/USDT")
    ctx.side = "LONG"
    ctx.filled_qty = 1.0
    ctx.entry_price = 100.0
    ctx.stop_loss = 0.0

    assert ctx.transition_to(OrderState.PROTECTED, reason="test missing SL") is False
    assert ctx.state == OrderState.EXECUTION_UNKNOWN
    assert ctx.is_protected() is False


def test_protected_state_rejects_directionally_wrong_long_stop():
    ctx = PositionContext("BTC/USDT")
    ctx.side = "LONG"
    ctx.filled_qty = 1.0
    ctx.entry_price = 100.0
    ctx.stop_loss = 101.0

    assert ctx.transition_to(OrderState.PROTECTED, reason="test wrong LONG SL") is False
    assert ctx.state == OrderState.EXECUTION_UNKNOWN


def test_protected_state_rejects_directionally_wrong_short_stop():
    ctx = PositionContext("BTC/USDT")
    ctx.side = "SHORT"
    ctx.filled_qty = 1.0
    ctx.entry_price = 100.0
    ctx.stop_loss = 99.0

    assert ctx.transition_to(OrderState.PROTECTED, reason="test wrong SHORT SL") is False
    assert ctx.state == OrderState.EXECUTION_UNKNOWN


def test_protected_state_accepts_valid_long_protection():
    ctx = PositionContext("BTC/USDT")
    ctx.side = "LONG"
    ctx.filled_qty = 1.0
    ctx.entry_price = 100.0
    ctx.stop_loss = 95.0

    assert ctx.transition_to(OrderState.PROTECTED, reason="test valid LONG") is True
    assert ctx.state == OrderState.PROTECTED


def test_protected_state_accepts_valid_short_protection():
    ctx = PositionContext("BTC/USDT")
    ctx.side = "SHORT"
    ctx.filled_qty = 1.0
    ctx.entry_price = 100.0
    ctx.stop_loss = 105.0

    assert ctx.transition_to(OrderState.PROTECTED, reason="test valid SHORT") is True
    assert ctx.state == OrderState.PROTECTED
