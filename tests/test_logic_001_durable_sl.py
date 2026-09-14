import json
from pathlib import Path

from execution.execution_result import ExecutionIntentJournal


def test_logic_001_no_synthetic_recovery_stop():
    recon = Path("core/reconciliation_engine.py").read_text(encoding="utf-8")
    assert "sl_dist = 0.02" not in recon
    assert "No synthetic 2% stop" in recon
    assert "NO_DURABLE_STRATEGY_SL" in recon
    assert "ORPHAN_NO_DURABLE_STRATEGY_SL" in recon
    assert "UNPROTECTED_NO_DURABLE_STRATEGY_SL" in recon


def test_live_entry_without_durable_protection_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING", "False")
    journal = ExecutionIntentJournal(tmp_path / "intents.jsonl")

    try:
        journal.create(
            intent_id="test-live-missing-sl",
            client_order_id="PS_TEST_MISSING_SL",
            venue="BINANCE",
            account_mode="spot",
            symbol="BTC/USDT",
            side="buy",
            requested_qty=0.01,
            order_role="ENTRY",
            price=100000.0,
            protection={},
        )
    except ValueError as exc:
        assert "requires durable protection" in str(exc)
    else:
        raise AssertionError("live ENTRY without durable protection must be rejected")

    assert not (tmp_path / "intents.jsonl").exists()


def test_durable_entry_protection_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING", "False")
    path = tmp_path / "intents.jsonl"
    journal = ExecutionIntentJournal(path)
    journal.create(
        intent_id="test-live-protected-entry",
        client_order_id="PS_TEST_PROTECTED",
        venue="BINANCE",
        account_mode="spot",
        symbol="BTC/USDT",
        side="buy",
        requested_qty=0.01,
        order_role="ENTRY",
        price=100000.0,
        protection={"stop_loss": 99500.0, "signal_entry_price": 100000.0, "position_side": "LONG"},
    )

    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["protection"]["stop_loss"] == 99500.0
    assert record["protection"]["position_side"] == "LONG"
