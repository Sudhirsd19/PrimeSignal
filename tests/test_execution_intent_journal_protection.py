import json

from execution.execution_result import ExecutionIntentJournal


def test_journal_persists_and_merges_protection_metadata(tmp_path):
    journal = ExecutionIntentJournal(tmp_path / "execution_intents.jsonl")
    protection = {
        "stop_loss": 61234.5,
        "risk_distance": 765.5,
        "position_side": "LONG",
    }

    journal.create(
        intent_id="intent-1",
        client_order_id="PS_INTENT1",
        venue="BINANCE",
        account_mode="futures",
        symbol="BTC/USDT",
        side="buy",
        requested_qty=0.01,
        order_role="ENTRY",
        price=62000.0,
        protection=protection,
    )

    records = journal.latest()
    assert records["intent-1"]["protection"] == protection

    # A later result event must not erase original protection metadata.
    journal.append({
        "event": "INTENT_RESULT",
        "intent_id": "intent-1",
        "state": "FILLED",
        "exchange_order_id": "12345",
        "filled_qty": 0.01,
    })
    merged = journal.latest()["intent-1"]
    assert merged["protection"] == protection
    assert merged["state"] == "FILLED"


def test_missing_protection_is_explicitly_distinguishable(tmp_path):
    journal = ExecutionIntentJournal(tmp_path / "execution_intents.jsonl")
    journal.create(
        intent_id="legacy-1",
        client_order_id="PS_LEGACY1",
        venue="BINANCE",
        account_mode="spot",
        symbol="ETH/USDT",
        side="buy",
        requested_qty=0.1,
        order_role="ENTRY",
        price=3000.0,
    )

    record = journal.latest()["legacy-1"]
    assert record.get("protection") == {}
    assert "stop_loss" not in record.get("protection", {})
