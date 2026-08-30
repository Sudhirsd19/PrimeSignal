"""Deterministic adversarial tests for Phase 1 execution semantics.

These tests use only standard-library unittest primitives so they can run in
the repository's bundled virtual environment without pytest.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from execution.coindcx_client import CoinDCXClient
from execution.execution_engine import ExecutionEngine
from execution.execution_result import (
    ExecutionIntentJournal,
    ExecutionResult,
    ExecutionState,
)
from core.order_state_machine import OrderState, PositionContext


class _Response:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status = status
        self._payload = payload if payload is not None else {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _TimeoutResponse:
    async def __aenter__(self):
        raise asyncio.TimeoutError("response lost")

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def post(self, _url, data=None, **_kwargs):
        self.payloads.append(json.loads(data) if isinstance(data, str) else data)
        return self.responses.pop(0)


class _BinanceClient:
    markets = {"BTC/USDT": {"limits": {"amount": {"min": 0.0001}}}}

    @staticmethod
    def amount_to_precision(_symbol, amount):
        return str(amount)

    async def create_market_order(self, *_args):
        raise TimeoutError("lost Binance response")

    async def fetch_open_orders(self, _symbol):
        return [{"id": "BN-1", "clientOrderId": "PS_DUP", "status": "filled", "amount": 1.0, "filled": 1.0}]

    async def fetch_orders(self, _symbol):
        # The same order appearing in both endpoints must be de-duplicated.
        return [{"id": "BN-1", "clientOrderId": "PS_DUP", "status": "filled", "amount": 1.0, "filled": 1.0}]


class _BinancePublic:
    async def close(self):
        return None


class Phase1ExecutionTests(unittest.TestCase):
    def test_coindcx_lost_response_lookup_succeeds_one_post(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                client = CoinDCXClient("k", "s", str(Path(td) / "intents.jsonl"))
                client.initialized = True
                client.markets_info = {"BTCINR": {"precision": 6, "min_quantity": 0.00001}}
                session = _Session([_TimeoutResponse()])
                client._get_session = lambda: _resolved(session)

                async def lookup(*_args, **_kwargs):
                    return ExecutionResult(
                        state=ExecutionState.FILLED,
                        requested_qty=0.5,
                        filled_qty=0.5,
                        exchange_order_id="DCX-1",
                        client_order_id="CID-1",
                        intent_id="INT-1",
                        venue="COINDCX",
                    )

                client._reconcile_ambiguous_order = lookup
                result = await client.place_order("buy", "market", 0.5, symbol="BTC/INR", intent_id="INT-1", client_order_id="CID-1")
                self.assertEqual(result.state, ExecutionState.FILLED)
                self.assertEqual(len(session.payloads), 1)

        asyncio.run(scenario())

    def test_coindcx_lost_response_all_reads_fail_no_second_post(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "intents.jsonl"
                client = CoinDCXClient("k", "s", str(path))
                client.initialized = True
                client.markets_info = {"BTCINR": {"precision": 6, "min_quantity": 0.00001}}
                session = _Session([_TimeoutResponse()])
                client._get_session = lambda: _resolved(session)

                async def lookup(*_args, **_kwargs):
                    return None

                client._reconcile_ambiguous_order = lookup
                result = await client.place_order("buy", "market", 0.5, symbol="BTC/INR", intent_id="INT-2", client_order_id="CID-2")
                self.assertEqual(result.state, ExecutionState.EXECUTION_UNKNOWN)
                self.assertEqual(len(session.payloads), 1)
                self.assertEqual(ExecutionIntentJournal(path).latest()["INT-2"]["symbol"], "BTCINR")

        asyncio.run(scenario())

    def test_coindcx_known_retry_reuses_same_client_id(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                client = CoinDCXClient("k", "s", str(Path(td) / "intents.jsonl"))
                client.initialized = True
                client.markets_info = {"BTCINR": {"precision": 6, "min_quantity": 0.00001}}
                session = _Session([
                    _Response(429, text="rate limit"),
                    _Response(429, text="rate limit"),
                    _Response(200, {"id": "DCX-3", "status": "filled", "total_quantity": 0.5, "filled_quantity": 0.5}),
                ])
                client._get_session = lambda: _resolved(session)

                async def no_sleep(_delay):
                    return None

                import execution.coindcx_client as module
                old_sleep = module.asyncio.sleep
                module.asyncio.sleep = no_sleep
                try:
                    result = await client.place_order("buy", "market", 0.5, symbol="BTC/INR", intent_id="INT-3", client_order_id="CID-3")
                finally:
                    module.asyncio.sleep = old_sleep
                self.assertEqual(result.state, ExecutionState.FILLED)
                self.assertEqual([p["client_order_id"] for p in session.payloads], ["CID-3", "CID-3", "CID-3"])

        asyncio.run(scenario())

    def test_partial_fill_quantity_mapping(self):
        result = ExecutionResult.from_exchange(
            {"id": "P-1", "status": "open", "amount": 1.0, "filled": 0.4, "remaining": 0.6, "price": 100.0},
            requested_qty=1.0,
        )
        self.assertEqual(result.state, ExecutionState.PARTIALLY_FILLED)
        self.assertAlmostEqual(result.filled_qty, 0.4)
        self.assertAlmostEqual(result.remaining_qty, 0.6)

    def test_binance_fill_poll_timeout_is_unknown(self):
        async def scenario():
            engine = object.__new__(ExecutionEngine)

            class Client:
                async def fetch_order(self, *_args):
                    return None

            engine.trade_client = Client()
            result = await engine.wait_for_fill("B-1", "BTC/USDT", timeout=0.0, requested_qty=1.0)
            self.assertEqual(result.state, ExecutionState.EXECUTION_UNKNOWN)
            self.assertFalse(result.is_fill_confirmed)

        asyncio.run(scenario())

    def test_exit_unknown_preserves_position_metadata(self):
        ctx = PositionContext("BTC/USDT")
        ctx.side = "LONG"
        ctx.entry_price = 100.0
        ctx.filled_qty = 1.0
        ctx.requested_qty = 1.0
        ctx.native_sl_order_id = "SL-1"
        ctx.transition_to(OrderState.PROTECTED, reason="setup")
        outcome = ExecutionResult(state=ExecutionState.EXECUTION_UNKNOWN, requested_qty=1.0)
        if not outcome.is_fill_confirmed:
            ctx.transition_to(OrderState.EXIT_UNKNOWN, reason="exit timeout")
        self.assertEqual(ctx.state, OrderState.EXIT_UNKNOWN)
        self.assertEqual(ctx.filled_qty, 1.0)
        self.assertEqual(ctx.entry_price, 100.0)
        self.assertEqual(ctx.native_sl_order_id, "SL-1")

    def test_coindcx_cancel_timeout_is_cancel_unknown(self):
        async def scenario():
            client = CoinDCXClient("k", "s")
            client.initialized = True
            session = _Session([_TimeoutResponse()])
            client._get_session = lambda: _resolved(session)
            result = await client.cancel_order("SL-1")
            self.assertEqual(result.state, ExecutionState.CANCEL_UNKNOWN)

        asyncio.run(scenario())

    def test_binance_duplicate_client_id_resolves_existing_order(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as td:
                engine = object.__new__(ExecutionEngine)
                engine.trade_client = _BinanceClient()
                engine.public_client = _BinancePublic()
                engine.coindcx_client = None
                engine._futures_initialized = False
                engine.intent_journal = ExecutionIntentJournal(Path(td) / "intents.jsonl")
                result = await engine.place_order(
                    "buy", "market", 1.0, symbol="BTC/USDT", is_exit_order=True,
                    intent_id="INT-BN", client_order_id="PS_DUP",
                )
                self.assertEqual(result.state, ExecutionState.FILLED)
                self.assertEqual(result.exchange_order_id, "BN-1")

        asyncio.run(scenario())

    def test_restart_recovers_original_intent_identity(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "intents.jsonl"
            journal = ExecutionIntentJournal(path)
            journal.create(
                intent_id="INT-R", client_order_id="CID-R", venue="COINDCX",
                account_mode="spot", symbol="BTCINR", side="buy",
                requested_qty=1.0, order_role="ENTRY", price=100.0,
            )
            journal.result(ExecutionResult(
                state=ExecutionState.EXECUTION_UNKNOWN,
                requested_qty=1.0, client_order_id="CID-R", intent_id="INT-R", venue="COINDCX",
            ))
            recovered = ExecutionIntentJournal(path).unresolved()
            self.assertEqual(recovered[0]["intent_id"], "INT-R")
            self.assertEqual(recovered[0]["client_order_id"], "CID-R")
            self.assertEqual(recovered[0]["symbol"], "BTCINR")

    def test_repeated_prepare_attempts_keep_one_intent_identity(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "intents.jsonl"
            journal = ExecutionIntentJournal(path)
            kwargs = dict(
                intent_id="INT-RETRY", client_order_id="CID-RETRY", venue="BINANCE",
                account_mode="spot", symbol="BTC/USDT", side="buy",
                requested_qty=1.0, order_role="ENTRY", price=100.0,
            )
            journal.create(**kwargs)
            journal.create(**kwargs)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["client_order_id"], "CID-RETRY")


async def _resolved(value):
    return value


if __name__ == "__main__":
    unittest.main()
