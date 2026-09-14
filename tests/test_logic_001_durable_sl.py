from pathlib import Path


def test_logic_001_no_synthetic_recovery_stop():
    recon = Path("core/reconciliation_engine.py").read_text(encoding="utf-8")
    assert "sl_dist = 0.02" not in recon
    assert "No synthetic 2% stop" in recon
    assert "NO_DURABLE_STRATEGY_SL" in recon
    assert "ORPHAN_NO_DURABLE_STRATEGY_SL" in recon
    assert "UNPROTECTED_NO_DURABLE_STRATEGY_SL" in recon
