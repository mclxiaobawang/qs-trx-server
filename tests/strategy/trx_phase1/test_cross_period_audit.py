import sqlite3

import pytest

from freqtrade.strategy.trx_phase1.cross_period_audit import direction_changes, original_window


def test_inherited_window_uses_boundary_equity_and_retained_states(tmp_path):
    path = tmp_path / "account.sqlite"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE samples (timestamp,equity,exposure,principal_trx,remaining_cost,"
            "interest_trx,trx_interest_value,realized_profit,principal_unrealized,anchor)"
        )
        db.executemany(
            "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (0, 2000, 0.6, 100, 1300, 10, 100, 400, -300, 1.5),
                (60000, 1500, 0.7, 100, 1300, 11, 110, 400, -800, 1.5),
                (120000, 2200, 0.6, 90, 1200, 12, 120, 450, -400, 1.6),
            ],
        )
        db.execute("CREATE TABLE risk_samples (timestamp,stale_cost)")
        db.executemany(
            "INSERT INTO risk_samples VALUES (?,?)", [(0, 100), (60000, 200), (120000, 300)]
        )
    result = original_window(path, 0, 120000)
    assert result["minutes"] == 2
    assert result["return_ratio"] == pytest.approx(0.1)
    assert result["max_drawdown"] == pytest.approx(0.25)
    assert result["entry_remaining_cost"] == 1300
    assert result["entry_retained_interest_trx"] == 10
    assert result["entry_anchor"] == 1.5
    assert result["realized_principal_change"] == 50
    assert result["mean_old_principal_cost_30_days"] == 250
    with pytest.raises(ValueError, match="incomplete"):
        original_window(path, 1, 120000)


def test_month_end_reversals_ignore_only_rounding_noise():
    assert direction_changes([0, 5, 1e-8, -3, -2, 4]) == 2
