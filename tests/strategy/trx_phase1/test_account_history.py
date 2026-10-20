import json
from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.account_history import AccountHistory, observation


NOW = 1800000000000


def snapshot():
    return {
        "completed_at": NOW + 1000,
        "balances": [
            {"coin": "USDT", "total": "100"},
            {"coin": "USDGO", "total": "200"},
            {"coin": "TRX", "total": "1000"},
        ],
        "prices": {
            "USDGO": {"bid": "1.002", "timestamp": NOW},
            "TRX": {"bid": "0.3", "timestamp": NOW},
        },
        "savings": {
            "pages_complete": True,
            "rows": [{"coin": "TRX", "quantity": "50", "total_profit": "10"}],
        },
    }


def test_actual_fx_marks_and_separate_savings_not_double_counted():
    source = snapshot()
    original = deepcopy(source)
    point = observation(source, "classic", NOW)
    assert point["spot_equity_usdt"] == "600.400"
    assert point["spot_quantities"]["TRX"] == "1000"
    assert point["savings_quantities"]["TRX"] == "50"
    assert point["valuation_status"] == "marked"
    assert source == original


@pytest.mark.parametrize("missing", ["balance", "quote", "savings", "uta", "failed"])
def test_missing_components_are_not_zero_or_previous_marks(missing):
    source = snapshot()
    if missing == "balance":
        source["balances"].pop()
    elif missing == "quote":
        source["prices"] = None
    elif missing == "savings":
        source["savings"] = None
    point = observation(
        None if missing == "failed" else source, "uta" if missing == "uta" else "classic", NOW
    )
    if missing == "savings":
        assert point["spot_equity_usdt"] == "600.400"
        assert all(v is None for v in point["savings_quantities"].values())
    else:
        assert point["spot_equity_usdt"] is None
        if missing in ("failed", "uta"):
            assert all(v is None for v in point["spot_quantities"].values())


def test_zero_exposure_does_not_need_a_quote_but_missing_coin_is_unknown():
    source = snapshot()
    for row in source["balances"][1:]:
        row["total"] = "0"
    source["prices"] = None
    assert observation(source, "classic", NOW)["spot_equity_usdt"] == "100"
    source["balances"].pop()
    assert observation(source, "classic", NOW)["spot_equity_usdt"] is None


def test_persistent_account_isolation_ordering_gaps_and_no_raw_identity(tmp_path):
    path = tmp_path / "account-observations.sqlite"
    history = AccountHistory(path)
    assert history.read(None, "classic")["points"] == []
    assert not path.exists()
    first = observation(snapshot(), "classic", NOW)
    gap = observation(None, "classic", NOW + 60000)
    history.append("hashed-account-a", "classic", gap)
    history.append("hashed-account-a", "classic", first)
    history.append("hashed-account-a", "classic", first)
    history.append("hashed-account-b", "classic", first)
    history.append("hashed-account-a", "uta", first)
    reloaded = AccountHistory(path).read("hashed-account-a", "classic")
    assert reloaded["points"] == [first, gap]
    assert reloaded["total_points"] == 2 and not reloaded["truncated"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert "hashed-account" not in json.dumps(reloaded)
