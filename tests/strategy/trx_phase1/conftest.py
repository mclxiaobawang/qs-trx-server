import pytest

from freqtrade.strategy.trx_phase1 import request_limits


@pytest.fixture(autouse=True)
def isolated_request_budget(tmp_path, monkeypatch):
    """Fixture transports must never consume the local real account's budget."""
    monkeypatch.setattr(request_limits, "DEFAULT_PATH", tmp_path / "request-limits.sqlite")
