from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from freqtrade.strategy.trx_phase1 import shadow_market
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_strategy import error_diagnostic
from freqtrade.strategy.trx_phase1.shadow_market import PublicMarketDataError, PublicShadowMarket
from tests.strategy.trx_phase1.test_live_strategy import live  # noqa: F401


NOW = 1793861045402


def public_market(monkeypatch, response):
    monkeypatch.setattr(shadow_market.time, "time", lambda: NOW / 1000)
    market = object.__new__(PublicShadowMarket)
    market.client = SimpleNamespace(
        fetch_time=lambda: NOW,
        publicSpotGetV2SpotMarketOrderbook=lambda params: response,
    )
    return market


def book(**changes):
    return {
        "code": "00000",
        "data": {"ts": str(NOW), "asks": [[".31", "10"]], "bids": [[".30", "10"]], **changes},
    }


@pytest.mark.parametrize(
    "response,code",
    [
        ([], "public_orderbook_response_invalid"),
        ({"code": "00000", "data": None}, "public_orderbook_response_invalid"),
        (book(asks=[["bad", "10"]]), "public_orderbook_response_invalid"),
        (book(asks=[[".31"]]), "public_orderbook_response_invalid"),
        (book(asks=[]), "invalid_orderbook"),
        (book(asks=[["nan", "10"]]), "invalid_orderbook"),
        (book(asks=[[".29", "10"]]), "crossed_orderbook"),
        (book(asks=[[".32", "10"], [".31", "10"]]), "unsorted_orderbook"),
        (book(ts=str(NOW - 10001)), "stale_execution_quote"),
    ],
)
def test_unusable_public_response_is_discarded_and_next_fresh_read_can_pass(
    monkeypatch, response, code
):
    market = public_market(monkeypatch, response)
    with pytest.raises(PublicMarketDataError, match=code) as raised:
        market.snapshot()
    assert raised.value.__cause__ is not None
    market.client.publicSpotGetV2SpotMarketOrderbook = lambda params: book()
    stamp, quote = market.snapshot()
    assert stamp == NOW and quote["bids"] == [[0.30, 10.0]]


def test_public_quote_failure_cannot_reach_income_or_writes_and_can_recover(live, monkeypatch):  # noqa: F811
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    before = deepcopy(runtime.execution.status()["book"])
    snapshot = market.snapshot

    def malformed():
        try:
            float("bad-price")
        except ValueError as exc:
            raise PublicMarketDataError(
                "public_orderbook_response_invalid", "public_orderbook_parse", {}
            ) from exc

    monkeypatch.setattr(market, "snapshot", malformed)
    with pytest.raises(PublicMarketDataError) as failure:
        runtime.tick(market.now)
    runtime.poll_error(failure.value, market.now)
    assert runtime.state["desired"] == "running" and runtime.state["phase"] == "recovering"
    diagnostic = runtime.state["last_error_diagnostic"]
    assert diagnostic["stage"] == "public_snapshot"
    assert "bad-price" in diagnostic["traceback"]
    assert broker.state()["writes"] == []
    assert runtime.execution.status()["book"] == before
    monkeypatch.setattr(market, "snapshot", snapshot)
    runtime.tick(market.now)
    assert runtime.state["error"] is None and runtime.engine.pending is None
    assert broker.state()["writes"] == []


@pytest.mark.parametrize(
    "error",
    [RuntimeError("unexpected-program-error"), ExecutionError("unrecognized_usdt_movement")],
)
def test_unknown_and_fund_errors_keep_operator_review(live, error):  # noqa: F811
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    runtime.poll_stage = "income_reconciliation"
    runtime.poll_error(error, market.now)
    assert runtime.state["desired"] == "paused" and runtime.state["phase"] == "blocked"
    assert runtime.state["last_error_diagnostic"]["message"] == str(error)
    assert broker.state()["writes"] == []


def test_traceback_redacts_credentials_query_values_and_authorization():
    broker = SimpleNamespace(
        transport=SimpleNamespace(
            settings=SimpleNamespace(
                api_key=SecretStr("private-key-value"),
                secret_key=SecretStr("private-secret-value"),
                passphrase=SecretStr("private-passphrase-value"),
            )
        )
    )
    try:
        raise RuntimeError(
            "private-key-value private-secret-value private-passphrase-value "
            "Authorization: Bearer private-token https://example.test/api?signature=private-query"
        )
    except RuntimeError as exc:
        diagnostic = error_diagnostic(exc, "income_reconciliation", None, broker)
    for secret in (
        "private-key-value",
        "private-secret-value",
        "private-passphrase-value",
        "private-token",
        "private-query",
    ):
        assert secret not in diagnostic["message"] and secret not in diagnostic["traceback"]
    assert (
        diagnostic["exception_type"] == "RuntimeError"
        and "test_traceback" in diagnostic["traceback"]
    )
