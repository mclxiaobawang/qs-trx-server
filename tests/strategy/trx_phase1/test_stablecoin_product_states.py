from copy import deepcopy
from types import SimpleNamespace

import pytest

from freqtrade.strategy.trx_phase1 import stablecoin_policy as policy
from freqtrade.strategy.trx_phase1.external_income_reinvestment import (
    ReinvestmentDeferred,
    available_sources,
)
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest
from tests.strategy.trx_phase1.test_external_income_reinvestment import (
    example as example,  # noqa: PLC0414
)


def terms(coin, status="in_progress", remaining="10000"):
    return {
        "product_id": coin,
        "status": status,
        "singleMaxAmount": "10000",
        "remainingAmount": remaining,
        "singleMinAmount": "0.1",
        "subscribePrecision": "8",
        "apyList": [
            {
                "minStepVal": "0",
                "maxStepVal": "1000",
                "currentApy": "11.57" if coin == "USDT" else "11" if coin == "USDC" else "9",
            },
            {
                "minStepVal": "1000",
                "maxStepVal": "10000",
                "currentApy": "3.57" if coin == "USDT" else "3.5" if coin == "USDC" else "8",
            },
        ],
    }


@pytest.mark.parametrize("closed", ["off_line", "sold_out", "paused", "completed", "not_started"])
def test_reopen_needs_two_new_reads_without_losing_other_products(closed):
    data = {c: terms(c, closed if c == "USDC" else "in_progress") for c in policy.COINS}
    one = policy.rate_confirmation({}, 0, data)
    two = policy.rate_confirmation(one, policy.POLL_MS, data)
    assert two["status"] == "confirmed"
    assert two["products"]["USDT"]["subscription_eligible"]
    assert not two["products"]["USDC"]["subscription_eligible"]
    data["USDC"]["status"] = "in_progress"
    reopened = policy.rate_confirmation(deepcopy(two), 2 * policy.POLL_MS, data)
    assert reopened["status"] == "awaiting_second_observation"
    assert reopened["products"]["USDT"]["subscription_eligible"]
    assert not reopened["products"]["USDC"]["subscription_eligible"]
    assert policy.rate_confirmation(reopened, 2 * policy.POLL_MS + 1, data) == reopened
    confirmed = policy.rate_confirmation(reopened, 3 * policy.POLL_MS, data)
    assert confirmed["products"]["USDC"]["subscription_eligible"]
    assert (
        policy.rate_confirmation(confirmed, 6 * policy.POLL_MS, data)["status"]
        == "awaiting_second_observation"
    )


def test_old_schema_cannot_count_as_first_product_confirmation():
    data = {c: terms(c) for c in policy.COINS}
    new = policy.rate_confirmation(
        {"checked_at": 0, "status": "confirmed", "fingerprint": "old"}, policy.POLL_MS, data
    )
    assert all(v["status"] == "awaiting_second_observation" for v in new["products"].values())


def test_failed_product_keeps_good_terms_and_records_request_stage():
    def read(coin, pid):
        if coin == "USDC":
            exc = ExecutionError("exchange_read_40014")
            exc.product_request_stage = "subscription_info"
            exc.read_diagnostic = {"code": "40014", "message": "provider failure"}
            raise exc
        return terms(coin)

    e = SimpleNamespace(
        stablecoin_policy={
            "starts_at": 0,
            "products": [{"coin": c, "product_id": c} for c in ["USDT", "USDC"]],
        },
        state={"products": [{"coin": "USDGO", "product_id": "USDGO"}]},
        broker=SimpleNamespace(product=read),
    )
    state = {}
    before = deepcopy(e.state)
    policy.poll(e, state, 0)
    value = state["stablecoin_rates"]
    assert set(value["terms"]) == {"USDT", "USDGO"} and value["status"] == "rates_unavailable"
    error = value["products"]["USDC"]["error"]
    assert error["stage"] == "subscription_info" and error["message"] == "exchange_read_40014"
    assert "provider failure" in error["response"] and "ExecutionError" in error["traceback"]
    assert e.state == before


def test_sold_out_zero_room_is_readable_but_empty_ladder_is_not():
    assert policy.tiers(terms("USDC", "sold_out", "0"), money(0)) == []
    bad = terms("USDC")
    bad["apyList"] = []
    with pytest.raises(ExecutionError):
        policy.tiers(bad, money(0))


def test_offline_product_keeps_existing_yield_without_new_capacity():
    b = Ledger(1905)
    b.cash = money(1000)
    b.locked["USDT"] = money(1000)
    b.usdc = money(905)
    b.usdc_locked = money(900)
    b.usdc_mark = money(1)
    positions = {
        c: {
            "coin": c,
            "product_id": c,
            "principal": str(q),
            "interest": "0",
            "unassigned_amount": "0",
        }
        for c, q in [("USDT", 1000), ("USDC", 900)]
    }
    remote = {
        c: {"coin": c, "product_id": c, "amount": str(q)}
        for c, q in [("USDT", 1000), ("USDC", 900)]
    }
    funding = SimpleNamespace(
        book=b,
        execution={"holdings": positions},
        policy={},
        rules={c: SimpleNamespace(product_id=c) for c in policy.COINS},
        markets={c + "USDT": {"bid": "1"} for c in ["USDC", "USDGO"]},
    )
    data = {c: terms(c, "off_line" if c == "USDC" else "in_progress") for c in policy.COINS}
    result = policy.allocation(funding, data, remote)
    assert result[5] == {"USDT": money(1000), "USDC": money(900), "USDGO": money(5)}
    assert result[3]["USDC"][-1][1] == money(900)


def test_external_reserved_tail_is_not_redeemable():
    reserved = "1.234567891234567891"
    position = {
        "coin": "USDGO",
        "product_id": "USDGO",
        "principal": "0",
        "interest": "0",
        "unassigned_amount": "11.358024671234567891",
        "reserved_amount": reserved,
    }
    funding = SimpleNamespace(
        execution={
            "holdings": {"h": position},
            "unassigned_offsets": dict.fromkeys(policy.COINS, "0"),
        },
        rules={c: SimpleNamespace(product_id=c) for c in policy.COINS},
    )
    original = deepcopy(position)
    _, owned = policy.outside_balances(
        funding,
        {"h": {"coin": "USDGO", "product_id": "USDGO", "amount": position["unassigned_amount"]}},
    )
    assert owned["USDGO"] == money("10.123456780000000000") and position == original


def test_usdc_location_receipt_cannot_consume_usdgo_sources(example):
    runtime, p, _, _, now = example
    expected = available_sources(runtime.state, p, now)
    rec = {
        "classification": "verified_savings_location_only",
        "coin": "USDC",
        "external_amount": "14",
        "external_interest_sources": {},
    }
    runtime.state.setdefault("verified_savings_location_movements", {})["usdc-manual"] = rec
    runtime.state["receipts"]["observed-savings-subscription:usdc-manual"] = digest(rec)
    assert available_sources(runtime.state, p, now) == expected
    rec["external_interest_sources"] = {"forged": "1"}
    runtime.state["receipts"]["observed-savings-subscription:usdc-manual"] = digest(rec)
    with pytest.raises(ReinvestmentDeferred):
        available_sources(runtime.state, p, now)


@pytest.mark.parametrize("stage", ["product_catalog", "subscription_info"])
def test_broker_product_error_names_actual_failed_endpoint(stage):
    from freqtrade.strategy.trx_phase1.live_broker import ClassicBroker

    broker = object.__new__(ClassicBroker)

    def get(path, *args):
        if path.endswith("product") and stage == "subscription_info":
            return [
                {
                    "coin": "USDC",
                    "productId": "USDC",
                    "periodType": "flexible",
                    "status": "in_progress",
                }
            ]
        raise ExecutionError("exchange_read_40014")

    broker.get = get
    with pytest.raises(ExecutionError) as found:
        broker.product("USDC", "USDC")
    assert found.value.product_request_stage == stage


def test_product_diagnostics_redact_credentials_without_frame_locals():
    from pydantic import SecretStr

    settings = SimpleNamespace(
        api_key=SecretStr("localkey"),
        secret_key=SecretStr("localsecret"),
        passphrase=SecretStr("localpass"),
    )
    exc = ExecutionError(
        "localkey localsecret localpass https://example.test/path?signature=abcdef password=private"
    )
    result = policy.product_error(
        exc, SimpleNamespace(transport=SimpleNamespace(settings=settings)), "USDC", "id", 1
    )
    import json

    rendered = json.dumps(result)
    assert not any(
        v in rendered for v in ["localkey", "localsecret", "localpass", "abcdef", "private"]
    )


def test_preflight_closure_revokes_confirmed_product_without_early_poll():
    data = {c: terms(c) for c in policy.COINS}
    one = policy.rate_confirmation({}, 0, data)
    state = {"stablecoin_rates": policy.rate_confirmation(one, policy.POLL_MS, data)}
    stamp = policy.POLL_MS + 5 * 60000
    policy.invalidate_subscription(state, "USDC", "USDC", stamp, "subscription_terms_not_satisfied")
    rates = state["stablecoin_rates"]
    assert rates["checked_at"] == policy.POLL_MS
    assert rates["products"]["USDT"]["subscription_eligible"]
    assert not rates["products"]["USDC"]["subscription_eligible"]
    assert rates["products"]["USDC"]["fingerprint"] is None
    unchanged = deepcopy(rates)
    assert policy.rate_confirmation(rates, stamp, data) == unchanged
    reopened = policy.rate_confirmation(rates, 2 * policy.POLL_MS, data)
    assert not reopened["products"]["USDC"]["subscription_eligible"]
    assert policy.rate_confirmation(reopened, 3 * policy.POLL_MS, data)["products"]["USDC"][
        "subscription_eligible"
    ]
    assert state["stablecoin_rates"] == unchanged


def test_preflight_invalidation_cannot_change_other_product_binding():
    data = {c: terms(c) for c in policy.COINS}
    state = {"stablecoin_rates": policy.rate_confirmation({}, 0, data)}
    before = deepcopy(state)
    policy.invalidate_subscription(
        state, "USDC", "wrong-product", 1, "subscription_terms_not_satisfied"
    )
    assert state == before


def test_native_strategy_preflight_failure_revokes_cached_eligibility():
    from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy

    data = {c: terms(c) for c in policy.COINS}
    rates = policy.rate_confirmation(policy.rate_confirmation({}, 0, data), policy.POLL_MS, data)
    flow = {
        "id": "natural",
        "status": "running",
        "index": 0,
        "steps": [{"kind": "subscribe", "coin": "USDC", "product_id": "USDC"}],
    }
    saved = {
        "flow": flow,
        "operation": {"status": "prepared"},
        "events": [
            {"kind": "read_or_reconciliation_blocked", "code": "subscription_terms_not_satisfied"}
        ],
    }
    stops = []
    runtime = object.__new__(LiveStrategy)
    runtime.state = {
        "intent": {"kind": "stablecoin_rebalance", "flow_id": "natural"},
        "desired": "running",
        "phase": "executing",
        "stablecoin_rates": rates,
    }
    runtime.execution = SimpleNamespace(
        state=saved,
        status=lambda: deepcopy(saved),
        tick=lambda now: None,
        reload=lambda: None,
        request_stop=lambda now: stops.append(now),
    )
    runtime.dispatch = lambda now: None
    runtime.persist = lambda: None
    assert runtime.resolve_execution(policy.POLL_MS + 300000, prepare_rebalance=True)
    assert (
        stops and not runtime.state["stablecoin_rates"]["products"]["USDC"]["subscription_eligible"]
    )
    assert runtime.state["stablecoin_rates"]["checked_at"] == policy.POLL_MS
