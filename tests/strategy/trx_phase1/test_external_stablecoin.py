from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.external_stablecoin import SOURCE
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import ExecutionStep, LiveExecution
from freqtrade.strategy.trx_phase1.stablecoin_policy import stage_activation
from tests.strategy.trx_phase1.test_income_allocation import approve
from tests.strategy.trx_phase1.test_live_execution import (
    NOW,
    RULES,
    ExchangeFixture,
    trade,
)


@pytest.fixture
def external_runtime(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    wallet = broker.state()
    wallet["wallet"].update(USDGO="10", USDC="3")
    broker.put(wallet)
    runtime = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    approve(runtime)
    runtime.import_interest(NOW + 2000)
    runtime.state["external_usdc"] = {
        "spot": "3",
        "savings": "0",
        "outside_strategy": True,
        "product_id": "USDC-flex",
    }
    policy = {
        "version": "old-stablecoin-v1",
        "account_id": runtime.account_id,
        "ledger_created_at": NOW,
        "starts_at": NOW + 3000,
        "external_rebalance": True,
        "products": [{"coin": coin, "product_id": f"{coin}-flex"} for coin in ("USDC", "USDT")],
    }
    runtime.state = stage_activation(runtime.state, policy, {}, {"USDC": "3"})
    runtime.save()
    return runtime


def test_external_flow_cannot_request_trx_or_mix_owners(external_runtime):
    with pytest.raises(ValueError, match="may only move stablecoins"):
        ExecutionStep.model_validate({**trade("TRXUSDT"), "source": SOURCE})
    with pytest.raises(ExecutionError, match="mixed_stablecoin_flow_ownership"):
        external_runtime.queue(
            [{**trade(side="sell", amount="2"), "source": SOURCE}, trade(amount="2")],
            now=NOW + 4000,
            expires_at=NOW + 120000,
            label="mixed fixture",
        )
    assert not external_runtime.broker.state()["writes"]


def test_external_fill_waits_for_wallet_and_survives_restart(external_runtime, tmp_path):
    runtime = external_runtime
    broker = runtime.broker
    original = runtime.status()
    runtime.queue(
        [{**trade(side="sell", amount="2"), "source": SOURCE}],
        now=NOW + 4000,
        expires_at=NOW + 120000,
        label="external fixture",
    )
    runtime.tick(NOW + 5000)
    runtime.tick(NOW + 6000)
    assert runtime.state["operation"]["status"] == "accepted"
    broker.lagged_balances = {
        coin: row["available"]
        for coin, row in runtime.state["operation"]["before_balances"].items()
    }
    runtime.tick(NOW + 7000)
    delayed = runtime.status()
    assert delayed["flow"]["index"] == 0
    assert delayed["unassigned_offsets"] == original["unassigned_offsets"]
    assert delayed["receipts"] == original["receipts"]

    broker.lagged_balances = None
    recovered = LiveExecution(tmp_path, broker, runtime.account_id)
    recovered.state["stablecoin_policy"]["external_rebalance"] = False
    recovered.save()
    recovered.tick(NOW + 8000)
    final = recovered.status()
    assert final["flow"]["status"] == "completed"
    assert final["book"] == original["book"]
    assert final["anchor_version"] == original["anchor_version"]
    assert money(final["unassigned_offsets"]["USDT"]) == money("2.002")
    assert money(final["unassigned_offsets"]["USDGO"]) == money(8)
    assert final["flow"]["outputs"][0]["attribution"] == SOURCE
    recovered.tick(NOW + 9000)
    assert recovered.status() == final
    assert len(broker.state()["writes"]) == 1


def test_external_earn_waits_for_bill_without_spending_strategy_cash(external_runtime):
    runtime = external_runtime
    broker = runtime.broker
    native_submit = broker.submit

    def submit(operation):
        result = native_submit(operation)
        provider = broker.state()
        provider["bills"][-1]["businessType"] = "SINGLE_USER_OUT"
        broker.put(provider)
        return result

    broker.submit = submit
    runtime.queue(
        [
            {
                "kind": "subscribe",
                "coin": "USDC",
                "product_id": "USDC-flex",
                "amount": "2",
                "source": SOURCE,
            }
        ],
        now=NOW + 4000,
        expires_at=NOW + 120000,
        label="external Earn fixture",
    )
    runtime.tick(NOW + 5000)
    runtime.tick(NOW + 6000)
    before = deepcopy(runtime.state)
    broker.hide_bills = True
    runtime.tick(NOW + 7000)
    assert runtime.state["flow"]["index"] == 0
    assert runtime.state["unassigned_offsets"] == before["unassigned_offsets"]
    broker.hide_bills = False
    runtime.tick(NOW + 8000)
    assert runtime.state["flow"]["status"] == "completed"
    assert runtime.state["book"] == before["book"]
    assert money(runtime.state["unassigned_offsets"]["USDC"]) == money(1)
    assert money(runtime.state["external_usdc"]["savings"]) == money(2)
    assert len(broker.state()["writes"]) == 1


def test_rejected_rebalance_redemption_stops_the_flow(external_runtime, mocker):
    runtime = external_runtime
    broker = runtime.broker
    provider = broker.state()
    provider["holdings"]["external-usdc"] = {
        "order_id": "external-usdc",
        "coin": "USDC",
        "product_id": "USDC-flex",
        "amount": "2",
    }
    broker.put(provider)
    runtime.state["holdings"]["external-usdc"] = {
        "coin": "USDC",
        "product_id": "USDC-flex",
        "principal": "0",
        "interest": "0",
        "unassigned_amount": "2",
        "subscription_ids": [],
        "lots": {},
    }
    next(r for r in runtime.state["stablecoin_policy"]["products"] if r["coin"] == "USDC").update(
        redemption_minimum="0.1", redemption_decimals=2, redemption_evidence="fixture"
    )
    runtime.save()
    original = runtime.status()
    runtime.queue(
        [
            {
                "kind": "redeem",
                "coin": "USDC",
                "product_id": "USDC-flex",
                "holding_id": "external-usdc",
                "amount": "1.00",
                "source": SOURCE,
            },
            {**trade(side="sell", amount="2"), "source": SOURCE},
        ],
        now=NOW + 4000,
        expires_at=NOW + 120000,
        label="C1 live · stablecoin_rebalance",
    )
    broker.fault = "reject"
    submit = mocker.spy(broker, "submit")
    for offset in (5000, 6000, 7000, 8000):
        runtime.tick(NOW + offset)
    state = runtime.status()
    assert state["flow"]["status"] == "failed" and state["flow"]["index"] == 0
    assert state["flow"]["rebalance_redemption_failure"]["code"] == "exchange_write_40014"
    assert state["operation"] is None
    assert state["book"] == original["book"]
    assert state["unassigned_offsets"] == original["unassigned_offsets"]
    submit.assert_called_once()
