from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import ExecutionStep
from freqtrade.strategy.trx_phase1.stablecoin_policy import (
    POLL_MS,
    minimum_apr,
    rate_confirmation,
    stage_activation,
)


def terms(rate="12"):
    return {
        "product_id": "usdc-flex",
        "singleMaxAmount": "10000",
        "remainingAmount": "9000",
        "tiers": [
            {"minStepVal": "0", "maxStepVal": "1000", "currentApy": rate},
            {"minStepVal": "1000", "maxStepVal": "10000", "currentApy": "3.5"},
        ],
    }


def test_confirmation_survives_restart_but_resets_on_new_rate_or_failed_read():
    first = rate_confirmation({}, 0, {"USDC": terms()})
    assert first["status"] == "awaiting_second_observation"
    assert rate_confirmation(first, POLL_MS - 1, {"USDC": terms()}) == first
    second = rate_confirmation(deepcopy(first), POLL_MS, {"USDC": terms()})
    assert second["status"] == "confirmed"
    changed = rate_confirmation(second, 2 * POLL_MS, {"USDC": terms("11.67")})
    assert changed["status"] == "awaiting_second_observation"
    failure = rate_confirmation(changed, 3 * POLL_MS, None)
    assert failure["status"] == "rates_unavailable"
    next_success = rate_confirmation(failure, 4 * POLL_MS, {"USDC": terms("11.67")})
    assert next_success["status"] == "awaiting_second_observation"


def test_outside_holding_counts_toward_marginal_placement_tier():
    assert minimum_apr(terms(), money(900), money(100)) == money(".12")
    assert minimum_apr(terms(), money(900), money(101)) == money(".035")
    with pytest.raises(ExecutionError, match="stablecoin_subscription_tier_changed"):
        minimum_apr(terms(), money(1000), money(9001))


def test_legacy_step_keeps_its_persisted_shape():
    legacy = ExecutionStep(kind="subscribe", coin="TRX", amount=money(1), product_id="trx-flex")
    assert "minimum_marginal_apr" not in legacy.saved()


def test_activation_keeps_external_usdc_outside_and_requires_exact_native_evidence():
    old = {
        "account_id": "old",
        "created_at": 10,
        "book": Ledger(1000).state(),
        "income_accounting": {"status": "active", "checkpoints": [{"timestamp": 10}]},
        "external_usdc": {
            "spot": "10",
            "savings": "20",
            "outside_strategy": True,
            "product_id": "usdc-flex",
        },
        "holdings": {},
        "unassigned_offsets": {},
    }
    policy = {
        "version": "old-stablecoin-v1",
        "account_id": "old",
        "ledger_created_at": 10,
        "starts_at": 20,
        "products": [
            {"coin": "USDT", "product_id": "usdt-flex", "redemption_decimals": 8},
            {"coin": "USDC", "product_id": "usdc-flex"},
        ],
    }
    remote = {"held": {"coin": "USDC", "product_id": "usdc-flex", "amount": "20"}}
    expected = deepcopy(old)
    activated = stage_activation(old, policy, remote, {"USDC": "10"})
    assert old == expected
    assert activated["book"] == old["book"]
    assert [row["redemption_decimals"] for row in activated["stablecoin_policy"]["products"]] == [
        2,
        2,
    ]
    assert all(
        "redemption_minimum" not in row for row in activated["stablecoin_policy"]["products"]
    )
    assert activated["holdings"]["held"]["principal"] == "0"
    assert activated["holdings"]["held"]["unassigned_amount"] == "20"
    with pytest.raises(ExecutionError, match="stablecoin_external_spot_unreconciled"):
        stage_activation(old, policy, remote, {"USDC": "10.1"})
    with pytest.raises(ExecutionError, match="stablecoin_policy_binding_invalid"):
        stage_activation(old, {**policy, "account_id": "new"}, remote, {"USDC": "10"})
