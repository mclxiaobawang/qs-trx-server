"""Synthetic exchange only: no credentials or real financial submissions."""

import json

import pytest

from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.redemption_check import RedemptionCheck, main
from tests.strategy.trx_phase1.test_live_execution import NOW, RULES, ExchangeFixture


@pytest.fixture
def check(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    state = broker.state()
    for coin, amount in [("USDGO", "1"), ("TRX", "4.46553")]:
        state["holdings"][coin + "-holding"] = {
            "order_id": coin + "-holding",
            "product_id": coin + "-flex",
            "coin": coin,
            "amount": amount,
        }
    broker.put(state)
    broker.writes_enabled = False
    rules = [dict(p) for p in RULES]
    for rule in rules:
        rule["redemption_minimum"] = "0.1" if rule["coin"] == "USDGO" else "0.00000001"
        rule["redemption_decimals"] = 2 if rule["coin"] == "USDGO" else 8
    (tmp_path / "live-product-rules.local.json").write_text(json.dumps({"products": rules}))
    clock = [NOW]
    return RedemptionCheck(tmp_path, broker, lambda: clock[0]), broker, clock


def test_preparation_is_read_only_and_uses_holding_identity_and_recorded_minimum(check):
    probe, broker, _ = check
    u, t = probe.prepare("USDGO"), probe.prepare("TRX")
    assert u["request"] == {
        "productId": "USDGO-flex",
        "periodType": "flexible",
        "amount": "0.1",
        "orderId": "USDGO-holding",
    }
    assert t["request"]["amount"] == "0.00000001"
    assert u["financial_write_attempts"] == 0 and broker.state()["writes"] == []
    assert not (probe.root / "live_execution").exists()
    assert probe.path("USDGO").stat().st_mode & 0o777 == 0o600


def test_final_submission_requires_operator_execute_flag(check):
    probe, broker, _ = check
    probe.prepare("USDGO")
    with pytest.raises(ExecutionError, match="operator_execute_flag_required"):
        probe.submit("USDGO")
    with pytest.raises(ExecutionError, match="operator_execute_flag_required"):
        probe.submit("USDGO", execute=True)
    assert broker.state()["writes"] == []


def test_synthetic_receipt_requires_result_bill_holding_and_spendable_balance(check):
    probe, broker, _ = check
    probe.prepare("USDGO")
    broker.writes_enabled = True
    accepted = probe.submit("USDGO", execute=True)
    assert accepted["status"] == "accepted_unverified"
    broker.writes_enabled = False
    broker.hide_bills = True
    pending = probe.verify("USDGO")
    assert pending["status"] == "pending_or_conflicting_evidence"
    broker.hide_bills = False
    verified = probe.verify("USDGO")
    assert verified["status"] == "verified" and all(verified["checks"].values())
    assert verified["arrival_status"] == "verified"
    assert verified["history_link"]["status"] == "verified"
    assert len(broker.state()["writes"]) == 1


def test_independent_history_id_is_auxiliary_after_primary_arrival_is_verified(check):
    probe, broker, clock = check
    probe.prepare("USDGO")
    probe.prepare("TRX")
    broker.writes_enabled = True
    accepted = probe.submit("USDGO", execute=True)
    state = broker.state()
    state["records"][0]["orderId"] = "independent-history-id"
    broker.put(state)
    value = probe.verify("USDGO")
    assert value["arrival_status"] == "verified"
    assert value["checks"]["arrival_bill"] is True
    assert all(value["checks"].values())
    assert value["status"] == "verified"
    assert value["verification_version"] == 2
    assert value["exchange_id"] == accepted["exchange_id"]
    link = value["history_link"]
    assert link["status"] == "record_id_unlinked"
    assert link["candidate_records"][0]["history_record_id"] == "independent-history-id"
    assert link["candidate_records"][0]["direct_order_id_match"] is False
    clock[0] += 61000
    assert probe.submit("TRX", execute=True)["status"] == "accepted_unverified"
    assert probe.verify("USDGO")["status"] == "verified"
    assert len(broker.state()["writes"]) == 2
    assert broker.state()["wallet"]["USDGO"] == "0.1"


@pytest.mark.parametrize("history", ["missing", "read_failure"])
def test_auxiliary_history_unavailable_does_not_block_preparation_or_confirmed_arrival(
    check, mocker, history
):
    probe, broker, _ = check
    if history == "missing":
        broker.hide_records = True
    else:
        mocker.patch.object(
            broker, "savings_records", side_effect=ExecutionError("exchange_response_unavailable")
        )
    prepared = probe.prepare("USDGO")
    assert prepared["status"] == "ready_for_operator_submission"
    broker.writes_enabled = True
    probe.submit("USDGO", execute=True)
    value = probe.verify("USDGO")
    assert value["status"] == "verified" and all(value["checks"].values())
    assert value["history_link"]["status"] == (
        "record_not_observed" if history == "missing" else "unavailable"
    )
    assert len(broker.state()["writes"]) == 1


@pytest.mark.parametrize("missing", ["result", "bill_identity", "holding_delta", "wallet_delta"])
def test_history_amount_and_time_never_replace_missing_order_or_arrival_proof(check, missing):
    probe, broker, _ = check
    probe.prepare("USDGO")
    broker.writes_enabled = True
    probe.submit("USDGO", execute=True)
    state = broker.state()
    state["records"][0]["orderId"] = "independent-history-id"
    if missing == "result":
        state["savings"].clear()
    elif missing == "bill_identity":
        state["bills"][0]["bizOrderId"] = "unrelated-order"
    elif missing == "holding_delta":
        state["holdings"]["USDGO-holding"]["amount"] = "0.8"
    else:
        state["wallet"]["USDGO"] = "0.2"
    broker.put(state)
    value = probe.verify("USDGO")
    assert value["arrival_status"] != "verified"
    assert value["status"] != "verified"
    assert value["history_link"]["status"] == "record_id_unlinked"


@pytest.mark.parametrize(
    "fault", ["timeout_before_accept", "timeout_after_accept", "crash_after_accept"]
)
def test_restart_never_repeats_an_ambiguous_request(check, fault):
    probe, broker, clock = check
    probe.prepare("USDGO")
    broker.writes_enabled = True
    broker.fault = fault
    if fault == "crash_after_accept":
        with pytest.raises(KeyboardInterrupt):
            probe.submit("USDGO", execute=True)
    else:
        assert probe.submit("USDGO", execute=True)["status"] == "outcome_unknown_requires_review"
    restarted = RedemptionCheck(probe.root, broker, lambda: clock[0])
    with pytest.raises(ExecutionError, match="request_already_attempted_or_not_prepared"):
        restarted.submit("USDGO", execute=True)
    assert len(broker.state()["writes"]) == int(fault != "timeout_before_accept")
    assert restarted.verify("USDGO")["exchange_id"] is None
    if fault != "timeout_before_accept":
        assert restarted.verify("USDGO", "exchange-1")["status"] == "verified"


def test_cooldown_is_shared_across_coins_and_both_minima_reconcile(check):
    probe, broker, clock = check
    probe.prepare("USDGO")
    probe.prepare("TRX")
    broker.writes_enabled = True
    probe.submit("USDGO", execute=True)
    with pytest.raises(ExecutionError, match="cooldown_61_seconds"):
        probe.submit("TRX", execute=True)
    clock[0] += 61000
    with pytest.raises(ExecutionError, match="verify_previous_redemption_before_next_coin"):
        probe.submit("TRX", execute=True)
    assert probe.verify("USDGO")["status"] == "verified"
    probe.submit("TRX", execute=True)
    assert probe.verify("TRX")["status"] == "verified"
    assert broker.state()["wallet"]["TRX"] == "1E-8"


@pytest.mark.parametrize("tamper", ["amount", "account", "expiry", "rule", "live_strategy"])
def test_changed_scope_or_expired_review_blocks_before_http(check, tamper):
    probe, broker, clock = check
    plan = probe.prepare("USDGO")
    broker.writes_enabled = True
    if tamper == "amount":
        plan["request"]["amount"] = "1000"
        probe.save(plan)
    if tamper == "account":
        plan["account_id"] = "another-account"
        probe.save(plan)
    if tamper == "expiry":
        clock[0] += 1800001
    if tamper == "rule":
        path = probe.root / "live-product-rules.local.json"
        rules = json.loads(path.read_text())
        rules["products"][0]["redemption_minimum"] = ".2"
        path.write_text(json.dumps(rules))
    if tamper == "live_strategy":
        (probe.root / "live-strategy-control.json").write_text("{}")
    with pytest.raises(ExecutionError):
        probe.submit("USDGO", execute=True)
    assert broker.state()["writes"] == []


def test_cli_missing_execute_flag_stops_before_any_credentials_or_connection(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "submit", "--coin", "USDGO"]) == 1
    assert "operator_coin_and_execute_flag_required" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
