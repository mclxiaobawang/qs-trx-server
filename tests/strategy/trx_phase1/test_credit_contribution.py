from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.credit_contribution import candidate
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest


NOW = 1792382000000


def sample():
    book = Ledger(100)
    book.buy("fill", "order", NOW - 10000, 10, 1, ".01")
    book.convert("fx", "to_usdgo", 50, 50, 1)
    record = {
        "bill_id": "bill",
        "order_id": "unlinked-order",
        "group": "other",
        "coin": "USDT",
        "amount": "0.04635835",
        "fees": "0",
        "timestamp": NOW - 2000,
        "recorded_at": NOW,
        "destination": "spot",
        "offset_delta": ".046358358466",
        "reported_amount": ".04635835",
        "precision_residual": ".000000008466",
        "attribution": "external_account_credit",
        "type": "REBATE_REWARDS",
    }
    state = {
        "account_id": "account",
        "created_at": NOW - 100000,
        "book": book.state(),
        "receipts": {
            "account-credit:bill": digest(
                {
                    k: record[k]
                    for k in (
                        "bill_id",
                        "order_id",
                        "coin",
                        "group",
                        "type",
                        "amount",
                        "fees",
                        "timestamp",
                    )
                }
            )
        },
        "quarantined_account_credits": {"account-credit:bill": record},
        "unassigned_offsets": {"USDT": ".046358358466"},
        "events": [],
        "operation": None,
        "flow": None,
        "holdings": {"original": "unchanged"},
        "asset_contributions": {},
        "error": None,
        "pending_income": {},
    }
    policy = {
        "version": "account-credit-capital-v1",
        "account_id": "account",
        "ledger_created_at": state["created_at"],
        "approved_at": NOW,
        "expires_at": NOW + 300000,
        "credits": [
            {
                "receipt_id": "account-credit:bill",
                "bill_id": "bill",
                "amount": record["offset_delta"],
            }
        ],
    }
    markets = {
        s: {"bid": p, "timestamp": NOW} for s, p in [("TRXUSDT", "1.1"), ("USDGOUSDT", "1.001")]
    }
    return state, policy, markets, book.free_cash + money(record["offset_delta"])


def test_capital_preserves_profit_nav_receipts_and_is_not_fee_income():
    state, policy, markets, wallet = sample()
    original = deepcopy(state)
    work, record = candidate(state, policy, markets, wallet, NOW)
    assert state == original
    before, after = Ledger.restore(state["book"]), Ledger.restore(work["book"])
    before.usdgo_mark = money(markets["USDGOUSDT"]["bid"])
    amount = money(record["value"])
    assert after.cash == before.cash + amount
    assert after.external_flows == before.external_flows + amount
    assert after.equity("1.1") == before.equity("1.1") + amount
    assert abs(
        after.equity("1.1") / after.fund_units - before.equity("1.1") / before.fund_units
    ) < money("1e-25")
    assert after.summary("1.1")["total_profit"] == before.summary("1.1")["total_profit"]
    for field in (
        "realized",
        "fees",
        "fee_rebates",
        "orders",
        "lots",
        "interest_trx",
        "stable_interest",
    ):
        assert work["book"][field] == state["book"][field]
    assert work["quarantined_account_credits"] == state["quarantined_account_credits"]
    assert work["receipts"] == state["receipts"]
    assert work["holdings"] == state["holdings"]
    assert money(work["unassigned_offsets"]["USDT"]) == 0
    assert record["source_credits"][0]["bill_id"] == "bill"
    assert len(work["events"]) == 1
    # Lost response/restart after commit returns the same record, even after expiry.
    again, same = candidate(work, policy, {}, 0, NOW + 999999)
    assert again == work and same == record
    # A new policy must never adopt the same source bill a second time.
    different = {**policy, "approved_at": NOW + 1}
    with pytest.raises(ExecutionError):
        candidate(work, different, markets, wallet, NOW + 2)


@pytest.mark.parametrize(
    "problem",
    [
        "wallet",
        "changed_credit",
        "expired",
        "pending",
        "inflight",
        "stale_quote",
        "duplicate_source",
        "missing_receipt",
        "other_account",
        "changed_fingerprint",
    ],
)
def test_changed_or_unverified_credit_leaves_state_untouched(problem):
    state, policy, markets, wallet = sample()
    if problem == "wallet":
        wallet -= money(".01")
    elif problem == "changed_credit":
        state["quarantined_account_credits"]["account-credit:bill"]["offset_delta"] = ".1"
    elif problem == "expired":
        policy["expires_at"] = NOW
    elif problem == "pending":
        state["pending_income"] = {"unresolved": {}}
    elif problem == "inflight":
        state["operation"] = {"status": "sending"}
    elif problem == "stale_quote":
        markets["TRXUSDT"]["timestamp"] = NOW - 60001
    elif problem == "duplicate_source":
        policy["credits"] *= 2
    elif problem == "missing_receipt":
        state["receipts"] = {}
    elif problem == "changed_fingerprint":
        state["quarantined_account_credits"]["account-credit:bill"]["order_id"] = "changed"
    else:
        policy["account_id"] = "other"
    before = deepcopy(state)
    with pytest.raises(ExecutionError):
        candidate(state, policy, markets, wallet, NOW)
    assert state == before
