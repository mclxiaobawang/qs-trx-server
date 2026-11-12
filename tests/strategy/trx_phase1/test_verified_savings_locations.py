"""Mixed USDC subscriptions need successful native evidence and exact conservation."""

from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.income_allocation import ownership
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.verified_savings_locations import candidate


@pytest.fixture
def usdc_sweep():
    book = Ledger(2000)
    book.usdc, book.usdc_locked, book.usdc_cost = map(money, ["100.15", "100", "100.15"])
    book.usdc_mark = money(1)
    state = {
        "account_id": "account",
        "created_at": 1,
        "interest_checked_at": 3600001,
        "book": book.state(),
        "operation": None,
        "flow": None,
        "flows": [],
        "error": None,
        "pending_income": {},
        "events": [],
        "receipts": {},
        "products": [{"coin": "USDGO", "product_id": "usdgo-flex"}],
        "stablecoin_policy": {
            "version": "old-stablecoin-v1",
            "account_id": "account",
            "ledger_created_at": 1,
            "starts_at": 2,
            "products": [
                {"coin": "USDT", "product_id": "usdt-flex"},
                {"coin": "USDC", "product_id": "usdc-flex"},
            ],
        },
        "unassigned_offsets": {"USDT": "0", "USDGO": "0", "TRX": "0", "USDC": "14.29"},
        "external_usdc": {
            "spot": "14.29",
            "savings": "200",
            "outside_strategy": True,
            "product_id": "usdc-flex",
        },
        "holdings": {
            "held": {
                "coin": "USDC",
                "product_id": "usdc-flex",
                "principal": "100",
                "interest": "0",
                "unassigned_amount": "200",
                "subscription_ids": [],
                "lots": {},
            }
        },
        "income_accounting": {"status": "active", "policy_sha256": "policy", "allocations": {}},
    }
    state["income_accounting"]["checkpoints"] = [
        {"timestamp": 1, "positions": ownership(state["holdings"])}
    ]
    evidence = {
        "account_id": "account",
        "account_binding_verified": True,
        "observed_at_ms": 3600003,
        "bills": {
            "USDT": [],
            "USDGO": [],
            "TRX": [],
            "USDC": [
                {
                    "billId": "bill",
                    "bizOrderId": "order",
                    "coin": "USDC",
                    "groupType": "financial",
                    "businessType": "SINGLE_USER_OUT",
                    "size": "-14.44",
                    "fees": "0",
                    "balance": "0",
                    "cTime": "3600002",
                }
            ],
        },
        "holdings": {"held": {"coin": "USDC", "product_id": "usdc-flex", "amount": "314.44"}},
        "earn_overview": [{"coin": "USDC", "amount": "314.44"}],
        "balances": {
            c: {"available": v}
            for c, v in {"USDT": "2000", "USDGO": "0", "TRX": "0", "USDC": "0"}.items()
        },
        "subscription_results": {"order": {"result": "success"}},
        "markets": {"USDGOUSDT": {"bid": "1"}, "USDCUSDT": {"bid": "1"}},
        "orders": {},
    }
    return state, evidence


def test_usdc_sweep_preserves_ownership_capital_history_and_is_idempotent(usdc_sweep):
    state, evidence = usdc_sweep
    before = deepcopy(state)
    new, records = candidate(state, evidence)
    assert state == before
    assert len(records) == 1
    assert records[0]["coin"] == "USDC"
    assert money(records[0]["strategy_amount"]) == money(".15")
    assert money(records[0]["external_amount"]) == money("14.29")
    assert money(new["book"]["usdc_locked"]) == money("100.15")
    assert money(new["holdings"]["held"]["principal"]) == money("100.15")
    assert money(new["external_usdc"]["savings"]) == money("214.29")
    assert money(new["external_usdc"]["spot"]) == 0
    for key in [
        "initial",
        "external_flows",
        "fund_units",
        "lots",
        "orders",
        "realized",
        "usdc",
        "usdc_cost",
    ]:
        assert new["book"][key] == before["book"][key]
    assert new["stablecoin_policy"] == before["stablecoin_policy"]
    assert candidate(new, evidence) == (new, [])


@pytest.mark.parametrize(
    "fault",
    [
        "pending",
        "missing",
        "amount",
        "wallet",
        "product",
        "overview",
        "account",
        "withdrawal",
        "partial",
        "accepted",
    ],
)
def test_unverified_or_ambiguous_usdc_never_changes_input(usdc_sweep, fault):
    state, evidence = usdc_sweep
    bill = evidence["bills"]["USDC"][0]
    if fault == "pending":
        evidence["subscription_results"]["order"]["result"] = "pending"
    elif fault == "missing":
        evidence["subscription_results"] = {}
    elif fault == "amount":
        bill["size"] = "-14.43"
    elif fault == "wallet":
        evidence["balances"]["USDC"]["available"] = ".01"
    elif fault == "product":
        evidence["holdings"]["held"]["product_id"] = "other"
    elif fault == "overview":
        evidence["earn_overview"][0]["amount"] = "314.43"
    elif fault == "account":
        evidence["account_id"] = "other"
    elif fault == "withdrawal":
        bill["businessType"] = "WITHDRAW"
        evidence["bills"]["USDC"].append(
            {**bill, "billId": "second", "businessType": "SINGLE_USER_OUT"}
        )
    elif fault == "partial":
        bill.update(size="-14.43", balance=".01")
        evidence["balances"]["USDC"]["available"] = ".01"
        evidence["holdings"]["held"]["amount"] = "314.43"
        evidence["earn_overview"][0]["amount"] = "314.43"
    else:
        state["operation"] = {"status": "accepted", "exchange_id": "native"}
    before = deepcopy(state)
    with pytest.raises(ExecutionError):
        candidate(state, evidence)
    assert state == before
