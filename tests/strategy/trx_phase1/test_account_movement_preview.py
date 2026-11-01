"""Manual movements must conserve independent native receipts and ownership."""

from copy import deepcopy
from decimal import Decimal

import pytest

from freqtrade.strategy.trx_phase1.account_movement_preview import preview
from freqtrade.strategy.trx_phase1.ledger import Ledger, Lot
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError


D = Decimal
HOUR = 3600000
START = 1792904400000


@pytest.fixture
def snapshot():
    book = Ledger(1000)
    book.cash = D(".2")
    book.usdgo = D(500)
    book.usdgo_cost = D(500)
    book.locked["USDGO"] = D(500)
    book.lots = [Lot("old-trx", START - HOUR, D(100), D(5), locked=D(100), remaining_cost=D(500))]
    positions = {
        "usdgo-hold": {
            "coin": "USDGO",
            "product_id": "usdgo-flex",
            "principal": "500",
            "interest": "0",
            "unassigned_amount": "0",
        },
        "trx-hold": {
            "coin": "TRX",
            "product_id": "trx-flex",
            "principal": "100",
            "interest": "0",
            "unassigned_amount": "0",
        },
    }
    history = [
        {
            "timestamp": START - HOUR,
            "positions": {
                k: {
                    "coin": p["coin"],
                    "product_id": p["product_id"],
                    "strategy": p["principal"],
                    "external": "0",
                }
                for k, p in positions.items()
            },
        }
    ]
    state = {
        "account_id": "account",
        "book": book.state(),
        "created_at": START - HOUR,
        "interest_checked_at": START,
        "operation": None,
        "holdings": positions,
        "unassigned_offsets": {"USDT": ".1", "USDGO": "0", "TRX": "0"},
        "income_accounting": {"checkpoints": history},
    }

    def bill(key, coin, group, kind, amount, balance, minute, business=None):
        return {
            "billId": key,
            "bizOrderId": business or key,
            "coin": coin,
            "groupType": group,
            "businessType": kind,
            "size": amount,
            "balance": balance,
            "fees": "0",
            "cTime": str(START + minute * 60000),
        }

    own = (D("49.9") * D(".07") / D("50.2")).quantize(D(".00000001"), rounding="ROUND_DOWN")
    evidence = {
        "account_id": "account",
        "account_binding_verified": True,
        "strategy_state": {"desired": "paused"},
        "observed_at_ms": START + HOUR,
        "bills": {
            "USDT": [
                bill("reward", "USDT", "other", "CASH_COUPON_ACTIVITY_USER_IN", ".03", ".33", 1),
                bill("deposit", "USDT", "deposit", "DEPOSIT", "100", "100.33", 2),
                bill("sub-usdt", "USDT", "financial", "SINGLE_USER_OUT", "-50", "50.33", 3),
                bill("fx-quote", "USDT", "transaction", "SELL", "-50.2", ".13", 4, "fill-native"),
            ],
            "USDGO": [
                bill("fx-base", "USDGO", "transaction", "BUY", "49.9", "49.9", 4, "fill-native"),
                bill("sub-usdgo", "USDGO", "financial", "SINGLE_USER_OUT", "-49.9", "0", 5),
            ],
            "TRX": [],
        },
        "orders": {
            "fill-native": {
                "native_trade_id": "fill-native",
                "order": {
                    "orderId": "distinct-order",
                    "symbol": "USDGOUSDT",
                    "side": "buy",
                    "status": "filled",
                },
                "fills": [
                    {
                        "id": "fill-native",
                        "order_id": "distinct-order",
                        "side": "buy",
                        "timestamp": START + 4 * 60000,
                        "quantity": "50",
                        "quote": "50.2",
                        "fees": [{"coin": "USDGO", "cost": ".1"}],
                    }
                ],
            }
        },
        "subscription_results": {
            "sub-usdt": {"result": "success"},
            "sub-usdgo": {"result": "success"},
        },
        "holdings": {
            "usdgo-hold": {"coin": "USDGO", "product_id": "usdgo-flex", "amount": "549.9"},
            "trx-hold": {"coin": "TRX", "product_id": "trx-flex", "amount": "100"},
            "usdt-hold": {"coin": "USDT", "product_id": "usdt-flex", "amount": "50"},
        },
        "earn_overview": [
            {"coin": "USDGO", "amount": "549.9"},
            {"coin": "TRX", "amount": "100"},
            {"coin": "USDT", "amount": "50"},
        ],
        "balances": {
            c: {"available": v, "frozen": "0", "locked": "0"}
            for c, v in [("USDT", ".13"), ("USDGO", "0"), ("TRX", "0")]
        },
    }
    return state, evidence, own


def test_complete_chain_and_source_ownership_do_not_mutate_inputs(snapshot):
    state, evidence, own = snapshot
    before = deepcopy((state, evidence))
    result = preview(state, evidence)
    assert result == preview(state, evidence)
    assert (state, evidence) == before
    assert result["original_strategy_USDT_used_by_manual_buy"] == ".07" or D(
        result["original_strategy_USDT_used_by_manual_buy"]
    ) == D(".07")
    assert D(result["strategy_USDGO_net_receipt_from_manual_buy"]) == own
    held = next(p for p in result["holding_checks"] if p["coin"] == "USDGO")
    assert D(held["strategy"]) == D(500) + own
    assert D(held["outside_strategy"]) == D("49.9") - own
    assert result["protected"]["trx_remaining_cost"] == "500"
    assert result["ledger_writes"] == result["exchange_writes"] == 0
    assert not result["formal_commit_ready"] and not result["resume_authorized"]


@pytest.mark.parametrize(
    "fault,code",
    [
        ("wrong_account", "preview_account_mismatch"),
        ("running", "preview_requires_paused"),
        ("missing_fill", "preview_fx_order_unverified"),
        ("fee_gap", "preview_fx_native_fill_conservation_failed"),
        ("unproven_sub", "preview_subscription_result_unverified"),
        ("unknown", "preview_unknown_account_movement"),
        ("final_gap", "preview_wallet_conservation_failed"),
        ("holding_gap", "preview_earn_overview_mismatch"),
        ("sequence_gap", "preview_native_balance_sequence_failed"),
        ("conflict", "preview_conflicting_bill"),
    ],
)
def test_incomplete_or_conflicting_evidence_never_passes(snapshot, fault, code):
    state, evidence, _ = snapshot
    if fault == "wrong_account":
        evidence["account_id"] = "other"
    elif fault == "running":
        evidence["strategy_state"]["desired"] = "running"
    elif fault == "missing_fill":
        evidence["orders"]["fill-native"]["fills"] = []
    elif fault == "fee_gap":
        evidence["orders"]["fill-native"]["fills"][0]["fees"][0]["cost"] = ".09"
    elif fault == "unproven_sub":
        evidence["subscription_results"]["sub-usdt"]["result"] = "unknown"
    elif fault == "unknown":
        evidence["bills"]["USDT"][1]["businessType"] = "UNKNOWN_TRANSFER"
    elif fault == "final_gap":
        evidence["balances"]["USDT"]["available"] = ".131"
    elif fault == "holding_gap":
        evidence["holdings"]["usdgo-hold"]["amount"] = "550"
    elif fault == "sequence_gap":
        evidence["bills"]["USDT"][1]["balance"] = "100.34"
    elif fault == "conflict":
        evidence["bills"]["USDT"].append({**evidence["bills"]["USDT"][0], "size": ".04"})
    before = deepcopy((state, evidence))
    with pytest.raises(ExecutionError, match=code):
        preview(state, evidence)
    assert (state, evidence) == before


def test_interest_uses_full_completed_hour_and_does_not_gift_external_interest(snapshot):
    state, evidence, _own = snapshot
    # First payout's prior full hour includes the subscription partway through;
    # newly subscribed units are excluded by the complete-hour minimum rule.
    evidence["observed_at_ms"] = START + HOUR + 60000
    evidence["bills"]["USDGO"].append(
        {
            "billId": "interest",
            "bizOrderId": "interest",
            "coin": "USDGO",
            "groupType": "financial",
            "businessType": "BATCH_INTEREST_USER_IN",
            "size": ".5",
            "balance": ".5",
            "fees": "0",
            "cTime": str(START + HOUR + 60000),
        }
    )
    evidence["balances"]["USDGO"]["available"] = ".5"
    result = preview(state, evidence)
    record = result["new_interest_allocations"][0]
    assert D(record["strategy_amount"]) == D(".5") and D(record["external_amount"]) == 0
    assert D(record["strategy_units"]) == 500 and D(record["external_units"]) == 0
