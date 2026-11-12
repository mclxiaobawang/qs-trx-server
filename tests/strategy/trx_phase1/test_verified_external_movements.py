import json
from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.income_allocation import ownership
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution, digest
from freqtrade.strategy.trx_phase1.reconciliation_wait import verified_income_candidate
from freqtrade.strategy.trx_phase1.verified_external_movements import VERSION, collect, stage


NOW = 1793696800000


def test_external_reserve_redemption_changes_location_only():
    state, proof, policy = example()
    state["holdings"]["usdgo"]["reserved_amount"] = "450"
    original = deepcopy(state)
    final, _ = stage(state, proof, policy)
    assert final["holdings"]["usdgo"]["reserved_amount"] == "0"
    assert final["book"] == original["book"]
    assert state == original
    checkpoints = final["income_accounting"]["checkpoints"]
    assert checkpoints[-1]["positions"]["usdgo"].get("reserve", "0") == "0"


def seal(policy):
    policy["sha256"] = digest({k: v for k, v in policy.items() if k != "sha256"})
    return policy


def bill(bid, oid, coin, kind, size, stamp):
    return {
        "billId": bid,
        "bizOrderId": oid,
        "coin": coin,
        "groupType": "transaction" if kind in {"BUY", "SELL"} else "financial",
        "businessType": kind,
        "size": str(size),
        "fees": "0",
        "cTime": str(stamp),
        "balance": "0",
    }


def example(tail=False):
    book = Ledger(1000)
    book.convert("start", "to_usdgo", money(900), money(900), money(1))
    book.locked["USDGO"] = money(900)
    positions = {
        "usdgo": {
            "coin": "USDGO",
            "product_id": "g",
            "principal": "900",
            "interest": "0",
            "unassigned_amount": "500",
        }
    }
    state = {
        "account_id": "account",
        "created_at": NOW - 3600000,
        "interest_checked_at": NOW,
        "book": book.state(),
        "holdings": positions,
        "operation": None,
        "flow": None,
        "error": None,
        "pending_income": {},
        "receipts": {},
        "events": [],
        "products": [{"coin": "USDGO", "product_id": "g"}],
        "unassigned_offsets": {"USDGO": "0", "USDT": "10", "TRX": "0"},
        "income_accounting": {
            "status": "active",
            "allocations": {},
            "checkpoints": [{"timestamp": NOW, "positions": ownership(positions)}],
        },
    }
    policy = seal(
        {
            "version": VERSION,
            "enabled": True,
            "account_id": "account",
            "ledger_created_at": state["created_at"],
            "approved_at": NOW,
            "scope": "verified_external_stablecoins_only",
            "external_first": True,
            "usdgo_holdings": {"usdgo": "g"},
            "usdc_product_id": "c",
        }
    )
    redeemed = money(400 if tail else 500)
    paid = redeemed + money("10.00005") if tail else redeemed
    rows = {
        "USDGO": [bill("1", "redeem", "USDGO", "SINGLE_RELEASE_USER_IN", redeemed, NOW + 1000)],
        "USDT": [],
        "USDC": [],
    }
    fills, orders = [], {}

    def trade(tid, oid, symbol, base, quote, quantity, amount, stamp, side="sell"):
        rows[base].append(
            bill(
                tid + "b",
                tid,
                base,
                "SELL" if side == "sell" else "BUY",
                -quantity if side == "sell" else quantity,
                stamp,
            )
        )
        rows[quote].append(
            bill(
                tid + "q",
                tid,
                quote,
                "BUY" if side == "sell" else "SELL",
                amount if side == "sell" else -amount,
                stamp,
            )
        )
        fills.append(
            {
                "tradeId": tid,
                "orderId": oid,
                "symbol": symbol,
                "side": side,
                "size": str(quantity),
                "amount": str(amount),
                "cTime": str(stamp),
                "feeDetail": {"feeCoin": quote, "totalFee": "0", "totalDeductionFee": "0"},
            }
        )
        orders[oid] = [
            {
                "orderId": oid,
                "symbol": symbol,
                "side": side,
                "status": "filled",
                "baseVolume": str(quantity),
                "quoteVolume": str(amount),
            }
        ]

    if tail:
        trade("2", "o1", "USDGOUSDT", "USDGO", "USDT", redeemed, redeemed, NOW + 2000)
        trade("3", "o2", "USDCUSDT", "USDC", "USDT", paid, paid, NOW + 2500, "buy")
        policy["historical_cash_tail"] = {
            "trade_id": "3",
            "amount_USDT": "0.00005",
            "reimbursement_holding_id": "usdgo",
        }
        seal(policy)
    else:
        trade("2", "o1", "USDGOUSDC", "USDGO", "USDC", paid, paid, NOW + 2000)
    rows["USDC"].append(bill("4", "subscribe", "USDC", "SINGLE_USER_OUT", -paid, NOW + 3000))
    proof = {
        "observed_at": NOW + 5000,
        "account_id": "account",
        "account_binding_verified": True,
        "bills": rows,
        "fills": fills,
        "orders": orders,
        "results": {"redeem": {"result": "success"}, "subscribe": {"result": "success"}},
        "records": {
            "USDGO": {
                "redeem": [
                    {
                        "orderId": "r1",
                        "coinName": "USDGO",
                        "settleCoinName": "USDGO",
                        "productType": "flexible",
                        "orderType": "redeem",
                        "amount": str(redeemed),
                        "ts": str(NOW + 900),
                    }
                ],
                "subscribe": [],
            },
            "USDC": {
                "redeem": [],
                "subscribe": [
                    {
                        "orderId": "r2",
                        "coinName": "USDC",
                        "settleCoinName": "USDC",
                        "productType": "flexible",
                        "orderType": "subscribe",
                        "amount": str(paid),
                        "ts": str(NOW + 3500),
                    }
                ],
            },
        },
        "balances": {"USDC": {"total": "0"}},
        "raw_holdings": [{"productCoin": "USDC", "productId": "c", "holdAmount": str(paid)}],
        "markets": {"USDGOUSDT": {"bid": "1"}},
    }
    return state, proof, policy


def test_complete_external_conversion_preserves_every_strategy_asset_and_capital_field():
    original, proof, policy = example()
    saved = deepcopy(original)
    after, records = stage(original, proof, policy)
    assert original == saved
    assert after["book"] == original["book"]
    assert after["holdings"]["usdgo"]["principal"] == "900"
    assert money(after["holdings"]["usdgo"]["unassigned_amount"]) == 0
    assert after["external_usdc"]["savings"] == "500"
    assert len(records) == 3
    assert len(after["account_movement_bill_receipts"]) == 4
    assert all(r["exchange_writes"] == 0 for r in records)
    again, more = stage(after, proof, policy)
    assert again == after and more == []


def external_usdc_withdrawal():
    state, proof, policy = example()
    state["stablecoin_policy"] = {"products": [{"coin": "USDC", "product_id": "c"}]}
    book = Ledger.restore(state["book"])
    book.convert("strategy-usdc", "to_usdc", money(100), money(100), money(1))
    book.usdc_locked = money(90)
    state["book"] = book.state()
    state["holdings"]["usdc"] = {
        "coin": "USDC",
        "product_id": "c",
        "principal": "90",
        "interest": "0",
        "unassigned_amount": "1000",
    }
    state["income_accounting"]["checkpoints"][0]["positions"] = ownership(state["holdings"])
    state["external_usdc"] = {"spot": "0", "savings": "1000", "product_id": "c"}
    state["unassigned_offsets"]["USDT"] = "0"
    policy["usdt_external_withdrawals"] = True
    seal(policy)
    proof["bills"] = {
        "USDGO": [],
        "USDC": [
            bill("r", "redeem", "USDC", "SINGLE_RELEASE_USER_IN", "800", NOW + 1000),
            bill("s", "fill", "USDC", "SELL", "-679.5937", NOW + 2000),
        ],
        "USDT": [
            bill("b", "fill", "USDT", "BUY", "680.23931401", NOW + 2000),
            {
                **bill("w", "withdraw", "USDT", "WITHDRAW", "-680.22", NOW + 3000),
                "groupType": "withdraw",
                "fees": "-0.01",
            },
        ],
    }
    proof["fills"] = [
        {
            "tradeId": "fill",
            "orderId": "sell",
            "symbol": "USDCUSDT",
            "side": "sell",
            "size": "679.5937",
            "amount": "680.239314015",
            "cTime": str(NOW + 2000),
            "feeDetail": {"feeCoin": "USDT", "totalFee": "0", "totalDeductionFee": "0"},
        }
    ]
    proof["orders"] = {
        "sell": [
            {
                "orderId": "sell",
                "symbol": "USDCUSDT",
                "side": "sell",
                "status": "filled",
                "baseVolume": "679.5937",
                "quoteVolume": "680.239314015",
            }
        ]
    }
    proof["results"] = {"redeem": {"result": "success"}}
    proof["records"]["USDC"]["redeem"] = [
        {
            "orderId": "savings-redemption",
            "coinName": "USDC",
            "settleCoinName": "USDC",
            "productType": "flexible",
            "orderType": "redeem",
            "amount": "800",
            "ts": str(NOW + 900),
        }
    ]
    proof["raw_holdings"] = [
        {"orderId": "usdc", "productCoin": "USDC", "productId": "c", "holdAmount": "290"}
    ]
    proof["balances"]["USDC"]["total"] = "130.4063"
    proof["withdrawals"] = {
        "withdraw": [
            {
                "orderId": "withdraw",
                "coin": "USDT",
                "type": "withdraw",
                "status": "success",
                "size": "680.23",
                "fee": "-0.01",
                "cTime": str(NOW + 2500),
                "uTime": str(NOW + 3500),
            }
        ]
    }
    return state, proof, policy


def test_usdc_redemption_trade_and_withdrawal_leave_strategy_capital_and_fills_intact():
    state, proof, policy = external_usdc_withdrawal()
    before = deepcopy(state)
    updated, movements = stage(state, proof, policy)
    assert state == before and updated["book"] == before["book"]
    assert updated["unassigned_offsets"]["USDT"] == "0.009314015"
    assert updated["unassigned_offsets"]["USDC"] == "120.4063"
    assert updated["holdings"]["usdc"]["principal"] == "90"
    assert updated["holdings"]["usdc"]["unassigned_amount"] == "200"
    assert updated["income_accounting"]["checkpoints"][-1]["positions"] == ownership(
        updated["holdings"]
    )
    assert [row["kind"] for row in movements] == ["SINGLE_RELEASE_USER_IN", "trade", "WITHDRAW"]
    assert movements[-1]["amount"] == "680.23" and movements[-1]["fee"] == "0.01"
    assert stage(updated, proof, policy) == (updated, [])


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("strategy_usdc", "external_funds_insufficient"),
        ("strategy_usdt", "strategy_cash_would_be_used"),
        ("pending", "evidence_pending"),
        ("missing_record", "evidence_pending"),
        ("record_fee", "withdrawal_settlement_unverified"),
        ("unexpected_precision", "trade_bill_delta"),
        ("wrong_holding", "usdc_holding_location_unverified"),
    ],
)
def test_external_withdrawal_keeps_unproven_or_mixed_movements_uncommitted(fault, reason):
    state, proof, policy = external_usdc_withdrawal()
    record = proof["withdrawals"]["withdraw"][0]
    if fault == "strategy_usdc":
        # The redemption could be funded from outside, but not the entire sale.
        redemption = proof["records"]["USDC"]["redeem"][0]
        redemption["amount"] = proof["bills"]["USDC"][0]["size"] = "650"
    elif fault == "strategy_usdt":
        proof["bills"]["USDT"][-1]["size"] = "-680.23"
        record["size"] = "680.24"  # The fee crosses into owned USDT.
    elif fault == "pending":
        record["status"] = "pending"
    elif fault == "missing_record":
        proof["withdrawals"] = {}
    elif fault == "record_fee":
        record["fee"] = "-0.02"
    elif fault == "unexpected_precision":
        proof["bills"]["USDT"][0]["size"] = "680.23931400"
    elif fault == "wrong_holding":
        proof["raw_holdings"][0].update(orderId="different", holdAmount="200")
        proof["raw_holdings"].append(
            {"orderId": "usdc", "productCoin": "USDC", "productId": "c", "holdAmount": "90"}
        )
    before = deepcopy(state)
    with pytest.raises(ExecutionError, match=reason):
        stage(state, proof, policy)
    assert state == before


def test_fresh_usdc_interest_is_left_for_native_mixed_ownership_allocation():
    state, proof, policy = external_usdc_withdrawal()
    proof["bills"]["USDC"].append(
        bill("interest", "payout", "USDC", "BATCH_INTEREST_USER_IN", "0.5", NOW + 2500)
    )
    proof["balances"]["USDC"]["total"] = "130.9063"
    updated, _ = stage(state, proof, policy)
    assert "interest" not in updated["account_movement_bill_receipts"]
    assert updated["unassigned_offsets"]["USDC"] == "120.4063"


def test_withdrawal_only_triggers_native_proof_query(mocker):
    state, proof, policy = external_usdc_withdrawal()
    native = mocker.Mock()
    native.account_id.return_value = state["account_id"]
    native.bills.side_effect = lambda coin, *_: proof["bills"][coin][-1:] if coin == "USDT" else []
    native.pages.return_value = proof["raw_holdings"]
    evidence = {
        "v2/spot/wallet/withdrawal-records": proof["withdrawals"]["withdraw"],
        "v2/spot/account/assets": [
            {"coin": "USDC", "available": "10", "frozen": "0", "locked": "0"}
        ],
        "v2/earn/account/assets": [{"coin": "USDC", "amount": "290"}],
    }
    native.get.side_effect = lambda path, *_: evidence[path]
    result = collect(mocker.Mock(state=state, broker=native), policy, proof["observed_at"])
    assert result["withdrawals"] == proof["withdrawals"]
    assert result["fills"] == [] and result["orders"] == {}
    native.get.assert_any_call("v2/spot/wallet/withdrawal-records", {"orderId": "withdraw"})


def test_wallet_reconciliation_admits_a_fully_proven_external_withdrawal(
    tmp_path, monkeypatch, mocker
):
    state, proof, policy = external_usdc_withdrawal()
    monkeypatch.setattr(
        LiveExecution, "stablecoin_policy", property(lambda self: self.state["stablecoin_policy"])
    )
    (tmp_path / "external-stable-movements.json").write_text(json.dumps(policy))
    execution = object.__new__(LiveExecution)
    execution.root = tmp_path / "live_execution" / "account"
    execution.account_id, execution.state, execution.broker = (
        state["account_id"],
        state,
        mocker.Mock(),
    )
    broker = execution.broker
    broker.account_id.return_value = state["account_id"]
    broker.bills.side_effect = lambda coin, *_: proof["bills"][coin]
    broker.pages.side_effect = lambda path, *_args, **_kwargs: (
        proof["fills"] if path.endswith("/fills") else proof["raw_holdings"]
    )
    broker.savings_records.side_effect = lambda coin, *_args, order_type: proof["records"][coin][
        order_type
    ]
    broker.result.return_value = proof["results"]["redeem"]
    broker.balances.return_value = {
        "USDT": {"available": "0.009314015"},
        "USDC": {"available": "130.4063"},
        "USDGO": {"available": "0"},
        "TRX": {"available": "0"},
    }
    raw_assets = [
        {"coin": coin, "available": row["available"], "frozen": "0", "locked": "0"}
        for coin, row in broker.balances.return_value.items()
    ]
    broker.get.side_effect = lambda path, params=None: {
        "v2/spot/wallet/withdrawal-records": proof["withdrawals"]["withdraw"],
        "v2/spot/trade/orderInfo": proof["orders"]["sell"],
        "v2/spot/account/assets": raw_assets,
        "v2/earn/account/assets": [{"coin": "USDC", "amount": "290"}],
    }[path]

    def income_candidate(self, book, positions, receipts, _now):
        if "w" not in self.state.get("account_movement_bill_receipts", {}):
            raise ExecutionError("unrecognized_usdt_movement")
        return book, positions, receipts, [], self.state["income_accounting"]

    # The unchanged-income case isolates the external movement recovery entry point.
    monkeypatch.setattr(LiveExecution, "income_candidate", income_candidate)
    book, *_ = verified_income_candidate(
        execution,
        Ledger.restore(state["book"]),
        state["holdings"],
        state["receipts"],
        proof["observed_at"],
    )
    assert book.state() == state["book"]
    assert set(execution.state["account_movement_bill_receipts"]) == {"r", "s", "b", "w"}
    assert execution.state["unassigned_offsets"]["USDT"] == "0.009314015"


def usdt_location(*, redeem):
    state, proof, policy = example()
    state["stablecoin_policy"] = {"products": [{"coin": "USDT", "product_id": "owned"}]}
    for key, principal, external in (
        ("owned", "20", "0"),
        ("outside", "0", "5" if redeem else "0"),
    ):
        state["holdings"][key] = {
            "coin": "USDT",
            "product_id": key,
            "principal": principal,
            "interest": "0",
            "unassigned_amount": external,
        }
    accounting = state["income_accounting"]
    accounting["checkpoints"][-1]["positions"] = ownership(state["holdings"])
    active = {"owned": {"product_id": "owned", "profit": "0.1"}}
    if redeem:
        active["outside"] = {"product_id": "outside", "profit": "0.2"}
    accounting["usdt_product_counter_anchor"] = {"observed_at": NOW, "holdings": active}
    policy["usdt_external_savings"] = True
    seal(policy)
    kind = "redeem" if redeem else "subscribe"
    proof["bills"] = {
        "USDGO": [],
        "USDC": [],
        "USDT": [
            bill(
                "5",
                "usdt-op",
                "USDT",
                "SINGLE_RELEASE_USER_IN" if redeem else "SINGLE_USER_OUT",
                "5" if redeem else "-5",
                NOW + 1000,
            )
        ],
    }
    proof["results"]["usdt-op"] = {"result": "success"}
    proof["records"]["USDT"] = {
        kind: [
            {
                "orderId": "usdt-record",
                "coinName": "USDT",
                "settleCoinName": "USDT",
                "productType": "flexible",
                "orderType": kind,
                "amount": "5",
                "ts": str(NOW + 900),
            }
        ]
    }
    proof["raw_holdings"] = [
        {
            "orderId": "owned",
            "productCoin": "USDT",
            "productId": "owned",
            "holdAmount": "20",
            "interestCoin": "USDT",
            "totalProfit": "0.1",
        }
    ]
    if not redeem:
        proof["raw_holdings"].append(
            {
                "orderId": "outside",
                "productCoin": "USDT",
                "productId": "outside",
                "holdAmount": "5",
                "interestCoin": "USDT",
                "totalProfit": "0",
            }
        )
    return state, proof, policy


@pytest.mark.parametrize("redeem", [True, False])
def test_usdt_savings_updates_only_external_location_and_counter(redeem):
    state, proof, policy = usdt_location(redeem=redeem)
    original = deepcopy(state)
    staged, [movement] = stage(state, proof, policy)
    assert state == original and staged["book"] == original["book"]
    assert staged["holdings"]["owned"] == original["holdings"]["owned"]
    assert staged["holdings"]["outside"]["unassigned_amount"] == ("0" if redeem else "5")
    assert staged["unassigned_offsets"]["USDT"] == ("15" if redeem else "5")
    accounting = staged["income_accounting"]
    assert ("outside" in accounting["usdt_product_counter_anchor"]["holdings"]) != redeem
    assert ("outside" in accounting["usdt_closed_holding_counters"]) == redeem
    assert accounting["checkpoints"][-1]["source_bill_id"] == "5"
    assert movement["exchange_writes"] == 0
    assert stage(staged, proof, policy) == (staged, [])


@pytest.mark.parametrize(
    "fault", ["disabled", "pending_interest", "mixed", "wrong_delta", "duplicate"]
)
def test_usdt_savings_leaves_ambiguous_funds_for_review(fault):
    state, proof, policy = usdt_location(redeem=True)
    if fault == "disabled":
        del policy["usdt_external_savings"]
        seal(policy)
    elif fault == "pending_interest":
        proof["bills"]["USDT"].append(
            bill("6", "interest", "USDT", "BATCH_INTEREST_USER_IN", "0.1", NOW + 2000)
        )
    elif fault == "mixed":
        state["holdings"]["outside"].update(principal="1", unassigned_amount="4")
    elif fault == "wrong_delta":
        proof["raw_holdings"][0]["holdAmount"] = "19"
    else:
        proof["bills"]["USDT"].append(
            bill("6", "other", "USDT", "SINGLE_RELEASE_USER_IN", "5", NOW + 2000)
        )
    original = deepcopy(state)
    with pytest.raises(ExecutionError):
        stage(state, proof, policy)
    assert state == original


@pytest.mark.parametrize(
    "fault",
    [
        "account",
        "policy",
        "result",
        "missing_bill",
        "missing_fill",
        "incomplete_order",
        "wrong_order_side",
        "changed_amount",
        "future_bill",
        "source_funds",
        "wallet",
        "wrong_product",
        "missing_position",
        "active_operation",
        "active_flow",
        "pending_income",
        "duplicate_conflict",
        "ambiguous_savings",
        "unknown_pair",
        "unverified_fee",
    ],
)
def test_ambiguous_or_nonconserving_evidence_is_rejected_atomically(fault):  # noqa: C901
    state, proof, policy = example()
    if fault == "account":
        proof["account_id"] = "other"
    elif fault == "policy":
        policy["external_first"] = False
    elif fault == "result":
        proof["results"]["redeem"]["result"] = "processing"
    elif fault == "missing_bill":
        proof["bills"]["USDGO"].pop()
    elif fault == "missing_fill":
        proof["fills"] = []
    elif fault == "incomplete_order":
        proof["orders"]["o1"][0]["baseVolume"] = "501"
    elif fault == "wrong_order_side":
        proof["orders"]["o1"][0]["side"] = "buy"
    elif fault == "changed_amount":
        proof["bills"]["USDC"][0]["size"] = "501"
    elif fault == "future_bill":
        proof["bills"]["USDGO"][0]["cTime"] = str(NOW + 6000)
    elif fault == "source_funds":
        state["holdings"]["usdgo"]["unassigned_amount"] = "499"
    elif fault == "wallet":
        proof["balances"]["USDC"]["total"] = "1"
    elif fault == "wrong_product":
        proof["raw_holdings"][0]["productId"] = "other"
    elif fault == "missing_position":
        proof["raw_holdings"] = []
    elif fault == "active_operation":
        state["operation"] = {"status": "accepted"}
    elif fault == "active_flow":
        state["flow"] = {"status": "running"}
    elif fault == "pending_income":
        state["pending_income"] = {"unverified": {}}
    elif fault == "duplicate_conflict":
        raw = deepcopy(proof["bills"]["USDGO"][0])
        raw["size"] = "501"
        proof["bills"]["USDGO"].append(raw)
    elif fault == "ambiguous_savings":
        proof["records"]["USDGO"]["redeem"] *= 2
    elif fault == "unknown_pair":
        proof["fills"][0]["symbol"] = "TRXUSDT"
    elif fault == "unverified_fee":
        proof["fills"][0]["feeDetail"]["totalDeductionFee"] = "1"
    saved = deepcopy(state)
    with pytest.raises(ExecutionError):
        stage(state, proof, policy)
    assert state == saved


def test_historical_tail_is_reimbursed_in_kind_without_new_capital_or_fictitious_cash():
    state, proof, policy = example(tail=True)
    saved = deepcopy(state)
    after, _ = stage(state, proof, policy)
    book = Ledger.restore(after["book"])
    assert book.cash == money("99.99995")
    assert book.usdgo == book.locked["USDGO"] == money("900.00005")
    assert book.equity(1, 1) == Ledger.restore(saved["book"]).equity(1, 1)
    for key in ["initial", "external_flows", "fund_units", "lots", "orders", "realized", "fees"]:
        assert after["book"][key] == saved["book"][key]
    assert after["external_usdc"]["savings"] == "410.00005"
    assert money(after["holdings"]["usdgo"]["unassigned_amount"]) == money("99.99995")
    again, records = stage(after, proof, policy)
    assert again == after and records == []


def test_future_external_conversion_cannot_spend_strategy_cash_even_for_a_small_tail():
    state, proof, policy = example(tail=True)
    del policy["historical_cash_tail"]
    seal(policy)
    with pytest.raises(ExecutionError, match="strategy_cash_would_be_used"):
        stage(state, proof, policy)


def test_changed_already_applied_bill_is_rejected():
    state, proof, policy = example()
    after, _ = stage(state, proof, policy)
    proof["bills"]["USDC"][0]["size"] = "999"
    with pytest.raises(ExecutionError, match="bill_changed"):
        stage(after, proof, policy)


def test_external_usdc_interest_is_kept_entirely_outside_strategy():
    state, proof, policy = example()
    after, _ = stage(state, proof, policy)
    follow = deepcopy(proof)
    follow["observed_at"] += 3600000
    follow["bills"] = {
        "USDT": [],
        "USDGO": [],
        "USDC": [bill("income", "payout", "USDC", "BATCH_INTEREST_USER_IN", "0.1", NOW + 3600000)],
    }
    follow["balances"]["USDC"]["total"] = "0.1"
    final, records = stage(after, follow, policy)
    assert final["book"] == after["book"]
    assert final["external_usdc"]["spot"] == "0.1"
    assert records[0]["strategy_income_changed"] is False
