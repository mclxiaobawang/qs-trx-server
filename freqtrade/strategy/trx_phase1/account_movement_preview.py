"""Pure evidence replay for a paused account; never writes a ledger or submits."""

import hashlib
import json
from copy import deepcopy
from decimal import ROUND_DOWN, localcontext

from .income_allocation import HOUR, effective_units
from .ledger import Ledger, money
from .live_broker import ExecutionError


COINS = ("USDT", "USDGO", "TRX")
UNIT = money("0.00000001")
EXTERNAL_CREDITS = {
    "CASH_COUPON_ACTIVITY_USER_IN": "outside_strategy_cash_coupon_reward",
    "REBATE_REWARDS": "strategy_order_link_unverified",
    "REBATE_COUPON_ACTIVITY_USER_IN": "outside_strategy_rebate_coupon_reward",
}


def require(condition, code):
    if not condition:
        raise ExecutionError(code)


def quantity(value):
    result = money(value)
    require(result.is_finite(), "preview_invalid_quantity")
    return result


def _preview(execution, evidence):  # noqa: C901 - chronological conservation proof
    original = deepcopy(execution)
    require(evidence.get("account_id") == execution["account_id"], "preview_account_mismatch")
    require(evidence.get("account_binding_verified") is True, "preview_account_unverified")
    require(
        evidence.get("strategy_state", {}).get("desired") == "paused", "preview_requires_paused"
    )
    require(not execution.get("operation"), "preview_execution_in_flight")
    book = Ledger.restore(execution["book"])
    coins = (*COINS, "USDC") if "USDC" in execution["unassigned_offsets"] else COINS
    since, until = execution["interest_checked_at"], evidence["observed_at_ms"]
    require(since <= until, "preview_invalid_window")
    spot = {
        "USDT": {
            "strategy": book.free_cash,
            "external": quantity(execution["unassigned_offsets"]["USDT"]),
        },
        "USDGO": {
            "strategy": book.usdgo - book.locked["USDGO"],
            "external": quantity(execution["unassigned_offsets"]["USDGO"]),
        },
        "TRX": {
            "strategy": book.principal
            + book.interest_trx
            - sum((p.locked for p in book.lots), money(0))
            - book.locked["TRX_INTEREST"],
            "external": quantity(execution["unassigned_offsets"]["TRX"]),
        },
    }
    if "USDC" in coins:
        spot["USDC"] = {
            "strategy": book.usdc - book.usdc_locked,
            "external": quantity(execution["unassigned_offsets"]["USDC"]),
        }
    positions = {
        key: {
            "coin": p["coin"],
            "product_id": p["product_id"],
            "strategy": quantity(p["principal"]) + quantity(p["interest"]),
            "external": quantity(p["unassigned_amount"]),
        }
        for key, p in execution["holdings"].items()
    }
    history = deepcopy(execution.get("income_accounting", {}).get("checkpoints", []))
    require(bool(history), "preview_hourly_ownership_history_required")
    rows, seen = [], {}
    for coin, bills in evidence["bills"].items():
        require(coin in coins, "preview_unexpected_coin")
        for raw in bills:
            require(raw.get("coin") == coin, "preview_bill_coin_mismatch")
            key = str(raw["billId"])
            if key in seen:
                require(seen[key] == raw, "preview_conflicting_bill")
                continue
            seen[key] = raw
            if not since < int(raw["cTime"]) <= until:
                continue
            require(quantity(raw["fees"]) == 0, "preview_bill_fee_semantics_unverified")
            rows.append(deepcopy(raw))
    transactions = {}
    for row in rows:
        if row["groupType"] == "transaction":
            transactions.setdefault(str(row["bizOrderId"]), []).append(row)
    consumed, timeline, rewards, interest = set(), [], [], []
    strategy_usdt_used = money(0)
    strategy_usdgo_from_fx = money(0)

    def checkpoint(stamp):
        require(stamp > history[-1]["timestamp"], "preview_checkpoint_order_invalid")
        history.append(
            {
                "timestamp": stamp,
                "positions": {
                    key: {**p, "strategy": str(p["strategy"]), "external": str(p["external"])}
                    for key, p in positions.items()
                },
            }
        )

    def spend(coin, amount):
        require(amount > 0, "preview_invalid_debit")
        external = min(spot[coin]["external"], amount)
        own = amount - external
        require(own <= spot[coin]["strategy"], "preview_insufficient_proven_funds")
        spot[coin]["external"] -= external
        spot[coin]["strategy"] -= own
        return own, external

    for row in sorted(rows, key=lambda r: (int(r["cTime"]), str(r["billId"]))):
        key, stamp, coin = str(row["billId"]), int(row["cTime"]), row["coin"]
        if key in consumed:
            continue
        amount, kind, group = quantity(row["size"]), row["businessType"], row["groupType"]
        detail = {
            "timestamp": stamp,
            "coin": coin,
            "bill_ids": [key],
            "type": kind,
            "account_amount": str(amount),
        }
        native_legs = [row]
        if kind in EXTERNAL_CREDITS:
            require(
                coin == "USDT" and group == "other" and amount > 0,
                "preview_reward_evidence_invalid",
            )
            require(amount % UNIT == 0, "preview_reward_precision_invalid")
            from .account_credits import rebate_coupon_link
            from .live_execution import LiveExecution

            detail.update(
                rebate_coupon_link(LiveExecution.interest_bill(row), execution.get("receipts", {}))
            )
            spot[coin]["external"] += amount
            detail.update(
                strategy_amount="0",
                external_amount=str(amount),
                classification="cash_coupon_outside_strategy",
            )
            rewards.append(deepcopy(detail))
        elif kind == "DEPOSIT":
            require(
                coin == "USDT" and group == "deposit" and amount > 0,
                "preview_deposit_evidence_invalid",
            )
            spot[coin]["external"] += amount
            detail.update(
                strategy_amount="0",
                external_amount=str(amount),
                classification="deposit_outside_strategy",
            )
        elif kind == "BATCH_INTEREST_USER_IN":
            require(group == "financial" and amount > 0, "preview_interest_evidence_invalid")
            end = stamp // HOUR * HOUR
            weights, products = effective_units(history, coin, end - HOUR, end)
            total = weights["strategy"] + weights["external"]
            require(total > 0 and len(products) == 1, "preview_interest_ownership_unverified")
            own = (amount * weights["strategy"] / total).quantize(UNIT, rounding=ROUND_DOWN)
            if coin == "USDT":
                from .account_movement_receipts import external_usdt_income_allowed

                if external_usdt_income_allowed(execution):
                    own = money(0)
            outside = amount - own
            spot[coin]["strategy"] += own
            spot[coin]["external"] += outside
            detail.update(
                strategy_amount=str(own),
                external_amount=str(outside),
                classification="native_interest_prior_full_hour",
                period_start=end - HOUR,
                period_end=end,
                strategy_units=str(weights["strategy"]),
                external_units=str(weights["external"]),
            )
            interest.append(deepcopy(detail))
        elif group == "transaction":
            order_id = str(row["bizOrderId"])
            legs = transactions[order_id]
            native_legs = legs
            require(
                len(legs) == 2 and {r["coin"] for r in legs} == {"USDT", "USDGO"},
                "preview_fx_legs_unverified",
            )
            debit = next(r for r in legs if r["coin"] == "USDT")
            credit = next(r for r in legs if r["coin"] == "USDGO")
            paid, received = -quantity(debit["size"]), quantity(credit["size"])
            require(
                paid > 0 and received > 0 and int(debit["cTime"]) == int(credit["cTime"]),
                "preview_fx_direction_unverified",
            )
            entry = evidence["orders"].get(order_id, {})
            order = entry.get("order", {})
            # Native transaction bills carry the trade ID, not the order ID.
            fills = [f for f in entry.get("fills", []) if str(f["id"]) == order_id]
            actual_order_id = str(order.get("orderId"))
            require(
                entry.get("native_trade_id") == order_id
                and order.get("symbol") == "USDGOUSDT"
                and order.get("side") == "buy"
                and order.get("status") == "filled"
                and bool(fills),
                "preview_fx_order_unverified",
            )
            require(len({f["id"] for f in fills}) == len(fills), "preview_duplicate_fill")
            gross, quote, base_fee, quote_fee = money(0), money(0), money(0), money(0)
            for fill in fills:
                require(
                    fill["order_id"] == actual_order_id
                    and fill["side"] == "buy"
                    and int(fill["timestamp"]) == stamp,
                    "preview_fx_fill_mismatch",
                )
                gross += quantity(fill["quantity"])
                quote += quantity(fill["quote"])
                for fee in fill["fees"]:
                    require(
                        fee["coin"] in {"USDT", "USDGO"} and quantity(fee["cost"]) >= 0,
                        "preview_third_currency_fee_unverified",
                    )
                    if fee["coin"] == "USDGO":
                        base_fee += quantity(fee["cost"])
                    else:
                        quote_fee += quantity(fee["cost"])
            require(
                quote + quote_fee == paid and gross - base_fee == received,
                "preview_fx_native_fill_conservation_failed",
            )
            own_paid, outside_paid = spend("USDT", paid)
            own_received = (received * own_paid / paid).quantize(UNIT, rounding=ROUND_DOWN)
            spot["USDGO"]["strategy"] += own_received
            spot["USDGO"]["external"] += received - own_received
            strategy_usdt_used += own_paid
            strategy_usdgo_from_fx += own_received
            consumed.update(str(r["billId"]) for r in legs)
            detail.update(
                coin="USDT→USDGO",
                type="manual_usdgo_buy",
                bill_ids=sorted(consumed & {str(r["billId"]) for r in legs}),
                order_id=actual_order_id,
                trade_id=order_id,
                strategy_paid_USDT=str(own_paid),
                external_paid_USDT=str(outside_paid),
                gross_USDGO=str(gross),
                fee_USDGO=str(base_fee),
                fee_USDT=str(quote_fee),
                net_USDGO=str(received),
                strategy_received_USDGO=str(own_received),
                external_received_USDGO=str(received - own_received),
                classification="ownership_preview_external_first_then_proportional_net_receipt",
            )
        elif kind == "SINGLE_USER_OUT":
            require(group == "financial" and amount < 0, "preview_subscription_bill_invalid")
            business_id = str(row["bizOrderId"])
            receipt = evidence["subscription_results"].get(business_id, {})
            require(receipt.get("result") == "success", "preview_subscription_result_unverified")
            candidates = [(hid, p) for hid, p in evidence["holdings"].items() if p["coin"] == coin]
            require(len(candidates) == 1, "preview_subscription_holding_ambiguous")
            hid, remote = candidates[0]
            if hid not in positions:
                positions[hid] = {
                    "coin": coin,
                    "product_id": remote["product_id"],
                    "strategy": money(0),
                    "external": money(0),
                }
            own, outside = spend(coin, -amount)
            positions[hid]["strategy"] += own
            positions[hid]["external"] += outside
            checkpoint(stamp)
            detail.update(
                strategy_amount=str(own),
                external_amount=str(outside),
                business_order_id=business_id,
                holding_id=hid,
                classification="observed_subscription_location_only",
            )
        else:
            raise ExecutionError("preview_unknown_account_movement")
        detail["native_balance_checks"] = []
        for leg in native_legs:
            reconstructed = sum(spot[leg["coin"]].values(), money(0))
            reported = quantity(leg["balance"])
            require(abs(reconstructed - reported) < UNIT, "preview_native_balance_sequence_failed")
            detail["native_balance_checks"].append(
                {
                    "coin": leg["coin"],
                    "native_balance": str(reported),
                    "reconstructed_balance": str(reconstructed),
                    "precision_difference": str(reconstructed - reported),
                }
            )
        for parts in spot.values():
            require(min(parts.values()) >= 0, "preview_negative_ownership")
        consumed.add(key)
        timeline.append(detail)

    wallet_checks, holding_checks = [], []
    overview = {r["coin"]: quantity(r["amount"]) for r in evidence["earn_overview"]}
    for coin in coins:
        actual = sum((quantity(v) for v in evidence["balances"][coin].values()), money(0))
        expected = sum(spot[coin].values(), money(0))
        account_credits = [
            row for row in timeline if row["coin"] == coin and row["type"] in EXTERNAL_CREDITS
        ]
        if coin == "USDT" and account_credits and actual != expected:
            # Match the existing native account-credit precision rule. Only a
            # sub-eight-decimal residue may remain outside; USDC stays exact.
            residual = actual - expected
            require(
                abs(residual) < UNIT * len(account_credits), "preview_wallet_conservation_failed"
            )
            account_credits[-1]["precision_residual"] = str(residual)
            spot[coin]["external"] += residual
            expected += residual
        require(actual == expected, "preview_wallet_conservation_failed")
        wallet_checks.append(
            {
                "coin": coin,
                "actual": str(actual),
                "strategy": str(spot[coin]["strategy"]),
                "outside_strategy": str(spot[coin]["external"]),
                "difference": str(actual - expected),
            }
        )
        held = sum(
            (quantity(p["amount"]) for p in evidence["holdings"].values() if p["coin"] == coin),
            money(0),
        )
        require(coin in overview and held == overview[coin], "preview_earn_overview_mismatch")
    require(set(positions) == set(evidence["holdings"]), "preview_unexplained_holding")
    for hid, p in positions.items():
        actual = evidence["holdings"][hid]
        require(
            p["coin"] == actual["coin"] and p["product_id"] == actual["product_id"],
            "preview_holding_identity_mismatch",
        )
        total = p["strategy"] + p["external"]
        require(total == quantity(actual["amount"]), "preview_holding_conservation_failed")
        holding_checks.append(
            {
                "holding_id": hid,
                "coin": p["coin"],
                "actual": str(total),
                "strategy": str(p["strategy"]),
                "outside_strategy": str(p["external"]),
                "difference": "0",
            }
        )
    require(execution == original, "preview_mutated_input")
    return {
        "version": "paused-account-movement-preview-v1",
        "status": "evidence_verified_preview_only",
        "account_id": execution["account_id"],
        "since": since,
        "observed_at": until,
        "policy": (
            "New rewards/deposits stay outside. Manual debits use external funds first; "
            "FX net receipts follow paid ownership, strategy rounded down to8decimals. "
            "Review assumption, not a committed ownership change."
        ),
        "cash_coupon_rewards": rewards,
        "cash_coupon_reward_total_USDT": str(
            sum((quantity(r["external_amount"]) for r in rewards), money(0))
        ),
        "timeline": timeline,
        "wallet_checks": wallet_checks,
        "holding_checks": holding_checks,
        "new_interest_allocations": interest,
        "original_strategy_USDT_used_by_manual_buy": str(strategy_usdt_used),
        "strategy_USDGO_net_receipt_from_manual_buy": str(strategy_usdgo_from_fx),
        "protected": {
            "initial": execution["book"]["initial"],
            "created_at": execution["created_at"],
            "fund_units": execution["book"]["fund_units"],
            "external_flows": execution["book"]["external_flows"],
            "trx_remaining_cost": str(book.cost),
            "trx_lots_sha256": hashlib.sha256(
                json.dumps(execution["book"]["lots"], sort_keys=True).encode()
            ).hexdigest(),
        },
        "formal_commit_ready": False,
        "resume_authorized": False,
        "ledger_writes": 0,
        "exchange_writes": 0,
    }


def preview(execution_state, evidence):
    """Return a deterministic proposal; neither input object is changed."""
    with localcontext() as context:
        context.prec = 60
        result = _preview(execution_state, evidence)
        result["review_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return result
