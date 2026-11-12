"""GET-proven external stablecoin movements; never submit an exchange operation.

An explicit account-bound policy selects external funds first. Every debit must
fit previously attributed external money. Native trading funds, TRX lots and
capital history are protected. A separately approved historical cash-tail repair
may exchange an existing external USDGO ownership slice for the swept cash claim.
"""

import json
from copy import deepcopy
from decimal import ROUND_DOWN

from .ledger import Ledger, money
from .live_broker import DAY, ExecutionError, identifier
from .live_execution import LiveExecution, digest
from .reconciliation_wait import EXTERNAL_MOVEMENT_EVIDENCE_PENDING


VERSION = "verified-external-stable-movements-v1"
POLICY_FILE = "external-stable-movements.json"
WITHDRAWAL_RECORDS = "v2/spot/wallet/withdrawal-records"
SAVINGS_MOVEMENTS = {"SINGLE_RELEASE_USER_IN", "SINGLE_USER_OUT"}
COINS = ("USDT", "USDGO", "USDC")
PAIRS = {
    "USDGOUSDT": ("USDGO", "USDT"),
    "USDCUSDT": ("USDC", "USDT"),
    "USDGOUSDC": ("USDGO", "USDC"),
}


def require(condition, code):
    if not condition:
        raise ExecutionError("external_movement_" + code)


def wait_for_evidence(condition):
    if not condition:
        raise ExecutionError(EXTERNAL_MOVEMENT_EVIDENCE_PENDING)


def _external_usdt_withdrawal(raw, records, now):
    """Verify a completed debit, including the fee, from two native sources."""
    oid = str(raw["bizOrderId"])
    require(raw["coin"] == "USDT" and raw["groupType"] == "withdraw", "withdrawal_scope")
    wait_for_evidence(bool(records))
    require(len(records) == 1, "withdrawal_record_ambiguous")
    record = records[0]
    require(
        str(record["orderId"]) == oid and record["coin"] == "USDT" and record["type"] == "withdraw",
        "withdrawal_identity",
    )
    wait_for_evidence(record["status"] != "pending")
    size, fee = money(raw["size"]), money(raw["fees"])
    created, updated, posted = int(record["cTime"]), int(record["uTime"]), int(raw["cTime"])
    wait_for_evidence(updated <= now)
    require(
        record["status"] == "success"
        and size < 0
        and fee <= 0
        and money(record["fee"]) == fee
        and money(record["size"]) == -(size + fee)
        and 0 < created <= min(updated, posted) <= max(updated, posted) <= now,
        "withdrawal_settlement_unverified",
    )
    return -(size + fee), -fee


def _spot_bill_matches(raw, quantity, fee, *, allow_truncation):
    """The wallet uses full fill precision even when a bill displays eight decimals."""
    signed_size, signed_fee = money(raw["size"]), money(raw["fees"])
    if not allow_truncation:
        return signed_size + signed_fee == quantity + fee
    unit = money("0.00000001")
    return all(
        actual == expected or actual == expected.quantize(unit, rounding=ROUND_DOWN)
        for actual, expected in ((signed_size, quantity), (signed_fee, fee))
    )


def read_policy(execution, now):
    path = execution.root.parents[1] / POLICY_FILE
    if not path.exists():
        return None
    require(path.stat().st_size <= 1048576, "policy_invalid")
    policy = json.loads(path.read_text())
    if policy.get("enabled") is False:
        return None
    validate_policy(execution.state, policy, now)
    return policy


def validate_policy(state, policy, now):
    require(
        policy.get("version") == VERSION
        and policy.get("enabled") is True
        and policy.get("account_id") == state["account_id"]
        and policy.get("ledger_created_at") == state["created_at"]
        and policy.get("scope") == "verified_external_stablecoins_only"
        and policy.get("external_first") is True
        and 0 < policy.get("approved_at", 0) <= now
        and policy.get("sha256") == digest({k: v for k, v in policy.items() if k != "sha256"}),
        "policy_invalid",
    )
    require(
        bool(policy.get("usdgo_holdings")) and bool(policy.get("usdc_product_id")),
        "policy_scope_missing",
    )
    if policy.get("usdt_external_withdrawals") is True:
        require(bool(state.get("stablecoin_policy")), "withdrawal_scope")


def _known(state, raw):
    prior = state.get("account_movement_bill_receipts", {}).get(str(raw["billId"]))
    if prior is not None:
        require(prior == digest(LiveExecution.interest_bill(raw)), "bill_changed")
        return True
    return False


def _native_known(state, raw):
    bid, oid = str(raw["billId"]), str(raw["bizOrderId"])
    return (
        any(
            bid in {str(v) for v in event.get("bill_ids", [])}
            for event in state.get("events", [])
            if event.get("kind") == "operation_settled"
        )
        or (
            raw.get("groupType") == "transaction"
            and any(
                key.startswith("spot:") and key.endswith(":" + oid) for key in state["receipts"]
            )
        )
        or bid in state.get("income_baseline_bills", {})
        or "income-bill:" + bid in state["receipts"]
        or "account-credit:" + bid in state["receipts"]
    )


def consumed_interest(state):
    """Receipt-backed consumption, including the reserved subset when present."""
    used, reserved = {}, {}
    for key, record in state.get("verified_external_movements", {}).items():
        require(
            record.get("classification") == "verified_external_movement"
            and state["receipts"].get("verified-external-movement:" + key) == digest(record),
            "interest_receipt_conflict",
        )
        sources = record.get("external_interest_sources", {})
        parts = record.get("reserved_interest_sources", {})
        require(set(parts) <= set(sources), "interest_reserved_conflict")
        require(
            sum((money(v) for v in parts.values()), money(0))
            == money(record.get("reserved_amount", 0)),
            "interest_reserved_conflict",
        )
        for source, amount in sources.items():
            amount, part = money(amount), money(parts.get(source, 0))
            require(0 <= part <= amount and amount > 0, "interest_source_conflict")
            used[source] = used.get(source, money(0)) + amount
            reserved[source] = reserved.get(source, money(0)) + part
    return used, reserved


def _sources(state, policy, now, amount, before):
    if not policy or amount <= 0:
        return {}
    from .external_income_reinvestment import available_sources

    chosen = {}
    for key, value in available_sources(state, policy, now):
        if state["income_accounting"]["allocations"][key]["timestamp"] > before:
            continue
        part = min(value, amount)
        if part:
            chosen[key] = str(part)
            amount -= part
    return chosen


def _update_usdt_counter(state, proof, observed, holding_id, product_id, stamp, bill_id, redeem):
    accounting = state["income_accounting"]
    anchor = accounting["usdt_product_counter_anchor"]
    active = anchor["holdings"]
    unchanged = set(observed) - {holding_id}
    require(
        anchor["observed_at"] < stamp
        and set(active) - {holding_id} == unchanged
        and (holding_id in active) == redeem
        and not any(
            r.get("businessType") == "BATCH_INTEREST_USER_IN"
            and int(r["cTime"]) > state["interest_checked_at"]
            and not _native_known(state, r)
            for r in proof["bills"]["USDT"]
        ),
        "usdt_counter_boundary_unverified",
    )
    require(
        all(
            active[key]["product_id"] == str(observed[key]["productId"])
            and observed[key]["interestCoin"] == "USDT"
            and money(active[key]["profit"]) == money(observed[key]["totalProfit"])
            for key in unchanged
        ),
        "usdt_counter_boundary_unverified",
    )
    closed = accounting.setdefault("usdt_closed_holding_counters", {})
    if redeem:
        counter = active.pop(holding_id)
        require(counter["product_id"] == product_id, "usdt_counter_product_changed")
        closed[holding_id] = {**counter, "closed_at": stamp, "source_bill_id": bill_id}
    else:
        previous = closed.get(holding_id, {"product_id": product_id, "profit": "0"})
        row = observed[holding_id]
        require(
            previous["product_id"] == product_id
            and previous.get("closed_at", 0) < stamp
            and row["interestCoin"] == "USDT"
            and money(previous["profit"]) == money(row["totalProfit"]),
            "usdt_opening_counter_unverified",
        )
        active[holding_id] = {"product_id": product_id, "profit": str(money(row["totalProfit"]))}


def _stage_external_usdt_savings(state, proof, amount, redeem, raw, native):
    """Require a unique holding delta and leave every strategy holding intact."""
    held = {key: row for key, row in state["holdings"].items() if row["coin"] == "USDT"}
    observed = {}
    for row in proof["raw_holdings"]:
        if row.get("productCoin") == "USDT":
            key = str(row["orderId"])
            require(key not in observed and money(row["holdAmount"]) > 0, "usdt_holding_invalid")
            observed[key] = row
    deltas = {}
    for key in held.keys() | observed.keys():
        prior, current = held.get(key), observed.get(key)
        if prior and current:
            require(prior["product_id"] == str(current["productId"]), "usdt_holding_changed")
        before = (
            sum((money(prior[k]) for k in ("principal", "interest", "unassigned_amount")), money(0))
            if prior
            else money(0)
        )
        after = money(current["holdAmount"]) if current else money(0)
        if after != before:
            deltas[key] = (before, after)
    require(len(deltas) == 1, "usdt_savings_source_ambiguous")
    holding_id, (before, after) = next(iter(deltas.items()))
    prior = held.get(holding_id)
    product_id = prior["product_id"] if prior else str(observed[holding_id]["productId"])
    (strategy_product,) = (
        r["product_id"] for r in state["stablecoin_policy"]["products"] if r["coin"] == "USDT"
    )
    require(
        product_id != strategy_product
        and after - before == (-amount if redeem else amount)
        and (prior is not None or (not redeem and holding_id not in state["holdings"]))
        and (
            prior is None
            or all(
                money(prior.get(k, 0)) == 0 for k in ("principal", "interest", "reserved_amount")
            )
        )
        and not any(
            r.get("coin") == "USDT" and r.get("savings_record_id") == str(native["orderId"])
            for r in state.get("verified_external_movements", {}).values()
        ),
        "usdt_savings_ownership_unverified",
    )
    stamp = int(native["ts"])
    if not before or not after:
        _update_usdt_counter(
            state, proof, observed, holding_id, product_id, stamp, str(raw["billId"]), redeem
        )
    if prior is None:
        prior = state["holdings"][holding_id] = {
            "coin": "USDT",
            "product_id": product_id,
            "principal": "0",
            "interest": "0",
            "unassigned_amount": "0",
            "owned_since": stamp,
            "lots": {},
            "subscription_ids": [],
        }
    prior["unassigned_amount"] = str(after)
    return holding_id, product_id


def stage(original, proof, policy, *, income_policy=None):  # noqa: C901
    """Pure staging. Caller must still run native income/holding/wallet checks."""
    now = proof["observed_at"]
    validate_policy(original, policy, now)
    require(
        proof.get("account_id") == original["account_id"]
        and proof.get("account_binding_verified") is True,
        "account_binding",
    )
    require(not original.get("operation") and not original.get("error"), "operation_active")
    require(
        not original.get("flow")
        or original["flow"]["status"] in {"completed", "expired", "failed"},
        "flow_active",
    )
    require(
        original.get("income_accounting", {}).get("status") == "active"
        and not original.get("pending_income"),
        "income_unresolved",
    )
    since = original.get("external_movements_checked_at", original["interest_checked_at"])
    state = deepcopy(original)
    allow_withdrawals = policy.get("usdt_external_withdrawals") is True
    native_usdc_income = money(0)
    offsets = {coin: money(state["unassigned_offsets"].get(coin, "0")) for coin in COINS}
    offsets["USDC"] = money(state.get("external_usdc", {}).get("spot", "0"))
    usdc_saved = money(state.get("external_usdc", {}).get("savings", "0"))
    relevant, seen = [], {}
    for coin in COINS:
        for raw in proof["bills"][coin]:
            require(raw.get("coin") == coin, "bill_coin")
            bid = str(raw["billId"])
            fingerprint = digest(LiveExecution.interest_bill(raw))
            require(bid not in seen or seen[bid] == fingerprint, "duplicate_conflict")
            if bid in seen:
                continue
            seen[bid] = fingerprint
            cutoff = since if coin == "USDC" else original["interest_checked_at"]
            if _known(original, raw) or _native_known(original, raw) or int(raw["cTime"]) <= cutoff:
                continue
            require(int(raw["cTime"]) <= now, "future_bill")
            kind = raw.get("businessType")
            if kind in SAVINGS_MOVEMENTS or (allow_withdrawals and kind == "WITHDRAW"):
                relevant.append(raw)
            elif raw.get("groupType") == "transaction":
                trade_id = str(raw["bizOrderId"])
                if any(
                    key.startswith("spot:") and key.endswith(":" + trade_id)
                    for key in original["receipts"]
                ):
                    continue
                relevant.append(raw)
            elif coin == "USDC":
                require(
                    kind == "BATCH_INTEREST_USER_IN"
                    and raw.get("groupType") == "financial"
                    and money(raw["size"]) > 0
                    and money(raw["fees"]) == 0,
                    "unknown_usdc_credit",
                )
                if allow_withdrawals:
                    # The native allocator applies the actual strategy/external split.
                    native_usdc_income += money(raw["size"])
                else:
                    relevant.append(raw)
    if not relevant:
        return state, []
    require(len(relevant) <= 3000, "read_limit")
    usdt_savings = [
        r
        for r in relevant
        if r["coin"] == "USDT"
        and r["businessType"] in {"SINGLE_RELEASE_USER_IN", "SINGLE_USER_OUT"}
    ]
    if usdt_savings:
        require(
            policy.get("usdt_external_savings") is True and original.get("stablecoin_policy"),
            "savings_scope",
        )
        require(len(usdt_savings) == 1, "usdt_savings_source_ambiguous")
    new_rows = []
    sources_left = deepcopy(original)
    settled = {
        str(bid)
        for event in original.get("events", [])
        if event.get("kind") == "operation_settled"
        for bid in event.get("bill_ids", [])
    }
    relevant = [r for r in relevant if str(r["billId"]) not in settled]
    groups = {}
    for raw in relevant:
        if raw.get("groupType") == "transaction":
            key = (int(raw["cTime"]), "trade", str(raw["bizOrderId"]))
        else:
            key = (int(raw["cTime"]), raw["businessType"], str(raw["bizOrderId"]))
        groups.setdefault(key, []).append(raw)
    tail_paid = money(0)
    tail = {} if allow_withdrawals else policy.get("historical_cash_tail", {})
    tail_done = bool(state.get("external_tail_reimbursements", {}).get(tail.get("trade_id")))
    for (stamp, kind, oid), rows in sorted(groups.items()):
        consumed = money(0)
        record = {
            "timestamp": stamp,
            "recorded_at": now,
            "classification": "verified_external_movement",
            "bill_ids": sorted(str(r["billId"]) for r in rows),
            "kind": kind,
            "business_id": oid,
            "external_interest_sources": {},
            "exchange_writes": 0,
        }
        if kind == "trade":
            fills = [r for r in proof["fills"] if str(r["tradeId"]) == oid]
            if allow_withdrawals:
                wait_for_evidence(bool(fills))
            require(len(fills) == 1, "fill_missing_or_duplicated")
            fill = fills[0]
            require(
                fill["symbol"] in PAIRS and fill["side"] in {"buy", "sell"}, "unsupported_trade"
            )
            order_id = str(fill["orderId"])
            orders = proof["orders"].get(order_id, [])
            if allow_withdrawals:
                wait_for_evidence(bool(orders))
            require(len(orders) == 1, "order_identity")
            order = orders[0]
            all_fills = [r for r in proof["fills"] if str(r["orderId"]) == order_id]
            require(
                str(order.get("orderId")) == order_id
                and order.get("status") == "filled"
                and order.get("symbol") == fill["symbol"]
                and order.get("side") == fill["side"]
                and money(order["baseVolume"])
                == sum((money(r["size"]) for r in all_fills), money(0))
                and money(order["quoteVolume"])
                == sum((money(r["amount"]) for r in all_fills), money(0)),
                "order_fills_incomplete",
            )
            base, quote = PAIRS[fill["symbol"]]
            sign = 1 if fill["side"] == "buy" else -1
            quantities = {base: sign * money(fill["size"]), quote: -sign * money(fill["amount"])}
            fees_by_coin = dict.fromkeys(quantities, money(0))
            require(
                min(money(fill["size"]), money(fill["amount"])) > 0 and int(fill["cTime"]) == stamp,
                "fill_values",
            )
            fees = fill["feeDetail"]
            fees = [fees] if isinstance(fees, dict) else fees
            for fee in fees:
                require(
                    fee["feeCoin"] in quantities
                    and money(fee["totalFee"]) <= 0
                    and money(fee.get("totalDeductionFee") or "0") == 0,
                    "fee_unverified",
                )
                fees_by_coin[fee["feeCoin"]] += money(fee["totalFee"])
            delta = {coin: quantity + fees_by_coin[coin] for coin, quantity in quantities.items()}
            if allow_withdrawals:
                wait_for_evidence(len(rows) >= 2)
            require(
                len(rows) == 2 and {r["coin"] for r in rows} == {base, quote}, "trade_bill_pair"
            )
            for raw in rows:
                coin = raw["coin"]
                require(
                    raw.get("businessType") in {"BUY", "SELL"}
                    and _spot_bill_matches(
                        raw,
                        quantities[coin],
                        fees_by_coin[coin],
                        allow_truncation=allow_withdrawals and fill["symbol"] == "USDCUSDT",
                    ),
                    "trade_bill_delta",
                )
                offsets[coin] += delta[coin]
                if coin == "USDGO" and delta[coin] < 0:
                    consumed += -delta[coin]
            if offsets["USDT"] < 0:
                require(
                    not tail_done
                    and oid == tail.get("trade_id")
                    and -offsets["USDT"] == money(tail.get("amount_USDT", 0))
                    and -offsets["USDT"] <= money("0.0001"),
                    "strategy_cash_would_be_used",
                )
                tail_paid = -offsets["USDT"]
                offsets["USDT"] = money(0)
            record.update(
                order_id=order_id,
                symbol=fill["symbol"],
                trade_id=oid,
                deltas={k: str(v) for k, v in delta.items()},
            )
        elif kind == "WITHDRAW":
            require(len(rows) == 1, "withdrawal_bill_ambiguous")
            records = proof.get("withdrawals", {}).get(oid, [])
            amount, fee = _external_usdt_withdrawal(rows[0], records, now)
            require(amount <= offsets["USDT"], "strategy_cash_would_be_used")
            offsets["USDT"] -= amount
            record.update(coin="USDT", amount=str(amount), fee=str(fee))
        elif kind in SAVINGS_MOVEMENTS:
            require(len(rows) == 1, "savings_bill_duplicate")
            raw = rows[0]
            coin = raw["coin"]
            require(
                coin in {"USDGO", "USDC", "USDT"}
                and raw.get("groupType") == "financial"
                and money(raw["fees"]) == 0,
                "savings_scope",
            )
            signed = money(raw["size"])
            amount = abs(signed)
            redeem = kind == "SINGLE_RELEASE_USER_IN"
            require(amount > 0 and (signed > 0) == redeem, "savings_direction")
            result = proof["results"].get(oid, {})
            require(result.get("result") == "success", "savings_result_missing")
            records = proof["records"][coin]["redeem" if redeem else "subscribe"]
            matches = [
                r
                for r in records
                if r.get("coinName") == coin
                and r.get("settleCoinName") == coin
                and r.get("productType") == "flexible"
                and r.get("orderType") == ("redeem" if redeem else "subscribe")
                and money(r["amount"]) == amount
                and abs(stamp - int(r["ts"])) <= 120000
            ]
            require(len(matches) == 1, "savings_record_ambiguous")
            if coin == "USDT":
                hid, product_id = _stage_external_usdt_savings(
                    state, proof, amount, redeem, raw, matches[0]
                )
                record.update(holding_id=hid, product_id=product_id)
            elif coin == "USDGO":
                choices = [
                    (hid, pos)
                    for hid, pos in state["holdings"].items()
                    if pos["coin"] == coin
                    and policy["usdgo_holdings"].get(hid) == pos["product_id"]
                ]
                require(len(choices) == 1, "savings_source_ambiguous")
                hid, pos = choices[0]
                outside = money(pos["unassigned_amount"]) - (amount if redeem else -amount)
                require(outside >= 0, "strategy_holding_would_be_used")
                pos["unassigned_amount"] = str(outside)
                if "reserved_amount" in pos:
                    # Keep the original reserve allocation immutable. Only its
                    # present USDGO location changes; USDC remains outside.
                    reserved = money(pos["reserved_amount"])
                    part = min(reserved, amount) if redeem else money(0)
                    require(redeem or reserved <= outside, "reserve_location_invalid")
                    pos["reserved_amount"] = str(reserved - part)
                    record["reserved_USDGO_redeemed"] = str(part)
                if not redeem:
                    consumed += amount
                record.update(holding_id=hid, product_id=pos["product_id"])
            else:
                usdc_saved += -amount if redeem else amount
                require(usdc_saved >= 0, "usdc_savings_negative")
                if allow_withdrawals:
                    choices = [
                        (hid, pos)
                        for hid, pos in state["holdings"].items()
                        if pos["coin"] == "USDC" and pos["product_id"] == policy["usdc_product_id"]
                    ]
                    require(len(choices) == 1, "usdc_savings_source_ambiguous")
                    hid, position = choices[0]
                    position["unassigned_amount"] = str(
                        money(position["unassigned_amount"]) + (-amount if redeem else amount)
                    )
                    require(
                        money(position["unassigned_amount"]) == usdc_saved, "usdc_holding_mismatch"
                    )
                    record.update(holding_id=hid, product_id=position["product_id"])
            offsets[coin] += amount if redeem else -amount
            record.update(
                coin=coin, amount=str(amount), savings_record_id=str(matches[0]["orderId"])
            )
        else:
            raw = rows[0]
            require(
                len(rows) == 1 and raw["coin"] == "USDC" and usdc_saved > 0, "usdc_income_unbacked"
            )
            offsets["USDC"] += money(raw["size"])
            record.update(coin="USDC", amount=raw["size"], strategy_income_changed=False)
        require(all(v >= 0 for v in offsets.values()), "external_funds_insufficient")
        if consumed:
            record["external_interest_sources"] = _sources(
                sources_left, income_policy, now, consumed, stamp
            )
            if original.get("usdgo_reserve"):
                from .external_income_ownership import reserved_for_selection
                from .external_income_reinvestment import available_sources

                record["reserved_interest_sources"] = reserved_for_selection(
                    sources_left,
                    record["external_interest_sources"],
                    available_sources(sources_left, income_policy, now),
                )
                record["reserved_amount"] = str(
                    sum((money(v) for v in record["reserved_interest_sources"].values()), money(0))
                )
                if kind == "SINGLE_USER_OUT" and record.get("coin") == "USDGO":
                    pos["reserved_amount"] = str(
                        money(pos.get("reserved_amount", 0)) + money(record["reserved_amount"])
                    )
        if kind in SAVINGS_MOVEMENTS and (
            record.get("coin") in {"USDGO", "USDT"} or allow_withdrawals
        ):
            from .income_allocation import ownership

            checkpoints = state["income_accounting"]["checkpoints"]
            checkpoint_at = int(matches[0]["ts"])
            require(checkpoint_at > checkpoints[-1]["timestamp"], "checkpoint_order_invalid")
            checkpoints.append(
                {
                    "timestamp": checkpoint_at,
                    "recorded_at": now,
                    "positions": ownership(state["holdings"]),
                    "source_bill_id": str(raw["billId"]),
                }
            )
        key = digest(record["bill_ids"])
        state.setdefault("verified_external_movements", {})[key] = record
        state["receipts"]["verified-external-movement:" + key] = digest(record)
        sources_left.setdefault("verified_external_movements", {})[key] = record
        sources_left["receipts"]["verified-external-movement:" + key] = digest(record)
        for raw in rows:
            state.setdefault("account_movement_bill_receipts", {})[str(raw["billId"])] = digest(
                LiveExecution.interest_bill(raw)
            )
        state["events"].append({"kind": "verified_external_movement_recorded", **record})
        new_rows.append(record)
    if not new_rows:
        return state, []
    if tail_paid:
        hid = tail.get("reimbursement_holding_id")
        pos = state["holdings"].get(hid, {})
        require(
            pos.get("coin") == "USDGO"
            and policy["usdgo_holdings"].get(hid) == pos.get("product_id"),
            "tail_source_identity",
        )
        mark = money(proof["markets"]["USDGOUSDT"]["bid"])
        require(mark > 0, "tail_valuation_missing")
        quantity = (tail_paid / mark).quantize(money("0.000000000000000001"))
        require(
            quantity > 0 and quantity <= money(pos["unassigned_amount"]),
            "tail_compensation_unavailable",
        )
        book = Ledger.restore(state["book"])
        require(tail_paid <= book.free_cash, "tail_cash_claim_invalid")
        book.cash -= tail_paid
        book.usdgo += quantity
        book.usdgo_cost += tail_paid
        book.locked["USDGO"] += quantity
        pos["principal"] = str(money(pos["principal"]) + quantity)
        pos["unassigned_amount"] = str(money(pos["unassigned_amount"]) - quantity)
        record = {
            "trade_id": tail["trade_id"],
            "amount_USDT": str(tail_paid),
            "reimbursed_USDGO": str(quantity),
            "USDGO_mark": str(mark),
            "holding_id": hid,
            "timestamp": now,
            "classification": "external_operation_in_kind_reimbursement",
            "capital_changed": False,
            "exchange_writes": 0,
        }
        key = "external-tail-reimbursement:" + tail["trade_id"]
        book.events.add(key)
        state["book"] = book.state()
        state.setdefault("external_tail_reimbursements", {})[tail["trade_id"]] = record
        state["receipts"][key] = digest(record)
        state["events"].append({"kind": "external_tail_reimbursed", **record})
        from .income_allocation import ownership

        state["income_accounting"]["checkpoints"].append(
            {
                "timestamp": now,
                "recorded_at": now,
                "positions": ownership(state["holdings"]),
                "source_bill_id": key,
            }
        )
    # The external proof covers only the remainder after native owned locations.
    native = Ledger.restore(state["book"])
    require(
        money(proof["balances"]["USDC"]["total"])
        == offsets["USDC"] + native.usdc - native.usdc_locked + native_usdc_income,
        "usdc_wallet_mismatch",
    )
    usdc_positions = [r for r in proof["raw_holdings"] if r.get("productCoin") == "USDC"]
    require(
        all(str(r["productId"]) == policy["usdc_product_id"] for r in usdc_positions)
        and sum((money(r["holdAmount"]) for r in usdc_positions), money(0))
        == usdc_saved + native.usdc_locked,
        "usdc_holding_mismatch",
    )
    if state.get("stablecoin_policy"):
        holdings = state["holdings"]
        observed = {str(r["orderId"]): money(r["holdAmount"]) for r in usdc_positions}
        for key in set(observed) | {k for k, p in holdings.items() if p["coin"] == "USDC"}:
            position = holdings.setdefault(
                key,
                {
                    "coin": "USDC",
                    "product_id": policy["usdc_product_id"],
                    "principal": "0",
                    "interest": "0",
                    "subscription_ids": [],
                    "lots": {},
                    "owned_since": now,
                    "unassigned_amount": "0",
                },
            )
            owned = money(position["principal"]) + money(position["interest"])
            require(observed.get(key, money(0)) >= owned, "usdc_owned_holding_changed")
            if allow_withdrawals:
                require(
                    observed.get(key, money(0)) == owned + money(position["unassigned_amount"]),
                    "usdc_holding_location_unverified",
                )
            position["unassigned_amount"] = str(observed.get(key, money(0)) - owned)
        state["unassigned_offsets"]["USDC"] = str(offsets["USDC"])
    state["external_usdc"] = {
        "spot": str(offsets["USDC"]),
        "savings": str(usdc_saved),
        "product_id": policy["usdc_product_id"],
        "outside_strategy": True,
    }
    for coin in ["USDT", "USDGO"]:
        state["unassigned_offsets"][coin] = str(offsets[coin])
    state["external_movements_checked_at"] = now
    return state, new_rows


def collect(execution, policy, now):
    """Use the existing GET-only/read-admitted broker; never instantiate a runtime."""
    broker, state = execution.broker, execution.state
    require(broker.account_id() == state["account_id"], "account_binding")
    since = min(state["interest_checked_at"], state.get("external_movements_checked_at", now))
    require(0 <= now - since < 89 * DAY, "history_window")
    start = max(state["created_at"], since - 1)
    bills = {
        c: broker.bills(
            c,
            start if c == "USDC" else max(state["created_at"], state["interest_checked_at"] - 1),
            now,
        )
        for c in COINS
    }
    unknown = [
        r
        for rows in bills.values()
        for r in rows
        if int(r["cTime"]) > since and not _known(state, r) and not _native_known(state, r)
    ]
    withdrawal_ids = (
        {identifier(r["bizOrderId"]) for r in unknown if r.get("businessType") == "WITHDRAW"}
        if policy.get("usdt_external_withdrawals") is True
        else set()
    )
    if not withdrawal_ids and not any(
        r.get("businessType") in SAVINGS_MOVEMENTS
        or r.get("groupType") == "transaction"
        or r.get("coin") == "USDC"
        for r in unknown
    ):
        return None
    require(len(unknown) <= 3000, "read_limit")
    manual_ids = {str(r["bizOrderId"]) for r in unknown if r.get("groupType") == "transaction"}
    fills = (
        broker.pages(
            "v2/spot/trade/fills",
            {"startTime": str(start), "endTime": str(now)},
            cursor_key="tradeId",
        )
        if manual_ids
        else []
    )
    orders = {
        oid: broker.get("v2/spot/trade/orderInfo", {"orderId": oid})
        for oid in sorted(
            {identifier(r["orderId"]) for r in fills if str(r["tradeId"]) in manual_ids}
        )
    }
    savings_coins = {
        r["coin"]
        for r in unknown
        if r.get("businessType") in SAVINGS_MOVEMENTS
        and (r["coin"] != "USDT" or policy.get("usdt_external_savings") is True)
    }
    records = {
        c: {
            kind: broker.savings_records(c, start, now, order_type=kind)
            for kind in ["subscribe", "redeem"]
        }
        for c in savings_coins
    }
    results = {
        str(r["bizOrderId"]): broker.result(
            "subscribe" if r["businessType"] == "SINGLE_USER_OUT" else "redeem",
            str(r["bizOrderId"]),
        )
        for r in unknown
        if r.get("businessType") in SAVINGS_MOVEMENTS
    }
    withdrawals = {
        oid: broker.get(WITHDRAWAL_RECORDS, {"orderId": oid}) for oid in sorted(withdrawal_ids)
    }
    raw_balances = broker.get("v2/spot/account/assets", {"assetType": "all"})
    balances = {
        r["coin"]: {
            "total": str(sum((money(r[k]) for k in ["available", "frozen", "locked"]), money(0)))
        }
        for r in raw_balances
        if r["coin"] in COINS
    }
    raw_holdings = broker.pages(
        "v2/earn/savings/assets",
        {
            "periodType": "flexible",
            "startTime": str(state["created_at"] - 1000),
            "endTime": str(now),
        },
        list_key="resultList",
        cursor_key="orderId",
    )
    overview = broker.get("v2/earn/account/assets")
    require(
        sum(
            (money(r["holdAmount"]) for r in raw_holdings if r.get("productCoin") == "USDC"),
            money(0),
        )
        == sum((money(r["amount"]) for r in overview if r.get("coin") == "USDC"), money(0)),
        "usdc_overview_incomplete",
    )
    return {
        "account_id": state["account_id"],
        "account_binding_verified": True,
        "observed_at": now,
        "bills": bills,
        "fills": fills,
        "orders": orders,
        "records": records,
        "results": results,
        "withdrawals": withdrawals,
        "balances": balances,
        "raw_holdings": raw_holdings,
        "markets": {"USDGOUSDT": broker.market("USDGOUSDT", now)},
    }


def _observe(execution, now):
    policy = read_policy(execution, now)
    if policy is None:
        return None
    proof = collect(execution, policy, now)
    if proof is None:
        return None
    from .external_income_reinvestment import read_policy as income_policy

    staged, records = stage(
        execution.state, proof, policy, income_policy=income_policy(execution, now)
    )
    if not records:
        return None
    work = object.__new__(LiveExecution)
    work.root, work.account_id, work.broker, work.state = (
        execution.root,
        execution.account_id,
        execution.broker,
        staged,
    )
    book, positions, receipts, rows, accounting = work.income_candidate(
        Ledger.restore(staged["book"]), staged["holdings"], staged["receipts"], now
    )
    balances = execution.broker.balances()
    work.check_wallet(balances, book, rows)
    work.commit_candidate(book, positions, receipts, rows, now, accounting)
    if policy.get("usdt_external_withdrawals") is True:
        wait_for_evidence(
            not work.state.get("pending_income")
            and work.state["income_accounting"]["status"] == "active"
        )
    require(
        not work.state.get("pending_income")
        and work.state["income_accounting"]["status"] == "active",
        "income_unresolved",
    )
    return work.state, balances


def observe(execution, now):
    try:
        return _observe(execution, now)
    except ExecutionError:
        raise
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError):
        raise ExecutionError("external_movement_evidence_invalid") from None


def observe_due(execution, now):
    """Hourly external-only metadata refresh; no failure can admit trading funds."""
    checked = execution.state.get(
        "external_metadata_attempted_at", execution.state.get("external_movements_checked_at", now)
    )
    if (
        not execution.state.get("external_usdc")
        or now - checked < 3600000
        or execution.state.get("operation")
        or (execution.state.get("flow") or {}).get("status", "completed")
        not in {"completed", "expired", "failed"}
    ):
        return None
    execution.state["external_metadata_attempted_at"] = now
    return observe(execution, now)
