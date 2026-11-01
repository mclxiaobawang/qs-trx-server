"""Recognize proven USDGO spot-to-selected-Savings movements; never submit one.

Mixed ownership is inferred automatically only for a complete spot sweep. Native
bill, result, product, holding overview and every wallet must conserve exactly.
Unknown trades, transfers, redemptions and ambiguous mixed partials stay blocked.
"""

from copy import deepcopy

from .account_movement_preview import preview, require
from .account_movement_reconcile import VERSION_ID
from .account_movement_reconcile import candidate as replay
from .ledger import money
from .live_execution import digest


def candidate(original, evidence, external_policy=None):  # noqa: C901 - atomic evidence checks
    """Pure all-or-nothing accounting; preserves original ownership and history."""
    operation = original.get("operation")
    if operation:
        require(
            operation.get("status") == "prepared"
            and not operation.get("exchange_id")
            and not operation.get("fills")
            and not operation.get("bill_ids"),
            "location_operation_in_flight",
        )
    since = original["interest_checked_at"]
    moves = [
        row
        for rows in evidence["bills"].values()
        for row in rows
        if since < int(row["cTime"]) <= evidence["observed_at_ms"]
        and row.get("businessType") == "SINGLE_USER_OUT"
    ]
    if not moves:
        return deepcopy(original), []
    require(
        all(row.get("coin") == "USDGO" and row.get("groupType") == "financial" for row in moves),
        "location_scope_unverified",
    )
    work = deepcopy(original)
    work["operation"] = None
    flow = work.get("flow")
    if flow and flow["status"] not in {"completed", "expired", "failed"}:
        require(
            flow.get("index") == 0
            and not flow.get("outputs")
            and len(flow.get("steps", [])) == 1
            and flow["steps"][0].get("kind") == "subscribe",
            "location_active_financial_flow",
        )
        work["flow"] = None  # Pure replay gate; restored unchanged below.
    omitted = {}
    for hid, pos in list(work["holdings"].items()):
        if (
            hid not in evidence["holdings"]
            and sum(
                (money(pos[k]) for k in ["principal", "interest", "unassigned_amount"]), money(0)
            )
            == 0
        ):
            omitted[hid] = work["holdings"].pop(hid)
    proof = deepcopy(evidence)
    overview_coins = [row["coin"] for row in proof["earn_overview"]]
    require(len(overview_coins) == len(set(overview_coins)), "location_overview_duplicate_coin")
    for coin in ["USDT", "USDGO", "TRX"]:
        if coin not in overview_coins:
            require(
                not any(
                    v["coin"] == coin and money(v["amount"]) for v in proof["holdings"].values()
                ),
                "location_overview_incomplete",
            )
            proof["earn_overview"].append({"coin": coin, "amount": "0"})
    proof["execution_state"] = deepcopy(work)
    proof["strategy_state"] = {"desired": "paused"}  # No runtime state is changed.
    review = preview(work, proof)
    manual = [
        row
        for row in review["timeline"]
        if row["type"] not in {"BATCH_INTEREST_USER_IN", "CASH_COUPON_ACTIVITY_USER_IN"}
    ]
    require(
        manual
        and all(row["type"] == "SINGLE_USER_OUT" and row["coin"] == "USDGO" for row in manual),
        "location_other_movement_requires_review",
    )
    selected = {v["product_id"] for v in original["products"] if v["coin"] == "USDGO"}
    for row in manual:
        hid = row["holding_id"]
        require(
            hid in original["holdings"] and original["holdings"][hid]["product_id"] in selected,
            "location_selected_existing_product_required",
        )
        # A mixed partial cannot establish which owner's coins were selected.
        require(
            not (money(row["strategy_amount"]) and money(row["external_amount"]))
            or all(money(v["native_balance"]) == 0 for v in row["native_balance_checks"]),
            "location_ambiguous_mixed_partial",
        )
    key = "verified-savings-location:" + digest(sorted(row["bill_ids"][0] for row in manual))
    policy = {
        "version": VERSION_ID,
        "account_id": original["account_id"],
        "reconciliation_id": key,
        "baseline_sha256": digest(work),
        "approved_at": evidence["observed_at_ms"],
        "external_first": True,
        "usdt_outside_strategy": True,
        "manual_bill_ids": [bid for row in manual for bid in row["bill_ids"]],
    }
    result, _ = replay(work, proof, policy)
    result["flow"] = deepcopy(original.get("flow"))
    result["operation"] = deepcopy(operation)
    current = result.get("flow")
    if (
        current
        and current["status"] not in {"completed", "expired", "failed"}
        and current["steps"][0].get("coin") == "USDGO"
    ):
        # The moved funds are already subscribed. Cancel only a proven
        # unsubmitted, single-step idle subscription, never an accepted op.
        current["status"] = "expired"
        current["expires_at"] = min(current["expires_at"], evidence["observed_at_ms"])
        result["operation"] = None
        result["events"].append(
            {
                "kind": "earn_flow_superseded_by_verified_location",
                "timestamp": evidence["observed_at_ms"],
                "flow_id": current["id"],
            }
        )
    result["holdings"].update(omitted)
    result["account_movement_reconciliations"] = deepcopy(
        original.get("account_movement_reconciliations", {})
    )
    if "outside_usdt_income_policy" in original:
        result["outside_usdt_income_policy"] = deepcopy(original["outside_usdt_income_policy"])
    # Keep general movement history unchanged; this narrower receipt has its own
    # journal and cannot authorize an ownership change or any exchange request.
    result["events"] = [
        event
        for event in result["events"]
        if not (
            event.get("kind") == "account_movement_reconciliation_completed"
            and event.get("reconciliation_id") == key
        )
    ]
    records = result.setdefault("verified_savings_location_movements", {})
    available = []
    if external_policy:
        from .external_income_reinvestment import available_sources

        source_state = deepcopy(result)
        # Enumerate verified unspent income independently of its new location.
        # The real wallet/holding conservation was proved by the replay above.
        source_state["unassigned_offsets"]["USDGO"] = str(
            sum(
                (
                    money(v["external_amount"])
                    for v in result["income_accounting"]["allocations"].values()
                    if v.get("coin") == "USDGO"
                ),
                money(0),
            )
        )
        available = [
            (key, money(value))
            for key, value in available_sources(
                source_state, external_policy, evidence["observed_at_ms"]
            )
        ]
    for row in manual:
        oid = row["business_order_id"]
        require(oid not in records, "location_order_already_recorded")
        outside = money(row["external_amount"])
        sources = {}
        for index, (source, value) in enumerate(available):
            if result["income_accounting"]["allocations"][source]["timestamp"] > row["timestamp"]:
                continue
            part = min(outside, value)
            if part:
                sources[source] = str(part)
                outside -= part
                available[index] = (source, value - part)
        record = {
            "classification": "verified_savings_location_only",
            "coin": "USDGO",
            "business_order_id": oid,
            "bill_ids": row["bill_ids"],
            "holding_id": row["holding_id"],
            "product_id": original["holdings"][row["holding_id"]]["product_id"],
            "timestamp": row["timestamp"],
            "recorded_at": evidence["observed_at_ms"],
            "strategy_amount": row["strategy_amount"],
            "external_amount": row["external_amount"],
            "external_interest_sources": sources,
            "exchange_writes": 0,
            "capital_changed": False,
            "ownership_changed": False,
        }
        records[oid] = record
        result["receipts"]["observed-savings-subscription:" + oid] = digest(record)
        result["events"].append(
            {
                "kind": "verified_savings_location_recorded",
                "timestamp": evidence["observed_at_ms"],
                **record,
            }
        )
    return result, [deepcopy(records[row["business_order_id"]]) for row in manual]


def observe(execution, now):
    """Bounded GET-only fallback after a genuine wallet/holding mismatch."""
    state, broker = execution.state, execution.broker
    if state.get("income_accounting", {}).get("status") != "active" or not any(
        pos.get("coin") == "USDGO" for pos in state.get("holdings", {}).values()
    ):
        return None
    require(broker.account_id() == execution.account_id, "location_account_binding_changed")
    since = state.get("interest_checked_at", state["created_at"])
    bills = {
        coin: broker.bills(coin, max(state["created_at"], since - 1), now)
        for coin in ["USDT", "USDGO", "TRX"]
    }
    moves = [
        row
        for rows in bills.values()
        for row in rows
        if since < int(row["cTime"]) <= now and row.get("businessType") == "SINGLE_USER_OUT"
    ]
    if not moves:
        return None
    require(len(moves) <= 10, "location_movement_read_limit")
    results = {
        str(row["bizOrderId"]): broker.result("subscribe", str(row["bizOrderId"])) for row in moves
    }
    holdings = broker.holdings(state["created_at"] - 1000, now)
    overview = broker.get("v2/earn/account/assets")
    balances = broker.balances()
    evidence = {
        "account_id": execution.account_id,
        "account_binding_verified": True,
        "observed_at_ms": now,
        "strategy_state": {"desired": "paused"},
        "execution_state": state,
        "bills": bills,
        "holdings": holdings,
        "earn_overview": overview,
        "balances": balances,
        "orders": {},
        "subscription_results": results,
        "markets": {"USDGOUSDT": broker.market("USDGOUSDT", now)},
    }
    from .external_income_reinvestment import read_policy

    result, records = candidate(state, evidence, read_policy(execution, now))
    return (result, balances) if records else None
