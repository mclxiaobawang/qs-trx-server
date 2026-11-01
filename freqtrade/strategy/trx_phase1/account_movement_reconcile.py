"""Confirmed movement replay. Builds one candidate; persistence is separately guarded."""

import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from decimal import localcontext
from pathlib import Path

from .account_movement_preview import preview, require
from .income_allocation import VERSION, ownership
from .ledger import Ledger, money
from .live_execution import LiveExecution, digest


VERSION_ID = "confirmed-account-movements-v1"


def _candidate(original, evidence, policy):  # noqa: C901 - one chronological accounting replay
    require(policy.get("version") == VERSION_ID, "movement_policy_required")
    require(policy.get("account_id") == original["account_id"], "movement_account_mismatch")
    require(
        policy.get("external_first") is True and policy.get("usdt_outside_strategy") is True,
        "movement_ownership_policy_required",
    )
    key = policy["reconciliation_id"]
    fingerprint = digest(policy)
    prior = original.get("account_movement_reconciliations", {}).get(key)
    if prior:
        require(prior["policy_sha256"] == fingerprint, "movement_policy_changed")
        return deepcopy(original), deepcopy(prior)
    require(digest(original) == policy["baseline_sha256"], "movement_baseline_changed")
    require(original == evidence["execution_state"], "movement_snapshot_changed")
    require(
        not original.get("pending_income") and not original.get("error"),
        "movement_unresolved_execution",
    )
    require(
        not original.get("flow")
        or original["flow"]["status"] in {"completed", "expired", "failed"},
        "movement_flow_not_terminal",
    )
    review = preview(original, evidence)
    manual = {
        bid
        for r in review["timeline"]
        if r["type"] not in {"BATCH_INTEREST_USER_IN", "CASH_COUPON_ACTIVITY_USER_IN"}
        for bid in r["bill_ids"]
    }
    require(manual == set(policy["manual_bill_ids"]), "movement_manual_scope_changed")
    mark = money(evidence["markets"]["USDGOUSDT"]["bid"])
    require(mark > 0, "movement_fx_valuation_required")
    state, book = deepcopy(original), Ledger.restore(original["book"])
    before_book = deepcopy(book.state())
    state.setdefault("account_movement_bill_receipts", {})
    state.setdefault("quarantined_account_credits", {})
    native = {str(r["billId"]): r for rows in evidence["bills"].values() for r in rows}
    now = evidence["observed_at_ms"]
    accounting = state["income_accounting"]
    require(accounting["status"] == "active", "movement_income_policy_not_active")

    def event(kind, timestamp, **fields):
        state["events"].append({"kind": kind, "timestamp": timestamp, **fields})

    def offset(coin, amount):
        state["unassigned_offsets"][coin] = str(
            money(state["unassigned_offsets"][coin]) + money(amount)
        )
        require(money(state["unassigned_offsets"][coin]) >= 0, "movement_negative_external_offset")

    for row in review["timeline"]:
        stamp, kind, coin = row["timestamp"], row["type"], row["coin"]
        bills = [native[bid] for bid in row["bill_ids"]]
        for raw in bills:
            bid = str(raw["billId"])
            require(
                bid not in state["account_movement_bill_receipts"], "movement_bill_already_applied"
            )
            state["account_movement_bill_receipts"][bid] = digest(LiveExecution.interest_bill(raw))
        if kind == "CASH_COUPON_ACTIVITY_USER_IN":
            credit = LiveExecution.interest_bill(bills[0])
            receipt = "account-credit:" + credit["bill_id"]
            state["receipts"][receipt] = digest(credit)
            offset("USDT", row["external_amount"])
            state["quarantined_account_credits"][receipt] = {
                **credit,
                "offset_delta": row["external_amount"],
                "reported_amount": credit["amount"],
                "precision_residual": "0",
                "attribution": "external_account_credit",
                "destination": "spot",
                "recorded_at": now,
                "reason": "outside_strategy_cash_coupon_reward",
            }
            event(
                "external_credit_quarantined",
                stamp,
                receipt_id=receipt,
                **{
                    k: v
                    for k, v in state["quarantined_account_credits"][receipt].items()
                    if k != "timestamp"
                },
            )
        elif kind == "DEPOSIT":
            offset(coin, row["external_amount"])
            event(
                "outside_strategy_deposit_observed",
                stamp,
                bill_id=row["bill_ids"][0],
                amount=row["external_amount"],
                coin=coin,
            )
        elif kind == "BATCH_INTEREST_USER_IN":
            credit = LiveExecution.interest_bill(bills[0])
            amount, outside = money(row["strategy_amount"]), money(row["external_amount"])
            offset(coin, outside)
            native_key = "income-bill:" + credit["bill_id"]
            state["receipts"][native_key] = digest(credit)
            if coin == "USDT":
                require(amount == 0, "movement_usdt_interest_must_remain_external")
                receipt = "account-credit:" + credit["bill_id"]
                state["receipts"][receipt] = digest(credit)
                state["quarantined_account_credits"][receipt] = {
                    **credit,
                    "offset_delta": str(outside),
                    "reported_amount": credit["amount"],
                    "precision_residual": "0",
                    "attribution": "external_account_credit",
                    "destination": "spot",
                    "recorded_at": now,
                    "reason": "outside_strategy_usdt_interest",
                }
            else:
                receipt = "income-allocation:" + credit["bill_id"]
                fx = mark if coin == "USDGO" and amount else None
                allocation = {
                    "coin": coin,
                    "bill_id": credit["bill_id"],
                    "timestamp": stamp,
                    "recorded_at": now,
                    "amount": str(amount),
                    "account_amount": credit["amount"],
                    "external_amount": str(outside),
                    "strategy_units": row["strategy_units"],
                    "external_units": row["external_units"],
                    "period_start": row["period_start"],
                    "period_end": row["period_end"],
                    "method": VERSION,
                    "policy_sha256": accounting["policy_sha256"],
                    "destination": "spot",
                    "attribution": "strategy_allocation",
                    "offset_delta": str(-amount),
                    "resolves_pending": native_key,
                    "usdgo_mark": str(fx) if fx is not None else None,
                    "valuation_at": now,
                }
                if amount:
                    book.credit_interest(receipt, coin, amount, fx)
                state["receipts"][receipt] = digest(allocation)
                accounting["allocations"][native_key] = allocation
                event(
                    "interest_allocation_recorded",
                    stamp,
                    receipt_id=receipt,
                    **{k: v for k, v in allocation.items() if k != "timestamp"},
                )
        elif kind == "manual_usdgo_buy":
            paid, received = money(row["strategy_paid_USDT"]), money(row["strategy_received_USDGO"])
            if paid:
                require(received > 0, "movement_strategy_fx_receipt_too_small")
                book.convert("observed-fx:" + row["trade_id"], "to_usdgo", paid, received, mark)
            offset("USDT", -money(row["external_paid_USDT"]))
            offset("USDGO", row["external_received_USDGO"])
            state["receipts"]["spot:" + row["order_id"] + ":" + row["trade_id"]] = digest(
                evidence["orders"][row["trade_id"]]
            )
            event(
                "observed_manual_fx_allocated",
                stamp,
                order_id=row["order_id"],
                trade_id=row["trade_id"],
                strategy_paid_USDT=str(paid),
                strategy_received_USDGO=str(received),
                classification="currency_conversion_only",
            )
        elif kind == "SINGLE_USER_OUT":
            own, outside = money(row["strategy_amount"]), money(row["external_amount"])
            offset(coin, -outside)
            hid = row["holding_id"]
            if hid not in state["holdings"]:
                remote = evidence["holdings"][hid]
                require(coin == "USDT", "movement_new_strategy_holding_not_allowed")
                state["holdings"][hid] = {
                    "coin": coin,
                    "product_id": remote["product_id"],
                    "principal": "0",
                    "interest": "0",
                    "unassigned_amount": "0",
                    "owned_since": stamp,
                    "subscription_ids": [],
                    "lots": {},
                }
            position = state["holdings"][hid]
            if own:
                require(
                    coin != "USDT" or policy.get("preserve_owned_usdt_location") is True,
                    "movement_owned_usdt_location_review_required",
                )
                available = book.usdgo if coin == "USDGO" else book.cash
                require(
                    coin in {"USDGO", "USDT"} and own <= available - book.locked[coin],
                    "movement_owned_subscription_invalid",
                )
                book.locked[coin] += own
                position["principal"] = str(money(position["principal"]) + own)
            position["unassigned_amount"] = str(money(position["unassigned_amount"]) + outside)
            if row["business_order_id"] not in position["subscription_ids"]:
                position["subscription_ids"].append(row["business_order_id"])
            accounting["checkpoints"].append(
                {
                    "timestamp": stamp,
                    "recorded_at": now,
                    "positions": ownership(state["holdings"]),
                    "source_bill_id": row["bill_ids"][0],
                }
            )
            event(
                "external_savings_location_recorded",
                stamp,
                coin=coin,
                holding_id=hid,
                bill_id=row["bill_ids"][0],
                strategy_amount=str(own),
                external_amount=str(outside),
            )
    state["book"] = {k: v for k, v in book.state().items() if k in original["book"]}
    state["interest_checked_at"] = now
    accounting.update(status="active", checked_at=now)
    outside_usdt = {
        hid: {"product_id": p["product_id"], "amount": p["unassigned_amount"]}
        for hid, p in state["holdings"].items()
        if p["coin"] == "USDT"
    }
    outside_policy = {
        "version": "verified-external-usdt-earn-v1",
        "account_id": original["account_id"],
        "ledger_created_at": original["created_at"],
        "approved_at": policy["approved_at"],
        "holdings": outside_usdt,
    }
    if any(money(p["principal"]) for p in state["holdings"].values() if p["coin"] == "USDT"):
        outside_policy.update(
            version="verified-usdt-earn-location-v2", interest_is_outside_strategy=True
        )
        for hid, entry in outside_usdt.items():
            entry["strategy_principal"] = state["holdings"][hid]["principal"]
    outside_policy["sha256"] = digest(outside_policy)
    state["outside_usdt_income_policy"] = outside_policy
    for check in review["wallet_checks"]:
        require(
            money(state["unassigned_offsets"][check["coin"]]) == money(check["outside_strategy"]),
            "movement_external_wallet_mismatch",
        )
    check = object.__new__(LiveExecution)
    check.state = state
    check.check_wallet(evidence["balances"], book)
    for item in review["holding_checks"]:
        p = state["holdings"][item["holding_id"]]
        require(
            money(p["principal"]) + money(p["interest"]) == money(item["strategy"])
            and money(p["unassigned_amount"]) == money(item["outside_strategy"]),
            "movement_holding_ownership_mismatch",
        )
    for field in [
        "initial",
        "lots",
        "orders",
        "realized",
        "fees",
        "fee_rebates",
        "external_flows",
        "fund_units",
        "pending_rebates",
        "rebate_receipts",
    ]:
        require(
            state["book"].get(field) == before_book.get(field),
            "movement_protected_capital_or_cost_changed",
        )
    for field in [
        "created_at",
        "anchor",
        "anchor_version",
        "products",
        "flow",
        "operation",
        "flows",
    ]:
        require(state.get(field) == original.get(field), "movement_original_history_changed")
    record = {
        "policy_sha256": fingerprint,
        "recorded_at": now,
        "review_sha256": review["review_sha256"],
        "policy": deepcopy(policy),
        "cash_coupon_reward_total_USDT": review["cash_coupon_reward_total_USDT"],
        "manual_bill_ids": sorted(manual),
        "native_bill_ids": [b for r in review["timeline"] for b in r["bill_ids"]],
        "original_strategy_USDT_converted": review["original_strategy_USDT_used_by_manual_buy"],
        "wallet_checks": deepcopy(review["wallet_checks"]),
        "holding_checks": deepcopy(review["holding_checks"]),
        "exchange_writes": 0,
        "resume_authorized": False,
    }
    state.setdefault("account_movement_reconciliations", {})[key] = record
    event(
        "account_movement_reconciliation_completed",
        now,
        reconciliation_id=key,
        policy_sha256=fingerprint,
        manual_bill_ids=record["manual_bill_ids"],
        resume_authorized=False,
    )
    return state, deepcopy(record)


def candidate(execution_state, evidence, policy):
    """Idempotent pure candidate. Never touches SQLite, transport or worker state."""
    with localcontext() as context:
        context.prec = 60
        return _candidate(execution_state, evidence, policy)


def persist(execution_path, strategy_path, expected, execution, strategy, backup_directory):
    """Back up both DBs, then compare-and-swap their state rows in one transaction."""
    require(
        strategy.get("desired") == "paused"
        and strategy.get("phase") == "paused"
        and not strategy.get("error"),
        "movement_commit_must_remain_paused",
    )
    paths = {"execution": Path(execution_path), "strategy": Path(strategy_path)}
    backup = Path(backup_directory)
    backup.mkdir(mode=0o700, parents=True, exist_ok=False)
    for key, path in paths.items():
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(backup / (key + ".sqlite"))) as destination:
                source.backup(destination)
        (backup / (key + ".sqlite")).chmod(0o600)
        with closing(
            sqlite3.connect((backup / (key + ".sqlite")).as_uri() + "?mode=ro", uri=True)
        ) as db:
            version, raw = db.execute(
                "SELECT version,state FROM trx_state WHERE id=?", (key,)
            ).fetchone()
        require(
            version == expected[key]["version"]
            and digest(json.loads(raw)) == expected[key]["sha256"],
            "movement_backup_baseline_changed",
        )
    (backup / "manifest.json").write_text(json.dumps(expected, indent=2))
    db = sqlite3.connect(paths["execution"], timeout=5, isolation_level=None)
    try:
        db.execute("ATTACH DATABASE ? AS strategy_db", (str(paths["strategy"]),))
        require(
            db.execute("PRAGMA main.journal_mode").fetchone()[0] == "delete"
            and db.execute("PRAGMA strategy_db.journal_mode").fetchone()[0] == "delete",
            "movement_atomic_journal_mode_required",
        )
        db.execute("BEGIN IMMEDIATE")
        for key, schema in [("execution", "main"), ("strategy", "strategy_db")]:
            version, raw = db.execute(
                f"SELECT version,state FROM {schema}.trx_state WHERE id=?", (key,)
            ).fetchone()
            require(
                version == expected[key]["version"]
                and digest(json.loads(raw)) == expected[key]["sha256"],
                "movement_concurrent_state_change",
            )
            if key == "strategy":
                require(
                    json.loads(raw)["desired"] == "paused", "movement_strategy_no_longer_paused"
                )
        for key, schema, value in [
            ("execution", "main", execution),
            ("strategy", "strategy_db", strategy),
        ]:
            row = db.execute(
                f"UPDATE {schema}.trx_state SET version=version+1,state=? WHERE id=? AND version=?",
                (json.dumps(value, allow_nan=False), key, expected[key]["version"]),
            )
            require(row.rowcount == 1, "movement_concurrent_state_change")
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return {key: expected[key]["version"] + 1 for key in paths}
