"""Replay one explicitly approved USDGO wallet sweep and native income receipts.

No exchange write exists here. A successful business-order result, primary bill,
complete wallet sweep and final holding conservation must all agree. The replay
is in memory and saves exactly once; timestamps on recovered events remain real.
"""

from copy import deepcopy

from freqtrade.strategy.trx_phase1.income_allocation import allocate, ownership
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import DAY, ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution, digest


class Snapshot:
    writes_enabled = False

    def __init__(self, bills, records, holdings, balances, markets):
        self.rows, self.records = bills, records
        self.assets, self.wallet, self.markets = holdings, balances, markets

    def history_window(self, start, end):
        if not 0 <= end - start < 90 * DAY:
            raise ExecutionError("history_window_exceeded")

    def bills(self, coin, start, end):
        return [deepcopy(r) for r in self.rows[coin] if start <= int(r["cTime"]) <= end]

    def savings_records(self, coin, start, end, *, order_type=None):
        return [
            deepcopy(r)
            for r in self.records[coin]
            if start <= int(r["ts"]) <= end and (order_type is None or r["orderType"] == order_type)
        ]

    def holdings(self, *args):
        return deepcopy(self.assets)

    def balances(self):
        return deepcopy(self.wallet)

    def market(self, symbol, *args):
        return deepcopy(self.markets[symbol])


def projected(positions):
    return {
        key: {
            "coin": p["coin"],
            "product_id": p["product_id"],
            "amount": str(
                sum((money(p[k]) for k in ("principal", "interest", "unassigned_amount")), money(0))
            ),
        }
        for key, p in positions.items()
    }


def repair(execution, policy, now):  # noqa: C901 - bounded replay and one atomic commit
    with execution.exclusive():
        execution.reload()
        original = execution.state
        spec = policy.get("subscription", {})
        if (
            policy.get("version") != "trx-account-repair-v1"
            or policy.get("account_id") != execution.account_id
            or policy.get("ledger_created_at") != original["created_at"]
            or policy.get("products") != original["products"]
            or not policy.get("approved_at")
            or policy["approved_at"] > now
            or spec.get("coin") != "USDGO"
            or money(spec.get("amount", 0)) <= 0
            or not original["created_at"] <= spec.get("timestamp", 0) <= now
        ):
            raise ExecutionError("account_repair_policy_required")
        fingerprint = digest(policy)
        old = original.get("account_repairs", {}).get(spec["bill_id"])
        if old:
            if old["policy_sha256"] != fingerprint:
                raise ExecutionError("account_repair_policy_changed")
            return deepcopy(old)
        if (
            original["operation"]
            or original["error"]
            or original.get("pending_income")
            or (
                original["flow"]
                and original["flow"]["status"] not in {"completed", "expired", "failed"}
            )
        ):
            raise ExecutionError("account_repair_requires_idle_execution")
        if execution.broker.account_id() != execution.account_id:
            raise ExecutionError("account_binding_mismatch")
        if (
            execution.broker.result("subscribe", spec["business_order_id"]).get("result")
            != "success"
        ):
            raise ExecutionError("account_repair_subscription_unverified")
        start = max(
            original["created_at"],
            min(original.get("interest_checked_at", now), spec["timestamp"]) - DAY,
        )
        bills = {c: execution.broker.bills(c, start, now) for c in ("USDT", "TRX", "USDGO")}
        matches = [r for r in bills["USDGO"] if str(r.get("billId")) == spec["bill_id"]]
        if len(matches) != 1:
            raise ExecutionError("account_repair_bill_required")
        raw = matches[0]
        if (
            str(raw.get("bizOrderId")) != spec["business_order_id"]
            or raw.get("coin") != "USDGO"
            or raw.get("groupType") != "financial"
            or raw.get("businessType") != "SINGLE_USER_OUT"
            or money(raw["fees"]) != 0
            or money(raw["size"]) != -money(spec["amount"])
            or int(raw["cTime"]) != spec["timestamp"]
            or money(raw["balance"]) != 0
        ):
            raise ExecutionError("account_repair_bill_mismatch")
        remote = execution.broker.holdings(original["created_at"] - 1000, now)
        expected = projected(original["holdings"])
        holding = original["holdings"].get(spec["holding_id"])
        if not holding or holding["coin"] != "USDGO" or holding["product_id"] != spec["product_id"]:
            raise ExecutionError("account_repair_holding_required")
        expected[spec["holding_id"]]["amount"] = str(
            money(expected[spec["holding_id"]]["amount"]) + money(spec["amount"])
        )
        if set(remote) != set(expected) or any(
            r["coin"] != expected[key]["coin"]
            or r["product_id"] != expected[key]["product_id"]
            or money(r["amount"]) != money(expected[key]["amount"])
            for key, r in remote.items()
        ):
            raise ExecutionError("account_repair_holding_mismatch")
        records = {
            c: execution.broker.savings_records(c, start, now, order_type="pay_interest")
            for c in ("TRX", "USDGO")
        }
        markets = {s: execution.broker.market(s, now) for s in ("TRXUSDT", "USDGOUSDT")}
        balances = execution.broker.balances()
        work = LiveExecution.__new__(LiveExecution)
        work.root, work.account_id = execution.root, execution.account_id
        work.state = deepcopy(original)
        work.broker = Snapshot(bills, records, remote, balances, markets)
        stream = [(spec["timestamp"], "subscription", raw)]
        for coin in ("TRX", "USDGO"):
            for item in bills[coin]:
                if (
                    item.get("groupType") != "financial"
                    or item.get("businessType") != "BATCH_INTEREST_USER_IN"
                ):
                    continue
                row = work.interest_bill(item)
                if row["bill_id"] in original.get("income_baseline_bills", {}):
                    continue
                key = "income-bill:" + row["bill_id"]
                if key in original["receipts"]:
                    continue  # The final native import also validates old fingerprints.
                if row["coin"] != coin or money(row["amount"]) <= 0 or money(row["fees"]) != 0:
                    raise ExecutionError("account_repair_income_unverified")
                if row["timestamp"] == spec["timestamp"]:
                    raise ExecutionError("account_repair_event_order_ambiguous")
                stream.append((row["timestamp"], "interest", row))
        imported, movement = 0, None
        for stamp, kind, row in sorted(stream, key=lambda x: (x[0], x[1])):
            if kind == "subscription":
                book = Ledger.restore(work.state["book"])
                strategy = book.usdgo - book.locked["USDGO"]
                external = money(work.state["unassigned_offsets"]["USDGO"])
                if min(strategy, external) < 0 or strategy + external != money(spec["amount"]):
                    raise ExecutionError("account_repair_complete_wallet_sweep_required")
                position = work.state["holdings"][spec["holding_id"]]
                book.locked["USDGO"] += strategy
                position["principal"] = str(money(position["principal"]) + strategy)
                position["unassigned_amount"] = str(money(position["unassigned_amount"]) + external)
                work.state["unassigned_offsets"]["USDGO"] = "0"
                history = work.state["income_accounting"]["checkpoints"]
                if history[-1]["timestamp"] >= stamp:
                    raise ExecutionError("account_repair_checkpoint_conflict")
                history.append(
                    {
                        "timestamp": stamp,
                        "recorded_at": now,
                        "positions": ownership(work.state["holdings"]),
                        "source_bill_id": spec["bill_id"],
                    }
                )
                movement = {
                    **spec,
                    "strategy_amount": str(strategy),
                    "external_amount": str(external),
                    "recorded_at": now,
                    "source": "verified_business_result_and_primary_bill",
                }
                key = "observed-earn:" + spec["bill_id"]
                work.state["receipts"][key] = digest(work.interest_bill(raw))
                book.events.add(key)
                work.state["book"] = book.state()
                work.event(
                    "external_savings_location_recorded",
                    stamp,
                    **{k: v for k, v in movement.items() if k != "timestamp"},
                )
                continue
            key = "income-bill:" + row["bill_id"]
            credit = {
                **row,
                "attribution": "unassigned",
                "destination": "spot",
                "offset_delta": row["amount"],
                "reason": "awaiting_exact_ownership_evidence",
                "eligible_holdings": [],
            }
            receipts = {**work.state["receipts"], key: digest(row)}
            book, positions, receipts, income_rows, accounting = allocate(
                work,
                Ledger.restore(work.state["book"]),
                deepcopy(work.state["holdings"]),
                receipts,
                [(key, credit)],
                projected(work.state["holdings"]),
                now,
            )
            if (
                not accounting
                or accounting["status"] != "active"
                or work.income_adjustments(income_rows)[1]
            ):
                raise ExecutionError("account_repair_income_allocation_required")
            work.commit_candidate(book, positions, receipts, income_rows, now, accounting)
            imported += 1
        book, positions, receipts, income_rows, accounting = work.income_candidate(
            Ledger.restore(work.state["book"]), work.state["holdings"], work.state["receipts"], now
        )
        work.check_wallet(balances, book, income_rows)
        if (
            not accounting
            or accounting["status"] != "active"
            or work.income_adjustments(income_rows)[1]
        ):
            raise ExecutionError("account_repair_final_income_unresolved")
        work.commit_candidate(book, positions, receipts, income_rows, now, accounting)
        for field in (
            "initial",
            "external_flows",
            "fund_units",
            "cash",
            "lots",
            "orders",
            "realized",
            "fees",
            "fee_rebates",
        ):
            if work.state["book"][field] != original["book"][field]:
                raise ExecutionError("account_repair_capital_or_trade_changed")
        if (
            work.state["anchor"] != original["anchor"]
            or work.state["anchor_version"] != original["anchor_version"]
        ):
            raise ExecutionError("account_repair_anchor_changed")
        result = {
            "policy_sha256": fingerprint,
            "recorded_at": now,
            "movement": movement,
            "native_interest_receipts": imported,
            "exchange_writes": 0,
        }
        work.state.setdefault("account_repairs", {})[spec["bill_id"]] = result
        work.event(
            "account_reconciliation_completed",
            now,
            bill_id=spec["bill_id"],
            imported_receipts=imported,
        )
        execution.state = work.state
        execution.save()
        return deepcopy(result)
