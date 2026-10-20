"""Human-operated, bounded live-chain acceptance. prepare/verify are GET-only.

Each execute command submits at most one reviewed request. The final test-only
sale may realize a small loss; the production strategy's profit guard is unchanged.
"""

import argparse
import fcntl
import json
import os
import time
from contextlib import contextmanager
from decimal import ROUND_UP
from pathlib import Path

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError, rounded
from freqtrade.strategy.trx_phase1.live_execution import (
    ExecutionStep,
    LiveExecution,
    ProductRule,
    allocation_lock,
    digest,
)
from freqtrade.strategy.trx_phase1.live_funding import LiveFunding
from freqtrade.strategy.trx_phase1.live_service import LiveService


PHASES = (
    "USDGO 现货买入",
    "USDGO 最小申购",
    "USDGO 赎回",
    "USDGO 现货卖出",
    "TRX 现货买入",
    "TRX 最小申购",
    "TRX 赎回",
    "TRX 测试卖出",
)


class MinimumChainExecution(LiveExecution):
    """A bounded closeout policy available only inside the isolated test ledger."""

    loss_limit = money("0.01")
    closeout_deviation = money("0.001")

    def __init__(self, folder, broker, control):
        super().__init__(folder, broker, control["account_id"])
        self.control = control

    def is_closeout(self):
        flow = self.state["flow"]
        return bool(flow and flow["label"] == MinimumChainCheck.label(7))

    def closeout_basis(self, book, step):
        flow, completed = self.state["flow"], self.control["completed"]
        if (
            self.root.resolve().parent.parent.name != "minimum-full-chain-2026-10-17"
            or self.root.resolve().parent.parent.parent.name != "execution_checks"
            or len(completed) != 7
            or [r["phase"] for r in completed] != list(PHASES[:7])
            or len(flow["steps"]) != 1
            or flow["index"] != 0
            or flow.get("strategy_snapshot") is not None
            or step.amount != book.principal
            or step.source != "principal"
            or step.signal_anchor is not None
            or len(book.lots) != 1
            or any(lot.locked for lot in book.lots)
            or not 0 < book.initial == money(self.control["capital"]) <= 2
        ):
            raise ExecutionError("acceptance_closeout_scope_invalid")
        buy_id = "trx-" + digest([self.account_id, completed[4]["flow_id"], 0])[:32]
        buys = [
            e
            for e in self.state["events"]
            if e["kind"] == "operation_settled" and e["operation_id"] == buy_id
        ]
        lot = book.lots[0]
        if (
            len(buys) != 1
            or buys[0]["output"]["coin"] != "TRX"
            or buys[0]["exchange_id"] != lot.lot_id
            or money(buys[0]["output"]["amount"]) != lot.quantity
            or lot.remaining_cost
            != sum(
                (
                    money(f["quote"])
                    + sum(
                        (money(fee["cost"]) for fee in f["fees"] if fee["coin"] == "USDT"),
                        money(0),
                    )
                    for f in buys[0]["fills"]
                ),
                money(0),
            )
        ):
            raise ExecutionError("acceptance_closeout_buy_receipt_mismatch")
        final_flows = [
            f
            for f in [*self.state["flows"], flow]
            if f and f["label"] in {MinimumChainCheck.label(7), "Minimum chain 8: TRX 盈利本金卖出"}
        ]
        final_ids = {"trx-" + digest([self.account_id, f["id"], 0])[:32] for f in final_flows}
        if any(
            e["kind"] == "write_attempt" and e["operation_id"] in final_ids
            for e in self.state["events"]
        ):
            raise ExecutionError("acceptance_closeout_already_attempted")
        return lot

    @classmethod
    def closeout_preview(cls, book, step, market, amount, maximum_debit):
        cost = book.cost / book.principal
        net = amount * money(market["bid"]) * (1 - money(market["fee"]))
        floor = (
            amount
            * step.reference_price
            * (1 - step.maximum_deviation)
            * (1 - money(market["fee"]))
        )
        return {
            "purpose": "isolated_execution_acceptance",
            "profit_required": False,
            "preflight_loss_limit_usdt": str(cls.loss_limit),
            "sell_quantity_trx": str(amount),
            "maximum_principal_debit_trx": str(maximum_debit),
            "expected_net_usdt": str(net),
            "expected_realized_pnl_usdt": str(net - amount * cost),
            "minimum_net_receipt_usdt": str(floor),
            "conservative_loss_at_price_floor_usdt": str(
                max(money(0), maximum_debit * cost - floor)
            ),
            "remaining_test_trx_if_quote_fee": str(book.principal - amount),
            "quote_timestamp": market["timestamp"],
            "note": (
                "Market fills can exceed preview limits; actual fills are booked and require "
                "review if below the receipt floor."
            ),
        }

    def principal_sale_allocations(self, book, step, market, amount, maximum_debit):
        if not self.is_closeout():
            return super().principal_sale_allocations(book, step, market, amount, maximum_debit)
        lot = self.closeout_basis(book, step)
        preview = self.closeout_preview(book, step, market, amount, maximum_debit)
        if (
            step.maximum_deviation != self.closeout_deviation
            or step.minimum_receive != money(preview["minimum_net_receipt_usdt"])
            or money(preview["conservative_loss_at_price_floor_usdt"]) > self.loss_limit
        ):
            raise ExecutionError("acceptance_closeout_loss_limit_exceeded")
        return [{"lot_id": lot.lot_id, "quantity": str(maximum_debit)}]

    def prepare(self, now):
        super().prepare(now)
        if self.is_closeout():
            op = self.state["operation"]
            self.state["flow"]["acceptance_closeout"] = self.closeout_preview(
                Ledger.restore(self.state["book"]),
                ExecutionStep.model_validate(op["step"]),
                op["market"],
                money(op["amount"]),
                money(op["maximum_debit"]),
            )


class MinimumChainCheck:
    def __init__(self, root, broker, clock=None):
        self.root, self.broker = Path(root), broker
        self.folder = self.root / "execution_checks/minimum-full-chain-2026-10-17"
        self.control = self.folder / "control.json"
        self.clock = clock or (lambda: int(time.time() * 1000))

    @contextmanager
    def locked(self):
        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.folder.chmod(0o700)
        with allocation_lock(self.root), (self.folder / ".lock").open("a+") as lock:
            (self.folder / ".lock").chmod(0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (self.root / "live-strategy-control.json").exists() or any(
                (self.root / "live_execution").glob("*/execution.sqlite")
            ):
                raise ExecutionError("standalone_test_requires_unallocated_account")
            for coin in ("USDGO", "TRX"):
                path = (
                    self.root / "execution_checks/minimum-redemption-2026-10-17" / (coin + ".json")
                )
                previous = json.loads(path.read_text()) if path.exists() else {}
                if previous.get("attempted_at") and previous.get("status") != "verified":
                    raise ExecutionError("verify_previous_redemption_before_full_chain")
            yield

    def load(self):
        return json.loads(self.control.read_text()) if self.control.exists() else None

    def save(self, control):
        temporary = self.control.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as out:
            json.dump(control, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(self.control)

    def environment(self, now):
        rules = json.loads((self.root / "live-product-rules.local.json").read_text())["products"]
        rules = [ProductRule.model_validate(r).model_dump(mode="json") for r in rules]
        if len(rules) != 2 or {r["coin"] for r in rules} != {"TRX", "USDGO"}:
            raise ExecutionError("both_live_products_required")
        terms = {r["coin"]: self.broker.product(r["coin"], r["product_id"]) for r in rules}
        if any(t["status"] != "in_progress" for t in terms.values()):
            raise ExecutionError("product_unavailable")
        markets = {s: self.broker.market(s, now) for s in ("TRXUSDT", "USDGOUSDT")}
        return rules, terms, markets

    def runtime(self, control):
        if self.broker.account_id() != control["account_id"]:
            raise ExecutionError("account_binding_mismatch")
        runtime = MinimumChainExecution(self.folder, self.broker, control)
        if runtime.state is None or digest(runtime.state["products"]) != control["rules_sha256"]:
            raise ExecutionError("test_journal_binding_mismatch")
        return runtime

    @staticmethod
    def subscription(rule, terms):
        if rule["redemption_minimum"] is None:
            raise ExecutionError("redemption_terms_unverified")
        amount = max(money(terms["singleMinAmount"]), money(rule["redemption_minimum"]))
        amount = amount.quantize(money(10) ** -rule["redemption_decimals"], rounding=ROUND_UP)
        if amount != rounded(amount, int(terms["subscribePrecision"])) or amount > min(
            money(terms["singleMaxAmount"]), money(terms["remainingAmount"])
        ):
            raise ExecutionError("subscription_terms_not_satisfied")
        return amount

    def initialize(self, rules, terms, markets, now):
        subscriptions = {r["coin"]: str(self.subscription(r, terms[r["coin"]])) for r in rules}
        budgets = {}
        for coin in ("USDGO", "TRX"):
            m = markets[coin + "USDT"]
            required = max(
                money(m["minimum_notional"]), money(subscriptions[coin]) * money(m["ask"])
            )
            budget = required * (1 + money(m["fee"])) * money("1.02")
            budgets[coin] = str(budget.quantize(money("0.0001"), rounding=ROUND_UP))
        capital = (max(map(money, budgets.values())) * money("1.02")).quantize(
            money("0.01"), rounding=ROUND_UP
        )
        runtime = LiveExecution(self.folder, self.broker, self.broker.account_id())
        if runtime.state is None:
            runtime = LiveExecution.create(self.folder, self.broker, capital, rules, now)
        elif (
            money(runtime.state["book"]["initial"]) != capital or runtime.state["products"] != rules
        ):
            raise ExecutionError("test_setup_requires_review")
        control = {
            "schema": 1,
            "account_id": runtime.account_id,
            "capital": str(capital),
            "budgets": budgets,
            "subscriptions": subscriptions,
            "rules_sha256": digest(rules),
            "completed": [],
        }
        self.save(control)
        return control

    @staticmethod
    def sale_step(coin, amount, funding, markets):
        step = funding.trade(coin + "USDT", "sell", amount)
        if coin == "TRX":
            market = markets["TRXUSDT"]
            step["maximum_deviation"] = str(MinimumChainExecution.closeout_deviation)
            quantity = rounded(amount / (1 + money(market["fee"])), market["quantity_decimals"])
            step["minimum_receive"] = str(
                quantity
                * money(step["reference_price"])
                * (1 - MinimumChainExecution.closeout_deviation)
                * (1 - money(market["fee"]))
            )
        return step

    def prepare(self):
        if self.broker.writes_enabled:
            raise ExecutionError("acceptance_preparation_requires_readonly_transport")
        with self.locked():
            now, control = self.clock(), self.load()
            if control and control.get("released_at"):
                return self.report(control, self.runtime(control))
            rules, terms, markets = self.environment(now)
            if control is None:
                control = self.initialize(rules, terms, markets, now)
            runtime = self.runtime(control)
            if digest(rules) != control["rules_sha256"]:
                raise ExecutionError("test_product_rules_changed")
            index = len(control["completed"])
            if index == len(PHASES):
                return self.report(control, runtime)
            state = runtime.status()
            if state["operation"] and state["operation"]["status"] != "prepared":
                raise ExecutionError("verify_previous_request_before_preparing")
            if (
                state["flow"]
                and state["flow"]["status"] == "completed"
                and state["flow"]["label"] == self.label(index)
            ):
                raise ExecutionError("verify_completed_phase_before_next")
            runtime.request_stop(now)
            runtime.reload()
            funding = LiveFunding(runtime.state, ResearchConfig(), markets, terms)
            coin = "USDGO" if index < 4 else "TRX"
            amount = money(control["subscriptions"][coin])
            if index in (0, 4):
                steps = [funding.trade(coin + "USDT", "buy", control["budgets"][coin])]
            elif index in (1, 5):
                steps = [funding.subscribe(coin, amount)]
            elif index in (2, 6):
                positions = [
                    (k, p)
                    for k, p in runtime.state["holdings"].items()
                    if p["coin"] == coin and money(p["principal"]) >= amount
                ]
                if len(positions) != 1:
                    raise ExecutionError("test_owned_holding_missing_or_ambiguous")
                steps = [funding.redemption(coin, amount, positions[0][0])]
            else:
                book = Ledger.restore(runtime.state["book"])
                amount = book.usdgo if coin == "USDGO" else book.principal
                steps = [self.sale_step(coin, amount, funding, markets)]
            runtime.queue(steps, now=now, expires_at=now + 600000, label=self.label(index))
            runtime.tick(self.clock())  # Prepare and validate only; the transport is GET-only.
            return self.report(control, runtime)

    @staticmethod
    def label(index):
        return f"Minimum chain {index + 1}: {PHASES[index]}"

    def execute(self, *, execute=False, accept_test_closeout=False):
        if not execute or not self.broker.writes_enabled:
            raise ExecutionError("operator_execute_flag_required")
        with self.locked():
            control = self.load()
            if (
                not control
                or control.get("released_at")
                or len(control["completed"]) == len(PHASES)
            ):
                raise ExecutionError("test_request_not_prepared")
            runtime = self.runtime(control)
            state = runtime.status()
            if (
                not state["flow"]
                or state["flow"]["label"] != self.label(len(control["completed"]))
                or not state["operation"]
                or state["operation"]["status"] != "prepared"
            ):
                raise ExecutionError("request_already_attempted_or_not_prepared")
            if runtime.is_closeout() and not accept_test_closeout:
                raise ExecutionError("test_closeout_confirmation_required")
            runtime.tick(self.clock())  # One bounded submission; never a watch loop.
            return self.report(control, runtime)

    def verify(self, exchange_id=None):
        if self.broker.writes_enabled:
            raise ExecutionError("acceptance_verification_requires_readonly_transport")
        with self.locked():
            control = self.load()
            if not control:
                raise ExecutionError("test_request_not_prepared")
            runtime = self.runtime(control)
            state = runtime.status()
            if exchange_id:
                if not state["operation"]:
                    raise ExecutionError("unresolved_operation_required")
                runtime.attach_exchange_order(
                    state["operation"]["id"], exchange_id, now=self.clock()
                )
            runtime.tick(self.clock())
            state = runtime.status()
            index = len(control["completed"])
            if (
                index < len(PHASES)
                and state["flow"]
                and state["flow"]["label"] == self.label(index)
                and state["flow"]["status"] == "completed"
            ):
                control["completed"].append(
                    {
                        "phase": PHASES[index],
                        "flow_id": state["flow"]["id"],
                        "verified_at": self.clock(),
                    }
                )
                if len(control["completed"]) == len(PHASES):
                    control["released_at"] = self.clock()
                self.save(control)
            return self.report(control, runtime)

    def cancel(self):
        if self.broker.writes_enabled:
            raise ExecutionError("acceptance_verification_requires_readonly_transport")
        with self.locked():
            control = self.load()
            if not control:
                raise ExecutionError("test_request_not_prepared")
            runtime = self.runtime(control)
            if any(e["kind"] == "write_attempt" for e in runtime.state["events"]):
                raise ExecutionError("finish_started_chain_before_release")
            runtime.request_stop(self.clock())
            control.update(released_at=self.clock(), cancelled=True)
            self.save(control)
            return self.report(control, runtime)

    @staticmethod
    def report(control, runtime):
        state = runtime.status()
        return {
            "capital_limit_usdt": control["capital"],
            "completed_count": len(control["completed"]),
            "phase_count": len(PHASES),
            "complete": len(control["completed"]) == len(PHASES),
            "cancelled": bool(control.get("cancelled")),
            "phases": PHASES,
            "completed": control["completed"],
            "execution": state,
            "test_closeout": (state.get("flow") or {}).get("acceptance_closeout"),
            "note": (
                "Final acceptance-only TRX sale can realize a small loss and requires explicit "
                "operator acknowledgement. Production profit rules are unchanged. "
                "Spot rounding/fee reserve dust may remain. No strategy worker is started."
            ),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("user_data/trx_phase1"))
    parser.add_argument("command", choices=("prepare", "execute", "verify", "status", "cancel"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-test-closeout", action="store_true")
    parser.add_argument("--exchange-order-id")
    args = parser.parse_args(argv)
    broker = None
    try:
        if args.command == "execute" and not args.execute:
            raise ExecutionError("operator_execute_flag_required")
        service = LiveService(args.root.parent)
        service.root = args.root
        broker = service.broker(args.command == "execute" and args.execute)
        check = MinimumChainCheck(args.root, broker)
        if args.command == "prepare":
            result = check.prepare()
        elif args.command == "execute":
            result = check.execute(
                execute=args.execute, accept_test_closeout=args.accept_test_closeout
            )
        elif args.command == "verify":
            result = check.verify(args.exchange_order_id)
        elif args.command == "cancel":
            result = check.cancel()
        else:
            with check.locked():
                control = check.load()
                result = (
                    check.report(control, check.runtime(control))
                    if control
                    else {"complete": False, "completed_count": 0, "phase_count": len(PHASES)}
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ExecutionError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    except Exception:
        print(json.dumps({"error": "acceptance_check_failed"}))
        return 1
    finally:
        if broker:
            LiveService.close_broker(broker)


if __name__ == "__main__":
    raise SystemExit(main())
