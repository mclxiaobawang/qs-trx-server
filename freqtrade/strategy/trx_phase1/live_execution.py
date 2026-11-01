"""Durable real-exchange execution workflows, driven explicitly by their caller.

No background worker starts on import and no research/shadow state is adopted.
One account has one serialized workflow. Each write intent commits before the
HTTP call; ambiguous Earn calls require an operator-supplied exchange order ID.
"""

import fcntl
import hashlib
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.ledger import Ledger, StateStore, money
from freqtrade.strategy.trx_phase1.live_broker import (
    COINS,
    DAY,
    SYMBOLS,
    ExecutionError,
    WriteNotSent,
    WriteRejected,
    identifier,
    positive,
    rounded,
)
from freqtrade.strategy.trx_phase1.reconciliation_wait import (
    verified_income_candidate,
    wallet_error,
)


CANCEL_RECOVERY_MS = 60000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def allocation_lock(root):
    path = Path(root) / "account-allocation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        path.chmod(0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class ProductRule(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    coin: Literal["TRX", "USDGO"]
    product_id: str
    # Subscription minimum/precision do not establish redemption rules. Empty
    # product API fields are not the backtest's assumed one-minute delay.
    redemption_minimum: Decimal | None = Field(default=None, gt=0)
    redemption_decimals: int | None = Field(default=None, ge=0, le=18)
    redemption_evidence: str = ""

    @model_validator(mode="after")
    def verified_redemption(self):
        identifier(self.product_id)
        if self.redemption_minimum is not None and (
            self.redemption_decimals is None or not self.redemption_evidence.strip()
        ):
            raise ValueError("Redemption minimum/precision require recorded product evidence")
        return self


class ExecutionStep(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["trade", "subscribe", "redeem"]
    amount: Decimal | Literal["previous"]
    symbol: Literal["TRXUSDT", "USDGOUSDT"] | None = None
    side: Literal["buy", "sell"] | None = None
    coin: Literal["TRX", "USDGO"] | None = None
    product_id: str | None = None
    holding_id: str | None = None
    source: Literal["principal", "interest", "external_interest"] = "principal"
    external_income_sources: dict[str, Decimal] = Field(default_factory=dict)
    external_income_policy_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reference_price: Decimal | None = Field(default=None, gt=0)
    maximum_deviation: Decimal = Field(default=Decimal("0.01"), gt=0, lt=Decimal("0.5"))
    minimum_receive: Decimal = Field(default=Decimal(0), ge=0)
    profit_margin: Decimal = Field(default=Decimal("0.0005"), ge=0, lt=Decimal("0.1"))
    continue_partial: bool = False
    signal_anchor: Decimal | None = Field(default=None, gt=0)
    signal_threshold: Decimal | None = Field(default=None, gt=0, lt=1)
    redeem_lots: dict[str, Decimal] = Field(default_factory=dict)

    @model_validator(mode="after")
    def funding_guards(self):
        if (self.signal_anchor is None) != (self.signal_threshold is None):
            raise ValueError("Signal guard requires both anchor and threshold")
        if self.signal_anchor is not None and (self.kind != "trade" or self.symbol != "TRXUSDT"):
            raise ValueError("Signal guard applies to TRX trades")
        if self.redeem_lots:
            if self.kind != "redeem" or self.coin != "TRX" or self.source != "principal":
                raise ValueError("Lot-directed funding applies to TRX principal redemption")
            if (
                self.amount == "previous"
                or any(q <= 0 for q in self.redeem_lots.values())
                or sum(self.redeem_lots.values()) != self.amount
            ):
                raise ValueError("Redemption lot quantities must equal the requested amount")
        return self

    @model_validator(mode="after")
    def complete_request(self):
        if self.amount != "previous":
            positive(self.amount)
        if self.source == "external_interest":
            if (
                self.kind != "subscribe"
                or self.coin != "USDGO"
                or self.amount == "previous"
                or not self.external_income_sources
                or not self.external_income_policy_sha256
                or any(q <= 0 for q in self.external_income_sources.values())
                or sum(self.external_income_sources.values(), money(0)) != self.amount
            ):
                raise ValueError(
                    "External interest only permits a receipt-funded USDGO subscription"
                )
        elif self.external_income_sources or self.external_income_policy_sha256:
            raise ValueError("External income evidence requires its dedicated source")
        if self.kind == "trade":
            if not self.symbol or not self.side or self.reference_price is None:
                raise ValueError("Trade requires symbol, side and reviewed reference price")
            if self.minimum_receive <= 0:
                raise ValueError("Trade requires a positive minimum net receipt")
            if self.source != "principal":
                raise ValueError("Ordinary execution never sells retained TRX interest")
        elif not self.coin or not self.product_id:
            raise ValueError("Savings requires an explicit coin and product ID")
        elif self.kind == "redeem" and not self.holding_id:
            raise ValueError("Redemption requires a strategy-owned holding order")
        if self.kind != "trade" and self.source == "interest" and self.coin != "TRX":
            raise ValueError("Separate retained-interest source applies only to TRX")
        return self

    @property
    def input_coin(self):
        if self.kind != "trade":
            return self.coin
        return "USDT" if self.side == "buy" else SYMBOLS[self.symbol]

    @property
    def output_coin(self):
        if self.kind != "trade":
            return self.coin
        return SYMBOLS[self.symbol] if self.side == "buy" else "USDT"


class LiveExecution:
    def __init__(self, root, broker, account_id):
        identifier(account_id)
        self.root = Path(root) / "live_execution" / account_id
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.root / "execution.sqlite"
        self.store = StateStore(self.path)
        self.path.chmod(0o600)
        self.broker, self.account_id = broker, account_id
        self.version, self.state = self.store.load("execution")

    @classmethod
    def create(cls, root, broker, capital, products, now):
        capital = positive(capital)
        account_id = broker.account_id()  # Classic compatibility and account binding.
        result = cls(root, broker, account_id)
        with result.exclusive():
            result.reload()
            if result.state is not None:
                raise ExecutionError("execution_account_already_allocated")
            balances = broker.balances()
            if money(balances["USDT"]["available"]) < capital:
                raise ExecutionError("allocated_capital_unavailable")
            observed = max(now, broker.clock()) if hasattr(broker, "clock") else now
            baseline = {
                identifier(b["billId"]): digest(cls.interest_bill(b))
                for coin in ("TRX", "USDGO")
                for b in broker.bills(coin, max(0, observed - DAY), observed)
                if b.get("groupType") == "financial"
                and b.get("businessType") == "BATCH_INTEREST_USER_IN"
            }
            cash_baseline = {
                identifier(b["billId"]): digest(cls.interest_bill(b))
                for coin in COINS
                for b in broker.bills(coin, max(0, observed - DAY), observed)
                if b.get("groupType") == "other"
                and (
                    b.get("businessType") == "ACTIVITY_ASSET_REWARD_USER_IN"
                    or (coin == "USDT" and b.get("businessType") == "REBATE_REWARDS")
                )
            }
            if broker.balances() != balances:
                raise ExecutionError("account_changed_during_allocation")
            rules = [ProductRule.model_validate(p).model_dump(mode="json") for p in products]
            if len({r["product_id"] for r in rules}) != len(rules):
                raise ExecutionError("duplicate_product_rule")
            offsets = {
                c: str(sum((money(v) for v in balances[c].values()), money(0))) for c in COINS
            }
            offsets["USDT"] = str(money(offsets["USDT"]) - capital)
            result.state = {
                "schema": "trx-classic-live-execution-v1",
                "account_id": account_id,
                "created_at": now,
                "book": Ledger(capital).state(),
                "products": rules,
                "unassigned_offsets": offsets,
                "income_baseline_bills": baseline,
                "account_credit_baseline_bills": cash_baseline,
                "pending_income": {},
                "holdings": {},
                "receipts": {},
                "flows": [],
                "flow": None,
                "operation": None,
                "events": [],
                "anchor": None,
                "anchor_version": 0,
                "error": None,
            }
            result.save()
        return result

    @contextmanager
    def exclusive(self):
        with (self.root / "execution.lock").open("a+") as handle:
            (self.root / "execution.lock").chmod(0o600)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ExecutionError("execution_busy") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def reload(self):
        self.version, self.state = self.store.load("execution")
        if self.state and self.state["account_id"] != self.account_id:
            raise ExecutionError("account_binding_mismatch")

    def save(self):
        self.version = self.store.save("execution", self.version, self.state)

    def status(self):
        _, state = self.store.load("execution")
        if state and state["account_id"] != self.account_id:
            raise ExecutionError("account_binding_mismatch")
        return deepcopy(state)

    def event(self, kind, now, **values):
        self.state["events"].append({"kind": kind, "timestamp": now, **values})

    def require_trading_authorized(self, code="adopted_trading_not_authorized"):
        if self.state.get("trading_authorized") is False:
            raise ExecutionError(code)
        if self.state.get("opening_adoption") and not self.state.get("performance_basis"):
            raise ExecutionError("actual_start_basis_required")

    def queue(self, steps, *, now, expires_at, label, strategy_snapshot=None, intent_id=None):
        self.require_trading_authorized()
        if strategy_snapshot is not None:
            verified = ResearchConfig.from_saved(strategy_snapshot).model_dump()
            if verified != strategy_snapshot:
                raise ExecutionError("complete_strategy_snapshot_required")
        steps = [ExecutionStep.model_validate(s) for s in steps]
        if not steps or len(steps) > 20 or not now < expires_at <= now + 7 * DAY:
            raise ExecutionError("invalid_flow")
        for index, step in enumerate(steps):
            if step.amount == "previous" and (
                index == 0 or steps[index - 1].output_coin != step.input_coin
            ):
                raise ExecutionError("previous_receipt_currency_mismatch")
            if index and steps[index - 1].kind == "subscribe" and step.amount == "previous":
                raise ExecutionError("subscribed_funds_are_not_spendable")
        with self.exclusive():
            self.reload()
            if intent_id is not None:
                identifier(intent_id)
                matches = [
                    f
                    for f in [*self.state["flows"], self.state["flow"]]
                    if f and f.get("intent_id") == intent_id
                ]
                if matches:
                    if (
                        matches[0]["steps"] != [s.model_dump(mode="json") for s in steps]
                        or matches[0]["strategy_snapshot"] != strategy_snapshot
                    ):
                        raise ExecutionError("strategy_intent_conflict")
                    return matches[0]["id"]
            if (
                self.state["operation"]
                or self.state["error"]
                or (
                    self.state["flow"]
                    and self.state["flow"]["status"] not in {"completed", "expired", "failed"}
                )
            ):
                raise ExecutionError("unresolved_flow")
            if self.state["flow"]:
                self.state["flows"].append(self.state["flow"])
            flow_id = str(uuid4())
            self.state["flow"] = {
                "id": flow_id,
                "label": label,
                "created_at": now,
                "expires_at": expires_at,
                "steps": [s.model_dump(mode="json") for s in steps],
                "index": 0,
                "status": "queued",
                "outputs": [],
                "strategy_snapshot": strategy_snapshot,
                "strategy_sha256": digest(strategy_snapshot) if strategy_snapshot else None,
                "intent_id": intent_id,
            }
            self.event("flow_queued", now, flow_id=flow_id)
            self.save()
            return flow_id

    def tick(self, now):
        """One bounded step. No automatic write resubmissions or retrospective signals."""
        self.require_trading_authorized()
        with self.exclusive():
            self.reload()
            if self.broker.account_id() != self.account_id:
                raise ExecutionError("account_binding_mismatch")
            flow = self.state["flow"]
            if not flow or self.state["error"]:
                return
            try:
                if self.state["operation"] is None:
                    if flow["status"] in {"completed", "expired", "failed", "needs_review"}:
                        return
                    if flow["index"] == len(flow["steps"]):
                        flow["status"] = "completed"
                        self.event("flow_completed", now, flow_id=flow["id"])
                    elif now >= flow["expires_at"]:
                        flow["status"] = "expired"
                        self.event("flow_expired", now, flow_id=flow["id"])
                    else:
                        self.prepare(now)
                    self.save()
                elif self.state["operation"]["status"] == "prepared":
                    if self.broker.writes_enabled:
                        self.send(now)
                else:
                    self.reconcile(now)
            except ExecutionError as exc:
                # A read failure is not a negative acknowledgement. Retain every
                # reservation and operation for the next bounded recovery poll.
                self.event("read_or_reconciliation_blocked", now, code=str(exc))
                self.save()
            except Exception:
                self.event(
                    "read_or_reconciliation_blocked", now, code="unverified_exchange_evidence"
                )
                self.save()

    def owned_spot(self, book):
        return {
            "USDT": book.free_cash,
            "USDGO": book.usdgo - book.locked["USDGO"],
            "TRX": book.principal
            + book.interest_trx
            - sum((lot.locked for lot in book.lots), money(0))
            - book.locked["TRX_INTEREST"],
        }

    def check_wallet(self, balances, book, income_records=()):
        offsets, _ = self.income_adjustments(income_records)
        for coin, owned in self.owned_spot(book).items():
            actual = sum((money(v) for v in balances[coin].values()), money(0))
            expected = owned + money(offsets[coin])
            if actual != expected:
                raise wallet_error("wallet_reconciliation_required", coin, actual, expected)
            if money(balances[coin]["available"]) < owned:
                raise wallet_error(
                    "owned_balance_not_spendable",
                    coin,
                    money(balances[coin]["available"]),
                    owned,
                    basis="available",
                )

    def product_rule(self, step):
        matches = [
            p
            for p in self.state["products"]
            if p["coin"] == step.coin and p["product_id"] == step.product_id
        ]
        if len(matches) != 1:
            raise ExecutionError("product_not_selected")
        return ProductRule.model_validate(matches[0])

    def request_stop(self, now):
        with self.exclusive():
            self.reload()
            flow, op = self.state["flow"], self.state["operation"]
            if not flow or flow["status"] in {"completed", "expired", "failed"}:
                return
            flow["expires_at"] = min(now, flow["expires_at"])
            if op is None or op["status"] == "prepared":
                self.state["operation"] = None
                flow["status"] = "expired"
            else:
                # Reconcile accepted/unknown writes; never discard their funding
                # reservation. A later executing poll may cancel a live spot order.
                flow["status"] = "stopping"
            self.event("operator_stop_requested", now)
            self.save()

    def principal_sale_allocations(self, book, step, market, amount, maximum_debit):
        allocations = book.qualifying(
            market["bid"], maximum_debit, market["fee"], step.profit_margin
        )
        if sum((money(a["quantity"]) for a in allocations), money(0)) < maximum_debit:
            raise ExecutionError("profitable_principal_unavailable")
        return allocations

    def prepare(self, now):  # noqa: C901 - ordered trade and savings preflight
        flow = self.state["flow"]
        step = ExecutionStep.model_validate(flow["steps"][flow["index"]])
        amount = positive(
            flow["outputs"][-1]["amount"] if step.amount == "previous" else step.amount
        )
        book = Ledger.restore(self.state["book"])
        book, positions, receipts, income_records, accounting, balances, verified_at = (
            verified_income_candidate(
                self, book, self.state["holdings"], self.state["receipts"], now
            )
        )
        flow = self.state["flow"]
        if flow["status"] == "expired":
            raise ExecutionError("idle_flow_superseded_by_verified_location")
        self.commit_candidate(book, positions, receipts, income_records, verified_at, accounting)
        client_id = "trx-" + digest([self.account_id, flow["id"], flow["index"]])[:32]
        op = {
            "id": client_id,
            "kind": step.kind,
            "step": step.model_dump(mode="json"),
            "amount": str(amount),
            "created_at": now,
            "status": "prepared",
            "before_balances": balances,
            "exchange_id": None,
            "fills": {},
            "cancel_attempted": False,
            "bill_ids": [],
            "allocations": [],
        }
        if step.kind == "trade":
            market = self.broker.market(step.symbol, now)
            price = money(market["ask" if step.side == "buy" else "bid"])
            if abs(price / step.reference_price - 1) > step.maximum_deviation:
                raise ExecutionError("price_deviation_exceeded")
            if step.signal_anchor is not None:
                change = (
                    1 - price / step.signal_anchor
                    if step.side == "buy"
                    else price / step.signal_anchor - 1
                )
                if change < step.signal_threshold:
                    raise ExecutionError("signal_price_condition_changed")
            decimals = (
                market["price_decimals"] if step.side == "buy" else market["quantity_decimals"]
            )
            # A step's amount is a maximum source-asset budget, including fees.
            # Market buys use quote units; market sells use base units.
            amount = rounded(amount / (1 + money(market["fee"])), decimals)
            notional = amount if step.side == "buy" else amount * price
            if notional < money(market["minimum_notional"]):
                raise ExecutionError("order_below_minimum")
            maximum_debit = amount * (1 + money(market["fee"]))
            if maximum_debit > self.owned_spot(book)[step.input_coin]:
                raise ExecutionError("strategy_funds_unavailable")
            levels = market["asks"] if step.side == "buy" else market["bids"]
            need = amount if step.side == "sell" else amount / price
            depth = sum(
                (
                    money(q)
                    for p, q in levels
                    if abs(money(p) / step.reference_price - 1) <= step.maximum_deviation
                ),
                money(0),
            )
            if depth < need:
                raise ExecutionError("insufficient_verified_depth")
            if step.symbol == "TRXUSDT" and step.side == "sell":
                op["allocations"] = self.principal_sale_allocations(
                    book, step, market, amount, maximum_debit
                )
            op.update(amount=str(amount), maximum_debit=str(maximum_debit), market=market)
            op["request"] = {
                "symbol": step.symbol,
                "side": step.side,
                "orderType": "market",
                "force": "gtc",
                "size": str(amount),
                "clientOid": client_id,
                "requestTime": str(now),
                "receiveWindow": "10000",
            }
        else:
            rule = self.product_rule(step)
            terms = self.broker.product(step.coin, step.product_id)
            holdings = self.broker.holdings(self.state["created_at"] - 1000, now)
            op["before_holdings"] = holdings
            if step.kind == "subscribe":
                decimals = int(terms["subscribePrecision"])
                if not 0 <= decimals <= 18:
                    raise ExecutionError("product_precision_unverified")
                amount = rounded(amount, decimals)
                if terms["status"] != "in_progress" or not money(
                    terms["singleMinAmount"]
                ) <= amount <= min(
                    money(terms["singleMaxAmount"]), money(terms["remainingAmount"])
                ):
                    raise ExecutionError("subscription_terms_not_satisfied")
                available = self.owned_spot(book)[step.coin]
                if step.source == "external_interest":
                    from freqtrade.strategy.trx_phase1.external_income_reinvestment import authorize

                    if amount != money(step.amount):
                        raise ExecutionError("external_income_precision_changed")
                    op["external_income_policy"] = authorize(self, step, balances, book, now)
                    available = money(self.state["unassigned_offsets"]["USDGO"])
                elif step.coin == "TRX":
                    available = (
                        (book.principal - sum((lot.locked for lot in book.lots), money(0)))
                        if step.source == "principal"
                        else book.interest_trx - book.locked["TRX_INTEREST"]
                    )
                if amount > available:
                    raise ExecutionError("strategy_funds_unavailable")
            else:
                if rule.redemption_minimum is None:
                    raise ExecutionError("redemption_terms_unverified")
                if (
                    amount < rule.redemption_minimum
                    or rounded(amount, rule.redemption_decimals) != amount
                ):
                    raise ExecutionError("redemption_terms_not_satisfied")
                owned = self.state["holdings"].get(step.holding_id)
                remote = holdings.get(step.holding_id)
                if (
                    not owned
                    or not remote
                    or owned["product_id"] != step.product_id
                    or remote["product_id"] != step.product_id
                    or remote["coin"] != step.coin
                    or amount > money(owned[step.source])
                    or amount > money(remote["amount"])
                ):
                    raise ExecutionError("owned_holding_unavailable")
                if step.redeem_lots and any(
                    q > money(owned.get("lots", {}).get(k, "0"))
                    for k, q in step.redeem_lots.items()
                ):
                    raise ExecutionError("redemption_lot_ownership_mismatch")
                last = max(
                    (
                        e["timestamp"]
                        for e in self.state["events"]
                        if e["kind"] == "write_attempt" and e.get("operation_kind") == "redeem"
                    ),
                    default=0,
                )
                if now - last <= 60000:
                    raise ExecutionError("redemption_cooldown")
            op["amount"] = str(amount)
            op["request"] = {
                "productId": step.product_id,
                "periodType": "flexible",
                "amount": str(amount),
            }
            if step.kind == "redeem":
                op["request"]["orderId"] = step.holding_id
        self.state["operation"] = op
        flow["status"] = "running"
        self.event("operation_prepared", now, operation_id=client_id)

    def send(self, now):
        started = time.monotonic()
        op, flow = self.state["operation"], self.state["flow"]
        if now >= flow["expires_at"]:
            self.state["operation"] = None
            flow["status"] = "expired"
            self.save()
            return
        # Refresh all preflight inputs immediately before a first submission.
        # No previous attempt exists in this branch; deterministic ID is retained.
        self.prepare(now)
        self.save()
        op = self.state["operation"]
        if hasattr(self.broker, "reserve_submission"):
            self.broker.reserve_submission(op, flow["expires_at"])
        now += int((time.monotonic() - started) * 1000)
        if now >= flow["expires_at"]:
            self.state["operation"] = None
            flow["status"] = "expired"
            self.event("flow_expired_during_preflight", now)
            self.save()
            return
        if op["kind"] == "trade":
            if not -2000 <= now - op["market"]["timestamp"] <= 10000:
                raise ExecutionError("quote_expired_during_preflight")
            op["request"]["requestTime"] = str(now)
        op["created_at"] = now
        # Preflight freezes the deadline. The durable sending intent precedes
        # actual shared-rate admission and the first HTTP submission.
        op["status"] = "sending"
        self.event("write_attempt", now, operation_id=op["id"], operation_kind=op["kind"])
        self.save()
        try:
            response = self.broker.submit(deepcopy(op))
            order_id = identifier(response["orderId"])
            if response.get("clientOid") not in (None, op["id"]):
                raise ExecutionError("acknowledgement_identity_mismatch")
            op["exchange_id"], op["status"] = order_id, "accepted"
        except WriteNotSent:
            op["status"] = "prepared"
            self.event("write_not_sent", now, operation_id=op["id"], code="request_rate_wait")
        except WriteRejected as exc:
            op["status"], flow["status"] = "rejected", "failed"
            self.event("write_rejected", now, code=str(exc), operation_id=op["id"])
            self.state["operation"] = None
        except Exception:
            # Includes malformed successful acknowledgements. A POST may have
            # committed even when the local response cannot be parsed.
            op["status"] = "unknown"
            self.event("write_outcome_unknown", now, operation_id=op["id"])
        self.save()

    def reconcile(self, now):
        op = self.state["operation"]
        if op["status"] == "rejected":
            return
        if now - op["created_at"] >= 89 * DAY:
            self.review("api_recovery_history_exceeded", now)
        elif op["kind"] == "trade":
            self.reconcile_trade(now)
        elif not op["exchange_id"]:
            # Unlike clientOid for spot orders, timestamp/coin/amount is not a
            # unique Earn request identifier; even one candidate may be manual.
            self.review("earn_submission_identity_unknown", now)
        else:
            self.reconcile_earn(now)

    def review(self, code, now):
        self.state["flow"]["status"] = "needs_review"
        self.state["operation"]["status"] = "needs_review"
        self.event("operator_review_required", now, code=code)
        self.save()

    def attach_exchange_order(self, operation_id, exchange_id, *, now):
        """An operator supplies an independently verified Earn order ID, locally.

        This never submits or retries a request. Result/bills/holdings and wallet
        evidence must still all pass before the workflow can continue.
        """
        with self.exclusive():
            self.reload()
            op = self.state["operation"]
            if not op or op["id"] != operation_id or op["kind"] == "trade" or op["exchange_id"]:
                raise ExecutionError("operation_cannot_be_attached")
            op["exchange_id"], op["status"] = identifier(exchange_id), "accepted"
            self.state["flow"]["status"] = "running"
            self.event("operator_attached_order", now, operation_id=operation_id)
            self.save()

    def apply_fill(self, book, op, fill):
        step = ExecutionStep.model_validate(op["step"])
        base = SYMBOLS[step.symbol]
        fees = {c: money(0) for c in (base, "USDT")}
        for fee in fill["fees"]:
            if fee["coin"] not in fees or money(fee["cost"]) < 0:
                raise ExecutionError("native_fee_requires_reconciliation")
            fees[fee["coin"]] += money(fee["cost"])
        q, quote = positive(fill["quantity"]), positive(fill["quote"])
        event_id = "spot:" + fill["order_id"] + ":" + fill["id"]
        if base == "TRX":
            if step.side == "buy":
                book.buy(
                    event_id,
                    fill["order_id"],
                    fill["timestamp"],
                    q,
                    quote / q,
                    fees["USDT"],
                    fees[base],
                    quote_amount=quote,
                )
            else:
                row = book.sell(
                    event_id,
                    fill["order_id"],
                    fill["timestamp"],
                    q,
                    quote / q,
                    op["allocations"],
                    fees["USDT"],
                    fees[base],
                    quote_amount=quote,
                )
                for part in row["allocations"]:
                    for allocation in op["allocations"]:
                        if allocation["lot_id"] == part["lot_id"]:
                            allocation["quantity"] = str(
                                money(allocation["quantity"]) - money(part["quantity"])
                            )
        elif step.side == "buy":
            book.convert(event_id, "to_usdgo", quote + fees["USDT"], q - fees[base], quote / q)
        else:
            book.convert(event_id, "to_usdt", q + fees[base], quote - fees["USDT"], quote / q)
        if base == "USDGO":
            book.fees += fees["USDT"] + fees[base] * (quote / q)
        sign = 1 if step.side == "buy" else -1
        return {base: sign * q - fees[base], "USDT": -sign * quote - fees["USDT"]}

    def reconcile_trade(self, now):  # noqa: C901 - validate a whole receipt batch before commit
        op = self.state["operation"]
        step = ExecutionStep.model_validate(op["step"])
        order = self.broker.order(op["id"])
        if order is None:
            if now >= self.state["flow"]["expires_at"]:
                self.review("spot_submission_not_found", now)
            return
        order_id = identifier(order["orderId"])
        if (
            order.get("symbol") != step.symbol
            or order.get("side") != step.side
            or order.get("orderType") != "market"
            or (op["exchange_id"] and op["exchange_id"] != order_id)
        ):
            raise ExecutionError("order_identity_mismatch")
        fills = self.broker.fills(order_id, step.symbol, max(0, op["created_at"] - 1000), now)
        if any(
            f["side"] != step.side or not op["created_at"] - 1000 <= f["timestamp"] <= now
            for f in fills
        ):
            raise ExecutionError("fill_identity_mismatch")
        base_total = sum((money(f["quantity"]) for f in fills), money(0))
        quote_total = sum((money(f["quote"]) for f in fills), money(0))
        if base_total != money(order["baseVolume"]) or quote_total != money(order["quoteVolume"]):
            raise ExecutionError("fill_history_not_complete")
        if (quote_total if step.side == "buy" else base_total) > money(op["amount"]):
            raise ExecutionError("fill_exceeds_request")
        status = order["status"]
        if status not in {"live", "partially_filled", "filled", "cancelled"}:
            raise ExecutionError("order_status_unverified")
        candidate, candidate_op = Ledger.restore(self.state["book"]), deepcopy(op)
        receipts = dict(self.state["receipts"])
        deltas = {c: money(0) for c in COINS}
        for fill in fills:
            key = "spot:" + order_id + ":" + fill["id"]
            fingerprint = digest(fill)
            if key in receipts and receipts[key] != fingerprint:
                raise ExecutionError("conflicting_fill_receipt")
            # Reconstruct aggregate settlement deltas from every receipt, but
            # apply each receipt to the principal ledger exactly once.
            fees = {c: money(0) for c in (SYMBOLS[step.symbol], "USDT")}
            for fee in fill["fees"]:
                if fee["coin"] not in fees or money(fee["cost"]) < 0:
                    raise ExecutionError("native_fee_requires_reconciliation")
                fees[fee["coin"]] += money(fee["cost"])
            sign = 1 if step.side == "buy" else -1
            deltas[SYMBOLS[step.symbol]] += (
                sign * money(fill["quantity"]) - fees[SYMBOLS[step.symbol]]
            )
            deltas["USDT"] += -sign * money(fill["quote"]) - fees["USDT"]
            if key not in receipts:
                self.apply_fill(candidate, candidate_op, fill)
                receipts[key] = fingerprint
        if -deltas[step.input_coin] > money(op["maximum_debit"]):
            raise ExecutionError("fill_exceeds_reserved_funds")
        self.state["book"], self.state["receipts"] = candidate.state(), receipts
        candidate_op.update(
            exchange_id=order_id, fills={f["id"]: f for f in fills}, status="settling"
        )
        self.state["operation"] = op = candidate_op
        if status not in {"filled", "cancelled"}:
            op["status"] = "open"
            if now >= self.state["flow"]["expires_at"]:
                self.cancel_open_order(op, step, order_id, now)
            self.save()
            return
        candidate, positions, receipts, income_records, accounting = self.income_candidate(
            candidate, self.state["holdings"], receipts, now
        )
        self.check_wallet(self.broker.balances(), candidate, income_records)
        self.commit_candidate(candidate, positions, receipts, income_records, now, accounting)
        received = deltas[step.output_coin]
        if base_total and step.symbol == "TRXUSDT" and not op.get("anchor_applied"):
            self.state["anchor"] = str(quote_total / base_total)
            self.state["anchor_version"] += 1
            op["anchor_applied"] = True
        if (
            base_total
            and abs((quote_total / base_total) / step.reference_price - 1) > step.maximum_deviation
        ):
            self.review("actual_fill_deviation_exceeded", now)
        elif (
            not base_total
            or received < step.minimum_receive
            or (status == "cancelled" and not step.continue_partial)
        ):
            self.review("partial_or_minimum_receipt_not_satisfied", now)
        else:
            self.complete({"coin": step.output_coin, "amount": str(received)}, now)
        self.save()

    def cancel_open_order(self, op, step, order_id, now):  # noqa: C901 - bounded cancel journal
        """Called only after a fresh order/fills query proves it is still open.

        An ambiguous dispatched cancel is never repeated. A locally proven
        unsent request may obtain a new permit on a later poll. Both waits are
        bounded across restarts; no reservation or actual fill is discarded.
        """
        if self.state["flow"]["status"] == "needs_review":
            op["status"] = "needs_review"
            return
        if "cancel_started_at" not in op:
            # Old snapshots have only cancel_attempted and journal events.
            previous = next(
                (
                    e["timestamp"]
                    for e in reversed(self.state["events"])
                    if e["kind"] == "cancel_attempt" and e.get("operation_id") == op["id"]
                ),
                min(now, self.state["flow"]["expires_at"]),
            )
            op["cancel_started_at"] = previous if op["cancel_attempted"] else now
        op["cancel_deadline_at"] = op["cancel_started_at"] + CANCEL_RECOVERY_MS
        code = None
        if op.get("cancel_outcome") == "rejected":
            code = "cancel_rejected_requires_review"
        elif now >= op["cancel_deadline_at"]:
            code = (
                "cancel_confirmation_timeout"
                if op["cancel_attempted"]
                else "cancel_admission_timeout"
            )
        if code:
            op["cancel_review_code"] = code
            self.review(code, now)
            return
        if op["cancel_attempted"] or not self.broker.writes_enabled:
            return  # The next poll queries the order again, including after restart.
        try:
            if hasattr(self.broker, "reserve_cancellation"):
                self.broker.reserve_cancellation()
        except WriteNotSent:
            op["cancel_outcome"] = "deferred"
            self.event("cancel_not_sent", now, operation_id=op["id"], code="request_rate_wait")
            return
        op.update(cancel_attempted=True, cancel_outcome="sending")
        self.event("cancel_attempt", now, operation_id=op["id"])
        self.save()  # A crash from here is ambiguous; recovery queries, never resends.
        try:
            response = self.broker.cancel(op["id"], step.symbol)
            if identifier(response["orderId"]) != order_id or response.get("clientOid") not in (
                None,
                op["id"],
            ):
                raise ExecutionError("cancel_acknowledgement_identity_mismatch")
            op["cancel_outcome"] = "accepted"
            self.event("cancel_acknowledged", now, operation_id=op["id"])
        except WriteNotSent:
            # The short permit may expire while committing the write intent.
            # Only this explicit proof permits another first transmission.
            op.update(cancel_attempted=False, cancel_outcome="deferred")
            self.event("cancel_not_sent", now, operation_id=op["id"], code="request_rate_wait")
        except WriteRejected as exc:
            op["cancel_outcome"] = "rejected"
            self.event("cancel_rejected", now, operation_id=op["id"], code=str(exc))
        except Exception:
            op["cancel_outcome"] = "unknown"
            self.event("cancel_outcome_unknown", now, operation_id=op["id"])

    def reconcile_earn(self, now):  # noqa: C901 - validate the receipt before one atomic commit
        op = self.state["operation"]
        step = ExecutionStep.model_validate(op["step"])
        result = self.broker.result(op["kind"], op["exchange_id"])
        if result.get("result") == "fail":
            self.review("savings_result_failed_requires_wallet_review", now)
            return
        if result.get("result") != "success":
            return
        start = max(0, op["created_at"] - 1000)
        # Principal only changes location. The acknowledged business order and
        # its bills establish ownership; subscription/redemption history rows
        # neither post another credit nor supply an assumed order-ID mapping.
        bills = [
            b
            for b in self.broker.bills(step.coin, start, now)
            if str(b.get("bizOrderId")) == op["exchange_id"]
            and b.get("coin") == step.coin
            and b.get("groupType") == "financial"
            and start <= int(b["cTime"]) <= now
        ]
        amount = money(op["amount"])
        expected = -amount if op["kind"] == "subscribe" else amount
        if not bills:
            return
        if any(money(b["fees"]) != 0 for b in bills):
            raise ExecutionError("earn_fee_semantics_unverified")
        if sum((money(b["size"]) for b in bills), money(0)) != expected:
            raise ExecutionError("savings_bill_amount_mismatch")
        holdings = self.broker.holdings(self.state["created_at"] - 1000, now)
        before = op["before_holdings"]
        if step.kind == "subscribe":
            changed = [
                key
                for key, row in holdings.items()
                if row["coin"] == step.coin
                and row["product_id"] == step.product_id
                and money(row["amount"]) - money(before.get(key, {}).get("amount", 0)) >= amount
            ]
            if len(changed) != 1:
                return
            holding_id = changed[0]
        else:
            holding_id = step.holding_id
            remaining = money(holdings.get(holding_id, {}).get("amount", 0))
            if money(before[holding_id]["amount"]) - remaining > amount:
                return
        if step.source == "external_interest":
            from freqtrade.strategy.trx_phase1.external_income_reinvestment import settle

            return settle(self, step, op, bills, holdings, holding_id, now)
        candidate = Ledger.restore(self.state["book"])
        positions = deepcopy(self.state["holdings"])
        position = positions.setdefault(
            holding_id,
            {
                "coin": step.coin,
                "product_id": step.product_id,
                "principal": "0",
                "interest": "0",
                "unassigned_amount": before.get(holding_id, {}).get("amount", "0"),
                "owned_since": op["created_at"],
                "subscription_ids": [],
                "lots": {},
            },
        )
        moved_lots = self.move_location(
            candidate,
            step.coin,
            amount if step.kind == "subscribe" else -amount,
            step.source,
            (step.redeem_lots or position.get("lots", {})) if step.kind == "redeem" else None,
        )
        for lot_id, quantity in moved_lots.items():
            position["lots"][lot_id] = str(money(position["lots"].get(lot_id, "0")) + quantity)
        position[step.source] = str(money(position[step.source]) - expected)
        if step.kind == "subscribe":
            position["subscription_ids"].append(op["exchange_id"])
        candidate, positions, receipts, income_records, accounting = self.income_candidate(
            candidate, positions, self.state["receipts"], now
        )
        # Wait for the holding snapshot too, allowing only separately verified
        # native income to explain a change beyond the principal movement.
        observed = holdings.get(holding_id)
        if observed is not None and (
            observed["coin"] != step.coin or observed["product_id"] != step.product_id
        ):
            raise ExecutionError("savings_holding_identity_mismatch")
        position = positions[holding_id]
        expected_holding = sum(
            (money(position[k]) for k in ("principal", "interest", "unassigned_amount")), money(0)
        )
        if money(observed["amount"] if observed else 0) != expected_holding:
            return
        self.check_wallet(self.broker.balances(), candidate, income_records)
        self.commit_candidate(candidate, positions, receipts, income_records, now, accounting)
        op["bill_ids"] = [identifier(b["billId"]) for b in bills]
        self.complete({"coin": step.coin, "amount": str(amount), "holding_id": holding_id}, now)
        self.save()

    @staticmethod
    def move_location(book, coin, amount, source, lot_allocations=None):
        moved_lots = {}
        if coin != "TRX":
            book.locked[coin] += amount
            if not 0 <= book.locked[coin] <= book.usdgo:
                raise ExecutionError("savings_location_mismatch")
        elif source == "interest":
            book.locked["TRX_INTEREST"] += amount
            if not 0 <= book.locked["TRX_INTEREST"] <= book.interest_trx:
                raise ExecutionError("interest_location_mismatch")
        else:
            remaining = abs(amount)
            for lot in book.lots:
                available = lot.quantity - lot.locked if amount > 0 else lot.locked
                if amount < 0:
                    available = min(available, money((lot_allocations or {}).get(lot.lot_id, "0")))
                part = min(available, remaining)
                lot.locked += part if amount > 0 else -part
                if part:
                    moved_lots[lot.lot_id] = part if amount > 0 else -part
                remaining -= part
                if remaining == 0:
                    break
            if remaining:
                raise ExecutionError("principal_location_mismatch")
        return moved_lots

    def complete(self, output, now):
        op, flow = self.state["operation"], self.state["flow"]
        flow["outputs"].append(output)
        self.event(
            "operation_settled",
            now,
            operation_id=op["id"],
            exchange_id=op["exchange_id"],
            output=output,
            bill_ids=op["bill_ids"],
            fills=list(op["fills"].values()),
        )
        flow["index"] += 1
        flow["status"] = "completed" if flow["index"] == len(flow["steps"]) else "running"
        self.state["operation"] = None

    @staticmethod
    def interest_bill(bill):
        return {
            "bill_id": identifier(bill["billId"]),
            "order_id": identifier(bill["bizOrderId"]),
            "coin": bill["coin"],
            "group": bill["groupType"],
            "type": bill.get("businessType"),
            "amount": str(money(bill["size"]).normalize()),
            "fees": str(money(bill["fees"]).normalize()),
            "timestamp": int(bill["cTime"]),
        }

    @staticmethod
    def matching_income_bills(rows, credit):
        return [
            b
            for b in rows
            if str(b.get("bizOrderId")) == credit["order_id"]
            and b.get("coin") == credit["coin"]
            and b.get("groupType") == "financial"
            and money(b["size"]) == money(credit["amount"])
            and money(b["fees"]) == 0
            and credit["timestamp"] <= int(b["cTime"])
        ]

    def income_candidate(self, book, positions, receipts, now):  # noqa: C901 - atomic income batch
        start = self.state.get("interest_checked_at", self.state["created_at"])
        self.broker.history_window(start, now)
        start = max(self.state["created_at"], start - DAY)
        remote = self.broker.holdings(self.state["created_at"] - 1000, now) if positions else {}
        book, positions = Ledger.restore(book.state()), deepcopy(positions)
        receipts = dict(receipts)
        income_records, grouped, used_bills = [], {}, set()
        bills_by_coin = {c: self.broker.bills(c, start, now) for c in ("TRX", "USDGO")}
        for coin in ("TRX", "USDGO"):
            records = (
                self.broker.savings_records(coin, start, now, order_type="pay_interest")
                if positions and not self.state.get("performance_basis")
                else []
            )
            for row in records:
                if row.get("orderType") != "pay_interest" or row.get("coinName") != coin:
                    continue
                matching = [
                    key
                    for key, p in positions.items()
                    if str(row["orderId"]) == key or str(row["orderId"]) in p["subscription_ids"]
                ]
                if not matching:
                    continue  # Unassigned account assets never become strategy income.
                if (
                    len(matching) != 1
                    or row.get("settleCoinName") != coin
                    or row.get("productType") != "flexible"
                ):
                    raise ExecutionError("interest_attribution_unverified")
                holding_id = matching[0]
                position = positions[holding_id]
                stamp, amount = int(row["ts"]), money(row["amount"])
                if not start <= stamp <= now or amount < 0:
                    raise ExecutionError("invalid_interest_record")
                if stamp < position.get("owned_since", self.state["created_at"]):
                    continue
                key = "income:" + digest([str(row["orderId"]), stamp, coin])
                normalized = {
                    "holding_id": holding_id,
                    "order_id": str(row["orderId"]),
                    "coin": coin,
                    "amount": str(amount),
                    "timestamp": stamp,
                }
                fingerprint = digest(normalized)
                allocated = self.state.get("income_accounting", {}).get("allocations", {})
                matched_bills = self.matching_income_bills(bills_by_coin[coin], normalized)
                allocated_ids = {entry["bill_id"] for entry in allocated.values()}
                if any(str(b["billId"]) in allocated_ids for b in matched_bills):
                    if len(matched_bills) != 1 or (
                        key in receipts and receipts[key] != fingerprint
                    ):
                        raise ExecutionError("conflicting_interest_record")
                    # Late history is corroboration, never another asset credit.
                    receipts[key] = fingerprint
                    used_bills.add(str(matched_bills[0]["billId"]))
                    continue
                if key in receipts:
                    if receipts[key] != fingerprint:
                        raise ExecutionError("conflicting_interest_record")
                    used_bills.update(
                        str(b["billId"])
                        for b in self.matching_income_bills(bills_by_coin[coin], normalized)
                    )
                    continue
                grouped[holding_id] = grouped.get(holding_id, money(0)) + amount
                income_records.append((key, normalized))
                receipts[key] = fingerprint
        destinations = {}
        for holding_id, amount in grouped.items():
            p = positions[holding_id]
            before = sum(
                (money(p[k]) for k in ("principal", "interest", "unassigned_amount")), money(0)
            )
            after = money(remote.get(holding_id, {}).get("amount", "0"))
            if after == before + amount:
                destinations[holding_id] = "earn"
            elif after == before:
                destinations[holding_id] = "spot"
            else:
                raise ExecutionError("interest_holding_evidence_incomplete")
        deferred = set()
        for key, credit in income_records:
            coin, amount = credit["coin"], money(credit["amount"])
            destination = destinations[credit["holding_id"]]
            position = positions[credit["holding_id"]]
            assigned = money(position["unassigned_amount"]) == 0
            if assigned and self.state.get("income_accounting"):
                from freqtrade.strategy.trx_phase1.income_allocation import HOUR, effective_units

                end = credit["timestamp"] // HOUR * HOUR
                weights, _ = effective_units(
                    self.state["income_accounting"]["checkpoints"], coin, end - HOUR, end
                )
                assigned = weights["external"] == 0
            if destination == "spot" and amount:
                matches = [
                    b
                    for b in self.matching_income_bills(bills_by_coin[coin], credit)
                    if str(b["billId"]) not in used_bills
                ]
                if len(matches) != 1:
                    if any(
                        b.get("coin") == coin
                        and b.get("groupType") == "financial"
                        and b.get("businessType") == "BATCH_INTEREST_USER_IN"
                        for b in bills_by_coin[coin]
                    ):
                        # Preserve the independently proven account receipt, but
                        # do not invent a history/business-order mapping.
                        deferred.add(key)
                        del receipts[key]
                        continue
                    raise ExecutionError("interest_arrival_bill_unverified")
                bill = self.interest_bill(matches[0])
                used_bills.add(bill["bill_id"])
                bill_key = "income-bill:" + bill["bill_id"]
                fingerprint = digest(bill)
                if bill_key in receipts and receipts[bill_key] != fingerprint:
                    raise ExecutionError("conflicting_interest_bill")
                pending = self.state.get("pending_income", {}).get(bill_key)
                if bill_key in receipts and pending is None:
                    raise ExecutionError("interest_bill_already_consumed")
                if pending:
                    assigned = assigned and credit["holding_id"] in pending["eligible_holdings"]
                    credit["resolves_pending"] = bill_key
                    credit["offset_delta"] = str(-amount if assigned else 0)
                else:
                    credit["offset_delta"] = str(0 if assigned else amount)
                receipts[bill_key] = fingerprint
                credit["bill_id"] = bill["bill_id"]
                credit.update(type=bill["type"], group=bill["group"])
            credit["destination"] = destination
            credit["attribution"] = "strategy" if assigned else "unassigned"
            if not assigned:
                credit["reason"] = "mixed_or_unproven_holding_ownership"
                if destination == "earn":
                    position["unassigned_amount"] = str(
                        money(position["unassigned_amount"]) + amount
                    )
                continue
            fx = money(self.broker.market("USDGOUSDT", now)["bid"]) if coin == "USDGO" else None
            book.credit_interest(key, coin, amount, fx)
            if destination == "earn":
                source = "interest" if coin == "TRX" else "principal"
                self.move_location(book, coin, amount, source)
                position[source] = str(money(position[source]) + amount)
        income_records = [(k, v) for k, v in income_records if k not in deferred]
        # A verified native interest bill proves an account receipt even while
        # product history lags. Quarantine it outside strategy funds/PnL until
        # an exact ownership link arrives; amount/time similarity is insufficient.
        for coin, bills in bills_by_coin.items():
            for raw in bills:
                if (
                    raw.get("groupType") != "financial"
                    or raw.get("businessType") != "BATCH_INTEREST_USER_IN"
                ):
                    continue
                bill = self.interest_bill(raw)
                if bill["coin"] != coin or not self.state["created_at"] <= bill["timestamp"] <= now:
                    raise ExecutionError("invalid_interest_bill")
                amount = money(bill["amount"])
                if amount <= 0 or money(bill["fees"]) != 0:
                    raise ExecutionError("invalid_interest_bill")
                fingerprint = digest(bill)
                baseline = self.state.get("income_baseline_bills", {})
                if bill["bill_id"] in baseline:
                    if baseline[bill["bill_id"]] != fingerprint:
                        raise ExecutionError("conflicting_interest_bill")
                    continue
                key = "income-bill:" + bill["bill_id"]
                if key in receipts:
                    if receipts[key] != fingerprint:
                        raise ExecutionError("conflicting_interest_bill")
                    continue
                if bill["bill_id"] in used_bills:
                    continue
                receipts[key] = fingerprint
                income_records.append(
                    (
                        key,
                        {
                            **bill,
                            "attribution": "unassigned",
                            "destination": "spot",
                            "offset_delta": str(amount),
                            "reason": "awaiting_exact_ownership_evidence",
                            "eligible_holdings": [
                                k
                                for k, p in positions.items()
                                if p["coin"] == coin
                                and money(p["unassigned_amount"]) == 0
                                and p.get("owned_since", self.state["created_at"])
                                <= bill["timestamp"]
                                and money(p["principal"]) + money(p["interest"]) > 0
                            ],
                        },
                    )
                )
        from freqtrade.strategy.trx_phase1.income_allocation import allocate

        book, positions, receipts, income_records, accounting = allocate(
            self, book, positions, receipts, income_records, remote, now
        )
        from freqtrade.strategy.trx_phase1.account_credits import quarantine

        receipts, income_records = quarantine(
            self, book, receipts, income_records, now, bills_by_coin=bills_by_coin
        )
        return book, positions, receipts, income_records, accounting

    def income_adjustments(self, income_records):
        offsets = dict(self.state["unassigned_offsets"])
        pending = deepcopy(self.state.get("pending_income", {}))
        for key, credit in income_records:
            coin = credit["coin"]
            offsets[coin] = str(money(offsets[coin]) + money(credit.get("offset_delta", "0")))
            if credit.get("resolves_pending"):
                pending.pop(credit["resolves_pending"], None)
            if credit.get("attribution") == "unassigned":
                pending[key] = credit
        return offsets, pending

    def commit_candidate(self, book, positions, receipts, income_records, now, accounting):
        offsets, pending = self.income_adjustments(income_records)
        self.state.update(
            book=book.state(),
            holdings=positions,
            receipts=receipts,
            interest_checked_at=now,
            unassigned_offsets=offsets,
            pending_income=pending,
        )
        if accounting is not None:
            self.state["income_accounting"] = accounting
        for key, credit in income_records:
            if credit.get("attribution") == "external_account_credit":
                self.state.setdefault("quarantined_account_credits", {})[key] = credit
            self.event(
                "external_credit_quarantined"
                if credit.get("attribution") == "external_account_credit"
                else "interest_allocation_recorded"
                if credit.get("attribution") == "strategy_allocation"
                else "unattributed_interest_observed"
                if credit.get("attribution") == "unassigned"
                else "credited_interest_imported",
                now,
                receipt_id=key,
                **credit,
            )

    def import_interest(self, now):
        """Import verified native credits atomically, without any exchange write."""
        self.require_trading_authorized("adoption_income_attribution_pending")
        with self.exclusive():
            self.reload()
            if self.broker.account_id() != self.account_id:
                raise ExecutionError("account_binding_mismatch")
            if self.state["operation"]:
                raise ExecutionError("resolve_operation_before_interest_import")
            book, positions, receipts, income_records, accounting, _, verified_at = (
                verified_income_candidate(
                    self,
                    Ledger.restore(self.state["book"]),
                    self.state["holdings"],
                    self.state["receipts"],
                    now,
                )
            )
            self.commit_candidate(
                book, positions, receipts, income_records, verified_at, accounting
            )
            self.save()
            return len(income_records)


def local_execution_status(root):
    """Authenticated UI/CLI inspection without credentials or exchange requests."""
    accounts = []
    for path in sorted((Path(root) / "live_execution").glob("*/execution.sqlite")):
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db, db:
            row = db.execute("SELECT state FROM trx_state WHERE id='execution'").fetchone()
        if row is None:
            continue
        state = json.loads(row[0])
        book = Ledger.restore(state["book"])
        accounts.append(
            {
                "account_id": state["account_id"],
                "flow": state["flow"],
                "operation": state["operation"],
                "holdings": state["holdings"],
                "book": {
                    "cash": str(book.cash),
                    "usdgo": str(book.usdgo),
                    "principal_trx": str(book.principal),
                    "retained_interest_trx": str(book.interest_trx),
                },
                "events": state["events"][-30:],
            }
        )
    return {"accounts": accounts, "page_can_submit_orders": False}
