"""Research-only funding intents outlive signals; only fresh plans may execute."""

from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field, model_validator

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine, Plan
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.research_earn import HOUR, HistoricalEarn


class FundingTerms(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    ready_hold_minutes: int = Field(default=240, ge=5, le=1440)
    cooldown_minutes: int = Field(default=60, ge=1, le=1440)
    switch_at: int | None = Field(default=None, ge=0)
    environment_changes: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def environment_only(self):
        allowed = {
            "redemption_delay_minutes",
            "usdt_redemption_delay_minutes",
            "trx_redemption_delay_minutes",
            "apr_scenario_scale",
            "trx_earn_cap",
            "usdgo_earn_cap",
        }
        if set(self.environment_changes) - allowed:
            raise ValueError("Funding research cannot change trading rules")
        if self.switch_at is not None and self.switch_at % 60000:
            raise ValueError("Funding transition must align to a minute")
        return self


class Preparation:
    def __init__(self, engine, terms):
        self.engine, self.terms = engine, terms
        self.intents = {"buy": None, "sell": None}
        self.cooldown = {"buy": 0, "sell": 0}
        self.sequence = 0
        self.events = []

    @property
    def earn(self):
        return self.engine.earn

    @property
    def ready_cash(self):
        row = self.intents["buy"]
        return float(row["remaining"]) if row and row["status"] == "ready" else 0

    @property
    def ready_trx(self):
        row = self.intents["sell"]
        return (
            sum(float(part["quantity"]) for part in row["allocations"])
            if row and row["status"] == "ready"
            else 0
        )

    def record(self, operation, timestamp, row, **values):
        event = {
            "operation": operation,
            "timestamp": timestamp,
            "intent_id": row["id"],
            "side": row["side"],
            **values,
        }
        self.events.append(event)
        self.earn.record(
            operation,
            timestamp,
            **{k: v for k, v in event.items() if k not in ("operation", "timestamp")},
        )

    def start(self, plan):
        side, now = plan.side, plan.created_at
        if self.intents[side] or now < self.cooldown[side]:
            return False
        self.sequence += 1
        row = {
            "id": f"preparation-{side}-{now}-{self.sequence}",
            "side": side,
            "source_signal_id": plan.plan_id,
            "source_signal_expires_at": plan.expires_at,
            "created_at": now,
            "status": "awaiting_arrival",
            "remaining": "0",
            "allocations": deepcopy(plan.allocations),
        }
        if side == "buy":
            self.earn.policy.prepare_cash(row["id"], now, plan.budget, expires=None)
            request = self.earn.policy.requests.get(row["id"])
            if not request:
                return False
            row["due_at"] = request["due_at"]
            row["budget_cap"] = plan.budget
        else:
            row["due_at"] = now + self.engine.config.trx_redemption_delay * 60000
            for part in row["allocations"]:
                lot = next(lot for lot in self.engine.book.lots if lot.lot_id == part["lot_id"])
                lot.locked += money(part["quantity"])
            self.sync_principal(row)
            self.earn.record(
                "redemption_requested",
                now,
                coin="TRX",
                plan_id=row["id"],
                quantity=plan.quantity,
                due_at=row["due_at"],
            )
        self.intents[side] = row
        self.record(
            "funding_intent_created",
            now,
            row,
            budget=plan.budget,
            quantity=plan.quantity,
            due_at=row["due_at"],
            source_signal_expires_at=plan.expires_at,
        )
        return True

    def sync_principal(self, row):
        self.earn.inactive_principal = [
            part for part in self.earn.inactive_principal if part.get("plan_id") != row["id"]
        ]
        self.earn.inactive_principal.extend(
            {
                "plan_id": row["id"],
                "lot_id": part["lot_id"],
                "quantity": str(part["quantity"]),
                "since_at": row["created_at"],
                "eligible_at": None,
            }
            for part in row["allocations"]
            if part["quantity"] > 0
        )

    def advance(self, timestamp):
        for side, row in self.intents.items():
            if not row:
                continue
            if row["status"] == "awaiting_arrival" and timestamp >= row["due_at"]:
                if side == "buy":
                    request = self.earn.policy.requests[row["id"]]
                    if request["status"] != "arrived":
                        continue
                    row["remaining"] = request["received"]
                else:
                    for part in row["allocations"]:
                        lot = next(
                            lot for lot in self.engine.book.lots if lot.lot_id == part["lot_id"]
                        )
                        lot.locked -= money(part["quantity"])
                    self.earn.record(
                        "redemption_received",
                        timestamp,
                        coin="TRX",
                        plan_id=row["id"],
                        quantity=self.quantity(row),
                        due_at=row["due_at"],
                    )
                row.update(
                    status="ready",
                    arrived_at=timestamp,
                    expires_at=timestamp + self.terms.ready_hold_minutes * 60000,
                )
                self.record("funding_intent_ready", timestamp, row, expires_at=row["expires_at"])
            if row["status"] == "ready" and timestamp >= row["expires_at"]:
                pending = self.engine.pending
                if pending and pending.side == side and pending.plan_id in self.earn.managed_plans:
                    self.engine.audit_end_reason = "funding_hold_expired"
                    self.engine.finish(timestamp)
                if self.intents[side] is not None:
                    self.release(side, timestamp, "ready_hold_expired")

    @staticmethod
    def quantity(row):
        return sum(float(part["quantity"]) for part in row["allocations"])

    def release(self, side, timestamp, reason):
        row = self.intents[side]
        if not row or row["status"] != "ready":
            raise ValueError("Only confirmed arrivals may be released to Earn")
        if side == "sell":
            self.earn.inactive_principal = [
                part for part in self.earn.inactive_principal if part.get("plan_id") != row["id"]
            ]
            self.earn.inactive_principal.extend(
                {
                    "lot_id": part["lot_id"],
                    "quantity": str(part["quantity"]),
                    "since_at": timestamp,
                    "eligible_at": (timestamp // HOUR + 1) * HOUR,
                }
                for part in row["allocations"]
                if part["quantity"] > 0
            )
        self.record(
            "funding_intent_closed",
            timestamp,
            row,
            reason=reason,
            unused_cash=float(row["remaining"]),
            unused_trx=self.quantity(row),
        )
        self.intents[side] = None
        self.cooldown[side] = timestamp + self.terms.cooldown_minutes * 60000

    def consume(self, plan, timestamp):
        row = self.intents[plan.side]
        if not row or row["status"] != "ready":
            return
        if plan.side == "buy":
            row["remaining"] = str(max(money(0), money(row["remaining"]) - money(plan.filled_cash)))
            empty = float(row["remaining"]) < self.engine.config.min_notional
        else:
            # Engine has already subtracted actual fills from the plan's allocations.
            used = {}
            for fill in self.engine.book.orders:
                if fill["order_id"] == plan.plan_id:
                    for part in fill["allocations"]:
                        used[part["lot_id"]] = used.get(part["lot_id"], 0) + part["quantity"]
            for part in row["allocations"]:
                part["quantity"] = max(0, part["quantity"] - used.get(part["lot_id"], 0))
            self.sync_principal(row)
            empty = self.quantity(row) * self.engine.last_price < self.engine.config.min_notional
        self.record(
            "funding_intent_used",
            timestamp,
            row,
            fresh_signal_id=plan.plan_id,
            fresh_signal_at=plan.created_at,
            filled_quantity=plan.filled_quantity,
            filled_cash=plan.filled_cash,
        )
        if empty:
            self.release(plan.side, timestamp, "consumed")

    def state(self):
        return {
            "intents": deepcopy(self.intents),
            "cooldown": dict(self.cooldown),
            "sequence": self.sequence,
            "events": list(self.events),
        }

    def restore(self, state):
        self.intents, self.cooldown = deepcopy(state["intents"]), dict(state["cooldown"])
        self.sequence, self.events = state["sequence"], list(state["events"])


class PreparedEarn(HistoricalEarn):
    def __init__(self, book, config, start_at, preparation):
        super().__init__(book, config, start_at)
        self.preparation = preparation
        self.preview = False
        self.managed_plans = set()

    def spendable_cash(self):
        if not self.config.usdt_redemption_delay_minutes:
            return self.book.free_cash
        invested = self.usdt_eligible + sum((q for _, q in self.usdt_waiting), money(0))
        return max(money(0), self.book.free_cash - invested)

    def sync_usdt(self, timestamp):
        held = money(self.preparation.ready_cash)
        self.book.locked["USDT"] += held
        try:
            super().sync_usdt(timestamp)
        finally:
            self.book.locked["USDT"] -= held

    def sweep_cash(self, timestamp, reserved=0):
        super().sweep_cash(timestamp, max(reserved, self.preparation.ready_cash))

    def advance(self, timestamp, price):
        if self.policy is not None:
            self.policy.advance(timestamp)
        self.preparation.advance(timestamp)
        super().advance(timestamp, price)

    def prepare_cash(self, plan_id, timestamp, required, expires=None):
        if self.preview:
            return self.fundable_cash(timestamp)
        return super().prepare_cash(plan_id, timestamp, required, expires)

    def prepare_plan(self, plan):
        return plan.created_at if self.preview else super().prepare_plan(plan)

    def refresh_plan(self, plan):
        if not self.preview and plan.plan_id not in self.managed_plans:
            super().refresh_plan(plan)

    def finish_plan(self, plan, timestamp):
        if plan.plan_id in self.managed_plans:
            if plan.filled_quantity:
                self.preparation.consume(plan, timestamp)
            self.managed_plans.discard(plan.plan_id)
        elif not self.preview:
            super().finish_plan(plan, timestamp)


class PreparedEngine(Engine):
    def __init__(self, config, terms, ledger=None):
        super().__init__(config, ledger)
        self.initial_config = config.model_dump()
        self.terms = FundingTerms(**terms) if isinstance(terms, dict) else terms
        self.preparation = Preparation(self, self.terms)
        self.transition = None

    def before_minute(self, timestamp):
        if self.terms.switch_at is None or self.transition or timestamp < self.terms.switch_at:
            return
        before = self.book.state()
        self.transition = deepcopy(
            {
                "timestamp": timestamp,
                "principal_trx": float(self.book.principal),
                "remaining_cost": float(self.book.cost),
                "interest_trx": float(self.book.interest_trx),
                "anchor": self.anchor,
                "anchor_version": self.anchor_version,
                "locked": {k: str(v) for k, v in self.book.locked.items()},
                "lots": [lot.state() for lot in self.book.lots],
                "pending": self.pending.plan_id if self.pending else None,
                "earn_state": self.earn.state(),
            }
        )
        self.config = ResearchConfig(
            **{**self.config.model_dump(), **self.terms.environment_changes}
        )
        self.earn.config = self.config
        if self.book.state() != before:
            raise ValueError("Funding transition changed the ledger")
        self.record("environment", timestamp, reason="predeclared_funding_transition")

    def funding_window(self, *args, **kwargs):
        return True if self.earn.preview else super().funding_window(*args, **kwargs)

    def make_plan(self, side, timestamp, price, threshold, expires, **kwargs):
        if not self.terms.enabled or self.config.earn_model != "net_benefit_hourly":
            return super().make_plan(side, timestamp, price, threshold, expires, **kwargs)
        intent = self.preparation.intents[side]
        if intent and intent["status"] == "awaiting_arrival":
            self.record(
                "environment", timestamp, reason="funding_intent_awaiting_arrival", side=side
            )
            return None
        if not intent and timestamp < self.preparation.cooldown[side]:
            return (
                super().make_plan(side, timestamp, price, threshold, expires, **kwargs)
                if (timestamp + self.delay(side) * 60000 < expires)
                else None
            )
        if not intent and timestamp + self.delay(side) * 60000 < expires:
            return super().make_plan(side, timestamp, price, threshold, expires, **kwargs)
        self.earn.preview = True
        try:
            plan = super().make_plan(side, timestamp, price, threshold, expires, **kwargs)
        finally:
            self.earn.preview = False
        if not plan:
            return None
        if intent and intent["status"] == "ready":
            if side == "buy":
                plan.budget = min(plan.budget, float(self.earn.spendable_cash()))
                plan.quantity = self.rounded(
                    min(plan.quantity, plan.budget / (price * (1 + self.config.buy_fee)))
                )
            else:
                ready = {part["lot_id"]: part["quantity"] for part in intent["allocations"]}
                plan.allocations = [
                    {**part, "quantity": min(part["quantity"], ready.get(part["lot_id"], 0))}
                    for part in plan.allocations
                    if ready.get(part["lot_id"], 0) > 0
                ]
                plan.quantity = self.rounded(sum(part["quantity"] for part in plan.allocations))
            plan.expires_at = min(plan.expires_at, intent["expires_at"])
            if (
                plan.quantity * price >= self.config.min_notional
                and plan.quantity >= self.config.min_quantity
            ):
                self.earn.managed_plans.add(plan.plan_id)
                self.record(
                    "funding_ready_plan",
                    timestamp,
                    id=plan.plan_id,
                    intent_id=intent["id"],
                    source_signal_id=intent["source_signal_id"],
                )
                return plan
        elif side == "buy" and float(self.earn.spendable_cash()) >= plan.budget:
            self.earn.managed_plans.add(plan.plan_id)
            return plan
        else:
            self.preparation.start(plan)
        self.record("finished", timestamp, id=plan.plan_id, reason="funding_separated_from_signal")
        self.pending = None
        self.reason = "funding_intent_waiting_for_fresh_signal"
        return None

    def delay(self, side):
        return (
            self.config.trx_redemption_delay
            if side == "sell"
            else max(
                self.config.usdt_redemption_delay_minutes, self.config.redemption_delay_minutes
            )
        )

    def research_state(self):
        return {
            "terms": self.terms.model_dump(),
            "initial_config": self.initial_config,
            "transition": self.transition,
            "preparation": self.preparation.state(),
            "final_config": self.config.model_dump(),
            "end_policy": "Keep actual in-flight funding and inventory; no fabricated arrival.",
        }

    def state(self):
        return {
            **super().state(),
            "funding": self.research_state(),
            "managed_plans": sorted(self.earn.managed_plans),
        }

    @classmethod
    def restore(cls, state):
        state = deepcopy(state)
        config = ResearchConfig.from_saved(state["config"])
        engine = cls(config, state["funding"]["terms"], Ledger.restore(state["book"]))
        for key, value in state.items():
            if key not in {"config", "book", "pending", "earn", "funding", "managed_plans"}:
                setattr(engine, key, value)
        engine.pending = Plan(**state["pending"]) if state["pending"] else None
        engine.preparation.restore(state["funding"]["preparation"])
        engine.initial_config = deepcopy(state["funding"]["initial_config"])
        engine.transition = deepcopy(state["funding"]["transition"])
        earn = PreparedEarn(engine.book, config, state["earn"]["start_at"], engine.preparation)
        restored = HistoricalEarn.restore(engine.book, config, state["earn"])
        for key, value in restored.__dict__.items():
            if key != "policy":
                setattr(earn, key, value)
        if earn.policy is not None:
            earn.policy.restore(state["earn"]["policy"])
        earn.managed_plans = set(state["managed_plans"])
        engine.earn, engine.prepare_cash = earn, earn.prepare_cash
        return engine


class FundingFactory:
    def __init__(self, terms):
        self.terms = FundingTerms(**terms)

    def __call__(self, config, start_at):
        if self.terms.switch_at is None:
            config = ResearchConfig(**{**config.model_dump(), **self.terms.environment_changes})
        engine = PreparedEngine(config, self.terms)
        earn = PreparedEarn(engine.book, config, start_at, engine.preparation)
        engine.earn, engine.prepare_cash = earn, earn.prepare_cash
        return engine, earn
