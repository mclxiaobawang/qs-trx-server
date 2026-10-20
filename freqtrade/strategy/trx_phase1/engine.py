import math
from dataclasses import asdict, dataclass, field
from decimal import ROUND_DOWN

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.ledger import Ledger, money


@dataclass
class Plan:
    plan_id: str
    side: str
    created_at: int
    expires_at: int
    anchor_version: int
    anchor: float
    signal_price: float
    threshold: float
    quantity: float
    budget: float = 0
    allocations: list[dict] = field(default_factory=list)
    filled_quantity: float = 0
    filled_notional: float = 0
    filled_cash: float = 0
    submitted: bool = False
    ready_at: int = 0
    reference_kind: str = "shared"
    startup: bool = False
    inventory_source: str = "principal"


class Engine:
    def __init__(self, config: ResearchConfig, ledger: Ledger | None = None, prepare_cash=None):
        self.config = config
        self.prepare_cash = prepare_cash
        self.earn = None
        self.book = ledger or Ledger(config.initial_cash)
        self.anchor = 0.0
        self.anchor_version = 0
        self.last_close = 0
        self.buy_period = config.periods[0]
        self.sell_period = config.periods[0]
        self.buy_at = 0
        self.sell_at = 0
        self.buy_threshold = config.buy_threshold
        self.sell_threshold = config.sell_threshold
        self.tier = 0
        self.relax_count = 0
        self.pending: Plan | None = None
        self.reason = "warming_up"
        self.paused = False
        self.accept_new_buys = True
        self.manual_preparation = None
        self.sequence = 0
        self.last_price = 0.0
        self.last_sigma = 0.0
        self.price_position_day = None
        self.price_position_reference = None
        self.price_position_events = []
        self.core_lot_ids = []
        self.sell_policy_events = []
        self.audit = None
        self.audit_end_reason = "execution_conditions_changed"
        self.initial_anchor_at = 0
        self.last_principal_fill_at = 0
        self.recovery_reference = 0.0
        self.recovery_started_at = 0
        self.recovery_expires_at = 0
        self.recovery_cooldown_until = 0

    def record(self, kind, timestamp, **values):
        if self.audit is not None:
            self.audit.event(kind, self, timestamp, values)

    def price_position_multiplier(self, side, timestamp, price, startup, apply=True):
        cfg = self.config
        if not cfg.price_position_enabled or startup:
            return 1.0
        day = (int(timestamp) + 480 * 60000) // 86400000
        if self.price_position_day != day:
            raise ValueError("Price-position sizing needs a causal reference for the signal day")
        reference = self.price_position_reference
        high = reference is not None and price > reference
        multiplier = getattr(cfg, f"price_position_{side}_multiplier") if high and apply else 1.0
        event = {
            "timestamp": int(timestamp),
            "side": side,
            "price": float(price),
            "reference_price": reference,
            "high_price": high,
            "multiplier": multiplier,
            "reference_available": reference is not None,
        }
        if not apply:
            event["application"] = "added_lot_quantity_unscaled"
        self.price_position_events.append(event)
        self.record(
            "price_position", timestamp, **{k: v for k, v in event.items() if k != "timestamp"}
        )
        return multiplier

    def added_lots(self):
        return [
            lot
            for lot in self.book.lots
            if lot.lot_id not in self.core_lot_ids and lot.quantity > 0
        ]

    def record_sell_policy(self, timestamp, plan_id, source, quantity, allocations):
        event = {
            "timestamp": timestamp,
            "plan_id": plan_id,
            "inventory_source": source,
            "target_quantity": quantity,
            "allocations": [dict(a) for a in allocations],
            "remaining_added_lots": len(self.added_lots()),
        }
        self.sell_policy_events.append(event)

    def occupation(self, price):
        equity = float(self.book.equity(price))
        basis = float(self.book.deployable)
        if equity <= 0 or basis <= 0:
            raise ValueError("Nonpositive equity/deployable capital; additions are paused")
        w = float(self.book.principal + self.book.interest_trx) * price / equity
        u = float(self.book.cost) / basis
        return w, u, min(1.0, max(w, u))

    @staticmethod
    def next_close(timestamp, period):
        duration = period * 60000
        return (timestamp // duration + 1) * duration

    def schedule(self, timestamp, price, sigma, buy=True, sell=True):
        cfg = self.config
        _, _, occupation = self.occupation(price)
        if cfg.mechanisms["periods"]:
            desired = sum(occupation >= boundary for boundary in (0.25, 0.55, 0.8))
            desired = min(len(cfg.periods) - 1, desired + int(sigma >= cfg.volatility_high))
            if desired > self.tier:
                self.tier, self.relax_count = desired, 0
            elif desired < self.tier:
                self.relax_count += 1
                if self.relax_count >= cfg.minimum_period_hold:
                    self.tier, self.relax_count = desired, 0
            else:
                self.relax_count = 0
            buy_period = cfg.periods[self.tier]
            sell_period = cfg.periods[max(0, self.tier - 1)]
        else:
            buy_period = sell_period = cfg.periods[0]
        if buy:
            self.buy_period = buy_period
            self.buy_at = self.next_close(timestamp, buy_period)
            self.buy_threshold = self.threshold("buy", sigma, occupation, buy_period)
        if sell:
            self.sell_period = sell_period
            self.sell_at = self.next_close(timestamp, sell_period)
            self.sell_threshold = self.threshold("sell", sigma, occupation, sell_period)
        self.record("schedule", timestamp, buy=buy, sell=sell, occupation=occupation, sigma=sigma)

    def threshold(self, side, sigma, occupation, period):
        cfg = self.config
        base = cfg.buy_threshold if side == "buy" else cfg.sell_threshold
        if not cfg.mechanisms["thresholds"]:
            return base
        volatility = sigma * math.sqrt(period / 60)
        feedback = (
            cfg.buy_occupation_multiplier * occupation
            if side == "buy"
            else -cfg.sell_occupation_multiplier * occupation
        )
        return min(
            cfg.maximum_threshold,
            max(cfg.minimum_threshold, base + cfg.volatility_multiplier * volatility + feedback),
        )

    def rounded(self, quantity):
        return float(
            money(quantity).quantize(
                money(10) ** -self.config.quantity_decimals, rounding=ROUND_DOWN
            )
        )

    def make_plan(  # noqa: C901
        self,
        side,
        timestamp,
        price,
        threshold,
        expires,
        startup=False,
        reference=None,
        recovery=False,
    ):
        cfg = self.config
        reference = self.anchor if reference is None else reference
        quote_price = (
            price * (1 + cfg.slippage if side == "buy" else 1 - cfg.slippage)
            if cfg.execution_preflight
            else price
        )
        _, _, occupation = self.occupation(price)
        self.sequence += 1
        plan_id = f"{timestamp}-{self.sequence}"
        self.record(
            "signal",
            timestamp,
            id=plan_id,
            side=side,
            threshold=threshold,
            startup=startup,
            expires_at=expires,
            reference_anchor=reference,
            reference_kind="recovery" if recovery else "shared",
        )
        if cfg.execution_preflight:
            change = 1 - quote_price / reference if side == "buy" else quote_price / reference - 1
            reason = "preflight_price_condition" if not startup and change < threshold else ""
            if reason:
                self.record(
                    "blocked",
                    timestamp,
                    id=plan_id,
                    reason=reason,
                    estimated_execution_price=quote_price,
                )
                self.reason = reason
                return None
        allocations = []
        budget = 0.0
        added_lots = self.added_lots() if cfg.sell_mode == "recent_lot_then_core" else []
        added_sale = side == "sell" and bool(added_lots)
        price_multiplier = self.price_position_multiplier(
            side, timestamp, price, startup, apply=not added_sale
        )
        if side == "buy":
            drop = max(0, 1 - price / reference)
            fraction = cfg.initial_buy_fraction if startup else cfg.buy_fraction
            factor = 1.0
            if cfg.sizing_enabled("buy") and not startup:
                factor = (
                    min(2.0, max(1.0, drop / threshold))
                    / (1 + 3 * occupation)
                    / (1 + self.last_sigma / cfg.volatility_high)
                )
            desired_budget = float(self.book.deployable) * fraction * factor * price_multiplier
            if recovery:
                desired_budget = min(
                    desired_budget, float(self.book.deployable) * cfg.recovery_buy_fraction
                )
            self.record(
                "sizing",
                timestamp,
                id=plan_id,
                planned_budget=desired_budget,
                planned_notional=desired_budget / (1 + cfg.buy_fee),
                sizing_factor=factor,
                planned_quantity=desired_budget / (price * (1 + cfg.buy_fee)),
            )
            if not self.fundable_order(desired_budget, price, timestamp):
                reason = "cash_unavailable" if not self.book.usdgo else "earn_funds_unavailable"
                self.record("blocked", timestamp, id=plan_id, reason=reason)
                self.reason = "insufficient_cash_or_profitable_principal"
                return None
            available = float(self.book.free_cash)
            if not self.funding_window(side, timestamp, expires, plan_id, desired_budget):
                return None
            if self.prepare_cash is not None:
                prepared = self.prepare_cash(plan_id, timestamp, desired_budget, expires)
                if prepared is not None:
                    available = prepared
                else:
                    available = float(self.book.free_cash)
            budget = min(available, desired_budget)
            quantity = self.rounded(budget / (price * (1 + cfg.buy_fee)))
        elif added_sale:
            # One eligible added lot per signal, newest first. Never substitute
            # profitable core inventory for an ineligible or untradable added lot.
            added_ids = {lot.lot_id for lot in added_lots}
            eligible = self.book.qualifying(
                quote_price, float(self.book.principal), cfg.sell_fee, cfg.execution_margin
            )
            selected = next((a for a in eligible if a["lot_id"] in added_ids), None)
            quantity = self.rounded(selected["quantity"]) if selected else 0.0
            allocations = [{"lot_id": selected["lot_id"], "quantity": quantity}] if selected else []
            self.record(
                "sizing",
                timestamp,
                id=plan_id,
                planned_notional=quantity * price,
                planned_quantity=quantity,
                sizing_factor=1.0,
                sell_quantity_basis="recent_added_lot",
                sizing_basis_quantity=quantity,
            )
            self.record_sell_policy(timestamp, plan_id, "added_lot", quantity, allocations)
        else:
            strength = (
                min(1.0, max(0.0, (price / reference - 1) / threshold - 1)) if threshold > 0 else 0
            )
            share = cfg.sell_fraction
            if cfg.sizing_enabled("sell"):
                share = min(1.0, share + 0.4 * occupation + cfg.sell_rebound_multiplier * strength)
            share = min(1.0, share * price_multiplier, cfg.sell_fraction_cap)
            basis = float(self.book.principal)
            if cfg.sell_quantity_basis == "profitable":
                basis = sum(
                    row["quantity"]
                    for row in self.book.qualifying(
                        quote_price, basis, cfg.sell_fee, cfg.execution_margin
                    )
                )
            quantity = self.rounded(basis * share)
            self.record(
                "sizing",
                timestamp,
                id=plan_id,
                planned_notional=quantity * price,
                planned_quantity=quantity,
                sizing_factor=share,
                sell_quantity_basis=cfg.sell_quantity_basis,
                sizing_basis_quantity=basis,
            )
            allocations = self.book.qualifying(
                quote_price, quantity, cfg.sell_fee, cfg.execution_margin
            )
            quantity = self.rounded(sum(item["quantity"] for item in allocations))
            free = float(sum((lot.quantity - lot.locked for lot in self.book.lots), money(0)))
            if cfg.tail_completion and 0 < free - quantity < max(
                cfg.min_quantity, cfg.min_notional / price
            ):
                all_eligible = self.book.qualifying(
                    quote_price, free, cfg.sell_fee, cfg.execution_margin
                )
                if abs(sum(item["quantity"] for item in all_eligible) - free) < 1e-9:
                    quantity = self.rounded(free)
            allocations = self.book.qualifying(
                quote_price, quantity, cfg.sell_fee, cfg.execution_margin
            )
            if cfg.sell_mode == "recent_lot_then_core":
                self.record_sell_policy(timestamp, plan_id, "core_fraction", quantity, allocations)
        if quantity < cfg.min_quantity or quantity * price < cfg.min_notional:
            reason = (
                "no_profitable_lots"
                if side == "sell" and not allocations
                else "below_minimum_quantity"
                if quantity < cfg.min_quantity
                else "below_minimum_notional"
            )
            self.record(
                "blocked",
                timestamp,
                id=plan_id,
                reason=reason,
                target_quantity=quantity,
                constrained_notional=quantity * price,
            )
            if self.earn is not None:
                self.earn.cancel_funding(plan_id, timestamp)
            self.reason = "insufficient_cash_or_profitable_principal"
            return None
        if side == "sell" and not self.funding_window(side, timestamp, expires, plan_id):
            return None
        self.pending = Plan(
            plan_id,
            side,
            timestamp,
            expires,
            self.anchor_version,
            reference,
            price,
            threshold,
            quantity,
            budget,
            allocations,
            reference_kind="recovery" if recovery else "shared",
            startup=startup,
            inventory_source=(
                "added_lot"
                if added_sale
                else "core_fraction"
                if side == "sell" and cfg.sell_mode == "recent_lot_then_core"
                else "principal"
            ),
        )
        if self.earn is not None and cfg.earn_model != "none":
            self.pending.ready_at = self.earn.prepare_plan(self.pending)
        self.record(
            "qualified",
            timestamp,
            id=plan_id,
            target_quantity=quantity,
            constrained_notional=quantity * price,
            budget=budget,
            ready_at=self.pending.ready_at,
        )
        self.reason = "buy_confirmed" if side == "buy" else "sell_confirmed"
        return self.pending

    def funding_window(self, side, timestamp, expires, plan_id, budget=0):
        cfg = self.config
        if not cfg.execution_preflight or self.earn is None or cfg.earn_model == "none":
            return True
        delay = cfg.trx_redemption_delay if side == "sell" else 0
        if side == "buy" and self.earn.policy is not None:
            invested = self.earn.usdt_eligible + sum(
                (q for _, q in self.earn.usdt_waiting), money(0)
            )
            if invested and budget > float(max(money(0), self.book.free_cash - invested)):
                delay = cfg.usdt_redemption_delay_minutes
            if budget > float(self.book.free_cash) and self.book.usdgo:
                delay = max(delay, cfg.redemption_delay_minutes)
        if timestamp + delay * 60000 >= expires:
            self.record(
                "blocked",
                timestamp,
                id=plan_id,
                reason="funding_window_too_short",
                ready_at=timestamp + delay * 60000,
                expires_at=expires,
            )
            self.reason = "funding_window_too_short"
            return False
        return True

    def fundable_order(self, budget, price, timestamp):
        if self.earn is None:
            return True
        quantity = self.rounded(
            min(budget, self.earn.fundable_cash(timestamp)) / (price * (1 + self.config.buy_fee))
        )
        return quantity >= self.config.min_quantity and quantity * price >= self.config.min_notional

    def observe(self, timestamp, price, sigma):
        if timestamp <= self.last_close:
            return False
        if self.last_close and timestamp != self.last_close + 60000:
            self.paused, self.reason = True, "missing_minute_data"
            raise ValueError("Unfilled minute-data gap; no new decision is permitted")
        self.last_close, self.last_price, self.last_sigma = timestamp, price, sigma
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid observed market price")
        if not math.isfinite(sigma) or sigma < 0:
            self.reason = "warming_up"
            return False
        if self.paused:
            self.reason = "paused"
            return False
        if (
            self.earn is not None
            and self.book.usdgo
            and not self.config.fx_quote_available
            and (
                any(self.config.mechanisms.values())
                or self.config.sizing_enabled("buy")
                or self.config.sizing_enabled("sell")
            )
        ):
            self.reason = "fx_valuation_pending"
            return False
        return True

    def on_close(self, timestamp: int, price: float, sigma: float):  # noqa: C901
        self.book.settle_fee_rebates(timestamp)
        if not self.observe(timestamp, price, sigma):
            self.record("environment", timestamp, reason=self.reason)
            return self.pending
        if not self.anchor:
            self.anchor = price
            self.initial_anchor_at = timestamp
            self.record("anchor", timestamp, previous=0, current=price, source="initial_close")
            self.schedule(timestamp, price, sigma)
            if self.config.startup == "small_buy" and self.accept_new_buys:
                return self.make_plan("buy", timestamp, price, 0, self.buy_at, startup=True)
            self.reason = "waiting_pullback"
            return None
        if (
            self.config.startup in ("half_position", "allocated_position")
            and not self.buy_at
            and not self.sell_at
        ):
            # Keep the actual startup fill anchor; begin normal confirmation only
            # after the same volatility warmup used by the waiting control.
            self.schedule(timestamp, price, sigma)
        if self.pending:
            self.record("environment", timestamp, reason="pending_plan_check")
            if timestamp >= self.pending.expires_at and not self.pending.submitted:
                self.audit_end_reason = "signal_expired"
                self.finish(timestamp)
                self.reason = "signal_expired"
            return self.pending
        buy_due, sell_due = timestamp == self.buy_at, timestamp == self.sell_at
        buy_threshold, sell_threshold = self.buy_threshold, self.sell_threshold
        if buy_due:
            self.update_recovery(timestamp, price, buy_threshold)
        buy_reference = self.anchor
        recovery = bool(
            self.recovery_reference
            and timestamp > self.recovery_started_at
            and 1 - price / self.anchor < buy_threshold
        )
        if recovery:
            buy_reference = self.recovery_reference
        for side, due, threshold in (
            ("buy", buy_due, buy_threshold),
            ("sell", sell_due, sell_threshold),
        ):
            if due:
                change = 1 - price / buy_reference if side == "buy" else price / self.anchor - 1
                self.record(
                    "evaluation",
                    timestamp,
                    side=side,
                    threshold=threshold,
                    period=getattr(self, side + "_period"),
                    met=change >= threshold,
                    price_change=change,
                    reference_anchor=buy_reference if side == "buy" else self.anchor,
                    reference_kind="recovery" if side == "buy" and recovery else "shared",
                )
            else:
                self.record("not_due", timestamp, sides=[side])
        buy_expiry = self.next_close(timestamp, self.buy_period)
        sell_expiry = self.next_close(timestamp, self.sell_period)
        if buy_due or sell_due or timestamp > min(self.buy_at, self.sell_at):
            self.schedule(
                timestamp,
                price,
                sigma,
                buy_due or timestamp > self.buy_at,
                sell_due or timestamp > self.sell_at,
            )
        if sell_due and price / self.anchor - 1 >= sell_threshold:
            plan = self.make_plan("sell", timestamp, price, sell_threshold, sell_expiry)
            if plan:
                return plan
        if self.accept_new_buys and buy_due and 1 - price / buy_reference >= buy_threshold:
            expiry = min(buy_expiry, self.recovery_expires_at) if recovery else buy_expiry
            return self.make_plan(
                "buy",
                timestamp,
                price,
                buy_threshold,
                expiry,
                reference=buy_reference,
                recovery=recovery,
            )
        self.reason = (
            "waiting_pullback" if not self.book.principal else "waiting_profitable_rebound"
        )
        return None

    def update_recovery(self, timestamp, price, threshold):
        cfg = self.config
        if (
            not cfg.buy_reference_recovery
            or not self.accept_new_buys
            or self.paused
            or self.pending
        ):
            return
        if self.recovery_reference and timestamp >= self.recovery_expires_at:
            self.record(
                "recovery",
                timestamp,
                reason="recovery_expired",
                reference_anchor=self.recovery_reference,
            )
            self.recovery_reference = 0
            self.recovery_cooldown_until = timestamp + cfg.recovery_cooldown_days * 86400000
        if self.recovery_reference or timestamp < self.recovery_cooldown_until:
            return
        since = self.last_principal_fill_at or self.initial_anchor_at or self.last_close
        if (
            timestamp - since < cfg.recovery_stagnation_days * 86400000
            or 1 - price / self.anchor >= threshold
        ):
            return
        if self.book.qualifying(
            price * (1 - cfg.slippage),
            float(self.book.principal),
            cfg.sell_fee,
            cfg.execution_margin,
        ):
            return
        budget = min(
            float(self.book.deployable) * cfg.recovery_buy_fraction,
            float(self.book.free_cash) if self.earn is None else self.earn.fundable_cash(timestamp),
        )
        if self.rounded(budget / (price * (1 + cfg.buy_fee))) * price < cfg.min_notional:
            return
        self.recovery_reference = price
        self.recovery_started_at = timestamp
        self.recovery_expires_at = timestamp + cfg.recovery_valid_days * 86400000
        self.record(
            "recovery",
            timestamp,
            reason="recovery_activated",
            reference_anchor=price,
            shared_anchor=self.anchor,
            expires_at=self.recovery_expires_at,
            budget_cap=budget,
        )

    def execution_quantity(self, price):
        plan = self.pending
        if not plan:
            return 0.0
        quantity = max(0.0, plan.quantity - plan.filled_quantity)
        if plan.side == "buy":
            available = max(0.0, min(float(self.book.free_cash), plan.budget - plan.filled_cash))
            quantity = min(quantity, available / (price * (1 + self.config.buy_fee)))
        return self.rounded(quantity)

    def executable(self, timestamp, price):
        plan = self.pending
        if (
            not plan
            or self.paused
            or timestamp >= plan.expires_at
            or timestamp < plan.ready_at
            or plan.anchor_version != self.anchor_version
        ):
            return False
        cfg = self.config
        quantity = self.execution_quantity(price)
        if quantity < cfg.min_quantity or quantity * price < cfg.min_notional:
            return False
        if abs(price / plan.signal_price - 1) > cfg.maximum_execution_deviation:
            return False
        if plan.side == "buy":
            if plan.threshold and 1 - price / plan.anchor < plan.threshold:
                return False
            return quantity * price * (1 + cfg.buy_fee) <= float(self.book.free_cash) + 1e-8
        if price / plan.anchor - 1 < plan.threshold:
            return False
        for allocation in plan.allocations:
            lot = next(item for item in self.book.lots if item.lot_id == allocation["lot_id"])
            if price * (1 - cfg.sell_fee) <= float(lot.unit_cost) * (1 + cfg.execution_margin):
                return False
        return True

    def fill(
        self, event_id, order_id, timestamp, quantity, price, fee_quote=0, fee_base=0, final=True
    ):
        plan = self.pending
        if not plan:
            raise ValueError("Fill has no execution plan")
        if event_id in self.book.events:
            return None
        if quantity > plan.quantity - plan.filled_quantity + 1e-8:
            raise ValueError("Fill exceeds reserved plan quantity")
        if self.audit is not None:
            self.audit.before_fill(self, event_id, timestamp, price)
        if plan.side == "buy":
            row = self.book.buy(event_id, order_id, timestamp, quantity, price, fee_quote, fee_base)
            if (
                self.config.sell_mode == "recent_lot_then_core"
                and plan.startup
                and order_id not in self.core_lot_ids
            ):
                self.core_lot_ids.append(order_id)
        else:
            row = self.book.sell(
                event_id,
                order_id,
                timestamp,
                quantity,
                price,
                plan.allocations,
                fee_quote,
                fee_base,
            )
            used = {item["lot_id"]: item["quantity"] for item in row["allocations"]}
            for allocation in plan.allocations:
                allocation["quantity"] = max(
                    0.0, allocation["quantity"] - used.get(allocation["lot_id"], 0)
                )
            if self.config.sell_mode == "recent_lot_then_core":
                row.update(sell_mode=self.config.sell_mode, inventory_source=plan.inventory_source)
        plan.filled_quantity += quantity
        plan.filled_notional += quantity * price
        plan.filled_cash += quantity * price + fee_quote
        if self.earn is not None:
            self.earn.refresh_plan(plan)
        self.book.queue_fee_rebate(
            event_id,
            (money(fee_quote) + money(fee_base) * money(price))
            * money(self.config.fee_rebate_ratio),
            self.config.rebate_due_at(timestamp),
        )
        row.update(
            signal_timestamp=plan.created_at,
            signal_anchor=plan.anchor,
            signal_threshold=plan.threshold,
            confirmation_period=self.buy_period if plan.side == "buy" else self.sell_period,
            equity_after_fill=float(self.book.equity(price)),
        )
        self.record(
            "fill", timestamp, id=plan.plan_id, fill_id=event_id, quantity=quantity, price=price
        )
        if self.audit is not None and plan.side == "buy":
            self.audit.operation(
                "subscribe",
                timestamp,
                coin="TRX",
                quantity=quantity,
                starts_at=self.next_close(timestamp, 60),
                plan_id=plan.plan_id,
            )
        if final:
            self.audit_end_reason = "filled"
            self.finish(timestamp)
        return row

    def finish(self, timestamp):
        previous_anchor = self.anchor
        previous_plan = self.pending
        if self.earn is not None and self.pending:
            self.earn.finish_plan(self.pending, timestamp)
        if self.pending and self.pending.filled_quantity:
            self.anchor = self.pending.filled_notional / self.pending.filled_quantity
            self.anchor_version += 1
            self.last_principal_fill_at = timestamp
            if self.recovery_reference:
                self.record(
                    "recovery",
                    timestamp,
                    reason="recovery_completed",
                    reference_anchor=self.recovery_reference,
                    shared_anchor=self.anchor,
                )
                self.recovery_reference = 0
                self.recovery_cooldown_until = (
                    timestamp + self.config.recovery_cooldown_days * 86400000
                )
            self.record(
                "anchor",
                timestamp,
                previous=previous_anchor,
                current=self.anchor,
                source="principal_fill_vwap",
                plan_id=self.pending.plan_id,
            )
        if previous_plan:
            self.record(
                "finished", timestamp, id=previous_plan.plan_id, reason=self.audit_end_reason
            )
        self.audit_end_reason = "execution_conditions_changed"
        self.pending = None
        if self.anchor and math.isfinite(self.last_sigma):
            self.schedule(timestamp, self.last_price, self.last_sigma)

    def execution_reason(self, timestamp, price):
        """Observational explanation for an already rejected execution check."""
        plan, cfg = self.pending, self.config
        if not plan:
            return "no_pending_plan"
        if self.paused:
            return "paused"
        if timestamp >= plan.expires_at:
            return "signal_expired"
        if timestamp < plan.ready_at:
            return "redemption_pending"
        if plan.anchor_version != self.anchor_version:
            return "anchor_changed"
        quantity = self.execution_quantity(price)
        if quantity < cfg.min_quantity:
            return "below_minimum_quantity"
        if quantity * price < cfg.min_notional:
            return "below_minimum_notional"
        if abs(price / plan.signal_price - 1) > cfg.maximum_execution_deviation:
            return "price_deviation"
        change = 1 - price / plan.anchor if plan.side == "buy" else price / plan.anchor - 1
        if plan.threshold and change < plan.threshold:
            return "price_condition_lost"
        return "cash_unavailable" if plan.side == "buy" else "profit_lost_after_costs"

    def manual_allocations(self, quantity):
        remaining, allocations = money(quantity), []
        for lot in sorted(self.book.lots, key=lambda row: row.bought_at, reverse=True):
            used = min(remaining, lot.quantity - lot.locked)
            if used > 0:
                allocations.append({"lot_id": lot.lot_id, "quantity": float(used)})
                remaining -= used
        if remaining > money("1e-9"):
            raise ValueError("Manual action exceeds free principal")
        return allocations

    def check_manual_exit(self, event_id, timestamp, source, quantity, price, accept_loss):
        if not accept_loss:
            raise ValueError("A manual exit needs explicit loss acceptance")
        if self.pending:
            raise ValueError("Cancel the pending principal plan before a manual exit")
        if quantity < self.config.min_quantity or quantity * price < self.config.min_notional:
            raise ValueError("Manual exit does not meet the market minimum")
        prepared = self.manual_preparation
        if (
            self.config.earn_model != "none"
            and self.config.trx_redemption_delay
            and prepared is None
        ):
            raise ValueError("Prepare the manual redemption and wait for arrival before exiting")
        if prepared is not None:
            if (prepared["event_id"], prepared["source"], prepared["quantity"]) != (
                event_id,
                source,
                quantity,
            ):
                raise ValueError("Manual exit does not match the prepared asset reservation")
            if timestamp < prepared["ready_at"]:
                raise ValueError("Manual redemption has not arrived")
        return prepared

    def manual_exit(self, event_id, timestamp, source, quantity, price, accept_loss=False):
        if event_id in self.book.events:
            return next(row for row in self.book.orders if row["id"] == event_id)
        quantity = self.rounded(quantity)
        prepared = self.check_manual_exit(event_id, timestamp, source, quantity, price, accept_loss)
        fee = quantity * price * self.config.sell_fee
        if prepared is not None:
            self.release_manual_reservation()
        if source == "principal":
            row = self.book.sell(
                event_id,
                event_id,
                timestamp,
                quantity,
                price,
                self.manual_allocations(quantity),
                fee,
            )
            row["kind"] = "manual_principal_exit"
        elif source == "interest":
            row = self.book.sell_interest(event_id, timestamp, quantity, price, fee)
            if self.earn is not None and prepared is None:
                self.earn.remove_retained_interest(money(quantity))
        else:
            raise ValueError("Unknown manual exit source")
        self.book.queue_fee_rebate(
            event_id,
            money(fee) * money(self.config.fee_rebate_ratio),
            self.config.rebate_due_at(timestamp),
        )
        if prepared is not None:
            if self.earn is not None:
                self.earn.inactive_principal = [
                    entry
                    for entry in self.earn.inactive_principal
                    if entry.get("plan_id") != f"manual-{event_id}"
                ]
            self.manual_preparation = None
        return row

    def prepare_manual_exit(self, event_id, timestamp, source, quantity):
        if event_id in self.book.events:
            return
        if self.pending:
            raise ValueError("Cancel the pending ordinary plan before manual preparation")
        if self.manual_preparation:
            if self.manual_preparation["event_id"] == event_id:
                return
            raise ValueError("A manual exit is already being prepared")
        quantity = self.rounded(quantity)
        if quantity <= 0:
            raise ValueError("Manual preparation requires a positive quantity")
        allocations = []
        if source == "principal":
            allocations = self.manual_allocations(quantity)
            for entry in allocations:
                next(
                    lot for lot in self.book.lots if lot.lot_id == entry["lot_id"]
                ).locked += money(entry["quantity"])
            if self.earn is not None:
                self.earn.inactive_principal.extend(
                    {
                        **entry,
                        "plan_id": f"manual-{event_id}",
                        "since_at": timestamp,
                        "eligible_at": None,
                    }
                    for entry in allocations
                )
        elif source == "interest":
            if money(quantity) > self.book.interest_trx - self.book.locked["TRX_INTEREST"]:
                raise ValueError("Manual preparation exceeds free retained interest")
            self.book.locked["TRX_INTEREST"] += money(quantity)
            if self.earn is not None:
                self.earn.remove_retained_interest(money(quantity))
        else:
            raise ValueError("Unknown manual preparation source")
        delay = self.config.trx_redemption_delay if self.config.earn_model != "none" else 0
        self.manual_preparation = {
            "event_id": event_id,
            "source": source,
            "quantity": quantity,
            "allocations": allocations,
            "ready_at": timestamp + delay * 60000,
        }

    def release_manual_reservation(self):
        prepared = self.manual_preparation
        if prepared["source"] == "principal":
            for entry in prepared["allocations"]:
                next(
                    lot for lot in self.book.lots if lot.lot_id == entry["lot_id"]
                ).locked -= money(entry["quantity"])
        else:
            self.book.locked["TRX_INTEREST"] -= money(prepared["quantity"])

    def cancel_manual_exit(self, timestamp):
        prepared = self.manual_preparation
        if not prepared:
            return
        self.release_manual_reservation()
        if self.earn is not None:
            self.earn.inactive_principal = [
                entry
                for entry in self.earn.inactive_principal
                if entry.get("plan_id") != f"manual-{prepared['event_id']}"
            ]
            eligible_at = (timestamp // 3600000 + 1) * 3600000
            if prepared["source"] == "principal":
                self.earn.inactive_principal.extend(
                    {**entry, "since_at": timestamp, "eligible_at": eligible_at}
                    for entry in prepared["allocations"]
                )
            else:
                self.earn.retained_waiting.append((eligible_at, money(prepared["quantity"])))
        self.manual_preparation = None

    def state(self):
        names = (
            "anchor",
            "anchor_version",
            "last_close",
            "buy_period",
            "sell_period",
            "buy_at",
            "sell_at",
            "buy_threshold",
            "sell_threshold",
            "tier",
            "relax_count",
            "reason",
            "paused",
            "accept_new_buys",
            "manual_preparation",
            "sequence",
            "last_price",
            "last_sigma",
            "initial_anchor_at",
            "last_principal_fill_at",
            "recovery_reference",
            "recovery_started_at",
            "recovery_expires_at",
            "recovery_cooldown_until",
            "core_lot_ids",
        )
        return {
            "config": self.config.model_dump(),
            "book": self.book.state(),
            **{key: getattr(self, key) for key in names},
            "pending": asdict(self.pending) if self.pending else None,
            "earn": self.earn.state() if self.earn is not None else None,
        }

    @classmethod
    def restore(cls, state):
        engine = cls(ResearchConfig.from_saved(state["config"]), Ledger.restore(state["book"]))
        for key, value in state.items():
            if key not in ("config", "book", "pending", "earn"):
                setattr(engine, key, value)
        engine.pending = Plan(**state["pending"]) if state["pending"] else None
        if state.get("earn") is not None:
            from freqtrade.strategy.trx_phase1.research_earn import HistoricalEarn

            engine.earn = HistoricalEarn.restore(engine.book, engine.config, state["earn"])
            engine.prepare_cash = engine.earn.prepare_cash
        return engine

    def status(self):
        price = self.last_price or self.anchor or 1
        w, u, _ = self.occupation(price)
        reserved_cash = 0.0
        if self.pending and self.pending.side == "buy":
            reserved_cash = max(0.0, self.pending.budget - self.pending.filled_cash)
        return {
            **self.book.summary(price),
            **(self.earn.summary(price) if self.earn is not None else {}),
            "anchor": self.anchor,
            "anchor_version": self.anchor_version,
            "buy_period": self.buy_period,
            "sell_period": self.sell_period,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "exposure": w,
            "occupation": u,
            "reason": self.reason,
            "paused": self.paused,
            "accept_new_buys": self.accept_new_buys,
            "last_close": self.last_close,
            "reserved_cash": reserved_cash,
            "available_cash": max(0.0, float(self.book.free_cash) - reserved_cash),
            "pending": asdict(self.pending) if self.pending else None,
        }
