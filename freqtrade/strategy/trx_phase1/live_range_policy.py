"""Opt-in live range rules and read-only previews over the existing financial book.

This module has no activation writer, worker start, exchange or ledger mutation.
An active policy needs matching account-bound markers in strategy and control.
"""

import json
import math
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool

from .active_price_band import DAY, OFFSET, BandHistory
from .engine import Engine
from .ledger import money
from .live_broker import ExecutionError
from .live_execution import digest


HISTORY_FILE = "range-price-history-v1.npz"


class RangePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal["trx-weighted-range-v1"] = "trx-weighted-range-v1"
    half_life_days: Literal[90] = 90
    width_cap_pct: Literal[4] = 4
    inside_buy_fraction: Literal[0.12] = 0.12
    outside_buy_fraction: Literal[0.09] = 0.09
    inside_buy_threshold: Literal[0.00725] = 0.00725
    inside_sell_threshold: Literal[0.01075] = 0.01075
    outside_buy_threshold: Literal[0.014] = 0.014
    outside_sell_threshold: Literal[0.012] = 0.012
    # Missing in existing account bindings: preserve the original policy.
    pause_upper_p75_buys: StrictBool = False


def upper_p75_intersection(price, upper, p75):
    return p75 is not None and price > upper and price > p75


class RangeEngine(Engine):
    policy = None
    band = None
    range_signal = None
    intersection_signal = None

    def set_band(self, band):
        self.band = dict(band)

    def on_close(self, timestamp, price, sigma):
        if self.paused:
            return super().on_close(timestamp, price, sigma)
        cutoff = ((timestamp + OFFSET) // DAY) * DAY - OFFSET
        if self.band is None or self.band["effective_at"] != cutoff:
            raise ExecutionError("current_range_reference_required")
        inside = self.band["lower"] <= price <= self.band["upper"]
        # Keep the research wrapper's first-schedule ordering exactly.
        if not self.buy_at and not self.sell_at:
            import math

            if math.isfinite(sigma):
                self.schedule(timestamp, price, sigma)
        # Preserve the frozen replay's floating-point operation order as well.
        self.buy_threshold = self.config.buy_threshold * (.725 / 1.4 if inside else 1)
        self.sell_threshold = self.config.sell_threshold * (1.075 / 1.2 if inside else 1)
        return super().on_close(timestamp, price, sigma)

    def make_plan(self, side, timestamp, price, threshold, expires, **kwargs):
        cutoff = ((timestamp + OFFSET) // DAY) * DAY - OFFSET
        if kwargs.get("startup") or self.band is None or self.band["effective_at"] != cutoff:
            raise ExecutionError("range_policy_completed_startup_required")
        p = self.policy
        inside = self.band["lower"] <= price <= self.band["upper"]
        expected = (
            (p.inside_buy_threshold if inside else p.outside_buy_threshold)
            if side == "buy"
            else (p.inside_sell_threshold if inside else p.outside_sell_threshold)
        )
        if abs(threshold - expected) > 1e-14:
            raise ExecutionError("range_signal_threshold_mismatch")
        fraction = p.inside_buy_fraction if inside else p.outside_buy_fraction
        if side == "buy" and p.pause_upper_p75_buys:
            p75 = self.price_position_reference
            if (
                self.price_position_day != (timestamp + OFFSET) // DAY
                or p75 is None
                or not math.isfinite(p75)
                or p75 <= 0
            ):
                raise ExecutionError("current_intersection_p75_reference_required")
            if upper_p75_intersection(price, self.band["upper"], p75):
                self.intersection_signal = {
                    "timestamp": timestamp,
                    "side": side,
                    "signal_price": price,
                    "band": dict(self.band),
                    "p75": p75,
                    "base_buy_fraction": fraction,
                    "p75_buy_multiplier": self.config.price_position_buy_multiplier,
                    "original_desired_budget": (
                        float(self.book.deployable)
                        * fraction
                        * self.config.price_position_buy_multiplier
                    ),
                    "selected_desired_budget": 0.0,
                    "result": "new_buy_skipped",
                    "reason": "upper_and_p75_new_buy_paused",
                }
                self.reason = "upper_and_p75_new_buy_paused"
                self.record(
                    "blocked", timestamp,
                    **{k: v for k, v in self.intersection_signal.items() if k != "timestamp"},
                )
                # No native plan/sequence increment, cash preparation or reservation.
                return None
        original = self.config
        if side == "buy":
            self.config = original.model_copy(update={"buy_fraction": fraction})
        try:
            plan = super().make_plan(side, timestamp, price, threshold, expires, **kwargs)
        finally:
            self.config = original
        if plan is not None:
            self.range_signal = {
                "plan_id": plan.plan_id,
                "timestamp": timestamp,
                "side": side,
                "inside": inside,
                "band": dict(self.band),
                "base_buy_fraction": fraction if side == "buy" else None,
                "threshold": threshold,
                "planned_quantity": plan.quantity,
                "reserved_budget": plan.budget,
            }
        return plan

    def state(self):
        state = {**super().state(), "band": self.band, "range_signal": self.range_signal}
        if self.intersection_signal is not None:
            state["intersection_signal"] = self.intersection_signal
        return state


def validate_binding(strategy, execution, control):
    b = strategy["range_policy"]
    if (
        set(b)
        != {
            "policy",
            "account_id",
            "source_sha256",
            "profit_started_at",
            "execution_created_at",
            "initial_capital",
            "history_sha256",
            "history_first_open_at",
            "history_rows",
        }
        or control.get("range_policy_sha256") != digest(b)
        or b["account_id"] != strategy["account_id"]
        or b["account_id"] != execution["account_id"]
        or b["account_id"] != control["account_id"]
        or b["source_sha256"] != strategy["source_sha256"]
        or b["source_sha256"] != control["source_sha256"]
        or digest(strategy["source_snapshot"]) != b["source_sha256"]
        or b["profit_started_at"] != strategy.get("activated_at")
        or b["execution_created_at"] != execution["created_at"]
        or b["initial_capital"] != execution["book"]["initial"]
        or strategy.get("startup_complete") is not True
    ):
        raise ExecutionError("range_policy_binding_mismatch")
    p = RangePolicy.model_validate(b["policy"])
    cfg = strategy["engine"]["config"]
    if (
        cfg["variant"] != "dynamic_period_only"
        or cfg["buy_fraction"] != p.inside_buy_fraction
        or cfg["buy_threshold"] != p.outside_buy_threshold
        or cfg["sell_threshold"] != p.outside_sell_threshold
        or not cfg["price_position_enabled"]
    ):
        raise ExecutionError("range_policy_base_configuration_mismatch")
    return p


def restore_engine(strategy, execution, root):
    control_path = root.parent.parent / "live-strategy-control.json"
    control = json.loads(control_path.read_text())
    policy = validate_binding(strategy, execution, control)
    b = strategy["range_policy"]
    history = BandHistory.load(
        root / HISTORY_FILE, b["history_sha256"], b["history_first_open_at"], b["history_rows"]
    )
    engine = RangeEngine.restore(strategy["engine"])
    engine.policy = policy
    engine.band = None  # Recompute from causal history; never trust a stale saved daily band.
    engine.band_history = history
    return engine


def preview(strategy, execution, control, history, recent, now):
    """Pure preview. No broker, stores, save(), queue(), or initializer is used."""
    from .config import ResearchConfig
    from .live_funding import LiveFunding

    s, e, c = deepcopy(strategy), deepcopy(execution), deepcopy(control)
    if (
        s.get("startup_complete") is not True
        or not s.get("activated_at")
        or s["account_id"] != e["account_id"]
        or s["account_id"] != c["account_id"]
        or s["source_sha256"] != c["source_sha256"]
        or digest(s["source_snapshot"]) != s["source_sha256"]
    ):
        raise ExecutionError("completed_bound_strategy_required")
    policy = RangePolicy()
    band = history.at(now, recent)
    cfg = ResearchConfig.from_saved(s["engine"]["config"])
    funding = LiveFunding(e, cfg, s["markets"], s["terms"])
    book = funding.planning_book()
    known = recent.loc[pd_close_times(recent) <= now]
    if known.empty:
        raise ExecutionError("finalized_market_data_stale")
    last = known.iloc[-1]
    closed_at = int(last.date.timestamp() * 1000) + 60000
    if not 0 <= now - closed_at <= 120000:
        raise ExecutionError("finalized_market_data_stale")
    from .price_position import build_price_reference

    p75 = build_price_reference(recent, cfg).value_at(closed_at)
    if p75 is None:
        raise ExecutionError("complete_p75_history_required")
    price = float(last.close)
    high = price > p75
    multiplier = money(cfg.price_position_buy_multiplier if high else 1)
    available = funding.available_cash()
    anchor = money(e["anchor"])
    scenarios = []
    for name, fraction, buy, sell in (
        ("current", cfg.buy_fraction, cfg.buy_threshold, cfg.sell_threshold),
        ("proposed_inside", 0.12, 0.00725, 0.01075),
        ("proposed_outside", 0.09, 0.014, 0.012),
    ):
        budget = min(available, book.deployable * money(fraction) * multiplier)
        scenarios.append(
            {
                "name": name,
                "base_buy_fraction": fraction,
                "p75_buy_multiplier_at_latest_close": str(multiplier),
                "estimated_budget_usdt": str(max(money(0), budget)),
                "buy_threshold": buy,
                "sell_threshold": sell,
                "buy_target": str(anchor * (1 - money(buy))),
                "sell_target": str(anchor * (1 + money(sell))),
            }
        )
    blockers = []
    if s.get("intent") or e.get("operation") or s["engine"].get("pending"):
        blockers.append("existing_intent_or_plan_must_finish_before_switch")
    if s.get("error") or e.get("error") or s.get("phase") == "blocked":
        blockers.append("existing_reconciliation_or_strategy_error")
    return {
        "schema": "trx-range-switch-preview-v1",
        "observed_at": now,
        "policy": policy.model_dump(),
        "current_daily_band": band,
        "latest_closed_at": closed_at,
        "latest_close": price,
        "latest_close_inside": band["lower"] <= price <= band["upper"],
        "p75": p75,
        "scenarios": scenarios,
        "budget_basis": "same_latest_confirmed_close_p75_tier_estimate_not_an_order",
        "financial_fingerprint": digest(execution),
        "book_initial": e["book"]["initial"],
        "net_capital_flows": e["book"]["external_flows"],
        "trx_remaining_cost": str(
            sum((money(lot["remaining_cost"]) for lot in e["book"]["lots"]), money(0))
        ),
        "profit_started_at": s["activated_at"],
        "blockers": blockers,
        "formal_activation_performed": False,
        "exchange_write_requests": 0,
    }


def pd_close_times(recent):
    import pandas as pd

    return pd.DatetimeIndex(recent.date).as_unit("ms").asi8 + 60000


def intersection_preview(strategy, execution, control, history, recent, now):
    """Compare a bound, completed range strategy without activating or writing it."""
    current = validate_binding(strategy, execution, control)
    result = preview(strategy, execution, control, history, recent, now)
    proposed = current.model_copy(update={"pause_upper_p75_buys": True})
    price, p75, band = result["latest_close"], result["p75"], result["current_daily_band"]
    inside = result["latest_close_inside"]
    fraction = current.inside_buy_fraction if inside else current.outside_buy_fraction
    from .config import ResearchConfig
    from .live_funding import LiveFunding

    cfg = ResearchConfig.from_saved(strategy["engine"]["config"])
    funding = LiveFunding(deepcopy(execution), cfg, strategy["markets"], strategy["terms"])
    book = funding.planning_book()
    multiplier = money(cfg.price_position_buy_multiplier if price > p75 else 1)
    intersection = upper_p75_intersection(price, band["upper"], p75)
    scenarios = []
    for name, policy in (("current", current), ("proposed", proposed)):
        skipped = policy.pause_upper_p75_buys and intersection
        desired = book.deployable * money(fraction) * multiplier
        scenarios.append({
            "name": name,
            "pause_upper_p75_buys": policy.pause_upper_p75_buys,
            "base_buy_fraction": fraction,
            "p75_buy_multiplier": str(multiplier),
            "intersection_multiplier": 0 if skipped else 1,
            "desired_budget_before_filter": str(desired),
            "estimated_budget_usdt": str(
                money(0) if skipped else max(money(0), min(funding.available_cash(), desired))
            ),
            "new_buy_skipped_at_latest_close": skipped,
            "buy_threshold": (
                current.inside_buy_threshold if inside else current.outside_buy_threshold
            ),
            "sell_threshold": (
                current.inside_sell_threshold if inside else current.outside_sell_threshold
            ),
        })
    return {
        **result,
        "schema": "trx-upper-p75-switch-preview-v1",
        "current_policy": current.model_dump(),
        "policy": proposed.model_dump(),
        "upper_p75_intersection_at_latest_close": intersection,
        "scenarios": scenarios,
        "budget_basis": "latest_confirmed_close_regime_p75_and_filter_estimate_not_an_order",
        "activation_requires_fresh_preflight": True,
        "usdt_interest_policy": deepcopy(execution.get("outside_usdt_income_policy")),
    }
