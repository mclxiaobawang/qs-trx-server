# ruff: noqa: RUF001
"""Read-only replay observations. The observer never supplies trading decisions."""

import csv
import io
import itertools
import json
import math
import sqlite3
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

from freqtrade.strategy.trx_phase1.fx_diagnostics import FxAudit


SCHEMA = "trx-diagnostics-v4"
SUPPORTED_SCHEMAS = {"trx-diagnostics-v3", SCHEMA}


SAMPLE_COLUMNS = [
    "timestamp",
    "price",
    "equity",
    "cash",
    "free_cash",
    "usdgo",
    "usdgo_value",
    "locked_usdgo",
    "principal_trx",
    "interest_trx",
    "remaining_cost",
    "deployable",
    "exposure",
    "occupation",
    "realized_profit",
    "principal_unrealized",
    "stable_interest",
    "fx_profit",
    "fx_unrealized",
    "trx_credit_value",
    "trx_interest_value",
    "gross_fees",
    "credited_rebates",
    "external_flows",
    "buy_threshold",
    "sell_threshold",
    "buy_period",
    "sell_period",
    "state",
    "reason",
    "fx_valuation_known",
    "usdt_earn_eligible",
    "usdgo_earn_eligible",
    "redemption_usdgo_value",
    "redemption_principal_trx",
    "anchor",
    "buy_lower_clipped",
    "buy_upper_clipped",
    "sell_lower_clipped",
    "sell_upper_clipped",
    "redemption_usdt",
    "recovery_reference",
]


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encoded(value):
    return json.dumps(clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def calendar(timestamp, timezone="Asia/Shanghai", pattern="%Y-%m"):
    return (
        datetime.fromtimestamp(timestamp / 1000, UTC)
        .astimezone(ZoneInfo(timezone))
        .strftime(pattern)
    )


def distribution(values):
    if not values:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": len(data),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.9)),
        "maximum": float(np.max(data)),
    }


class Recorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix(".building")
        self.db = sqlite3.connect(self.temporary)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        fields = ",".join(
            f"{key} {'TEXT' if key in ('state', 'reason') else 'REAL'}" for key in SAMPLE_COLUMNS
        )
        self.db.execute(f"CREATE TABLE samples ({fields})")
        self.db.execute("CREATE INDEX samples_time ON samples(timestamp)")
        self.db.execute(
            "CREATE TABLE records (kind TEXT,id TEXT,timestamp REAL,side TEXT,"
            "status TEXT,reason TEXT,payload TEXT)"
        )
        self.db.execute("CREATE INDEX record_time ON records(kind,timestamp)")
        self.samples, self.records = [], []
        self.opportunities = {}
        self.evaluations = {}
        self.conditions = {}
        self.environment = Counter()
        self.no_signal = {"buy": Counter(), "sell": Counter()}
        self.peak, self.episode = None, None
        self.drawdowns = []
        self.fill_context = {}
        self.period_switches = 0
        self.sample_count = 0
        self.active_signals, self.aliases = {}, {}
        self.arrived_plans = set()
        self.fx = FxAudit()

    def abort(self):
        self.db.close()
        self.temporary.unlink(missing_ok=True)

    def write(self, kind, row):
        self.records.append(
            (
                kind,
                str(row.get("id", "")),
                row.get("timestamp", 0),
                row.get("side", ""),
                row.get("status", ""),
                row.get("reason", ""),
                encoded(row),
            )
        )
        if len(self.records) >= 1000:
            self.flush()

    def flush(self):
        if self.samples:
            self.db.executemany(
                f"INSERT INTO samples VALUES ({','.join('?' for _ in SAMPLE_COLUMNS)})",
                self.samples,
            )
            self.samples.clear()
        if self.records:
            self.db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", self.records)
            self.records.clear()

    @staticmethod
    def context(engine, timestamp, price=None):
        price = price or engine.last_price
        book, cfg = engine.book, engine.config
        w, u, occupation = engine.occupation(price)
        allocations = book.qualifying(
            price, float(book.principal), cfg.sell_fee, cfg.execution_margin
        )
        return {
            "timestamp": timestamp,
            "anchor": engine.anchor,
            "price": price,
            "volatility": engine.last_sigma,
            "exposure": w,
            "cost_occupation": u,
            "occupation": occupation,
            "deployable": float(book.deployable),
            "available_cash": float(book.free_cash),
            "redemption_usdt": float(book.locked["USDT"]),
            "recovery_reference": engine.recovery_reference,
            "usdgo": float(book.usdgo),
            "locked_usdgo": float(book.locked["USDGO"]),
            "principal_trx": float(book.principal),
            "interest_trx": float(book.interest_trx),
            "qualifying_quantity": sum(item["quantity"] for item in allocations),
            "qualifying_lots": len(allocations),
        }

    def event(self, kind, engine, timestamp, values):  # noqa: C901
        cfg = engine.config
        if kind == "schedule":
            for side in ("buy", "sell"):
                if not values.get(side):
                    continue
                period = getattr(engine, side + "_period")
                threshold = getattr(engine, side + "_threshold")
                occupation = values["occupation"]
                base = cfg.buy_threshold if side == "buy" else cfg.sell_threshold
                dynamic = cfg.mechanisms["thresholds"]
                vol = (
                    cfg.volatility_multiplier * values["sigma"] * math.sqrt(period / 60)
                    if dynamic
                    else 0
                )
                feedback = (
                    (
                        cfg.buy_occupation_multiplier
                        if side == "buy"
                        else -cfg.sell_occupation_multiplier
                    )
                    * occupation
                    if dynamic
                    else 0
                )
                old = self.conditions.get(side)
                row = {
                    "timestamp": timestamp,
                    "side": side,
                    "period": period,
                    "threshold": threshold,
                    "base": base,
                    "volatility_adjustment": vol,
                    "occupation_adjustment": feedback,
                    "unclipped": base + vol + feedback,
                    "lower_clipped": dynamic and base + vol + feedback < cfg.minimum_threshold,
                    "upper_clipped": dynamic and base + vol + feedback > cfg.maximum_threshold,
                    "occupation": occupation,
                    "volatility": values["sigma"],
                    "due_at": getattr(engine, side + "_at"),
                    "tier": engine.tier,
                    "reason": "occupation_or_volatility_tier"
                    if cfg.mechanisms["periods"]
                    else "fixed_period",
                    "relax_count": engine.relax_count,
                }
                self.conditions[side] = row
                if old is None or old["period"] != period:
                    self.period_switches += old is not None
                    self.write(
                        "period_switches",
                        {**row, "previous_period": old["period"] if old else None},
                    )
            return
        if kind == "environment":
            self.environment[values["reason"]] += 1
            return
        if kind == "evaluation":
            side = values["side"]
            row = {
                **self.context(engine, timestamp),
                **self.conditions.get(side, {}),
                **values,
                "timestamp": timestamp,
            }
            row["reason"] = "price_condition_met" if values["met"] else "price_change_insufficient"
            self.evaluations[side] = row
            self.write("evaluations", row)
            if not values["met"]:
                self.no_signal[side]["price_change_insufficient"] += 1
                self.active_signals.pop(side, None)
            return
        if kind == "not_due":
            for side in values["sides"]:
                self.no_signal[side][values.get("reason", "not_selected_close")] += 1
            return
        if kind == "signal":
            side, signal_id = values["side"], values["id"]
            key = (
                engine.anchor_version,
                side,
                values.get("reference_kind", "shared"),
                values.get("reference_anchor", engine.anchor),
            )
            existing = self.active_signals.get(side)
            if existing and existing[0] == key:
                self.aliases[signal_id] = existing[1]
                row = self.opportunities[existing[1]]
                row["rechecks"] = row.get("rechecks", 0) + 1
                row["last_signal_at"] = timestamp
                row["last_attempt"] = {
                    **self.context(engine, timestamp),
                    **self.evaluations.get(side, {}),
                    **values,
                    "timestamp": timestamp,
                }
                return
            row = {
                **self.context(engine, timestamp),
                **self.evaluations.get(side, {}),
                **values,
                "timestamp": timestamp,
                "side": side,
                "id": signal_id,
                "status": "formed",
                "reason": "",
                "tags": [],
                "fills": 0,
                "submitted_orders": 0,
                "filled_quantity": 0,
                "filled_notional": 0,
            }
            self.opportunities[signal_id] = row
            self.active_signals[side] = (key, signal_id)
            return
        if kind in ("sizing", "qualified", "blocked", "submitted", "execution_check", "finished"):
            row = self.opportunities.get(self.aliases.get(values["id"], values["id"]))
            if row is None:
                return
            if kind == "submitted":
                row["submitted_orders"] += 1
                row["submitted_quantity"] = values.get("quantity", row.get("target_quantity", 0))
                row["submitted_at"] = timestamp
                row["status"] = "submitted"
            elif kind == "execution_check":
                reason = values["reason"]
                if reason and reason not in row["tags"]:
                    row["tags"].append(reason)
                row["execution_check"] = {**values, "timestamp": timestamp}
            elif kind == "blocked":
                row.update(status="ineligible", reason=values["reason"], ended_at=timestamp)
            elif kind == "finished":
                quantity = row.get("submitted_quantity", row.get("target_quantity", 0))
                if row["fills"]:
                    row["status"] = (
                        "filled"
                        if row["filled_quantity"] >= quantity - 1e-8
                        else "partially_executed_closed"
                    )
                    row["reason"] = "" if row["status"] == "filled" else "unfilled_remainder_closed"
                else:
                    row["status"] = "not_executed"
                    row["reason"] = values["reason"]
                row["ended_at"] = timestamp
                row["anchor_after"] = engine.anchor
            else:
                row.update({key: value for key, value in values.items() if key != "id"})
                if kind == "qualified":
                    row["qualified"] = True
                    row["reason"] = ""
                    limited = (
                        row.get("planned_quantity", 0)
                        > row.get("target_quantity", 0) + 10**-cfg.quantity_decimals
                        if row["side"] == "sell"
                        else row.get("budget", 0) + 1e-8 < row.get("planned_budget", 0)
                    )
                    if limited:
                        tag = (
                            "profitable_quantity_limited"
                            if row["side"] == "sell"
                            else "available_budget_limited"
                        )
                        if tag not in row["tags"]:
                            row["tags"].append(tag)
                    row["status"] = (
                        "funding" if row.get("ready_at", timestamp) > timestamp else "ready"
                    )
            return
        if kind == "fill":
            row = self.opportunities.get(self.aliases.get(values["id"], values["id"]))
            if row:
                row["fills"] += 1
                row["filled_quantity"] += values["quantity"]
                row["filled_notional"] += values["quantity"] * values["price"]
                row.setdefault("first_fill_at", timestamp)
                row["last_fill_at"] = timestamp
                self.fill_context.setdefault(
                    values["fill_id"], self.context(engine, timestamp, values["price"])
                )
            return
        if kind == "anchor":
            self.active_signals.clear()
            self.write("anchors", {"timestamp": timestamp, **values})
        if kind == "recovery":
            self.active_signals.pop("buy", None)
            self.write(
                "anchors", {"timestamp": timestamp, "source": "buy_reference_recovery", **values}
            )

    def operation(self, kind, timestamp, **values):
        if values.get("plan_id"):
            values["opportunity_id"] = self.aliases.get(values["plan_id"], values["plan_id"])
        if kind == "subscribe" and values.get("coin") == "USDGO":
            self.fx.subscribe(values["quantity"], timestamp)
        self.write("operations", {"timestamp": timestamp, "operation": kind, **values})

    def conversion(self, row, mark, forecast):
        cost = (
            row.get("paid", row["quantity"]) - row["quantity"] * mark
            if row["side"] == "to_usdgo"
            else row["quantity"] * mark - row.get("received", row["quantity"])
        )
        self.write(
            "conversions",
            {
                **row,
                "id": str(row["timestamp"]),
                "reference_mark": mark,
                "comprehensive_cost": cost,
                "forecast": forecast,
            },
        )

    def before_close(self, engine):
        self.active = {
            "buy_threshold": engine.buy_threshold,
            "sell_threshold": engine.sell_threshold,
            "buy_period": engine.buy_period,
            "sell_period": engine.sell_period,
            "conditions": {side: dict(row) for side, row in self.conditions.items()},
        }

    def initial(self, timestamp, price, equity):
        self.peak = {"timestamp": timestamp, "price": price, "equity": equity}
        self.peak["funds"] = {
            "timestamp": timestamp,
            "cash": equity,
            "free_cash": equity,
            "principal_trx": 0,
            "interest_trx": 0,
            "usdgo_value": 0,
            "redemption_usdgo_value": 0,
            "exposure": 0,
        }
        self.peak_lots = {}

    def before_fill(self, engine, event_id, timestamp, price):
        self.fill_context[event_id] = {
            **self.context(engine, timestamp, price),
            "opportunity_id": self.aliases.get(engine.pending.plan_id, engine.pending.plan_id),
            "snapshot_timing": "before principal execution",
        }

    def arrival(self, engine, timestamp):
        plan = engine.pending
        if (
            plan
            and plan.side == "sell"
            and timestamp >= plan.ready_at
            and plan.plan_id not in self.arrived_plans
        ):
            self.arrived_plans.add(plan.plan_id)
            self.operation(
                "redemption_received",
                timestamp,
                coin="TRX",
                quantity=plan.quantity,
                plan_id=plan.plan_id,
                due_at=plan.ready_at,
            )

    @staticmethod
    def lot_snapshot(book, price):
        return {
            lot.lot_id: {
                "quantity": float(lot.quantity),
                "cost": float(lot.remaining_cost),
                "unrealized": float(lot.quantity) * price - float(lot.remaining_cost),
            }
            for lot in book.lots
            if lot.quantity
        }

    def sample(self, engine, earn, timestamp, price, equity):
        book = engine.book
        principal, cost = float(book.principal), float(book.cost)
        basis = float(book.deployable)
        exposure = (principal + float(book.interest_trx)) * price / equity
        occupation = min(1, max(exposure, cost / basis))
        state = (
            "principal_empty_wait"
            if not principal and not book.interest_trx
            else "interest_only"
            if not principal
            else "near_full"
            if occupation >= 0.8
            else "principal_held"
        )
        sg_value = float(book.usdgo) * float(book.usdgo_mark or 0)
        self.samples.append(
            [
                timestamp,
                price,
                equity,
                float(book.cash),
                float(book.free_cash),
                float(book.usdgo),
                sg_value,
                float(book.locked["USDGO"]),
                principal,
                float(book.interest_trx),
                cost,
                basis,
                exposure,
                occupation,
                float(book.realized),
                principal * price - cost,
                float(book.stable_interest),
                float(book.fx_profit),
                sg_value - float(book.usdgo_cost),
                float(earn.trx_credit_value),
                float(book.interest_trx) * price,
                float(book.fees),
                float(book.fee_rebates),
                float(book.external_flows),
                self.active["buy_threshold"],
                self.active["sell_threshold"],
                self.active["buy_period"],
                self.active["sell_period"],
                state,
                engine.reason,
                int(engine.config.fx_quote_available),
                float(earn.usdt_eligible),
                float(earn.usdgo_eligible),
                float(book.locked["USDGO"]) * float(book.usdgo_mark or 0),
                sum(
                    float(row["quantity"])
                    for row in earn.inactive_principal
                    if row.get("eligible_at") is None
                ),
                engine.anchor,
                *[
                    int(self.active["conditions"].get(side, {}).get(key, False))
                    for side in ("buy", "sell")
                    for key in ("lower_clipped", "upper_clipped")
                ],
                float(book.locked["USDT"]),
                engine.recovery_reference,
            ]
        )
        self.sample_count += 1
        if self.peak is None or equity >= self.peak["equity"]:
            if self.episode:
                self.episode.update(recovered_at=timestamp, end_at=timestamp)
                self.drawdowns.append(self.episode)
                self.episode = None
            self.peak = {"timestamp": timestamp, "equity": equity, "price": price}
            self.peak["funds"] = dict(zip(SAMPLE_COLUMNS, self.samples[-1], strict=True))
            self.peak_lots = self.lot_snapshot(book, price)
        elif self.episode is None or equity < self.episode["trough_equity"]:
            if self.episode is None:
                self.episode = {
                    "id": str(self.peak["timestamp"]),
                    "peak_at": self.peak["timestamp"],
                    "peak_equity": self.peak["equity"],
                    "peak_price": self.peak["price"],
                    "peak_lots": self.peak_lots,
                    "peak_funds": self.peak["funds"],
                    "recovered_at": None,
                }
            self.episode.update(
                trough_at=timestamp,
                trough_equity=equity,
                trough_price=price,
                drawdown=1 - equity / self.peak["equity"],
                end_at=timestamp,
                trough_lots=self.lot_snapshot(book, price),
                trough_funds=dict(zip(SAMPLE_COLUMNS, self.samples[-1], strict=True)),
            )
        if self.sample_count % 1000 == 0:
            self.drawdowns = sorted(self.drawdowns, key=lambda r: r["drawdown"], reverse=True)[:10]
            self.flush()

    def finish(self, result, engine, earn):
        for cycle in self.fx.finish(engine.last_close):
            self.write("fx_cycles", cycle)
        for receipt in earn.receipts:
            self.operation(
                "interest_credited",
                receipt["timestamp"],
                **{key: value for key, value in receipt.items() if key != "timestamp"},
            )
            self.operation(
                "accrual_completed",
                receipt["timestamp"] - 3600000,
                coin=receipt["coin"],
                quantity=receipt["eligible_quantity"],
                paid_at=receipt["timestamp"],
            )
        if self.episode:
            self.episode["end_at"] = engine.last_close
            self.drawdowns.append(self.episode)
        for row in self.opportunities.values():
            if row["status"] in ("formed", "ready", "submitted", "funding"):
                row.update(status="not_executed", reason="end_of_data", ended_at=engine.last_close)
            self.write("opportunities", row)
        reported = {row["id"]: row for row in engine.book.reported_fills()}
        for fill in engine.book.orders:
            self.write(
                "fills",
                {
                    **reported[fill["id"]],
                    **self.fill_context.get(fill["id"], {}),
                    "timestamp": fill["timestamp"],
                    "id": fill["id"],
                },
            )
        for lot in engine.book.lots:
            allocations = [
                part
                for row in reported.values()
                for part in row["allocations"]
                if part["lot_id"] == lot.lot_id
            ]
            sold = sum(part["quantity"] for part in allocations)
            ending = float(lot.quantity)
            qualified = (
                bool(
                    engine.book.qualifying(
                        engine.last_price,
                        ending,
                        engine.config.sell_fee,
                        engine.config.execution_margin,
                    )
                )
                if ending
                else False
            )
            exits = [
                row["timestamp"]
                for row in reported.values()
                if any(part["lot_id"] == lot.lot_id for part in row["allocations"])
            ]
            self.write(
                "lots",
                {
                    **lot.state(),
                    "id": lot.lot_id,
                    "timestamp": lot.bought_at,
                    "original_quantity": ending + sold,
                    "sold_quantity": sold,
                    "remaining_quantity": ending,
                    "remaining_cost": float(lot.remaining_cost),
                    "unit_cost": float(lot.unit_cost),
                    "qualified": qualified,
                    "realized_profit": sum(part["profit"] for part in allocations),
                    "unrealized_profit": ending * engine.last_price - float(lot.remaining_cost),
                    "status": "held" if ending else "exited",
                    "deployable": float(engine.book.deployable),
                    "held_minutes": (
                        (engine.last_close if ending else max(exits, default=lot.bought_at))
                        - lot.bought_at
                    )
                    / 60000,
                },
            )
        for episode in sorted(self.drawdowns, key=lambda item: item["drawdown"], reverse=True)[:10]:
            peak_lots, trough_lots = episode.pop("peak_lots"), episode.pop("trough_lots")
            episode["lot_changes"] = sorted(
                [
                    {
                        "lot_id": key,
                        "peak": peak_lots.get(key),
                        "trough": trough_lots.get(key),
                        "unrealized_change": trough_lots.get(key, {}).get("unrealized", 0)
                        - peak_lots.get(key, {}).get("unrealized", 0),
                    }
                    for key in set(peak_lots) | set(trough_lots)
                ],
                key=lambda item: item["unrealized_change"],
            )
            episode.update(
                timestamp=episode["peak_at"],
                decline_minutes=(episode["trough_at"] - episode["peak_at"]) / 60000,
                recovery_minutes=(
                    (episode["recovered_at"] - episode["trough_at"]) / 60000
                    if episode["recovered_at"]
                    else None
                ),
            )
            self.write("drawdowns", episode)
        self.flush()
        self.db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
        attribution = {
            "principal_realized": result["summary"]["realized_profit"],
            "principal_unrealized": result["summary"]["unrealized_profit"],
            "trx_interest_credit_value": float(earn.trx_credit_value),
            "trx_interest_price_pnl": float(engine.book.interest_trx) * engine.last_price
            + float(engine.book.interest_sale_proceeds)
            - float(earn.trx_credit_value),
            "stable_interest": float(engine.book.stable_interest),
            "fx_realized": float(engine.book.fx_profit),
            "fx_unrealized": float(engine.book.usdgo) * float(engine.book.usdgo_mark or 0)
            - float(engine.book.usdgo_cost),
        }
        expected = (
            result["summary"]["equity"]
            - engine.config.initial_cash
            - float(engine.book.external_flows)
        )
        metadata = {
            "schema": SCHEMA,
            "collected": True,
            "sample_count": self.sample_count,
            "parameters": result["config"],
            "provenance": result.get("provenance"),
            "data_fingerprint": result["data_fingerprint"],
            "start_utc": result["start_utc"],
            "end_utc": result["end_utc"],
            "warmup_minutes": result["warmup_minutes"],
            "source": {
                "exchange": "bitget",
                "pair": "TRX/USDT",
                "market": "spot",
                "timeframe": "1m",
            },
            "attribution": attribution,
            "equity_change_net_flows": expected,
            "attribution_residual": expected - sum(attribution.values()),
            "no_signal": self.no_signal,
            "environment": self.environment,
            "period_switch_count": self.period_switches,
            "funds_semantics": {
                "free_cash": "immediately deployable USDT; USDT Earn is modeled as liquid",
                "usdgo": "includes subscribed and pending accrual; requires conversion/redemption",
                "locked_usdgo": "awaiting redemption; not cash",
                "interest_trx": (
                    "retained interest, NAV and exposure included, ordinary sales excluded"
                ),
            },
            "fx_attribution": (
                "USDGO eligible income pro-rata; waiting withdrawals LIFO then "
                "eligible pro-rata; open cycle has no future exit cost"
            ),
            "model_support": {
                "submission": "virtual replay submission, not exchange acceptance",
                "orderbook_partial_fills": False,
                "exchange_rejections": False,
                "monthly_resets": False,
                "observed_history": True,
                "sample_timing": (
                    "minute close account state; time conditions locked before that close"
                ),
                "fees": (
                    "already included in principal/FX/interest-sale components; do not "
                    "subtract again"
                ),
            },
        }
        self.db.execute("INSERT INTO metadata VALUES ('overview',?)", (encoded(metadata),))
        self.db.commit()
        self.db.close()
        self.temporary.replace(self.path)
        return {"collected": True, "schema": metadata["schema"], "sample_count": self.sample_count}


def connect(path):
    db = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def available(path):
    if not Path(path).exists():
        return False
    with connect(path) as db:
        saved = json.loads(
            db.execute("SELECT value FROM metadata WHERE key='overview'").fetchone()[0]
        )
        return saved.get("schema") in SUPPORTED_SCHEMAS


def records(
    db, kind, start=None, stop=None, side=None, status=None, reason=None, offset=0, limit=100
):
    where, values = ["kind=?"], [kind]
    for key, value, operator in (
        ("timestamp", start, ">="),
        ("timestamp", stop, "<"),
        ("side", side, "="),
        ("status", status, "="),
        ("reason", reason, "="),
    ):
        if value is not None:
            where.append(f"{key}{operator}?")
            values.append(value)
    clause = " AND ".join(where)
    total = db.execute("SELECT count(*) FROM records WHERE " + clause, values).fetchone()[0]
    rows = db.execute(
        "SELECT payload FROM records WHERE " + clause + " ORDER BY timestamp,id LIMIT ? OFFSET ?",
        [*values, limit, offset],
    )
    return {"total": total, "rows": [json.loads(row[0]) for row in rows]}


def overview(path, timezone="Asia/Shanghai"):  # noqa: C901
    with connect(path) as db:
        result = json.loads(
            db.execute("SELECT value FROM metadata WHERE key='overview'").fetchone()[0]
        )
        if result["schema"] not in SUPPORTED_SCHEMAS:
            return {"collected": False, "message": "未采集当前诊断口径，需要重跑"}
        all_opportunities = records(db, "opportunities", limit=1000000)["rows"]
        fills = records(db, "fills", limit=1000000)["rows"]
        months, periods, states = {}, {"buy": {}, "sell": {}}, Counter()
        gaps = []

        def month_for(timestamp):
            key = calendar(timestamp, timezone)
            return months.setdefault(
                key,
                {
                    "month": key,
                    "buy_decisions": 0,
                    "sell_decisions": 0,
                    "buy_orders": 0,
                    "sell_orders": 0,
                    "buy_fills": 0,
                    "sell_fills": 0,
                    "active_days": set(),
                    "minutes": 0,
                },
            )

        start = int(db.execute("SELECT min(timestamp) FROM samples").fetchone()[0]) - 60000
        end = int(db.execute("SELECT max(timestamp) FROM samples").fetchone()[0])
        executions = sorted({row["timestamp"] for row in fills})
        points = [start, *executions, end]
        gaps = [
            {"start": a, "end": b, "minutes": (b - a) / 60000}
            for a, b in itertools.pairwise(points)
            if b > a
        ]
        for row in db.execute("SELECT timestamp,state,buy_period,sell_period,anchor FROM samples"):
            key = calendar(row[0] - 60000, timezone)
            month = months.setdefault(
                key,
                {
                    "month": key,
                    "buy_decisions": 0,
                    "sell_decisions": 0,
                    "buy_orders": 0,
                    "sell_orders": 0,
                    "buy_fills": 0,
                    "sell_fills": 0,
                    "active_days": set(),
                    "minutes": 0,
                },
            )
            month["minutes"] += 1
            states[row[1]] += 1
            if not row[4]:
                continue
            for side, period in (("buy", int(row[2])), ("sell", int(row[3]))):
                periods[side].setdefault(
                    period, {"period": period, "minutes": 0, "evaluations": 0, "signals": 0}
                )["minutes"] += 1
        for row in all_opportunities:
            month = month_for(row["timestamp"])
            month[row["side"] + "_decisions"] += 1
            period = int(row.get("period", result["parameters"]["periods"][0]))
            periods[row["side"]].setdefault(
                period, {"period": period, "minutes": 0, "evaluations": 0, "signals": 0}
            )["signals"] += 1
            if row["submitted_orders"]:
                submitted_month = month_for(row["submitted_at"])
                submitted_month[row["side"] + "_orders"] += row["submitted_orders"]
        for row in fills:
            month = month_for(row["timestamp"])
            month[row["side"] + "_fills"] += 1
            month["active_days"].add(calendar(row["timestamp"], timezone, "%Y-%m-%d"))
        evaluations = records(db, "evaluations", limit=1000000)["rows"]
        for row in evaluations:
            period = int(row["period"])
            periods[row["side"]].setdefault(
                period, {"period": period, "minutes": 0, "evaluations": 0, "signals": 0}
            )["evaluations"] += 1
        for month in months.values():
            month["active_days"] = len(month["active_days"])
        funnel = {}
        waiting_ids = {
            row.get("opportunity_id", row.get("plan_id"))
            for row in (
                json.loads(value[0])
                for value in db.execute(
                    "SELECT payload FROM records WHERE kind='operations' "
                    "AND json_extract(payload,'$.operation')='redemption_requested' "
                    "AND json_extract(payload,'$.due_at')>timestamp"
                )
            )
        }
        for side in ("buy", "sell"):
            rows = [row for row in all_opportunities if row["side"] == side]
            times = sorted({row["timestamp"] for row in fills if row["side"] == side})
            funnel[side] = {
                "evaluations": sum(row["side"] == side for row in evaluations),
                "signals": len(rows),
                "qualified": sum(bool(row.get("qualified")) for row in rows),
                "funding_wait": sum(row["id"] in waiting_ids for row in rows),
                "submitted": sum(row["submitted_orders"] for row in rows),
                "executed_decisions": sum(bool(row["fills"]) for row in rows),
                "fill_fragments": sum(row["fills"] for row in rows),
                "outcomes": dict(Counter(row["status"] for row in rows)),
                "terminal_reasons": dict(
                    Counter(row["reason"] for row in rows if not row["fills"])
                ),
                "gaps_minutes": distribution(
                    [(b - a) / 60000 for a, b in itertools.pairwise(times)]
                ),
            }
        thresholds = {}
        for side in ("buy", "sell"):
            values = [
                row[0] for row in db.execute(f"SELECT {side}_threshold FROM samples WHERE anchor>0")
            ]
            actual = [row["threshold"] for row in evaluations if row["side"] == side]
            minimum, maximum = (
                result["parameters"]["minimum_threshold"],
                result["parameters"]["maximum_threshold"],
            )
            thresholds[side] = {
                "time": distribution(values),
                "decisions": distribution(actual),
                "at_lower_ratio": sum(abs(v - minimum) < 1e-12 for v in values) / len(values)
                if values
                else None,
                "at_upper_ratio": sum(abs(v - maximum) < 1e-12 for v in values) / len(values)
                if values
                else None,
            }
        notionals = {
            side: [row["quantity"] * row["price"] for row in fills if row["side"] == side]
            for side in ("buy", "sell")
        }
        lots = records(db, "lots", limit=1000000)["rows"]
        result.update(
            timezone=timezone,
            months=list(months.values()),
            funnel=funnel,
            periods={side: list(rows.values()) for side, rows in periods.items()},
            state_minutes=dict(states),
            thresholds=thresholds,
            longest_no_trade=max(gaps, key=lambda r: r["minutes"]) if gaps else None,
            no_trade_intervals=gaps,
            amounts={
                side: {
                    "total": sum(values),
                    **distribution(values),
                    "deployable_fraction": distribution(
                        [
                            row["quantity"] * row["price"] / row["deployable"]
                            for row in fills
                            if row["side"] == side and row.get("deployable", 0) > 0
                        ]
                    ),
                    "planned_total": sum(
                        row.get("planned_notional", 0)
                        for row in all_opportunities
                        if row["side"] == side
                    ),
                    "constrained_total": sum(
                        row.get("constrained_notional", 0)
                        for row in all_opportunities
                        if row["side"] == side
                    ),
                }
                for side, values in notionals.items()
            },
            lots_summary={
                "loss_locked_cost": sum(
                    row["remaining_cost"] for row in lots if row["unrealized_profit"] < 0
                ),
                "held_duration": distribution(
                    [row["held_minutes"] for row in lots if row["status"] == "held"]
                ),
                "exited_duration": distribution(
                    [row["held_minutes"] for row in lots if row["status"] == "exited"]
                ),
                "stale_cost_30_days": sum(
                    row["remaining_cost"]
                    for row in lots
                    if row["status"] == "held" and row["held_minutes"] >= 30 * 1440
                ),
                "stale_capital_ratio": sum(
                    row["remaining_cost"]
                    for row in lots
                    if row["status"] == "held" and row["held_minutes"] >= 30 * 1440
                )
                / (lots[0]["deployable"] if lots else 1),
            },
            cash_only=not fills,
            drawdowns=sorted(
                records(db, "drawdowns", limit=10)["rows"],
                key=lambda row: row["drawdown"],
                reverse=True,
            ),
            fx_cycles_summary={
                "closed": db.execute(
                    "SELECT count(*) FROM records WHERE kind='fx_cycles' AND status='closed'"
                ).fetchone()[0],
                "open": db.execute(
                    "SELECT count(*) FROM records WHERE kind='fx_cycles' AND status='open'"
                ).fetchone()[0],
            },
        )
        return result


def sample_rows(db, start=None, stop=None, limit=2000):
    bounds, values = [], []
    for value, operator in ((start, ">="), (stop, "<")):
        if value is not None:
            bounds.append("timestamp" + operator + "?")
            values.append(value)
    clause = " WHERE " + " AND ".join(bounds) if bounds else ""
    count = db.execute("SELECT count(*) FROM samples" + clause, values).fetchone()[0]
    stride = max(1, math.ceil(count / limit))
    where = clause + (" AND " if clause else " WHERE ") + "(rowid % ?=0)"
    rows = [
        dict(row)
        for row in db.execute(
            "SELECT * FROM samples" + where + " ORDER BY timestamp", [*values, stride]
        )
    ]
    by_time = {row["timestamp"]: row for row in rows}
    for order in ("timestamp", "timestamp DESC", "equity", "equity DESC", "price", "price DESC"):
        row = db.execute(
            "SELECT * FROM samples" + clause + " ORDER BY " + order + " LIMIT 1", values
        ).fetchone()
        if row:
            by_time[row["timestamp"]] = dict(row)
    rows = sorted(by_time.values(), key=lambda row: row["timestamp"])
    return {"total": count, "stride": stride, "rows": rows}


def export_package(result, diagnostic_path, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + "." + str(uuid4()) + ".building")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        # Only research records are exported. Account/server configuration is never read.
        archive.writestr("result.json", encoded(result))
        archive.writestr("effective_parameters.json", encoded(result["config"]))
        archive.writestr(
            "data_and_rules.json",
            encoded(
                {
                    key: result.get(key)
                    for key in (
                        "provenance",
                        "data_fingerprint",
                        "start_utc",
                        "end_utc",
                        "warmup_minutes",
                        "candles",
                        "assumptions",
                        "earn_assumptions",
                    )
                }
            ),
        )
        datasets = {
            key: result.get(key, [])
            for key in (
                "orders",
                "fills",
                "lots",
                "interest_receipts",
                "rebate_receipts",
                "cash_conversions",
                "cash_forecasts",
            )
        }
        for name, rows in datasets.items():
            archive.writestr(name + ".json", encoded(rows))
        if diagnostic_path and available(diagnostic_path):
            with connect(diagnostic_path) as db:
                with archive.open("continuous_equity_funds.csv", "w") as target:
                    text = io.TextIOWrapper(target, encoding="utf-8", newline="")
                    writer = csv.writer(text)
                    samples = db.execute("SELECT * FROM samples ORDER BY timestamp")
                    writer.writerow([column[0] for column in samples.description])
                    writer.writerows(tuple(row) for row in samples)
                    text.flush()
                    text.detach()
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='risk_samples'").fetchone():
                    with archive.open("continuous_principal_occupation.csv", "w") as target:
                        text = io.TextIOWrapper(target, encoding="utf-8", newline="")
                        writer = csv.writer(text)
                        samples = db.execute("SELECT * FROM risk_samples ORDER BY timestamp")
                        writer.writerow([column[0] for column in samples.description])
                        writer.writerows(tuple(row) for row in samples)
                        text.flush()
                        text.detach()
                for kind in (
                    "evaluations",
                    "opportunities",
                    "period_switches",
                    "anchors",
                    "operations",
                    "lots",
                    "drawdowns",
                    "conversions",
                    "fx_cycles",
                ):
                    with archive.open(kind + ".jsonl", "w") as target:
                        for row in db.execute(
                            "SELECT payload FROM records WHERE kind=? ORDER BY timestamp,id",
                            (kind,),
                        ):
                            target.write((row[0] + "\n").encode())
                archive.writestr("diagnostic_overview.json", encoded(overview(diagnostic_path)))
                collected_available = True
        else:
            collected_available = False
        archive.writestr(
            "README.txt",
            (
                "TRX 历史研究包。时间戳为 UTC Unix 毫秒；金额为研究账本记录。\n"
                "月度统计从同一次连续回放切分，没有重置本金或锚点。\n"
                "机会 ID 唯一；最终状态与主要终止原因只记一次。成交片段与交易决策不是同一口径。\n"
                "收入分解包含已入账费用及返佣，不可再次扣减；关闭理财重跑是独立路径。\n"
                "当前回放不模拟真实交易所拒单、盘口部分成交或真实账户理财。\n"
                "已观察历史的诊断不能重新声明为未见测试集。\n"
                + (
                    "完整诊断已采集。\n"
                    if collected_available
                    else "未采集新诊断，需要重跑；未根据汇总数字补造记录。\n"
                )
            ),
        )
    temp.replace(destination)
    return destination


def interval_details(path, start, stop):
    with connect(path) as db:
        first = db.execute(
            "SELECT * FROM samples WHERE timestamp>=? ORDER BY timestamp LIMIT 1", (start,)
        ).fetchone()
        last = db.execute(
            "SELECT * FROM samples WHERE timestamp<? ORDER BY timestamp DESC LIMIT 1", (stop,)
        ).fetchone()
        fills = records(db, "fills", start, stop, limit=2000)
        opportunities = records(db, "opportunities", start, stop, limit=2000)
        changes = (
            {
                key: last[key] - first[key]
                for key in (
                    "realized_profit",
                    "principal_unrealized",
                    "stable_interest",
                    "fx_profit",
                    "fx_unrealized",
                    "trx_credit_value",
                    "trx_interest_value",
                    "gross_fees",
                    "credited_rebates",
                )
            }
            if first and last
            else {}
        )
        return {
            "start": start,
            "stop": stop,
            "first": dict(first) if first else None,
            "last": dict(last) if last else None,
            "changes": changes,
            "fills": fills,
            "opportunities": opportunities,
            "samples": sample_rows(db, start, stop),
            "note": (
                "Balance changes between first and last minute-close snapshots; "
                "fees are included, not additional deductions."
            ),
        }
