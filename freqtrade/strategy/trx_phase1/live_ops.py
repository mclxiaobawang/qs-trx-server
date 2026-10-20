"""Independent read-only live views and durable alarms. Never creates a trading worker."""
# ruff: noqa: RUF001 - user-facing Chinese punctuation

import hashlib
import json
import math
import shutil
import sqlite3
import time
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN
from pathlib import Path
from urllib.request import Request, urlopen

from .auto_rebate import read_status as rebate_status
from .income_allocation import summary as income_summary
from .ledger import Ledger, money
from .live_observation import dashboard


class ViewUnavailable(Exception):
    pass


ALARM_TITLES = {
    "strategy_error": ("策略运行异常", "策略错误已解除"),
    "worker_stale": ("策略心跳更新超时", "策略心跳告警已解除"),
    "minute_stale": ("分钟行情更新超时", "行情更新告警已解除"),
    "reconciliation_stale": ("余额与理财对账更新超时", "余额与理财对账告警已解除"),
    "signal_overdue_buy": ("买入信号判断延迟", "买入信号延迟告警已解除"),
    "signal_overdue_sell": ("卖出信号判断延迟", "卖出信号延迟告警已解除"),
    "income_pending": ("理财利息归属处理超时", "理财利息归属告警已解除"),
    "source_unavailable": ("策略数据读取失败", "策略数据读取已恢复"),
    "disk_usage": ("服务器磁盘占用过高", "磁盘空间告警已解除"),
    "observer_memory": ("告警容器内存占用过高", "告警容器内存告警已解除"),
    "trading_memory": ("交易容器内存占用过高", "交易容器内存告警已解除"),
    "trading_memory_unavailable": ("交易容器内存监测数据异常", "交易容器内存指标告警已解除"),
    "backup_stale": ("自动备份更新超时", "自动备份更新告警已解除"),
    "backup_failed": ("自动备份失败", "自动备份错误已解除"),
    "backup_capacity": ("自动备份占用超过预算", "备份容量告警已解除"),
}

LIFECYCLE_TITLES = {
    "started": "策略启动成功",
    "paused": "已暂停新增委托",
    "resumed": "策略已恢复运行",
}


def lifecycle_status(state, execution, now):
    """Confirm running from persisted progress, never from a start-button request."""
    phase = state.get("phase")
    if state.get("desired") == "paused":
        if state.get("error") or phase == "blocked":
            return "paused", "已暂停新增委托，等待人工核对"
        if phase == "pausing" or execution.get("operation"):
            return "paused", "已暂停新增委托，既有执行与到账仍在核对"
        return "paused", "已暂停新增委托，保留现有持仓和账本"
    if state.get("desired") != "running":
        return None, "运行状态待核对"
    if state.get("error") or execution.get("error"):
        return None, "运行异常，等待恢复核对"
    engine = state.get("engine", {})
    stamps = (
        (state.get("last_poll_at"), 120000),
        (engine.get("last_close"), 180000),
        (execution.get("interest_checked_at"), 180000),
    )
    fresh = all(isinstance(t, (int, float)) and -5000 <= now - t <= limit for t, limit in stamps)
    if not state.get("startup_complete") or phase not in ("observing", "executing") or not fresh:
        return None, "启动或恢复检查中，尚未确认正常运行"
    text = "运行中，正在执行既有流程" if phase == "executing" else "运行中，等待新信号"
    return "running", text


def lifecycle_message(item, current):
    return (
        "状态变更确认时间："
        + beijing_time(item["observed_at"])
        + "\n变更确认时状态："
        + item["message"]
        + "\n当前状态："
        + current
        + "\n以上状态由独立监控核验；确认时间可能晚于点击时间。"
    )


def notification_content(code, transition, item, created_at, current=None):
    """Render durable event time separately from the status at delivery."""
    prefix = "【测试】" if item.get("test") else ""
    if transition in LIFECYCLE_TITLES:
        return (
            prefix + "[TRX策略状态] " + LIFECYCLE_TITLES[transition],
            lifecycle_message(item, current or "暂时无法读取，请查看实盘运行页面")
            + "\n请在实盘运行页面查看完整状态。",
        )
    if transition == "trade":
        title = "买入 TRX" if item["side"] == "buy" else "卖出 TRX"
        return (
            prefix + "[TRX策略成交] " + title,
            item["message"] + "\n同一订单如有分批成交，将按成交明细分别通知。",
        )
    recovered = transition == "recovered"
    title = ALARM_TITLES.get(code, ("其他运行异常", "其他运行告警已解除"))[recovered]
    message = item["message"]
    if recovered:
        message = (
            "恢复时间："
            + beijing_time(item.get("recovered_at", created_at))
            + "\n当前状态："
            + (item["message"] if item.get("recovered_at") is not None else "该项告警已解除。")
            + "\n以上为恢复时核验的状态。"
        )
    return (
        prefix + f"[TRX策略{'恢复' if recovered else '告警'}] " + title,
        message + "\n请访问真实账户页面核对。此通知不会触发暂停、重启或交易。",
    )


def memory_usage(root=Path("/sys/fs/cgroup")):
    """Read this container's cgroup v2 budget without loading process libraries."""
    try:
        current = int((root / "memory.current").read_text())
        limit = int((root / "memory.max").read_text())
        if current < 0 or limit <= 0:
            return None
        return {"current": current, "limit": limit, "percent": 100 * current / limit}
    except (OSError, ValueError):
        return None  # Non-cgroup hosts and unlimited budgets have no finite percentage.


def trading_memory(path, now):
    if path is None:
        return None
    try:
        path = Path(path)
        if path.stat().st_size > 65536:
            raise ValueError()
        value = json.loads(path.read_text())
        if (
            value.get("container_name") != "trx-phase1-live"
            or value.get("error")
            or not value.get("running")
            or not -5000 <= now - value["checked_at"] <= 90000
            or not isinstance(value["current"], int)
            or not isinstance(value["limit"], int)
            or value["current"] < 0
            or value["limit"] <= 0
        ):
            raise ValueError()
        return {**value, "percent": 100 * value["current"] / value["limit"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {"available": False}


def fill_message(fill):
    """Describe an already booked native fill; this function never places orders."""
    side = {"buy": "买入", "sell": "卖出"}[fill["side"]]
    stamp = datetime.fromtimestamp(fill["timestamp"] / 1000, timezone(timedelta(hours=8)))
    quantity, price, fee = (money(fill[k]) for k in ("quantity", "price", "fee"))
    if min(quantity, price) <= 0 or fee < 0:
        raise ValueError("invalid_booked_fill")
    lines = [
        f"TRX {side}已成交并核对入账（单笔成交明细）",
        f"成交时间：{stamp:%Y-%m-%d %H:%M:%S} 北京时间",
        f"成交数量：{quantity:.8f} TRX",
        f"成交价格：{price:.8f} USDT",
        f"成交额约：{quantity * price:.8f} USDT",
        f"手续费折合：{fee:.8f} USDT（返佣前）",
        f"订单号：{fill['order_id']}",
        f"成交编号：{fill['id']}",
    ]
    if fill.get("net_quantity") is not None:
        lines.insert(4, f"净到账数量：{money(fill['net_quantity']):.8f} TRX")
    if fill.get("profit") is not None:
        lines.append(f"该笔本金已实现利润：{money(fill['profit']):.8f} USDT")
    return "\n".join(lines)


def beijing_time(stamp):
    if stamp is None:
        return "暂无记录"
    value = datetime.fromtimestamp(stamp / 1000, timezone(timedelta(hours=8)))
    return f"{value:%Y-%m-%d %H:%M:%S} 北京时间（UTC+8）"


def activity_summary(execution, now):
    """Aggregate complete persisted evidence; never fetch, credit, or mutate assets."""
    book = Ledger.restore(execution["book"])
    fills = {row["id"]: row for row in book.reported_fills()}
    created = execution["created_at"]
    midnight = datetime.fromtimestamp(now / 1000, timezone(timedelta(hours=8))).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day = int(midnight.timestamp() * 1000)
    starts = {"today": max(created, day), "week": max(created, day - 6 * 86400000), "all": created}
    # Native USDGO conversion fills are not in the TRX order list. Their exact
    # receipt identity must already be in the ledger, including an in-flight op.
    conversions = {}
    groups = [e.get("fills", []) for e in execution.get("events", [])]
    groups.append(list((execution.get("operation") or {}).get("fills", {}).values()))
    for group in groups:
        for row in group:
            key = "spot:" + row["order_id"] + ":" + row["id"]
            if key in book.events and key not in fills:
                conversions[key] = row
    conversion_fees = [
        (
            row["timestamp"],
            sum(
                (
                    money(f["cost"])
                    * (money(row["quote"]) / money(row["quantity"]) if f["coin"] == "USDGO" else 1)
                    for f in row["fees"]
                ),
                money(0),
            ),
        )
        for row in conversions.values()
    ]
    allocations = (execution.get("income_accounting") or {}).get("allocations", {})
    income = {}
    for row in allocations.values():
        key = "income-allocation:" + row["bill_id"]
        if key in book.events:
            income[key] = row
    for row in execution.get("events", []):
        key = row.get("receipt_id")
        if row["kind"] == "credited_interest_imported" and key in book.events:
            income.setdefault(key, row)
    credited_keys = {k for k in book.events if k.startswith(("income:", "income-allocation:"))}
    complete_income = credited_keys == income.keys()
    # Refuse a plausible but incomplete fee/PnL subtotal if older evidence is missing.
    fees = sum((money(r["fee"]) for r in fills.values()), money(0)) + sum(
        (fee for _, fee in conversion_fees), money(0)
    )
    realized = sum(
        (money(r["profit"]) for r in fills.values() if r.get("profit") is not None), money(0)
    )
    fee_complete = abs(fees - book.fees) <= money("0.00000001")
    profit_complete = abs(realized - book.realized) <= money("0.00000001")
    periods = {}
    for name, start in starts.items():
        selected = [r for r in fills.values() if start <= r["timestamp"] <= now]
        native = {c: money(0) for c in ("TRX", "USDGO", "USDT")}
        for row in income.values():
            if start <= row["timestamp"] <= now:
                native[row["coin"]] += money(row["amount"])
        trx_fees = sum((money(r["fee"]) for r in selected), money(0))
        fx_fees = sum((fee for stamp, fee in conversion_fees if start <= stamp <= now), money(0))
        periods[name] = {
            "start": start,
            "end": now,
            "buy_fills": sum(r["side"] == "buy" for r in selected),
            "sell_fills": sum(r["side"] == "sell" for r in selected),
            "turnover": str(
                sum((money(r["quantity"]) * money(r["price"]) for r in selected), money(0))
            ),
            "realized_profit": str(
                sum((money(r["profit"]) for r in selected if r.get("profit") is not None), money(0))
            )
            if profit_complete
            else None,
            "interest": {c: str(v) if complete_income else None for c, v in native.items()},
            "trx_fees": str(trx_fees),
            "conversion_fees": str(fx_fees) if fee_complete else None,
            "gross_fees": str(trx_fees + fx_fees) if fee_complete else None,
        }
    return {
        "observed_at": now,
        "created_at": created,
        "timezone": "Asia/Shanghai",
        "complete": {"fees": fee_complete, "profit": profit_complete, "interest": complete_income},
        "periods": periods,
    }


def recovery_status(code, state, execution, metrics):
    """Describe evidence at recovery, without repeating the original alarm text."""
    state, execution = state or {}, execution or {}
    running = state.get("desired") == "running"
    phase = {
        "observing": "运行中，等待新信号",
        "executing": "运行中，正在执行既有流程",
        "recovering": "恢复检查中",
        "warming_up": "指标初始化中",
        "blocked": "等待人工核对",
        "paused": "已暂停",
    }.get(state.get("phase"), "状态待核对")
    strategy = f"策略状态：{phase if running else '已暂停'}。"
    if code == "strategy_error":
        return "策略错误状态已清除；" + strategy
    if code == "source_unavailable":
        return "策略快照已恢复可读；" + strategy
    if code in ("worker_stale", "minute_stale", "reconciliation_stale"):
        label, stamp = {
            "worker_stale": ("策略心跳", state.get("last_poll_at")),
            "minute_stale": ("确认分钟行情", state.get("engine", {}).get("last_close")),
            "reconciliation_stale": ("余额与理财到账核对", execution.get("interest_checked_at")),
        }[code]
        status = f"{label}已恢复更新" if running else f"策略已暂停，{label}时效监测不适用"
        return f"{status}；最近更新时间：{beijing_time(stamp)}。"
    if code.startswith("signal_overdue_"):
        return "信号超时告警已解除；" + strategy
    if code == "disk_usage":
        return f"磁盘占用已低于85%告警线，恢复时占用 {metrics['disk']['used_percent']:.2f}%。"
    if code in ("observer_memory", "trading_memory", "trading_memory_unavailable"):
        reading = metrics["memory" if code == "observer_memory" else "trading_memory"]
        label = "告警容器" if code == "observer_memory" else "交易容器"
        if reading and reading.get("percent") is not None:
            return f"{label}内存指标可读取，恢复时占用 {reading['percent']:.1f}%。"
        return f"{label}当前没有可用的有限内存额度指标，此项占用告警已解除。"
    if code == "income_pending":
        pending = len(execution.get("pending_income", {}))
        return f"已无超过15分钟待归属的利息；当前待归属 {pending} 条。"
    backup = metrics["backup"]
    if code == "backup_capacity":
        return "自动备份占用已回到容量预算内。"
    if code in ("backup_stale", "backup_failed"):
        status = "自动备份已恢复更新" if code == "backup_stale" else "自动备份错误状态已清除"
        return f"{status}；最近成功时间：{beijing_time(backup.get('last_success_at'))}。"
    return "该项监测告警已解除，请在真实账户页面查看其他运行指标。"


def read_state(path, key):
    """Read a short snapshot without creating a database or acquiring a writer lock."""
    try:
        with closing(
            sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.15)
        ) as db:
            db.execute("PRAGMA query_only=ON")
            row = db.execute("SELECT version,state FROM trx_state WHERE id=?", (key,)).fetchone()
        if row is None:
            raise ViewUnavailable("missing_state")
        return row[0], json.loads(row[1])
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise ViewUnavailable("source_unavailable") from exc


def floor(value, decimals):
    return money(value).quantize(money(1).scaleb(-int(decimals)), rounding=ROUND_DOWN)


def preview(state, engine, execution, book, panel, now):
    """Explain the existing execution fee reserve; never construct a Plan or unlock assets."""
    market = state.get("markets", {}).get("TRXUSDT", {})
    cfg = engine["config"]
    stamp = market.get("timestamp")
    common = {
        "observed_at": stamp,
        "fresh": bool(stamp and -5000 <= now - stamp <= 60000),
        "basis": "threshold_price_existing_execution_fee_reserve",
        "buy": None,
        "sell": None,
    }
    if not market or not panel["signal"]["principal_location_verified"]:
        return common
    fee = money(market["fee"])
    for side in ("buy", "sell"):
        signal = panel["signal"]["sides"][side]
        dynamic = cfg.get(side + "_quantity_dynamic")
        if dynamic or (
            dynamic is None
            and cfg["variant"] not in ("dynamic_period_only", "fixed", "dynamic_threshold_only")
        ):
            continue  # A fixed-fraction preview must not pretend to model dynamic sizing.
        target = money(signal["target"] or 0)
        if target <= 0 or signal["target_fraction"] is None:
            continue
        if side == "buy":
            if signal.get("target_budget_estimate") is None:
                continue
            budget = money(signal["target_budget_estimate"])
            quote = floor(budget / (1 + fee), market["price_decimals"])
            gross = quote / target
            common[side] = {
                "budget": str(budget),
                "order_quote": str(quote),
                "gross_trx": str(gross),
                "net_trx": str(gross * (1 - fee)),
                "fee_rate": str(fee),
                "fee_assumption": "buy_fee_in_trx",
                "meets_minimum": quote >= money(market["minimum_notional"])
                and gross * target >= money(cfg["min_notional"]),
            }
        else:
            eligible = money(signal.get("target_eligible_trx", 0))
            basis = eligible if cfg.get("sell_quantity_basis") == "profitable" else book.principal
            reserved = floor(basis * money(signal["target_fraction"]), market["quantity_decimals"])
            reserved = min(reserved, floor(eligible, market["quantity_decimals"]))
            if (
                cfg.get("tail_completion")
                and 0
                < book.principal - reserved
                < max(money(cfg["min_quantity"]), money(cfg["min_notional"]) / target)
                and eligible >= book.principal
            ):
                reserved = floor(book.principal, market["quantity_decimals"])
            quantity = floor(reserved / (1 + fee), market["quantity_decimals"])
            common[side] = {
                "target_trx": str(reserved),
                "order_trx": str(quantity),
                "net_usdt": str(quantity * target * (1 - fee)),
                "fee_rate": str(fee),
                "meets_minimum": reserved * target >= money(cfg["min_notional"])
                and quantity * target >= money(market["minimum_notional"]),
            }
    return common


class LiveView:
    def __init__(self, source, account_id, clock=None):
        self.source, self.account_id = Path(source), account_id
        self.clock = clock or (lambda: int(time.time() * 1000))

    def snapshot(self):
        for _ in range(3):
            ev, execution = read_state(self.source / "execution.sqlite", "execution")
            sv, state = read_state(self.source / "strategy.sqlite", "strategy")
            ev2, _ = read_state(self.source / "execution.sqlite", "execution")
            if ev == ev2:
                if (
                    execution.get("account_id") != self.account_id
                    or state.get("account_id") != self.account_id
                ):
                    raise ViewUnavailable("account_mismatch")
                return ev, sv, execution, state
        raise ViewUnavailable("snapshot_changing")

    def history(self, curve=False):
        try:
            with closing(
                sqlite3.connect(
                    (self.source / "strategy.sqlite").resolve().as_uri() + "?mode=ro",
                    uri=True,
                    timeout=0.15,
                )
            ) as db:
                db.execute("PRAGMA query_only=ON")
                row = db.execute("SELECT payload FROM live_observation_stats WHERE id=1").fetchone()
                stats = (
                    json.loads(row[0])
                    if row
                    else {"started_at": None, "latest_at": None, "samples": 0, "max_drawdown": None}
                )
                result = {"history": {k: v for k, v in stats.items() if k != "peak_nav"}}
                if curve:
                    stride = max(1, math.ceil(stats["samples"] / 1998))
                    rows = db.execute(
                        "SELECT payload FROM live_samples WHERE rowid % ? = 0 "
                        "OR timestamp IN (?,?) ORDER BY timestamp",
                        (stride, stats["started_at"], stats["latest_at"]),
                    ).fetchall()
                    result["equity_curve"] = [json.loads(r[0]) for r in rows]
                return result
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise ViewUnavailable("observations_unavailable") from exc

    def view(self):
        now = self.clock()
        ev, sv, execution, state = self.snapshot()
        engine = state["engine"]
        book = Ledger.restore(execution["book"])
        markets = state.get("markets", {})
        if book.usdgo:
            book.usdgo_mark = money(markets["USDGOUSDT"]["bid"])
        result = {
            k: deepcopy(state.get(k))
            for k in (
                "phase",
                "desired",
                "error",
                "startup_complete",
                "last_poll_at",
                "intent",
                "warmup_until",
                "source_sha256",
                "price_reference",
                "savings_observation",
                "indicator_initialization",
                "startup_resume",
                "terms",
            )
        }
        # Worker liveness remains the last explicit response from the trading API;
        # this reader reports persisted heartbeat freshness, never a fabricated thread state.
        result["summary"] = book.summary(
            markets.get("TRXUSDT", {}).get("bid") or engine["last_price"] or 1
        )
        result.update(self.history())
        result["summary"].update(
            initial_capital=float(book.initial),
            added_capital=float(book.external_flows),
            trading_turnover=float(
                sum((money(f["quantity"]) * money(f["price"]) for f in book.orders), money(0))
            ),
            max_drawdown=result["history"]["max_drawdown"],
        )
        result["schedule"] = {
            k: engine[k]
            for k in (
                "anchor",
                "anchor_version",
                "buy_period",
                "sell_period",
                "buy_at",
                "sell_at",
                "reason",
                "last_close",
            )
        }
        result["effective_config"] = engine["config"]
        times = [int(q["timestamp"]) for q in markets.values() if q.get("timestamp")]
        result["valuation"] = {
            "observed_at": min(times) if times else None,
            "stale": len(times) < 2 or any(not -5000 <= now - t <= 60000 for t in times),
        }
        result["events"] = state.get("events", [])[-30:]
        result["decisions"] = state.get("decisions", [])[-10:]
        result["evaluations"] = state.get("evaluations", [])[-12:]
        result["dashboard"] = dashboard(state, engine, execution, book, now)
        result["order_preview"] = preview(state, engine, execution, book, result["dashboard"], now)
        result["activity"] = activity_summary(execution, now)
        result["rebate_capital_automation"] = rebate_status(self.source, execution, now)
        result["execution"] = {
            "flow": execution.get("flow"),
            "operation": execution.get("operation"),
            "holdings": execution.get("holdings", {}),
            "events": execution.get("events", [])[-30:],
            "orders": execution["book"]["orders"][-20:],
            "order_count": len(book.orders),
            "lots": execution["book"]["lots"],
            "pending_income": execution.get("pending_income", {}),
            "income_accounting": income_summary(execution.get("income_accounting")),
            "asset_contributions": list(execution.get("asset_contributions", {}).values()),
            "quarantined_account_credits": list(
                execution.get("quarantined_account_credits", {}).values()
            )[-30:],
            "checked_at": execution.get("interest_checked_at"),
        }
        return {
            "account_id": self.account_id,
            "observed_at": now,
            "versions": {"execution": ev, "strategy": sv},
            "runtime": result,
        }


def send_notification(config, subject, body):
    """Telegram delivery uses only an explicitly configured recipient and bot token."""
    if not config.get("enabled"):
        return False
    if (
        config.get("channel") != "telegram"
        or not config.get("bot_token")
        or not config.get("chat_id")
    ):
        raise ValueError("telegram_configuration_incomplete")
    token = config["bot_token"]
    if any(
        character not in "0123456789:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-"
        for character in token
    ):
        raise ValueError("telegram_token_invalid")
    request = Request(
        "https://api.telegram.org/bot" + token + "/sendMessage",
        data=json.dumps(
            {
                "chat_id": config["chat_id"],
                "text": subject + "\n" + body,
                "disable_web_page_preview": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS Telegram host
        result = json.loads(response.read(65536))
    if result.get("ok") is not True:
        raise ValueError("telegram_delivery_unconfirmed")
    return True


class Operations:
    def __init__(
        self,
        view,
        database,
        backup_status,
        disk_path,
        notification_file=None,
        clock=None,
        resource_file=None,
    ):
        self.view, self.database = view, Path(database)
        self.backup_status, self.disk_path = Path(backup_status), disk_path
        self.notification_file = Path(notification_file) if notification_file else None
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.resource_file = Path(resource_file) if resource_file else None
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS ops_state(id INTEGER PRIMARY KEY,payload TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS notifications("
                "id INTEGER PRIMARY KEY,code TEXT,transition TEXT,created_at INTEGER,"
                "payload TEXT,sent_at INTEGER,attempted_at INTEGER,attempts INTEGER DEFAULT 0,"
                "delivery_state TEXT NOT NULL DEFAULT 'pending')"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS trade_notification_accounts("
                "account_id TEXT PRIMARY KEY,baseline_at INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS notified_fills("
                "account_id TEXT,fill_id TEXT,observed_at INTEGER,notification_id INTEGER,"
                "PRIMARY KEY(account_id,fill_id))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS strategy_lifecycle("
                "account_id TEXT PRIMARY KEY,payload TEXT NOT NULL)"
            )
        self.database.chmod(0o600)

    def notification(self):
        try:
            return json.loads(self.notification_file.read_text()) if self.notification_file else {}
        except (OSError, ValueError):
            return {}

    def observe_trades(self, execution, now):
        """Baseline existing fills once, then atomically enqueue each new native fill."""
        account = self.view.account_id
        if execution.get("account_id") != account:
            raise ValueError("account_mismatch")
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            initialized = db.execute(
                "SELECT 1 FROM trade_notification_accounts WHERE account_id=?", (account,)
            ).fetchone()
            for fill in execution["book"]["orders"]:
                key = str(fill["id"])
                if not key.startswith("spot:") or fill.get("side") not in ("buy", "sell"):
                    continue
                if db.execute(
                    "SELECT 1 FROM notified_fills WHERE account_id=? AND fill_id=?",
                    (account, key),
                ).fetchone():
                    continue
                notification_id = None
                if initialized:
                    code = (
                        "trade_fill:" + hashlib.sha256((account + ":" + key).encode()).hexdigest()
                    )
                    payload = {"message": fill_message(fill), "side": fill["side"], "fill": fill}
                    row = db.execute(
                        "INSERT INTO notifications"
                        "(code,transition,created_at,payload,delivery_state) "
                        "VALUES(?,?,?,?,?)",
                        (code, "trade", now, json.dumps(payload, ensure_ascii=False), "pending"),
                    )
                    notification_id = row.lastrowid
                db.execute(
                    "INSERT INTO notified_fills VALUES(?,?,?,?)",
                    (account, key, now, notification_id),
                )
            if not initialized:
                db.execute("INSERT INTO trade_notification_accounts VALUES(?,?)", (account, now))

    def observe_lifecycle(self, execution, state, now):
        """Persist sampled transitions and their outbox atomically in the observer DB."""
        account = self.view.account_id
        if execution.get("account_id") != account or state.get("account_id") != account:
            raise ValueError("account_mismatch")
        status, text = lifecycle_status(state, execution, now)
        marker = (state.get("indicator_initialization") or {}).get("initialized_at")
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM strategy_lifecycle WHERE account_id=?", (account,)
            ).fetchone()
            if row is None:
                # Rollout establishes a baseline, not a fictitious new start/pause.
                historical_run = bool(state.get("activated_at") and state.get("startup_complete"))
                prior = {
                    "status": status
                    or (
                        "running" if historical_run and state.get("desired") == "running" else None
                    ),
                    "ever_running": status == "running" or historical_run,
                    "run_marker": marker,
                    "sequence": 0,
                    "observed_at": now,
                }
            else:
                prior = json.loads(row[0])
                transition = None
                if status == "running":
                    new_run = (
                        marker is not None
                        and prior["run_marker"] is not None
                        and marker != prior["run_marker"]
                    )
                    if prior["status"] != "running" or new_run:
                        transition = "resumed" if prior["ever_running"] else "started"
                    prior.update(ever_running=True, run_marker=marker)
                elif status == "paused" and prior["ever_running"] and prior["status"] != status:
                    transition = "paused"
                if transition:
                    prior["sequence"] += 1
                    code = (
                        "strategy_lifecycle:"
                        + hashlib.sha256(f"{account}:{prior['sequence']}".encode()).hexdigest()
                    )
                    item = {"observed_at": now, "message": text, "sequence": prior["sequence"]}
                    db.execute(
                        "INSERT INTO notifications"
                        "(code,transition,created_at,payload,delivery_state) VALUES(?,?,?,?,?)",
                        (code, transition, now, json.dumps(item, ensure_ascii=False), "pending"),
                    )
                # Temporary network/health failures already have their own alarms;
                # do not turn them into invented user pauses or duplicate resumes.
                if status is not None:
                    prior["status"] = status
                prior["observed_at"] = now
            db.execute(
                "INSERT INTO strategy_lifecycle VALUES(?,?) ON CONFLICT(account_id) "
                "DO UPDATE SET payload=excluded.payload",
                (account, json.dumps(prior)),
            )

    def status(self):
        with closing(sqlite3.connect(self.database)) as db:
            row = db.execute("SELECT payload FROM ops_state WHERE id=1").fetchone()
            pending = db.execute(
                "SELECT count(*) FROM notifications "
                "WHERE delivery_state IN ('pending','unknown','sending')"
            ).fetchone()[0]
        with closing(sqlite3.connect(self.database)) as db:
            unconfirmed = db.execute(
                "SELECT count(*) FROM notifications WHERE delivery_state IN ('sending','unknown')"
            ).fetchone()[0]
            baseline = db.execute(
                "SELECT baseline_at FROM trade_notification_accounts WHERE account_id=?",
                (self.view.account_id,),
            ).fetchone()
            fills = db.execute(
                "SELECT count(*),count(notification_id) FROM notified_fills WHERE account_id=?",
                (self.view.account_id,),
            ).fetchone()
        value = json.loads(row[0]) if row else {"checked_at": None, "alerts": [], "backup": {}}
        notification = self.notification()
        ready = all(notification.get(k) for k in ("enabled", "bot_token", "chat_id"))
        value["notification"] = {
            "configured": bool(ready),
            "pending": pending,
            "last_error": "telegram_delivery_unconfirmed"
            if unconfirmed
            else value.get("notification_error"),
        }
        value.pop("notification_error", None)
        value["trade_notifications"] = {
            "baseline_at": baseline[0] if baseline else None,
            "observed_fills": fills[0],
            "queued_fills": fills[1],
        }
        return value

    def check(self):  # noqa: C901 - independent alarm conditions, no trading actions
        now = self.clock()
        findings = {}
        state = execution = None

        def alarm(code, message, severity="warning"):
            findings[code] = {"code": code, "message": message, "severity": severity}

        try:
            _, _, execution, state = self.view.snapshot()
            self.observe_trades(execution, now)
            self.observe_lifecycle(execution, state, now)
            running = state.get("desired") == "running"
            if state.get("error"):
                alarm("strategy_error", "策略报告运行异常，请查看真实账户页面。", "critical")
            if running:
                checks = (
                    ("worker_stale", state.get("last_poll_at"), 120000, "策略心跳超过2分钟未更新"),
                    (
                        "minute_stale",
                        state["engine"].get("last_close"),
                        180000,
                        "确认分钟行情超过3分钟未更新",
                    ),
                    (
                        "reconciliation_stale",
                        execution.get("interest_checked_at"),
                        180000,
                        "余额与理财到账核对超过3分钟未更新",
                    ),
                )
                for code, stamp, limit, message in checks:
                    if not stamp or not -5000 <= now - stamp <= limit:
                        alarm(code, message, "critical")
                if state.get("phase") == "observing" and not state.get("intent"):
                    for side in ("buy", "sell"):
                        due = state["engine"].get(side + "_at")
                        if due and now - due > 180000:
                            alarm(
                                "signal_overdue_" + side,
                                "信号确认时间超过3分钟仍未推进",
                                "critical",
                            )
            pending = execution.get("pending_income", {})
            if any(now - int(v["timestamp"]) > 900000 for v in pending.values()):
                alarm("income_pending", "存在超过15分钟尚未归属的到账利息")
        except (ViewUnavailable, KeyError, TypeError, ValueError):
            alarm(
                "source_unavailable", "独立监控无法读取策略快照，未修改策略运行状态。", "critical"
            )
        usage = shutil.disk_usage(self.disk_path)
        percent = 100 * usage.used / usage.total
        if percent >= 85:
            alarm(
                "disk_usage",
                "宿主机磁盘使用率达到85%，请检查日志与备份增长。",
                "critical" if percent >= 95 else "warning",
            )
        memory = memory_usage()
        if memory and memory["percent"] >= 80:
            alarm(
                "observer_memory",
                "独立告警容器内存占用达到80%，请核对资源预算；策略未被暂停。",
                "critical" if memory["percent"] >= 90 else "warning",
            )
        trading = trading_memory(self.resource_file, now)
        if trading and trading.get("available") is False:
            alarm("trading_memory_unavailable", "交易容器内存指标缺失、过期或读取失败。")
        elif trading and trading["percent"] >= 80:
            alarm(
                "trading_memory",
                f"交易容器内存占用 {trading['percent']:.1f}%，"
                f"当前 {trading['current'] / 1024**2:.1f} MiB，"
                f"上限 {trading['limit'] / 1024**2:.0f} MiB。",
                "critical" if trading["percent"] >= 90 else "warning",
            )
        try:
            backup = json.loads(self.backup_status.read_text())
        except (OSError, ValueError):
            backup = {}
        if not backup.get("last_success_at") or now - backup["last_success_at"] > 7200000:
            alarm("backup_stale", "最近2小时没有通过核验的自动备份")
        if backup.get("retention", {}).get("over_limit"):
            alarm("backup_capacity", "自动备份已超过容量预算，请核对保留策略。")
        if backup.get("last_error"):
            alarm("backup_failed", "最近一次自动备份失败；策略未被暂停。")
        old = self.status()
        previous = {a["code"]: a for a in old["alerts"]}
        alerts = []
        result = {
            "checked_at": now,
            "alerts": alerts,
            "backup": backup,
            "disk": {"used_percent": round(percent, 2), "free_bytes": usage.free},
            "notification_error": old.get("notification", {}).get("last_error"),
            "memory": memory,
            "trading_memory": trading,
        }
        with closing(sqlite3.connect(self.database)) as db, db:
            for code in previous.keys() | findings.keys():
                prior, current = previous.get(code), findings.get(code)
                if current:
                    current["first_seen_at"] = prior["first_seen_at"] if prior else now
                    current["observations"] = (prior.get("observations", 0) if prior else 0) + 1
                    current["active"] = current["observations"] >= 2
                    alerts.append(current)
                transition = (
                    "opened"
                    if current and current["active"] and not (prior and prior.get("active"))
                    else "recovered"
                    if prior and prior.get("active") and not current
                    else None
                )
                if transition:
                    if transition == "recovered":
                        db.execute(
                            "UPDATE notifications SET delivery_state='superseded' "
                            "WHERE code=? AND transition='opened' "
                            "AND delivery_state='pending'",
                            (code,),
                        )
                    item = dict(current or prior)
                    if transition == "recovered":
                        item.update(
                            active=False,
                            recovered_at=now,
                            message=recovery_status(code, state, execution, result),
                        )
                    db.execute(
                        "INSERT INTO notifications"
                        "(code,transition,created_at,payload,delivery_state) "
                        "VALUES(?,?,?,?,?)",
                        (
                            code,
                            transition,
                            now,
                            json.dumps(item, ensure_ascii=False),
                            "pending"
                            if transition == "opened"
                            or db.execute(
                                "SELECT count(*) FROM notifications WHERE code=? "
                                "AND transition='opened' AND attempted_at IS NOT NULL",
                                (code,),
                            ).fetchone()[0]
                            else "superseded",
                        ),
                    )
            db.execute(
                "INSERT INTO ops_state VALUES(1,?) ON CONFLICT(id) "
                "DO UPDATE SET payload=excluded.payload",
                (json.dumps(result),),
            )
        return self.status()

    def deliver(self):
        config = self.notification()
        if not all(config.get(k) for k in ("enabled", "bot_token", "chat_id")):
            return
        now = self.clock()
        with closing(sqlite3.connect(self.database)) as db, db:
            # A Telegram result lost after acceptance is ambiguous. Do not auto-resend;
            # record it for review instead of creating duplicate notifications.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id,code,transition,payload,created_at FROM notifications "
                "WHERE delivery_state='pending' AND attempted_at IS NULL "
                "ORDER BY CASE WHEN transition='trade' THEN 2 "
                "WHEN transition IN ('started','paused','resumed') THEN 1 ELSE 0 END,id LIMIT 1"
            ).fetchone()
            if not row:
                return
            db.execute(
                "UPDATE notifications SET attempted_at=?,attempts=attempts+1,"
                "delivery_state='sending' WHERE id=?",
                (now, row[0]),
            )
        error = None
        try:
            item = json.loads(row[3])
            current = None
            if row[2] in LIFECYCLE_TITLES:
                try:
                    _, _, execution, state = self.view.snapshot()
                    _, current = lifecycle_status(state, execution, now)
                except (ViewUnavailable, KeyError, TypeError, ValueError):
                    current = "暂时无法读取，请查看实盘运行页面"
            title, message = notification_content(row[1], row[2], item, row[4], current)
            send_notification(config, title, message)
        except Exception:
            # Never expose API responses, bot tokens or recipient IDs.
            error = "telegram_delivery_unconfirmed"
        with closing(sqlite3.connect(self.database)) as db, db:
            if error is None:
                db.execute(
                    "UPDATE notifications SET sent_at=?,delivery_state='sent' WHERE id=?",
                    (now, row[0]),
                )
            if error is not None:
                db.execute(
                    "UPDATE notifications SET delivery_state='unknown' WHERE id=?", (row[0],)
                )
            payload = json.loads(
                db.execute("SELECT payload FROM ops_state WHERE id=1").fetchone()[0]
            )
            payload["notification_error"] = error
            db.execute("UPDATE ops_state SET payload=? WHERE id=1", (json.dumps(payload),))
