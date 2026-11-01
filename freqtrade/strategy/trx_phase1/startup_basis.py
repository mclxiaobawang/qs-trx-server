"""Actual-start valuation of existing assets; previews never publish financial state.

The previous adoption receipt remains an audit record. Only an explicitly
authorized start can commit a fresh basis. Both financial journals change in one
SQLite transaction, and a retry recovers that basis instead of revaluing it.
"""

import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from decimal import localcontext
from pathlib import Path

from freqtrade.strategy.trx_phase1.adoption_preview import adoption_preview
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest
from freqtrade.strategy.trx_phase1.opening_adoption import (
    MODE,
    atomic_json,
    read_state,
)
from freqtrade.strategy.trx_phase1.opening_adoption import (
    documents as adoption_documents,
)


POLICY = "actual-start-existing-positions-v1"
HOUR = 3600000


def pending(control, strategy):
    return bool(
        control and control.get("allocation_mode") == MODE and not strategy.get("activated_at")
    )


def idle(execution, strategy):
    if (
        strategy.get("activated_at")
        or execution.get("performance_basis")
        or execution["book"]["orders"]
        or execution.get("flow")
        or execution.get("flows")
        or execution.get("operation")
        or strategy.get("intent")
        or strategy["engine"].get("pending")
        or strategy.get("desired") != "paused"
    ):
        raise ExecutionError("startup_basis_requires_unused_paused_account")


def context(root):
    root = Path(root)
    path = root / "live-strategy-control.json"
    if not path.exists():
        return None
    control = json.loads(path.read_text())
    if control.get("allocation_mode") != MODE:
        return None
    directory = root / "live_execution" / control["account_id"]
    execution = read_state(directory / "execution.sqlite", "execution")
    strategy = read_state(directory / "strategy.sqlite", "strategy")
    return control, execution, strategy


def project(observed, control, execution, strategy, now):
    result = adoption_preview(observed, now_ms=now)
    result.update(
        schema="trx-startup-basis-preview-v1",
        policy=POLICY,
        performance_started=False,
        started_at=None,
        fresh_snapshot_required_at_start=True,
        historical_income_queried=False,
        previous_adoption_at=execution["opening_adoption"]["adopted_at"],
        previous_adoption_is_performance_basis=False,
        income_policy={
            "performance_starts_on_actual_strategy_start": True,
            "pre_start_assets_are_initial_capital": True,
            "historical_income_is_strategy_profit": False,
            "delayed_pre_start_interest_is_strategy_profit": False,
            "cross_start_period_is_quarantined": True,
            "daily_interest_summary_is_another_credit": False,
            "usdt_is_outside_strategy": True,
            "first_purchase_required": False,
            "fees_counted_once": True,
        },
    )
    try:
        idle(execution, strategy)
    except ExecutionError as exc:
        result["blockers"].append(str(exc))
    if result["manifest"]:
        m = result["manifest"]
        m.update(
            schema="trx-startup-basis-manifest-v1", cost_basis="actual_strategy_start_market_value"
        )
        for asset in m["assets"]:
            asset["startup_unit_cost_usdt"] = asset.pop("adoption_unit_cost_usdt")
        result["manifest_id"] = digest(m)
        result["scope_hash"] = digest(
            {
                "account_id": control["account_id"],
                "locations": [
                    {k: row[k] for k in ("coin", "location", "product_id", "order_id")}
                    for row in m["locations"]
                ],
                "excluded_coins": [row["coin"] for row in m["excluded_assets"]],
                "policy": POLICY,
            }
        )
    result["reviewable"] = bool(result["manifest"] and not result["blockers"])
    result["warnings"] = [
        "preview_is_not_start_price",
        "pre_start_income_is_opening_capital",
        "cross_start_income_pending",
    ]
    return result


def preview(root, observed, now):
    saved = context(root)
    if saved is None:
        return None
    control, execution, strategy = saved
    if strategy.get("activated_at"):
        return None
    return project(observed, control, execution, strategy, now)


def capture(account, service):
    from freqtrade.strategy.trx_phase1.opening_adoption_capture import capture as capture_positions

    original, account_id, identity, now = capture_positions(
        account, service, current_positions_only=True
    )
    saved = context(service.root)
    if saved is None or saved[0]["account_id"] != account_id:
        raise ExecutionError("account_binding_mismatch")
    # Reuse the account client's final independently checked quantities/quotes.
    observed = account.status()
    observed = deepcopy(observed)
    observed["snapshot"]["prices"] = original["opening_quotes"]
    observed["snapshot"]["completed_at"] = now
    result = project(observed, *saved, now)
    if not result["reviewable"]:
        raise ExecutionError("fresh_startup_snapshot_required")
    result["opening_quotes"] = original["opening_quotes"]
    result["captured_at"] = now
    result["credentials_identity"] = identity
    return result


def documents(captured, control, execution, strategy, source):
    """Build the exact future journals without allocating, buying, or persisting."""
    idle(execution, strategy)
    if not captured["reviewable"] or captured["policy"] != POLICY:
        raise ExecutionError("fresh_startup_snapshot_required")
    if captured["manifest_id"] != digest(captured["manifest"]) or any(
        money(asset["startup_unit_cost_usdt"])
        != money(captured["opening_quotes"][asset["coin"]]["bid"])
        for asset in captured["manifest"]["assets"]
    ):
        raise ExecutionError("startup_snapshot_mismatch")
    now = captured["captured_at"]
    raw = deepcopy(captured)
    for asset in raw["manifest"]["assets"]:
        asset["adoption_unit_cost_usdt"] = asset["startup_unit_cost_usdt"]
    with localcontext() as ctx:
        ctx.prec = 50
        _, new_execution, new_strategy, _ = adoption_documents(
            raw, control["account_id"], captured["credentials_identity"], source, now
        )
    if {(p["coin"], p["product_id"]) for p in new_execution["products"]} != {
        (p["coin"], p["product_id"]) for p in control["setup"]["products"]
    }:
        raise ExecutionError("startup_product_identity_changed")
    basis = {
        "schema": "trx-performance-start-basis-v1",
        "policy": POLICY,
        "started_at": now,
        "manifest": deepcopy(captured["manifest"]),
        "manifest_id": captured["manifest_id"],
        "quotes": deepcopy(captured["opening_quotes"]),
        "scope_hash": captured["scope_hash"],
        "initial_equity_usdt": new_execution["book"]["initial"],
        "income_policy": deepcopy(captured["income_policy"]),
        "startup_actions": [],
    }
    basis["sha256"] = digest(basis)
    price = captured["opening_quotes"]["TRX"]["bid"]
    from freqtrade.strategy.trx_phase1.income_allocation import VERSION, ownership

    owned = ownership(new_execution["holdings"])
    external = deepcopy(owned)
    for row in external.values():
        row["external"], row["strategy"] = row["strategy"], "0"
    policy = {
        "version": VERSION,
        "account_id": control["account_id"],
        "approved_at": now,
        "products": deepcopy(control["setup"]["products"]),
        "checkpoints": [
            {"timestamp": now // HOUR * HOUR - HOUR, "positions": external},
            {"timestamp": now, "positions": owned},
        ],
    }
    basis["income_policy_sha256"] = digest(policy)
    basis["sha256"] = digest({k: v for k, v in basis.items() if k != "sha256"})
    authorized = control.get("trading_authorized") is True
    new_execution.update(
        opening_adoption=deepcopy(execution["opening_adoption"]),
        performance_basis=basis,
        startup_income_policy=policy,
        products=deepcopy(control["setup"]["products"]),
        trading_authorized=authorized,
        anchor=price,
        anchor_version=1,
        events=[
            {"kind": "strategy_start_basis", "timestamp": now, "basis_sha256": basis["sha256"]}
        ],
    )
    new_strategy.update(
        performance_basis=deepcopy(basis),
        activated_at=now,
        trading_authorized=authorized,
        events=[{"kind": "existing_positions_started", "timestamp": now}],
    )
    new_strategy["engine"].update(
        anchor=float(money(price)),
        anchor_version=1,
        initial_anchor_at=now,
        accept_new_buys=authorized,
    )
    return new_execution, new_strategy


def commit(service, captured, *, review_hash=None):
    """Called only by authorized start, never by GET, refresh or preflight."""
    control, execution, strategy = context(service.root)
    if control.get("trading_authorized") is not True:
        raise ExecutionError("adopted_trading_not_authorized")
    if control["credentials_identity"] != service.credential_identity():
        raise ExecutionError("credentials_changed_requires_preflight")
    if control["source_sha256"] != digest(service.source()):
        raise ExecutionError("frozen_live_setup_mismatch")
    directory = service.root / "live_execution" / control["account_id"]
    if not execution.get("performance_basis"):
        if (
            not control.get("review_hash")
            or (review_hash is not None and review_hash != control["review_hash"])
            or service.clock() >= control.get("review_expires_at", 0)
            or control.get("startup_scope_hash") != captured["scope_hash"]
        ):
            raise ExecutionError("fresh_preflight_required")
        if captured["credentials_identity"] != service.credential_identity() or any(
            not -5000 <= service.clock() - q["timestamp"] <= 10000
            for q in captured["opening_quotes"].values()
        ):
            raise ExecutionError("startup_snapshot_expired")
        new_execution, new_strategy = documents(
            captured, control, execution, strategy, service.source()
        )
        with closing(sqlite3.connect(directory / "execution.sqlite")) as db:
            db.execute("ATTACH DATABASE ? AS strategy_db", (str(directory / "strategy.sqlite"),))
            # Multi-database atomicity requires rollback journals on this filesystem.
            if any(
                db.execute(f"PRAGMA {name}.journal_mode").fetchone()[0] != "delete"
                for name in ("main", "strategy_db")
            ):
                raise ExecutionError("startup_atomic_journal_required")
            try:
                db.execute("BEGIN IMMEDIATE")
                for name, previous, updated, table in (
                    ("execution", execution, new_execution, "main"),
                    ("strategy", strategy, new_strategy, "strategy_db"),
                ):
                    version, raw = db.execute(
                        f"SELECT version,state FROM {table}.trx_state WHERE id=?", (name,)
                    ).fetchone()
                    if json.loads(raw) != previous:
                        raise ExecutionError("startup_state_changed")
                    db.execute(
                        f"UPDATE {table}.trx_state SET version=?,state=? WHERE id=?",
                        (version + 1, json.dumps(updated), name),
                    )
                db.commit()
            except BaseException:
                db.rollback()
                raise
        execution = new_execution
    basis = execution["performance_basis"]
    control["setup"]["capital"] = basis["initial_equity_usdt"]
    control.update(
        performance_started_at=basis["started_at"], performance_basis_sha256=basis["sha256"]
    )
    atomic_json(service.control, control)
    service.runtime = None
    return basis
