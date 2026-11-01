"""Explicit local opening adoption. No broker, order or worker is constructed.

Both native journals and the immutable opening receipt are published together.
The control file is the final commit marker; a crash before it can only finish
that same adoption, never allocate a second time or revalue its opening cost.
"""

import json
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from copy import deepcopy
from decimal import localcontext
from pathlib import Path

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, Lot, StateStore, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import allocation_lock, digest
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategyStore


MODE = "existing_positions_market_value_v1"


def read_state(path, key):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro")) as db:
        row = db.execute("SELECT state FROM trx_state WHERE id=?", (key,)).fetchone()
    if not row:
        raise ExecutionError("adoption_journal_unavailable")
    return json.loads(row[0])


def status(root):
    root = Path(root)
    control = root / "live-strategy-control.json"
    if not control.exists():
        return {"committed": False, "receipt": None}
    saved = json.loads(control.read_text())
    if saved.get("allocation_mode") != MODE:
        return {"committed": False, "receipt": None}
    receipt = validate_journals(root, saved)
    return {"committed": True, "receipt": receipt}


def validate_journals(root, control):
    if not re.fullmatch(r"[a-f0-9]{64}", control["account_id"]):
        raise ExecutionError("account_binding_mismatch")
    directory = root / "live_execution" / control["account_id"]
    execution = read_state(directory / "execution.sqlite", "execution")
    strategy = read_state(directory / "strategy.sqlite", "strategy")
    receipt = execution["opening_adoption"]
    basis = execution.get("performance_basis")
    initial = basis["initial_equity_usdt"] if basis else receipt["manifest"]["initial_equity_usdt"]
    if (
        receipt["receipt_sha256"]
        != digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        or receipt["manifest_id"] != control["adoption_manifest_id"]
        or execution["account_id"] != control["account_id"]
        or strategy["account_id"] != control["account_id"]
        or strategy["source_sha256"] != control["source_sha256"]
        or money(execution["book"]["initial"]) != money(initial)
        or (basis and basis["sha256"] != digest({k: v for k, v in basis.items() if k != "sha256"}))
    ):
        raise ExecutionError("adoption_journal_mismatch")
    return deepcopy(receipt)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=".opening-control-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
        sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def opening_book(manifest, now):
    book = Ledger(manifest["initial_equity_usdt"])
    book.cash = money(0)
    assets = {r["coin"]: r for r in manifest["assets"]}
    usdgo = assets["USDGO"]
    book.usdgo, book.usdgo_cost = money(usdgo["quantity"]), money(usdgo["strategy_cost_usdt"])
    book.usdgo_mark = money(usdgo["adoption_unit_cost_usdt"])
    positions = {}
    for index, row in enumerate(manifest["locations"]):
        quantity, coin = money(row["quantity"]), row["coin"]
        if not quantity:
            continue
        held = row["location"] == "flexible_savings"
        lot_id = "opening-principal-" + str(index)
        if coin == "TRX":
            price = money(assets[coin]["adoption_unit_cost_usdt"])
            book.lots.append(
                Lot(lot_id, now, quantity, price, quantity if held else money(0), quantity * price)
            )
        elif held:
            book.locked["USDGO"] += quantity
        if held:
            positions[row["order_id"]] = {
                "coin": coin,
                "product_id": row["product_id"],
                "principal": str(quantity),
                "interest": "0",
                "unassigned_amount": "0",
                "owned_since": now,
                "subscription_ids": [],
                "lots": {lot_id: str(quantity)} if coin == "TRX" else {},
            }
    book.events.add("opening-existing-positions")
    if book.equity(assets["TRX"]["adoption_unit_cost_usdt"]) != book.initial:
        raise ExecutionError("opening_equity_mismatch")
    return book, positions


def strategy_state(account_id, source, book, quotes, now):
    config = ResearchConfig.from_saved(source).model_copy(
        update={
            "initial_cash": float(book.initial),
            "fee_rebate_ratio": 0,
            "slippage": 0,
            "trx_apr": 0,
            "usdgo_apr": 0,
            "usdt_apr": 0,
        }
    )
    engine = Engine(config, Ledger.restore(book.state()))
    quotes = {coin + "USDT": deepcopy(quote) for coin, quote in quotes.items()}
    engine.last_price = float(money(quotes["TRXUSDT"]["bid"]))
    engine.paused, engine.accept_new_buys = True, False
    engine.reason = "adopted_trading_not_authorized"
    engine.core_lot_ids = [lot.lot_id for lot in book.lots]
    return {
        "schema": "trx-c1-live-strategy-v1",
        "account_id": account_id,
        "source_snapshot": source,
        "source_sha256": digest(source),
        "created_at": now,
        "desired": "paused",
        "phase": "paused",
        "error": None,
        "engine": engine.state(),
        "activated_at": None,
        "warmup_until": None,
        "indicator_initialization": None,
        "startup_complete": True,
        "startup_mode": MODE,
        "trading_authorized": False,
        "intent": None,
        "last_flow": None,
        "last_poll_at": None,
        "price_reference": None,
        "last_cash_switch": 0,
        "markets": quotes,
        "terms": {},
        "decisions": [],
        "events": [{"kind": "existing_positions_adopted_paused", "timestamp": now}],
    }


def documents(preview, account_id, identity, source, now):
    manifest = deepcopy(preview["manifest"])
    book, holdings = opening_book(manifest, now)
    products = []
    for coin in ("TRX", "USDGO"):
        ids = {
            r["product_id"]
            for r in manifest["locations"]
            if r["coin"] == coin and r["location"] == "flexible_savings"
        }
        if len(ids) != 1:
            raise ExecutionError("adoption_product_mapping_required")
        products.append(
            {
                "coin": coin,
                "product_id": ids.pop(),
                "redemption_minimum": None,
                "redemption_decimals": None,
                "redemption_evidence": "",
            }
        )
    receipt = {
        "schema": "trx-opening-adoption-receipt-v1",
        "committed": True,
        "preview_only": False,
        "adopted_at": now,
        "manifest_id": preview["manifest_id"],
        "manifest": manifest,
        "trading_authorized": False,
        "trading_enabled": False,
        "earn_write_enabled": False,
        "startup_actions": [],
        "income_policy": preview["income_policy"],
        "quotes": deepcopy(preview["opening_quotes"]),
        "cross_period_income": "pending_attribution",
        "source_sha256": digest(source),
    }
    receipt["receipt_sha256"] = digest(receipt)
    excluded = manifest["excluded_assets"][0]
    execution = {
        "schema": "trx-classic-live-execution-v1",
        "account_id": account_id,
        "created_at": now,
        "book": book.state(),
        "products": products,
        "unassigned_offsets": {"TRX": "0", "USDGO": "0", "USDT": excluded["spot_quantity"]},
        "income_baseline_bills": {},
        "account_credit_baseline_bills": {},
        "pending_income": {},
        "holdings": holdings,
        "receipts": {},
        "flows": [],
        "flow": None,
        "operation": None,
        "events": [{"kind": "opening_adoption", "timestamp": now}],
        "anchor": None,
        "anchor_version": 0,
        "error": None,
        "interest_checked_at": now,
        "opening_adoption": receipt,
        "trading_authorized": False,
    }
    control = {
        "account_id": account_id,
        "source_sha256": digest(source),
        "setup": {"capital": manifest["initial_equity_usdt"], "products": products},
        "review_hash": "",
        "reviewed_at": now,
        "review_expires_at": 0,
        "credentials_identity": identity,
        "allocation_mode": MODE,
        "adoption_manifest_id": preview["manifest_id"],
        "trading_authorized": False,
    }
    return (
        receipt,
        execution,
        strategy_state(account_id, source, book, receipt["quotes"], now),
        control,
    )


def adopt(account, service, *, capture=None):
    """Operator-only entry point; deliberately not exposed by any POST route."""
    if capture is None:
        from freqtrade.strategy.trx_phase1.opening_adoption_capture import capture

    root = service.root
    with allocation_lock(root):
        if service.control.exists():
            existing = status(root)
            if not existing["committed"]:
                raise ExecutionError("execution_account_already_allocated")
            if service.load()["credentials_identity"] != service.credential_identity():
                raise ExecutionError("account_binding_mismatch")
            return existing["receipt"]
        parent = root / "live_execution"
        previous = list(parent.iterdir()) if parent.exists() else []
        if previous:
            if len(previous) != 1 or not (previous[0] / "opening-control.json").is_file():
                raise ExecutionError("unresolved_existing_allocation")
            control = json.loads((previous[0] / "opening-control.json").read_text())
            if control["credentials_identity"] != service.credential_identity():
                raise ExecutionError("account_binding_mismatch")
            receipt = validate_journals(root, control)
            atomic_json(service.control, control)
            return receipt
        preview, account_id, identity, now = capture(account, service)
        if not preview["reviewable"] or preview["committed"]:
            raise ExecutionError("fresh_complete_adoption_snapshot_required")
        with localcontext() as context:
            context.prec = 50
            receipt, execution, strategy, control = documents(
                preview, account_id, identity, service.source(), now
            )
        temporary = Path(tempfile.mkdtemp(prefix=".opening-adoption-", dir=root))
        try:
            path = temporary / account_id
            path.mkdir(mode=0o700)
            StateStore(path / "execution.sqlite").save("execution", 0, execution)
            LiveStrategyStore(path / "strategy.sqlite").save("strategy", 0, strategy)
            atomic_json(path / "opening-control.json", control)
            for file in path.iterdir():
                file.chmod(0o600)
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
            sync_directory(path)
            if identity != service.credential_identity() or any(
                not -5000 <= service.clock() - q["timestamp"] <= 10000
                for q in receipt["quotes"].values()
            ):
                raise ExecutionError("adoption_quote_or_identity_changed_before_commit")
            parent.mkdir(mode=0o700, exist_ok=True)
            path.rename(parent / account_id)
            sync_directory(parent)
            atomic_json(service.control, control)
        finally:
            shutil.rmtree(temporary)
        return receipt
