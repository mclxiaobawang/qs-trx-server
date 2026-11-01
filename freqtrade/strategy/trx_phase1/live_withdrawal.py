"""Reviewed USDGO withdrawal to the same account's external spot balance.

The existing strategy thread owns all redemption writes. A confirmation first
commits an outbox in the execution journal, before any flow can be queued. Only
verified completed redemptions allow an atomic capital debit. No transfer or
on-chain withdrawal API is called by this module.
"""

from copy import deepcopy

from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_contribution import candidate, quotes
from freqtrade.strategy.trx_phase1.live_execution import digest
from freqtrade.strategy.trx_phase1.live_funding import LiveFunding
from freqtrade.strategy.trx_phase1.withdrawal_preview import WithdrawalPreviewError, preview


DESTINATION = "same_account_spot"
ACTIVE = {"confirmed"}


def active(state):
    rows = [r for r in state.get("asset_withdrawals", {}).values() if r["status"] in ACTIVE]
    if len(rows) > 1:
        raise ExecutionError("withdrawal_multiple_active_requests")
    return rows[0] if rows else None


def idle(runtime):
    if (
        not runtime.state["startup_complete"]
        or runtime.state["phase"] not in {"observing", "paused", "prepared"}
        or runtime.state["error"]
        or runtime.state["intent"]
        or runtime.engine.pending
    ):
        raise ExecutionError("withdrawal_requires_idle_strategy")


def project(runtime, amount, now):
    execution = runtime.execution
    candidate(execution, now, allow_cross_start_pending=True)
    markets = quotes(execution, now)
    now = max(now, execution.broker.clock()) if hasattr(execution.broker, "clock") else now
    rule = next(p for p in execution.state["products"] if p["coin"] == "USDGO")
    if execution.broker.product("USDGO", rule["product_id"])["status"] != "in_progress":
        raise ExecutionError("withdrawal_product_unavailable")
    state = deepcopy(runtime.state)
    state["engine"] = runtime.engine.state()
    state["markets"] = {coin + "USDT": row for coin, row in markets.items()}
    try:
        result = preview(execution.state, state, amount, now)
    except WithdrawalPreviewError as error:
        raise ExecutionError(str(error)) from None
    return result, markets, now


def review(runtime, amount, now):
    idle(runtime)
    execution = runtime.execution
    with execution.exclusive():
        execution.reload()
        execution.require_trading_authorized()
        if active(execution.state):
            raise ExecutionError("withdrawal_already_pending")
        result, _, now = project(runtime, amount, now)
        result.pop("preview_id")
        result.update(
            preview_only=False,
            execution_enabled=True,
            destination=DESTINATION,
            expires_at=now + 5 * 60000,
            book_sha256=digest(execution.state["book"]),
        )
        result["review_hash"] = digest(result)
        execution.state["asset_withdrawal_review"] = result
        execution.save()
        return deepcopy(result)


def confirm(runtime, review_hash, now):
    execution = runtime.execution
    with execution.exclusive():
        execution.reload()
        records = execution.state.get("asset_withdrawals", {})
        if review_hash in records:
            return deepcopy(records[review_hash])
        idle(runtime)
        execution.require_trading_authorized()
        if runtime.state["desired"] != "running":
            raise ExecutionError("withdrawal_requires_running_worker")
        if active(execution.state):
            raise ExecutionError("withdrawal_already_pending")
        saved = execution.state.get("asset_withdrawal_review")
        if (
            not saved
            or saved.get("review_hash") != review_hash
            or digest({k: v for k, v in saved.items() if k != "review_hash"}) != review_hash
            or saved["account_id"] != execution.account_id
            or saved["destination"] != DESTINATION
            or not saved["created_at"] <= now < saved["expires_at"]
        ):
            raise ExecutionError("withdrawal_review_expired")
        current, _, now = project(runtime, saved["withdrawal"]["quantity"], now)
        if now >= saved["expires_at"]:
            raise ExecutionError("withdrawal_review_expired")
        if saved["performance_basis_sha256"] != current["performance_basis_sha256"]:
            raise ExecutionError("withdrawal_start_basis_required")
        if (
            saved["book_sha256"] != digest(execution.state["book"])
            or saved["sources"] != current["sources"]
        ):
            raise ExecutionError("withdrawal_assets_changed")
        if abs(
            money(current["withdrawal"]["mark_usdt"]) / money(saved["withdrawal"]["mark_usdt"]) - 1
        ) > money(".01"):
            raise ExecutionError("withdrawal_price_changed")
        record = {
            "review_hash": review_hash,
            "account_id": execution.account_id,
            "status": "confirmed",
            "destination": DESTINATION,
            "amount": current["withdrawal"]["quantity"],
            "confirmed_at": now,
            "confirmed_mark_usdt": current["withdrawal"]["mark_usdt"],
            "steps": current["sources"]["redemptions"],
            "performance_basis_sha256": current["performance_basis_sha256"],
        }
        execution.state.setdefault("asset_withdrawals", {})[review_hash] = record
        execution.state["asset_withdrawal_review"] = None
        execution.event("capital_withdrawal_confirmed", now, withdrawal=record)
        execution.save()
        return deepcopy(record)


def advance(runtime, now):
    """Recover the committed outbox before the next automatic trading decision."""
    runtime.execution.reload()
    record = active(runtime.execution.state)
    if not record:
        return
    intent = runtime.state["intent"]
    if intent:
        if intent.get("withdrawal_id") != record["review_hash"]:
            raise ExecutionError("withdrawal_intent_binding_mismatch")
        return
    if runtime.state["desired"] != "running":
        finish(runtime, record["review_hash"], {"status": "expired"}, now)
        return
    if record["steps"]:
        runtime.make_intent(
            record["steps"],
            "withdrawal",
            now,
            now + 10 * 60000,
            withdrawal_id=record["review_hash"],
        )
    else:
        finish(runtime, record["review_hash"], {"status": "completed"}, now)
        sync(runtime)


def sync(runtime):
    runtime.execution.reload()
    if runtime.state["markets"]:
        runtime.sync_book(
            LiveFunding(
                runtime.execution.state,
                runtime.engine.config,
                runtime.state["markets"],
                runtime.state["terms"],
            )
        )
    runtime.persist()


def finish(runtime, review_hash, flow, now):
    execution = runtime.execution
    with execution.exclusive():
        execution.reload()
        record = execution.state["asset_withdrawals"][review_hash]
        if record["status"] != "confirmed":
            return
        if flow["status"] != "completed":
            record.update(
                status="not_withdrawn", reason="redemption_not_completed", completed_at=now
            )
            execution.event("capital_withdrawal_not_completed", now, withdrawal=record)
            execution.save()
            return
        book, balances = candidate(execution, now, allow_cross_start_pending=True)
        markets = quotes(execution, now)
        now = max(now, execution.broker.clock()) if hasattr(execution.broker, "clock") else now
        price, fx = money(markets["TRX"]["bid"]), money(markets["USDGO"]["bid"])
        if abs(fx / money(record["confirmed_mark_usdt"]) - 1) > money(".01"):
            record.update(
                status="not_withdrawn", reason="price_changed_requires_new_review", completed_at=now
            )
            execution.event("capital_withdrawal_not_completed", now, withdrawal=record)
            execution.save()
            return
        if execution.state["performance_basis"]["sha256"] != record["performance_basis_sha256"]:
            raise ExecutionError("withdrawal_start_basis_required")
        amount = money(record["amount"])
        old = book.state()
        try:
            book.withdraw_usdgo("capital-withdrawal:" + review_hash, amount, price, fx)
        except ValueError:
            raise ExecutionError("withdrawal_spot_or_capital_unavailable") from None
        offsets = execution.state["unassigned_offsets"]
        offsets["USDGO"] = str(money(offsets["USDGO"]) + amount)
        execution.check_wallet(balances, book)
        record.update(
            status="completed",
            completed_at=now,
            value_usdt=str(amount * fx),
            mark_usdt=str(fx),
            fund_units_before=old["fund_units"],
            fund_units_after=str(book.fund_units),
            retained_valuation_profit_usdt=str(
                book.withdrawn_usdgo_profit - money(old.get("withdrawn_usdgo_profit", "0"))
            ),
            flow_id=flow.get("id"),
        )
        execution.state["book"] = book.state()
        execution.event("capital_withdrawal_recorded", now, withdrawal=record)
        execution.save()
