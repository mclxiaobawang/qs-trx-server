"""Bounded account-evidence waits; never approves a balance gap or changes funds."""

# ruff: noqa: RUF001 - Chinese wait-status text
import logging
from time import monotonic


EXTERNAL_MOVEMENT_EVIDENCE_PENDING = "external_movement_evidence_pending"
WAIT_CODES = frozenset(
    {
        "account_credit_wallet_evidence_required",
        "wallet_reconciliation_required",
        "owned_balance_not_spendable",
        "interest_holding_evidence_incomplete",
        "interest_arrival_bill_unverified",
        EXTERNAL_MOVEMENT_EVIDENCE_PENDING,
    }
)
WAIT_LIMIT_MS = 300000
REFRESH_CODES = frozenset(
    {"wallet_reconciliation_required", "account_credit_wallet_evidence_required"}
)


def pending_wait(state, now, execution=None):
    since = state.get("evidence_wait_started_at")
    if (
        state.get("desired") != "running"
        or state.get("phase") not in {"recovering", "executing"}
        or (execution or {}).get("error")
        or state.get("error") not in WAIT_CODES
        or type(since) is not int
        or not 0 <= now - since < WAIT_LIMIT_MS
    ):
        return None
    return {
        "code": state["error"],
        "started_at": since,
        "deadline": since + WAIT_LIMIT_MS,
        "age_ms": now - since,
        "message": "等待余额与到账流水同步，核对通过前不提交新增操作",
    }


def clear_verified_wait(state, execution, now):
    if (
        state.get("desired") != "running"
        or state.get("phase") not in {"recovering", "executing", "observing"}
        or state.get("error") not in WAIT_CODES
        or type(state.get("evidence_wait_started_at")) is not int
        or state["evidence_wait_started_at"] > now
        or execution.get("error")
    ):
        return None
    last_error = max(
        [state.get("evidence_wait_last_seen_at") or state["evidence_wait_started_at"]]
        + [
            e["timestamp"]
            for e in state.get("events", [])
            if e.get("kind") == "strategy_poll_blocked" and e.get("code") in WAIT_CODES
        ]
    )
    verified = execution.get("interest_checked_at")
    if type(verified) is not int or not last_error < verified <= now + 120000:
        return None
    code = state["error"]
    state.update(error=None, evidence_wait_started_at=None, evidence_wait_last_seen_at=None)
    return {"code": code, "wallet_verified_at": verified}


def wallet_error(code, coin, actual, expected, *, basis="total"):
    # Imported only by the trading caller; the independent monitor stays lightweight.
    from .live_broker import ExecutionError

    detail = {
        "coin": coin,
        "actual": str(actual),
        "expected": str(expected),
        "difference": str(actual - expected),
        "basis": basis,
    }
    logging.getLogger(__name__).warning(
        "Wallet reconciliation: code=%s coin=%s basis=%s actual=%s expected=%s difference=%s",
        code,
        coin,
        basis,
        actual,
        expected,
        actual - expected,
    )
    error = ExecutionError(code)
    error.wallet_mismatch = detail
    return error


def verified_income_candidate(execution, book, positions, receipts, now):  # noqa: C901 - bounded fallbacks
    """Re-read at most once; every candidate must pass the exact original wallet check."""
    from .live_broker import ExecutionError

    started = monotonic()
    for attempt in range(2):
        read_at = now + max(0, int((monotonic() - started) * 1000))
        try:
            candidate = execution.income_candidate(book, positions, receipts, read_at)
            balances = execution.broker.balances()
            execution.check_wallet(balances, candidate[0], candidate[3])
            from .verified_external_movements import observe_due

            try:
                external = observe_due(execution, read_at)
            except ExecutionError:
                # USDC is wholly outside this strategy. A metadata-only read
                # failure cannot block a separately verified native wallet.
                external = None
            if external:
                from .ledger import Ledger

                execution.state, balances = external
                state = execution.state
                return (
                    Ledger.restore(state["book"]),
                    state["holdings"],
                    state["receipts"],
                    [],
                    state["income_accounting"],
                    balances,
                    now,
                )
            # Keep the durable poll watermark at the caller's logical time.
            # A retry can read newer bills, but a following call in the same poll
            # must not acquire a history start later than its own cutoff.
            return (*candidate, balances, now)
        except ExecutionError as exc:
            if (attempt or str(exc) not in REFRESH_CODES) and str(exc) in (
                REFRESH_CODES
                | {
                    "interest_holding_evidence_incomplete",
                    "unrecognized_usdt_movement",
                    "unrecognized_account_movement",
                    "account_credit_evidence_required",
                }
            ):
                from .ledger import Ledger
                from .verified_external_movements import observe as observe_external

                try:
                    external = observe_external(execution, read_at)
                except ExecutionError as evidence_error:
                    if str(evidence_error) == EXTERNAL_MOVEMENT_EVIDENCE_PENDING:
                        raise evidence_error from None
                    external = None
                if external:
                    execution.state, balances = external
                    state = execution.state
                    return (
                        Ledger.restore(state["book"]),
                        state["holdings"],
                        state["receipts"],
                        [],
                        state["income_accounting"],
                        balances,
                        now,
                    )
            mismatch = getattr(exc, "wallet_mismatch", {})
            if (
                (attempt or str(exc) not in REFRESH_CODES)
                and str(exc) in (REFRESH_CODES | {"interest_holding_evidence_incomplete"})
                and mismatch.get("coin", "USDGO") in {"USDGO", "USDC"}
            ):
                from .ledger import Ledger
                from .verified_savings_locations import observe

                try:
                    recognized = observe(execution, read_at)
                except ExecutionError:
                    # Delayed/ambiguous evidence remains the original bounded
                    # reconciliation wait; never relax its balance requirements.
                    recognized = None
                if recognized:
                    state, balances = recognized
                    execution.state = state
                    return (
                        Ledger.restore(state["book"]),
                        state["holdings"],
                        state["receipts"],
                        [],
                        state["income_accounting"],
                        balances,
                        now,
                    )
            if str(exc) not in REFRESH_CODES or attempt:
                raise
            logging.getLogger(__name__).info(
                "Account evidence refresh: code=%s attempt=1 limit=1 read_only=true", str(exc)
            )
    raise AssertionError("bounded_evidence_refresh_exhausted")
