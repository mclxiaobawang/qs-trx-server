"""Short-lived native executor, run by the host timer in the existing container.

Sibling modules are installed in a versioned tools directory. Never patches the
running worker, places exchange orders, or writes the strategy planning database.
"""

import fcntl
import hashlib
import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path


if __package__:
    from .auto_rebate import (
        AdmissionDeferred,
        batch_policy,
        fingerprint,
        read_json,
        supervised_commit,
    )
    from .credit_contribution import candidate
else:
    from auto_rebate import (
        AdmissionDeferred,
        batch_policy,
        fingerprint,
        read_json,
        supervised_commit,
    )
    from credit_contribution import candidate

from freqtrade.strategy.trx_phase1.account import load_settings
from freqtrade.strategy.trx_phase1.live_broker import (
    ClassicBroker,
    ClassicTransport,
    ExecutionError,
)
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from freqtrade.strategy.trx_phase1.request_limits import RequestLimiter


def run(root):  # noqa: C901 - keep the bounded native transaction protocol together.
    control = read_json(root / "live-strategy-control.json")
    account = control["account_id"]
    source = root / "live_execution" / account
    policy_path = source / "auto-rebate-policy.json"
    policy = read_json(policy_path)
    policy_hash = fingerprint(policy)
    worker_file = Path("/app/freqtrade/strategy/trx_phase1/live_strategy.py")
    if hashlib.sha256(worker_file.read_bytes()).hexdigest() != policy["worker_module_sha256"]:
        raise AdmissionDeferred("worker_compatibility_changed")

    def window():
        age = (time.time_ns() - (source / "strategy.sqlite").stat().st_mtime_ns) / 1e9
        with closing(
            sqlite3.connect(
                (source / "strategy.sqlite").as_uri() + "?mode=ro", uri=True, timeout=0.1
            )
        ) as db:
            db.execute("PRAGMA query_only=ON")
            if db.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise AdmissionDeferred("worker_journal_changed")
            version, raw = db.execute(
                "SELECT version,state FROM trx_state WHERE id='strategy'"
            ).fetchone()
        state = json.loads(raw)
        now = int(time.time() * 1000)
        valid = (
            0.15 <= age < 1.0
            and 0 <= now - state["last_poll_at"] < 10000
            and state["desired"] == "running"
            and state["phase"] == "observing"
            and state["startup_complete"]
            and not state.get("error")
            and not state.get("intent")
            and not state["engine"].get("pending")
            and state["source_sha256"] == policy["source_sha256"]
            and min(state["engine"]["buy_at"], state["engine"]["sell_at"]) - now > 5000
        )
        return valid, version, state

    settings, _, complete = load_settings(root / "bitget-account.local.json")
    if not complete or not settings.enabled or settings.account_type != "classic":
        raise AdmissionDeferred("account_read_unavailable")
    transport = ClassicTransport(
        settings, writes_enabled=False, limiter=RequestLimiter(root / "request-limits.sqlite")
    )
    try:
        transport.sync_clock()
        broker = ClassicBroker(transport)
        if broker.writes_enabled or broker.account_id() != account:
            raise AdmissionDeferred("account_binding_mismatch")
        balances = broker.balances()
        wallet_at = int(time.time() * 1000)
        wallet = balances["USDT"]["available"]
        if any(float(balances["USDT"][k]) != 0 for k in ("frozen", "locked", "restricted")):
            raise AdmissionDeferred("wallet_not_idle")
        execution = LiveExecution(root, broker, account)
    finally:
        transport.session.close()  # No open network/SQLite handles enter the child.

    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        valid, version, _ = window()
        if valid:

            def commit(version=version):
                started = time.monotonic()
                with execution.exclusive():
                    valid, current_version, state = window()
                    if not valid or version != current_version:
                        raise AdmissionDeferred("worker_window_changed")
                    if fingerprint(read_json(policy_path)) != policy_hash:
                        raise AdmissionDeferred("authorization_changed")
                    if fingerprint(read_json(root / "live-strategy-control.json")) != fingerprint(
                        control
                    ):
                        raise AdmissionDeferred("runtime_control_changed")
                    now = int(time.time() * 1000)
                    if now - wallet_at >= 45000:
                        raise AdmissionDeferred("wallet_observation_expired")
                    execution.reload()
                    before = execution.state
                    batch = batch_policy(before, policy, now)
                    if not batch:
                        return {"phase": "idle", "adopted": 0}
                    work, record = candidate(before, batch, state["markets"], wallet, now)
                    for field in (
                        "initial",
                        "lots",
                        "orders",
                        "fees",
                        "fee_rebates",
                        "realized",
                        "interest_trx",
                        "stable_interest",
                    ):
                        if work["book"][field] != before["book"][field]:
                            raise AdmissionDeferred("capital_invariant_failed")
                    for field in ("receipts", "holdings", "quarantined_account_credits"):
                        if work[field] != before[field]:
                            raise AdmissionDeferred("receipt_invariant_failed")
                    if time.monotonic() - started >= 0.15 or window()[1] != version:
                        raise AdmissionDeferred("precommit_deadline")
                    execution.state = work
                    execution.save()  # Existing optimistic CAS and atomic journal transaction.
                return {
                    "phase": "admitted",
                    "adopted": len(record["source_credits"]),
                    "value": record["value"],
                    "review_hash": record["review_hash"],
                    "timestamp": record["timestamp"],
                    "commit_seconds": time.monotonic() - started,
                }

            if threading.active_count() != 1:
                raise AdmissionDeferred("native_executor_not_single_threaded")
            return supervised_commit(commit, timeout=2.0)
        time.sleep(0.05)
    raise AdmissionDeferred("waiting_for_worker_idle_window")


if __name__ == "__main__":
    try:
        with (Path(__file__).parent / "admission-job.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run(Path("/app/user_data/trx_phase1"))
    except (AdmissionDeferred, ExecutionError) as exc:
        result = {"phase": "deferred", "reason": str(exc)}
    except Exception:
        result = {"phase": "deferred", "reason": "native_read_unavailable"}
    print(json.dumps(result), flush=True)
