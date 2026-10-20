"""Standing authorization and read model for verified USDT rebates as new capital.

This module intentionally has only standard-library dependencies. The observer
can read it without importing the trading runtime or obtaining exchange access.
"""

import hashlib
import json
import os
import select
import signal
import time
from decimal import Decimal
from pathlib import Path


class AdmissionDeferred(Exception):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError("automation_file_too_large")
    return json.loads(path.read_text())


def validate_policy(state, policy, now):
    if (
        policy.get("version") != "auto-rebate-capital-v1"
        or policy.get("enabled") is not True
        or policy.get("account_id") != state["account_id"]
        or policy.get("ledger_created_at") != state["created_at"]
        or not 0 < policy.get("approved_at", 0) <= now
        or policy.get("starts_after_recorded_at") != policy["approved_at"]
        or policy.get("scope") != "verified_native_usdt_rebates"
        or policy.get("classification") != "new_capital"
        or not policy.get("source_sha256")
        or not policy.get("worker_image_id")
    ):
        raise AdmissionDeferred("policy_not_authorized")


def eligible(state, policy, now):
    """Only native reconciled receipts; all previous capital admissions deduplicate."""
    validate_policy(state, policy, now)
    used = {
        r["receipt_id"]
        for c in state.get("asset_contributions", {}).values()
        for r in c.get("source_credits", [])
    }
    result = []
    for key, row in state.get("quarantined_account_credits", {}).items():
        if key in used or row.get("recorded_at", 0) < policy["starts_after_recorded_at"]:
            continue
        if row.get("coin") != "USDT" or row.get("type") != "REBATE_REWARDS":
            continue
        native = {
            k: row.get(k)
            for k in ("bill_id", "order_id", "coin", "group", "type", "amount", "fees", "timestamp")
        }
        if (
            key != "account-credit:" + str(row.get("bill_id"))
            or state.get("receipts", {}).get(key) != fingerprint(native)
            or row.get("group") != "other"
            or Decimal(row["fees"]) != 0
            or row.get("attribution") != "external_account_credit"
            or row.get("destination") != "spot"
            or not state["created_at"] <= row["timestamp"] <= row["recorded_at"] <= now
            or not Decimal(row["offset_delta"]).is_finite()
            or Decimal(row["offset_delta"]) <= 0
        ):
            raise AdmissionDeferred("receipt_evidence_mismatch")
        result.append({"receipt_id": key, "bill_id": row["bill_id"], "amount": row["offset_delta"]})
    return sorted(result, key=lambda r: r["receipt_id"])


def batch_policy(state, policy, now):
    rows = eligible(state, policy, now)[:16]
    if not rows:
        return None
    return {
        "version": "account-credit-capital-v1",
        "account_id": state["account_id"],
        "ledger_created_at": state["created_at"],
        "approved_at": policy["approved_at"],
        "expires_at": now + 120000,
        "authorization_policy_sha256": fingerprint(policy),
        "credits": rows,
    }


def supervised_commit(commit, timeout=2.0):
    """Kill a delayed commit before the worker's 15-second sleep ends.

    The child must close network/SQLite handles before forking and use native
    transactional persistence. Death releases flock/SQLite locks; replay checks
    the durable receipt IDs even if COMMIT succeeded but its response was lost.
    """
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            result = {"ok": True, "result": commit()}
        except BaseException:
            result = {"ok": False, "reason": "commit_deferred"}
        try:
            payload = json.dumps(result, allow_nan=False).encode()
            if len(payload) > 4096:
                payload = b'{"ok":false,"reason":"commit_response_too_large"}'
            os.write(writer, payload)
        finally:
            os.close(writer)
            os._exit(0)
    os.close(writer)
    deadline = time.monotonic() + timeout
    try:
        if not select.select([reader], [], [], max(0, deadline - time.monotonic()))[0]:
            raise AdmissionDeferred("commit_deadline")
        payload = os.read(reader, 4096)
        if not payload:
            raise AdmissionDeferred("commit_response_lost")
        value = json.loads(payload)
        if not value["ok"]:
            raise AdmissionDeferred(value["reason"])
        return value["result"]
    finally:
        # Always reap the child, including malformed/lost responses and timeouts.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        os.close(reader)


def read_status(source, state, now):
    """A stale daemon can never masquerade as healthy; totals come from the book."""
    records = [
        c
        for c in state.get("asset_contributions", {}).values()
        if c.get("classification") == "automatic_capital_unlinked_rebate"
    ]
    result = {
        "enabled": False,
        "phase": "unavailable",
        "fresh": False,
        "checked_at": None,
        "approved_at": None,
        "last_success_at": None,
        "adopted_receipts": sum(len(c["source_credits"]) for c in records),
        "adopted_amount": str(sum((Decimal(c["value"]) for c in records), Decimal(0))),
        "pending_receipts": None,
    }
    try:
        policy = read_json(Path(source) / "auto-rebate-policy.json")
        validate_policy(state, policy, now)
        result.update(enabled=True, approved_at=policy["approved_at"])
        result["pending_receipts"] = len(eligible(state, policy, now))
        status = read_json(Path(source) / "auto-rebate-status.json")
        if status.get("account_id") != state["account_id"] or status.get(
            "policy_sha256"
        ) != fingerprint(policy):
            raise ValueError("status_binding_mismatch")
        checked = status["checked_at"]
        fresh = -5000 <= now - checked <= 90000
        result.update(
            checked_at=checked,
            fresh=fresh,
            phase=status["phase"] if fresh else "stale",
            reason=status.get("reason"),
            last_success_at=max((c["timestamp"] for c in records), default=None),
        )
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError, AdmissionDeferred):
        pass
    return result
