"""Host systemd one-shot coordinator. Idle polls perform local read-only work only."""

import json
import os
import sqlite3
import subprocess
import tempfile
import time
from contextlib import closing
from pathlib import Path


if __package__:
    from .auto_rebate import AdmissionDeferred, eligible, fingerprint, read_json
else:
    from auto_rebate import AdmissionDeferred, eligible, fingerprint, read_json


def read_state(source, key):
    with closing(
        sqlite3.connect((source / (key + ".sqlite")).as_uri() + "?mode=ro", uri=True, timeout=0.15)
    ) as db:
        db.execute("PRAGMA query_only=ON")
        return json.loads(
            db.execute("SELECT state FROM trx_state WHERE id=?", (key,)).fetchone()[0]
        )


def atomic_status(path, value):
    fd, name = tempfile.mkstemp(prefix=".auto-rebate-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.chown(name, 10001, 10001)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def run(root):
    account = read_json(root / "live-strategy-control.json")["account_id"]
    source = root / "live_execution" / account
    policy = read_json(source / "auto-rebate-policy.json")
    status = {"account_id": account, "policy_sha256": fingerprint(policy), "phase": "deferred"}
    try:
        now = int(time.time() * 1000)
        execution = read_state(source, "execution")
        rows = eligible(execution, policy, now)
        status["pending_receipts"] = len(rows)
        if rows:
            state = read_state(source, "strategy")
            if (
                state["desired"] != "running"
                or state["phase"] != "observing"
                or not state["startup_complete"]
                or state.get("error")
                or execution.get("error")
                or execution.get("operation")
                or state.get("intent")
                or state["engine"].get("pending")
                or now - state["last_poll_at"] > 120000
                or state["source_sha256"] != policy["source_sha256"]
            ):
                raise AdmissionDeferred("waiting_for_idle_reconciliation")
            worker = json.loads(
                subprocess.check_output(["docker", "inspect", "trx-phase1-live"], timeout=10)
            )[0]
            if not worker["State"]["Running"] or worker["Image"] != policy["worker_image_id"]:
                raise AdmissionDeferred("worker_compatibility_changed")
            script = "/app/user_data/trx_phase1/operator-tools/auto-rebate-v1/auto_rebate_native.py"
            run = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-e",
                    "OPENBLAS_NUM_THREADS=1",
                    "-e",
                    "OMP_NUM_THREADS=1",
                    "-e",
                    "NUMEXPR_NUM_THREADS=1",
                    "trx-phase1-live",
                    "python",
                    script,
                ],
                capture_output=True,
                text=True,
                timeout=80,
                check=True,
            )
            status.update(json.loads(run.stdout))
        else:
            status["phase"] = "idle"
    except AdmissionDeferred as exc:
        status["reason"] = str(exc)
    except Exception:
        status["reason"] = "automation_read_unavailable"
    status["checked_at"] = int(time.time() * 1000)
    atomic_status(source / "auto-rebate-status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    run(Path("/opt/trx-phase1/user_data/trx_phase1"))
