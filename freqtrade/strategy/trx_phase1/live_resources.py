"""Host-side, fixed-container resource sampling. No Docker socket enters the observer."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def collect(proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    result = {"container_name": "trx-phase1-live", "checked_at": int(time.time() * 1000)}
    try:
        response = subprocess.run(
            ["docker", "inspect", "trx-phase1-live"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        container = json.loads(response.stdout)[0]
        state = container["State"]
        if not state["Running"]:
            raise ValueError("container_not_running")
        group = (proc_root / str(state["Pid"]) / "cgroup").read_text().strip().split("::")[-1]
        root = cgroup_root / group.lstrip("/")
        current, limit = (
            int((root / name).read_text()) for name in ("memory.current", "memory.max")
        )
        if current < 0 or limit <= 0:
            raise ValueError("memory_budget_unavailable")
        result.update(
            running=True,
            container_id=container["Id"],
            started_at=state["StartedAt"],
            restarts=container["RestartCount"],
            current=current,
            limit=limit,
            error=None,
        )
    except Exception as exc:
        result.update(running=False, error=type(exc).__name__)
    return result


def save(path, value, owner=None):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".trading-memory-", dir=path.parent)
    temporary = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        if owner is not None:
            os.fchown(fd, owner, owner)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--owner", type=int)
    args = parser.parse_args()
    save(args.output, collect(), args.owner)
