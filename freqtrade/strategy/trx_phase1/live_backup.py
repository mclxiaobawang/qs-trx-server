"""Bounded online backups; source SQLite files are always opened read-only.

This standalone stdlib script is suitable for a restricted systemd oneshot. It
never loads a broker, trading service, credentials or strategy initializer.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value, owner=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        temporary.chmod(0o600)
        if owner is not None:
            os.chown(temporary, *owner)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def state(path, key):
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.15)
    ) as db:
        db.execute("PRAGMA query_only=ON")
        row = db.execute("SELECT version,state FROM trx_state WHERE id=?", (key,)).fetchone()
    if row is None:
        raise ValueError("backup_source_state_missing")
    return row[0], json.loads(row[1])


def copy_database(source, destination, deadline):
    def progress(status, remaining, total):
        if time.monotonic() > deadline:
            raise TimeoutError("backup_time_budget_exceeded")

    with closing(
        sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.15)
    ) as reader:
        reader.execute("PRAGMA query_only=ON")
        with closing(sqlite3.connect(destination)) as writer:
            reader.backup(writer, pages=64, progress=progress, sleep=0.02)
    Path(destination).chmod(0o600)


def verify(directory, account):
    """Validate copied files in isolation; no restore is ever applied to the live account."""
    directory = Path(directory)
    for name in ("execution", "strategy"):
        with closing(
            sqlite3.connect(
                (directory / (name + ".sqlite")).resolve().as_uri() + "?mode=ro", uri=True
            )
        ) as db:
            until = time.monotonic() + 20
            db.set_progress_handler(lambda deadline=until: int(time.monotonic() > deadline), 10000)
            if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("backup_database_check_failed")
    ev, execution = state(directory / "execution.sqlite", "execution")
    sv, strategy = state(directory / "strategy.sqlite", "strategy")
    control = json.loads((directory / "control.json").read_text())
    if any(value.get("account_id") != account for value in (execution, strategy, control)):
        raise ValueError("backup_account_mismatch")
    if strategy.get("source_sha256") != control.get("source_sha256"):
        raise ValueError("backup_strategy_mismatch")
    orders = execution["book"].get("orders", [])
    if len({r["id"] for r in orders}) != len(orders):
        raise ValueError("backup_duplicate_fill")
    return {
        "execution_version": ev,
        "strategy_version": sv,
        "startup_complete": bool(strategy.get("startup_complete")),
        "order_count": len(orders),
        "created_at": execution["created_at"],
        "inflight": bool(execution.get("operation") or strategy.get("intent")),
        "restore_validation": "offline_database_and_identity_checks",
    }


def prune(root, maximum_bytes, keep_hourly=24, keep_daily=7):
    """Only prune this script's complete managed archives; preserve at least two."""
    managed = []
    for manifest in root.glob("trx-auto-*.json"):
        try:
            data = json.loads(manifest.read_text())
            archive = manifest.with_suffix(".tar.gz")
            if data.get("format") == "trx-online-backup-v1" and archive.is_file():
                managed.append((data["created_at"], archive, manifest))
        except (OSError, ValueError, KeyError):
            continue
    managed.sort(reverse=True)
    keep = {row[1] for row in managed[:keep_hourly]}
    days = set()
    for stamp, archive, _manifest in managed:
        day = datetime.fromtimestamp(stamp / 1000, UTC).date()
        if day not in days and len(days) < keep_daily:
            days.add(day)
            keep.add(archive)
    newest = {row[1] for row in managed[:2]}
    total = sum(a.stat().st_size + m.stat().st_size for _, a, m in managed)
    for _, archive, manifest in reversed(managed):
        if archive not in newest and (archive not in keep or total > maximum_bytes):
            total -= archive.stat().st_size + manifest.stat().st_size
            archive.unlink()
            manifest.unlink()
    return {
        "archives": sum(1 for _, a, _ in managed if a.exists()),
        "bytes": total,
        "limit_bytes": maximum_bytes,
        "over_limit": total > maximum_bytes,
    }


def run_backup(config):
    root, source = Path(config["destination"]), Path(config["source"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    status_path = Path(config["status"])
    status_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = status_path.parent / "backup.lock"
    with lock_path.open("a+") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prior = json.loads(status_path.read_text())
        except (OSError, ValueError):
            prior = {}
        try:
            required = sum(
                (source / (n + ".sqlite")).stat().st_size for n in ("execution", "strategy")
            )
            if shutil.disk_usage(root).free < max(1024**3, required * 3):
                raise ValueError("backup_insufficient_free_space")
            start = int(time.time() * 1000)
            deadline = time.monotonic() + 90
            with tempfile.TemporaryDirectory(prefix=".trx-auto-", dir=root) as temporary:
                folder = Path(temporary)
                for _attempt in range(3):
                    control = Path(config["control"]).read_bytes()
                    before = {
                        n: state(source / (n + ".sqlite"), n)[0] for n in ("execution", "strategy")
                    }
                    for name in ("execution", "strategy"):
                        copy_database(
                            source / (name + ".sqlite"), folder / (name + ".sqlite"), deadline
                        )
                    after = {
                        n: state(source / (n + ".sqlite"), n)[0] for n in ("execution", "strategy")
                    }
                    if before == after and Path(config["control"]).read_bytes() == control:
                        break
                else:
                    raise ValueError("backup_source_changed_retry_later")
                # Frozen control contains identity hashes, never exchange or SMTP secrets.
                frozen = json.loads(control)
                (folder / "control.json").write_text(json.dumps(frozen))
                (folder / "control.json").chmod(0o600)
                validation = verify(folder, config["account_id"])
                if (
                    validation["execution_version"] != before["execution"]
                    or validation["strategy_version"] != before["strategy"]
                ):
                    raise ValueError("backup_version_mismatch")
                if config.get("history") and Path(config["history"]).is_file():
                    copy_database(config["history"], folder / "history.sqlite", deadline)
                files = {p.name: checksum(p) for p in folder.iterdir() if p.is_file()}
                manifest = {
                    "format": "trx-online-backup-v1",
                    "created_at": start,
                    "account_id": config["account_id"],
                    "files": files,
                    "validation": validation,
                    "credential_files_included": False,
                    "deployment": config.get("deployment", {}),
                }
                (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
                name = "trx-auto-" + datetime.fromtimestamp(start / 1000, UTC).strftime(
                    "%Y%m%dT%H%M%S"
                )
                archive = root / (name + ".tar.gz")
                pending = root / (name + ".partial")
                with tarfile.open(pending, "w:gz", compresslevel=1) as tar:
                    for file in sorted(folder.iterdir()):
                        tar.add(file, arcname=file.name)
                pending.chmod(0o600)
                # Read back the archive in a fresh directory, including file hashes.
                with tempfile.TemporaryDirectory(prefix=".trx-check-", dir=root) as check:
                    unpack_archive(pending, check)
                    verify(check, config["account_id"])
                manifest["archive_sha256"] = checksum(pending)
                pending.replace(archive)
                atomic_json(root / (name + ".json"), manifest)
            retention = prune(root, int(config.get("maximum_bytes", 2 * 1024**3)))
            result = {
                "last_success_at": int(time.time() * 1000),
                "last_error": None,
                "last_archive": archive.name,
                "archive_bytes": archive.stat().st_size,
                "sha256": manifest["archive_sha256"],
                "verification": validation,
                "retention": retention,
                "offsite_configured": False,
                "schedule": "hourly",
                "credential_files_included": False,
            }
            atomic_json(status_path, result, config.get("status_owner"))
            return result
        except Exception as exc:
            atomic_json(
                status_path,
                {
                    **prior,
                    "last_attempt_at": int(time.time() * 1000),
                    "last_error": type(exc).__name__,
                },
                config.get("status_owner"),
            )
            raise


def unpack_archive(archive, destination):
    with tarfile.open(archive) as tar:
        names = {
            "execution.sqlite",
            "strategy.sqlite",
            "control.json",
            "history.sqlite",
            "manifest.json",
        }
        for member in tar.getmembers():
            if member.name not in names or not member.isfile():
                raise ValueError("backup_archive_member_invalid")
        tar.extractall(destination, filter="data")
    manifest = json.loads((Path(destination) / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if name not in names or checksum(Path(destination) / name) != expected:
            raise ValueError("backup_checksum_mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_backup(json.loads(Path(args.config).read_text()))
    print(
        json.dumps(
            {
                "ok": True,
                "last_success_at": result["last_success_at"],
                "archive_bytes": result["archive_bytes"],
            }
        )
    )
