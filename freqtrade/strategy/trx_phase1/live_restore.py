"""Opt-in server recovery of an existing completed-startup live account only.

No initializer is called here. Missing, partial or mismatched migrated state
cannot become a new allocation. Manual pauses and review blocks remain paused.
"""

import json
import logging
import re
import sqlite3
import threading
from contextlib import closing

from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest
from freqtrade.strategy.trx_phase1.live_service import LiveSetup, terms_review


logger = logging.getLogger(__name__)


def existing_state(service):
    control = service.load()
    if not control or not re.fullmatch("[a-f0-9]{64}", control.get("account_id", "")):
        raise ExecutionError("existing_live_account_required")
    account = control["account_id"]
    rows = {}
    for name, key in (("execution", "execution"), ("strategy", "strategy")):
        path = service.root / "live_execution" / account / f"{name}.sqlite"
        if not path.is_file():
            raise ExecutionError("existing_live_account_required")
        try:
            with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db, db:
                row = db.execute("SELECT state FROM trx_state WHERE id=?", (key,)).fetchone()
                if not row:
                    raise ExecutionError("existing_live_account_required")
                rows[name] = json.loads(row[0])
        except (sqlite3.Error, ValueError):
            raise ExecutionError("existing_live_state_invalid") from None
    execution, strategy = rows["execution"], rows["strategy"]
    if strategy.get("startup_complete") is not True or not strategy.get("activated_at"):
        raise ExecutionError("completed_live_startup_required")
    source = digest(service.source())
    if (
        execution.get("account_id") != account
        or strategy.get("account_id") != account
        or control.get("source_sha256") != source
        or strategy.get("source_sha256") != source
        or digest(strategy.get("source_snapshot")) != source
        or execution.get("products") != control["setup"]["products"]
        or money(execution["book"]["initial"]) != money(control["setup"]["capital"])
    ):
        raise ExecutionError("frozen_live_setup_mismatch")
    return control, strategy


class ExistingLiveRestore:
    def __init__(self, service):
        self.service = service
        self.stopping = threading.Event()
        self.activation = threading.Lock()
        self.thread = None

    def once(self):
        if self.stopping.is_set():
            return False
        service = self.service
        with service.lock:
            runtime = service.runtime
            if runtime and runtime.thread and runtime.thread.is_alive():
                return False
            control, saved = existing_state(service)
            if saved["desired"] != "running" or saved["phase"] == "blocked":
                return False
            if service.credential_identity() != control["credentials_identity"]:
                raise ExecutionError("credentials_changed_requires_preflight")
            runtime = service.get_runtime(control)
            reader = service.broker_factory(False)
            writer, market = None, None
            try:
                if reader.account_id() != control["account_id"]:
                    raise ExecutionError("account_binding_mismatch")
                runtime.execution.broker = reader
                funding = runtime.environment(service.clock())
                if (
                    terms_review(
                        service.source(), LiveSetup.model_validate(control["setup"]), funding
                    )
                    != control["review_hash"]
                ):
                    runtime.state.update(
                        desired="paused",
                        phase="blocked",
                        error="live_terms_changed_review_required",
                    )
                    runtime.persist()
                    return False
                writer = service.broker_factory(True)
                if writer.account_id() != control["account_id"]:
                    raise ExecutionError("account_binding_mismatch")
                market = service.market_factory()
                runtime.execution.broker, runtime.market = writer, market
                expected = control["credentials_identity"]
                runtime.credential_guard = lambda: service.credential_identity() == expected
                with self.activation:
                    if self.stopping.is_set():
                        return False
                    runtime.start()
                # The worker owns these transports after successful activation.
                writer, market = None, None
                logger.info("Existing TRX live account restored; completed startup retained.")
                return True
            finally:
                service.close_broker(reader)
                if writer is not None:
                    service.close_broker(writer)
                if market is not None:
                    market.close()

    def run(self):
        while not self.stopping.is_set():
            try:
                self.once()
            except Exception as exc:
                # Transport exceptions may contain signed request details.
                logger.warning("Existing TRX live restoration deferred (%s).", type(exc).__name__)
            self.stopping.wait(30)

    def start(self):
        self.thread = threading.Thread(target=self.run, name="trx-cloud-restore", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        with self.activation:
            pass
        if self.thread:
            self.thread.join(timeout=2)
