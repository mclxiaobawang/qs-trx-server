"""One fixed batch review; the original live API remains the capital journal writer.

Review preparation writes only the private review store. A user's confirmation can
install bounded metadata and apply that review while the caller holds the
original runtime lock. This helper must never run as an external journal writer.
No exchange client or order endpoint is present here.
"""

import fcntl
import json
import re
import sqlite3
import time
from contextlib import closing, contextmanager
from copy import deepcopy
from pathlib import Path


if __package__:
    from .contribution_preview import ContributionPreviewError, fingerprint, preview
    from .ledger import money
else:
    from contribution_preview import ContributionPreviewError, fingerprint, preview
    from ledger import money


class SelectedContributionError(ValueError):
    pass


class CoreUnavailable(Exception):
    """No definitive response; the same review is retained for reconciliation."""


class CoreRejected(Exception):
    """A completed HTTP response; journal evidence still decides the outcome."""

    def __init__(self, code):
        self.code = code if re.fullmatch(r"[a-z0-9_]{1,160}", code) else "core_confirmation_failed"


class SelectedContribution:
    METADATA = {"asset_contribution_review", "selected_contribution_batches"}

    def __init__(self, config, *, core, clock=None):
        self.config = config
        self.source = Path(config["source"])
        self.folder = Path(config["state"])
        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.scope = {
            "version": "fixed-usdgo-earn-contribution-v1",
            **{
                k: config[k]
                for k in [
                    "account_id",
                    "ledger_created_at",
                    "performance_started_at",
                    "source_sha256",
                    "initial_capital",
                ]
            },
            **deepcopy(config["selection"]),
        }
        self.policy_hash = fingerprint(self.scope)
        self.core = core
        self.reviews = self.folder / "reviews.sqlite"
        with closing(sqlite3.connect(self.reviews)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reviews (hash TEXT PRIMARY KEY,payload TEXT NOT NULL)"
            )
        self.reviews.chmod(0o600)

    def read(self, name):
        path = self.source / (name + ".sqlite")
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
        ) as db:
            row = db.execute("SELECT version,state FROM trx_state WHERE id=?", (name,)).fetchone()
        if row is None:
            raise SelectedContributionError("contribution_source_unavailable")
        state = json.loads(row[1])
        if state.get("account_id") != self.scope["account_id"]:
            raise SelectedContributionError("contribution_account_mismatch")
        return row[0], state

    def selection(self, requested=None):
        for key, value in (requested or {}).items():
            expected = self.scope.get(key)
            if expected is None or str(value) != str(expected):
                raise SelectedContributionError("contribution_fixed_selection_mismatch")

    def project(self, execution=None):
        e = execution if execution is not None else self.read("execution")[1]
        s = self.read("strategy")[1]
        if (
            e["created_at"] != self.scope["ledger_created_at"]
            or s["activated_at"] != self.scope["performance_started_at"]
            or s["source_sha256"] != self.scope["source_sha256"]
            or money(e["book"]["initial"]) != money(self.scope["initial_capital"])
        ):
            raise SelectedContributionError("contribution_original_basis_changed")
        if any(r.get("status") == "confirmed" for r in e.get("asset_withdrawals", {}).values()):
            raise SelectedContributionError("contribution_withdrawal_pending")
        try:
            result = preview(
                e,
                s,
                self.scope["coin"],
                self.scope["amount"],
                self.clock(),
                holding_id=self.scope["holding_id"],
            )
        except ContributionPreviewError as exc:
            raise SelectedContributionError(str(exc)) from None
        if result["selection"]["product_id"] != self.scope["product_id"]:
            raise SelectedContributionError("contribution_fixed_product_mismatch")
        return result

    def marker(self, state):
        marker = state.get("selected_contribution_batches", {}).get(self.scope["batch_id"])
        if marker and marker.get("policy_sha256") != self.policy_hash:
            raise SelectedContributionError("contribution_batch_policy_changed")
        return marker

    def receipt(self, state, marker):
        if not marker:
            return None
        record = state.get("asset_contributions", {}).get(marker["review_hash"])
        if record is None:
            return None
        assets = record.get("assets", [])
        expected = self.scope
        if (
            record.get("review_hash") != marker["review_hash"]
            or len(assets) != 1
            or assets[0].get("coin") != "USDGO"
            or money(assets[0]["amount"]) != money(expected["amount"])
            or len(assets[0].get("locations", [])) != 1
        ):
            raise SelectedContributionError("contribution_receipt_scope_mismatch")
        location = assets[0]["locations"][0]
        if any(location.get(k) != expected[k] for k in ["holding_id", "product_id"]):
            raise SelectedContributionError("contribution_receipt_scope_mismatch")
        if location.get("location") != "earn" or money(location["amount"]) != money(
            expected["amount"]
        ):
            raise SelectedContributionError("contribution_receipt_scope_mismatch")
        return deepcopy(record)

    def status(self):
        state = self.read("execution")[1]
        marker = self.marker(state)
        record = self.receipt(state, marker)
        return {
            "status": "confirmed"
            if record
            else "confirmation_pending"
            if marker
            else "not_confirmed",
            "selection": deepcopy(self.config["selection"]),
            "review_hash": marker["review_hash"] if marker else None,
            "record": record,
        }

    def prepare(self, requested=None):
        self.selection(requested)
        status = self.status()
        if status["status"] == "confirmed":
            raise SelectedContributionError("contribution_batch_already_confirmed")
        if status["status"] == "confirmation_pending":
            return self.response(self.saved(status["review_hash"]), "confirmation_pending")
        projection = self.project()
        now = self.clock()
        selected = projection["selection"]
        review = {
            "account_id": self.scope["account_id"],
            "created_at": now,
            "expires_at": now + 5 * 60000,
            "basis": "contribution_market_value",
            "selection_scope": deepcopy(self.scope),
            "assets": [
                {
                    "coin": "USDGO",
                    "amount": self.scope["amount"],
                    "mark": selected["mark_usdt"],
                    "value": selected["value_usdt"],
                    "locations": [
                        {
                            "location": "earn",
                            "holding_id": self.scope["holding_id"],
                            "product_id": self.scope["product_id"],
                            "amount": self.scope["amount"],
                        }
                    ],
                }
            ],
            "value": selected["value_usdt"],
        }
        review["review_hash"] = fingerprint(review)
        saved = {"review": review, "projection": projection}
        with closing(sqlite3.connect(self.reviews)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO reviews VALUES(?,?)",
                (review["review_hash"], json.dumps(saved, allow_nan=False)),
            )
        return self.response(saved)

    @staticmethod
    def response(saved, status="reviewed"):
        return {
            **deepcopy(saved["review"]),
            "projection": deepcopy(saved["projection"]),
            "status": status,
            "fixed_selection": True,
            "requires_user_confirmation": True,
        }

    def saved(self, review_hash):
        with closing(sqlite3.connect(self.reviews)) as db:
            row = db.execute("SELECT payload FROM reviews WHERE hash=?", (review_hash,)).fetchone()
        if row is None:
            raise SelectedContributionError("contribution_selected_review_not_found")
        saved = json.loads(row[0])
        review = saved["review"]
        if (
            review.get("review_hash") != review_hash
            or fingerprint({k: v for k, v in review.items() if k != "review_hash"}) != review_hash
            or review.get("selection_scope") != self.scope
        ):
            raise SelectedContributionError("contribution_selected_review_changed")
        return saved

    @contextmanager
    def lock(self, path):
        with path.open("a+") as handle:
            path.chmod(0o600)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SelectedContributionError("contribution_confirmation_busy") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def backup(self, review_hash):
        target = self.folder / "backups" / review_hash
        target.mkdir(parents=True, mode=0o700, exist_ok=True)
        for key in ["execution", "strategy"]:
            path = target / (key + ".sqlite")
            if not path.exists():
                with closing(
                    sqlite3.connect(
                        (self.source / (key + ".sqlite")).resolve().as_uri() + "?mode=ro", uri=True
                    )
                ) as source:
                    with closing(sqlite3.connect(path)) as destination:
                        source.backup(destination)
                path.chmod(0o600)

    def patch(self, version, old, new):
        # This service may edit only the bounded review and once-per-batch marker.
        if {k: v for k, v in old.items() if k not in self.METADATA} != {
            k: v for k, v in new.items() if k not in self.METADATA
        }:
            raise SelectedContributionError("contribution_metadata_boundary_violation")
        path = (self.source / "execution.sqlite").resolve().as_uri() + "?mode=rw"
        with closing(sqlite3.connect(path, uri=True, timeout=2)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "UPDATE trx_state SET version=version+1,state=? WHERE id='execution' AND version=?",
                (json.dumps(new, allow_nan=False), version),
            ).rowcount
            if count != 1:
                raise SelectedContributionError("contribution_execution_changed")

    def install(self, review):
        with self.lock(self.source / "execution.lock"):
            version, state = self.read("execution")
            marker = self.marker(state)
            record = self.receipt(state, marker)
            if record:
                return record
            if marker and marker["review_hash"] != review["review_hash"]:
                raise SelectedContributionError("contribution_previous_confirmation_pending")
            if marker:
                # Even an expired pending request must reach the native idempotent
                # handler, which either finds its receipt or definitively rejects it.
                return None
            if not review["created_at"] <= self.clock() < review["expires_at"]:
                raise SelectedContributionError("contribution_selected_review_expired")
            current = self.project(state)
            if abs(
                money(current["selection"]["mark_usdt"]) / money(review["assets"][0]["mark"]) - 1
            ) > money(".01"):
                raise SelectedContributionError("contribution_price_changed")
            previous = state.get("asset_contribution_review")
            if previous and previous.get("expires_at", 0) > self.clock():
                raise SelectedContributionError("contribution_other_review_active")
            self.backup(review["review_hash"])
            new = deepcopy(state)
            new["asset_contribution_review"] = deepcopy(review)
            new.setdefault("selected_contribution_batches", {})[self.scope["batch_id"]] = {
                "review_hash": review["review_hash"],
                "policy_sha256": self.policy_hash,
                "created_at": self.clock(),
                "selection": deepcopy(self.config["selection"]),
            }
            self.patch(version, state, new)
        return None

    def clear_rejected(self, review_hash):
        with self.lock(self.source / "execution.lock"):
            version, state = self.read("execution")
            marker = self.marker(state)
            if not marker or marker["review_hash"] != review_hash or self.receipt(state, marker):
                return
            new = deepcopy(state)
            del new["selected_contribution_batches"][self.scope["batch_id"]]
            if (new.get("asset_contribution_review") or {}).get("review_hash") == review_hash:
                new["asset_contribution_review"] = None
            self.patch(version, state, new)

    def confirm(self, review_hash, authorization):
        saved = self.saved(review_hash)
        with self.lock(self.folder / "confirm.lock"):
            current = self.status()
            if current["record"]:
                return current["record"]
            record = self.install(saved["review"])
            if record:
                return record
            rejected = None
            try:
                self.core.confirm(review_hash, authorization)
            except CoreRejected as exc:
                rejected = exc.code
            except CoreUnavailable:
                pass
            current = self.status()
            if current["record"]:
                return current["record"]
            if rejected:
                self.clear_rejected(review_hash)
                raise SelectedContributionError(rejected)
            raise SelectedContributionError("contribution_confirmation_pending_check_status")


def fixed_policy(source):
    """Load the privately approved batch; never infer account or asset scope."""
    source = Path(source)
    path = source / "selected-contribution-policy.local.json"
    fields = {
        "account_id",
        "ledger_created_at",
        "performance_started_at",
        "source_sha256",
        "initial_capital",
        "selection",
    }
    try:
        if not path.is_file() or path.stat().st_size > 65536:
            raise ValueError
        policy = json.loads(path.read_text())
        if not isinstance(policy, dict) or set(policy) != fields:
            raise ValueError
        selection = policy["selection"]
        if not isinstance(selection, dict) or set(selection) != {
            "batch_id",
            "coin",
            "amount",
            "holding_id",
            "product_id",
        }:
            raise ValueError
        if (
            not all(
                isinstance(policy[key], str) and 0 < len(policy[key]) <= 128
                for key in ("account_id", "source_sha256", "initial_capital")
            )
            or not re.fullmatch(r"[a-f0-9]{64}", policy["source_sha256"])
            or not all(
                type(policy[key]) is int and policy[key] > 0
                for key in ("ledger_created_at", "performance_started_at")
            )
            or not money(policy["initial_capital"]).is_finite()
            or money(policy["initial_capital"]) <= 0
            or not all(
                isinstance(value, str) and 0 < len(value) <= 128 for value in selection.values()
            )
            or selection["coin"] != "USDGO"
            or not re.fullmatch(r"\d+(?:\.\d{1,8})?", selection["amount"])
            or money(selection["amount"]) <= 0
        ):
            raise ValueError
    except (OSError, ValueError, TypeError, ArithmeticError):
        raise SelectedContributionError("contribution_private_policy_required") from None
    return {
        "source": str(source),
        "state": str(source / "selected-contribution"),
        **policy,
    }
