"""Local management of a frozen C1 runtime. Merely visiting never connects."""

import hashlib
import json
import os
import secrets
import threading
import time
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from freqtrade.strategy.trx_phase1.account import load_settings
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import (
    ClassicBroker,
    ClassicTransport,
    ExecutionError,
)
from freqtrade.strategy.trx_phase1.live_execution import (
    LiveExecution,
    ProductRule,
    allocation_lock,
    digest,
)
from freqtrade.strategy.trx_phase1.live_funding import LiveFunding
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy
from freqtrade.strategy.trx_phase1.price_position import build_price_reference
from freqtrade.strategy.trx_phase1.request_limits import RequestLimiter
from freqtrade.strategy.trx_phase1.shadow_market import (
    PublicShadowMarket,
    seed_history,
    validate_quote,
)


EXPERIMENT = "a108ad7e-ed76-41e9-ac10-aa07b0e6b95e"
RUN = "8c2b6b9d-dcd2-40c1-a9af-4d624b5990cf"
LABEL = "C1 D | 首次 TRX 10% | 仅动态周期 + P75 365天"


def terms_review(snapshot, setup, funding):
    return digest(
        {
            "source": digest(snapshot),
            "setup": setup.model_dump(mode="json"),
            "fees": {s: m["fee"] for s, m in funding.markets.items()},
            "terms": {
                coin: {
                    k: t.get(k)
                    for k in (
                        "status",
                        "subscribePrecision",
                        "singleMinAmount",
                        "singleMaxAmount",
                        "tiers",
                    )
                }
                for coin, t in funding.terms.items()
            },
        }
    )


class LiveSetup(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    capital: str
    products: list[ProductRule] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def valid_setup(self):
        if money(self.capital) <= 0:
            raise ValueError("Positive allocated capital required")
        if {p.coin for p in self.products} != {"TRX", "USDGO"} or any(
            p.redemption_minimum is None for p in self.products
        ):
            raise ValueError("Both selected products require redemption evidence")
        return self


class LiveService:
    def __init__(
        self,
        user_dir,
        *,
        broker_factory=None,
        market_factory=PublicShadowMarket,
        clock=None,
        existing_only=False,
    ):
        self.root = Path(user_dir) / "trx_phase1"
        self.control = self.root / "live-strategy-control.json"
        self.source_path = self.root / "experiments" / EXPERIMENT / "results" / (RUN + ".json")
        self.seed = (
            self.root
            / "control_matrices/c1-seven-earn-oct16-window-2026-10-16/inputs"
            / "history-through-2026-10-16.feather"
        )
        self.broker_factory = broker_factory or self.broker
        self.market_factory = (
            (
                lambda: PublicShadowMarket(
                    limiter=RequestLimiter(self.root / "request-limits.sqlite")
                )
            )
            if market_factory is PublicShadowMarket
            else market_factory
        )
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.lock = threading.RLock()
        self.runtime = None
        self._source = None
        self.existing_only = existing_only

    def source(self):
        if self._source is not None:
            return deepcopy(self._source)
        saved = json.loads(self.source_path.read_text())
        cfg = ResearchConfig.from_saved(saved["config"])
        if saved["config"] != cfg.model_dump():
            raise ExecutionError("complete_strategy_snapshot_required")
        self._source = cfg.model_dump()
        return deepcopy(self._source)

    def broker(self, writes=False):
        settings, _, complete = load_settings(self.root / "bitget-account.local.json")
        if not complete or not settings.enabled or settings.account_type != "classic":
            raise ExecutionError("enabled_classic_credentials_required")
        transport = ClassicTransport(
            settings,
            writes_enabled=writes,
            limiter=RequestLimiter(self.root / "request-limits.sqlite"),
        )
        transport.sync_clock()
        return ClassicBroker(transport)

    def credential_identity(self):
        settings, _, complete = load_settings(self.root / "bitget-account.local.json")
        if not complete or not settings.enabled or settings.account_type != "classic":
            raise ExecutionError("enabled_classic_credentials_required")
        return hashlib.sha256(
            (
                settings.model_dump_json()
                + "\0".join(
                    v.get_secret_value()
                    for v in (settings.api_key, settings.secret_key, settings.passphrase)
                )
            ).encode()
        ).hexdigest()

    def load(self):
        return json.loads(self.control.read_text()) if self.control.exists() else None

    def save(self, value):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.control.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(self.control)

    def get_runtime(self, control):
        if self.runtime is None:
            execution = LiveExecution(self.root, None, control["account_id"])
            self.runtime = LiveStrategy(execution, None)
            if self.runtime.state is None:
                raise ExecutionError("strategy_state_unavailable")
        return self.runtime

    def redemption_verified(self, rule, account_id):
        if not account_id:
            return None
        path = (
            self.root / "execution_checks/minimum-redemption-2026-10-17" / (rule["coin"] + ".json")
        )
        try:
            if path.stat().st_size > 65536:
                return None
            evidence = json.loads(path.read_text())
            request = evidence["request"]
            checks = evidence["checks"]
            if (
                evidence["account_id"] == account_id
                and evidence["rules_sha256"] == digest(rule)
                and evidence["verification_version"] == 2
                and evidence["status"] == "verified"
                and request["productId"] == rule["product_id"]
                and money(request["amount"]) == money(rule["redemption_minimum"])
                and evidence.get("exchange_id")
                and all(
                    checks.get(k) is True
                    for k in (
                        "exchange_result",
                        "arrival_bill",
                        "holding_reduction",
                        "spendable_arrival",
                    )
                )
            ):
                return evidence["checked_at"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def acceptance_pending(self):
        path = self.root / "execution_checks/minimum-full-chain-2026-10-17/control.json"
        if not path.exists():
            return False
        try:
            if path.stat().st_size > 65536:
                raise ValueError("oversize")
            return not json.loads(path.read_text()).get("released_at")
        except (OSError, ValueError, TypeError):
            raise ExecutionError("invalid_acceptance_state") from None

    def configuration(self, account=None, control=None, account_id=None):  # noqa: C901 - coherent draft
        """Derive a display/preflight draft, never allocate or import account assets."""
        account = account or {}
        snapshot = account.get("snapshot") or {}
        checks = account.get("checks", {})
        observed_at = snapshot.get("observed_at")
        stale = bool(account.get("stale") or account.get("error")) or not observed_at
        if observed_at:
            stale = stale or not 0 <= self.clock() - observed_at <= 60000
        path = self.root / "live-product-rules.local.json"
        rules, notes, rules_error = [], {}, None
        try:
            if path.stat().st_size > 65536:
                raise ValueError("oversize")
            saved = json.loads(path.read_text())
            if saved["schema"] != 1:
                raise ValueError("schema")
            rules = LiveSetup(capital="1", products=saved["products"]).model_dump(mode="json")[
                "products"
            ]
            notes = {str(k): str(v)[:1000] for k, v in saved.get("notes", {}).items()}
        except FileNotFoundError:
            rules_error = "missing_product_rules"
        except Exception:
            rules_error = "invalid_product_rules"
        frozen = control.get("setup") if control else None
        if frozen:
            rules, rules_error = frozen["products"], None
        available, restricted, spendable = None, None, None
        if checks.get("balances", {}).get("status") == "success":
            row = next((b for b in snapshot.get("balances") or [] if b["coin"] == "USDT"), None)
            if row and row.get("restricted_available") is not None:
                try:
                    available, restricted = (
                        money(row["available"]),
                        money(row["restricted_available"]),
                    )
                    if not 0 <= restricted <= available:
                        raise ValueError("invalid_balance")
                    spendable = str(available - restricted)
                    available, restricted = str(available), str(restricted)
                except Exception:
                    available, restricted, spendable = None, None, None
        products = []
        for rule in rules:
            verified_at = self.redemption_verified(
                rule, account_id or (control["account_id"] if control else None)
            )
            product = next(
                (
                    p
                    for p in (snapshot.get("products") or {}).get("rows", [])
                    if p["coin"] == rule["coin"] and p["product_id"] == rule["product_id"]
                ),
                None,
            )
            held = None
            if checks.get("savings", {}).get("status") == "success":
                held = str(
                    sum(
                        (
                            money(p["quantity"])
                            for p in (snapshot.get("savings") or {}).get("rows", [])
                            if p["coin"] == rule["coin"] and p["product_id"] == rule["product_id"]
                        ),
                        money(0),
                    )
                )
            products.append(
                {
                    **rule,
                    "current_product": product,
                    "account_savings_quantity": held,
                    "note": "最小额赎回已通过。运行精度为保守限制。交易所最大赎回精度仍未验证。"
                    if verified_at
                    else notes.get(rule["coin"], "沿用已冻结的策略规则" if frozen else ""),
                    "redemption_api_verified": verified_at is not None,
                    "redemption_verified_at": verified_at,
                }
            )
        capital = frozen["capital"] if frozen else spendable
        status = (
            rules_error
            or (
                "account_binding_mismatch"
                if control and account_id and control["account_id"] != account_id
                else None
            )
            or (
                "finish_acceptance_before_strategy_allocation"
                if self.acceptance_pending()
                else None
            )
            or ("account_refresh_required" if stale else None)
            or ("classic_account_required" if account.get("account_type") != "classic" else None)
            or ("account_balance_unavailable" if spendable is None else None)
            or ("insufficient_available_usdt" if capital is None or money(capital) <= 0 else None)
            or (
                "selected_product_terms_unavailable"
                if any(
                    not p["current_product"]
                    or p["current_product"].get("terms_status") != "success"
                    or p["current_product"].get("status") != "in_progress"
                    for p in products
                )
                else None
            )
            or "ready"
        )
        if control and control.get("trading_authorized") is False:
            status = "adopted_trading_not_authorized"
        return {
            "status": status,
            "ready": status == "ready",
            "observed_at": observed_at,
            "stale": bool(stale),
            "capital": capital,
            "capital_source": "frozen_strategy" if frozen else "available_usdt_less_restricted",
            "available_usdt": available,
            "restricted_usdt": restricted,
            "rules_file": str(path),
            "products": products,
            "setup": {"capital": capital, "products": rules}
            if capital is not None and rules
            else None,
        }

    def prepare_account(self, account, account_id):
        with self.lock:
            draft = self.configuration(account, self.load())
            if not draft["ready"]:
                raise ExecutionError(draft["status"])
            if not account_id:
                raise ExecutionError("account_binding_mismatch")
            self.prepare(draft["setup"], expected_account_id=account_id)
            return self.status(account, account_id)

    def status(self, account=None, account_id=None):
        with self.lock:
            result = {
                "candidate": {
                    "label": LABEL,
                    "experiment_id": EXPERIMENT,
                    "run_id": RUN,
                    "snapshot": self.source(),
                },
                "prepared": None,
                "runtime": None,
                "real_trading_running": False,
                "rate_limit": RequestLimiter.status(self.root / "request-limits.sqlite"),
            }
            control = self.load()
            result["configuration"] = self.configuration(account, control, account_id)
            if control is None:
                return result
            runtime = self.get_runtime(control)
            state = runtime.status()
            execution = state.pop("execution")
            engine = state.pop("engine")
            book = Ledger.restore(execution["book"])
            price = (
                state.get("markets", {}).get("TRXUSDT", {}).get("bid", engine["last_price"] or 1)
            )
            if book.usdgo:
                book.usdgo_mark = money(state["markets"]["USDGOUSDT"]["bid"])
            state["summary"] = book.summary(price)
            observations = runtime.store.observations()
            state.update(observations)
            state["summary"].update(
                initial_capital=float(book.initial),
                added_capital=float(book.external_flows),
                trading_turnover=float(
                    sum(
                        (money(fill["quantity"]) * money(fill["price"]) for fill in book.orders),
                        money(0),
                    )
                ),
                max_drawdown=observations["history"]["max_drawdown"],
            )
            quotes = state.get("markets", {})
            quote_times = [int(q["timestamp"]) for q in quotes.values() if q.get("timestamp")]
            state["valuation"] = {
                "observed_at": min(quote_times) if quote_times else None,
                "stale": len(quote_times) < 2
                or any(not -5000 <= self.clock() - timestamp <= 60000 for timestamp in quote_times),
            }
            state["schedule"] = {
                k: engine[k]
                for k in (
                    "anchor",
                    "anchor_version",
                    "buy_period",
                    "sell_period",
                    "buy_at",
                    "sell_at",
                    "reason",
                    "last_close",
                )
            }
            from .range_display import display_config, range_view

            state["effective_config"] = display_config(state, engine, self.clock())
            state["active_range_policy"] = range_view(state, engine, self.clock())
            if control.get("allocation_mode") == "existing_positions_market_value_v1":
                state["performance"] = {
                    "status": "started" if state.get("performance_basis") else "not_started",
                    "started_at": (state.get("performance_basis") or {}).get("started_at"),
                    "basis": "actual_strategy_start_market_value",
                    "previous_adoption_is_performance_basis": False,
                }
            from freqtrade.strategy.trx_phase1.live_observation import dashboard

            state["dashboard"] = dashboard(state, engine, execution, book, self.clock())
            from freqtrade.strategy.trx_phase1.income_allocation import summary as income_summary
            from freqtrade.strategy.trx_phase1.live_reserve import summary as reserve_summary

            state["execution"] = {
                "usdgo_reserve": reserve_summary(execution),
                "flow": execution["flow"],
                "operation": execution["operation"],
                "holdings": execution["holdings"],
                "events": execution["events"][-30:],
                "orders": execution["book"]["orders"][-20:],
                "order_count": len(execution["book"]["orders"]),
                "lots": execution["book"]["lots"],
                "pending_income": execution.get("pending_income", {}),
                "income_accounting": income_summary(execution.get("income_accounting")),
                "asset_contributions": list(execution.get("asset_contributions", {}).values()),
                "asset_withdrawals": list(execution.get("asset_withdrawals", {}).values()),
                "withdrawal_enabled": True,
                "quarantined_account_credits": list(
                    execution.get("quarantined_account_credits", {}).values()
                )[-30:],
                "account_repairs": list(execution.get("account_repairs", {}).values()),
                "checked_at": execution.get("interest_checked_at"),
            }
            state["events"] = state["events"][-30:]
            state["intent"] = (
                {k: v for k, v in state["intent"].items() if k != "steps"}
                if state["intent"]
                else None
            )
            result.update(
                prepared={k: v for k, v in control.items() if k != "credentials_identity"},
                runtime=state,
                real_trading_running=state["worker_running"] and state["desired"] == "running",
            )
            return result

    def selected_contribution(self, review_hash=None, selection=None, status_only=False):
        """One fixed old-account batch, serialized with the running strategy."""
        from freqtrade.strategy.trx_phase1 import live_contribution
        from freqtrade.strategy.trx_phase1.selected_contribution import (
            CoreRejected,
            SelectedContribution,
            SelectedContributionError,
            fixed_policy,
        )

        with self.lock:
            control = self.load()
            if not control:
                raise ExecutionError("existing_live_account_required")
            runtime = self.get_runtime(control)
            with runtime.lock:
                if self.credential_identity() != control["credentials_identity"]:
                    raise ExecutionError("credentials_changed_requires_preflight")
                reader = self.broker_factory(False)
                try:
                    execution = LiveExecution(self.root, reader, control["account_id"])

                    class NativeWriter:
                        def confirm(writer, selected_hash, _authorization):
                            try:
                                return live_contribution.apply(
                                    execution, selected_hash, self.clock()
                                )
                            except ExecutionError as exc:
                                raise CoreRejected(str(exc)) from None

                    helper = SelectedContribution(
                        fixed_policy(execution.root), core=NativeWriter(), clock=self.clock
                    )
                    if status_only:
                        return helper.status()
                    if (
                        not runtime.state["startup_complete"]
                        or runtime.state["intent"]
                        or runtime.engine.pending
                        or runtime.state["error"]
                        or runtime.state["phase"] not in {"observing", "paused", "prepared"}
                    ):
                        raise ExecutionError("contribution_requires_idle_strategy")
                    helper.selection(selection)
                    result = (
                        helper.confirm(review_hash, None)
                        if review_hash
                        else helper.prepare(selection)
                    )
                    runtime.execution.reload()
                    if review_hash:
                        funding = LiveFunding(
                            runtime.execution.state,
                            runtime.engine.config,
                            runtime.state["markets"],
                            runtime.state["terms"],
                        )
                        runtime.sync_book(funding)
                        runtime.persist()
                    return result
                except SelectedContributionError as exc:
                    raise ExecutionError(str(exc)) from None
                finally:
                    runtime.execution.reload()
                    self.close_broker(reader)

    def contribution(self, review_hash=None):
        """Preview is read-only at the exchange; applying requires an explicit review hash."""
        from freqtrade.strategy.trx_phase1 import live_contribution

        with self.lock:
            control = self.load()
            if not control:
                raise ExecutionError("existing_live_account_required")
            runtime = self.get_runtime(control)
            with runtime.lock:
                from freqtrade.strategy.trx_phase1.live_withdrawal import active

                if active(runtime.execution.status()):
                    raise ExecutionError("withdrawal_already_pending")
                if (
                    not runtime.state["startup_complete"]
                    or runtime.state["intent"]
                    or runtime.engine.pending
                    or runtime.state["error"]
                    or runtime.state["phase"] not in {"observing", "paused", "prepared"}
                ):
                    raise ExecutionError("contribution_requires_idle_strategy")
                if self.credential_identity() != control["credentials_identity"]:
                    raise ExecutionError("credentials_changed_requires_preflight")
                reader = self.broker_factory(False)
                try:
                    execution = LiveExecution(self.root, reader, control["account_id"])
                    now = self.clock()
                    result = (
                        live_contribution.apply(execution, review_hash, now)
                        if review_hash
                        else live_contribution.preview(execution, now)
                    )
                    runtime.execution.reload()
                    if review_hash:
                        # The execution journal is authoritative after a crash here;
                        # the next worker tick reloads it without applying twice.
                        funding = LiveFunding(
                            runtime.execution.state,
                            runtime.engine.config,
                            runtime.state["markets"],
                            runtime.state["terms"],
                        )
                        runtime.sync_book(funding)
                        runtime.persist()
                    return result
                finally:
                    runtime.execution.reload()
                    self.close_broker(reader)

    def reserve_half(self, account_id, allocation_id):
        """Explicit one-time ownership change; the broker is read-only."""
        from freqtrade.strategy.trx_phase1.live_reserve import allocate_half

        with self.lock:
            control = self.load()
            if not control or control["account_id"] != account_id:
                raise ExecutionError("account_binding_mismatch")
            runtime = self.get_runtime(control)
            with runtime.lock:
                if self.credential_identity() != control["credentials_identity"]:
                    raise ExecutionError("credentials_changed_requires_preflight")
                saved = runtime.execution.status().get("usdgo_reserve")
                if saved:
                    if saved["allocation_id"] != allocation_id:
                        raise ExecutionError("reserve_already_allocated")
                    return deepcopy(saved)
                if not runtime.thread or not runtime.thread.is_alive():
                    raise ExecutionError("reserve_requires_running_worker")
                reader = self.broker_factory(False)
                try:
                    execution = LiveExecution(self.root, reader, account_id)
                    state = {**runtime.state, "engine": runtime.engine.state()}
                    result = allocate_half(execution, state, allocation_id, self.clock())
                    runtime.execution.reload()
                    runtime.sync_book(
                        LiveFunding(
                            runtime.execution.state,
                            runtime.engine.config,
                            runtime.state["markets"],
                            runtime.state["terms"],
                        )
                    )
                    runtime.persist()
                    return result
                finally:
                    runtime.execution.reload()
                    self.close_broker(reader)

    def withdrawal(self, amount=None, review_hash=None):
        """Only a human confirmation queues a durable withdrawal for the current worker."""
        from freqtrade.strategy.trx_phase1 import live_withdrawal

        with self.lock:
            control = self.load()
            if not control:
                raise ExecutionError("existing_live_account_required")
            runtime = self.get_runtime(control)
            with runtime.lock:
                if self.credential_identity() != control["credentials_identity"]:
                    raise ExecutionError("credentials_changed_requires_preflight")
                # A repeated confirmation returns its existing receipt without new work.
                if review_hash:
                    saved = runtime.execution.status().get("asset_withdrawals", {}).get(review_hash)
                    if saved:
                        return deepcopy(saved)
                    if (
                        runtime.state["desired"] != "running"
                        or not runtime.thread
                        or not runtime.thread.is_alive()
                    ):
                        raise ExecutionError("withdrawal_requires_running_worker")
                try:
                    return (
                        live_withdrawal.confirm(runtime, review_hash, self.clock())
                        if review_hash
                        else live_withdrawal.review(runtime, amount, self.clock())
                    )
                finally:
                    runtime.execution.reload()

    def products(self):
        broker = self.broker_factory(False)
        result = {}
        try:
            for coin in ("TRX", "USDGO"):
                rows = broker.get("v2/earn/savings/product", {"coin": coin, "filter": "available"})
                result[coin] = [
                    {
                        "product_id": str(r["productId"]),
                        "coin": coin,
                        "status": r["status"],
                        "level": r.get("productLevel", "unknown"),
                        "tiers": r.get("apyList", []),
                    }
                    for r in rows
                    if r.get("coin") == coin and r.get("periodType") == "flexible"
                ]
            return result
        finally:
            self.close_broker(broker)

    @staticmethod
    def close_broker(broker):
        if hasattr(broker, "transport"):
            broker.transport.session.close()

    def confirmed_minutes(self, market, data):
        # Keep the same public client across confirmation reads. A fresh client
        # cannot confirm recent bars on its first observation.
        for attempt in range(3):
            observed, quote = market.snapshot()
            now = self.clock()
            validate_quote(quote, now)
            if abs(observed - now) > 30000:
                raise ExecutionError("market_clock_mismatch")
            data = market.update(data, now)
            now = self.clock()
            age = now - int(pd_timestamp(data.date.iloc[-1])) - 60000
            if 0 <= age <= 120000:
                return data, now
            if attempt < 2:
                time.sleep(11)
        raise ExecutionError("finalized_market_data_stale")

    def check_existing_only(self):
        if (self.load() or {}).get("trading_authorized") is False:
            raise ExecutionError("adopted_trading_not_authorized")
        if self.existing_only:
            from freqtrade.strategy.trx_phase1.live_restore import existing_state

            existing_state(self)

    def awaiting_existing_start(self, control):
        return bool(
            control
            and control.get("allocation_mode") == "existing_positions_market_value_v1"
            and not self.get_runtime(control).state.get("activated_at")
        )

    def prepare(self, setup, *, expected_account_id=None):
        with self.lock:
            control = self.load()
            self.recover_startup_control(control)
            control = self.load()
            if self.awaiting_existing_start(control):
                with allocation_lock(self.root):
                    return self.prepare_existing_start(setup, expected_account_id)
            return self.prepare_allocated(setup, expected_account_id=expected_account_id)

    def recover_startup_control(self, control):
        """Finish an already committed start after a control-marker crash."""
        if control and control.get("allocation_mode") == "existing_positions_market_value_v1":
            basis = self.get_runtime(control).execution.status().get("performance_basis")
            if basis and control.get("performance_basis_sha256") != basis["sha256"]:
                from freqtrade.strategy.trx_phase1.startup_basis import commit

                with allocation_lock(self.root):
                    commit(self, None)

    def prepare_allocated(self, setup, *, expected_account_id=None):
        with self.lock, allocation_lock(self.root):
            self.check_existing_only()
            if self.acceptance_pending():
                raise ExecutionError("finish_acceptance_before_strategy_allocation")
            if self.runtime and self.runtime.thread and self.runtime.thread.is_alive():
                raise ExecutionError("pause_strategy_before_preflight")
            setup = LiveSetup.model_validate(setup)
            snapshot, now = self.source(), self.clock()
            identity = self.credential_identity()
            broker, market = self.broker_factory(False), self.market_factory()
            try:
                account_id = broker.account_id()
                if expected_account_id is not None and account_id != expected_account_id:
                    raise ExecutionError("account_binding_mismatch")
                broker.require_bgb_deduction_off()
                control = self.load()
                products = [p.model_dump(mode="json") for p in setup.products]
                if control and (
                    control["account_id"] != account_id
                    or control["setup"] != setup.model_dump(mode="json")
                    or control["source_sha256"] != digest(snapshot)
                ):
                    raise ExecutionError("frozen_live_setup_mismatch")
                data = (
                    seed_history(self.seed, now, snapshot["price_position_lookback_days"])
                    if not control
                    else self.get_runtime(control).store.candles()
                )
                data, now = self.confirmed_minutes(market, data)
                path = self.root / "live_execution" / account_id / "execution.sqlite"
                if path.exists():
                    execution = LiveExecution(self.root, broker, account_id)
                    if execution.state is not None and (
                        money(execution.state["book"]["initial"]) != money(setup.capital)
                        or execution.state["products"] != products
                    ):
                        raise ExecutionError("frozen_live_setup_mismatch")
                else:
                    execution = None
                if execution is None or execution.state is None:
                    # Validate product terms, native APR and history BEFORE allocation.
                    scratch = {
                        "book": Ledger(setup.capital).state(),
                        "products": products,
                        "holdings": {},
                    }
                    markets = {s: broker.market(s, now) for s in ("TRXUSDT", "USDGOUSDT")}
                    terms = {p.coin: broker.product(p.coin, p.product_id) for p in setup.products}
                    funding = LiveFunding(
                        scratch, ResearchConfig.from_saved(snapshot), markets, terms
                    )
                    self.check_startup(funding)
                    from freqtrade.strategy.trx_phase1.price_position import build_price_reference

                    if (
                        build_price_reference(data, ResearchConfig.from_saved(snapshot)).value_at(
                            now
                        )
                        is None
                    ):
                        raise ExecutionError("complete_p75_history_required")
                    execution = LiveExecution.create(
                        self.root, broker, setup.capital, products, now
                    )
                runtime = self.get_runtime(control) if control else LiveStrategy(execution, market)
                if runtime.state is None:
                    runtime = LiveStrategy.create(execution, market, snapshot, data, now)
                runtime.execution.broker, runtime.market = broker, market
                now = self.clock()
                runtime.execution.import_interest(now)
                runtime.execution.reload()
                runtime.sync_savings_observation()
                funding = runtime.environment(now)
                self.review_remaining_startup(runtime, funding, now)
                if not runtime.state["startup_complete"] and not runtime.state["activated_at"]:
                    self.check_startup(funding)
                runtime.data = data
                runtime.store.add_candles(data)
                runtime.reference_at(now)
                runtime.sync_book(funding)
                runtime.initialize_indicators(now)
                flow = runtime.execution.state["flow"]
                if not runtime.execution.state["operation"] and (
                    not flow or flow["status"] in {"completed", "expired"}
                ):
                    runtime.state.update(phase="prepared", desired="paused", error=None)
                runtime.persist()
                review = terms_review(snapshot, setup, funding)
                self.runtime = runtime
                self.save(
                    {
                        **(control or {}),
                        "account_id": account_id,
                        "source_sha256": digest(snapshot),
                        "setup": setup.model_dump(mode="json"),
                        "review_hash": review,
                        "reviewed_at": now,
                        "review_expires_at": now + 5 * 60000,
                        "credentials_identity": identity,
                        **{
                            key: control[key]
                            for key in ("range_policy_sha256", "range_policy_activated_at")
                            if control and key in control
                        },
                    }
                )
                return self.status()
            finally:
                self.close_broker(broker)
                market.close()

    def capture_startup(self):
        from freqtrade.strategy.trx_phase1.account import AccountService
        from freqtrade.strategy.trx_phase1.startup_basis import capture

        return capture(AccountService(self.root.parent), self)

    def prepare_existing_start(self, setup, expected_account_id=None):
        """Fresh read-only preflight for an explicitly authorized adopted account.

        Price and capital remain provisional here. The actual start captures them
        again through this preflight and commits only after its review matches.
        """
        from freqtrade.strategy.trx_phase1.startup_basis import POLICY, documents

        control = self.load()
        if control.get("trading_authorized") is not True:
            raise ExecutionError("adopted_trading_not_authorized")
        if self.credential_identity() != control["credentials_identity"]:
            raise ExecutionError("credentials_changed_requires_preflight")
        if expected_account_id and expected_account_id != control["account_id"]:
            raise ExecutionError("account_binding_mismatch")
        setup = LiveSetup.model_validate(setup)
        if setup.model_dump(mode="json")["products"] != control["setup"]["products"]:
            raise ExecutionError("frozen_live_setup_mismatch")
        runtime = self.get_runtime(control)
        if runtime.thread and runtime.thread.is_alive():
            raise ExecutionError("pause_strategy_before_preflight")
        reader, market = self.broker_factory(False), self.market_factory()
        try:
            if reader.account_id() != control["account_id"]:
                raise ExecutionError("account_binding_mismatch")
            reader.require_bgb_deduction_off()
            data = runtime.store.candles()
            if data is None:
                data = seed_history(
                    self.seed, self.clock(), self.source()["price_position_lookback_days"]
                )
            data, now = self.confirmed_minutes(market, data)
            if (
                build_price_reference(data, ResearchConfig.from_saved(self.source())).value_at(now)
                is None
            ):
                raise ExecutionError("complete_p75_history_required")
            markets = {s: reader.market(s, now) for s in ("TRXUSDT", "USDGOUSDT")}
            terms = {p.coin: reader.product(p.coin, p.product_id) for p in setup.products}
            # Final exchange reads are the startup quantities and prices.
            captured = self.capture_startup()
            scratch, _ = documents(
                captured, control, runtime.execution.status(), runtime.status(), self.source()
            )
            funding = LiveFunding(scratch, ResearchConfig.from_saved(self.source()), markets, terms)
            for coin, term in terms.items():
                if term["status"] != "in_progress" or not self.redemption_verified(
                    funding.rules[coin].model_dump(mode="json"), control["account_id"]
                ):
                    raise ExecutionError("redemption_terms_unverified")
            # Review the account/product scope and terms. Both quantities and
            # prices are captured again at start, never frozen by a preview.
            review = digest(
                {
                    "terms": terms_review(
                        self.source(), setup.model_copy(update={"capital": "1"}), funding
                    ),
                    "scope": captured["scope_hash"],
                    "policy": POLICY,
                }
            )
            now = self.clock()
            self._startup_capture, self._startup_data = captured, data
            self.save(
                {
                    **control,
                    "review_hash": review,
                    "reviewed_at": now,
                    "review_expires_at": now + 5 * 60000,
                    "startup_scope_hash": captured["scope_hash"],
                }
            )
            return self.status()
        finally:
            self.close_broker(reader)
            market.close()

    def start_existing_positions(self, review_hash):
        from freqtrade.strategy.trx_phase1.startup_basis import commit

        control = self.load()
        self.prepare(control["setup"])
        if self.load()["review_hash"] != review_hash:
            raise ExecutionError("live_terms_changed_review_required")
        writer = self.broker_factory(True)
        market = None
        try:
            if writer.account_id() != control["account_id"]:
                raise ExecutionError("account_binding_mismatch")
            writer.require_bgb_deduction_off()
            with allocation_lock(self.root):
                basis = commit(self, self._startup_capture, review_hash=review_hash)
            runtime = self.get_runtime(self.load())
            market = self.market_factory()
            runtime.execution.broker, runtime.market = writer, market
            runtime.data = self._startup_data
            runtime.store.add_candles(runtime.data)
            runtime.initialize_indicators(basis["started_at"])
            runtime.credential_guard = lambda: (
                self.credential_identity() == control["credentials_identity"]
            )
            runtime.start()
            self.save({**self.load(), "review_expires_at": 0})
            return self.status()
        except BaseException:
            self.close_broker(writer)
            if market is not None:
                market.close()
            raise

    @staticmethod
    def review_remaining_startup(runtime, funding, now):
        state = runtime.execution.state
        intent = runtime.state["intent"]
        flow = state["flow"]
        if (
            not intent
            or intent["kind"] != "startup"
            or not flow
            or flow["status"] != "expired"
            or state["operation"]
        ):
            return
        if flow["id"] != intent["flow_id"]:
            raise ExecutionError("strategy_execution_binding_mismatch")
        remaining = deepcopy(flow["steps"][flow["index"] :])
        if not remaining:
            raise ExecutionError("startup_state_requires_review")
        if remaining[0]["amount"] == "previous":
            remaining[0]["amount"] = flow["outputs"][-1]["amount"]
        for index, step in enumerate(remaining):
            if step["kind"] == "trade":
                remaining[index] = funding.trade(step["symbol"], step["side"], step["amount"])
        runtime.state.update(
            intent=None, startup_resume=remaining, phase="prepared", desired="paused", error=None
        )
        runtime.event(
            "startup_remaining_steps_reviewed",
            now,
            flow_id=flow["id"],
            completed_steps=flow["index"],
        )

    @staticmethod
    def check_startup(funding):
        for coin, share in (
            ("TRX", money(funding.config.initial_buy_fraction)),
            ("USDGO", 1 - money(funding.config.initial_buy_fraction)),
        ):
            m = funding.markets[coin + "USDT"]
            budget = funding.book.initial * share
            from freqtrade.strategy.trx_phase1.live_broker import rounded

            quantity = rounded(
                budget / (money(m["ask"]) * (1 + money(m["fee"]))),
                min(int(m["quantity_decimals"]), int(funding.terms[coin]["subscribePrecision"])),
            )
            if (
                budget / (1 + money(m["fee"])) < money(m["minimum_notional"])
                or funding.eligible_subscription(coin, quantity) < quantity
            ):
                raise ExecutionError("startup_product_quota_or_minimum_unsatisfied")
            if quantity < funding.rules[coin].redemption_minimum:
                raise ExecutionError("startup_below_redemption_minimum")
            funding.apr_income(coin, quantity)
            if funding.terms[coin]["status"] != "in_progress":
                raise ExecutionError("product_unavailable")

    def start(self, review_hash):
        with self.lock:
            control = self.load()
            if control and control.get("trading_authorized") is False:
                raise ExecutionError("adopted_trading_not_authorized")
            if (
                not control
                or not secrets.compare_digest(review_hash, control["review_hash"])
                or self.clock() >= control["review_expires_at"]
            ):
                raise ExecutionError("fresh_preflight_required")
            if self.awaiting_existing_start(control):
                return self.start_existing_positions(review_hash)
            refreshed = self.prepare(control["setup"])
            current = self.load()
            if current["review_hash"] != review_hash:
                raise ExecutionError("live_terms_changed_review_required")
            broker = self.broker_factory(True)
            try:
                if broker.account_id() != current["account_id"]:
                    raise ExecutionError("account_binding_mismatch")
                broker.require_bgb_deduction_off()
            except BaseException:
                self.close_broker(broker)
                raise
            runtime = self.runtime
            runtime.execution.broker, runtime.market = broker, self.market_factory()
            expected = current["credentials_identity"]
            runtime.credential_guard = lambda: self.credential_identity() == expected
            try:
                runtime.start()
            except Exception:
                self.close_broker(broker)
                runtime.market.close()
                raise
            self.save({**current, "review_expires_at": 0})
            return {**refreshed, **self.status()}

    def pause(self):
        with self.lock:
            control = self.load()
            if control:
                runtime = self.get_runtime(control)
                runtime.pause()
                runtime.execution.request_stop(self.clock())
                self.save({**control, "review_expires_at": 0})
            return self.status()

    def reconcile(self, operation_id=None, exchange_id=None):
        with self.lock:
            control = self.load()
            if not control:
                raise ExecutionError("strategy_not_prepared")
            if self.awaiting_existing_start(control):
                raise ExecutionError("actual_start_required_before_reconciliation")
            runtime = self.get_runtime(control)
            if runtime.thread and runtime.thread.is_alive():
                raise ExecutionError("pause_strategy_before_reconciliation")
            broker = self.broker_factory(False)
            runtime.execution.broker = broker
            try:
                if broker.account_id() != control["account_id"]:
                    raise ExecutionError("account_binding_mismatch")
                if operation_id or exchange_id:
                    runtime.execution.attach_exchange_order(
                        operation_id, exchange_id, now=self.clock()
                    )
                runtime.resolve_execution(self.clock())
                if (
                    runtime.state["startup_complete"]
                    and not runtime.execution.status()["operation"]
                ):
                    policy = self.root / "account-repair-policy.json"
                    if policy.exists():
                        from freqtrade.strategy.trx_phase1.account_repair import repair

                        repair(runtime.execution, json.loads(policy.read_text()), self.clock())
                    runtime.execution.import_interest(self.clock())
                    runtime.execution.reload()
                    runtime.sync_savings_observation()
                    funding = runtime.environment(self.clock())
                    runtime.sync_book(funding)
                    runtime.capture_observation(self.clock())
                if runtime.state["phase"] == "blocked" and not runtime.state["intent"]:
                    runtime.state.update(phase="paused", error=None, evidence_wait_started_at=None,
                                         evidence_wait_last_seen_at=None)
                runtime.persist()
                return self.status()
            finally:
                self.close_broker(broker)

    def shutdown(self):
        runtime = self.runtime
        if runtime and runtime.thread and runtime.thread.is_alive():
            runtime.stop_event.set()
            # Revoke new POSTs immediately, without touching an in-flight receipt.
            if hasattr(runtime.execution.broker, "transport"):
                runtime.execution.broker.transport.writes_enabled = False
            runtime.thread.join(timeout=2)


def pd_timestamp(value):
    return value.timestamp() * 1000
