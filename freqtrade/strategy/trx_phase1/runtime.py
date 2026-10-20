import threading
from pathlib import Path

import pandas as pd

from freqtrade.strategy.trx_phase1.candles import normalized_volatility
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import StateStore
from freqtrade.strategy.trx_phase1.research_earn import HistoricalEarn


class PaperRuntime:
    """Persistent paper portfolio on public live data; never sends exchange orders."""

    def __init__(self, user_dir: Path, data_path: Path, exchange, config: ResearchConfig):
        if config.sell_mode != "proportional":
            raise ValueError("Recent-lot sales are currently a historical replay mode only")
        if config.price_position_enabled:
            raise ValueError("Price-position sizing is currently a historical replay mode only")
        if config.startup in ("half_position", "allocated_position"):
            raise ValueError("Half-position startup is currently a historical replay mode only")
        self.data_path, self.exchange = data_path, exchange
        self.store = StateStore(user_dir / "trx_phase1/paper.sqlite")
        self.version, saved = self.store.load("paper")
        self.engine = Engine.restore(saved) if saved else Engine(config)
        if self.engine.config.model_dump() != config.model_dump():
            raise ValueError(
                "Paper configuration differs from saved state; restore its settings first"
            )
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.thread = None
        self.error = None
        self.last_poll = None

    @property
    def running(self):
        return bool(self.thread and self.thread.is_alive() and not self.stop_event.is_set())

    def start(self):
        if self.engine.manual_preparation:
            raise ValueError("Complete or cancel the prepared manual exit before resuming")
        if self.running:
            raise ValueError("Paper runtime already running")
        self.stop_event.clear()
        self.engine.paused = False
        self.thread = threading.Thread(target=self.run, name="trx-paper", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=45)
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Paper worker is still completing a request; stop remains pending")
        with self.lock:
            self.engine.finish(self.engine.last_close)
            self.engine.paused, self.engine.reason = True, "paused"
            self.save()

    def save(self):
        self.version = self.store.save("paper", self.version, self.engine.state())

    def poll(self):
        existing = pd.read_feather(self.data_path)
        last = int(pd.Timestamp(existing["date"].iloc[-1]).timestamp() * 1000)
        now = pd.Timestamp.now(tz="UTC")
        if now.timestamp() * 1000 - last > 35 * 86400000:
            raise ValueError("Local data is too old; download historical gaps before resuming")
        merged = self.exchange.update(existing, int(now.timestamp() * 1000))
        if len(merged) > len(existing):
            temporary = self.data_path.with_name(self.data_path.name + ".paper.tmp")
            merged.to_feather(temporary, compression="lz4", compression_level=9)
            temporary.replace(self.data_path)
        sigma = normalized_volatility(merged, self.engine.config.volatility_window)
        timestamps = pd.DatetimeIndex(merged["date"]).as_unit("ms").asi8
        with self.lock:
            was_paused = self.engine.paused
            if self.engine.earn is None:
                # Start prospective subscriptions now; old paper states do not
                # acquire hypothetical income for periods before activation.
                start_at = (
                    int(now.timestamp() * 1000) if self.engine.last_close else int(timestamps[-1])
                )
                self.engine.earn = HistoricalEarn(self.engine.book, self.engine.config, start_at)
                self.engine.prepare_cash = self.engine.earn.prepare_cash
                self.engine.earn.sweep_cash(start_at)
            latest_close = int(timestamps[-1] + 60000)
            if not self.engine.last_close:
                self.engine.last_close = latest_close - 60000
            recovering = latest_close - self.engine.last_close > 120000
            if recovering:
                self.engine.finish(self.engine.last_close)
                self.engine.paused = True
            for index in range(
                max(0, int(timestamps.searchsorted(self.engine.last_close))), len(merged)
            ):
                timestamp = int(timestamps[index])
                row = merged.iloc[index]
                if timestamp + 60000 <= self.engine.last_close:
                    continue
                self.engine.earn.advance(timestamp, self.engine.last_price or float(row["open"]))
                if (
                    self.engine.pending
                    and timestamp >= self.engine.pending.ready_at
                    and not recovering
                    and not self.stop_event.is_set()
                ):
                    plan = self.engine.pending
                    price = float(row["open"]) * (
                        1 + self.engine.config.slippage
                        if plan.side == "buy"
                        else 1 - self.engine.config.slippage
                    )
                    if self.engine.executable(timestamp, price):
                        fee_rate = (
                            self.engine.config.buy_fee
                            if plan.side == "buy"
                            else self.engine.config.sell_fee
                        )
                        quantity = self.engine.execution_quantity(price)
                        self.engine.fill(
                            plan.plan_id,
                            plan.plan_id,
                            timestamp,
                            quantity,
                            price,
                            quantity * price * fee_rate,
                        )
                    else:
                        self.engine.finish(timestamp)
                self.engine.earn.advance(timestamp + 60000, float(row["close"]))
                plan = self.engine.on_close(
                    timestamp + 60000, float(row["close"]), float(sigma.iloc[index])
                )
                self.engine.earn.allocate_cash(timestamp + 60000, plan)
            if recovering:
                self.engine.paused = False
                self.engine.schedule(latest_close, self.engine.last_price, self.engine.last_sigma)
                self.engine.reason = "recovered_data_waiting_new_signal"
            if was_paused and self.stop_event.is_set():
                self.engine.paused, self.engine.reason = True, "paused"
            self.last_poll, self.error = now.isoformat(), None
            self.save()

    def configure(self, config):
        if config.sell_mode != "proportional":
            raise ValueError("Recent-lot sales are currently a historical replay mode only")
        if config.price_position_enabled:
            raise ValueError("Price-position sizing is currently a historical replay mode only")
        if self.engine.manual_preparation:
            raise ValueError(
                "Complete or cancel the prepared manual exit before changing parameters"
            )
        if self.running:
            raise ValueError("Pause paper execution before changing its parameters")
        with self.lock:
            if config.initial_cash != self.engine.config.initial_cash:
                raise ValueError("An existing paper portfolio cannot change its initial capital")
            changed = (config.earn_model, config.trx_apr, config.usdgo_apr, config.usdt_apr) != (
                self.engine.config.earn_model,
                self.engine.config.trx_apr,
                self.engine.config.usdgo_apr,
                self.engine.config.usdt_apr,
            )
            if (
                changed
                and self.engine.earn is not None
                and (self.engine.book.usdgo or self.engine.book.interest_trx)
            ):
                raise ValueError(
                    "Preserve existing Earn terms while the paper portfolio holds Earn assets"
                )
            self.engine.finish(self.engine.last_close)
            self.engine.config = config
            self.engine.tier = self.engine.relax_count = 0
            self.engine.buy_at = self.engine.sell_at = 0
            self.engine.buy_period = self.engine.sell_period = config.periods[0]
            if changed:
                self.engine.earn, self.engine.prepare_cash = None, None
            elif self.engine.earn is not None:
                self.engine.earn.config = config
            self.engine.paused, self.engine.reason = True, "parameters_updated_waiting_new_signal"
            self.save()

    def set_additions(self, enabled):
        with self.lock:
            self.engine.accept_new_buys = enabled
            if not enabled and self.engine.pending and self.engine.pending.side == "buy":
                self.engine.finish(self.engine.last_close)
            self.save()

    def refresh_paused(self):
        if self.running:
            raise ValueError("Pause paper execution before refreshing it manually")
        self.stop_event.set()
        self.engine.paused = True
        self.poll()
        self.engine.paused, self.engine.reason = True, "paused"
        self.save()

    def manual_exit(self, event_id, source, quantity, accept_loss):
        if self.running:
            raise ValueError("Pause paper execution before a manual exit")
        with self.lock:
            now = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
            if not self.engine.last_price or now - self.engine.last_close > 120000:
                raise ValueError("Refresh the paused paper market data before a manual exit")
            self.engine.manual_exit(
                event_id,
                now,
                source,
                quantity,
                self.engine.last_price * (1 - self.engine.config.slippage),
                accept_loss,
            )
            self.engine.reason = "manual_exit_recorded"
            self.save()

    def prepare_exit(self, event_id, source, quantity):
        if self.running:
            raise ValueError("Pause paper execution before preparing a manual exit")
        with self.lock:
            self.engine.prepare_manual_exit(
                event_id, int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), source, quantity
            )
            self.save()

    def cancel_exit(self):
        with self.lock:
            self.engine.cancel_manual_exit(int(pd.Timestamp.now(tz="UTC").timestamp() * 1000))
            self.save()

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.poll()
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.engine.paused, self.engine.reason = True, "data_or_accounting_error"
                    self.save()
                self.stop_event.set()
            self.stop_event.wait(10)

    def status(self):
        with self.lock:
            return {
                "mode": "paper_only",
                "running": self.running,
                "error": self.error,
                "last_poll": self.last_poll,
                "config": self.engine.config.model_dump(),
                "summary": self.engine.status(),
                "orders": self.engine.book.order_summary(),
                "lots": [lot.state() for lot in self.engine.book.lots],
                "manual_preparation": self.engine.manual_preparation,
            }
