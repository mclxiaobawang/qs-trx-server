import json
import sqlite3
import threading
from copy import deepcopy

import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.shadow import ShadowRuntime, ShadowStore, create_shadow, digest
from freqtrade.strategy.trx_phase1.shadow_execution import (
    ShadowExecution,
    SimulatedBroker,
    check_execution_faults,
)
from freqtrade.strategy.trx_phase1.shadow_market import (
    PublicBitget,
    PublicShadowMarket,
    validate_quote,
)


NOW = int(pd.Timestamp("2026-01-01T12:03:20Z").timestamp() * 1000)
RULES = {"quantity_decimals": 4, "min_quantity": 3, "min_notional": 1, "observed_at": NOW}


def config(**updates):
    return ResearchConfig(
        version="trx-phase1-v0.8",
        startup="allocated_position",
        initial_cash=10000,
        initial_buy_fraction=0.1,
        buy_threshold=0.014,
        sell_threshold=0.012,
        buy_fraction=0.12,
        sell_fraction=0.12,
        variant="dynamic_period_only",
        periods=[60, 240, 1440],
        price_position_enabled=True,
        price_position_lookback_days=1,
        price_position_quantile=0.75,
        price_position_buy_multiplier=0.25,
        price_position_sell_multiplier=1.75,
        volatility_window=20,
        earn_model="net_benefit_hourly",
        trx_apr=0.1033,
        usdgo_apr=0.1,
        trx_earn_cap=10000000,
        usdgo_earn_cap=100000,
        slippage=0,
        redemption_delay_minutes=1,
        **updates,
    )


class Market:
    def __init__(self):
        self.now = NOW
        self.price = 0.3

    def snapshot(self):
        return self.now, {
            "timestamp": self.now,
            "asks": [[self.price, 100000]],
            "bids": [[self.price - 0.0001, 100000]],
        }

    def rules(self):
        return {**RULES, "observed_at": self.now}

    def update(self, existing, now):
        dates = pd.date_range(
            existing.date.iloc[-1] + pd.Timedelta("1min"),
            pd.Timestamp(now // 60000 * 60000 - 60000, unit="ms", tz="UTC"),
            freq="1min",
        )
        if not len(dates):
            return existing
        extra = pd.DataFrame(
            {
                "date": dates,
                "open": self.price,
                "high": self.price,
                "low": self.price,
                "close": self.price,
                "volume": 1000.0,
            }
        )
        return pd.concat([existing, extra], ignore_index=True)


def make_runtime(tmp_path):
    seed = tmp_path / "history.feather"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-12-29", periods=3 * 1440, freq="1min", tz="UTC"),
            "open": 0.3,
            "high": 0.3,
            "low": 0.3,
            "close": 0.3,
            "volume": 1000.0,
        }
    ).to_feather(seed)
    source = tmp_path / "source.json"
    saved = {
        "run_id": "frozen-run",
        "experiment_id": "frozen-experiment",
        "config": config().model_dump(),
        "start_utc": "2025-01-01",
        "end_utc": "2026-01-01",
        "provenance": {"exchange": "bitget", "pair": "TRX/USDT"},
    }
    source.write_text(json.dumps(saved))
    identifier = create_shadow(tmp_path / "shadows", source, seed)
    return ShadowRuntime(tmp_path / "shadows" / identifier / "shadow.sqlite", Market()), source


def prepared(tmp_path, fault="none"):
    cfg = config()
    engine = Engine(cfg)
    engine.last_price = 0.3
    engine.make_plan("buy", NOW, 0.3, 0, NOW + 60000, startup=True, reference=0.3)
    broker = SimulatedBroker(tmp_path / "broker.sqlite", fault)
    driver = ShadowExecution(broker, "test")
    driver.prepare(engine, Market().snapshot()[1], NOW, RULES)
    return engine, driver


def test_frozen_source_and_independent_prospective_startup(tmp_path):
    runtime, source = make_runtime(tmp_path)
    source.write_text("{}")  # Running parameters survive changes to the source/form.
    runtime.poll()
    state = runtime.status()
    assert state["source"]["snapshot_sha256"] == digest(config().model_dump())
    assert state["source"]["parameters"] == config().model_dump()
    assert state["summary"]["usdgo"] == 9000
    assert state["summary"]["principal_trx"] == pytest.approx(3330.0033)
    assert len(state["orders"]) == 1
    assert state["orders"][0]["timestamp"] == NOW
    assert state["orders"][0]["price"] == 0.3
    assert state["activated_at"] == NOW and state["phase"] == "paused"  # No worker in this test.
    assert runtime.state["phase"] == "warming_up"
    assert state["summary"]["interest_trx"] == 0 and state["summary"]["stable_interest"] == 0
    assert not state["real_trading_enabled"] and not state["real_earn_enabled"]
    runtime.poll()
    assert len(runtime.engine.book.orders) == 1
    assert not (tmp_path / "paper.sqlite").exists()


def test_gap_accrues_frozen_native_interest_without_retroactive_trades(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    runtime.poll()
    principal = runtime.engine.book.principal
    restarted = ShadowRuntime(runtime.store.path, Market())
    restarted.market.now = NOW + 4 * 3600000
    restarted.market.price = 0.4  # A missed rise cannot create a retrospective sell.
    restarted.poll()
    assert restarted.engine.book.principal == principal
    assert len(restarted.engine.book.orders) == 1
    assert restarted.engine.book.interest_trx > 0
    assert restarted.engine.book.stable_interest > 0
    assert restarted.engine.sell_at > restarted.market.now
    assert restarted.engine.last_close <= restarted.market.now
    assert restarted.engine.price_position_reference == 0.3  # Prior complete day, not today's rise.
    restored = Engine.restore(restarted.store.load("shadow")[1]["engine"])
    assert restored.book.state() == restarted.engine.book.state()
    assert restored.earn.state() == restarted.engine.earn.state()


def test_new_confirmed_sell_after_warmup_uses_current_bid_and_keeps_interest(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    runtime.poll()
    runtime.market.now = NOW + 2 * 3600000
    runtime.poll()  # Offline warmup, no stale signals.
    due = runtime.engine.next_close(runtime.market.now, 60)
    runtime.engine.buy_at = runtime.engine.sell_at = due
    runtime.market.now = due - 40000
    runtime.poll()  # Reconnected before the next confirmation; observe its final minute.
    runtime.market.now = due + 20000
    runtime.market.price = 0.31
    runtime.poll()
    assert runtime.engine.pending and runtime.engine.pending.side == "sell"
    runtime.market.now = due + 80000
    runtime.market.price = 0.3105
    runtime.poll()  # The saved one-minute TRX redemption is honored, too.
    assert runtime.engine.book.orders[-1]["side"] == "sell"
    assert runtime.engine.book.orders[-1]["price"] == pytest.approx(0.3104)
    assert runtime.engine.book.orders[-1]["timestamp"] == due + 80000
    assert runtime.engine.anchor == pytest.approx(0.3104)
    assert all(part["profit"] > 0 for part in runtime.engine.book.orders[-1]["allocations"])


def test_funding_waits_for_virtual_redemption_before_buy(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    runtime.poll()
    runtime.market.now = NOW + 2 * 3600000
    runtime.poll()
    due = runtime.engine.next_close(runtime.market.now, 60)
    runtime.engine.buy_at = runtime.engine.sell_at = due
    runtime.market.now = due - 40000
    runtime.poll()
    runtime.market.now, runtime.market.price = due + 20000, 0.29
    runtime.poll()
    assert len(runtime.engine.book.orders) == 1
    assert runtime.engine.pending and runtime.engine.pending.ready_at > runtime.market.now
    runtime.market.now = due + 80000
    runtime.poll()
    assert len(runtime.engine.book.orders) == 2
    assert runtime.engine.book.orders[-1]["side"] == "buy"
    assert runtime.engine.book.orders[-1]["timestamp"] == due + 80000


def test_fault_suite_restores_every_case_without_duplicate_balance(tmp_path):
    report = check_execution_faults(tmp_path / "checks", config())
    assert report["passed"] and len(report["checks"]) == 7
    assert json.loads((tmp_path / "checks/fault-check.json").read_text()) == report


def test_receipt_batch_is_atomic_and_duplicates_are_idempotent(tmp_path):
    engine, driver = prepared(tmp_path, "partial")
    broker_order = driver.broker.submit(deepcopy(driver.outbox))
    engine = driver.ingest(engine, broker_order, NOW)
    before, receipts = deepcopy(engine.state()), deepcopy(driver.receipts)
    same = driver.ingest(engine, broker_order, NOW)
    assert same.state() == before
    bad = deepcopy(broker_order)
    bad["fills"][0]["price"] += 0.001
    with pytest.raises(ValueError, match="conflicting_fill_receipt"):
        driver.ingest(engine, bad, NOW)
    assert engine.state() == before and driver.receipts == receipts
    bad = deepcopy(broker_order)
    bad["fills"] += [{**bad["fills"][0], "event_id": "foreign:fill"}]
    with pytest.raises(ValueError, match="unexpected_fill_receipt"):
        driver.ingest(engine, bad, NOW)
    assert engine.state() == before and driver.receipts == receipts


def test_depth_walk_respects_precision_and_fee_inclusive_cash(tmp_path):
    engine, driver = prepared(tmp_path)
    driver.outbox = None
    engine.pending.submitted = False
    quote = {"timestamp": NOW, "asks": [[0.3, 100], [0.3001, 100000]], "bids": [[0.2999, 100000]]}
    driver.prepare(engine, quote, NOW, RULES)
    engine = driver.resolve(engine, NOW, lambda candidate: None)
    assert len(engine.book.orders) == 2
    cost = sum(money(row["cost"]) for row in engine.book.orders)
    assert cost <= money(1000)
    assert engine.anchor > 0.3
    assert engine.book.cash >= 0
    assert all(money(row["quantity"]) % money(".0001") == 0 for row in engine.book.orders)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"timestamp": NOW - 11000}, "stale_execution_quote"),
        ({"asks": [[float("nan"), 1]]}, "invalid_orderbook"),
        ({"asks": [[0.29, 1]]}, "crossed_orderbook"),
        ({"asks": [[0.31, 1], [0.3, 1]]}, "unsorted_orderbook"),
    ],
)
def test_invalid_public_quotes_cannot_execute(change, error):
    quote = Market().snapshot()[1]
    quote.update(change)
    with pytest.raises(ValueError, match=error):
        validate_quote(quote, NOW)


@pytest.mark.parametrize(
    "api,path,method",
    [
        (["private", "spot"], "v2/spot/trade/place-order", "POST"),
        (["private", "spot"], "v2/spot/account/assets", "GET"),
        (["public", "spot"], "v2/spot/trade/place-order", "GET"),
    ],
)
def test_public_transport_cannot_access_private_or_financial_paths(api, path, method):
    client = PublicBitget()
    with pytest.raises(PermissionError, match="public GET only"):
        client.request(path, api, method)


def test_concurrent_snapshot_never_commits_sample(tmp_path):
    store = ShadowStore(tmp_path / "shadow.sqlite")
    store.save("shadow", 0, {"value": 1})
    store.save("shadow", 1, {"value": 2})
    with pytest.raises(RuntimeError, match="concurrent_shadow_change"):
        store.save("shadow", 1, {"value": 3}, {"timestamp": 10, "equity": 500})
    assert store.load("shadow")[1] == {"value": 2}
    assert store.curve() == []


def test_freeze_rejects_incomplete_source(tmp_path):
    runtime, source = make_runtime(tmp_path)
    saved = json.loads(source.read_text())
    del saved["config"]["buy_fraction"]
    source.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="complete_parameter_snapshot_required"):
        create_shadow(tmp_path / "shadows", source, tmp_path / "history.feather")
    with sqlite3.connect(runtime.store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM trx_state").fetchone()[0] == 1


def test_pause_during_market_connection_prevents_startup_and_competing_worker(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = runtime.market.snapshot

    def slow_snapshot():
        entered.set()
        assert release.wait(5)
        return original()

    runtime.market.snapshot = slow_snapshot
    runtime.start()
    assert entered.wait(5)
    second = ShadowRuntime(runtime.store.path, Market())
    with pytest.raises(RuntimeError, match="shadow_already_running"):
        second.start()
    runtime.stop()
    release.set()
    runtime.thread.join(5)
    assert not runtime.thread.is_alive()
    assert runtime.status()["phase"] == "paused"
    assert runtime.engine.book.principal == 0 and runtime.engine.book.usdgo == 0
    assert runtime.state["activated_at"] is None


def test_missing_price_reference_blocks_all_startup_actions(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    runtime.engine.config.price_position_lookback_days = 10
    with pytest.raises(ValueError, match="complete_p75_history_required"):
        runtime.poll()
    assert runtime.engine.book.principal == 0 and runtime.engine.book.usdgo == 0
    assert runtime.state["activated_at"] is None


def test_recent_candle_requires_spaced_consistent_reads_and_ignores_unclosed_bar():
    market = PublicShadowMarket()
    end = NOW // 60000 * 60000
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime([end - 120000, end - 60000, end], unit="ms", utc=True),
            "open": 0.3,
            "high": 0.3,
            "low": 0.3,
            "close": 0.3,
            "volume": 10.0,
        }
    )
    last = end - 120000
    assert len(market.confirm_recent(frame, last, NOW)) == 1
    assert len(market.confirm_recent(frame, last, NOW + 5000)) == 1
    confirmed = market.confirm_recent(frame, last, NOW + 15000)
    assert len(confirmed) == 2
    assert int(confirmed.date.iloc[-1].timestamp() * 1000) + 60000 <= NOW + 15000


def test_provisional_revision_restarts_confirmation_without_overwriting_saved_bar():
    market = PublicShadowMarket()
    end = NOW // 60000 * 60000
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime([end - 60000], unit="ms", utc=True),
            "open": 0.3,
            "high": 0.31,
            "low": 0.3,
            "close": 0.3,
            "volume": 10.0,
        }
    )
    assert market.confirm_recent(frame, end - 120000, NOW).empty
    frame.loc[0, "close"] = 0.31
    assert market.confirm_recent(frame, end - 120000, NOW + 15000).empty
    assert len(market.confirm_recent(frame, end - 120000, NOW + 30000)) == 1


def test_full_account_restores_accepted_startup_after_crash_before_ledger_commit(tmp_path):
    runtime, _ = make_runtime(tmp_path)

    class CrashAfterAccept(SimulatedBroker):
        def submit(self, request):
            super().submit(request)  # Separate broker transaction has committed.
            raise SystemExit("process_crash_before_fill_commit")

    runtime.execution.broker = CrashAfterAccept(runtime.store.path)
    with pytest.raises(SystemExit, match="process_crash_before_fill_commit"):
        runtime.poll()
    saved = runtime.store.load("shadow")[1]
    assert saved["execution"]["outbox"]["status"] == "submitting"
    assert saved["engine"]["book"]["orders"] == []
    restored = ShadowRuntime(runtime.store.path, Market())
    restored.poll()
    assert restored.state["startup_complete"]
    assert len(restored.engine.book.orders) == 1
    assert restored.engine.book.usdgo == 9000
    assert float(restored.engine.book.principal) == pytest.approx(3330.0033)
    assert restored.engine.earn.usdgo_waiting
    assert restored.engine.earn.start_at == NOW
    restored.poll()
    assert len(restored.engine.book.orders) == 1
