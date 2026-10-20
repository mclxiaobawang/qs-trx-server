from copy import deepcopy

import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.candles import normalized_volatility
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import LiveExecution
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy, LiveStrategyStore
from tests.strategy.trx_phase1.test_live_execution import (
    NOW,
    RULES,
    ExchangeFixture,
    credit_provider_interest,
)


class Market:
    def __init__(self, broker, now):
        self.broker, self.now = broker, now
        start = pd.Timestamp(now, unit="ms", tz="UTC").floor("D") - pd.Timedelta(days=2)
        self.data = pd.DataFrame(
            {
                "date": pd.date_range(
                    start,
                    pd.Timestamp(now, unit="ms", tz="UTC").floor("min") - pd.Timedelta(minutes=1),
                    freq="min",
                ),
                "open": 0.3,
                "high": 0.3,
                "low": 0.3,
                "close": 0.3,
                "volume": 1000.0,
            }
        )

    def snapshot(self):
        price = float(self.broker.state().get("trx_price", "0.3"))
        return self.now, {
            "timestamp": self.now,
            "asks": [[price, 1000000]],
            "bids": [[price, 1000000]],
        }

    def update(self, data, now):
        last = self.data.date.iloc[-1]
        stop = pd.Timestamp(now, unit="ms", tz="UTC").floor("min") - pd.Timedelta(minutes=1)
        if stop > last:
            price = float(self.broker.state().get("trx_price", "0.3"))
            new = pd.DataFrame(
                {
                    "date": pd.date_range(last + pd.Timedelta(minutes=1), stop, freq="min"),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1000.0,
                }
            )
            self.data = pd.concat([self.data, new], ignore_index=True)
        return self.data.copy()

    def close(self):
        pass


def candidate():
    return ResearchConfig(
        version="trx-phase1-v0.8",
        startup="allocated_position",
        initial_buy_fraction=0.1,
        variant="dynamic_period_only",
        periods=[60, 240, 1440],
        buy_threshold=0.014,
        sell_threshold=0.012,
        buy_fraction=0.12,
        sell_fraction=0.12,
        price_position_enabled=True,
        price_position_lookback_days=1,
        price_position_quantile=0.75,
        price_position_buy_multiplier=0.25,
        price_position_sell_multiplier=1.75,
        volatility_window=20,
        earn_model="net_benefit_hourly",
        slippage=0,
    )


@pytest.fixture
def live(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    original = broker.product
    broker.product = lambda coin, pid: {
        **original(coin, pid),
        "tiers": [{"minStepVal": "0", "maxStepVal": "10000000", "currentApy": "10"}],
    }
    market = Market(broker, NOW)
    execution = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    runtime = LiveStrategy.create(execution, market, candidate().model_dump(), market.data, NOW)
    return runtime, broker, market, tmp_path


def advance(live, seconds=15):
    runtime, _, market, _ = live
    market.now += seconds * 1000
    runtime.tick(market.now)


def finish(live):
    runtime = live[0]
    for _ in range(60):
        if runtime.state["intent"] is None:
            return
        advance(live)
    raise AssertionError(runtime.execution.status())


def startup(live):
    runtime, _, market, _ = live
    runtime.state["desired"] = "running"
    runtime.tick(market.now)
    finish(live)
    assert runtime.state["startup_complete"]


def next_buy_close(live):
    runtime, _, market, _ = live
    due = runtime.engine.buy_at + 30000
    while market.now + 60000 < due:
        advance(live, 60)
    market.now = due
    runtime.tick(market.now)


def test_creation_status_and_pause_cannot_activate_or_submit(live):
    runtime, broker, market, _ = live
    assert not runtime.status()["worker_running"]
    runtime.tick(market.now)
    assert broker.state()["writes"] == []


def test_product_profit_observation_survives_restart_without_crediting_strategy_income(live):
    runtime, broker, market, root = live
    startup(live)
    before = runtime.execution.status()
    broker.savings_observation = {
        "observed_at": market.now,
        "holdings": {
            "pooled": {
                "coin": "TRX",
                "product_id": "trx-flex",
                "interest_coin": "TRX",
                "total_profit": "0.5",
            }
        },
    }
    advance(live, 60)
    after = runtime.execution.status()
    for key in ("book", "holdings", "receipts", "anchor", "anchor_version"):
        assert after[key] == before[key]
    restored = LiveStrategy(LiveExecution(root, broker, runtime.execution.account_id), market)
    assert restored.state["savings_observation"] == broker.savings_observation
    assert restored.execution.status()["book"]["interest_trx"] == "0"


def test_live_observations_survive_restart_skip_duplicates_and_keep_full_drawdown(tmp_path):
    store = LiveStrategyStore(tmp_path / "strategy.sqlite")
    for timestamp, nav in [(1, 1.0), (2, 1.1), (3, 0.88), (4, 1.2)]:
        store.observe({"timestamp": timestamp, "equity": 1000 * nav}, nav)
    store.observe({"timestamp": 3, "equity": 1}, 0.001)
    restored = LiveStrategyStore(tmp_path / "strategy.sqlite").observations()
    assert [p["timestamp"] for p in restored["equity_curve"]] == [1, 2, 3, 4]
    assert restored["history"]["samples"] == 4
    assert restored["history"]["max_drawdown"] == pytest.approx(0.2)
    assert restored["history"]["started_at"] == 1
    assert restored["history"]["latest_at"] == 4


def test_live_observation_uses_owned_ledger_and_actual_fx_without_mutating_it(live):
    runtime, _, market, _ = live
    startup(live)
    before = runtime.execution.status()
    runtime.state["markets"]["TRXUSDT"].update(bid="0.31", timestamp=market.now + 1)
    runtime.state["markets"]["USDGOUSDT"].update(bid="0.99", timestamp=market.now + 1)
    runtime.capture_observation(market.now + 1)
    after = runtime.store.observations()["equity_curve"][-1]
    book = Ledger.restore(before["book"])
    assert after["principal_trx"] == float(book.principal)
    assert after["usdgo"] == float(book.usdgo)
    assert after["equity"] == pytest.approx(float(book.equity("0.31", "0.99")))
    assert runtime.execution.status() == before
    count = runtime.store.observations()["history"]["samples"]
    for row in runtime.state["markets"].values():
        row["timestamp"] = market.now - 61000
    runtime.capture_observation(market.now)
    assert runtime.store.observations()["history"]["samples"] == count


def test_live_curve_is_bounded_but_drawdown_uses_every_observation(tmp_path):
    store = LiveStrategyStore(tmp_path / "strategy.sqlite")
    for timestamp in range(1, 2010):
        nav = 0.5 if timestamp == 999 else 1.0
        store.observe({"timestamp": timestamp, "equity": 1000 * nav}, nav)
    result = store.observations()
    assert len(result["equity_curve"]) <= 2000
    assert result["equity_curve"][0]["timestamp"] == 1
    assert result["equity_curve"][-1]["timestamp"] == 2009
    assert result["history"]["samples"] == 2009
    assert result["history"]["max_drawdown"] == 0.5


def test_history_initializes_indicators_without_replaying_any_candle(live, mocker):
    runtime, broker, market, _ = live
    window = runtime.engine.config.volatility_window
    indices = runtime.data.index[-window - 1 :]
    prices = [0.3 * (1 + (i % 3) * 0.001) for i in range(window + 1)]
    for col in ("open", "high", "low", "close"):
        runtime.data.loc[indices, col] = prices
    expected = normalized_volatility(runtime.data.tail(window + 1), window).iloc[-1]
    observe = mocker.spy(runtime.engine, "on_close")
    runtime.initialize_indicators(market.now)
    assert runtime.engine.last_sigma == pytest.approx(expected)
    assert runtime.engine.last_sigma > 0
    assert runtime.engine.last_close == market.now
    assert runtime.state["indicator_initialization"]["signals_after"] == market.now
    assert runtime.state["warmup_until"] is None
    assert runtime.state["decisions"] == []
    observe.assert_not_called()
    assert broker.state()["writes"] == []


def test_ready_history_can_produce_a_new_dynamic_signal_before_two_hours(tmp_path):
    broker = ExchangeFixture(tmp_path / "exchange.sqlite")
    original = broker.product
    broker.product = lambda coin, pid: {
        **original(coin, pid),
        "tiers": [{"minStepVal": "0", "maxStepVal": "10000000", "currentApy": "10"}],
    }
    market = Market(broker, NOW)
    snapshot = candidate().model_copy(update={"volatility_window": 120}).model_dump()
    execution = LiveExecution.create(tmp_path, broker, "1000.4", RULES, NOW)
    runtime = LiveStrategy.create(execution, market, snapshot, market.data, NOW)
    live = (runtime, broker, market, tmp_path)
    startup(live)
    assert runtime.state["indicator_initialization"]["rows_used"] == 121
    assert runtime.state["warmup_until"] is None
    assert market.now < NOW + 120 * 60000
    assert runtime.state["decisions"] == []
    state = broker.state()
    state["trx_price"] = "0.294"
    broker.put(state)
    next_buy_close(live)
    assert runtime.engine.pending.side == "buy"
    assert NOW < runtime.engine.pending.created_at < NOW + 120 * 60000
    assert runtime.state["intent"]["kind"] == "trade"
    assert len(broker.state()["writes"]) == 4  # only startup; new intent has not submitted


def test_legacy_warmup_migration_preserves_holdings_costs_anchor_and_receipts(live):
    runtime, broker, market, root = live
    startup(live)
    before = runtime.execution.status()
    activated = runtime.state["activated_at"]
    runtime.state.pop("indicator_initialization", None)
    runtime.state["warmup_until"] = NOW + 120 * 60000
    runtime.persist()
    restored = LiveStrategy(LiveExecution(root, broker, runtime.execution.account_id), market)
    advance((restored, broker, market, root), 60)
    after = restored.execution.status()
    for key in ("book", "holdings", "receipts", "anchor", "anchor_version"):
        assert after[key] == before[key]
    assert restored.state["activated_at"] == activated
    assert restored.state["startup_complete"]
    assert restored.state["warmup_until"] is None
    assert restored.state["phase"] == "observing"
    assert restored.state["intent"] is None
    assert restored.engine.buy_at > market.now
    assert len(broker.state()["writes"]) == 4


def test_restart_skips_a_missed_signal_but_accepts_the_next_forward_confirmation(live):
    runtime, broker, market, root = live
    startup(live)
    missed = runtime.engine.buy_at
    state = broker.state()
    state["trx_price"] = "0.294"
    broker.put(state)
    market.now = missed + 60000
    restored = LiveStrategy(LiveExecution(root, broker, runtime.execution.account_id), market)
    restored.tick(market.now)
    assert restored.state["intent"] is None
    assert restored.state["decisions"] == []
    assert restored.engine.pending is None
    assert restored.engine.buy_at > market.now
    assert len(broker.state()["writes"]) == 4
    resumed_at = market.now
    next_buy_close((restored, broker, market, root))
    assert restored.engine.pending.side == "buy"
    assert restored.engine.pending.created_at > resumed_at
    assert all(d["timestamp"] > resumed_at for d in restored.state["decisions"])


@pytest.mark.parametrize("invalid", ["insufficient", "stale", "gap"])
def test_indicator_initialization_refuses_incomplete_history_before_any_order(live, invalid):
    runtime, broker, market, _ = live
    if invalid == "insufficient":
        runtime.data = runtime.data.tail(runtime.engine.config.volatility_window)
        code = "volatility_history_incomplete"
    elif invalid == "stale":
        runtime.data = runtime.data.iloc[:-5]
        code = "finalized_market_data_stale"
    else:
        runtime.data = runtime.data.drop(runtime.data.index[-2])
        code = "gaps"
    with pytest.raises((ExecutionError, ValueError), match=code):
        runtime.initialize_indicators(market.now)
    assert runtime.state["indicator_initialization"] is None
    assert runtime.engine.last_close == 0
    assert broker.state()["writes"] == []


def test_indicator_initialization_excludes_a_future_unclosed_candle(live):
    runtime, broker, market, _ = live
    runtime.initialize_indicators(market.now)
    expected = runtime.engine.last_sigma
    reference = runtime.engine.price_position_reference
    row = runtime.data.iloc[-1].copy()
    row["date"] = row["date"] + pd.Timedelta(minutes=1)
    for key in ("open", "high", "low", "close"):
        row[key] = 100
    runtime.data = pd.concat([runtime.data, pd.DataFrame([row])], ignore_index=True)
    runtime.reference_day = None
    runtime.initialize_indicators(market.now)
    assert runtime.engine.last_sigma == expected
    assert runtime.engine.price_position_reference == reference
    assert runtime.engine.last_close == market.now
    assert broker.state()["writes"] == []
    assert runtime.engine.earn is None
    assert runtime.engine.config.fee_rebate_ratio == 0


def test_transient_public_quote_recovers_without_pausing_or_backfilling(live):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    runtime.poll_error(ValueError("stale_execution_quote"), market.now)
    assert runtime.state["phase"] == "recovering"
    assert runtime.state["desired"] == "running"
    runtime.tick(market.now)
    assert runtime.state["error"] is None
    assert runtime.engine.pending is None
    assert broker.state()["writes"] == []


def test_evidence_delay_retries_for_five_minutes_then_requires_review(live):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    runtime.poll_error(ExecutionError("wallet_reconciliation_required"), market.now)
    assert runtime.state["phase"] == "recovering"
    runtime.poll_error(ExecutionError("interest_arrival_bill_unverified"), market.now + 299999)
    assert runtime.state["desired"] == "running"
    runtime.poll_error(ExecutionError("wallet_reconciliation_required"), market.now + 300000)
    assert runtime.state["phase"] == "blocked"
    assert runtime.state["desired"] == "paused"
    assert runtime.state["error"] == "account_evidence_wait_expired"
    assert broker.state()["writes"] == []


def test_conflicting_market_data_still_blocks_instead_of_retrying(live):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    runtime.poll_error(ValueError("conflicting_history_pages"), market.now)
    assert runtime.state["phase"] == "blocked"
    assert runtime.state["desired"] == "paused"
    assert broker.state()["writes"] == []


def test_real_workflow_startup_exactly_once_and_native_owned_positions(live):
    runtime, broker, _, _ = live
    startup(live)
    book = Ledger.restore(runtime.execution.status()["book"])
    assert book.initial == money("1000.4")
    assert 99 < book.cost < 101
    assert 898 < book.usdgo < 901
    assert 0 <= book.principal - book.lots[0].locked < money("0.00000001")
    assert book.locked["USDGO"] == book.usdgo
    assert book.stable_interest == book.interest_trx == 0
    count = len(broker.state()["writes"])
    advance(live)
    assert len(broker.state()["writes"]) == count


def test_saved_intent_recovers_queue_ack_crash_without_duplicate_startup(live, mocker):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    real_queue = runtime.execution.queue

    def crash(*args, **kwargs):
        real_queue(*args, **kwargs)
        raise RuntimeError("simulated_process_loss_after_queue_commit")

    mocker.patch.object(runtime.execution, "queue", side_effect=crash)
    with pytest.raises(RuntimeError):
        runtime.tick(market.now)
    assert runtime.state["intent"]["flow_id"] is None
    restarted = LiveStrategy(runtime.execution, market)
    live = (restarted, broker, market, live[3])
    finish(live)
    assert restarted.state["startup_complete"]
    assert len([w for w in broker.state()["writes"] if w["kind"] == "trade"]) == 2


def test_dynamic_period_signal_funds_buy_and_retains_real_anchor(live):
    runtime, broker, market, _ = live
    startup(live)
    market.now += 60000
    runtime.tick(market.now)  # catch-up observes without retrospective trading
    state = broker.state()
    state["trx_price"] = "0.294"
    broker.put(state)
    next_buy_close(live)
    assert runtime.engine.pending and runtime.engine.pending.side == "buy"
    steps = runtime.state["intent"]["steps"]
    assert [s["kind"] for s in steps] == ["redeem", "trade", "trade"]
    assert steps[-1]["signal_anchor"] == str(runtime.engine.anchor)
    assert runtime.engine.pending.budget == pytest.approx(120.048)
    finish(live)
    execution = runtime.execution.status()
    assert money(execution["anchor"]) == pytest.approx(money("0.294"), abs=money("0.000001"))
    assert execution["anchor_version"] == 2


def test_signal_guard_rechecks_after_funding_before_trx_write(live):
    runtime, broker, market, _ = live
    startup(live)
    market.now += 60000
    runtime.tick(market.now)
    state = broker.state()
    state["trx_price"] = "0.294"
    broker.put(state)
    next_buy_close(live)
    state = broker.state()
    state["trx_price"] = "0.3"
    broker.put(state)
    finish(live)
    assert runtime.execution.status()["flow"]["status"] == "expired"
    trx_buys = [
        order
        for order in broker.state()["orders"].values()
        if order["symbol"] == "TRXUSDT" and order["side"] == "buy"
    ]
    assert len(trx_buys) == 1


def test_p75_does_not_use_current_day_and_restart_does_not_autostart(live):
    runtime, broker, market, _ = live
    runtime.reference_at(market.now)
    reference = runtime.engine.price_position_reference
    current = (market.data.date + pd.Timedelta(hours=8)).dt.floor("D") == (
        pd.Timestamp(market.now, unit="ms", tz="UTC") + pd.Timedelta(hours=8)
    ).floor("D")
    runtime.data.loc[current, ["open", "high", "low", "close"]] = 100
    runtime.reference_day = None
    runtime.reference_at(market.now)
    assert runtime.engine.price_position_reference == reference
    runtime.state["desired"] = "running"
    runtime.persist()
    restored = LiveStrategy(runtime.execution, market)
    assert restored.status()["phase"] == "restart_requires_preflight"
    assert broker.state()["writes"] == []


def test_funding_projection_never_unlocks_execution_lots_or_adds_fake_cash(live):
    runtime, _, market, _ = live
    startup(live)
    runtime.execution.reload()
    before = deepcopy(runtime.execution.state)
    funding = runtime.environment(market.now)
    book = funding.planning_book()
    assert all(lot.locked == 0 for lot in book.lots)
    assert runtime.execution.state == before
    assert book.cash == money(before["book"]["cash"])
    assert funding.available_cash() > book.free_cash


def test_missing_history_and_wallet_mismatch_stop_new_actions(live):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    runtime.data = runtime.data.tail(100).reset_index(drop=True)
    market.data = runtime.data
    with pytest.raises(ExecutionError, match="complete_p75_history_required"):
        runtime.tick(market.now)
    assert broker.state()["writes"] == []


def test_engine_trade_decision_uses_fixed_snapshot_with_actual_environment(live):
    runtime, _, market, _ = live
    startup(live)
    market.now += 60000
    runtime.tick(market.now)
    assert runtime.engine.config.variant == "dynamic_period_only"
    assert runtime.engine.config.buy_fraction == runtime.engine.config.sell_fraction == 0.12
    assert runtime.engine.config.buy_fee == runtime.engine.config.sell_fee == 0.001
    assert runtime.engine.config.trx_apr == runtime.engine.config.usdgo_apr == 0
    assert Engine.restore(runtime.engine.state()).earn is None


def test_rebound_redeems_selected_profitable_lots_and_never_sells_interest(live):
    runtime, broker, market, _ = live
    startup(live)
    market.now += 60000
    runtime.tick(market.now)
    state = broker.state()
    state["trx_price"] = "0.294"
    broker.put(state)
    next_buy_close(live)
    finish(live)
    advance(live, 60)  # return the newly acquired batch to real Savings
    finish(live)
    market.now += 1000
    credit_provider_interest(runtime.execution, "TRX", "0.01", "earn", market.now)
    state = broker.state()
    state["trx_price"] = "0.31"
    broker.put(state)
    for _ in range(240):
        advance(live, 60)
        if runtime.engine.pending and runtime.engine.pending.side == "sell":
            break
        finish(live)
    else:
        raise AssertionError("No forward sell signal")
    assert runtime.engine.pending and runtime.engine.pending.side == "sell"
    steps = runtime.state["intent"]["steps"]
    assert steps[0]["kind"] == "redeem" and steps[0]["redeem_lots"]
    source = runtime.execution.status()
    target = sum(money(q) for q in steps[0]["redeem_lots"].values())
    assert target > 0
    finish(live)
    book = Ledger.restore(runtime.execution.status()["book"])
    assert book.realized > 0 and book.interest_trx == money("0.01")
    assert all(order["profit"] > 0 for order in book.orders if order["side"] == "sell")
    assert source["book"]["initial"] == book.state()["initial"]


def test_pause_after_accepted_startup_prevents_remaining_writes_but_keeps_receipts(live):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    runtime.tick(market.now)
    advance(live)
    advance(live)
    assert len(broker.state()["writes"]) == 1
    runtime.pause()
    for _ in range(6):
        advance(live)
    assert len(broker.state()["writes"]) == 1
    book = Ledger.restore(runtime.execution.status()["book"])
    assert book.usdgo > 0  # accepted conversion is accounted for, never discarded
    assert not runtime.state["startup_complete"]


def test_exchange_acceptance_timeout_recovers_before_new_signal(live):
    runtime, broker, market, _ = live
    runtime.state["desired"] = "running"
    runtime.tick(market.now)
    advance(live)
    broker.fault = "timeout_after_accept"
    advance(live)
    assert runtime.execution.status()["operation"]["status"] == "unknown"
    broker.fault = None
    restarted = LiveStrategy(runtime.execution, market)
    finish((restarted, broker, market, live[3]))
    assert restarted.state["startup_complete"]
    assert len(broker.state()["writes"]) == 4


def test_unconfirmed_cancel_blocks_automatic_strategy_without_another_financial_write(live):
    runtime, broker, market, root = live
    runtime.state["desired"] = "running"
    runtime.tick(market.now)
    advance(live)
    broker.fault = "partial"
    advance(live)
    runtime.execution.reload()
    runtime.execution.state["flow"]["expires_at"] = market.now + 1000
    runtime.execution.save()
    broker.fault = "cancel_timeout_before_accept"
    advance(live)
    assert runtime.execution.status()["operation"]["cancel_outcome"] == "unknown"
    restored = LiveStrategy(LiveExecution(root, broker, runtime.execution.account_id), market)
    advance((restored, broker, market, root), 60)
    assert restored.state["phase"] == "blocked"
    assert restored.state["desired"] == "paused"
    assert restored.state["error"] == "cancel_confirmation_timeout"
    assert restored.execution.status()["flow"]["status"] == "needs_review"
    assert len(broker.state()["writes"]) == 1


def test_inflight_interest_subscription_does_not_lose_a_confirmation_signal(live):
    runtime, broker, market, _ = live
    startup(live)
    market.now += 60000
    runtime.tick(market.now)
    due = runtime.engine.buy_at
    while market.now + 60000 < due - 10000:
        advance(live, 60)
    market.now = due - 10000
    credit_provider_interest(runtime.execution, "TRX", "0.2", "spot", market.now)
    runtime.tick(market.now)
    assert runtime.state["intent"]["kind"] == "earn"
    state = broker.state()
    state["trx_price"] = "0.294"
    broker.put(state)
    market.now = due + 30000
    runtime.tick(market.now)
    assert runtime.state["intent"]["kind"] == "earn"
    assert runtime.engine.last_close >= due
    assert runtime.engine.pending.side == "buy" and runtime.engine.pending.created_at == due
    finish(live)
    advance(live)
    assert runtime.state["intent"]["kind"] == "trade"
    assert runtime.engine.pending.created_at == due
    finish(live)
    assert money(runtime.execution.status()["anchor"]) == pytest.approx(
        money("0.294"), abs=money("0.000001")
    )
