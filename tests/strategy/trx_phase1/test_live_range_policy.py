import hashlib
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.active_price_band import DAY, OFFSET, BandHistory
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest
from freqtrade.strategy.trx_phase1.live_range_policy import (
    HISTORY_FILE,
    RangeEngine,
    RangePolicy,
    preview,
    validate_binding,
)
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy
from tests.strategy.trx_phase1.test_live_strategy import live as live_fixture
from tests.strategy.trx_phase1.test_live_strategy import startup


@pytest.fixture
def live(tmp_path):
    return live_fixture.__wrapped__(tmp_path)


@pytest.fixture
def completed(live):
    startup(live)
    runtime, broker, market, _ = live
    now = market.now
    cutoff = ((now + OFFSET) // DAY) * DAY - OFFSET
    first = cutoff - 366 * DAY
    ticks = np.full(366 * 1440, 300000, dtype=np.int64)
    path = runtime.execution.root / HISTORY_FILE
    np.savez_compressed(path, first_open_at=np.array(first), close_ticks=ticks)
    b = {
        "policy": RangePolicy().model_dump(),
        "account_id": runtime.state["account_id"],
        "source_sha256": runtime.state["source_sha256"],
        "profit_started_at": runtime.state["activated_at"],
        "execution_created_at": runtime.execution.state["created_at"],
        "initial_capital": runtime.execution.state["book"]["initial"],
        "history_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "history_first_open_at": first,
        "history_rows": len(ticks),
    }
    control = {"account_id": b["account_id"], "source_sha256": b["source_sha256"]}
    return runtime, broker, market, b, control


def test_preview_is_pure_and_does_not_activate_a_completed_account(completed):
    runtime, broker, market, binding, control = completed
    before = deepcopy((runtime.state, runtime.execution.state, broker.state()))
    history = BandHistory.load(
        runtime.execution.root / HISTORY_FILE,
        binding["history_sha256"],
        binding["history_first_open_at"],
        binding["history_rows"],
    )
    result = preview(
        runtime.state, runtime.execution.state, control, history, market.data, market.now
    )
    assert result["formal_activation_performed"] is False
    assert result["exchange_write_requests"] == 0
    assert result["profit_started_at"] == runtime.state["activated_at"]
    assert result["book_initial"] == runtime.execution.state["book"]["initial"]
    assert [s["base_buy_fraction"] for s in result["scenarios"]] == [0.12, 0.12, 0.09]
    assert (runtime.state, runtime.execution.state, broker.state()) == before
    assert type(runtime.engine) is Engine


@pytest.mark.parametrize(
    "field", ["account_id", "source_sha256", "profit_started_at", "initial_capital"]
)
def test_binding_cannot_attach_to_another_origin_or_source(completed, field):
    runtime, _, _, b, c = completed
    b[field] = "bad" if isinstance(b[field], str) else 0
    c["range_policy_sha256"] = digest(b)
    state = {**runtime.state, "range_policy": b}
    with pytest.raises((ExecutionError, ValueError)):
        validate_binding(state, runtime.execution.state, c)


def test_unconfirmed_policy_marker_cannot_enable_range_rules(completed):
    runtime, _, _, b, c = completed
    with pytest.raises(ExecutionError, match="range_policy_binding_mismatch"):
        validate_binding({**runtime.state, "range_policy": b}, runtime.execution.state, c)


def test_opted_in_restore_preserves_book_origin_contributions_and_no_initial_buy(completed):
    runtime, broker, market, b, c = completed
    runtime.state["range_policy"] = b
    runtime.persist()
    c["range_policy_sha256"] = digest(b)
    control = runtime.execution.root.parent.parent / "live-strategy-control.json"
    control.write_text(json.dumps(c))
    before = deepcopy(runtime.execution.state)
    writes = deepcopy(broker.state()["writes"])
    restored = LiveStrategy(runtime.execution, market)
    assert isinstance(restored.engine, RangeEngine)
    restored.data = market.data
    restored.reference_at(market.now)
    assert restored.state["active_price_band"]["lower"] == 0.3
    assert restored.engine.band["upper"] == 0.3
    assert restored.execution.state == before
    assert broker.state()["writes"] == writes
    assert restored.state["startup_complete"]
    assert restored.state["activated_at"] == runtime.state["activated_at"]
    assert restored.engine.config.fee_rebate_ratio == 0


@pytest.mark.parametrize(
    "inside,high,fraction",
    [(True, False, 0.12), (False, False, 0.09), (True, True, 0.03), (False, True, 0.0225)],
)
def test_signal_budget_keeps_p75_and_pending_plan_across_band_changes(
    completed, inside, high, fraction
):
    runtime, _, market, _, _ = completed
    engine = RangeEngine(runtime.engine.config, Ledger(10000))
    engine.policy = RangePolicy()
    t = ((market.now + OFFSET) // DAY) * DAY - OFFSET
    price = 0.33 if inside else 0.31
    engine.set_band({"effective_at": t, "lower": 0.32, "upper": 0.3328})
    engine.anchor = 0.36
    engine.price_position_day = (t + OFFSET) // DAY
    engine.price_position_reference = 0.2 if high else 0.4
    threshold = 0.00725 if inside else 0.014
    plan = engine.make_plan("buy", t, price, threshold, t + 2 * DAY)
    assert plan.budget == pytest.approx(10000 * fraction)
    assert engine.config.buy_fraction == 0.12
    quantity, budget = plan.quantity, plan.budget
    engine.set_band({"effective_at": t + DAY, "lower": 0.1, "upper": 0.104})
    assert engine.pending.quantity == quantity and engine.pending.budget == budget
    saved = engine.state()
    restored = RangeEngine.restore(saved)
    assert restored.pending.quantity == quantity and restored.pending.threshold == threshold
    assert money(restored.book.state()["initial"]) == money(10000)


def test_history_uses_only_closed_minutes_before_midnight_and_rejects_conflicts():
    cutoff = int(pd.Timestamp("2026-10-26", tz="Asia/Shanghai").timestamp() * 1000)
    first = cutoff - 366 * DAY
    ticks = np.full(366 * 1440 + 1440, 300000, dtype=np.int64)
    ticks[-1440:] = 800000
    recent = pd.DataFrame(
        {
            "date": pd.to_datetime(
                np.arange(cutoff - DAY, cutoff + DAY, 60000), unit="ms", utc=True
            ),
            "close": np.r_[np.full(1440, 0.3), np.full(1440, 0.8)],
        }
    )
    a = BandHistory(first, ticks).at(cutoff + 1000, recent)
    assert a["lower"] == a["upper"] == 0.3
    assert a["latest_training_close_at"] == cutoff
    changed = ticks.copy()
    changed[-1440:] = 100000
    future = recent.copy()
    future.loc[future.date.ge(pd.Timestamp(cutoff, unit="ms", tz="UTC")), "close"] = 0.1
    assert BandHistory(first, changed).at(cutoff + 1000, future) == a
    recent.loc[0, "close"] = 0.29
    with pytest.raises(ExecutionError, match="range_history_conflict"):
        BandHistory(first, ticks).at(cutoff + 1000, recent)


def test_missing_history_never_falls_back_to_a_shorter_window():
    cutoff = int(pd.Timestamp("2026-10-26", tz="Asia/Shanghai").timestamp() * 1000)
    first = cutoff - 366 * DAY
    seed = np.full(365 * 1440, 300000, dtype=np.int64)
    recent = pd.DataFrame(
        {"date": pd.to_datetime([cutoff - 60000], unit="ms", utc=True), "close": [0.3]}
    )
    with pytest.raises(ExecutionError, match="range_history_incomplete"):
        BandHistory(first, seed).at(cutoff, recent)


def test_preflight_and_start_preserve_account_bound_policy_for_restart(tmp_path, mocker):
    from tests.strategy.trx_phase1.test_live_restore import completed as completed_service
    from tests.strategy.trx_phase1.test_live_service import service as base_service

    svc, broker, clock, _, _ = completed_service.__wrapped__(
        base_service.__wrapped__(tmp_path)
    )
    control = svc.load()
    runtime = svc.get_runtime(control)
    execution = deepcopy(runtime.execution.state)
    writes = deepcopy(broker.state()["writes"])
    cutoff = ((clock[0] + OFFSET) // DAY) * DAY - OFFSET
    path = runtime.execution.root / HISTORY_FILE
    ticks = np.full(366 * 1440, 300000, dtype=np.int64)
    first = cutoff - 366 * DAY
    np.savez_compressed(path, first_open_at=np.array(first), close_ticks=ticks)
    binding = {
        "policy": RangePolicy().model_dump(),
        "account_id": control["account_id"],
        "source_sha256": control["source_sha256"],
        "profit_started_at": runtime.state["activated_at"],
        "execution_created_at": execution["created_at"],
        "initial_capital": execution["book"]["initial"],
        "history_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "history_first_open_at": first,
        "history_rows": len(ticks),
    }
    runtime.state.update(desired="paused", phase="paused", range_policy=binding)
    runtime.persist()
    control.update(range_policy_sha256=digest(binding), range_policy_activated_at=clock[0])
    svc.save(control)
    svc.runtime = None
    checked = svc.prepare(control["setup"])
    assert svc.load()["range_policy_sha256"] == digest(binding)
    assert svc.load()["range_policy_activated_at"] == clock[0]
    start = mocker.patch.object(LiveStrategy, "start")
    svc.start(checked["prepared"]["review_hash"])
    start.assert_called_once()
    svc.runtime = None
    restored = svc.get_runtime(svc.load())
    assert isinstance(restored.engine, RangeEngine)
    validate_binding(restored.state, restored.execution.state, svc.load())
    assert restored.execution.state["book"] == execution["book"]
    assert restored.state["activated_at"] == binding["profit_started_at"]
    assert broker.state()["writes"] == writes
