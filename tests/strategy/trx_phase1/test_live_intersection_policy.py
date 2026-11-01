from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.active_price_band import DAY, OFFSET, BandHistory
from freqtrade.strategy.trx_phase1.ledger import Ledger
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.live_execution import digest
from freqtrade.strategy.trx_phase1.live_observation import observed_close
from freqtrade.strategy.trx_phase1.live_range_policy import (
    HISTORY_FILE,
    RangeEngine,
    RangePolicy,
    intersection_preview,
    validate_binding,
)
from tests.strategy.trx_phase1.test_live_range_policy import completed as completed_fixture
from tests.strategy.trx_phase1.test_live_range_policy import live as live_fixture


@pytest.fixture
def live(tmp_path):
    return live_fixture.__wrapped__(tmp_path)


@pytest.fixture
def completed(live):
    return completed_fixture.__wrapped__(live)


def engine_for(completed, enabled=True):
    runtime, _, market, _, _ = completed
    engine = RangeEngine(runtime.engine.config, Ledger(10000))
    engine.policy = RangePolicy(pause_upper_p75_buys=enabled)
    t = ((market.now + OFFSET) // DAY) * DAY - OFFSET + 60000
    engine.set_band({"effective_at": t - 60000, "lower": 0.30, "upper": 0.312})
    engine.anchor = 0.4
    engine.last_close = t - 60000
    engine.last_sigma = 0.001
    engine.buy_at, engine.sell_at = t, t + DAY
    engine.price_position_day = (t + OFFSET) // DAY
    engine.price_position_reference = 0.34
    return engine, t


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "price,p75,filtered,fraction",
    [
        (0.35, 0.34, True, 0.0225),
        (0.35, 0.36, False, 0.09),
        (0.312, 0.30, False, 0.03),  # Equality to upper remains inside.
        (0.35, 0.35, False, 0.09),  # Equality to P75 is not high.
        (0.31, 0.30, False, 0.03),
        (0.29, 0.28, False, 0.0225),  # Below band but high P75 does not filter.
        (0.29, 0.34, False, 0.09),
    ],
)
def test_only_strict_double_high_new_buys_are_filtered(
    completed, mocker, enabled, price, p75, filtered, fraction
):
    engine, t = engine_for(completed, enabled)
    engine.price_position_reference = p75
    before = deepcopy(engine.book.state())
    prepare = mocker.Mock(return_value=10000.0)
    engine.prepare_cash = prepare
    plan = engine.make_plan("buy", t, price, 0.00725 if 0.30 <= price <= 0.312 else 0.014, t + DAY)
    if enabled and filtered:
        assert plan is None and engine.pending is None and engine.sequence == 0
        assert engine.reason == "upper_and_p75_new_buy_paused"
        prepare.assert_not_called()
        assert engine.intersection_signal["original_desired_budget"] == pytest.approx(225)
    else:
        assert plan.budget == pytest.approx(10000 * fraction)
        prepare.assert_called_once()
    assert engine.book.state() == before
    assert engine.anchor == 0.4 and engine.config.buy_fraction == 0.12


@pytest.mark.parametrize("reference,day_offset", [(None, 0), (float("nan"), 0), (0.34, -1)])
def test_enabled_filter_requires_a_valid_same_day_p75(completed, reference, day_offset):
    engine, t = engine_for(completed)
    engine.price_position_reference = reference
    engine.price_position_day += day_offset
    before = deepcopy(engine.state())
    with pytest.raises(ExecutionError, match="current_intersection_p75_reference_required"):
        engine.make_plan("buy", t, 0.35, 0.014, t + DAY)
    assert engine.state() == before


def test_skipping_advances_schedule_records_reason_and_does_not_queue_a_catchup(completed):
    engine, t = engine_for(completed)
    before = deepcopy(engine.book.state())
    plan, rows = observed_close(engine, t, 0.35, 0.001, t)
    assert plan is None and not engine.paused and engine.pending is None
    assert engine.buy_at > t and engine.sequence == 0 and engine.anchor == 0.4
    assert rows[0]["reason"] == "upper_and_p75_new_buy_paused"
    assert rows[0]["result"] == "blocked"
    # Leaving the intersection before the next original evaluation does not trade.
    engine.price_position_reference = 0.36
    assert engine.on_close(t + 60000, 0.35, 0.001) is None
    assert engine.book.state() == before
    saved = engine.state()
    restored = RangeEngine.restore(saved)
    assert restored.intersection_signal == engine.intersection_signal
    assert restored.book.state() == before


def test_existing_buy_plan_is_not_retroactively_cancelled_on_entry_to_intersection(completed):
    engine, t = engine_for(completed)
    engine.price_position_reference = 0.30
    plan = engine.make_plan("buy", t, 0.31, 0.00725, t + DAY)
    engine.last_close = t
    before = deepcopy(engine.state()["pending"])
    assert engine.on_close(t + 60000, 0.35, 0.001) is plan
    assert engine.state()["pending"] == before
    # Existing execution price/threshold protection is still present.
    assert not engine.executable(t + 60000, 0.35)


def test_profitable_sells_are_identical_with_filter_enabled(completed):
    outputs = []
    for enabled in [False, True]:
        engine, t = engine_for(completed, enabled)
        engine.book.buy("lot", "lot", t - DAY, 1000, 0.30, 0.30)
        before = deepcopy(engine.book.state())
        plan = engine.make_plan("sell", t, 0.35, 0.012, t + DAY)
        assert plan.side == "sell" and plan.quantity == pytest.approx(210)
        assert engine.book.state() == before
        outputs.append(deepcopy(engine.state()))
    assert outputs[0] == outputs[1]


def test_candidate_marker_requires_both_account_bound_copies_to_match(completed):
    runtime, _, _, binding, control = completed
    binding["policy"]["pause_upper_p75_buys"] = True
    state = {**runtime.state, "range_policy": binding}
    with pytest.raises(ExecutionError, match="range_policy_binding_mismatch"):
        validate_binding(state, runtime.execution.state, control)
    control["range_policy_sha256"] = digest(binding)
    assert validate_binding(state, runtime.execution.state, control).pause_upper_p75_buys
    # A historical binding without the new field preserves the old behavior.
    del binding["policy"]["pause_upper_p75_buys"]
    control["range_policy_sha256"] = digest(binding)
    assert not validate_binding(state, runtime.execution.state, control).pause_upper_p75_buys


def test_intersection_switch_preview_is_pure_and_keeps_account_origin(completed):
    runtime, broker, market, binding, control = completed
    state = {**deepcopy(runtime.state), "range_policy": binding}
    control["range_policy_sha256"] = digest(binding)
    history = BandHistory.load(
        runtime.execution.root / HISTORY_FILE,
        binding["history_sha256"],
        binding["history_first_open_at"],
        binding["history_rows"],
    )
    before = deepcopy((state, runtime.execution.state, control, broker.state()))
    result = intersection_preview(
        state, runtime.execution.state, control, history, market.data, market.now
    )
    assert result["current_policy"]["pause_upper_p75_buys"] is False
    assert result["policy"]["pause_upper_p75_buys"] is True
    assert result["exchange_write_requests"] == 0 and not result["formal_activation_performed"]
    assert result["profit_started_at"] == state["activated_at"]
    assert result["book_initial"] == runtime.execution.state["book"]["initial"]
    assert (state, runtime.execution.state, control, broker.state()) == before
