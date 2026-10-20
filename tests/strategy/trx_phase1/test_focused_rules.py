import csv
import json
from uuid import uuid4

import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.protocol import protocol_cases, protocol_parameters, run_protocol
from freqtrade.strategy.trx_phase1.rates import annual_interest
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_earn import HOUR, HistoricalEarn
from tests.strategy.trx_phase1.test_diagnostics import bars


def cfg(**values):
    return ResearchConfig(
        **{"version": "trx-phase1-v0.5", "variant": "fixed", "min_notional": 1, **values}
    )


def test_quantity_tiers_actual_credit_and_marginal_forecast():
    config = cfg(
        earn_model="net_benefit_hourly",
        usdt_apr_tiers=[(300, 0.0737), (120000000, 0.0271)],
        usdgo_apr_tiers=[(100000, 0.066), (1000000, 0.055)],
        fx_fee=0.0005,
    )
    assert annual_interest(config, "USDT", 1000) == money("41.08")
    book = Ledger(1000)
    earn = HistoricalEarn(book, config, 0)
    earn.sync_usdt(0)
    earn.advance(2 * HOUR, 1)
    assert earn.receipts[-1]["apr"] == pytest.approx(0.04108)
    assert earn.usdt_credit_value == money("41.08") / (365 * 24)
    forecast = earn.policy.forecast(2 * HOUR, money(700))
    cash = book.free_cash
    expected = annual_interest(config, "USDT", cash) - annual_interest(
        config, "USDT", cash - money(forecast["spend"])
    )
    assert forecast["usdt_alternative_interest"] == pytest.approx(float(expected) * 23 / (365 * 24))
    assert annual_interest(cfg(usdt_apr_tiers=[(300, 0.1)]), "USDT", 1000) == 30
    with pytest.raises(ValueError):
        cfg(usdt_apr_tiers=[(300, 0.1), (200, 0.01)])


def test_usdt_wait_has_no_spendable_cash_until_arrival_and_restores_exactly():
    config = cfg(earn_model="net_benefit_hourly", usdt_apr=0.05, usdt_redemption_delay_minutes=2)
    book = Ledger(1000)
    earn = HistoricalEarn(book, config, 0)
    earn.sync_usdt(0)
    prepared = earn.prepare_cash("buy", 1, 100, 10 * 60000)
    assert prepared == 1000
    assert book.cash == 1000 and book.free_cash == 900
    assert earn.policy.requests["buy"]["due_at"] == 120001
    clone = HistoricalEarn.restore(Ledger.restore(book.state()), config, earn.state())
    earn.advance(120000, 1)
    assert book.free_cash == 900
    earn.advance(120001, 1)
    clone.advance(120001, 1)
    assert book.free_cash == 1000
    assert earn.state() == clone.state()
    assert book.state() == clone.book.state()
    earn.prepare_cash("cancel", 120002, 100, 120003)
    earn.advance(120003, 1)
    assert book.free_cash == 1000
    assert earn.policy.requests["cancel"]["status"] == "canceled_before_arrival"


def test_preflight_rejects_price_and_funding_without_redemption_and_retains_profit_checks():
    engine = Engine(cfg(execution_preflight=True))
    engine.anchor = 1
    calls = []
    engine.prepare_cash = lambda *args: calls.append(args)
    assert engine.make_plan("buy", 60000, 0.988, 0.012, 360000) is None
    assert engine.reason == "preflight_price_condition" and not calls
    engine.config = cfg(
        execution_preflight=True, earn_model="net_benefit_hourly", usdt_redemption_delay_minutes=60
    )
    engine.earn = HistoricalEarn(engine.book, engine.config, 0)
    engine.earn.sync_usdt(0)
    engine.prepare_cash = engine.earn.prepare_cash
    assert engine.make_plan("buy", 60000, 0.98, 0.012, 360000) is None
    assert engine.reason == "funding_window_too_short"
    assert not engine.earn.policy.requests
    assert engine.book.free_cash == engine.book.cash
    seller = Engine(cfg(execution_preflight=True, tail_completion=False))
    seller.anchor = 1
    seller.book.buy("lot", "lot", 0, 100, 1.02)
    assert seller.book.qualifying(1.0217, 100, 0.001, 0.0005)
    assert seller.make_plan("sell", 60000, 1.0217, 0.012, 360000) is None


def test_profitable_inventory_basis_preserves_low_cost_batch():
    quantities = []
    for basis in ("principal", "profitable"):
        engine = Engine(cfg(initial_cash=2000, sell_quantity_basis=basis, tail_completion=False))
        engine.anchor = 0.8
        engine.book.buy("old", "old", 0, 900, 1.5)
        engine.book.buy("cheap", "cheap", 1, 100, 0.7)
        plan = engine.make_plan("sell", 60000, 1, 0.012, 360000)
        assert plan
        quantities.append(plan.quantity)
        assert all(row["lot_id"] == "cheap" for row in plan.allocations)
        engine.fill(plan.plan_id, plan.plan_id, 120000, plan.quantity, 1, plan.quantity * 0.001)
        assert engine.book.realized > 0
        assert engine.book.lots[0].remaining_cost == 1350
    assert quantities == [100, 30]


def stagnant_engine():
    engine = Engine(cfg(buy_reference_recovery=True, buy_fraction=0.08))
    engine.anchor = 0.5
    engine.initial_anchor_at = 1
    engine.book.buy("old", "old", 1, 100, 1.5)
    return engine


def test_recovery_is_frozen_bounded_and_does_not_rewrite_cost_or_fill_anchor():
    engine = stagnant_engine()
    now = 31 * 86400000
    engine.last_close = now - 60000
    engine.buy_at = engine.sell_at = now
    assert engine.on_close(now, 1, 0.001) is None
    assert engine.recovery_reference == 1 and engine.anchor == 0.5
    restored = Engine.restore(engine.state())
    assert restored.recovery_reference == 1 and restored.recovery_expires_at == now + 7 * 86400000
    engine.last_close = engine.buy_at - 60000
    plan = engine.on_close(engine.buy_at, 0.98, 0.001)
    assert plan and plan.reference_kind == "recovery" and plan.anchor == 1
    assert plan.budget <= 20 and engine.book.lots[0].remaining_cost == 150
    engine.fill(
        plan.plan_id,
        plan.plan_id,
        engine.last_close + 60000,
        plan.quantity,
        0.98,
        plan.quantity * 0.98 * 0.001,
    )
    assert engine.anchor == 0.98 and not engine.recovery_reference
    assert engine.book.lots[0].remaining_cost == 150
    assert engine.last_principal_fill_at > now
    expired = stagnant_engine()
    expired.update_recovery(now, 1, 0.012)
    expired.update_recovery(now + 7 * 86400000, 1.1, 0.012)
    assert not expired.recovery_reference
    assert expired.recovery_cooldown_until == now + 14 * 86400000
    assert expired.anchor == 0.5


def test_recovery_does_not_bypass_pause_profitable_inventory_or_funds():
    now = 31 * 86400000
    for mode in ("pause", "profitable", "funds", "stop_buys"):
        engine = stagnant_engine()
        if mode == "pause":
            engine.paused = True
        elif mode == "profitable":
            engine.book.buy("cheap", "cheap", 2, 10, 0.7)
        elif mode == "funds":
            engine.book.cash = money(0)
        else:
            engine.accept_new_buys = False
        engine.update_recovery(now, 1, 0.012)
        assert not engine.recovery_reference


def test_focused_protocol_pairs_and_full_benchmark_csvs(tmp_path):
    frame = bars(250)
    saved = replay(frame, cfg(volatility_window=20))
    saved["run_id"] = str(uuid4())
    snapshot = tmp_path / "rates.json"
    snapshot.write_text(
        json.dumps(
            {
                "source_page": "https://www.bitget.com/finance/savings/",
                "captured_at_utc": "2026-10-13 11:51:42 UTC",
                "ordinary_flexible_products": [
                    {"coin": "TRX", "displayed_apr": 0.1014},
                    {"coin": "USDGO", "tiers": [{"quantity_to": 100000, "apr": 0.066}]},
                    {
                        "coin": "USDT",
                        "tiers": [
                            {"quantity_to": 300, "apr": 0.0737},
                            {"quantity_to": 120000000, "apr": 0.0271},
                        ],
                    },
                ],
            }
        )
    )
    baseline, _ = protocol_parameters(saved, snapshot)
    cases = protocol_cases(baseline)
    by_key = {key: config for key, _, _, config in cases}
    environments = {key: scenario for key, _, scenario, _ in cases}
    for key, parent, scenario, config in cases:
        if parent:
            assert environments[parent] == scenario
        if not parent or scenario != "ordinary_snapshot":
            continue
        different = {
            field
            for field in config.model_dump()
            if config.model_dump()[field] != by_key[parent].model_dump()[field]
        }
        assert len(different) == (2 if key == "combined_08" else 1)
    study_id = str(uuid4())
    result = run_protocol(frame, saved, tmp_path, study_id, snapshot, collect_diagnostics=False)
    assert len(result["entries"]) == 9 and len(result["benchmarks"]) == 8
    pressure_baseline = next(row for row in result["entries"] if row["case"] == "pressure_08")
    assert pressure_baseline["compare_to"] is None
    assert pressure_baseline["return_change"] == pressure_baseline["drawdown_change"] == 0
    assert (
        json.loads((tmp_path / study_id / "manifest.json").read_text())["group"]
        == "focused_protocol"
    )
    for row in result["benchmarks"]:
        path = tmp_path / study_id / "benchmarks" / row["scenario"] / row["curve_file"]
        with path.open() as handle:
            values = list(csv.DictReader(handle))
        assert len(values) == len(frame)
        assert float(values[-1]["equity"]) == pytest.approx(row["summary"]["equity"])


@pytest.mark.parametrize(
    "extra",
    [
        {"buy_reference_recovery": True},
        {"sell_quantity_basis": "profitable"},
        {"sell_quantity_basis": "profitable", "buy_reference_recovery": True},
    ],
)
def test_new_rules_observation_is_invariant(tmp_path, extra):
    frame = bars()
    config = cfg(
        execution_preflight=True,
        earn_model="net_benefit_hourly",
        usdt_redemption_delay_minutes=1,
        redemption_delay_minutes=1,
        usdt_apr_tiers=[(300, 0.0737), (120000000, 0.0271)],
        usdgo_apr_tiers=[(100000, 0.066)],
        **extra,
    )
    plain = replay(frame, config)
    observed = replay(frame, config, diagnostic_path=tmp_path / "trace.sqlite")
    assert all(observed[key] == value for key, value in plain.items())


def test_v3_diagnostics_remain_readable_and_export_their_actual_columns(tmp_path):
    import sqlite3
    import zipfile

    from freqtrade.strategy.trx_phase1.diagnostics import available, export_package

    path = tmp_path / "trace.sqlite"
    result = replay(bars(150), cfg(volatility_window=20), diagnostic_path=path)
    result["run_id"] = str(uuid4())
    with sqlite3.connect(path) as db:
        data = json.loads(
            db.execute("SELECT value FROM metadata WHERE key='overview'").fetchone()[0]
        )
        data["schema"] = "trx-diagnostics-v3"
        db.execute("UPDATE metadata SET value=? WHERE key='overview'", [json.dumps(data)])
        db.execute("ALTER TABLE samples DROP COLUMN redemption_usdt")
        db.execute("ALTER TABLE samples DROP COLUMN recovery_reference")
    assert available(path)
    target = export_package(result, path, tmp_path / "legacy.zip")
    with zipfile.ZipFile(target) as archive:
        lines = list(csv.reader(archive.read("continuous_equity_funds.csv").decode().splitlines()))
    assert all(len(row) == len(lines[0]) for row in lines)
    assert "redemption_usdt" not in lines[0]
