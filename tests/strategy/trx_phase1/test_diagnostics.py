import json
import zipfile
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from freqtrade.rpc.api_server.api_trx import add_original_lot_costs, assert_reproduced
from freqtrade.strategy.trx_phase1.benchmarks import fair_benchmarks
from freqtrade.strategy.trx_phase1.config import MECHANISMS, ResearchConfig
from freqtrade.strategy.trx_phase1.diagnostics import connect, export_package, overview, records
from freqtrade.strategy.trx_phase1.experiments import run_experiment
from freqtrade.strategy.trx_phase1.replay import replay


def bars(count=1500, constant=False):
    close = np.full(count, 1.0) if constant else 1 + np.sin(np.arange(count) / 19) * 0.12
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-31 14:30", periods=count, freq="1min", tz="UTC"),
            "open": opening,
            "close": close,
            "high": np.maximum(opening, close),
            "low": np.minimum(opening, close),
            "volume": 10.0,
        }
    )


def config(**values):
    return ResearchConfig(
        **{
            "periods": [1, 5, 15, 60],
            "volatility_window": 20,
            "min_notional": 1,
            "earn_model": "net_benefit_hourly",
            **values,
        }
    )


@pytest.mark.parametrize("variant", MECHANISMS)
@pytest.mark.parametrize(
    "stress",
    [
        {},
        {"redemption_delay_minutes": 60},
        {
            "fx_fee": 0.0005,
            "fx_spread": 0.002,
            "fx_slippage": 0.0005,
            "fx_shock_after_minutes": 800,
            "fx_shock_rate": 0.97,
        },
    ],
)
def test_observer_never_changes_replay_and_reconciles(tmp_path, variant, stress):
    parameters = config(variant=variant, **stress)
    frame = bars()
    original = replay(frame, parameters)
    path = tmp_path / "trace.sqlite"
    observed = replay(frame, parameters, diagnostic_path=path)
    assert all(observed[key] == value for key, value in original.items())
    diagnosis = overview(path)
    assert abs(diagnosis["attribution_residual"]) < 1e-8
    assert diagnosis["sample_count"] == len(frame)
    if not parameters.redemption_delay_minutes:
        assert all(row["funding_wait"] == 0 for row in diagnosis["funnel"].values())
    else:
        assert diagnosis["funnel"]["buy"]["funding_wait"] > 0
    assert sum(row["buy_fills"] for row in diagnosis["months"]) == original["summary"]["buy_orders"]
    assert (
        sum(row["sell_fills"] for row in diagnosis["months"]) == original["summary"]["sell_orders"]
    )
    for month in diagnosis["months"]:
        for side in ("buy", "sell"):
            assert month[side + "_orders"] == month[side + "_fills"]
    with connect(path) as db:
        opportunities = records(db, "opportunities", limit=10000)["rows"]
        assert len({row["id"] for row in opportunities}) == len(opportunities)
        for row in opportunities:
            if row["fills"]:
                assert row["status"] == "filled"
                assert row["filled_quantity"] == pytest.approx(row["submitted_quantity"])
        executed = records(db, "fills", limit=10000)["rows"]
        if executed:
            assert executed[0]["snapshot_timing"] == "before principal execution"
            assert executed[0]["principal_trx"] == 0
        for side in ("buy", "sell"):
            signals = [r for r in opportunities if r["side"] == side]
            assert diagnosis["funnel"][side]["signals"] == len(signals)
            assert sum(diagnosis["funnel"][side]["terminal_reasons"].values()) == sum(
                not r["fills"] for r in signals
            )
        largest = max((r["drawdown"] for r in diagnosis["drawdowns"]), default=0)
        assert largest == pytest.approx(original["summary"]["max_drawdown"])
        cycles = records(db, "fx_cycles", limit=10000)["rows"]
        assert sum(row["credited_interest_value"] for row in cycles) == pytest.approx(
            original["summary"]["usdgo_interest_credited_value"], abs=1e-8
        )
        assert all(
            row["full_roundtrip_cost_recovered"] is None
            for row in cycles
            if row["status"] == "open"
        )


def test_cash_only_missing_history_and_export_are_explicit(tmp_path):
    path = tmp_path / "trace.sqlite"
    result = replay(bars(constant=True), config(), diagnostic_path=path)
    result["run_id"] = str(uuid4())
    assert overview(path)["cash_only"]
    package = export_package(result, path, tmp_path / "export.zip")
    with zipfile.ZipFile(package) as archive:
        assert len(archive.read("continuous_equity_funds.csv").splitlines()) == len(bars()) + 1
        assert json.loads(archive.read("effective_parameters.json")) == result["config"]
        assert "password" not in archive.read("data_and_rules.json").decode()
        assert "opportunities.jsonl" in archive.namelist()
    legacy = export_package(result, None, tmp_path / "legacy.zip")
    with zipfile.ZipFile(legacy) as archive:
        assert "continuous_equity_funds.csv" not in archive.namelist()
        assert "未采集" in archive.read("README.txt").decode()


def test_split_sizing_switches_inherit_and_isolate():
    frame = bars()
    for variant in MECHANISMS:
        original = replay(frame, config(variant=variant))
        explicit = replay(
            frame,
            config(
                variant=variant,
                buy_quantity_dynamic=MECHANISMS[variant][1],
                sell_quantity_dynamic=MECHANISMS[variant][1],
            ),
        )
        assert original["fills"] == explicit["fills"]
        assert original["summary"] == explicit["summary"]
    assert config(variant="fixed", buy_quantity_dynamic=True).sizing_enabled("buy")
    assert not config(variant="fixed", buy_quantity_dynamic=True).sizing_enabled("sell")


def test_controlled_comparison_keeps_baseline_and_one_group(tmp_path):
    frame = bars(300)
    saved = replay(frame, config())
    saved["run_id"] = str(uuid4())
    study = run_experiment(
        frame,
        saved,
        tmp_path,
        str(uuid4()),
        "buy_quantity_dynamic",
        [False, True],
        assert_reproduced,
    )
    assert len(study["entries"]) == 3
    for row in study["entries"][1:]:
        assert all(
            v == saved["config"][k] for k, v in row["config"].items() if k != "buy_quantity_dynamic"
        )
    assert len(study["benchmarks"]) == 4
    assert study["stage"] == "observed_history"
    with pytest.raises(ValueError):
        run_experiment(
            frame, saved, tmp_path, str(uuid4()), "buy_quantity_dynamic", [0.5], assert_reproduced
        )


def test_passive_benchmarks_charge_costs_and_do_not_trade_repeatedly():
    frame = bars(300)
    cfg = config(fx_fee=0.0005, fx_spread=0.002, fx_slippage=0.0005)
    alternatives = fair_benchmarks(frame, cfg)
    by_name = {row["name"]: row for row in alternatives}
    assert by_name["usdt"]["summary"]["equity"] == cfg.initial_cash
    assert by_name["usdgo"]["summary"]["equity"] < cfg.initial_cash
    assert by_name["trx_hold"]["summary"]["buy_orders"] <= 1
    assert by_name["low_exposure"]["summary"]["buy_orders"] <= 1
    assert all(row["summary"]["sell_orders"] == 0 for row in alternatives)


def test_original_lot_costs_include_exited_lots_and_credited_refunds(tmp_path):
    path = tmp_path / "trace.sqlite"
    result = replay(bars(), config(variant="fixed"), diagnostic_path=path)
    with connect(path) as db:
        lots = add_original_lot_costs(db, records(db, "lots", limit=10000)["rows"])
    assert any(row["status"] == "exited" for row in lots)
    buys = {row["id"]: row for row in result["orders"] if row["side"] == "buy"}
    for lot in lots:
        purchase = buys[lot["lot_id"]]
        assert lot["original_cost"] == pytest.approx(
            purchase["quantity"] * purchase["price"] + purchase["net_fee"]
        )
        assert lot["original_unit_cost"] > 0


def test_usdgo_passive_holding_keeps_fx_risk_with_zero_earn():
    cfg = config(earn_model="none", fx_shock_after_minutes=150, fx_shock_rate=0.97)
    alternatives = {row["name"]: row for row in fair_benchmarks(bars(300), cfg)}
    summary = alternatives["usdgo"]["summary"]
    assert summary["usdgo"] == 1000
    assert summary["equity"] == 970
    assert summary["earn_income_market_value"] == 0
    assert alternatives["usdt"]["summary"]["equity"] == 1000
