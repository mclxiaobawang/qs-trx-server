import json
from uuid import uuid4

import pandas as pd
import pytest

from freqtrade.rpc.api_server.api_trx import assert_reproduced
from freqtrade.strategy.trx_phase1.candidate_attribution import explain_c1
from freqtrade.strategy.trx_phase1.candidate_stability import (
    equal_diagnostics,
    load_stability_source,
    run_stability_study,
    stability_cases,
)
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_metrics import StudyRecorder, risk_metrics
from tests.strategy.trx_phase1.test_diagnostics import bars, config


def sources(tmp_path):
    frame = bars(1200)
    directory = tmp_path / "experiments" / str(uuid4())
    saved, entries = {}, []
    for key, variant, fraction in (
        ("C1", "dynamic_period_only", 0.09),
        ("C2", "dynamic_period_only", 0.06),
        ("legacy_dynamic_period_only", "dynamic_period_only", 0.12),
        ("legacy_dynamic_quantity", "dynamic_quantity", 0.12),
    ):
        run_id = str(uuid4())
        path = directory / "results" / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = config(
            variant=variant,
            buy_fraction=fraction,
            periods=[5],
            redemption_delay_minutes=1,
            usdt_redemption_delay_minutes=1,
        )
        result = replay(
            frame,
            cfg,
            diagnostic_path=path.with_suffix(".diagnostics.sqlite"),
            recorder_factory=StudyRecorder,
        )
        result.update(run_id=run_id)
        risk, diagnosis = risk_metrics(path.with_suffix(".diagnostics.sqlite"), cfg.min_notional)
        result["risk_metrics"] = risk
        path.write_text(json.dumps(result))
        saved[key] = result
        entries.append(
            {
                "case": key,
                "run_id": run_id,
                "config": result["config"],
                "summary": result["summary"],
                "lots_summary": diagnosis["lots_summary"],
                "amounts": diagnosis["amounts"],
                "risk_metrics": risk,
                "baseline": False,
                "cash_only": False,
            }
        )
    source = {k: saved["C1"][k] for k in ("start_utc", "end_utc", "data_fingerprint")}
    source.update(
        experiment_id=directory.name,
        source_experiment_id=str(uuid4()),
        rate_snapshot={},
        entries=entries,
    )
    (directory / "manifest.json").write_text(json.dumps({**source, "group": "risk_matched_rules"}))
    return frame, source, saved, directory


def test_only_two_neighbors_and_independent_predeclared_pairs(tmp_path):
    _, source, saved, _ = sources(tmp_path)
    source["end_utc"] = "2027-01-01 00:00:00+00:00"
    cases = stability_cases(source, saved)
    assert len(cases) == 6
    for row in cases[:2]:
        before, after = saved["C1"]["config"], row["config"].model_dump()
        assert {key for key in before if before[key] != after[key]} == {"buy_fraction"}
    assert [row["config"].buy_fraction for row in cases[:2]] == [0.08, 0.10]
    for row in cases[2:]:
        assert row["config"].model_dump() == saved["legacy_dynamic_quantity"]["config"]
        assert row["terms"].ready_hold_minutes == 240
        assert row["terms"].cooldown_minutes == 60
    assert cases[-1]["terms"].switch_at == int(
        (pd.Timestamp(source["start_utc"]) + pd.Timedelta(days=180)).timestamp() * 1000
    )


def test_source_loader_requires_complete_consistent_c1_ledgers(tmp_path):
    _, source, _, directory = sources(tmp_path)
    manifest, saved = load_stability_source(tmp_path, source["experiment_id"])
    assert manifest["data_fingerprint"] == saved["C1"]["data_fingerprint"]
    path = directory / "results" / f"{saved['C1']['run_id']}.json"
    changed = json.loads(path.read_text())
    changed["summary"]["equity"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="saved complete ledger"):
        load_stability_source(tmp_path, source["experiment_id"])


def test_attribution_reconciles_corrected_sales_costs_and_monthly_paths(tmp_path):
    frame, _, saved, directory = sources(tmp_path)
    report = explain_c1(
        directory / "results" / f"{saved['legacy_dynamic_period_only']['run_id']}.json",
        directory / "results" / f"{saved['C1']['run_id']}.json",
        frame,
        tmp_path / "analysis",
    )
    assert abs(report["attribution_residual"]) < 1e-8
    assert sum(row["change"] for row in report["net_components"]) == pytest.approx(
        report["net_equity_change"]
    )
    assert sum(row["equity_gap_change"] for row in report["monthly"]) == pytest.approx(
        report["net_equity_change"]
    )
    assert report["batches"]["source"] == len(saved["legacy_dynamic_period_only"]["lots"])
    assert report["batches"]["concentration"]["realized_change"]["net"] == pytest.approx(
        saved["C1"]["summary"]["realized_profit"]
        - saved["legacy_dynamic_period_only"]["summary"]["realized_profit"]
    )
    assert (tmp_path / "analysis/c1-source-minute-paths.csv").is_file()


def test_normal_no_interference_and_continuous_funding_prefix(tmp_path, monkeypatch):
    frame, source, saved, _ = sources(tmp_path)
    declared = {**source, "end_utc": "2027-01-01 00:00:00+00:00"}
    cases = stability_cases(declared, saved)
    at = int(frame.date.iloc[600].timestamp() * 1000)
    for row in cases[-2:]:
        row["terms"].switch_at = at
    monkeypatch.setattr(
        "freqtrade.strategy.trx_phase1.candidate_stability.stability_cases", lambda *_: cases
    )
    result = run_stability_study(frame, source, saved, tmp_path, str(uuid4()), assert_reproduced)
    assert len(result["entries"]) == 9
    assert result["verification"]["normal_funding_preparation_events"] == 0
    assert result["verification"]["normal_funding_full_path_equality"]["samples"] == len(frame)
    rows = [row for row in result["entries"] if row["scenario"] == "after_180_days_both_60"]
    assert rows[0]["funding_transition"] == rows[1]["funding_transition"]
    assert result["verification"]["day180_prefix_equality"]["samples"] == 600
    left = (
        tmp_path
        / "experiments"
        / result["experiment_id"]
        / "results"
        / f"{rows[0]['run_id']}.diagnostics.sqlite"
    )
    right = left.with_name(f"{rows[1]['run_id']}.diagnostics.sqlite")
    with pytest.raises(ValueError, match="paths differ"):
        equal_diagnostics(left, right)
