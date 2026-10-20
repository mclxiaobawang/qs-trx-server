import json
from functools import partial
from hashlib import sha256
from uuid import uuid4

import pytest

from freqtrade.rpc.api_server.api_trx import assert_reproduced
from freqtrade.strategy.trx_phase1.candles import data_fingerprint
from freqtrade.strategy.trx_phase1.cross_period import (
    CASES,
    PROTOCOL,
    ContinuousRecorder,
    load_cross_period_inputs,
    plain_account,
    run_cross_period,
)
from freqtrade.strategy.trx_phase1.cross_period_audit import audit_cross_period
from freqtrade.strategy.trx_phase1.history_coverage import coverage
from freqtrade.strategy.trx_phase1.replay import replay
from tests.strategy.trx_phase1.test_diagnostics import bars, config


def inputs(root):
    data = bars(1800)
    original = data.iloc[600:].reset_index(drop=True)
    source = root / "experiments" / str(uuid4())
    source.mkdir(parents=True)
    rows = []
    definitions = (
        ("dynamic_period_only", 0.09, "principal", False),
        ("dynamic_quantity", 0.12, "principal", False),
        ("dynamic_period", 0.08, "profitable", True),
        ("dynamic_period", 0.06, "profitable", True),
    )
    for case, (variant, fraction, basis, preflight) in zip(CASES, definitions, strict=True):
        cfg = config(
            variant=variant,
            buy_fraction=fraction,
            sell_quantity_basis=basis,
            execution_preflight=preflight,
        )
        result = replay(original, cfg)
        run_id = str(uuid4())
        result["run_id"] = run_id
        path = source / "results" / f"{run_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result))
        rows.append(
            {
                "case": case,
                "source_experiment_id": source.name,
                "source_run_id": run_id,
                "source_result_sha256": sha256(path.read_bytes()).hexdigest(),
                "saved_config": result["config"],
                "effective_config": cfg.model_dump(mode="json"),
                "saved_summary": result["summary"],
                "original_data_fingerprint": result["data_fingerprint"],
            }
        )
    target = root / "experiments" / str(uuid4())
    (target / "inputs").mkdir(parents=True)
    original.to_feather(target / "inputs/original-control-1m.feather")
    data.to_feather(target / "inputs/continuous-1m.feather")
    freeze = {
        "protocol_id": PROTOCOL,
        "candidates": rows,
        "candidate_source_experiment_id": source.name,
    }
    (target / "candidate-freeze.json").write_text(json.dumps(freeze))
    report = coverage(data)
    report.update(selected=report["intervals"][0], selected_data_fingerprint=data_fingerprint(data))
    (target / "data-coverage.json").write_text(json.dumps(report))
    return target, data, original


def test_read_only_checkpoints_preserve_the_original_engine_and_earn(tmp_path):
    frame = bars(1500)
    cfg = config(variant="dynamic_quantity")
    boundary = int(frame.date.iloc[600].timestamp() * 1000)
    plain = replay(frame, cfg)
    observed = replay(
        frame,
        cfg,
        diagnostic_path=tmp_path / "observed.sqlite",
        engine_factory=plain_account,
        recorder_factory=partial(ContinuousRecorder, boundaries=[boundary]),
    )
    assert_reproduced(plain, observed)
    assert plain["summary"] == observed["summary"]
    snapshot = json.loads((tmp_path / "observed.checkpoints.json").read_text())[str(boundary)]
    book = snapshot["engine"]["book"]
    assert sum(float(lot["quantity"]) for lot in book["lots"]) > 0
    assert float(book["interest_trx"]) > 0
    assert snapshot["engine"]["anchor"] is not None
    assert len(book["orders"]) == sum(row["timestamp"] < boundary for row in observed["fills"])
    assert snapshot["engine"]["earn"]["next_payment"] >= boundary


def test_four_controls_and_four_inherited_paths_with_calendar_reports(tmp_path):
    target, data, original = inputs(tmp_path)
    manifest = run_cross_period(tmp_path, target.name, str(uuid4()), assert_reproduced)
    assert len(manifest["entries"]) == 8
    assert len(manifest["benchmarks"]) == 4
    assert {row["case"] for row in manifest["entries"]} == set(CASES)
    for row in manifest["entries"]:
        if row["scenario"] != "continuous_history":
            continue
        stats = row["continuous_metrics"]
        assert sum(p["profit"] for p in stats["monthly"]) == pytest.approx(
            row["summary"]["total_profit"]
        )
        assert sum(p["minutes"] for p in stats["monthly"]) == len(data)
        assert stats["mean_interest_exposure"] > 0
    for benchmark in manifest["benchmarks"]:
        months = benchmark["continuous_periods"]["monthly"]
        assert months[1]["start_equity"] == months[0]["end_equity"]
        assert benchmark["full_curve_rows"] == len(data)
    audit = audit_cross_period(tmp_path, manifest["experiment_id"])
    assert all(
        row["minutes"] == len(original) for row in audit["original_calendar_window"].values()
    )
    for window in audit["shared_drawdown_windows"].values():
        assert set(window["accounts"]) == set(CASES)
        assert all(
            row["start_at"] == window["start_at"] and row["end_at"] == window["end_at"]
            for row in window["accounts"].values()
        )


def test_effective_parameters_cannot_drift_from_saved_candidate(tmp_path):
    target, _, _ = inputs(tmp_path)
    freeze = json.loads((target / "candidate-freeze.json").read_text())
    freeze["candidates"][0]["effective_config"]["buy_fraction"] = 0.10
    (target / "candidate-freeze.json").write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="Effective candidate parameters"):
        load_cross_period_inputs(tmp_path, target.name)
