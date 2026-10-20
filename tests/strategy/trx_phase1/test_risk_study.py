import json
import sqlite3
from uuid import uuid4

import pytest

from freqtrade.rpc.api_server.api_trx import TwoTrackRequest, assert_reproduced, two_track_source
from freqtrade.strategy.trx_phase1.benchmarks import fair_benchmarks
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_metrics import StudyRecorder
from freqtrade.strategy.trx_phase1.risk_study import SOURCE_CASES, risk_cases, run_risk_study
from tests.strategy.trx_phase1.test_diagnostics import bars, config


def saved_sources(frame):
    saved = {}
    for key, changes in zip(
        SOURCE_CASES,
        [
            {
                "buy_fraction": 0.08,
                "sell_quantity_basis": "profitable",
                "execution_preflight": True,
            },
            {"variant": "dynamic_quantity", "buy_fraction": 0.16},
            {"variant": "dynamic_period_only", "buy_fraction": 0.14},
        ],
        strict=True,
    ):
        saved[key] = replay(frame, config(**changes))
        saved[key]["run_id"] = str(uuid4())
    return saved


def test_six_single_field_changes_use_saved_allocations():
    saved = saved_sources(bars(100))
    rows = {row["case"]: row for row in risk_cases(saved)}
    assert len(rows) == 9
    assert [rows[k]["config"].buy_fraction for k in ("A1", "B1", "B2", "C1", "C2")] == [
        0.06,
        0.12,
        0.08,
        0.10500000000000001,
        0.07,
    ]
    for row in rows.values():
        if row["compare_to"]:
            source = ResearchConfig.from_saved(saved[row["compare_to"]]["config"]).model_dump()
            changed = row["config"].model_dump()
            assert {k for k in source if source[k] != changed[k]} == {row["changed_field"]}
    assert not rows["B3"]["config"].execution_preflight


def test_rule_study_controls_and_passive_allocations(tmp_path):
    frame = bars(200)
    saved = saved_sources(frame)
    source = {
        "experiment_id": str(uuid4()),
        "data_fingerprint": saved[SOURCE_CASES[0]]["data_fingerprint"],
        "start_utc": saved[SOURCE_CASES[0]]["start_utc"],
        "end_utc": saved[SOURCE_CASES[0]]["end_utc"],
        "rate_snapshot": {"frozen": True},
    }
    checked = []

    def verify(original, actual):
        assert_reproduced(original, actual)
        checked.append(original["run_id"])

    result = run_risk_study(frame, source, saved, tmp_path, str(uuid4()), verify)
    assert len(checked) == 3 and len(result["entries"]) == 9
    assert [row["allocation_fraction"] for row in result["benchmarks"]] == [0.4, 0.7]
    assert all(abs(row["risk_metrics"]["attribution_residual"]) < 1e-8 for row in result["entries"])
    with pytest.raises(ValueError, match="data differs"):
        run_risk_study(frame.iloc[:-1], source, saved, tmp_path, str(uuid4()), verify)


def test_longitudinal_observer_keeps_economics_and_has_every_minute(tmp_path):
    frame = bars(600)
    cfg = config()
    original = replay(frame, cfg)
    path = tmp_path / "risk.sqlite"
    observed = replay(frame, cfg, diagnostic_path=path, recorder_factory=StudyRecorder)
    assert_reproduced(original, observed)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM risk_samples").fetchone()[0] == len(frame)
        assert db.execute(
            "SELECT min(underwater_cost),min(stale_cost) FROM risk_samples"
        ).fetchone() == (0, 0)
    with pytest.raises(ValueError, match="Invalid passive"):
        fair_benchmarks(frame, cfg, allocations=[("../../invalid", 0.4)])


def test_followup_study_resolves_its_frozen_original_source(tmp_path):
    original_id, next_id = str(uuid4()), str(uuid4())
    root = tmp_path / "trx_phase1"
    original, later = root / "experiments" / original_id, root / "experiments" / next_id
    (original / "results").mkdir(parents=True)
    later.mkdir(parents=True)
    saved = saved_sources(bars(100))
    entries = []
    for key, detail in saved.items():
        (original / "results" / f"{detail['run_id']}.json").write_text(json.dumps(detail))
        entries.append(
            {
                "case": key,
                "scenario": "ordinary_snapshot",
                **{k: detail[k] for k in ("config", "summary", "run_id")},
            }
        )
    manifest = {
        "group": "legacy_environment",
        "experiment_id": original_id,
        "entries": entries,
        "data_fingerprint": saved[SOURCE_CASES[0]]["data_fingerprint"],
    }
    (original / "manifest.json").write_text(json.dumps(manifest))
    (later / "manifest.json").write_text(
        json.dumps({"group": "risk_matched_rules", "source_experiment_id": original_id})
    )
    _, source, actual = two_track_source(
        {"user_data_dir": tmp_path}, TwoTrackRequest(source_experiment_id=next_id)
    )
    assert source["experiment_id"] == original_id and actual == saved
