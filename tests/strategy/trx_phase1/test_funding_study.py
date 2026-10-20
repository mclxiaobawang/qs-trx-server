import json
from uuid import UUID, uuid4

import pandas as pd
import pytest

from freqtrade.rpc.api_server.api_trx import assert_reproduced, experiment_result
from freqtrade.strategy.trx_phase1.funding_preparation import FundingFactory, FundingTerms
from freqtrade.strategy.trx_phase1.funding_study import funding_cases, run_funding_study
from freqtrade.strategy.trx_phase1.replay import replay
from tests.strategy.trx_phase1.test_diagnostics import bars, config


def test_predeclared_pairs_freeze_trading_rules_and_split_redemption_sides():
    source = {"start_utc": "2025-10-12 00:00:00+00:00"}
    saved = {"config": config(variant="dynamic_quantity").model_dump()}
    rows = funding_cases(source, saved)
    assert len(rows) == 17 and sum(row["terms"].enabled for row in rows) == 8
    assert all(row["config"].model_dump() == saved["config"] for row in rows)
    for row in rows[1:]:
        if row["case"].startswith("after_90_days"):
            assert row["terms"].switch_at == int(
                pd.Timestamp("2026-01-10", tz="UTC").timestamp() * 1000
            )
    stable = next(row for row in rows if row["case"] == "from_cash_stable_60_control")
    trx = next(row for row in rows if row["case"] == "from_cash_trx_60_control")
    engine, _ = FundingFactory(stable["terms"].model_dump())(stable["config"], 0)
    assert engine.config.trx_redemption_delay == 1
    assert (
        engine.config.usdt_redemption_delay_minutes == engine.config.redemption_delay_minutes == 60
    )
    engine, _ = FundingFactory(trx["terms"].model_dump())(trx["config"], 0)
    assert engine.config.trx_redemption_delay == 60
    assert engine.config.redemption_delay_minutes == 0


def test_paired_study_checks_reproduction_and_preserves_continuous_transition(
    tmp_path, monkeypatch
):
    frame = bars(1000)
    cfg = config(
        variant="dynamic_quantity",
        periods=[5],
        usdt_redemption_delay_minutes=1,
        redemption_delay_minutes=1,
    )
    saved = replay(frame, cfg)
    saved["run_id"] = str(uuid4())
    source = {k: saved[k] for k in ("start_utc", "end_utc", "data_fingerprint")}
    source.update(experiment_id=str(uuid4()), rate_snapshot={"frozen": True})
    at = int(frame.date.iloc[500].timestamp() * 1000)
    definitions = [
        {
            "case": "funding_ordinary_control",
            "scenario": "ordinary_snapshot",
            "config": cfg,
            "terms": FundingTerms(),
            "compare_to": None,
        }
    ]
    for enabled in (False, True):
        definitions.append(
            {
                "case": "prepared" if enabled else "control",
                "scenario": "both",
                "config": cfg,
                "compare_to": "control" if enabled else None,
                "terms": FundingTerms(
                    enabled=enabled,
                    switch_at=at,
                    environment_changes={
                        "redemption_delay_minutes": 60,
                        "usdt_redemption_delay_minutes": 60,
                    },
                ),
            }
        )
    monkeypatch.setattr(
        "freqtrade.strategy.trx_phase1.funding_study.funding_cases", lambda *_: definitions
    )
    result = run_funding_study(frame, source, saved, tmp_path, str(uuid4()), assert_reproduced)
    left, right = result["entries"][1:]
    assert left["funding_transition"] == right["funding_transition"]
    assert left["funding_transition"]["principal_trx"] > 0
    assert abs(right["risk_metrics"]["attribution_residual"]) < 1e-8
    with pytest.raises(ValueError, match="data differs"):
        run_funding_study(frame.iloc[:-1], source, saved, tmp_path, str(uuid4()), assert_reproduced)


def test_overview_omits_duplicate_history_without_mutating_saved_evidence(tmp_path):
    experiment_id = str(uuid4())
    directory = tmp_path / "trx_phase1/experiments" / experiment_id
    directory.mkdir(parents=True)
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "group": "funding_preparation",
                "entries": [
                    {
                        "funding_transition": {
                            "principal_trx": 100,
                            "earn_state": {"receipts": [1, 2, 3]},
                        }
                    }
                ],
            }
        )
    )
    before = path.read_bytes()
    overview = experiment_result(UUID(experiment_id), {"user_data_dir": tmp_path})
    assert overview["entries"][0]["funding_transition"] == {"principal_trx": 100}
    assert path.read_bytes() == before
    assert json.loads(before)["entries"][0]["funding_transition"]["earn_state"]["receipts"] == [
        1,
        2,
        3,
    ]
