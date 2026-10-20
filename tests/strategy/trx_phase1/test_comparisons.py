import json
import sqlite3
from copy import deepcopy
from uuid import uuid4

import pandas as pd
import pytest

from freqtrade.rpc.api_server import api_trx
from freqtrade.strategy.trx_phase1.candles import data_fingerprint
from freqtrade.strategy.trx_phase1.comparisons import VARIANTS, comparison_groups
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.replay import replay


def frame():
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="1min", tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )


def cohort():
    results = []
    comparison_id = str(uuid4())
    for variant in VARIANTS:
        result = replay(frame(), ResearchConfig(variant=variant, volatility_window=20))
        result.update(run_id=str(uuid4()), comparison_id=comparison_id)
        results.append(result)
    return results


def test_complete_same_data_same_assumptions_comparison():
    results = cohort()
    groups = comparison_groups(results)
    assert len(groups) == 1
    assert [r["variant"] for r in groups[0]["variants"]] == list(VARIANTS)
    assert groups[0]["candles"] == 40
    assert groups[0]["parameters"]["fee_rebate_ratio"] == 0.5
    assert comparison_groups(results[:2]) == []
    assert comparison_groups([*results, deepcopy(results[0])]) == []


@pytest.mark.parametrize("source", ["amounts", "diagnostics", "result"])
def test_experiment_trading_totals_use_saved_executions_without_rewriting(tmp_path, source):
    experiment_id, owner_id, run_id = uuid4(), uuid4(), uuid4()
    directory = tmp_path / "trx_phase1/experiments" / str(experiment_id)
    source_directory = directory.parent / str(owner_id) / "results"
    directory.mkdir(parents=True)
    source_directory.mkdir(parents=True)
    fills = [
        {"side": "buy", "quantity": 3, "price": 0.4, "fee": 0.05},
        {"side": "buy", "quantity": 2, "price": 0.5, "fee": 0.025},
        {"side": "sell", "quantity": 1, "price": 0.6, "fee": 0.02},
    ]
    entry = {
        "run_id": str(run_id),
        "source_experiment_id": str(owner_id),
        "summary": {"fees": 0.075, "fee_rebates": 0.02, "pending_fee_rebates": 0.01},
        "amounts": {},
    }
    if source == "amounts":
        entry["amounts"] = {"buy": {"total": 2.2}, "sell": {"total": 0.6}}
        entry["summary"]["gross_fees"] = 0.095
    elif source == "diagnostics":
        with sqlite3.connect(source_directory / f"{run_id}.diagnostics.sqlite") as db:
            db.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
            db.execute(
                "INSERT INTO metadata VALUES ('overview', ?)",
                (json.dumps({"schema": api_trx.diagnostics.SCHEMA}),),
            )
            db.execute("CREATE TABLE records (kind TEXT, payload TEXT)")
            db.executemany(
                "INSERT INTO records VALUES ('fills', ?)",
                [(json.dumps(fill),) for fill in fills],
            )
            db.execute(
                "INSERT INTO records VALUES ('cash_operations', ?)",
                (json.dumps({"quantity": 1000, "price": 1, "fee": 10}),),
            )
    else:
        (source_directory / f"{run_id}.json").write_text(
            json.dumps({"fills": fills, "orders": [{"quantity": 999, "price": 999}]})
        )
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"group": "earn_horizons", "entries": [entry]}))
    original = manifest.read_bytes()
    saved = api_trx.experiment_result(experiment_id, {"user_data_dir": tmp_path})
    summary = saved["entries"][0]["summary"]
    assert summary["trading_turnover"] == pytest.approx(2.8)
    assert summary["gross_fees"] == pytest.approx(0.095)
    assert summary["fees"] == 0.075  # Pending rebates do not reduce net fees.
    assert manifest.read_bytes() == original


def test_trading_totals_distinguish_zero_execution_from_missing_saved_values(tmp_path):
    saved = {
        "entries": [
            {"run_id": str(uuid4()), "summary": {}, "amounts": {}},
            {"run_id": "../outside", "summary": {}, "amounts": {}},
            {"summary": {}, "amounts": {"buy": {"total": 0}, "sell": {"total": 0}}},
        ],
        "benchmarks": [
            {"cash_only": True, "summary": {"fees": 0, "gross_fees": 0}},
            {
                "cash_only": False,
                "execution": "one initial purchase after identical warmup",
                "summary": {
                    "buy_orders": 2,
                    "sell_orders": 0,
                    "remaining_cost": 100.6,
                    "fees": 0.6,
                    "gross_fees": 1,
                    "pending_fee_rebates": 0.1,
                },
            },
            {"cash_only": False, "summary": {"fees": 0.6}},
        ],
    }
    api_trx.add_trading_totals(saved, tmp_path / str(uuid4()))
    assert "trading_turnover" not in saved["entries"][0]["summary"]
    assert "trading_turnover" not in saved["entries"][1]["summary"]
    assert saved["entries"][2]["summary"]["trading_turnover"] == 0
    assert saved["benchmarks"][0]["summary"]["trading_turnover"] == 0
    assert saved["benchmarks"][1]["summary"]["trading_turnover"] == pytest.approx(100)
    assert "trading_turnover" not in saved["benchmarks"][2]["summary"]


def test_api_batch_preserves_earn_rates_and_credits_all_three_variants(tmp_path, mocker):
    directory = tmp_path / "data"
    directory.mkdir()
    data = frame().iloc[:1].copy()
    data = pd.concat([data] * 180, ignore_index=True)
    data["date"] = pd.date_range("2026-01-01", periods=180, freq="1min", tz="UTC")
    data.to_feather(directory / "TRX_USDT-1m.feather")
    run_id = str(uuid4())
    mocker.patch.object(api_trx, "_jobs", {run_id: {"status": "running"}})
    request = api_trx.ReplayRequest(
        parameters=ResearchConfig(
            earn_model="simplified_hourly", trx_apr=0.08, usdgo_apr=0.13, volatility_window=20
        ),
        compare_variants=True,
    )
    api_trx.run_replay(
        run_id,
        request,
        {"exchange": {"name": "bitget"}, "datadir": directory, "user_data_dir": tmp_path},
    )
    assert api_trx.job_status(run_id)["status"] == "success"
    results = [
        json.loads(path.read_text()) for path in (tmp_path / "trx_phase1/results").glob("*.json")
    ]
    assert len(comparison_groups(results)) == 1
    for result in results:
        assert result["config"]["trx_apr"] == 0.08
        assert result["config"]["usdgo_apr"] == 0.13
        assert result["summary"]["usdgo_interest_quantity"] == pytest.approx(
            2 * 1000 * 0.13 / (365 * 24)
        )
        assert len(result["interest_receipts"]) == 2


@pytest.mark.parametrize("difference", ["interval", "capital", "fees", "data", "legacy"])
def test_incompatible_results_are_never_presented_as_comparable(difference):
    results = cohort()
    if difference == "interval":
        results[1]["end_utc"] = "2026-01-02 00:00:00+00:00"
    elif difference == "capital":
        results[1]["config"]["initial_cash"] = 2000
    elif difference == "fees":
        results[1]["config"]["fee_rebate_ratio"] = 0
    elif difference == "data":
        changed = frame()
        changed.loc[0, "volume"] = 2
        results[1]["data_fingerprint"] = data_fingerprint(changed)
    else:
        for result in results:
            result.pop("data_fingerprint")
    assert comparison_groups(results) == []


def test_batch_freezes_data_even_if_live_file_changes(tmp_path, mocker):
    directory = tmp_path / "data"
    directory.mkdir()
    source = directory / "TRX_USDT-1m.feather"
    original = frame()
    original.to_feather(source)
    replay_count = 0

    def concurrent_update(data, parameters, progress, diagnostic_path=None):
        nonlocal replay_count
        result = replay(data, parameters, progress, diagnostic_path=diagnostic_path)
        replay_count += 1
        changed = original.copy()
        changed.loc[0, "volume"] = replay_count + 1
        changed.to_feather(source)
        return result

    mocker.patch.object(api_trx, "replay", side_effect=concurrent_update)
    run_id = str(uuid4())
    mocker.patch.object(api_trx, "_jobs", {run_id: {"status": "running"}})
    request = api_trx.ReplayRequest(
        parameters=ResearchConfig(volatility_window=20), compare_variants=True
    )
    api_trx.run_replay(
        run_id,
        request,
        {"exchange": {"name": "bitget"}, "datadir": directory, "user_data_dir": tmp_path},
    )
    job = api_trx.job_status(run_id)
    assert job["status"] == "success"
    assert len(job["result_ids"]) == replay_count == 3
    results = [
        json.loads(path.read_text()) for path in (tmp_path / "trx_phase1/results").glob("*.json")
    ]
    assert {r["data_fingerprint"] for r in results} == {data_fingerprint(original)}
    assert data_fingerprint(pd.read_feather(source)) != data_fingerprint(original)
    assert len(comparison_groups(results)) == 1
