"""Frozen candidates on one longer uninterrupted account path per candidate."""

import json
import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import UTC, datetime
from functools import partial
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from freqtrade.strategy.trx_phase1.benchmarks import fair_benchmarks
from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.environment_study import ENVIRONMENT_FIELDS
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_earn import HistoricalEarn
from freqtrade.strategy.trx_phase1.research_metrics import StudyRecorder, read_json, risk_metrics
from freqtrade.strategy.trx_phase1.studies import rule_fingerprint


GROUP = "cross_period_validation"
PROTOCOL = "TRX-PHASE1-CROSS-PERIOD-VALIDATION-01"
CASES = ("C1", "legacy_dynamic_quantity", "inventory_08", "A1")
LABELS = {
    "C1": "C1: 9%",
    "legacy_dynamic_quantity": "Legacy dynamic quantity",
    "inventory_08": "Current eligible inventory sale: 8%",
    "A1": "A1: 6% with profitable inventory sizing",
}


def plain_account(config, timestamp):
    # Equivalent original engine/Earn pair; only redundant legacy approximate
    # benchmarks are omitted. The common complete passive paths are run once.
    engine = Engine(config)
    return engine, HistoricalEarn(engine.book, config, timestamp)


class ContinuousRecorder(StudyRecorder):
    def __init__(self, path, boundaries):
        super().__init__(path)
        self.boundaries = set(boundaries)
        self.checkpoints = {}

    def sample(self, engine, earn, timestamp, price, equity):
        super().sample(engine, earn, timestamp, price, equity)
        if timestamp in self.boundaries:
            held = [lot for lot in engine.book.lots if lot.quantity]
            self.checkpoints[str(timestamp)] = deepcopy(
                {
                    "timestamp": timestamp,
                    "phase": "after_close_before_next_open",
                    "price": price,
                    "equity": equity,
                    "engine": engine.state(),
                    "oldest_principal_days": max(
                        ((timestamp - lot.bought_at) / 86400000 for lot in held), default=0
                    ),
                    "principal_cost_older_365_days": sum(
                        float(lot.remaining_cost)
                        for lot in held
                        if timestamp - lot.bought_at >= 365 * 86400000
                    ),
                }
            )

    def finish(self, result, engine, earn):
        output = super().finish(result, engine, earn)
        path = Path(self.path).with_suffix(".checkpoints.json")
        path.write_text(json.dumps(self.checkpoints, allow_nan=False))
        result["continuous_checkpoints"] = {
            key: {k: v for k, v in row.items() if k != "engine"}
            for key, row in self.checkpoints.items()
        }
        return output


def longest_run(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return int((ends - starts).max()) if len(starts) else 0


def period_statistics(values, initial_cash, frequency):
    dates = pd.to_datetime(values.timestamp.to_numpy(dtype="int64") - 60000, unit="ms", utc=True)
    periods = (
        pd.DatetimeIndex(dates).tz_convert("Asia/Shanghai").tz_localize(None).to_period(frequency)
    )
    output = []
    for key, part in values.groupby(periods, sort=True):
        first, last = int(part.index[0]), int(part.index[-1])
        beginning = float(values.equity.iloc[first - 1]) if first else initial_cash
        local_drawdown = 1 - part.equity / part.equity.cummax().clip(lower=beginning)
        before_realized = float(values.realized_profit.iloc[first - 1]) if first else 0
        output.append(
            {
                "period": str(key),
                "start_at": int(part.timestamp.iloc[0] - 60000),
                "end_at": int(part.timestamp.iloc[-1]),
                "minutes": len(part),
                "start_equity": beginning,
                "end_equity": float(part.equity.iloc[-1]),
                "profit": float(part.equity.iloc[-1]) - beginning,
                "return_ratio": float(part.equity.iloc[-1]) / beginning - 1,
                "max_drawdown_in_period": float(local_drawdown.max()),
                "mean_exposure": float(part.exposure.mean()),
                "high_exposure_minutes_80": int((part.exposure >= 0.8).sum()),
                "longest_high_exposure_minutes_80": longest_run(part.exposure >= 0.8),
                "mean_interest_exposure": float((part.trx_interest_value / part.equity).mean()),
                "ending_interest_exposure": float(
                    part.trx_interest_value.iloc[-1] / part.equity.iloc[-1]
                ),
                "mean_old_principal_cost_30_days": float(part.stale_cost.mean()),
                "peak_old_principal_cost_30_days": float(part.stale_cost.max()),
                "realized_principal_change": float(part.realized_profit.iloc[-1]) - before_realized,
                "ending_principal_unrealized": float(part.principal_unrealized.iloc[-1]),
                "ending_interest_trx": float(part.interest_trx.iloc[-1]),
                "global_drawdown_at_end": float(values.global_drawdown.iloc[last]),
            }
        )
    return output


def continuous_metrics(path, initial_cash, output):
    with sqlite3.connect(path) as db:
        values = pd.read_sql_query(
            "SELECT timestamp,equity,price,exposure,interest_trx,trx_interest_value,"
            "realized_profit,principal_unrealized,remaining_cost,principal_trx "
            "FROM samples ORDER BY timestamp",
            db,
        )
        risk = pd.read_sql_query(
            "SELECT timestamp,stale_cost FROM risk_samples ORDER BY timestamp", db
        )
    if not np.array_equal(values.timestamp.to_numpy(), risk.timestamp.to_numpy()):
        raise ValueError("Minute risk and equity paths are not aligned")
    values["stale_cost"] = risk.stale_cost
    values["global_drawdown"] = 1 - values.equity / values.equity.cummax().clip(lower=initial_cash)
    monthly = period_statistics(values, initial_cash, "M")
    annual = period_statistics(values, initial_cash, "Y")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(monthly).to_csv(output / "monthly.csv", index=False)
    pd.DataFrame(annual).to_csv(output / "annual.csv", index=False)
    daily = values.iloc[np.unique(np.r_[np.arange(0, len(values), 1440), len(values) - 1])]
    daily.to_csv(output / "daily-path.csv", index=False)
    return {
        "month_calendar": "Asia/Shanghai; actual candle minutes, continuous account",
        "monthly": monthly,
        "annual": annual,
        "high_exposure_minutes_80": int((values.exposure >= 0.8).sum()),
        "longest_high_exposure_minutes_80": longest_run(values.exposure >= 0.8),
        "mean_interest_exposure": float((values.trx_interest_value / values.equity).mean()),
        "max_interest_exposure": float((values.trx_interest_value / values.equity).max()),
        "ending_interest_exposure": float(
            values.trx_interest_value.iloc[-1] / values.equity.iloc[-1]
        ),
        "daily_paths": json.loads(
            daily[
                [
                    "timestamp",
                    "equity",
                    "exposure",
                    "global_drawdown",
                    "trx_interest_value",
                    "stale_cost",
                ]
            ].to_json(orient="records")
        ),
    }


def load_cross_period_inputs(root, experiment_id):
    directory = Path(root) / "experiments" / str(UUID(str(experiment_id)))
    if not (directory / "candidate-freeze.json").exists():
        manifest = read_json(directory / "manifest.json")
        if manifest["group"] != GROUP:
            raise ValueError("Choose a completed cross-period coverage and candidate freeze")
        return load_cross_period_inputs(root, manifest["coverage_experiment_id"])
    freeze = read_json(directory / "candidate-freeze.json")
    cover = read_json(directory / "data-coverage.json")
    if (
        freeze["protocol_id"] != PROTOCOL
        or tuple(row["case"] for row in freeze["candidates"]) != CASES
    ):
        raise ValueError("Cross-period protocol does not contain the four frozen candidates")
    configurations = [
        ResearchConfig(**row["effective_config"]).model_dump(mode="json")
        for row in freeze["candidates"]
    ]
    if any(
        any(cfg[field] != configurations[0][field] for field in ENVIRONMENT_FIELDS)
        for cfg in configurations
    ):
        raise ValueError("Frozen candidates do not share one environment")
    saved = {}
    for row in freeze["candidates"]:
        path = (
            Path(root)
            / "experiments"
            / str(UUID(row["source_experiment_id"]))
            / "results"
            / f"{UUID(row['source_run_id'])}.json"
        )
        if sha256(path.read_bytes()).hexdigest() != row["source_result_sha256"]:
            raise ValueError("A frozen candidate source was changed")
        detail = read_json(path)
        if detail["config"] != row["saved_config"] or detail["summary"] != row["saved_summary"]:
            raise ValueError("A frozen candidate differs from its complete saved account")
        if (
            ResearchConfig.from_saved(detail["config"]).model_dump(mode="json")
            != row["effective_config"]
        ):
            raise ValueError("Effective candidate parameters differ from their frozen source")
        saved[row["case"]] = detail
    original = validate_minutes(pd.read_feather(directory / "inputs/original-control-1m.feather"))
    data = validate_minutes(pd.read_feather(directory / "inputs/continuous-1m.feather"))
    if (
        data_fingerprint(original) != freeze["candidates"][0]["original_data_fingerprint"]
        or data_fingerprint(data) != cover["selected_data_fingerprint"]
    ):
        raise ValueError("Cross-period source prices changed after coverage selection")
    overlap = data.loc[data.date >= original.date.iloc[0]].reset_index(drop=True)
    if data_fingerprint(overlap) != data_fingerprint(original):
        raise ValueError("The extended path does not preserve the original control year")
    return directory, freeze, cover, saved, original, data


def benchmark_periods(path, initial_cash):
    values = pd.read_csv(path, usecols=["timestamp", "equity", "exposure", "interest_trx", "price"])
    dates = pd.to_datetime(values.timestamp.to_numpy(dtype="int64") - 60000, unit="ms", utc=True)
    calendar = pd.DatetimeIndex(dates).tz_convert("Asia/Shanghai").tz_localize(None)
    reports = {}
    for label, frequency in (("monthly", "M"), ("annual", "Y")):
        rows = []
        for key, part in values.groupby(calendar.to_period(frequency), sort=True):
            first = int(part.index[0])
            beginning = float(values.equity.iloc[first - 1]) if first else initial_cash
            rows.append(
                {
                    "period": str(key),
                    "start_equity": beginning,
                    "end_equity": float(part.equity.iloc[-1]),
                    "return_ratio": float(part.equity.iloc[-1]) / beginning - 1,
                    "max_drawdown_in_period": float(
                        (1 - part.equity / part.equity.cummax().clip(lower=beginning)).max()
                    ),
                    "mean_exposure": float(part.exposure.mean()),
                    "ending_interest_trx": float(part.interest_trx.iloc[-1]),
                }
            )
        reports[label] = rows
    return reports


def run_case(frame, candidate, directory, scenario, boundaries):
    run_id = str(uuid4())
    path = Path(directory) / "results" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = ResearchConfig(**candidate["effective_config"])
    result = replay(
        frame,
        cfg,
        diagnostic_path=path.with_suffix(".diagnostics.sqlite"),
        recorder_factory=partial(ContinuousRecorder, boundaries=boundaries),
        engine_factory=plain_account,
    )
    risk, diagnosis = risk_metrics(path.with_suffix(".diagnostics.sqlite"), cfg.min_notional)
    result.update(
        run_id=run_id,
        experiment_id=path.parent.parent.name,
        risk_metrics=risk,
        protocol={
            "case": candidate["case"],
            "label": LABELS[candidate["case"]],
            "group": GROUP,
            "protocol_id": PROTOCOL,
            "stage": "observed_history",
            "scenario": scenario,
            "environment_label": (
                "Real historical prices plus frozen Earn conditions, "
                "not historical actual product returns"
            ),
            "continuous_account": True,
        },
    )
    stats = (
        continuous_metrics(
            path.with_suffix(".diagnostics.sqlite"),
            cfg.initial_cash,
            path.parent.parent / "analysis" / run_id,
        )
        if scenario == "continuous_history"
        else None
    )
    path.write_text(json.dumps(result, allow_nan=False))
    return {
        "case": candidate["case"],
        "run_id": run_id,
        "scenario": scenario,
        "config": result["config"],
        "summary": result["summary"],
        "risk_metrics": risk,
        "continuous_metrics": stats,
        "lots_summary": diagnosis["lots_summary"],
        "amounts": diagnosis["amounts"],
        "cash_only": not result["fills"],
        "baseline": scenario == "original_control",
        "value": candidate["case"],
        "source_run_id": candidate["source_run_id"],
        "compare_to": None,
        "return_change": 0,
        "drawdown_change": 0,
    }


def run_cross_period(root, source_id, experiment_id, verify, progress=None, workers=2):
    source_dir, freeze, cover, saved, original, data = load_cross_period_inputs(root, source_id)
    directory = Path(root) / "experiments" / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = rule_fingerprint()
    dates = pd.date_range(
        data.date.iloc[0].tz_convert("Asia/Shanghai").normalize(),
        data.date.iloc[-1].tz_convert("Asia/Shanghai"),
        freq="YS",
    )
    boundaries = [int(date.timestamp() * 1000) for date in dates] + [
        int(original.date.iloc[0].timestamp() * 1000)
    ]
    protocol = {
        **freeze,
        "experiment_id": experiment_id,
        "coverage_experiment_id": source_dir.name,
        "selected_interval": cover["selected"],
        "data_fingerprint": cover["selected_data_fingerprint"],
        "rules_sha256": fingerprint,
        "declared_at": datetime.now(UTC).isoformat(),
        "new_runs": [
            {"case": c["case"], "scenario": scenario}
            for scenario in ("original_control", "continuous_history")
            for c in freeze["candidates"]
        ],
        "checkpoint_boundaries": boundaries,
    }
    (directory / "protocol.json").write_text(json.dumps(protocol, allow_nan=False, indent=2))
    entries = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(
                run_case,
                frame,
                candidate,
                directory,
                scenario,
                boundaries if scenario == "continuous_history" else [],
            ): (scenario, candidate)
            for scenario, frame in (("original_control", original), ("continuous_history", data))
            for candidate in freeze["candidates"]
        }
        for future in as_completed(pending):
            scenario, candidate = pending[future]
            brief = future.result()
            if scenario == "original_control":
                result = read_json(directory / "results" / f"{brief['run_id']}.json")
                verify(saved[candidate["case"]], result)
                if result["summary"] != saved[candidate["case"]]["summary"]:
                    raise ValueError("Original-start control changed a saved complete summary")
            entries.append(brief)
            print(f"Completed {scenario} {candidate['case']}", flush=True)
            if progress:
                progress(len(entries) / 10)
    benchmarks = fair_benchmarks(
        data,
        ResearchConfig(**freeze["candidates"][0]["effective_config"]),
        directory / "benchmarks/continuous_history",
    )
    for row in benchmarks:
        row.update(scenario="continuous_history", plot_sample_minutes=1440)
        curve = row.get("equity_curve", [])
        row["equity_curve"] = curve[::24] + ([curve[-1]] if curve and len(curve) % 24 != 1 else [])
        row["continuous_periods"] = benchmark_periods(
            directory / "benchmarks/continuous_history" / row["curve_file"],
            freeze["candidates"][0]["effective_config"]["initial_cash"],
        )
    if rule_fingerprint() != fingerprint:
        raise ValueError("Frozen research source changed during cross-period validation")
    manifest = {
        "experiment_id": experiment_id,
        "group": GROUP,
        "protocol_id": PROTOCOL,
        "stage": "observed_history",
        "created_at": datetime.now(UTC).isoformat(),
        "coverage_experiment_id": source_dir.name,
        "source_experiment_id": freeze["candidate_source_experiment_id"],
        "start_utc": cover["selected"]["start_utc"],
        "end_utc": cover["selected"]["end_utc"],
        "data_fingerprint": cover["selected_data_fingerprint"],
        "rules_sha256": fingerprint,
        "candles": len(data),
        "data_coverage": cover,
        "candidate_freeze": freeze,
        "entries": sorted(entries, key=lambda row: (row["scenario"], CASES.index(row["case"]))),
        "benchmarks": benchmarks,
        "notes": [
            (
                "Four original-start controls reproduce saved complete summaries and paths; "
                "earlier-start inherited accounts need not reproduce the original annual return."
            ),
            (
                "One continuous engine/Earn account per candidate; reporting cuts never "
                "restart lots, anchor, cash, retained interest, queues or in-flight funds."
            ),
            (
                "Previously unused earlier prices are historical validation, "
                "not strict independent out-of-sample statistical evidence."
            ),
            (
                "Real historical prices plus a frozen quotation/product/cost scenario, "
                "not restored historical actual Earn returns."
            ),
            (
                "No trading-rule, candidate-priority, PRD-default or paused-paper-account "
                "changes; funding preparation remains separate."
            ),
        ],
    }
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, allow_nan=False))
    temporary.replace(directory / "manifest.json")
    if progress:
        progress(1)
    return manifest
