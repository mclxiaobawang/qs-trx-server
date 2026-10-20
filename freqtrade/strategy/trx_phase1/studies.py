import json
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import pandas as pd

from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes
from freqtrade.strategy.trx_phase1.config import MECHANISMS, ResearchConfig
from freqtrade.strategy.trx_phase1.replay import replay


EXPERIMENTS = tuple(MECHANISMS)


def rule_fingerprint():
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def marginal_effects(entries):
    by_mask = {
        tuple(row["mechanisms"][key] for key in ("thresholds", "quantities", "periods")): row
        for row in entries
    }
    if set(by_mask) != set(MECHANISMS.values()):
        raise ValueError("A mechanism study requires all eight independent combinations")
    effects = []
    for index, mechanism in enumerate(("thresholds", "quantities", "periods")):
        pairs = []
        for mask, baseline in by_mask.items():
            if mask[index]:
                continue
            enabled = list(mask)
            enabled[index] = True
            changed = by_mask[tuple(enabled)]
            pairs.append(
                {
                    "baseline_variant": baseline["variant"],
                    "enabled_variant": changed["variant"],
                    "profit_change": changed["summary"]["total_profit"]
                    - baseline["summary"]["total_profit"],
                    "return_change": changed["summary"]["return_ratio"]
                    - baseline["summary"]["return_ratio"],
                    "drawdown_change": changed["summary"]["max_drawdown"]
                    - baseline["summary"]["max_drawdown"],
                }
            )
        effects.append(
            {
                "mechanism": mechanism,
                "mean_profit_change": sum(row["profit_change"] for row in pairs) / len(pairs),
                "mean_return_change": sum(row["return_change"] for row in pairs) / len(pairs),
                "mean_drawdown_change": sum(row["drawdown_change"] for row in pairs) / len(pairs),
                "pairs": pairs,
            }
        )
    return effects


def compact(result):
    return {
        "run_id": result["run_id"],
        "variant": result["config"]["variant"],
        "mechanisms": result["mechanisms"],
        "summary": result["summary"],
        "benchmarks": result["benchmarks"],
    }


def run_study(frame, config: ResearchConfig, directory: Path, study_id=None, progress=None):
    data = validate_minutes(frame).reset_index(drop=True)
    study_id = study_id or str(uuid4())
    frozen_rules = rule_fingerprint()
    target = directory / study_id
    target.mkdir(parents=True, exist_ok=True)
    details = target / "results"
    details.mkdir(exist_ok=True)
    boundaries = [data.date.iloc[0]]
    while boundaries[-1] + pd.DateOffset(months=6) <= data.date.iloc[-1] + pd.Timedelta("1min"):
        boundaries.append(boundaries[-1] + pd.DateOffset(months=3))
    if len(boundaries) > 1:
        boundaries.append(data.date.iloc[-1] + pd.Timedelta("1min"))
    segments = [
        (a, b)
        for a, b in pairwise(boundaries)
        if b > a and len(data.loc[(data.date >= a) & (data.date < b)]) > config.volatility_window
    ]
    # Eight mechanisms and chronological resets; past observed history is not a
    # genuinely unseen holdout. No coefficient is fitted or selected here.
    pressures = [
        ("zero_earn", {"earn_model": "none"}),
        ("half_apr", {"trx_apr": config.trx_apr / 2, "usdgo_apr": config.usdgo_apr / 2}),
        ("higher_spot_costs", {"buy_fee": 0.002, "sell_fee": 0.002, "slippage": 0.001}),
        (
            "fx_costs",
            {
                "earn_model": "net_benefit_hourly",
                "fx_fee": 0.0005,
                "fx_spread": 0.002,
                "fx_slippage": 0.0005,
            },
        ),
        ("redemption_wait", {"earn_model": "net_benefit_hourly", "redemption_delay_minutes": 60}),
        (
            "usdgo_deviation",
            {
                "earn_model": "net_benefit_hourly",
                "fx_shock_after_minutes": len(data) // 2,
                "fx_shock_rate": 0.97,
            },
        ),
    ]
    total = len(EXPERIMENTS) * (1 + len(segments)) + len(pressures)
    completed = 0

    def execute(bars, parameters):
        nonlocal completed
        result_id = str(uuid4())
        result = replay(
            bars,
            parameters,
            progress=lambda fraction: (
                progress((completed + fraction) / total) if progress else None
            ),
            diagnostic_path=details / f"{result_id}.diagnostics.sqlite",
        )
        result["run_id"] = result_id
        path = details / f"{result['run_id']}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(path)
        completed += 1
        if progress:
            progress(completed / total)
        return compact(result)

    annual = [
        execute(data, config.model_copy(update={"variant": variant})) for variant in EXPERIMENTS
    ]
    windows = []
    for start, end in segments:
        subset = data.loc[(data.date >= start) & (data.date < end)].reset_index(drop=True)
        entries = [
            execute(subset, config.model_copy(update={"variant": variant}))
            for variant in EXPERIMENTS
        ]
        windows.append(
            {
                "start_utc": str(subset.date.iloc[0]),
                "end_utc": str(subset.date.iloc[-1]),
                "candles": len(subset),
                "data_fingerprint": data_fingerprint(subset),
                "variants": entries,
                "effects": marginal_effects(entries),
            }
        )
    stressed = []
    for name, updates in pressures:
        settings = config.model_copy(update={**updates, "variant": "dynamic_period"})
        stressed.append({"scenario": name, "changes": updates, **execute(data, settings)})
    if rule_fingerprint() != frozen_rules:
        raise ValueError("Research source changed during the study; rerun on frozen rules")
    manifest = {
        "study_id": study_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "complete",
        "rule_version": config.version,
        "rule_fingerprint": frozen_rules,
        "parameters": config.model_dump(exclude={"variant"}),
        "start_utc": str(data.date.iloc[0]),
        "end_utc": str(data.date.iloc[-1]),
        "candles": len(data),
        "data_fingerprint": data_fingerprint(data),
        "protocol": {
            "coefficient_search": False,
            "known_history": True,
            "chronological_resets": True,
            "interpretation": (
                "Retrospective component and stress diagnosis on already observed history; "
                "not an unseen holdout or a live approval."
            ),
        },
        "variants": annual,
        "effects": marginal_effects(annual),
        "windows": windows,
        "stress": stressed,
    }
    temporary = target / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, allow_nan=False))
    temporary.replace(target / "manifest.json")
    return manifest


def list_studies(directory):
    if not directory.exists():
        return []
    studies = []
    for path in sorted(
        directory.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        result = json.loads(path.read_text())
        studies.append(
            {
                key: result[key]
                for key in (
                    "study_id",
                    "created_at",
                    "status",
                    "rule_version",
                    "start_utc",
                    "end_utc",
                    "candles",
                )
            }
        )
    return studies
