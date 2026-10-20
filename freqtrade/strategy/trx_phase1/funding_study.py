"""Paired funding experiments freeze the old dynamic-quantity trading rules."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.funding_preparation import FundingTerms
from freqtrade.strategy.trx_phase1.research_metrics import read_json
from freqtrade.strategy.trx_phase1.risk_study import execute_case
from freqtrade.strategy.trx_phase1.studies import rule_fingerprint


GROUP = "funding_preparation"
SCENARIOS = {
    "stable_60": {
        "usdt_redemption_delay_minutes": 60,
        "redemption_delay_minutes": 60,
        "trx_redemption_delay_minutes": 1,
    },
    "trx_60": {"trx_redemption_delay_minutes": 60},
    "both_60": {
        "usdt_redemption_delay_minutes": 60,
        "redemption_delay_minutes": 60,
        "trx_redemption_delay_minutes": 60,
    },
    "pressure_60": {
        "usdt_redemption_delay_minutes": 60,
        "redemption_delay_minutes": 60,
        "trx_redemption_delay_minutes": 60,
        "apr_scenario_scale": 0.5,
        "trx_earn_cap": 1000,
        "usdgo_earn_cap": 500,
    },
}


def funding_cases(source, saved):
    cfg = ResearchConfig.from_saved(saved["config"])
    # Fixed before any new run: 90 complete days after the first candle open.
    switch = int((pd.Timestamp(source["start_utc"]) + pd.Timedelta(days=90)).timestamp() * 1000)
    cases = [
        {
            "case": "funding_ordinary_control",
            "scenario": "ordinary_snapshot",
            "config": cfg,
            "terms": FundingTerms(),
            "compare_to": None,
        }
    ]
    for regime in ("from_cash", "after_90_days"):
        for scenario, changes in SCENARIOS.items():
            for enabled in (False, True):
                key = f"{regime}_{scenario}_{'prepared' if enabled else 'control'}"
                terms = FundingTerms(
                    enabled=enabled,
                    switch_at=switch if regime == "after_90_days" else None,
                    environment_changes=changes,
                )
                cases.append(
                    {
                        "case": key,
                        "scenario": f"{regime}_{scenario}",
                        "config": cfg,
                        "terms": terms,
                        "compare_to": f"{regime}_{scenario}_control" if enabled else None,
                    }
                )
    return cases


def run_funding_study(frame, source, saved, root, experiment_id, verify, progress=None, workers=2):
    frame = validate_minutes(frame).reset_index(drop=True)
    if data_fingerprint(frame) != source["data_fingerprint"]:
        raise ValueError("Funding study data differs from saved source")
    directory = Path(root) / "experiments" / experiment_id
    cases = funding_cases(source, saved)
    fingerprint, completed = rule_fingerprint(), {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(
                execute_case,
                frame,
                row["config"].model_dump(),
                directory / "results",
                row["case"],
                GROUP,
                row["case"],
                row["terms"].model_dump(),
            ): row
            for row in cases
        }
        for future in as_completed(pending):
            definition, brief = pending[future], future.result()
            saved_path = directory / "results" / f"{brief['run_id']}.json"
            result = read_json(saved_path)
            if definition["case"] == "funding_ordinary_control":
                verify(saved, result)
            funding = result["funding_research"]
            brief.update(
                funding_terms=definition["terms"].model_dump(),
                funding_transition=funding["transition"],
                funding_events=len(funding["preparation"]["events"]),
            )
            completed[definition["case"]] = brief
            if progress:
                progress(len(completed) / len(cases))
    entries = []
    for definition in cases:
        row = completed[definition["case"]]
        control = completed[definition["compare_to"]] if definition["compare_to"] else row
        if definition["compare_to"] and row["funding_transition"] != control["funding_transition"]:
            raise ValueError("Prepared/control continuous states differ at the funding transition")
        row.update(
            case=definition["case"],
            label=definition["case"],
            value=definition["case"],
            baseline=definition["compare_to"] is None,
            compare_to=definition["compare_to"],
            scenario=definition["scenario"],
            source_run_id=saved["run_id"],
            return_change=row["summary"]["return_ratio"] - control["summary"]["return_ratio"],
            drawdown_change=row["summary"]["max_drawdown"] - control["summary"]["max_drawdown"],
            earn_value_change=row["summary"]["earn_income_market_value"]
            - control["summary"]["earn_income_market_value"],
        )
        entries.append(row)
    if rule_fingerprint() != fingerprint:
        raise ValueError("Research source changed during funding study")
    manifest = {
        "experiment_id": experiment_id,
        "group": GROUP,
        "stage": "observed_history",
        "created_at": datetime.now(UTC).isoformat(),
        "source_experiment_id": source["experiment_id"],
        "source_run_id": saved["run_id"],
        "source_config": saved["config"],
        "start_utc": source["start_utc"],
        "end_utc": source["end_utc"],
        "candles": len(frame),
        "data_fingerprint": source["data_fingerprint"],
        "rules_sha256": fingerprint,
        "rate_snapshot": source["rate_snapshot"],
        "entries": entries,
        "benchmarks": [],
        "notes": [
            "Old dynamic quantity trading parameters frozen; funding-only enabled/control pairs.",
            "Four scenarios from cash and after 90 continuous days; no ledger/anchor reset.",
            "The ordinary control must exactly reproduce the saved source.",
            (
                "Each side: one intent, bounded by its source budget/qualifying principal; "
                "240-minute ready hold and 60-minute cooldown. No expired order execution."
            ),
            "APR is the frozen scenario, not historical actual rates; no account activation.",
            "Judge total-equity changes alongside Earn, FX costs, drawdowns and occupation.",
        ],
    }
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, allow_nan=False))
    temporary.replace(directory / "manifest.json")
    return manifest
