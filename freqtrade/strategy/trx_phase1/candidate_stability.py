"""Two C1 neighbors and two independent funding pairs; no parameter search."""

import json
import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pandas as pd

from freqtrade.strategy.trx_phase1.candidate_attribution import explain_c1
from freqtrade.strategy.trx_phase1.candidate_audit import audit_stability
from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.funding_preparation import FundingTerms
from freqtrade.strategy.trx_phase1.funding_study import SCENARIOS
from freqtrade.strategy.trx_phase1.research_metrics import read_json
from freqtrade.strategy.trx_phase1.risk_study import execute_case
from freqtrade.strategy.trx_phase1.studies import rule_fingerprint


GROUP = "candidate_stability"
CONTEXT = ("C2", "C1", "legacy_dynamic_period_only")


def load_stability_source(root, experiment_id):
    directory = Path(root) / "experiments" / str(UUID(str(experiment_id)))
    manifest = read_json(directory / "manifest.json")
    if manifest["group"] == GROUP:
        return load_stability_source(root, manifest["risk_experiment_id"])
    if manifest["group"] != "risk_matched_rules":
        raise ValueError("Choose the completed C1 rule-study batch")
    entries = {row["case"]: row for row in manifest["entries"]}
    saved = {}
    for key in (*CONTEXT, "legacy_dynamic_quantity"):
        row = entries[key]
        path = directory / "results" / f"{UUID(row['run_id'])}.json"
        result = read_json(path)
        if (
            result["summary"] != row["summary"]
            or result["config"] != row["config"]
            or result["data_fingerprint"] != manifest["data_fingerprint"]
        ):
            raise ValueError("Candidate source differs from its saved complete ledger")
        if not path.with_suffix(".diagnostics.sqlite").is_file():
            raise ValueError("Candidate source is missing its full-minute diagnostic package")
        with sqlite3.connect(path.with_suffix(".diagnostics.sqlite")) as db:
            metadata = json.loads(
                db.execute("SELECT value FROM metadata WHERE key='overview'").fetchone()[0]
            )
        if (
            metadata["data_fingerprint"] != result["data_fingerprint"]
            or metadata["sample_count"] != result["candles"]
            or metadata["parameters"] != result["config"]
        ):
            raise ValueError("Candidate minute package differs from its saved ledger")
        saved[key] = result
    c1 = ResearchConfig.from_saved(saved["C1"]["config"])
    old = ResearchConfig.from_saved(saved["legacy_dynamic_period_only"]["config"])
    if c1.buy_fraction != 0.09 or old.buy_fraction != 0.12:
        raise ValueError("This protocol requires the saved 9% C1 and 12% old-period source")
    if {k for k in c1.model_dump() if c1.model_dump()[k] != old.model_dump()[k]} != {
        "buy_fraction"
    }:
        raise ValueError("C1/source differ in fields other than buy allocation")
    return manifest, saved


def stability_cases(source, saved):
    c1 = ResearchConfig.from_saved(saved["C1"]["config"])
    funding = ResearchConfig.from_saved(saved["legacy_dynamic_quantity"]["config"])
    if (
        funding.redemption_delay_minutes,
        funding.usdt_redemption_delay_minutes,
        funding.trx_redemption_delay,
    ) != (1, 1, 1):
        raise ValueError("Normal funding pair requires the saved one-minute environment")
    switch = int((pd.Timestamp(source["start_utc"]) + pd.Timedelta(days=180)).timestamp() * 1000)
    if pd.Timestamp(source["end_utc"]).timestamp() * 1000 <= switch:
        raise ValueError("The continuous source interval does not reach day 180")
    rows = [
        {
            "case": key,
            "scenario": "c1_neighbors",
            "config": ResearchConfig(**{**c1.model_dump(), "buy_fraction": fraction}),
            "terms": None,
            "compare_to": "C1",
            "source_case": "C1",
        }
        for key, fraction in (("C_08", 0.08), ("C_10", 0.10))
    ]
    for scenario, at, changes in (
        ("normal_funding", None, {}),
        ("after_180_days_both_60", switch, SCENARIOS["both_60"]),
    ):
        for enabled in (False, True):
            rows.append(
                {
                    "case": f"{scenario}_{'prepared' if enabled else 'control'}",
                    "scenario": scenario,
                    "config": funding,
                    "terms": FundingTerms(
                        enabled=enabled, switch_at=at, environment_changes=changes
                    ),
                    "compare_to": f"{scenario}_control" if enabled else None,
                    "source_case": "legacy_dynamic_quantity",
                }
            )
    return rows


def equal_diagnostics(left, right, stop=None):
    """Compare actual minute and operation rows in bounded memory."""
    counts = {}
    with sqlite3.connect(left) as a, sqlite3.connect(right) as b:
        for table in ("samples", "risk_samples", "records"):
            where = " WHERE timestamp <= ?" if stop is not None else ""
            # At a funding transition, operations at the boundary can already
            # reflect the new environment; close-time samples precede the switch.
            if stop is not None and table == "records":
                # Lots, opportunities and drawdown objects are finalized later
                # and can contain post-switch outcomes despite an earlier start.
                where = (
                    " WHERE timestamp < ? AND kind IN "
                    "('operations','fills','anchors','period_switches')"
                    " AND (kind != 'operations' OR "
                    "coalesce(json_extract(payload,'$.paid_at'),timestamp) <= ?)"
                )
            query = f"SELECT * FROM {table}{where} ORDER BY rowid"
            args = (
                (stop, stop)
                if stop is not None and table == "records"
                else ((stop,) if stop is not None else ())
            )
            x, y = a.execute(query, args), b.execute(query, args)
            count = 0
            while True:
                first, second = x.fetchmany(10000), y.fetchmany(10000)
                if first != second:
                    raise ValueError(f"Paired {table} paths differ in the required common interval")
                count += len(first)
                if not first:
                    break
            counts[table] = count
    return counts


def run_stability_study(  # noqa: C901
    frame, source, saved, root, experiment_id, verify, progress=None, workers=2
):
    frame = validate_minutes(frame).reset_index(drop=True)
    if data_fingerprint(frame) != source["data_fingerprint"]:
        raise ValueError("Candidate stability data differs from its saved source")
    directory = Path(root) / "experiments" / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    definitions = stability_cases(source, saved)
    fingerprint = rule_fingerprint()
    protocol = {
        "created_at": datetime.now(UTC).isoformat(),
        "risk_experiment_id": source["experiment_id"],
        "data_fingerprint": source["data_fingerprint"],
        "rules_sha256": fingerprint,
        "new_cases": [
            {
                **{k: v for k, v in row.items() if k not in {"config", "terms"}},
                "config": row["config"].model_dump(),
                "funding_terms": row["terms"].model_dump() if row["terms"] else None,
            }
            for row in definitions
        ],
        "stage": "observed_history",
    }
    (directory / "protocol.json").write_text(json.dumps(protocol, allow_nan=False))
    source_dir = Path(root) / "experiments" / source["experiment_id"] / "results"
    attribution = explain_c1(
        source_dir / f"{saved['legacy_dynamic_period_only']['run_id']}.json",
        source_dir / f"{saved['C1']['run_id']}.json",
        frame,
        directory / "analysis",
    )
    completed = {}
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
                row["terms"].model_dump() if row["terms"] else None,
            ): row
            for row in definitions
        }
        for future in as_completed(pending):
            definition, brief = pending[future], future.result()
            path = directory / "results" / f"{brief['run_id']}.json"
            result = read_json(path)
            if definition["case"] == "normal_funding_control":
                verify(saved["legacy_dynamic_quantity"], result)
            if definition["terms"]:
                funding = result["funding_research"]
                brief.update(
                    funding_terms=definition["terms"].model_dump(),
                    funding_transition=funding["transition"],
                    funding_events=len(funding["preparation"]["events"]),
                )
            completed[definition["case"]] = brief
            if progress:
                progress(len(completed) / 7)
    normal_left, normal_right = (
        completed[f"normal_funding_{key}"] for key in ("control", "prepared")
    )
    normal_results = [
        read_json(directory / "results" / f"{row['run_id']}.json")
        for row in (normal_left, normal_right)
    ]
    verify(*normal_results)
    for field in (
        "summary",
        "interest_receipts",
        "cash_conversions",
        "cash_forecasts",
        "lots",
        "rebate_receipts",
    ):
        if normal_results[0][field] != normal_results[1][field]:
            raise ValueError(f"Normal funding preparation changes {field}")
    if normal_right["funding_events"] or any(
        normal_results[1]["funding_research"]["preparation"]["intents"].values()
    ):
        raise ValueError("Normal funding preparation unnecessarily intervenes")

    def db_path(row):
        return directory / "results" / f"{row['run_id']}.diagnostics.sqlite"

    normal_equality = equal_diagnostics(db_path(normal_left), db_path(normal_right))
    held_left, held_right = (
        completed[f"after_180_days_both_60_{key}"] for key in ("control", "prepared")
    )
    transition = held_left["funding_transition"]
    if (
        transition != held_right["funding_transition"]
        or not transition
        or transition["principal_trx"] <= 0
    ):
        raise ValueError("Day-180 funding pair does not inherit the same held principal state")
    prefix_equality = equal_diagnostics(
        db_path(held_left), db_path(held_right), transition["timestamp"]
    )
    entries = []
    for key in CONTEXT:
        original = next(row for row in source["entries"] if row["case"] == key)
        entries.append(
            {
                **original,
                "scenario": "c1_neighbors",
                "source_experiment_id": source["experiment_id"],
                "reused_saved_result": True,
                "compare_to": "C1" if key != "C1" else None,
                "return_change": original["summary"]["return_ratio"]
                - saved["C1"]["summary"]["return_ratio"],
                "drawdown_change": original["summary"]["max_drawdown"]
                - saved["C1"]["summary"]["max_drawdown"],
            }
        )
    for definition in definitions:
        row = completed[definition["case"]]
        baseline = (
            completed.get(
                definition["compare_to"], {"summary": saved[definition["source_case"]]["summary"]}
            )
            if definition["compare_to"]
            else row
        )
        row.update(
            case=definition["case"],
            label=definition["case"],
            value=definition["case"],
            scenario=definition["scenario"],
            baseline=definition["compare_to"] is None,
            compare_to=definition["compare_to"],
            source_run_id=saved[definition["source_case"]]["run_id"],
            return_change=row["summary"]["return_ratio"] - baseline["summary"]["return_ratio"],
            drawdown_change=row["summary"]["max_drawdown"] - baseline["summary"]["max_drawdown"],
            earn_value_change=row["summary"]["earn_income_market_value"]
            - baseline["summary"]["earn_income_market_value"],
        )
        entries.append(row)
    entries.sort(key=lambda row: (row["scenario"], row["config"]["buy_fraction"], row["case"]))
    if rule_fingerprint() != fingerprint:
        raise ValueError("Research source changed during candidate validation")
    manifest = {
        "experiment_id": experiment_id,
        "group": GROUP,
        "stage": "observed_history",
        "created_at": datetime.now(UTC).isoformat(),
        "risk_experiment_id": source["experiment_id"],
        "source_experiment_id": source["source_experiment_id"],
        "start_utc": source["start_utc"],
        "end_utc": source["end_utc"],
        "candles": len(frame),
        "data_fingerprint": source["data_fingerprint"],
        "rules_sha256": fingerprint,
        "rate_snapshot": source["rate_snapshot"],
        "entries": entries,
        "benchmarks": [],
        "c1_attribution": attribution,
        "verification": {
            "normal_funding_full_path_equality": normal_equality,
            "normal_funding_preparation_events": 0,
            "day180_prefix_equality": prefix_equality,
        },
        "notes": [
            (
                "Only 8% and 10% are new rule parameters; 6%, 9% and 12% reuse complete "
                "saved continuous accounts."
            ),
            (
                "Funding uses only old dynamic quantity; normal one-minute pair and "
                "day-180 both-60 pair, hold 240 and cooldown 60 minutes."
            ),
            (
                "Day 180 and all six runs are declared before inspecting new results; "
                "no search or removal of adverse trades."
            ),
            (
                "Previously observed history and frozen quotation APR, not out-of-sample "
                "evidence or realized historical product rates."
            ),
            (
                "PRD defaults and paused account are unchanged; historical drawdown "
                "is not a future loss limit."
            ),
        ],
    }
    audit_stability(root, manifest)
    temp = directory / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, allow_nan=False))
    temp.replace(directory / "manifest.json")
    if progress:
        progress(1)
    return manifest
