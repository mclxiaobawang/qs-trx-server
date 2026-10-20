"""Six predeclared rule changes with frozen terms and three reproduced sources."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from freqtrade.strategy.trx_phase1.benchmarks import fair_benchmarks
from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.environment_study import ENVIRONMENT_FIELDS, pareto
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_metrics import StudyRecorder, read_json, risk_metrics
from freqtrade.strategy.trx_phase1.studies import rule_fingerprint


GROUP = "risk_matched_rules"
SOURCE_CASES = ("inventory_08", "legacy_dynamic_quantity", "legacy_dynamic_period_only")
LABELS = {
    "A1": "A1: 6% with profitable inventory sizing",
    "B1": "B1: legacy dynamic quantity, 75% buy allocation",
    "B2": "B2: legacy dynamic quantity, 50% buy allocation",
    "B3": "B3: legacy dynamic quantity, profitable inventory sizing",
    "C1": "C1: legacy dynamic period, 75% buy allocation",
    "C2": "C2: legacy dynamic period, 50% buy allocation",
}


def load_risk_source(root, experiment_id):
    directory = Path(root) / "experiments" / str(UUID(str(experiment_id)))
    source = read_json(directory / "manifest.json")
    if source["group"] != "legacy_environment":
        raise ValueError("Choose the completed matched legacy environment")
    rows = {row["case"]: row for row in source["entries"] if row["scenario"] == "ordinary_snapshot"}
    if not set(SOURCE_CASES) <= rows.keys():
        raise ValueError("Required saved rule sources are missing")
    saved = {}
    for key in SOURCE_CASES:
        row = rows[key]
        detail = read_json(directory / "results" / f"{UUID(row['run_id'])}.json")
        if (
            detail["summary"] != row["summary"]
            or detail["config"] != row["config"]
            or detail["data_fingerprint"] != source["data_fingerprint"]
        ):
            raise ValueError("Saved rule source and manifest differ")
        saved[key] = detail
    return source, saved


def risk_cases(saved):
    definitions = [
        ("A1", "inventory_08", "buy_fraction", 0.06),
        ("B1", "legacy_dynamic_quantity", "buy_fraction", None),
        ("B2", "legacy_dynamic_quantity", "buy_fraction", None),
        ("B3", "legacy_dynamic_quantity", "sell_quantity_basis", "profitable"),
        ("C1", "legacy_dynamic_period_only", "buy_fraction", None),
        ("C2", "legacy_dynamic_period_only", "buy_fraction", None),
    ]
    result = []
    for key in SOURCE_CASES:
        result.append(
            {
                "case": key,
                "compare_to": None,
                "source_run_id": saved[key]["run_id"],
                "config": ResearchConfig.from_saved(saved[key]["config"]),
            }
        )
    for key, parent, field, value in definitions:
        cfg = ResearchConfig.from_saved(saved[parent]["config"])
        if value is None:
            value = cfg.buy_fraction * (0.75 if key.endswith("1") else 0.5)
        changed = ResearchConfig(**{**cfg.model_dump(), field: value})
        result.append(
            {
                "case": key,
                "compare_to": parent,
                "changed_field": field,
                "source_run_id": saved[parent]["run_id"],
                "config": changed,
            }
        )
    common = result[0]["config"].model_dump()
    if not all(
        all(row["config"].model_dump()[k] == common[k] for k in ENVIRONMENT_FIELDS)
        for row in result
    ):
        raise ValueError("Rule experiment changed the common environment")
    return result


def execute_case(frame, config, directory, case, group, label, funding=None):
    cfg = ResearchConfig(**config)
    run_id = str(uuid4())
    path = Path(directory) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {}
    if funding is not None:
        from freqtrade.strategy.trx_phase1.funding_preparation import FundingFactory

        options["engine_factory"] = FundingFactory(funding)
    result = replay(
        frame,
        cfg,
        diagnostic_path=path.with_suffix(".diagnostics.sqlite"),
        recorder_factory=StudyRecorder,
        **options,
    )
    result.update(
        run_id=run_id,
        experiment_id=path.parent.parent.name,
        protocol={"case": case, "label": label, "group": group, "stage": "observed_history"},
    )
    risk, diagnosis = risk_metrics(path.with_suffix(".diagnostics.sqlite"), cfg.min_notional)
    result["risk_metrics"] = risk
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, allow_nan=False))
    temp.replace(path)
    return {
        "run_id": run_id,
        "config": result["config"],
        "summary": result["summary"],
        "cash_only": not result["fills"],
        "risk_metrics": risk,
        "lots_summary": diagnosis["lots_summary"],
        "amounts": diagnosis["amounts"],
    }


def run_risk_study(frame, source, saved, root, experiment_id, verify, progress=None, workers=2):
    frame = validate_minutes(frame).reset_index(drop=True)
    if data_fingerprint(frame) != source["data_fingerprint"]:
        raise ValueError("Rule study data differs from saved source")
    directory = Path(root) / "experiments" / experiment_id
    cases = risk_cases(saved)
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
                LABELS.get(row["case"], row["case"]),
            ): row
            for row in cases
        }
        for future in as_completed(pending):
            row, brief = pending[future], future.result()
            if row["compare_to"] is None:
                verify(
                    saved[row["case"]], read_json(directory / "results" / f"{brief['run_id']}.json")
                )
            completed[row["case"]] = brief
            if progress:
                progress(len(completed) / 10)
    entries = []
    for definition in cases:
        row = completed[definition["case"]]
        control = completed[definition["compare_to"]] if definition["compare_to"] else row
        row.update(
            {k: v for k, v in definition.items() if k != "config"},
            scenario="ordinary_snapshot",
            label=LABELS.get(definition["case"], definition["case"]),
            value=definition["case"],
            baseline=definition["compare_to"] is None,
            return_change=row["summary"]["return_ratio"] - control["summary"]["return_ratio"],
            drawdown_change=row["summary"]["max_drawdown"] - control["summary"]["max_drawdown"],
        )
        entries.append(row)
    benchmarks = fair_benchmarks(
        frame,
        cases[0]["config"],
        directory / "benchmarks/ordinary_snapshot",
        allocations=[("exposure_40", 0.4), ("exposure_70", 0.7)],
    )
    for row in benchmarks:
        row["scenario"] = "ordinary_snapshot"
    if rule_fingerprint() != fingerprint:
        raise ValueError("Research source changed during risk study")
    manifest = {
        "experiment_id": experiment_id,
        "group": GROUP,
        "stage": "observed_history",
        "created_at": datetime.now(UTC).isoformat(),
        "source_experiment_id": source["experiment_id"],
        "start_utc": source["start_utc"],
        "end_utc": source["end_utc"],
        "candles": len(frame),
        "data_fingerprint": source["data_fingerprint"],
        "rules_sha256": fingerprint,
        "rate_snapshot": source["rate_snapshot"],
        "entries": entries,
        "benchmarks": benchmarks,
        "pareto": {"ordinary_snapshot": pareto(entries)},
        "notes": [
            "Six predeclared single-field changes; three source controls reproduce exactly.",
            "Source buy fractions and all other switches are saved values, not UI inputs.",
            "40%/70% are initial passive allocations, not matched annual average exposure.",
            "Previously observed history; no parameter search or automatic account activation.",
        ],
    }
    temp = directory / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, allow_nan=False))
    temp.replace(directory / "manifest.json")
    if progress:
        progress(1)
    return manifest
