"""Replay saved legacy rule bundles against a frozen, already-published environment."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from freqtrade.strategy.trx_phase1.benchmarks import fair_benchmarks
from freqtrade.strategy.trx_phase1.candles import data_fingerprint, validate_minutes
from freqtrade.strategy.trx_phase1.config import MECHANISMS, ResearchConfig
from freqtrade.strategy.trx_phase1.protocol import CASE_LABELS, _execute
from freqtrade.strategy.trx_phase1.studies import rule_fingerprint


GROUP = "legacy_environment"
CURRENT_CASES = ("buy_06", "buy_08", "inventory_08", "combined_08")
VARIANT_LABELS = {
    "fixed": "Fixed periods, thresholds and fractions",
    "dynamic_threshold": "Dynamic thresholds only",
    "dynamic_quantity": "Dynamic quantities only",
    "dynamic_period_only": "Dynamic periods only",
    "dynamic_size": "Dynamic thresholds and quantities",
    "dynamic_threshold_period": "Dynamic thresholds and periods",
    "dynamic_quantity_period": "Dynamic quantities and periods",
    "dynamic_period": "Dynamic thresholds, quantities and periods",
}
# Capital, start/warmup, market constraints and all cash/Earn conditions are common.
# Price thresholds, sizing, period selection and the new rule switches remain per bundle.
ENVIRONMENT_FIELDS = (
    "version",
    "initial_cash",
    "startup",
    "volatility_window",
    "buy_fee",
    "sell_fee",
    "fee_rebate_ratio",
    "rebate_settlement_hour",
    "rebate_utc_offset_minutes",
    "slippage",
    "min_notional",
    "min_quantity",
    "quantity_decimals",
    "execution_margin",
    "maximum_execution_deviation",
    "earn_model",
    "trx_apr",
    "usdgo_apr",
    "usdt_apr",
    "trx_apr_tiers",
    "usdgo_apr_tiers",
    "usdt_apr_tiers",
    "apr_source",
    "apr_observed_at_utc",
    "apr_scenario_scale",
    "cash_forecast_hours",
    "cash_min_switch_value",
    "cash_switch_margin",
    "cash_min_hold_minutes",
    "fx_rate",
    "fx_spread",
    "fx_fee",
    "fx_slippage",
    "fx_quantity_decimals",
    "fx_deviation_limit",
    "redemption_delay_minutes",
    "trx_redemption_delay_minutes",
    "usdt_redemption_delay_minutes",
    "fx_quote_available",
    "fx_shock_after_minutes",
    "fx_shock_rate",
    "trx_product_available",
    "usdgo_product_available",
    "trx_earn_cap",
    "usdgo_earn_cap",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_sources(root, reference_id, legacy_id=None):  # noqa: C901 - Validate both archived sources.
    root = Path(root)
    reference_path = root / "experiments" / str(UUID(str(reference_id))) / "manifest.json"
    reference = read_json(reference_path)
    if reference["group"] == GROUP:
        reference_id = reference["reference_experiment_id"]
        reference_path = root / "experiments" / str(UUID(reference_id)) / "manifest.json"
        reference = read_json(reference_path)
    if reference["group"] != "focused_protocol":
        raise ValueError("Choose a published focused-protocol environment")
    by_case = {row["case"]: row for row in reference["entries"]}
    if not {*CURRENT_CASES, "pressure_08", "pressure_combined"} <= set(by_case):
        raise ValueError("The saved environment lacks the required current controls")
    for key in (*CURRENT_CASES, "pressure_08", "pressure_combined"):
        row = by_case[key]
        detail = read_json(reference_path.parent / "results" / f"{UUID(row['run_id'])}.json")
        if detail["data_fingerprint"] != reference["data_fingerprint"]:
            raise ValueError("Current reference data fingerprints differ")
        if detail["config"] != row["config"] or detail["summary"] != row["summary"]:
            raise ValueError("Current reference manifest and detail differ")
    candidates = (
        [root / "studies" / str(UUID(str(legacy_id))) / "manifest.json"]
        if legacy_id
        else sorted(
            (root / "studies").glob("*/manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )
    for path in candidates:
        legacy = read_json(path)
        if legacy.get("data_fingerprint") != reference["data_fingerprint"]:
            continue
        rows = legacy.get("variants", [])
        if len(rows) != len(MECHANISMS) or {r["variant"] for r in rows} != set(MECHANISMS):
            continue
        saved = {}
        for row in rows:
            detail = read_json(path.parent / "results" / f"{UUID(row['run_id'])}.json")
            if detail["data_fingerprint"] != reference["data_fingerprint"]:
                raise ValueError("Legacy detail has a different data fingerprint")
            if detail["config"]["variant"] != row["variant"]:
                raise ValueError("Legacy variant and saved configuration differ")
            if detail["summary"] != row["summary"]:
                raise ValueError("Legacy summary and saved detail differ")
            saved[row["variant"]] = detail
        return reference, legacy, saved
    raise ValueError("No complete eight-rule legacy study uses this exact saved data interval")


def study_cases(reference, legacy_results):
    by_case = {row["case"]: row for row in reference["entries"]}
    cases = []
    for scenario, profile_key in (
        ("ordinary_snapshot", "buy_08"),
        ("conservative_pressure", "pressure_08"),
    ):
        prefix = "" if scenario == "ordinary_snapshot" else "pressure_"
        profile = ResearchConfig.from_saved(by_case[profile_key]["config"]).model_dump()
        common = {key: profile[key] for key in ENVIRONMENT_FIELDS}
        for variant in MECHANISMS:
            source = legacy_results[variant]
            original = ResearchConfig.from_saved(source["config"]).model_dump()
            cfg = ResearchConfig(**{**original, **common})
            if (
                cfg.execution_preflight
                or cfg.buy_reference_recovery
                or cfg.sell_quantity_basis != "principal"
            ):
                raise ValueError("Legacy source already contains the new rule switches")
            label = "Legacy rules: " + VARIANT_LABELS[variant]
            cases.append(
                {
                    "case": prefix + "legacy_" + variant,
                    "scenario": scenario,
                    "config": cfg,
                    "family": "legacy",
                    "label": label,
                    "source_run_id": source["run_id"],
                    "original_config": source["config"],
                    "original_summary": source["summary"],
                }
            )
        for key in CURRENT_CASES:
            row = by_case[key]
            original = ResearchConfig.from_saved(row["config"]).model_dump()
            cfg = ResearchConfig(**{**original, **common})
            expected = (
                row["run_id"]
                if not prefix
                else {
                    "buy_08": by_case["pressure_08"]["run_id"],
                    "combined_08": by_case["pressure_combined"]["run_id"],
                }.get(key)
            )
            cases.append(
                {
                    "case": prefix + key,
                    "scenario": scenario,
                    "config": cfg,
                    "family": "current",
                    "label": CASE_LABELS[key],
                    "source_run_id": row["run_id"],
                    "verification_run_id": expected,
                    "original_config": row["config"],
                    "original_summary": row["summary"],
                }
            )
        for row in cases:
            if row["scenario"] == scenario and not all(
                row["config"].model_dump()[key] == common[key] for key in ENVIRONMENT_FIELDS
            ):
                raise ValueError("A rule bundle has different common environment parameters")
    return cases


def pareto(entries):
    eligible = [row for row in entries if not row["cash_only"]]
    front = [
        row["case"]
        for row in eligible
        if not any(
            other["summary"]["return_ratio"] >= row["summary"]["return_ratio"]
            and other["summary"]["max_drawdown"] <= row["summary"]["max_drawdown"]
            and (
                other["summary"]["return_ratio"] > row["summary"]["return_ratio"]
                or other["summary"]["max_drawdown"] < row["summary"]["max_drawdown"]
            )
            for other in eligible
        )
    ]
    return {"front": front, "excluded_cash_only": [r["case"] for r in entries if r["cash_only"]]}


def run_environment_study(
    frame,
    reference,
    legacy,
    legacy_results,
    root,
    experiment_id,
    verify,
    progress=None,
    collect_diagnostics=True,
    workers=2,
):
    data = validate_minutes(frame).reset_index(drop=True)
    if data_fingerprint(data) != reference["data_fingerprint"]:
        raise ValueError("Replay data differs from the frozen environment")
    cases = study_cases(reference, legacy_results)
    directory = Path(root) / "experiments" / experiment_id
    details = directory / "results"
    details.mkdir(parents=True, exist_ok=True)
    rules = rule_fingerprint()
    completed = {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(
                _execute,
                data,
                row["config"].model_dump(),
                details,
                str(uuid4()),
                collect_diagnostics,
                row["case"],
                GROUP,
                ("Pressure environment: " if row["scenario"] == "conservative_pressure" else "")
                + row["label"],
            ): row
            for row in cases
        }
        for future in as_completed(pending):
            row = pending[future]
            brief = future.result()
            path = details / f"{brief['run_id']}.json"
            replayed = read_json(path)
            if row["family"] == "current" and row["verification_run_id"]:
                saved = read_json(
                    Path(root)
                    / "experiments"
                    / reference["experiment_id"]
                    / "results"
                    / f"{row['verification_run_id']}.json"
                )
                if ResearchConfig.from_saved(saved["config"]) != row["config"]:
                    raise ValueError("Current replay changed saved rules or environment")
                verify(saved, replayed)
            path_hash = sha256(
                json.dumps(
                    {k: replayed[k] for k in ("orders", "fills", "equity_curve")},
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            source_info = {
                k: row[k]
                for k in ("family", "source_run_id", "original_config", "original_summary")
            }
            replayed["protocol"].update(
                source_info,
                reference_experiment_id=reference["experiment_id"],
                legacy_study_id=legacy["study_id"],
                economic_path_sha256=path_hash,
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(replayed, allow_nan=False))
            temporary.replace(path)
            brief.update(
                source_info,
                economic_path_sha256=path_hash,
                case=row["case"],
                scenario=row["scenario"],
                family=row["family"],
            )
            completed[row["case"]] = brief
            if progress:
                progress(len(completed) / (len(cases) + 2))
    entries, benchmarks, fronts = [], [], {}
    for scenario, prefix in (("ordinary_snapshot", ""), ("conservative_pressure", "pressure_")):
        baseline_key = prefix + "buy_08"
        baseline = completed[baseline_key]
        rows = []
        for definition in cases:
            if definition["scenario"] != scenario:
                continue
            row = completed[definition["case"]]
            row.update(
                value=row["case"],
                label=row["case"],
                baseline=row["case"] == baseline_key,
                compare_to=None if row["case"] == baseline_key else baseline_key,
                return_change=row["summary"]["return_ratio"] - baseline["summary"]["return_ratio"],
                drawdown_change=row["summary"]["max_drawdown"]
                - baseline["summary"]["max_drawdown"],
            )
            rows.append(row)
        entries.extend(rows)
        fronts[scenario] = pareto(rows)
        passive = fair_benchmarks(
            data, ResearchConfig(**baseline["config"]), directory / "benchmarks" / scenario
        )
        for row in passive:
            row["scenario"] = scenario
        benchmarks.extend(passive)
        if progress:
            progress((len(cases) + len(benchmarks) / 4) / (len(cases) + 2))
    if rule_fingerprint() != rules:
        raise ValueError("Research source changed during the matched environment study")
    manifest = {
        "experiment_id": experiment_id,
        "group": GROUP,
        "stage": "observed_history",
        "created_at": datetime.now(UTC).isoformat(),
        "reference_experiment_id": reference["experiment_id"],
        "legacy_study_id": legacy["study_id"],
        "start_utc": str(data.date.min()),
        "end_utc": str(data.date.max()),
        "candles": len(data),
        "data_fingerprint": reference["data_fingerprint"],
        "rules_sha256": rules,
        "rate_snapshot": reference["rate_snapshot"],
        "environment_fields": ENVIRONMENT_FIELDS,
        "entries": entries,
        "benchmarks": benchmarks,
        "pareto": fronts,
        "notes": [
            (
                "Eight saved legacy rule bundles and four current alternatives per environment; "
                "no parameter search."
            ),
            (
                "All rows within an environment share the exact data, capital, warmup, costs, "
                "rebates and cash/Earn conditions."
            ),
            (
                "Legacy trading rules retain their saved thresholds, sizing, periods and switches; "
                "the new switches stay disabled."
            ),
            (
                "Current controls are replayed and must exactly reproduce saved executions "
                "and economics before publication."
            ),
            "Original summaries are provenance only, not same-condition comparison figures.",
            (
                "This is observed-history bundle comparison, not a single-factor effect estimate, "
                "unseen test or automatic rule selection."
            ),
            (
                "Cash-only outcomes are excluded from the return/drawdown frontier; "
                "APR remains the frozen 2026-10-13 scenario."
            ),
        ],
    }
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, allow_nan=False))
    temporary.replace(directory / "manifest.json")
    return manifest
