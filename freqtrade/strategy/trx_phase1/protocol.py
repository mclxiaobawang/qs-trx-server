"""A small, predeclared study for allocation, recovery and profitable inventory."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from freqtrade.strategy.trx_phase1.benchmarks import fair_benchmarks
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.diagnostics import overview
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.studies import rule_fingerprint


CASE_LABELS = {
    "control_12": "Common terms, 12% buy allocation",
    "preflight_12": "P0: preflight, 12% buy allocation",
    "buy_08": "P1: 8% buy allocation",
    "buy_06": "P1: 6% buy allocation",
    "recovery_08": "P2: 8% with restricted recovery",
    "inventory_08": "P3: 8% with profitable inventory sizing",
    "combined_08": "Combined recovery and inventory experiment",
    "pressure_08": "Pressure baseline, 8% allocation",
    "pressure_combined": "Pressure combined experiment",
}


def protocol_parameters(saved, snapshot_path):
    raw = Path(snapshot_path).read_bytes()
    snapshot = json.loads(raw)
    products = {row["coin"]: row for row in snapshot["ordinary_flexible_products"]}
    values = {
        **ResearchConfig.from_saved(saved["config"]).model_dump(),
        "version": "trx-phase1-v0.5",
        "earn_model": "net_benefit_hourly",
        "apr_source": snapshot["source_page"] + "#snapshot-sha256=" + sha256(raw).hexdigest(),
        "apr_observed_at_utc": snapshot["captured_at_utc"],
        "apr_scenario_scale": 1,
        "trx_apr": products["TRX"]["displayed_apr"],
        "usdgo_apr": products["USDGO"]["tiers"][0]["apr"],
        "usdt_apr": products["USDT"]["tiers"][0]["apr"],
        "trx_apr_tiers": [],
        "usdgo_apr_tiers": [(r["quantity_to"], r["apr"]) for r in products["USDGO"]["tiers"]],
        "usdt_apr_tiers": [(r["quantity_to"], r["apr"]) for r in products["USDT"]["tiers"]],
        "fx_fee": 0.0005,
        "fx_spread": 0.0005,
        "fx_slippage": 0.0002,
        "redemption_delay_minutes": 1,
        "usdt_redemption_delay_minutes": 1,
        "trx_earn_cap": 5000,
        "usdgo_earn_cap": 1000,
        "trx_product_available": True,
        "usdgo_product_available": True,
        "fx_quote_available": True,
        "fx_rate": 1,
        "fx_shock_after_minutes": 0,
        "fx_shock_rate": 0,
        "buy_fraction": 0.12,
        "execution_preflight": False,
        "buy_reference_recovery": False,
        "recovery_stagnation_days": 30,
        "recovery_valid_days": 7,
        "recovery_cooldown_days": 7,
        "recovery_buy_fraction": 0.02,
        "sell_quantity_basis": "principal",
        "sell_fraction_cap": 1,
    }
    return ResearchConfig(**values), snapshot


def protocol_cases(config):
    definitions = [
        ("control_12", None, {}),
        ("preflight_12", "control_12", {"execution_preflight": True}),
        ("buy_08", "preflight_12", {"buy_fraction": 0.08}),
        ("buy_06", "preflight_12", {"buy_fraction": 0.06}),
        ("recovery_08", "buy_08", {"buy_reference_recovery": True}),
        ("inventory_08", "buy_08", {"sell_quantity_basis": "profitable"}),
        (
            "combined_08",
            "buy_08",
            {"buy_reference_recovery": True, "sell_quantity_basis": "profitable"},
        ),
    ]
    configs, cases = {}, []
    for key, parent, changes in definitions:
        source = configs[parent] if parent else config
        cfg = ResearchConfig(**{**source.model_dump(), **changes})
        configs[key] = cfg
        cases.append((key, parent, "ordinary_snapshot", cfg))
    pressure = {
        "apr_scenario_scale": 0.5,
        "trx_earn_cap": 1000,
        "usdgo_earn_cap": 500,
        "redemption_delay_minutes": 60,
        "usdt_redemption_delay_minutes": 60,
    }
    for key, source, parent in [
        ("pressure_08", "buy_08", None),
        ("pressure_combined", "combined_08", "pressure_08"),
    ]:
        cases.append(
            (
                key,
                parent,
                "conservative_pressure",
                ResearchConfig(**{**configs[source].model_dump(), **pressure}),
            )
        )
    return cases


def _execute(
    frame,
    values,
    directory,
    result_id,
    collect_diagnostics,
    case,
    group="focused_protocol",
    label=None,
):
    cfg = ResearchConfig(**values)
    path = Path(directory) / f"{result_id}.json"
    result = replay(
        frame,
        cfg,
        diagnostic_path=path.with_suffix(".diagnostics.sqlite") if collect_diagnostics else None,
    )
    result["run_id"] = result_id
    result["experiment_id"] = Path(directory).parent.name
    result["protocol"] = {
        "case": case,
        "label": label or CASE_LABELS[case],
        "group": group,
        "stage": "observed_history",
    }
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, allow_nan=False))
    temp.replace(path)
    if collect_diagnostics:
        metrics = overview(path.with_suffix(".diagnostics.sqlite"))
        lots, amounts = metrics["lots_summary"], metrics["amounts"]
    else:
        end = int(frame.date.iloc[-1].timestamp() * 1000) + 60000
        lots = {
            "stale_cost_30_days": sum(
                float(row["remaining_cost"])
                for row in result["lots"]
                if float(row["quantity"]) > 0 and end - row["bought_at"] >= 30 * 86400000
            )
        }
        amounts = {
            side: {
                "total": sum(
                    row["quantity"] * row["price"] for row in result["fills"] if row["side"] == side
                )
            }
            for side in ("buy", "sell")
        }
    return {
        "run_id": result_id,
        "config": cfg.model_dump(),
        "summary": result["summary"],
        "cash_only": not result["fills"],
        "lots_summary": lots,
        "amounts": amounts,
    }


def run_protocol(
    frame,
    saved,
    root,
    experiment_id,
    snapshot_path,
    progress=None,
    collect_diagnostics=True,
    workers=2,
):
    config, snapshot = protocol_parameters(saved, snapshot_path)
    cases = protocol_cases(config)
    directory = Path(root) / experiment_id
    details = directory / "results"
    details.mkdir(parents=True, exist_ok=True)
    rules = rule_fingerprint()
    completed = {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {
            pool.submit(
                _execute, frame, cfg.model_dump(), details, str(uuid4()), collect_diagnostics, key
            ): key
            for key, _, _, cfg in cases
        }
        for future in as_completed(pending):
            completed[pending[future]] = future.result()
            if progress:
                progress(len(completed) / (len(cases) + 2))
    entries = []
    for key, parent, scenario, _ in cases:
        row = completed[key]
        reference = completed[parent] if parent else row
        row.update(
            case=key,
            label=key,
            baseline=parent is None,
            value=key,
            compare_to=parent,
            scenario=scenario,
            return_change=row["summary"]["return_ratio"] - reference["summary"]["return_ratio"],
            drawdown_change=row["summary"]["max_drawdown"] - reference["summary"]["max_drawdown"],
        )
        entries.append(row)
    benchmarks = []
    for scenario, cfg in [("ordinary_snapshot", config), ("conservative_pressure", cases[-2][3])]:
        rows = fair_benchmarks(frame, cfg, directory / "benchmarks" / scenario)
        for row in rows:
            row["scenario"] = scenario
        benchmarks.extend(rows)
        if progress:
            progress((len(cases) + len(benchmarks) / 4) / (len(cases) + 2))
    if rule_fingerprint() != rules:
        raise ValueError("Research source changed during focused protocol")
    manifest = {
        "experiment_id": experiment_id,
        "group": "focused_protocol",
        "stage": "observed_history",
        "baseline_run_id": saved["run_id"],
        "start_utc": str(frame.date.min()),
        "end_utc": str(frame.date.max()),
        "data_fingerprint": saved["data_fingerprint"],
        "rules_sha256": rules,
        "entries": entries,
        "benchmarks": benchmarks,
        "rate_snapshot": snapshot,
        "notes": [
            (
                "Ordinary public APR snapshot is a constant-rate historical scenario, "
                "not past actual rates."
            ),
            (
                "Finite Earn caps (5,000 TRX / 1,000 USDGO), 1-minute redemption waits "
                "and FX costs are explicit unverified research assumptions."
            ),
            (
                "A second paired pressure scenario halves APR, limits quotas and waits 60 minutes; "
                "it is not ranked against ordinary-snapshot runs."
            ),
            (
                "Seven predeclared strategy cases and one paired pressure scenario; "
                "no parameter search or automatic selection."
            ),
            (
                "Recovery is a separate buy reference; old lot costs and the actual shared fill "
                "anchor are preserved until a real simulated principal fill."
            ),
            (
                "Passive benchmark CSVs contain every minute; UI curves use hourly samples. "
                "Cash-only benchmarks flag an unexecuted initial TRX purchase."
            ),
        ],
    }
    temp = directory / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, allow_nan=False))
    temp.replace(directory / "manifest.json")
    return manifest
