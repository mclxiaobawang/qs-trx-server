"""Small controlled comparisons, kept separate from the existing eight-mask study."""

import json
from pathlib import Path
from uuid import uuid4

from freqtrade.strategy.trx_phase1.benchmarks import fair_benchmarks
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.diagnostics import overview
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.studies import rule_fingerprint


GROUPS = {
    "buy_threshold": ("buy_threshold",),
    "sell_threshold": ("sell_threshold",),
    "buy_fraction": ("buy_fraction",),
    "sell_fraction": ("sell_fraction",),
    "buy_quantity_dynamic": ("buy_quantity_dynamic",),
    "sell_quantity_dynamic": ("sell_quantity_dynamic",),
    "disable_earn": ("earn_model",),
    "execution_preflight": ("execution_preflight",),
    "buy_reference_recovery": ("buy_reference_recovery",),
    "sell_quantity_basis": ("sell_quantity_basis",),
    "sell_fraction_cap": ("sell_fraction_cap",),
}


def run_experiment(data, saved, root, experiment_id, group, values, verify, progress=None):
    if group not in GROUPS or not 1 <= len(values) <= 5:
        raise ValueError("Choose one supported parameter group and 1-5 candidates")
    baseline = ResearchConfig.from_saved(saved["config"])
    candidates = [baseline]
    for value in values:
        if (
            group.endswith("_dynamic") or group in ("execution_preflight", "buy_reference_recovery")
        ) and type(value) is not bool:
            raise ValueError("Quantity switches require boolean values")
        if group == "disable_earn" and value != "none":
            raise ValueError("Only the independent zero-Earn path is supported here")
        candidates.append(
            ResearchConfig(
                **{
                    **baseline.model_dump(),
                    group if group != "disable_earn" else "earn_model": value,
                }
            )
        )
    folder = Path(root) / experiment_id
    details = folder / "results"
    details.mkdir(parents=True, exist_ok=True)
    rule_hash = rule_fingerprint()
    entries = []
    for index, cfg in enumerate(candidates):
        run_id = str(uuid4())
        path = details / f"{run_id}.json"
        replayed = replay(
            data,
            cfg,
            diagnostic_path=path.with_suffix(".diagnostics.sqlite"),
            progress=lambda value, i=index: (
                progress((i + value) / (len(candidates) + 1)) if progress else None
            ),
        )
        if index == 0:
            verify(saved, replayed)
        replayed.update(
            run_id=run_id,
            experiment_id=experiment_id,
            protocol={"stage": "observed_history", "group": group, "baseline": index == 0},
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(replayed, allow_nan=False))
        temporary.replace(path)
        diagnosis = overview(path.with_suffix(".diagnostics.sqlite"))
        entries.append(
            {
                "run_id": run_id,
                "baseline": index == 0,
                "value": getattr(cfg, GROUPS[group][0]),
                "config": cfg.model_dump(),
                "summary": replayed["summary"],
                "cash_only": diagnosis["cash_only"],
                "amounts": diagnosis["amounts"],
                "lots_summary": diagnosis["lots_summary"],
                "return_change": replayed["summary"]["return_ratio"]
                - entries[0]["summary"]["return_ratio"]
                if entries
                else 0,
                "drawdown_change": replayed["summary"]["max_drawdown"]
                - entries[0]["summary"]["max_drawdown"]
                if entries
                else 0,
            }
        )
    benchmarks = fair_benchmarks(data, baseline)
    if rule_fingerprint() != rule_hash:
        raise ValueError("Research source changed while running; comparison not published")
    manifest = {
        "experiment_id": experiment_id,
        "group": group,
        "baseline_run_id": saved["run_id"],
        "stage": "observed_history",
        "data_fingerprint": saved["data_fingerprint"],
        "start_utc": saved["start_utc"],
        "end_utc": saved["end_utc"],
        "rules_sha256": rule_hash,
        "entries": entries,
        "benchmarks": benchmarks,
        "notes": [
            "Only one parameter group changes; the complete baseline is frozen.",
            "Observed historical exploration; not an unseen holdout or forward simulation.",
            "Risk threshold filters research results; it is not a stop-loss rule.",
        ],
    }
    temp = folder / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, allow_nan=False))
    temp.replace(folder / "manifest.json")
    if progress:
        progress(1)
    return manifest
