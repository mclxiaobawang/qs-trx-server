"""Same-calendar comparisons from completed immutable continuous account ledgers."""

import json
import sqlite3
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.strategy.trx_phase1.research_metrics import read_json


def direction_changes(values):
    signs = [int(np.sign(value)) for value in values if abs(value) > 1e-7]
    return sum(left != right for left, right in pairwise(signs))


def original_window(path, begin, end):
    with sqlite3.connect(path) as db:
        frame = pd.read_sql_query(
            "SELECT timestamp,equity,exposure,principal_trx,remaining_cost,interest_trx,"
            "trx_interest_value,realized_profit,principal_unrealized,anchor "
            "FROM samples WHERE timestamp>=? AND timestamp<=? ORDER BY timestamp",
            db,
            params=(begin, end),
        )
        old_cost = db.execute(
            "SELECT avg(stale_cost),max(stale_cost) FROM risk_samples "
            "WHERE timestamp>? AND timestamp<=?",
            (begin, end),
        ).fetchone()
    if frame.empty or frame.timestamp.iloc[0] != begin or frame.timestamp.iloc[-1] != end:
        raise ValueError("Account calendar window is incomplete")
    first, last = frame.iloc[0], frame.iloc[-1]
    part = frame.iloc[1:]
    return {
        "start_at": begin,
        "end_at": end,
        "minutes": len(part),
        "start_equity": float(first.equity),
        "end_equity": float(last.equity),
        "return_ratio": float(last.equity / first.equity - 1),
        "max_drawdown": float(
            (1 - part.equity / part.equity.cummax().clip(lower=first.equity)).max()
        ),
        "mean_exposure": float(part.exposure.mean()),
        "mean_interest_exposure": float((part.trx_interest_value / part.equity).mean()),
        "mean_old_principal_cost_30_days": old_cost[0],
        "peak_old_principal_cost_30_days": old_cost[1],
        "entry_principal_trx": float(first.principal_trx),
        "entry_remaining_cost": float(first.remaining_cost),
        "entry_retained_interest_trx": float(first.interest_trx),
        "entry_anchor": float(first.anchor),
        "realized_principal_change": float(last.realized_profit - first.realized_profit),
        "ending_principal_unrealized": float(last.principal_unrealized),
    }


def audit_cross_period(root, experiment_id):
    directory = Path(root) / "experiments" / experiment_id
    manifest = read_json(directory / "manifest.json")
    source = Path(root) / "experiments" / manifest["coverage_experiment_id"]
    original = pd.read_feather(source / "inputs/original-control-1m.feather")
    begin = int(original.date.iloc[0].timestamp() * 1000)
    end = int(original.date.iloc[-1].timestamp() * 1000) + 60000
    rows = [row for row in manifest["entries"] if row["scenario"] == "continuous_history"]
    windows = {}
    for row in rows:
        windows[row["case"]] = original_window(
            directory / "results" / f"{row['run_id']}.diagnostics.sqlite", begin, end
        )
        if windows[row["case"]]["minutes"] != len(original):
            raise ValueError("Inherited window does not cover all original-year minutes")
    stress = {}
    for origin in rows:
        if not origin["risk_metrics"]["drawdowns"]:
            continue
        worst = max(origin["risk_metrics"]["drawdowns"], key=lambda value: value["drawdown"])
        key = f"{worst['peak_at']}-{worst['trough_at']}"
        if key in stress:
            stress[key]["maximum_drawdown_sources"].append(origin["case"])
            continue
        stress[key] = {
            "start_at": int(worst["peak_at"]),
            "end_at": int(worst["trough_at"]),
            "maximum_drawdown_sources": [origin["case"]],
            "accounts": {
                row["case"]: original_window(
                    directory / "results" / f"{row['run_id']}.diagnostics.sqlite",
                    int(worst["peak_at"]),
                    int(worst["trough_at"]),
                )
                for row in rows
            },
        }
    comparisons, table = {}, []
    series = {
        row["case"]: {p["period"]: p["end_equity"] for p in row["continuous_metrics"]["monthly"]}
        for row in rows
    }
    for row in manifest["benchmarks"]:
        if row["name"] in {"usdt", "trx_hold"}:
            series[row["name"]] = {
                p["period"]: p["end_equity"] for p in row["continuous_periods"]["monthly"]
            }
    pairs = list(combinations([row["case"] for row in rows], 2))
    pairs += [(row["case"], name) for row in rows for name in ("usdt", "trx_hold")]
    for left, right in pairs:
        if series[left].keys() != series[right].keys():
            raise ValueError("Month-end comparison calendars differ")
        months = sorted(series[left])
        gaps = [series[left][month] - series[right][month] for month in months]
        comparisons[f"{left}-minus-{right}"] = {
            "month_end_direction_changes": direction_changes(gaps),
            "months_positive_gap": sum(value > 1e-7 for value in gaps),
            "months_negative_gap": sum(value < -1e-7 for value in gaps),
            "months": len(months),
            "ending_equity_gap": gaps[-1],
            "minimum_month_end_gap": min(gaps),
            "maximum_month_end_gap": max(gaps),
        }
        table.extend(
            {
                "month": month,
                "left": left,
                "right": right,
                "left_equity": series[left][month],
                "right_equity": series[right][month],
                "equity_gap": gap,
            }
            for month, gap in zip(months, gaps, strict=True)
        )
    report = {
        "original_calendar_window": windows,
        "shared_drawdown_windows": stress,
        "month_end_comparisons": comparisons,
        "notes": [
            (
                "Month-end direction changes ignore only numerical noise below 1e-7 USDT; "
                "they do not count intramonth reversals."
            ),
            (
                "The inherited original-year window begins at actual carried equity "
                "and retains all original states. It need not repeat original-start return."
            ),
            (
                "Shared stress windows use every candidate's actual maximum-drawdown "
                "peak-to-trough dates after replay, solely for diagnosis; all accounts "
                "are measured on identical dates without removing trades or restarting."
            ),
        ],
    }
    (directory / "analysis/same-calendar-audit.json").write_text(
        json.dumps(report, allow_nan=False)
    )
    pd.DataFrame(table).to_csv(directory / "analysis/month-end-equity-gaps.csv", index=False)
    return report
