"""Read-only follow-up on fixed candidate paths and the shared funding window."""

import json
import sqlite3
from pathlib import Path

from freqtrade.strategy.trx_phase1.candidate_attribution import aligned_samples, lot_contributions
from freqtrade.strategy.trx_phase1.research_metrics import read_json


def audit_stability(root, manifest):
    directory = Path(root) / "experiments" / manifest["experiment_id"]
    output = directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    entries = {row["case"]: row for row in manifest["entries"]}

    def path(key):
        row = entries[key]
        owner = row.get("source_experiment_id", manifest["experiment_id"])
        return Path(root) / "experiments" / owner / "results" / f"{row['run_id']}.json"

    left_key, right_key = "after_180_days_both_60_control", "after_180_days_both_60_prepared"
    a, b = aligned_samples(
        path(left_key).with_suffix(".diagnostics.sqlite"),
        path(right_key).with_suffix(".diagnostics.sqlite"),
    )
    switch = entries[left_key]["funding_transition"]["timestamp"]
    tail = a.timestamp >= switch
    if not tail.any() or a.loc[tail, "equity"].iloc[0] != b.loc[tail, "equity"].iloc[0]:
        raise ValueError("Funding window does not start from identical continuous equity")
    start_equity = float(a.loc[tail, "equity"].iloc[0])
    funding = {
        "start_at": switch,
        "end_at": int(a.timestamp.iloc[-1]),
        "start_equity": start_equity,
    }
    for key, values in (("control", a), ("prepared", b)):
        part = values.loc[tail]
        drawdown = 1 - part.equity / part.equity.cummax()
        detail = read_json(path(left_key if key == "control" else right_key))
        annual_drawdown = 1 - values.equity / values.equity.cummax().clip(
            lower=detail["config"]["initial_cash"]
        )
        fills = [row for row in detail["fills"] if row["timestamp"] >= switch]
        funding[key] = {
            "return_since_switch": float(part.equity.iloc[-1] / start_equity - 1),
            "max_drawdown_since_switch": float(drawdown.max()),
            "drawdown_trough_at": int(part.loc[drawdown.idxmax(), "timestamp"]),
            "mean_exposure_since_switch": float(part.exposure.mean()),
            "buy_fills_since_switch": sum(row["side"] == "buy" for row in fills),
            "sell_fills_since_switch": sum(row["side"] == "sell" for row in fills),
            "annual_max_drawdown": detail["summary"]["max_drawdown"],
            "annual_drawdown_trough_at": int(values.loc[annual_drawdown.idxmax(), "timestamp"]),
            "ending_equity": detail["summary"]["equity"],
        }
    funding["notes"] = [
        (
            "The shared post-switch window starts at the same actual account equity; "
            "positions and interest continue without reset."
        ),
        (
            "Local drawdown resets only its measurement peak at the boundary, "
            "never the simulated account. Inspect annual and post-switch drawdown separately."
        ),
    ]
    (output / "funding-after180.json").write_text(json.dumps(funding, allow_nan=False))
    c1 = read_json(path("C1"))
    c1_orders = {(row["timestamp"], row["side"]): row for row in c1["orders"]}
    neighbors = {}
    for key in ("C_08", "C_10"):
        detail = read_json(path(key))
        orders = {(row["timestamp"], row["side"]): row for row in detail["orders"]}
        asymmetric = sorted(c1_orders.keys() ^ orders.keys())
        timestamp, side = asymmetric[0] if asymmetric else (None, None)
        context = {}
        for name in ("C1", key):
            with sqlite3.connect(path(name).with_suffix(".diagnostics.sqlite")) as db:
                reason_counts = db.execute(
                    "SELECT side,status,reason,count(*) FROM records "
                    "WHERE kind='opportunities' GROUP BY side,status,reason"
                ).fetchall()
                evaluations = (
                    []
                    if timestamp is None
                    else [
                        json.loads(row[0])
                        for row in db.execute(
                            "SELECT payload FROM records WHERE kind='evaluations' "
                            "AND timestamp<=? AND side=? ORDER BY timestamp DESC LIMIT 2",
                            (timestamp, side),
                        )
                    ]
                )
                context[name] = {
                    "opportunity_reason_counts": [
                        {"side": row[0], "status": row[1], "reason": row[2], "count": row[3]}
                        for row in reason_counts
                    ],
                    "last_evaluations_before_first_asymmetric_fill": evaluations,
                }
        neighbors[key] = {
            "common_fill_minutes": len(c1_orders.keys() & orders.keys()),
            "c1_only_fill_minutes": len(c1_orders.keys() - orders.keys()),
            "neighbor_only_fill_minutes": len(orders.keys() - c1_orders.keys()),
            "first_asymmetric_fill_at": timestamp,
            "first_asymmetric_side": side,
            "c1_first_fill": c1_orders.get((timestamp, side)),
            "neighbor_first_fill": orders.get((timestamp, side)),
            "diagnostic_context": context,
        }
    completed = {}
    for key in ("legacy_dynamic_period_only", "C1"):
        detail = read_json(path(key))
        lots = lot_contributions(detail)
        closed = lots.remaining_quantity == 0
        completed[key] = {
            "fully_exited_batches": int(closed.sum()),
            "fully_exited_batch_net_profit": float(lots.loc[closed, "realized"].sum()),
            "partial_batch_realized_net_profit": float(lots.loc[~closed, "realized"].sum()),
        }
    audit = {
        "neighbors": neighbors,
        "completed_batches": completed,
        "notes": [
            (
                "Fill matching uses actual timestamp and side, not sequence IDs. "
                "Costs and interest remain on the original uninterrupted path."
            ),
            (
                "First asymmetric fills explain observable path divergence; "
                "no adverse trade was deleted and no extra parameter was tested."
            ),
        ],
    }
    (output / "neighbor-boundaries.json").write_text(json.dumps(audit, allow_nan=False))
    return funding, audit
