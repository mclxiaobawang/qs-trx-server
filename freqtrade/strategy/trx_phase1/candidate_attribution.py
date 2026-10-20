"""Read complete saved ledgers; explain differences without altering either path."""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from freqtrade.strategy.trx_phase1.candles import data_fingerprint
from freqtrade.strategy.trx_phase1.research_metrics import read_json


SAMPLE_COLUMNS = (
    "timestamp,price,equity,exposure,principal_trx,interest_trx,remaining_cost,"
    "realized_profit,principal_unrealized,stable_interest,fx_profit,fx_unrealized,"
    "trx_credit_value,trx_interest_value,gross_fees,credited_rebates,anchor,"
    "buy_period,sell_period"
)


def samples(path):
    with sqlite3.connect(path) as db:
        return pd.read_sql_query(f"SELECT {SAMPLE_COLUMNS} FROM samples ORDER BY timestamp", db)


def aligned_samples(left, right):
    a, b = samples(left), samples(right)
    if not a.timestamp.equals(b.timestamp) or not a.price.equals(b.price):
        raise ValueError("Saved minute paths are not on the same calendar and price path")
    return a, b


def verify_ending_sample(values, result):
    ending = values.iloc[-1]
    for column, field in (
        ("equity", "equity"),
        ("realized_profit", "realized_profit"),
        ("principal_unrealized", "unrealized_profit"),
        ("stable_interest", "stable_interest"),
        ("trx_credit_value", "trx_interest_credited_value"),
        ("trx_interest_value", "trx_interest_market_value"),
    ):
        if abs(ending[column] - result["summary"][field]) > 1e-7:
            raise ValueError("Ending minute ledger differs from its saved summary")


def lot_contributions(result):
    """Corrected sale allocations include already credited buy/sell fee refunds."""
    bought = {row["order_id"]: row for row in result["orders"] if row["side"] == "buy"}
    sold = {key: {"quantity": 0.0, "profit": 0.0} for key in bought}
    for row in result["orders"]:
        for part in row["allocations"]:
            sold[part["lot_id"]]["quantity"] += part["quantity"]
            sold[part["lot_id"]]["profit"] += part["profit"]
    rows = []
    for lot in result["lots"]:
        buy = bought[lot["lot_id"]]
        rows.append(
            {
                "timestamp": int(buy["timestamp"]),
                "lot_id": lot["lot_id"],
                "bought_quantity": buy["quantity"],
                "purchase_cost": buy["cost"],
                "sold_quantity": sold[lot["lot_id"]]["quantity"],
                "realized": sold[lot["lot_id"]]["profit"],
                "remaining_quantity": lot["quantity"],
                "remaining_cost": float(lot["remaining_cost"]),
                "unrealized": lot["unrealized_profit"],
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        if frame.timestamp.duplicated().any():
            raise ValueError("Ambiguous buy-batch calendar matching")
        for field, total in (("realized", "realized_profit"), ("unrealized", "unrealized_profit")):
            if abs(frame[field].sum() - result["summary"][total]) > 1e-7:
                raise ValueError("Corrected batch allocation does not reconcile to the ledger")
    return frame


def execution_costs(result, frame):
    """Mechanical cost decomposition on the actual path, not a zero-cost replay."""
    opening = dict(zip(pd.DatetimeIndex(frame.date).as_unit("ms").asi8, frame.open, strict=True))
    lots = {row["lot_id"]: row for row in result["lots"]}
    orders = {row["order_id"]: row for row in result["orders"]}
    held_fee = held_slippage = buy_slippage = sell_slippage = 0.0
    for fill in result["fills"]:
        raw = float(opening[fill["timestamp"]])
        expected = raw * (
            1 + result["config"]["slippage"]
            if fill["side"] == "buy"
            else 1 - result["config"]["slippage"]
        )
        if abs(fill["price"] - expected) > 1e-12:
            raise ValueError("Execution price is not the frozen next-open slippage model")
        if fill["side"] == "buy":
            if fill["net_quantity"] != fill["quantity"]:
                raise ValueError("This decomposition requires quote-denominated buy fees")
            cost = fill["quantity"] * (fill["price"] - raw)
            buy_slippage += cost
            held_slippage += (
                cost * lots[fill["order_id"]]["quantity"] / orders[fill["order_id"]]["quantity"]
            )
        else:
            sell_slippage += fill["quantity"] * (raw - fill["price"])
    for key, lot in lots.items():
        row = orders[key]
        held_fee += row["net_fee"] * lot["quantity"] / row["quantity"]
    summary = result["summary"]
    net_fees = summary["gross_fees"] - summary["fee_rebates"]
    total = net_fees + buy_slippage + sell_slippage
    return {
        "gross_fees": summary["gross_fees"],
        "credited_rebates": summary["fee_rebates"],
        "net_fees": net_fees,
        "buy_slippage": buy_slippage,
        "sell_slippage": sell_slippage,
        "total": total,
        "held_costs": held_fee + held_slippage,
        "realized_costs": total - held_fee - held_slippage,
    }


def common_windows(a, b, episodes):
    rows = []
    for episode in episodes:
        start, end = episode["peak_at"], episode.get("recovered_at") or int(a.timestamp.iloc[-1])
        window = (a.timestamp >= start) & (a.timestamp <= end)
        row = {"start_at": start, "end_at": end, "source_trough_at": episode["trough_at"]}
        for name, values in (("source", a), ("c1", b)):
            part = values.loc[window]
            trough = part.loc[part.equity.idxmin()]
            local_dd = 1 - part.equity / part.equity.cummax()
            ref = float(part.equity.iloc[0])
            recovery = values.loc[
                (values.timestamp > trough.timestamp) & (values.equity >= ref), "timestamp"
            ]
            row[name] = {
                "start_equity": ref,
                "end_equity": float(part.equity.iloc[-1]),
                "minimum_equity": float(trough.equity),
                "minimum_at": int(trough.timestamp),
                "max_drawdown_in_common_window": float(local_dd.max()),
                "mean_exposure": float(part.exposure.mean()),
                "trough_exposure": float(trough.exposure),
                "trough_principal_unrealized": float(trough.principal_unrealized),
                "recovered_start_equity_at": int(recovery.iloc[0]) if len(recovery) else None,
            }
        rows.append(row)
    return rows


def explain_c1(source_path, c1_path, frame, output):
    """Use full-minute samples and corrected allocations, never aggregate tables alone."""
    source_path, c1_path, output = map(Path, (source_path, c1_path, output))
    output.mkdir(parents=True, exist_ok=True)
    left, right = read_json(source_path), read_json(c1_path)
    if left["data_fingerprint"] != right["data_fingerprint"]:
        raise ValueError("C1/source price fingerprints differ")
    if data_fingerprint(frame) != left["data_fingerprint"]:
        raise ValueError("Attribution candles differ from the saved complete price path")
    a, b = aligned_samples(
        source_path.with_suffix(".diagnostics.sqlite"), c1_path.with_suffix(".diagnostics.sqlite")
    )
    for values, result in ((a, left), (b, right)):
        verify_ending_sample(values, result)
    costs = {"source": execution_costs(left, frame), "c1": execution_costs(right, frame)}
    components = []
    definitions = (
        ("principal_realized_net", "realized_profit"),
        ("principal_unrealized_net", "unrealized_profit"),
        ("trx_interest_credit_value", "trx_interest_credited_value"),
        ("trx_interest_price_pnl", "trx_interest_price_pnl"),
        ("stable_interest", "stable_interest"),
        ("fx_realized_net", "fx_profit"),
        ("fx_unrealized_net", "usdgo_unrealized_profit"),
    )
    for key, field in definitions:
        old, new = left["summary"][field], right["summary"][field]
        components.append({"component": key, "source": old, "c1": new, "change": new - old})
    additive = []
    for row in components:
        copy = row.copy()
        if row["component"].startswith("principal_"):
            cost_field = (
                "realized_costs" if row["component"] == "principal_realized_net" else "held_costs"
            )
            copy.update(
                component=row["component"].replace("_net", "_before_execution_costs"),
                source=row["source"] + costs["source"][cost_field],
                c1=row["c1"] + costs["c1"][cost_field],
            )
            copy["change"] = copy["c1"] - copy["source"]
        additive.append(copy)
    additive.append(
        {
            "component": "execution_costs",
            "source": -costs["source"]["total"],
            "c1": -costs["c1"]["total"],
            "change": costs["source"]["total"] - costs["c1"]["total"],
        }
    )
    delta = right["summary"]["equity"] - left["summary"]["equity"]
    residual = delta - sum(row["change"] for row in additive)
    if abs(residual) > 1e-7:
        raise ValueError("C1 execution-cost attribution failed conservation")
    joined = a.add_prefix("source_").join(b.drop(columns=["timestamp", "price"]).add_prefix("c1_"))
    joined = joined.rename(columns={"source_timestamp": "timestamp", "source_price": "price"})
    joined["equity_gap"] = b.equity - a.equity
    joined["source_drawdown"] = 1 - a.equity / a.equity.cummax().clip(
        lower=left["config"]["initial_cash"]
    )
    joined["c1_drawdown"] = 1 - b.equity / b.equity.cummax().clip(
        lower=right["config"]["initial_cash"]
    )
    joined.to_csv(output / "c1-source-minute-paths.csv", index=False)
    dates = pd.to_datetime(a.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    ends = a.groupby(dates.dt.strftime("%Y-%m")).tail(1).index
    monthly = []
    previous = {
        "equity": 0.0,
        "realized_profit": 0.0,
        "principal_unrealized": 0.0,
        "stable_interest": 0.0,
        "trx_interest_value": 0.0,
    }
    for index in ends:
        row = {"month": dates.iloc[index].strftime("%Y-%m"), "end_at": int(a.timestamp.iloc[index])}
        for field, last_gap in previous.items():
            gap = float(b.loc[index, field] - a.loc[index, field])
            row[f"{field}_gap"] = gap
            row[f"{field}_gap_change"] = gap - last_gap
            previous[field] = gap
        mask = dates.dt.strftime("%Y-%m") == row["month"]
        row.update(
            source_mean_exposure=float(a.loc[mask, "exposure"].mean()),
            c1_mean_exposure=float(b.loc[mask, "exposure"].mean()),
        )
        monthly.append(row)
    pd.DataFrame(monthly).to_csv(output / "c1-source-monthly.csv", index=False)
    old_lots, new_lots = lot_contributions(left), lot_contributions(right)
    lot_diff = old_lots.merge(
        new_lots, on="timestamp", how="outer", suffixes=("_source", "_c1"), indicator=True
    )
    for field in ("realized", "unrealized", "remaining_cost", "bought_quantity"):
        lot_diff[f"{field}_change"] = lot_diff[f"{field}_c1"].fillna(0) - lot_diff[
            f"{field}_source"
        ].fillna(0)
    lot_diff["principal_pnl_change"] = lot_diff.realized_change + lot_diff.unrealized_change
    lot_diff.to_csv(output / "c1-source-batches.csv", index=False)
    concentration = {}
    for field in ("realized_change", "principal_pnl_change"):
        ordered = lot_diff.reindex(lot_diff[field].abs().sort_values(ascending=False).index)
        absolute = float(ordered[field].abs().sum())
        concentration[field] = {
            "net": float(ordered[field].sum()),
            "absolute": absolute,
            "top5_absolute_share": float(ordered[field].head(5).abs().sum() / absolute)
            if absolute
            else 0,
            "top10_absolute_share": float(ordered[field].head(10).abs().sum() / absolute)
            if absolute
            else 0,
            "top10": json.loads(
                ordered[["timestamp", "_merge", field]].head(10).to_json(orient="records")
            ),
        }
    report = {
        "source_run_id": left["run_id"],
        "c1_run_id": right["run_id"],
        "minute_rows": len(a),
        "net_equity_change": delta,
        "net_components": components,
        "additive_components_before_execution_costs": additive,
        "execution_costs": costs,
        "attribution_residual": residual,
        "monthly": monthly,
        "common_drawdown_windows": common_windows(a, b, left["risk_metrics"]["drawdowns"][:3]),
        "batches": {
            "source": len(old_lots),
            "c1": len(new_lots),
            "common_buy_minutes": int((lot_diff._merge == "both").sum()),
            "added_buy_minutes": int((lot_diff._merge == "right_only").sum()),
            "removed_buy_minutes": int((lot_diff._merge == "left_only").sum()),
            "concentration": concentration,
        },
        "hourly_paths": json.loads(
            joined.iloc[::60][
                [
                    "timestamp",
                    "source_equity",
                    "c1_equity",
                    "equity_gap",
                    "source_exposure",
                    "c1_exposure",
                    "source_drawdown",
                    "c1_drawdown",
                ]
            ].to_json(orient="records")
        ),
        "notes": [
            "Monthly calendar is Asia/Shanghai; first/last months are partial; no account restart.",
            (
                "Corrected sold allocations include credited fee rebates; "
                "raw execution fills are immutable."
            ),
            (
                "Cost decomposition restores fees and next-open slippage on the observed "
                "quantities only; it is not a hypothetical zero-cost account path."
            ),
            (
                "Net principal components already include execution costs; "
                "use only one decomposition at a time."
            ),
            (
                "Batches match actual buy minutes, not path-dependent sequence IDs; "
                "no trade was deleted or replayed."
            ),
            (
                "FX components are net of conversion execution costs; "
                "these two paths have no conversions."
            ),
        ],
    }
    (output / "c1-attribution.json").write_text(json.dumps(report, allow_nan=False))
    return report
