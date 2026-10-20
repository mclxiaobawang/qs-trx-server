"""Supplement existing minute diagnostics with principal occupation through time."""

import json
import sqlite3

from freqtrade.strategy.trx_phase1.diagnostics import Recorder, overview


class StudyRecorder(Recorder):
    def __init__(self, path):
        super().__init__(path)
        self.db.execute(
            "CREATE TABLE risk_samples (timestamp INTEGER,underwater_cost REAL,"
            "stale_cost REAL,spendable_cash REAL,prepared_cash REAL,prepared_trx REAL)"
        )
        self.risk_samples = []

    def sample(self, engine, earn, timestamp, price, equity):
        super().sample(engine, earn, timestamp, price, equity)
        lots = [lot for lot in engine.book.lots if lot.quantity]
        net_price = price * (1 - engine.config.sell_fee)
        preparation = getattr(engine, "preparation", None)
        self.risk_samples.append(
            (
                timestamp,
                sum(float(lot.remaining_cost) for lot in lots if net_price < float(lot.unit_cost)),
                sum(
                    float(lot.remaining_cost)
                    for lot in lots
                    if timestamp - lot.bought_at >= 30 * 86400000
                ),
                float(
                    earn.spendable_cash()
                    if hasattr(earn, "spendable_cash")
                    else engine.book.free_cash
                ),
                preparation.ready_cash if preparation else 0,
                preparation.ready_trx if preparation else 0,
            )
        )
        if len(self.risk_samples) >= 1000:
            self.flush_risk()

    def flush_risk(self):
        if self.risk_samples:
            self.db.executemany("INSERT INTO risk_samples VALUES (?,?,?,?,?,?)", self.risk_samples)
            self.risk_samples.clear()

    def finish(self, result, engine, earn):
        self.flush_risk()
        return super().finish(result, engine, earn)


def risk_metrics(path, min_notional):
    diagnosis = overview(path)
    with sqlite3.connect(path) as db:
        mean_loss, peak_loss, loss_minutes, mean_stale, peak_stale, cash_short = db.execute(
            "SELECT avg(underwater_cost),max(underwater_cost),sum(underwater_cost>0),"
            "avg(stale_cost),max(stale_cost),sum(spendable_cash<?) FROM risk_samples",
            (min_notional,),
        ).fetchone()
    return {
        "mean_underwater_principal_cost": mean_loss,
        "peak_underwater_principal_cost": peak_loss,
        "underwater_principal_minutes": loss_minutes,
        "mean_stale_principal_cost_30_days": mean_stale,
        "peak_stale_principal_cost_30_days": peak_stale,
        "cash_below_minimum_minutes": cash_short,
        "drawdowns": diagnosis["drawdowns"],
        "attribution_residual": diagnosis["attribution_residual"],
        "definition": "Underwater cost uses each remaining lot's net-sale mark, not aggregate PnL; "
        "cost occupation is not a loss amount. Minute samples retain the continuous ledger.",
    }, diagnosis


def read_json(path):
    return json.loads(path.read_text())
