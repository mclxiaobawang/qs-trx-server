"""Passive alternatives use the same ledger, hourly products, costs and execution checks."""

import csv
from contextlib import ExitStack
from decimal import ROUND_DOWN
from pathlib import Path

import pandas as pd

from freqtrade.strategy.trx_phase1.candles import normalized_volatility, validate_minutes
from freqtrade.strategy.trx_phase1.cash_policy import CashPolicy
from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.research_earn import HistoricalEarn


def fair_benchmarks(frame, config, curve_directory=None, allocations=None):
    with ExitStack() as stack:
        return _fair_benchmarks(frame, config, curve_directory, stack, allocations)


def _fair_benchmarks(frame, config, curve_directory, stack, allocations=None):  # noqa: C901
    data = validate_minutes(frame)
    sigma = normalized_volatility(data, config.volatility_window).to_numpy()
    stamps = pd.DatetimeIndex(data.date).as_unit("ms").asi8
    opening, closing = data.open.to_numpy(), data.close.to_numpy()
    results = []
    choices = (
        allocations
        if allocations is not None
        else [("usdt", 0), ("usdgo", 0), ("trx_hold", 1), ("low_exposure", 0.25)]
    )
    if len({name for name, _ in choices}) != len(choices) or any(
        name not in {"usdt", "usdgo", "trx_hold", "low_exposure", "exposure_40", "exposure_70"}
        or not 0 <= fraction <= 1
        for name, fraction in choices
    ):
        raise ValueError("Invalid passive allocations")
    for name, fraction in choices:
        curve, writer = [], None
        if curve_directory is not None:
            directory = Path(curve_directory)
            directory.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context((directory / f"{name}.csv").open("w", newline=""))
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "equity",
                    "price",
                    "cash",
                    "usdgo_value",
                    "principal_trx",
                    "interest_trx",
                    "exposure",
                ],
            )
            writer.writeheader()
        cfg = ResearchConfig(
            **{
                **config.model_dump(),
                "startup": "small_buy",
                "initial_buy_fraction": fraction or 0.02,
            }
        )
        engine = Engine(cfg)
        earn = HistoricalEarn(engine.book, cfg, int(stamps[0]))
        engine.earn, engine.prepare_cash = earn, earn.prepare_cash
        if fraction:
            earn.sweep_cash(int(stamps[0]))
        elif name == "usdgo":
            if earn.policy is None and cfg.earn_model != "simplified_hourly":
                # Passive asset holding still needs FX marking when interest is disabled.
                # TRX-only Earn does not authorize stablecoin interest.
                earn.config = cfg.model_copy(update={"usdt_apr": 0, "usdt_apr_tiers": []})
                earn.policy = CashPolicy(earn)
            if cfg.earn_model == "simplified_hourly":
                earn.sweep_cash(int(stamps[0]))
            elif (
                earn.policy
                and cfg.usdgo_product_available
                and earn.policy.quote_usable(int(stamps[0]))
            ):
                mark, _, ask = earn.policy.quote(int(stamps[0]))
                quantity = engine.book.free_cash / (money(ask) * (1 + money(cfg.fx_fee)))
                if cfg.usdgo_earn_cap:
                    quantity = min(quantity, money(cfg.usdgo_earn_cap))
                quantity = money(quantity).quantize(
                    money(10) ** -cfg.fx_quantity_decimals, rounding=ROUND_DOWN
                )
                spend = quantity * money(ask) * (1 + money(cfg.fx_fee))
                if spend >= money(cfg.cash_min_switch_value) and quantity > 0:
                    engine.book.convert("passive-usdgo", "to_usdgo", spend, quantity, mark)
                    earn.subscribe_usdgo(quantity, int(stamps[0]))
                    earn.reconcile_usdgo()
        peak, drawdown, exposure, started = cfg.initial_cash, 0, 0, False
        for index, (timestamp, price_open, price) in enumerate(
            zip(stamps, opening, closing, strict=True)
        ):
            earn.advance(int(timestamp), float(closing[max(0, index - 1)]))
            if engine.pending and timestamp >= engine.pending.ready_at:
                execution = float(price_open) * (1 + cfg.slippage)
                if engine.executable(int(timestamp), execution):
                    plan = engine.pending
                    quantity = engine.execution_quantity(execution)
                    if quantity * execution >= cfg.min_notional:
                        engine.fill(
                            plan.plan_id,
                            plan.plan_id,
                            int(timestamp),
                            quantity,
                            execution,
                            quantity * execution * cfg.buy_fee,
                        )
                else:
                    engine.finish(int(timestamp))
            end = int(timestamp + 60000)
            earn.advance(end, float(price))
            engine.book.settle_fee_rebates(end)
            if not started and fraction and index >= cfg.volatility_window:
                engine.on_close(end, float(price), float(sigma[index]))
                started = True
            else:
                engine.observe(end, float(price), float(sigma[index]))
                if engine.pending and end >= engine.pending.expires_at:
                    engine.finish(end)
            if fraction:
                earn.allocate_cash(end, engine.pending)
            equity = float(engine.book.equity(float(price)))
            peak = max(peak, equity)
            drawdown = max(drawdown, 1 - equity / peak)
            exposure += float(engine.book.principal + engine.book.interest_trx) * price / equity
            if writer is not None:
                sample = {
                    "timestamp": end,
                    "equity": equity,
                    "price": float(price),
                    "cash": float(engine.book.cash),
                    "usdgo_value": float(engine.book.usdgo * (engine.book.usdgo_mark or money(0))),
                    "principal_trx": float(engine.book.principal),
                    "interest_trx": float(engine.book.interest_trx),
                    "exposure": float(engine.book.principal + engine.book.interest_trx)
                    * price
                    / equity,
                }
                writer.writerow(sample)
                if index % 60 == 0 or index == len(data) - 1:
                    curve.append(sample)
        if engine.pending:
            engine.finish(int(stamps[-1] + 60000))
        summary = {**engine.book.summary(float(closing[-1])), **earn.summary(float(closing[-1]))}
        summary.update(
            max_drawdown=drawdown,
            mean_trx_exposure=float(exposure / len(data)),
            buy_orders=len(engine.book.orders),
            sell_orders=0,
        )
        results.append(
            {
                "name": name,
                "summary": summary,
                "cash_only": not engine.book.orders,
                "allocation_fraction": fraction,
                "stage": "observed_history",
                "execution": (
                    "one initial purchase after identical warmup; expiry and redemption "
                    "checks apply"
                ),
                "end_inventory": "marked to market, no forced liquidation",
                "usdgo_policy": (
                    "passive allocation subject to quote, cost, product, quota and "
                    "minimum; no benefit gate"
                ),
            }
        )
        if writer is not None:
            results[-1].update(
                equity_curve=curve, curve_file=f"{name}.csv", full_curve_rows=len(data)
            )
    return results
