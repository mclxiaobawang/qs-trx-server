import time
from decimal import ROUND_DOWN

import numpy as np
import pandas as pd

from freqtrade.strategy.trx_phase1.candles import (
    data_fingerprint,
    normalized_volatility,
    validate_minutes,
)
from freqtrade.strategy.trx_phase1.config import ASSUMPTIONS, ResearchConfig
from freqtrade.strategy.trx_phase1.diagnostics import Recorder
from freqtrade.strategy.trx_phase1.engine import Engine
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.price_position import build_price_reference
from freqtrade.strategy.trx_phase1.research_earn import HistoricalEarn


def initialize_half_position(engine, earn, timestamp, opening):
    """Explicit historical allocation at the first open, before observing its close."""
    cfg, book, policy = engine.config, engine.book, earn.policy
    if policy is None or not cfg.usdgo_product_available or not policy.quote_usable(timestamp):
        raise ValueError("Initial USDGO allocation needs an available product and usable FX quote")
    trx_budget = money(cfg.initial_cash) * money(cfg.initial_buy_fraction)
    usdgo_budget = money(cfg.initial_cash) - trx_budget
    mark, _, ask = policy.quote(timestamp)
    quantity = (usdgo_budget / (money(ask) * (1 + money(cfg.fx_fee)))).quantize(
        money(10) ** -cfg.fx_quantity_decimals, rounding=ROUND_DOWN
    )
    if cfg.usdgo_earn_cap and quantity > money(cfg.usdgo_earn_cap):
        raise ValueError("Initial USDGO allocation exceeds the saved product quota")
    price = opening * (1 + cfg.slippage)
    buy_quantity = engine.rounded(float(trx_budget) / (price * (1 + cfg.buy_fee)))
    if quantity <= 0 or buy_quantity < cfg.min_quantity or buy_quantity * price < cfg.min_notional:
        raise ValueError("Initial half-position allocation is below market minimums")
    paid = quantity * money(ask) * (1 + money(cfg.fx_fee))
    book.convert(f"initial-usdgo-{timestamp}", "to_usdgo", paid, quantity, mark)
    earn.subscribe_usdgo(quantity, timestamp)
    earn.reconcile_usdgo()
    earn.conversions.append(
        {
            "timestamp": timestamp,
            "side": "to_usdgo",
            "quantity": float(quantity),
            "paid": float(paid),
            "received": float(quantity),
            "rate": ask,
            "fee": float(paid - quantity * money(ask)),
            "simulated": True,
            "reason": (
                "initial_half_position"
                if cfg.startup == "half_position"
                else "initial_allocated_position"
            ),
        }
    )
    earn.observe_conversion()
    engine.last_price, engine.last_sigma = price, float("nan")
    plan = engine.make_plan(
        "buy", timestamp, price, 0, timestamp + 60000, startup=True, reference=price
    )
    if plan is None or plan.ready_at > timestamp:
        raise ValueError("Initial TRX purchase must use immediately available startup USDT")
    engine.record("submitted", timestamp, id=plan.plan_id, quantity=plan.quantity)
    fill = engine.fill(
        plan.plan_id,
        plan.plan_id,
        timestamp,
        plan.quantity,
        price,
        plan.quantity * price * cfg.buy_fee,
    )
    fill.update(startup=True, confirmation_period=0)
    engine.initial_anchor_at = timestamp
    return {
        "mode": cfg.startup,
        "timestamp": timestamp,
        "price_source": "first_minute_open",
        "usdgo_budget_usdt": float(usdgo_budget),
        "usdgo_quantity": float(quantity),
        "trx_budget_including_fee_usdt": float(trx_budget),
        "trx_quantity": fill["quantity"],
        "trx_fill_price": price,
        "trx_fee_usdt": fill["fee"],
        "remaining_usdt": float(book.cash),
        "anchor": engine.anchor,
        "ordinary_signals_after_existing_warmup": True,
    }


def _replay(  # noqa: C901
    frame,
    config: ResearchConfig,
    progress=None,
    audit=None,
    engine_factory=None,
    price_reference=None,
):
    data = validate_minutes(frame)
    sigma = normalized_volatility(data, config.volatility_window).to_numpy()
    timestamps = pd.DatetimeIndex(data["date"]).as_unit("ms").asi8
    opens = data["open"].to_numpy()
    closes = data["close"].to_numpy()
    if config.price_position_enabled:
        price_reference = price_reference or build_price_reference(data, config)
        if (
            price_reference.lookback_days != config.price_position_lookback_days
            or price_reference.quantile != config.price_position_quantile
        ):
            raise ValueError("Price-position reference differs from saved parameters")
    if engine_factory is None:
        engine = Engine(config)
        earn = HistoricalEarn(engine.book, config, int(timestamps[0]))
    else:
        engine, earn = engine_factory(config, int(timestamps[0]))
    engine.audit = earn.audit = audit
    engine.earn = earn
    engine.prepare_cash = earn.prepare_cash
    if audit:
        audit.initial(int(timestamps[0]), float(opens[0]), config.initial_cash)
    startup_allocation = None
    if config.startup in ("half_position", "allocated_position"):
        startup_allocation = initialize_half_position(
            engine, earn, int(timestamps[0]), float(opens[0])
        )
    earn.sweep_cash(int(timestamps[0]))
    equity_curve, decisions = [], []
    peak = config.initial_cash
    drawdown = 0.0
    full_inventory_minutes = 0
    exposure_sum = 0.0
    last_progress = time.monotonic()
    start_trade_index = config.volatility_window
    reference_minutes = high_price_minutes = 0
    for index, (timestamp, opening, close) in enumerate(
        zip(timestamps, opens, closes, strict=True)
    ):
        if hasattr(engine, "before_minute"):
            engine.before_minute(int(timestamp))
        earn.advance(int(timestamp), float(closes[max(0, index - 1)]))
        if audit:
            audit.arrival(engine, int(timestamp))
        if engine.pending and timestamp >= engine.pending.ready_at:
            execution_price = opening * (
                1 + config.slippage if engine.pending.side == "buy" else 1 - config.slippage
            )
            if engine.executable(int(timestamp), float(execution_price)):
                plan = engine.pending
                quantity = engine.execution_quantity(float(execution_price))
                if plan.side == "buy":
                    quantity = min(
                        quantity,
                        engine.rounded(
                            float(engine.book.free_cash) / (execution_price * (1 + config.buy_fee))
                        ),
                    )
                if quantity * execution_price >= config.min_notional:
                    engine.record("submitted", int(timestamp), id=plan.plan_id, quantity=quantity)
                    fee = (
                        quantity
                        * execution_price
                        * (config.buy_fee if plan.side == "buy" else config.sell_fee)
                    )
                    engine.fill(
                        plan.plan_id,
                        plan.plan_id,
                        int(timestamp),
                        quantity,
                        float(execution_price),
                        fee,
                    )
            else:
                rejection = engine.execution_reason(int(timestamp), float(execution_price))
                engine.record(
                    "execution_check", int(timestamp), id=engine.pending.plan_id, reason=rejection
                )
                engine.audit_end_reason = rejection
                engine.finish(int(timestamp))
                engine.reason = "execution_conditions_changed"
        close_time = int(timestamp + 60000)
        if config.price_position_enabled:
            engine.price_position_day = (close_time + 480 * 60000) // 86400000
            engine.price_position_reference = price_reference.value_at(close_time)
            if engine.price_position_reference is not None:
                reference_minutes += 1
                high_price_minutes += close > engine.price_position_reference
        earn.advance(close_time, float(close))
        if audit:
            audit.before_close(engine)
        plan = engine.on_close(close_time, float(close), float(sigma[index]))
        earn.allocate_cash(close_time, plan)
        if plan:
            decisions.append(
                {
                    "timestamp": close_time,
                    "side": plan.side,
                    "anchor": plan.anchor,
                    "period": engine.buy_period if plan.side == "buy" else engine.sell_period,
                    "threshold": plan.threshold,
                    "quantity": plan.quantity,
                }
            )
        equity = float(engine.book.equity(float(close)))
        if audit:
            audit.sample(engine, earn, close_time, float(close), equity)
        stable_value = float(
            engine.book.cash + engine.book.usdgo * (engine.book.usdgo_mark or money(0))
        )
        exposure_sum += (equity - stable_value) / equity
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
        if earn.liquid_cash() < config.min_notional:
            full_inventory_minutes += 1
        if index % 60 == 0 or index == len(data) - 1:
            equity_curve.append(
                {
                    "timestamp": close_time,
                    "price": float(close),
                    "equity": equity,
                    "principal_trx": float(engine.book.principal),
                    "cash": float(engine.book.cash),
                    "usdgo": float(engine.book.usdgo),
                    "usdgo_mark": float(engine.book.usdgo_mark or money(0)),
                    "interest_trx": float(engine.book.interest_trx),
                    "earn_income_credited_value": earn.summary(float(close))[
                        "earn_income_credited_value"
                    ],
                    "anchor": engine.anchor,
                    "realized_profit": float(engine.book.realized),
                }
            )
        if progress and time.monotonic() - last_progress > 0.5:
            progress(index / len(data))
            last_progress = time.monotonic()
    # The last close can produce a signal, but the next minute open is not in the dataset.
    if engine.pending:
        engine.audit_end_reason = "end_of_data"
        engine.finish(int(timestamps[-1] + 60000))
        engine.reason = "end_of_data_unexecuted_signal"
    earn.sweep_cash(int(timestamps[-1] + 60000))
    summary = engine.book.summary(float(closes[-1]))
    summary.update(earn.summary(float(closes[-1])))
    summary["mean_trx_exposure"] = exposure_sum / len(data)
    bought = sum(row["side"] == "buy" for row in engine.book.orders)
    sold = sum(row["side"] == "sell" for row in engine.book.orders)
    summary.update(
        {
            "max_drawdown": drawdown,
            "buy_orders": bought,
            "sell_orders": sold,
            "full_inventory_minutes": full_inventory_minutes,
            "oldest_remaining_lot_days": max(
                (
                    (timestamps[-1] - lot.bought_at) / 86400000
                    for lot in engine.book.lots
                    if lot.quantity
                ),
                default=0,
            ),
            "cash_wait_minutes": int(
                np.searchsorted(timestamps, engine.book.orders[0]["timestamp"])
            )
            if engine.book.orders
            else len(data),
        }
    )
    benchmarks = (
        benchmark_equities(config, timestamps, opens, closes, start_trade_index)
        if engine_factory is None
        else {}
    )
    earn_assumptions = {
        "model": config.earn_model,
        "trx_apr": config.trx_apr,
        "usdgo_apr": config.usdgo_apr,
        "basis": "Fixed scenario APR / 365 / 24; only complete eligible hours are credited.",
        "timing": (
            "Subscribe immediately, accrue next UTC hour, pay after that full hour; "
            "resubscribe credited interest with the same delay. "
            "Redeemed quantities lose the incomplete hour."
        ),
        "usdgo": (
            "Only simplified_hourly: idle USDT converts at a fixed 1:1 research rate, "
            "zero FX fees and instant redemption; "
            "no quotas or verified historical product availability."
        ),
        "authorization": "Historical simulation only; no exchange Earn or account operation.",
    }
    result = {
        "strategy": "TRXPhase1Strategy",
        "mode": "historical_research",
        "config": config.model_dump(),
        "assumptions": ASSUMPTIONS,
        "earn_assumptions": earn_assumptions,
        "summary": summary,
        "start_utc": str(data["date"].iloc[0]),
        "end_utc": str(data["date"].iloc[-1]),
        "warmup_minutes": min(config.volatility_window, len(data)),
        "candles": len(data),
        "data_fingerprint": data_fingerprint(data),
        "orders": engine.book.order_summary(),
        "fills": engine.book.orders,
        "rebate_receipts": engine.book.rebate_receipts,
        "interest_receipts": earn.receipts,
        "cash_conversions": earn.conversions,
        "cash_forecasts": earn.policy.forecasts if earn.policy is not None else [],
        "mechanisms": config.mechanisms,
        "decisions": decisions,
        "equity_curve": equity_curve,
        "lots": [
            {
                **lot.state(),
                "quantity": float(lot.quantity),
                "unit_cost": float(lot.unit_cost),
                "unrealized_profit": float(lot.quantity)
                * (float(closes[-1]) - float(lot.unit_cost)),
            }
            for lot in engine.book.lots
        ],
        "benchmarks": benchmarks,
        "limitations": [
            "Minute OHLCV cannot reconstruct order-book depth or intraminute order sequence.",
            "Fees and slippage are explicit research inputs, not your verified account fees.",
            (
                "Earn, when enabled, uses user-defined research rates and timing; "
                "it does not reconstruct verified historical product returns."
            ),
            "End positions remain marked to market; no forced exit is counted as a strategy sale.",
            (
                "Daily fee rebates are simulated USDT credits, not verified account receipts. "
                "Unpaid rebates are excluded from equity and order profitability checks."
            ),
        ],
    }
    if startup_allocation is not None:
        result["startup_allocation"] = startup_allocation
    if config.sell_mode == "recent_lot_then_core":
        result["sell_policy"] = {
            "mode": config.sell_mode,
            "core_lot_ids": engine.core_lot_ids,
            "events": engine.sell_policy_events,
            "definition": (
                "One newest profitable added lot's actual remaining quantity per sell signal; "
                "no P75 quantity multiplier or core spillover. Any remaining added lot, "
                "including unprofitable or untradable dust, blocks core sales. Only later "
                "signals with no added lots use the original proportional core sale. "
                "Shared fill anchors, confirmations, costs and execution profit checks remain."
            ),
        }
    if config.price_position_enabled:
        result["price_position"] = {
            **price_reference.describe(int(timestamps[0]), int(timestamps[-1] + 60000)),
            "reference_available_minutes": int(reference_minutes),
            "high_price_minutes": int(high_price_minutes),
            "sizing_events": engine.price_position_events,
            "startup_excluded": True,
        }
    if hasattr(engine, "research_state"):
        result["funding_research"] = engine.research_state()
        result["config"] = engine.initial_config
    if audit:
        from freqtrade.strategy.trx_phase1.studies import rule_fingerprint

        result["provenance"] = {
            "rules_sha256": rule_fingerprint(),
            "research_stage": "observed_history",
            "exchange": "bitget",
            "market": "spot",
            "pair": "TRX/USDT",
            "timeframe": "1m",
        }
        result["diagnostics"] = audit.finish(result, engine, earn)
    return result


def benchmark_equities(config, timestamps, opens, closes, start_trade_index):
    hold_index = min(start_trade_index, len(timestamps) - 1)
    entry_price = float(opens[hold_index]) * (1 + config.slippage)
    hold_at = int(timestamps[hold_index])
    zero_book = Ledger(config.initial_cash)
    zero_engine = Engine(config, zero_book)
    quantity = zero_engine.rounded(config.initial_cash / (entry_price * (1 + config.buy_fee)))
    fee = money(quantity) * money(entry_price) * money(config.buy_fee)
    zero_book.buy("hold", "hold", hold_at, quantity, entry_price, fee)
    zero_book.queue_fee_rebate(
        "hold", fee * money(config.fee_rebate_ratio), config.rebate_due_at(hold_at)
    )
    zero_book.settle_fee_rebates(int(timestamps[-1] + 60000))

    hold_book = Ledger(config.initial_cash)
    hold_engine = Engine(config, hold_book)
    hold_earn = HistoricalEarn(hold_book, config, int(timestamps[0]))
    cash_book = Ledger(config.initial_cash)
    cash_earn = HistoricalEarn(cash_book, config, int(timestamps[0]))
    hold_earn.sweep_cash(int(timestamps[0]))
    cash_earn.sweep_cash(int(timestamps[0]))
    requested = False
    purchased = False
    for index, (timestamp, close) in enumerate(zip(timestamps, closes, strict=True)):
        hold_earn.advance(int(timestamp), float(closes[max(0, index - 1)]))
        if index >= hold_index and not requested:
            budget = hold_earn.liquid_cash()
            hold_earn.prepare_cash("benchmark-hold", int(timestamp), budget)
            requested = True
        if requested and not purchased and hold_book.free_cash >= config.min_notional:
            entry_price = float(opens[index]) * (1 + config.slippage)
            amount = hold_engine.rounded(
                float(hold_book.free_cash) / (entry_price * (1 + config.buy_fee))
            )
            buy_fee = money(amount) * money(entry_price) * money(config.buy_fee)
            hold_book.buy("hold", "hold", int(timestamp), amount, entry_price, buy_fee)
            hold_book.queue_fee_rebate(
                "hold",
                buy_fee * money(config.fee_rebate_ratio),
                config.rebate_due_at(int(timestamp)),
            )
            purchased = True
        close_time = int(timestamp + 60000)
        hold_book.settle_fee_rebates(close_time)
        hold_earn.advance(close_time, float(close))
        hold_earn.sweep_cash(close_time)
        cash_earn.advance(close_time, float(close))
    hold_equity = float(zero_book.equity(float(closes[-1])))
    return {
        "cash_zero_interest_equity": config.initial_cash,
        "hold_trx_zero_interest_equity": hold_equity,
        "hold_trx_return": hold_equity / config.initial_cash - 1,
        "hold_trx_fee_rebates": float(zero_book.fee_rebates),
        "hold_trx_pending_fee_rebates": zero_book.summary(float(closes[-1]))["pending_fee_rebates"],
        "cash_with_earn_equity": float(cash_book.equity(float(closes[-1]))),
        "hold_trx_with_earn_equity": float(hold_book.equity(float(closes[-1]))),
        "hold_benchmark_filled": purchased,
    }


def replay(
    frame,
    config: ResearchConfig,
    progress=None,
    diagnostic_path=None,
    recorder_factory=Recorder,
    engine_factory=None,
    price_reference=None,
):
    from freqtrade.strategy.trx_phase1.studies import rule_fingerprint

    source = rule_fingerprint() if diagnostic_path else None
    audit = recorder_factory(diagnostic_path) if diagnostic_path else None
    try:
        result = _replay(frame, config, progress, audit, engine_factory, price_reference)
        if source and source != rule_fingerprint():
            raise ValueError(
                "Research source changed during replay; diagnostic result not published"
            )
        return result
    except BaseException:
        if audit:
            audit.abort()
            audit.path.unlink(missing_ok=True)
        raise
