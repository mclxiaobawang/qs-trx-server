from decimal import Decimal

import pandas as pd
import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.ledger import Ledger
from freqtrade.strategy.trx_phase1.replay import replay
from freqtrade.strategy.trx_phase1.research_earn import HOUR, HOURS_PER_YEAR, HistoricalEarn
from tests.strategy.trx_phase1.test_phase1 import candles, research


def test_trx_hourly_delay_retention_and_reinvestment_do_not_change_principal_cost():
    book = Ledger(1000)
    book.buy("buy", "buy", 20 * 60000, 100, 1)
    earn = HistoricalEarn(book, ResearchConfig(earn_model="trx_hourly"), 0)
    earn.advance(HOUR, 2)
    assert book.interest_trx == 0
    earn.advance(2 * HOUR, 2)
    hourly = Decimal(100) * Decimal("0.1") / HOURS_PER_YEAR
    assert float(book.interest_trx) == pytest.approx(float(hourly))
    assert earn.receipts[0]["timestamp"] == 2 * HOUR
    earn.advance(2 * HOUR, 2)
    assert len(earn.receipts) == 1
    earn.advance(3 * HOUR, 2)
    assert float(book.interest_trx) == pytest.approx(float(2 * hourly))
    assert book.cost == 100 and book.lots[0].unit_cost == 1
    allocation = book.qualifying(2, 100, 0)
    book.sell("sell", "sell", 3 * HOUR + 60000, 100, 2, allocation)
    assert book.principal == 0
    earn.advance(4 * HOUR, 3)
    assert earn.receipts[-1]["eligible_quantity"] == pytest.approx(float(hourly))
    assert book.interest_trx > 2 * hourly
    assert earn.summary(3)["trx_interest_price_pnl"] != 0


def test_usdgo_conversion_credits_and_redemption_conserve_cash_and_do_not_double_count_income():
    book = Ledger(1000)
    earn = HistoricalEarn(book, ResearchConfig(earn_model="simplified_hourly"), 0)
    earn.sweep_cash(0)
    assert book.cash == 0 and book.usdgo == 1000 and book.equity(1) == 1000
    earn.advance(HOUR, 1)
    assert book.stable_interest == 0
    earn.advance(2 * HOUR, 1)
    hourly = Decimal(1000) * Decimal("0.12") / HOURS_PER_YEAR
    assert float(book.stable_interest) == pytest.approx(float(hourly))
    assert book.usdgo_cost == book.usdgo
    earn.prepare_cash("buy-plan", 2 * HOUR, 400)
    assert book.cash == 400
    assert float(book.equity(1)) == pytest.approx(1000 + float(hourly))
    assert book.fx_profit == 0
    assert book.usdgo == earn.usdgo_eligible + sum(
        (entry[1] for entry in earn.usdgo_waiting), Decimal(0)
    )
    earn.sweep_cash(2 * HOUR, reserved=300)
    assert book.cash == 300
    assert earn.liquid_cash() == pytest.approx(1000 + float(hourly))


def test_usdgo_interest_requires_valuation_and_is_idempotent():
    book = Ledger(100)
    with pytest.raises(ValueError, match="value"):
        book.credit_interest("income", "USDGO", 2)
    book.credit_interest("income", "USDGO", 2, fx=1)
    book.credit_interest("income", "USDGO", 2, fx=1)
    book.convert("redeem", "to_usdt", 2, 2, 1)
    assert book.cash == 102 and book.stable_interest == 2 and book.fx_profit == 0
    restored = Ledger.restore(book.state())
    assert restored.summary(1) == book.summary(1)


def test_usdgo_roundoff_can_reconcile_without_creating_money_and_material_mismatch_fails():
    book = Ledger(1000)
    earn = HistoricalEarn(book, ResearchConfig(earn_model="simplified_hourly"), 0)
    earn.sweep_cash(0)
    for hour in range(1, 601):
        earn.advance(hour * HOUR, 1)
        if hour % 17 == 0:
            before = book.equity(1)
            earn.prepare_cash(f"full-redemption-{hour}", hour * HOUR, 100000)
            assert book.usdgo == 0
            assert abs(earn.usdgo_eligible) < Decimal("1e-18")
            assert abs(sum((entry[1] for entry in earn.usdgo_waiting), Decimal(0))) < Decimal(
                "1e-18"
            )
            assert book.equity(1) == before
            earn.sweep_cash(hour * HOUR)
    before = book.state()
    earn.usdgo_eligible -= Decimal("1e-24")
    earn.reconcile_usdgo()
    assert book.state() == before
    earn.usdgo_eligible -= Decimal("0.01")
    with pytest.raises(ValueError, match="do not reconcile"):
        earn.reconcile_usdgo()


@pytest.mark.parametrize("variant", ["fixed", "dynamic_size", "dynamic_period"])
@pytest.mark.parametrize("model", ["trx_hourly", "simplified_hourly"])
def test_earn_replay_is_causal_and_separately_accounts_for_retained_interest(variant, model):
    config = research(variant=variant, startup="small_buy", earn_model=model)
    data = candles(1000)
    full, prefix = replay(data, config), replay(data.iloc[:600], config)
    cutoff = int((data.date.iloc[599] + pd.Timedelta("1min")).timestamp() * 1000)
    assert [row for row in full["interest_receipts"] if row["timestamp"] <= cutoff] == prefix[
        "interest_receipts"
    ]
    assert [row for row in full["fills"] if row["timestamp"] < cutoff] == prefix["fills"]
    assert [row for row in full["equity_curve"] if row["timestamp"] < cutoff] == [
        row for row in prefix["equity_curve"] if row["timestamp"] < cutoff
    ]
    summary = full["summary"]
    assert summary["interest_trx"] > 0
    assert summary["equity"] == pytest.approx(
        summary["cash"]
        + summary["usdgo_market_value"]
        + (summary["principal_trx"] + summary["interest_trx"]) * data.close.iloc[-1]
    )
    assert summary["total_profit"] == pytest.approx(
        summary["realized_profit"]
        + summary["unrealized_profit"]
        + summary["stable_interest"]
        + summary["trx_interest_market_value"]
        + summary["fx_profit"]
        + summary["usdgo_unrealized_profit"]
    )
    assert all(
        allocation["profit"] > 0
        for sale in full["orders"]
        if sale["side"] == "sell"
        for allocation in sale["allocations"]
    )
    assert summary["usdgo_interest_quantity"] == pytest.approx(
        sum(row["quantity"] for row in full["interest_receipts"] if row["coin"] == "USDGO")
    )
    if model == "trx_hourly":
        assert summary["usdgo_interest_quantity"] == 0


@pytest.mark.parametrize("variant", ["fixed", "dynamic_size", "dynamic_period"])
def test_zero_rates_reproduce_the_original_trading_path(variant):
    config = research(variant=variant, startup="small_buy")
    data = candles(600)
    zero = replay(data, config)
    scenario = replay(
        data,
        config.model_copy(update={"earn_model": "simplified_hourly", "trx_apr": 0, "usdgo_apr": 0}),
    )
    assert scenario["fills"] == zero["fills"]
    for field in (
        "equity",
        "total_profit",
        "remaining_cost",
        "principal_trx",
        "realized_profit",
        "full_inventory_minutes",
    ):
        assert scenario["summary"][field] == pytest.approx(zero["summary"][field])
    assert scenario["interest_receipts"] == []


def test_identical_rate_benchmarks_pay_only_completed_hours():
    data = candles(180)
    for column in ("open", "high", "low", "close"):
        data[column] = 1.0
    config = research(earn_model="simplified_hourly", buy_fee=0, sell_fee=0, slippage=0)
    result = replay(data, config)
    assert result["benchmarks"]["cash_with_earn_equity"] == pytest.approx(
        1000 + 2 * 1000 * 0.12 / HOURS_PER_YEAR
    )
    assert result["benchmarks"]["hold_trx_with_earn_equity"] == pytest.approx(
        1000 + 2 * 1000 * 0.1 / HOURS_PER_YEAR
    )
    shorter = replay(data.iloc[:119], config)
    assert shorter["interest_receipts"] == []
    assert shorter["summary"]["total_profit"] == 0


def test_legacy_reports_stay_zero_earn_and_paper_can_explicitly_use_the_shared_model(tmp_path):
    from freqtrade.strategy.trx_phase1.runtime import PaperRuntime

    assert ResearchConfig.from_saved({"version": "trx-phase1-v0.2"}).earn_model == "none"
    runtime = PaperRuntime(
        tmp_path, tmp_path / "unused.feather", None, ResearchConfig(earn_model="simplified_hourly")
    )
    assert runtime.engine.config.earn_model == "simplified_hourly"
    assert runtime.running is False
