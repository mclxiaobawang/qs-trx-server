"""Cash reinvestment accepts normal precision loss while preserving funding limits."""

from copy import deepcopy

import pytest

from freqtrade.strategy.trx_phase1.config import ResearchConfig
from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_funding import LiveFunding
from tests.strategy.trx_phase1.test_live_execution import RULES


def funding():
    book = Ledger(10000)
    book.cash = money("171.7106353739706")
    book.usdgo = book.locked["USDGO"] = money("8583.87308165")
    execution = {
        "book": book.state(),
        "products": deepcopy(RULES),
        "holdings": {
            "pool": {
                "coin": "USDGO",
                "principal": "8583.87308165",
                "interest": "0",
                "unassigned_amount": "19968.10820856",
            }
        },
        "unassigned_offsets": {"USDT": "1000000"},
    }
    market = {
        "bid": "1.0013",
        "ask": "1.0014",
        "fee": "0",
        "price_decimals": 4,
        "quantity_decimals": 4,
    }
    terms = {
        "singleMinAmount": ".1",
        "singleMaxAmount": "10000000",
        "remainingAmount": "10000000",
        "subscribePrecision": "8",
        "tiers": [{"minStepVal": "0", "maxStepVal": "100000", "currentApy": "9"}],
    }
    return LiveFunding(
        execution,
        ResearchConfig(),
        {"USDGOUSDT": market},
        {"USDGO": terms, "TRX": deepcopy(terms)},
    )


def test_sale_cash_with_fractional_fx_can_reinvest_without_using_external_money():
    f = funding()
    before = deepcopy(f.execution)
    steps = f.idle(4 * 3600000, 0)
    assert len(steps) == 2
    assert steps[0]["symbol"] == "USDGOUSDT" and steps[0]["side"] == "buy"
    assert money(steps[0]["amount"]) == money("171.7106353739706")
    assert (
        steps[1]["kind"] == "subscribe"
        and steps[1]["coin"] == "USDGO"
        and steps[1]["amount"] == "previous"
    )
    assert f.execution == before


@pytest.mark.parametrize(
    "fault", ["minimum", "capacity", "holdoff", "unprofitable", "cash_minimum"]
)
def test_cash_reinvestment_still_requires_valid_limits_and_net_benefit(fault):
    f = funding()
    last = 0
    if fault == "minimum":
        f.terms["USDGO"]["singleMinAmount"] = "200"
    elif fault == "capacity":
        f.terms["USDGO"]["remainingAmount"] = "100"
    elif fault == "holdoff":
        last = 4 * 3600000 - 1000
    elif fault == "unprofitable":
        f.markets["USDGOUSDT"]["bid"] = ".99"
    else:
        f.book.cash = money("4")
    assert f.idle(4 * 3600000, last) == []
