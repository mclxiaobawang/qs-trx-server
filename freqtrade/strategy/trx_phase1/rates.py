"""Quantity-tier research rates shared by actual accrual and causal forecasts."""

from freqtrade.strategy.trx_phase1.ledger import money


def annual_interest(config, coin, quantity):
    quantity = money(quantity)
    if quantity < 0:
        raise ValueError("Interest-bearing quantity cannot be negative")
    tiers = getattr(config, coin.lower() + "_apr_tiers")
    if not tiers:
        return (
            quantity
            * money(getattr(config, coin.lower() + "_apr"))
            * money(config.apr_scenario_scale)
        )
    total, previous, remaining = money(0), money(0), quantity
    for cap, rate in tiers:
        width = money(cap) - previous
        part = min(remaining, width)
        total += part * money(rate) * money(config.apr_scenario_scale)
        remaining -= part
        previous = money(cap)
        if not remaining:
            return total
    # Above published rate coverage, grant no unverified yield.
    return total


def effective_apr(config, coin, quantity):
    return float(annual_interest(config, coin, quantity) / money(quantity)) if quantity else 0
