from decimal import Decimal

from freqtrade.strategy.trx_phase1.stablecoin_simulation import Product, allocate


def test_marginal_tiers_are_shared_and_usdt_wins_ties():
    d = Decimal
    products = {
        "t": Product("USDT", ((d(0), d(1000), d(".12")), (d(1000), d(9000), d(".04"))), d(1), 8),
        "c": Product("USDC", ((d(0), d(500), d(".12")), (d(500), d(9000), d(".035"))), d(1), 8),
        "g": Product("USDGO", ((d(0), d(9000), d(".09")),), d(1), 8),
    }
    targets = allocate(
        products,
        {p.coin: d(1) for p in products.values()},
        {"strategy": d(1200), "outside": d(600)},
    )

    assert targets == {
        "strategy": {"t": d(1000), "c": d(200)},
        "outside": {"c": d(300), "g": d(300)},
    }
