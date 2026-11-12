"""Offline old-account funding replay. Inputs are previously collected read-only snapshots."""

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path

from freqtrade.strategy.trx_phase1.engine import Plan
from freqtrade.strategy.trx_phase1.ledger import money


ZERO = Decimal(0)
OWNERS = ("strategy", "outside")
STABLES = ("USDT", "USDC", "USDGO")
REDEMPTION_INTERVAL_SECONDS = 61


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def rounded(value, decimals, mode=ROUND_DOWN):
    return value.quantize(Decimal(1).scaleb(-decimals), rounding=mode)


@dataclass(frozen=True)
class Product:
    coin: str
    tiers: tuple
    minimum: Decimal
    decimals: int


@dataclass
class Wallet:
    spot: dict = field(default_factory=lambda: defaultdict(Decimal))
    earn: dict = field(default_factory=lambda: defaultdict(Decimal))

    def value(self, products, marks):
        return sum((self.spot[c] * marks[c] for c in STABLES), ZERO) + sum(
            (amount * marks[products[pid].coin] for pid, amount in self.earn.items()), ZERO
        )


class Market:
    def __init__(self, raw, quote_mark):
        definition = raw["definition"]
        require(definition["status"] == "online", "market_unavailable")
        self.decimals = int(definition["quantityPrecision"])
        self.minimum = money(definition["minTradeUSDT"])
        self.fee = money(raw["fee"]["takerFeeRate"])
        self.quote_mark = quote_mark
        self.bids = [list(map(money, row)) for row in raw["depth"]["bids"]]
        self.asks = [list(map(money, row)) for row in raw["depth"]["asks"]]

    def quantity_for(self, side, value):
        quantity = ZERO
        for price, available in self.bids if side == "sell" else self.asks:
            take = min(available, value / price)
            quantity += take
            value -= take * price
            if value <= Decimal("1e-21"):
                return quantity
        raise ValueError("insufficient_market_depth")

    def fill(self, side, quantity):
        remaining, total = quantity, ZERO
        updates = []
        for row in self.bids if side == "sell" else self.asks:
            price, available = row
            take = min(available, remaining)
            if take:
                total += price * take
                remaining -= take
                updates.append((row, take))
            if not remaining:
                break
        require(not remaining, "insufficient_market_depth")
        require(quantity > 0 and total * self.quote_mark >= self.minimum, "below_minimum_trade")
        for row, take in updates:
            row[1] -= take
        return total


class Simulation:
    def __init__(self, snapshot, bill_evidence):
        state = snapshot["ledger_after"]
        require(state == snapshot["ledger_before"], "snapshot_ledger_changed")
        require(
            not state["operation"] and not state["error"] and not state["book"].get("reserved"),
            "execution_unsettled",
        )
        self.book = state["book"]
        self.wallets = {owner: Wallet() for owner in OWNERS}
        self.load_ownership(snapshot, state, bill_evidence)
        self.products = self.load_products(snapshot)
        raw = snapshot["markets"]
        self.marks = {
            coin: money(raw[f"{coin}USDT"]["depth"]["bids"][0][0])
            for coin in ("USDGO", "USDC", "TRX")
        } | {"USDT": Decimal(1)}
        self.markets = {
            symbol: Market(row, self.marks["USDC"] if symbol == "USDGOUSDC" else Decimal(1))
            for symbol, row in raw.items()
        }
        self.events = []

    def load_ownership(self, snapshot, state, bill_evidence):
        strategy, outside = self.wallets.values()
        actual_spot = {}
        for row in snapshot["spot"]:
            if row["coin"] in STABLES:
                require(
                    all(not money(row.get(k) or 0) for k in ("frozen", "locked", "limitAvailable")),
                    "exchange_spot_not_free",
                )
                actual_spot[row["coin"]] = money(row["available"])
        strategy.spot.update(
            USDT=money(self.book["free_cash"]),
            USDGO=money(self.book["usdgo"]) - money(state["locked"]["USDGO"]),
        )
        for coin in ("USDT", "USDGO"):
            outside.spot[coin] = money(state["unassigned_offsets"][coin])
        external = state["external_usdc"]
        require(external["outside_strategy"], "unverified_usdc_ownership")
        usdc_gap = actual_spot["USDC"] - money(external["spot"])
        interest = [
            r
            for r in bill_evidence["rows"]
            if r["businessType"] == "BATCH_INTEREST_USER_IN"
            and money(r["size"]) == usdc_gap
            and money(r["balance"]) == actual_spot["USDC"]
            and int(r["cTime"]) <= snapshot["balances_observed_at"]
            and money(r["fees"]) == 0
        ]
        require(usdc_gap == 0 or (usdc_gap > 0 and len(interest) == 1), "unverified_usdc_spot")
        outside.spot["USDC"] = actual_spot["USDC"]
        for coin in STABLES:
            require(
                strategy.spot[coin] >= 0
                and outside.spot[coin] >= 0
                and strategy.spot[coin] + outside.spot[coin] == actual_spot[coin],
                "spot_ownership_mismatch",
            )

        matched_orders = set()
        overview = {r["coin"]: money(r["amount"]) for r in snapshot["earn_overview"]}
        venue_earn = defaultdict(Decimal)
        for row in snapshot["savings"]:
            coin, pid, oid = row["productCoin"], row["productId"], row["orderId"]
            if coin not in STABLES:
                continue
            require(oid not in matched_orders, "duplicate_savings_order")
            matched_orders.add(oid)
            total = money(row["holdAmount"])
            venue_earn[coin] += total
            saved = state["holdings"].get(oid)
            if saved and saved["coin"] == coin and saved["product_id"] == pid:
                owned = money(saved["principal"])
                unassigned = money(saved.get("unassigned_amount") or 0)
                require(not money(saved.get("reserved_amount") or 0), "savings_reserved")
            else:
                require(
                    coin == "USDC" and pid == external["product_id"] and not outside.earn[pid],
                    "unverified_savings_owner",
                )
                owned, unassigned = ZERO, money(external["savings"])
            require(
                owned >= 0 and unassigned >= 0 and owned + unassigned == total,
                "savings_ownership_mismatch",
            )
            strategy.earn[pid] += owned
            outside.earn[pid] += unassigned
        for coin in STABLES:
            require(venue_earn[coin] == overview.get(coin, ZERO), "savings_overview_mismatch")
        for coin in ("USDGO", "USDT"):
            owned = sum(
                (money(p["principal"]) for p in state["holdings"].values() if p["coin"] == coin),
                ZERO,
            )
            require(owned == money(state["locked"][coin]), "strategy_earn_lock_mismatch")

    def load_products(self, snapshot):
        held = defaultdict(Decimal)
        for wallet in self.wallets.values():
            for pid, quantity in wallet.earn.items():
                held[pid] += quantity
        products = {}
        for row in snapshot["products"]:
            pid, coin = row["productId"], row["coin"]
            if (
                coin not in STABLES
                or row["productLevel"] != "normal"
                or row["status"] != "in_progress"
                or row["periodType"] != "flexible"
            ):
                continue
            terms = snapshot["terms"][pid]
            cap = min(money(terms["singleMaxAmount"]), held[pid] + money(terms["remainingAmount"]))
            tiers, cursor, previous = [], ZERO, Decimal(100)
            for band in terms["apyList"]:
                lo, hi, rate = map(
                    money, (band["minStepVal"], band["maxStepVal"], band["currentApy"])
                )
                require(lo == cursor and hi > lo and 0 <= rate <= previous, "invalid_apy_tiers")
                if lo < cap:
                    tiers.append((lo, min(hi, cap), rate / 100))
                cursor, previous = hi, rate
            if tiers:
                products[pid] = Product(
                    coin,
                    tuple(tiers),
                    money(terms["singleMinAmount"]),
                    int(terms["subscribePrecision"]),
                )
        require(all(pid in products for pid, qty in held.items() if qty), "missing_held_terms")
        return products

    def equity(self):
        trx = money(self.book["principal"]) + money(self.book["interest_trx"])
        return self.wallets["strategy"].value(self.products, self.marks) + trx * self.marks["TRX"]

    def daily_interest(self):
        prior = defaultdict(Decimal)
        daily = {}
        for owner, wallet in self.wallets.items():
            total = ZERO
            for pid, qty in wallet.earn.items():
                product = self.products[pid]
                end = prior[pid] + qty
                for lo, hi, rate in product.tiers:
                    total += (
                        max(ZERO, min(hi, end) - max(lo, prior[pid]))
                        * rate
                        * self.marks[product.coin]
                        / 365
                    )
                prior[pid] = end
            daily[owner] = total
        return daily

    def targets(self):
        """Allocate each marginal band once across both owners."""
        return allocate(
            self.products,
            self.marks,
            {
                owner: wallet.value(self.products, self.marks)
                for owner, wallet in self.wallets.items()
            },
        )

    def move(self, owner, pid, qty, kind, time=None):
        wallet, product = self.wallets[owner], self.products[pid]
        if not qty:
            return
        require(qty > 0, "invalid_movement_quantity")
        coin = product.coin
        if kind == "redeem":
            require(wallet.earn[pid] >= qty, "insufficient_owned_savings")
            wallet.earn[pid] -= qty
            wallet.spot[coin] += qty
        else:
            require(wallet.spot[coin] >= qty >= product.minimum, "invalid_subscription_amount")
            total = sum((w.earn[pid] for w in self.wallets.values()), ZERO)
            require(total + qty <= product.tiers[-1][1], "shared_product_cap")
            wallet.spot[coin] -= qty
            wallet.earn[pid] += qty
        self.events.append(
            {
                "owner": owner,
                "kind": kind,
                "coin": coin,
                "product": pid,
                "amount": qty,
                "at_s": time,
            }
        )

    def swap(self, owner, base, quote, side, qty, time=None):
        market, wallet = self.markets[base + quote], self.wallets[owner]
        require(market.fee == 0, "stablecoin_route_requires_zero_fee")
        if side == "sell":
            require(wallet.spot[base] >= qty, "insufficient_owned_spot")
            received = market.fill(side, qty)
            wallet.spot[base] -= qty
            wallet.spot[quote] += received
        else:
            # Determine the exact cost on a copy before consuming any depth.
            cost = deepcopy(market).fill(side, qty)
            require(wallet.spot[quote] >= cost, "insufficient_owned_spot")
            market.fill(side, qty)
            wallet.spot[quote] -= cost
            wallet.spot[base] += qty
            received = qty
        self.events.append(
            {
                "owner": owner,
                "kind": "swap",
                "symbol": base + quote,
                "side": side,
                "base_quantity": qty,
                "receive": received,
                "at_s": time,
            }
        )
        return received

    def rebalance(self, targets):
        # This replay only executes the conversion graph validated against this snapshot.
        desired = {
            (owner, self.products[p].coin): (p, q)
            for owner, rows in targets.items()
            for p, q in rows.items()
            if q
        }
        require(
            sum(bool(q) for rows in targets.values() for q in rows.values()) == 3
            and set(desired) == {("strategy", "USDT"), ("strategy", "USDC"), ("outside", "USDC")},
            "tier_plan_outside_validated_conversion_route",
        )
        usdt, reserve = desired["strategy", "USDT"]
        usdc, _ = desired["strategy", "USDC"]
        require(usdc == desired["outside", "USDC"][0], "usdc_products_differ")
        # Use account headroom through the selected marginal band, even if venue capacity is higher.
        usdc_ceiling = next(
            hi
            for lo, hi, _ in self.products[usdc].tiers
            if sum(row.get(usdc, ZERO) for row in targets.values()) <= hi
        )
        for owner, wallet in self.wallets.items():
            for pid, qty in list(wallet.earn.items()):
                if pid != usdc:
                    self.move(owner, pid, qty, "redeem")
            goal = reserve if owner == "strategy" else ZERO
            market = self.markets["USDGOUSDT"]
            if owner == "strategy" and wallet.spot["USDT"] < goal:
                needed = market.quantity_for("sell", goal - wallet.spot["USDT"])
                self.swap(
                    owner, "USDGO", "USDT", "sell", rounded(needed, market.decimals, ROUND_UP)
                )
            direct = self.markets["USDGOUSDC"]
            remainder = rounded(wallet.spot["USDGO"], direct.decimals)
            if remainder * direct.bids[0][0] * direct.quote_mark >= direct.minimum:
                self.swap(owner, "USDGO", "USDC", "sell", remainder)
            if owner == "outside" and wallet.spot["USDT"] >= self.markets["USDCUSDT"].minimum:
                market = self.markets["USDCUSDT"]
                qty = rounded(market.quantity_for("buy", wallet.spot["USDT"]), market.decimals)
                self.swap(owner, "USDC", "USDT", "buy", qty)
            if goal:
                self.move(owner, usdt, goal, "subscribe")
            amount = rounded(wallet.spot["USDC"], self.products[usdc].decimals)
            if amount >= self.products[usdc].minimum:
                require(
                    sum(w.earn[usdc] for w in self.wallets.values()) + amount <= usdc_ceiling,
                    "fills_exceed_selected_usdc_band",
                )
                self.move(owner, usdc, amount, "subscribe")
        return usdt, usdc


def allocate(products, marks, budgets):
    bands = sorted(
        (-rate, p.coin != "USDT", pid, lo, hi)
        for pid, p in products.items()
        for lo, hi, rate in p.tiers
    )
    shared = defaultdict(Decimal)
    targets = {owner: defaultdict(Decimal) for owner in OWNERS}
    for owner in OWNERS:
        remaining = budgets[owner]
        for _, _, pid, lo, hi in bands:
            coin = products[pid].coin
            require(shared[pid] >= lo, "non_contiguous_allocation")
            quantity = min(hi - shared[pid], remaining / marks[coin])
            if quantity:
                targets[owner][pid] += quantity
                shared[pid] += quantity
                remaining = max(ZERO, remaining - quantity * marks[coin])
            if not remaining:
                break
        require(remaining <= Decimal("1e-18"), "insufficient_verified_tier_capacity")
    return targets


def fund_buy(sim, plan, usdt_product, usdc_product, receipt_delays):
    """Replay a frozen buy plan's funding only. All times are simulated seconds."""
    wallet = sim.wallets["strategy"]
    budget = money(plan.budget)
    require(plan.side == "buy" and budget > 0, "invalid_buy_plan")
    clock, redemption_ready = plan.created_at // 1000, plan.created_at // 1000
    expiry = plan.expires_at // 1000
    status = "ready_for_original_buy"
    fx = sim.markets["USDCUSDT"]
    paid = ZERO
    quantity = ZERO
    try:
        require(clock < expiry, "plan_expired")
        gap = max(ZERO, budget - wallet.spot["USDT"])
        t = rounded(min(gap, wallet.earn[usdt_product]), 8)
        gap -= t
        c = ZERO
        if gap:
            required = max(gap, fx.minimum)
            c = rounded(fx.quantity_for("sell", required), fx.decimals, ROUND_UP)
            require(
                wallet.spot["USDC"] + wallet.earn[usdc_product] >= c,
                "insufficient_strategy_owned_usdc",
            )
        for pid, amount in (
            (usdt_product, t),
            (usdc_product, max(ZERO, c - wallet.spot["USDC"])),
        ):
            if not amount:
                continue
            clock = max(clock, redemption_ready)
            coin = sim.products[pid].coin
            delay = receipt_delays[coin]
            require(clock < expiry, "plan_expired")
            # One shared cooldown for both savings products.
            redemption_ready = clock + REDEMPTION_INTERVAL_SECONDS
            sim.events.append(
                {
                    "owner": "strategy",
                    "kind": "redeem_submitted",
                    "coin": coin,
                    "amount": amount,
                    "at_s": clock,
                }
            )
            if delay is None or clock + delay >= expiry:
                raise ValueError("await_reconciliation_no_buy")
            clock += delay
            sim.move("strategy", pid, amount, "redeem", clock)
        if c:
            require(clock < expiry, "plan_expired")
            sim.swap("strategy", "USDC", "USDT", "sell", c, time=clock)
        require(wallet.spot["USDT"] >= budget, "settled_cash_below_budget")
        require(clock < expiry, "plan_expired")
        trx = sim.markets["TRXUSDT"]
        quantity = rounded(trx.quantity_for("buy", budget / (1 + trx.fee)), trx.decimals)
        quote_cost = trx.fill("buy", quantity)
        average = quote_cost / quantity
        require(
            average <= money(plan.anchor) * (1 - money(plan.threshold)),
            "original_buy_price_condition_failed",
        )
        paid = quote_cost * (1 + trx.fee)
        require(paid <= budget, "buy_budget_exceeded")
    except ValueError as exc:
        status = str(exc)
        quantity = ZERO
    return {
        "status": status,
        "plan_budget_usdt": budget,
        "simulated_elapsed_seconds": clock - plan.created_at // 1000,
        "settled_strategy_usdt": wallet.spot["USDT"],
        "trx_quote_quantity": quantity,
        "trx_quote_total_with_fee": paid,
        "events": sim.events,
    }


def wallet_summary(sim):
    return {
        owner: {
            "spot": dict(wallet.spot),
            "earn": {sim.products[pid].coin: qty for pid, qty in wallet.earn.items() if qty},
            "stable_value_usdt": wallet.value(sim.products, sim.marks),
        }
        for owner, wallet in sim.wallets.items()
    }


def run(snapshot, bill_evidence):
    sim = Simulation(snapshot, bill_evidence)
    before, before_equity, before_daily = wallet_summary(sim), sim.equity(), sim.daily_interest()
    targets = sim.targets()
    usdt, usdc = sim.rebalance(targets)
    after, after_equity, after_daily = wallet_summary(sim), sim.equity(), sim.daily_interest()
    allocation_events = sim.events
    sim.events = []
    base_budget = money(sim.book["deployable"]) * money(snapshot["buy_config"]["buy_fraction"])
    timestamp = snapshot["balances_observed_at"] // 1000 * 1000
    # A synthetic, already-admitted funding fixture. It does not test signal generation.
    plan = Plan(
        "offline-funding-fixture",
        "buy",
        timestamp,
        timestamp + 180_000,
        0,
        0.33,
        float(sim.marks["TRX"]),
        snapshot["buy_config"]["buy_threshold"],
        0.0,
        budget=float(base_budget),
    )
    delays = {"USDT": 40, "USDC": 45}
    funded = fund_buy(deepcopy(sim), plan, usdt, usdc, delays)
    late = fund_buy(deepcopy(sim), plan, usdt, usdc, delays | {"USDC": 160})
    missing = fund_buy(deepcopy(sim), plan, usdt, usdc, delays | {"USDC": None})
    short = deepcopy(sim)
    short.wallets["strategy"].earn[usdc] = ZERO
    outside_only = fund_buy(short, plan, usdt, usdc, delays)
    adverse = deepcopy(plan)
    adverse.anchor = 0.324
    price_failed = fund_buy(deepcopy(sim), adverse, usdt, usdc, delays)
    with_reserve = deepcopy(sim)
    with_reserve.move("strategy", usdt, with_reserve.wallets["strategy"].earn[usdt], "redeem")
    reserve_daily = with_reserve.daily_interest()
    with_reserve.events = []
    funded_with_reserve = fund_buy(with_reserve, plan, usdt, usdc, delays)
    require(funded["status"] == "ready_for_original_buy", "fixture_did_not_fund")
    require(
        late["status"] == missing["status"] == "await_reconciliation_no_buy",
        "fixture_receipt_guard_failed",
    )
    require(
        outside_only["status"] == "insufficient_strategy_owned_usdc", "fixture_owner_guard_failed"
    )
    require(
        price_failed["status"] == "original_buy_price_condition_failed",
        "fixture_price_guard_failed",
    )
    require(funded_with_reserve["status"] == "ready_for_original_buy", "fixture_reserve_failed")
    return {
        "mode": "offline_snapshot_replay",
        "snapshot_bjt": datetime.fromtimestamp(
            timestamp / 1000, timezone(timedelta(hours=8))
        ).isoformat(),
        "marks_usdt": sim.marks,
        "before": before,
        "after": after,
        "strategy_equity_before": before_equity,
        "strategy_equity_after": after_equity,
        "strategy_equity_mark_difference": after_equity - before_equity,
        "unchanged_accounting_basis": {
            k: sim.book[k]
            for k in (
                "initial",
                "external_flows",
                "fund_units",
                "deployable",
                "principal",
                "interest_trx",
            )
        },
        "marginal_product_targets": targets,
        "product_terms": {
            pid: {"coin": p.coin, "tiers": p.tiers} for pid, p in sim.products.items()
        },
        "daily_interest_before_usdt": before_daily,
        "daily_interest_after_usdt": after_daily,
        "daily_interest_change_usdt": {o: after_daily[o] - before_daily[o] for o in OWNERS},
        "daily_interest_change_with_1000_usdt_in_spot": {
            o: reserve_daily[o] - before_daily[o] for o in OWNERS
        },
        "funding_assumptions": {
            "redeem_terms_verified": False,
            "receipt_delays_seconds": delays,
            "shared_redemption_interval_seconds": REDEMPTION_INTERVAL_SECONDS,
            "synthetic_plan_valid_for_seconds": 180,
            "quotes": "historical_snapshot_held_constant",
            "trx_order_sent": False,
        },
        "allocation_events": allocation_events,
        "funding_fixtures": {
            "owned_cash_and_usdc": funded,
            "with_1000_usdt_already_in_spot": funded_with_reserve,
            "late_receipt": late,
            "missing_receipt": missing,
            "outside_usdc_only": outside_only,
            "original_price_condition_failed": price_failed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("usdc_bill_evidence", type=Path)
    args = parser.parse_args()
    output = run(
        json.loads(args.snapshot.read_text()), json.loads(args.usdc_bill_evidence.read_text())
    )
    print(json.dumps(output, default=str, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
