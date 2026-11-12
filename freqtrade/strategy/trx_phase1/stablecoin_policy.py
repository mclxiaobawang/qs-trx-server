"""Account-bound stablecoin allocation for the native C1 execution journal."""

import re
import traceback
from copy import deepcopy
from hashlib import sha256
from json import dumps

from freqtrade.strategy.trx_phase1.ledger import Ledger, money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError, rounded


COINS = ("USDT", "USDC", "USDGO")
POLL_MS = 30 * 60_000
REDEMPTION_SPACING_MS = 61_000
REDEMPTION_DECIMALS = 2
ZERO = money(0)


def policy_for(state):
    policy = (state or {}).get("stablecoin_policy")
    if policy is None:
        return None
    products = policy.get("products", [])
    if (
        policy.get("version") != "old-stablecoin-v1"
        or policy.get("account_id") != state["account_id"]
        or policy.get("ledger_created_at") != state["created_at"]
        or type(policy.get("starts_at")) is not int
        or policy["starts_at"] < state["created_at"]
        or len(products) != 2
        or {row["coin"] for row in products} != {"USDT", "USDC"}
        or not state.get("external_usdc", {}).get("outside_strategy")
        or state["external_usdc"]["product_id"]
        != next(row["product_id"] for row in products if row["coin"] == "USDC")
    ):
        raise ExecutionError("stablecoin_policy_binding_invalid")
    # The old account uses a conservative request precision by policy.
    # The venue minimum and its supporting evidence remain independent.
    return {
        **policy,
        "products": [{**row, "redemption_decimals": REDEMPTION_DECIMALS} for row in products],
    }


def stage_activation(original, policy, remote, spot):
    """Prepare an offline state candidate from reconciled native ownership evidence."""
    if (
        original.get("stablecoin_policy")
        or original.get("operation")
        or original.get("error")
        or original.get("pending_income")
        or (original.get("flow") or {}).get("status")
        not in (None, "completed", "expired", "failed")
        or original.get("income_accounting", {}).get("status") != "active"
    ):
        raise ExecutionError("stablecoin_activation_requires_reconciled_idle_ledger")
    state = deepcopy(original)
    state["stablecoin_policy"] = deepcopy(policy)
    state["stablecoin_policy"] = policy_for(state)
    book = Ledger.restore(state["book"])
    if book.usdc or book.locked["USDT"]:
        raise ExecutionError("stablecoin_activation_requires_unallocated_native_earn")
    external = state["external_usdc"]
    if money(spot["USDC"]) != money(external["spot"]):
        raise ExecutionError("stablecoin_external_spot_unreconciled")
    native_usdc = {key: row for key, row in remote.items() if row["coin"] == "USDC"}
    if any(row["product_id"] != external["product_id"] for row in native_usdc.values()) or sum(
        (money(r["amount"]) for r in native_usdc.values()), ZERO
    ) != money(external["savings"]):
        raise ExecutionError("stablecoin_external_savings_unreconciled")
    state["unassigned_offsets"]["USDC"] = external["spot"]
    for key, row in native_usdc.items():
        if key in state["holdings"]:
            raise ExecutionError("stablecoin_holding_id_already_mapped")
        state["holdings"][key] = {
            "coin": "USDC",
            "product_id": row["product_id"],
            "owned_since": policy["starts_at"],
            "principal": "0",
            "interest": "0",
            "unassigned_amount": row["amount"],
            "subscription_ids": [],
            "lots": {},
        }
    from freqtrade.strategy.trx_phase1.income_allocation import ownership

    history = state["income_accounting"]["checkpoints"]
    if history[-1]["timestamp"] >= policy["starts_at"]:
        raise ExecutionError("stablecoin_activation_checkpoint_not_forward")
    history.append({"timestamp": policy["starts_at"], "positions": ownership(state["holdings"])})
    return state


def tiers(terms, existing):
    capacity = min(money(terms["singleMaxAmount"]), existing + money(terms["remainingAmount"]))
    rows, previous, floor = [], money(100), ZERO
    for row in terms.get("apyList") or terms.get("tiers") or []:
        low, high, rate = map(money, (row["minStepVal"], row["maxStepVal"], row["currentApy"]))
        if low != floor or high <= low or not 0 <= rate <= previous:
            raise ExecutionError("stablecoin_tiers_unverified")
        if capacity > low:
            rows.append((low, min(capacity, high), rate / 100))
        floor, previous = high, rate
    if floor == 0 or (not rows and existing > 0) or (rows and rows[-1][1] < existing):
        raise ExecutionError("stablecoin_capacity_unverified")
    return rows


def rate_confirmation(previous, now, terms, errors=None):
    """Confirm each product's rates AND subscription state on distinct half-hours."""
    checked = previous.get("checked_at")
    if checked is not None and now - checked < POLL_MS:
        return previous
    if terms is None:
        return {"checked_at": now, "status": "rates_unavailable"}
    products = {}
    for coin, data in sorted(terms.items()):
        status = data.get("status", "in_progress")
        body = {
            "id": data["product_id"],
            "status": status,
            "bands": data.get("apyList") or data["tiers"],
        }
        fingerprint = sha256(dumps(body, sort_keys=True).encode()).hexdigest()
        prior = previous.get("products", {}).get(coin, {})
        confirmed = (
            prior.get("fingerprint") == fingerprint
            and 0 < now - prior.get("checked_at", -2 * POLL_MS) < 2 * POLL_MS
        )
        products[coin] = {
            "product_id": data["product_id"],
            "native_status": status,
            "checked_at": now,
            "fingerprint": fingerprint,
            "status": "confirmed" if confirmed else "awaiting_second_observation",
            "subscription_eligible": status == "in_progress" and confirmed,
        }
    for coin, error in (errors or {}).items():
        products[coin] = {
            "checked_at": now,
            "status": "read_error",
            "subscription_eligible": False,
            "error": error,
        }
    confirmed = bool(products) and all(p["status"] == "confirmed" for p in products.values())
    fingerprint = sha256(
        dumps({c: p.get("fingerprint") for c, p in products.items()}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "checked_at": now,
        "schema": "per-product-v1",
        "status": "rates_unavailable"
        if errors
        else "confirmed"
        if confirmed
        else "awaiting_second_observation",
        "fingerprint": fingerprint,
        "terms": terms,
        "products": products,
    }


def product_error(exc, broker, coin, product_id, now):
    """Bounded diagnostics, without credentials, request headers or frame locals."""
    settings = getattr(getattr(broker, "transport", None), "settings", None)
    secrets = []
    for name in ("api_key", "secret_key", "passphrase"):
        value = getattr(settings, name, None)
        if value is not None:
            secrets.append(value.get_secret_value())

    def redact(value, limit):
        text = str(value)
        for secret in secrets:
            if secret:
                text = text.replace(secret, "<redacted>")
        text = re.sub(r"(?i)(Bearer |Basic )\S+", r"\1<redacted>", text)
        text = re.sub(
            r"(?i)((?:access[-_]?key|access[-_]?sign|secret|passphrase|password|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s,;\"'}]+",
            r"\1<redacted>",
            text,
        )
        text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?<redacted>", text)
        return text[:limit]

    return {
        "coin": coin,
        "product_id": product_id,
        "checked_at": now,
        "exception_type": type(exc).__name__,
        "message": redact(exc, 1500),
        "stage": getattr(exc, "product_request_stage", "product_validation"),
        "response": redact(dumps(getattr(exc, "read_diagnostic", {})), 2000),
        "traceback": redact("".join(traceback.format_exception(exc, limit=8)), 5000),
    }


def poll(execution, state, now):
    previous = state.get("stablecoin_rates", {})
    if now < execution.stablecoin_policy["starts_at"] or (
        now - previous.get("checked_at", -POLL_MS) < POLL_MS
    ):
        return
    rules = execution.state["products"] + execution.stablecoin_policy["products"]
    terms, errors = {}, {}
    for rule in rules:
        coin, pid = rule["coin"], rule["product_id"]
        if coin not in COINS:
            continue
        try:
            terms[coin] = {"product_id": pid, **execution.broker.product(coin, pid)}
            if terms[coin]["status"] not in {
                "in_progress",
                "not_started",
                "paused",
                "completed",
                "sold_out",
                "off_line",
            }:
                raise ExecutionError("stablecoin_product_status_unverified")
            tiers(terms[coin], ZERO)
        except (ExecutionError, ArithmeticError, ValueError, KeyError, TypeError) as exc:
            terms.pop(coin, None)
            errors[coin] = product_error(exc, execution.broker, coin, pid, now)
    if set(terms) | set(errors) != set(COINS):
        errors["configuration"] = {"message": "stablecoin_product_unverified", "checked_at": now}
    state["stablecoin_rates"] = rate_confirmation(previous, now, terms, errors)


def balances(funding, remote):
    book = funding.book
    spot = {
        "USDT": book.free_cash,
        "USDGO": book.usdgo - book.locked["USDGO"],
        "USDC": book.usdc - book.usdc_locked,
    }
    owned = {coin: ZERO for coin in COINS}
    account = dict(owned)
    for hid, row in remote.items():
        coin = row["coin"]
        if coin not in COINS or row["product_id"] != funding.rules[coin].product_id:
            continue
        value = money(row["amount"])
        mapped = funding.execution["holdings"].get(hid, {})
        quantity = money(mapped.get("principal", 0))
        if quantity < 0 or quantity > value:
            raise ExecutionError("stablecoin_holding_ownership_mismatch")
        owned[coin] += quantity
        account[coin] += value
    if owned != {
        "USDT": book.locked["USDT"],
        "USDGO": book.locked["USDGO"],
        "USDC": book.usdc_locked,
    }:
        raise ExecutionError("stablecoin_owned_savings_mismatch")
    return spot, owned, account


def allocate_amounts(spot, owned, fixed, bands, marks, reserve):
    budget = sum(((spot[c] + owned[c]) * marks[c] for c in COINS), ZERO) - reserve
    target = dict.fromkeys(COINS, ZERO)
    ordered = sorted(
        (-rate, coin != "USDT", coin, low, high)
        for coin, rows in bands.items()
        for low, high, rate in rows
    )
    for _, _, coin, low, high in ordered:
        room = max(ZERO, high - max(low, fixed[coin] + target[coin]))
        take = min(room, budget / marks[coin])
        target[coin] += take
        budget = max(ZERO, budget - take * marks[coin])
    return target


def outside_balances(funding, remote):
    from freqtrade.strategy.trx_phase1.external_stablecoin import holding_amount

    offsets = funding.execution["unassigned_offsets"]
    spot = {coin: money(offsets[coin]) for coin in COINS}
    owned = dict.fromkeys(COINS, ZERO)
    tracked = funding.execution["holdings"]
    for key in remote.keys() | tracked.keys():
        row, saved = remote.get(key, {}), tracked.get(key, {})
        coin = (row or saved)["coin"]
        if coin not in COINS or (row or saved)["product_id"] != funding.rules[coin].product_id:
            continue
        if not saved and money(row["amount"]):
            raise ExecutionError("external_stablecoin_holding_unmapped")
        if not saved:
            continue
        external = holding_amount(saved)
        total = sum((money(saved[k]) for k in ("principal", "interest", "unassigned_amount")), ZERO)
        if external < 0 or money(row.get("amount", 0)) != total:
            raise ExecutionError("external_stablecoin_holding_mismatch")
        if row and (row["coin"] != saved["coin"] or row["product_id"] != saved["product_id"]):
            raise ExecutionError("external_stablecoin_holding_mismatch")
        owned[coin] += external
    if min(spot.values()) < 0:
        raise ExecutionError("external_stablecoin_spot_mismatch")
    return spot, owned


def allocation(funding, observed, remote, *, outside=False):
    """Take the strategy slice first. Existing outside holdings still occupy shared tiers."""
    spot, owned, account = balances(funding, remote)
    marks = {
        c: money(funding.markets[c + "USDT"]["bid"]) if c != "USDT" else money(1) for c in COINS
    }
    # A closed product can retain existing yield, but has no new placement room.
    bands = {
        coin: tiers(
            observed[coin]
            if observed[coin].get("status", "in_progress") == "in_progress"
            else {**observed[coin], "remainingAmount": "0"},
            account[coin],
        )
        for coin in COINS
    }
    external = {coin: account[coin] - owned[coin] for coin in COINS}
    reserve = min(spot["USDT"], money(funding.policy.get("spot_reserve_usdt", 0)))
    if reserve < 0:
        raise ExecutionError("stablecoin_cash_reserve_invalid")
    target = allocate_amounts(spot, owned, external, bands, marks, reserve)
    if outside:
        strategy = owned
        spot, owned = outside_balances(funding, remote)
        external = {coin: account[coin] - owned[coin] for coin in COINS}
        # Leave room for the strategy's outstanding target, even if its
        # current holdings have not yet reached that target.
        fixed = {
            coin: external[coin] - strategy[coin] + max(strategy[coin], target[coin])
            for coin in COINS
        }
        reserve = ZERO
        target = allocate_amounts(spot, owned, fixed, bands, marks, reserve)
    return spot, owned, external, bands, marks, target, reserve


def marginal_floor(bands, held, amount):
    if not bands or amount <= 0 or held + amount > bands[-1][1]:
        raise ExecutionError("stablecoin_subscription_tier_changed")
    return min(apr for lo, hi, apr in bands if lo < held + amount and hi > held)


def minimum_apr(terms, held, amount):
    try:
        return marginal_floor(tiers(terms, held), held, amount)
    except (ExecutionError, KeyError, ArithmeticError, TypeError, ValueError):
        raise ExecutionError("stablecoin_subscription_tier_changed") from None


def rebalance(funding, observed, remote, *, outside=False):
    """Generate a single owned conversion/placement flow from the confirmed portfolio target."""
    spot, owned, external, bands, marks, target, reserve = allocation(
        funding, observed, remote, outside=outside
    )
    for coin in COINS:
        if observed[coin].get("status", "in_progress") != "in_progress":
            continue
        add = min(spot[coin] - (reserve if coin == "USDT" else ZERO), target[coin] - owned[coin])
        amount = funding.subscription(coin, add, observed)
        if amount:
            step = funding.subscribe(coin, amount)
            step["minimum_marginal_apr"] = str(
                marginal_floor(bands[coin], external[coin] + owned[coin], amount)
            )
            return owned_steps([step], outside)
    for dest in COINS:
        if observed[dest].get("status", "in_progress") != "in_progress":
            continue
        deficit = target[dest] - owned[dest] - spot[dest] + (reserve if dest == "USDT" else ZERO)
        if deficit <= 0 or funding.rules[dest].redemption_minimum is None:
            continue
        for source in COINS:
            excess = spot[source] + owned[source] - target[source]
            if source == dest or excess <= 0:
                continue
            if source == "USDT":
                excess -= reserve
            amount = min(excess, deficit * marks[dest] / marks[source])
            if amount <= 0:
                continue
            actions = switch(
                funding, observed, source, dest, amount, spot, reserve, outside=outside
            )
            if not actions:
                continue
            actions[-1]["minimum_marginal_apr"] = str(
                marginal_floor(
                    bands[dest], external[dest] + owned[dest], money(actions[-1]["amount"])
                )
            )
            return owned_steps(actions, outside)
    return []


def owned_steps(steps, outside):
    if not outside:
        return steps
    from freqtrade.strategy.trx_phase1.external_stablecoin import SOURCE

    return [{**step, "source": SOURCE} for step in steps]


def switch(funding, observed, source, dest, amount, spot, reserve, *, outside=False):
    """Prefer the quoted direct pair when it exists."""
    actions = []
    available = spot[source] - (reserve if source == "USDT" else ZERO)
    amount = min(amount, available + funding.redeemable_stable(source, outside=outside))
    if source == "USDT":
        base, quote_coin, side = dest, "USDT", "buy"
    elif dest == "USDT":
        base, quote_coin, side = source, "USDT", "sell"
    else:
        base, quote_coin = "USDGO", "USDC"
        side = "sell" if source == "USDGO" else "buy"
    market = funding.markets[base + quote_coin]
    quote_mark = money(funding.markets["USDCUSDT"]["bid"]) if quote_coin == "USDC" else money(1)
    precision = market["quantity_decimals" if side == "sell" else "price_decimals"]
    value_per_unit = (money(market["bid"]) if side == "sell" else money(1)) * quote_mark
    spot_spend = rounded(min(amount, max(ZERO, available)), precision)
    if spot_spend * value_per_unit >= money(market["minimum_notional"]):
        # Use independently tradable spot first, including already-settled redemptions.
        amount = spot_spend
    spend = rounded(amount, precision)
    notional = spend * value_per_unit
    if notional < money(market["minimum_notional"]):
        return None
    if spend > available:
        try:
            actions.extend(funding.redeem_stable(source, spend - available, outside=outside))
        except ExecutionError:
            return None
    trade = funding.stable_trade(base, side, spend, quote_coin)
    received = money(trade["minimum_receive"])
    actions.append(trade)
    subscribed = funding.subscription(dest, received, observed)
    if not subscribed:
        return None
    # Fixed cap at the guaranteed receipt and the reviewed product capacity.
    actions.append(funding.subscribe(dest, subscribed))
    return actions


def invalidate_subscription(state, coin, product_id, now, code):
    """An execution-time failure revokes cached eligibility, never the poll clock."""
    rates = state.get("stablecoin_rates", {})
    product = rates.get("products", {}).get(coin)
    if product is None or product.get("product_id") != product_id:
        return
    product.update(
        status="awaiting_second_observation",
        subscription_eligible=False,
        fingerprint=None,
        invalidated_at=now,
        error={
            "stage": "subscription_preflight",
            "message": code,
            "coin": coin,
            "product_id": product_id,
            "checked_at": now,
        },
    )
    rates["status"] = "awaiting_second_observation"
