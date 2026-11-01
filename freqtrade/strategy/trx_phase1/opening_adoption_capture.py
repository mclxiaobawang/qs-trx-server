"""Fresh opening observations using only the account client's GET allowlist."""

import hashlib
from copy import deepcopy

from freqtrade.strategy.trx_phase1.account import (
    AccountReader,
    load_settings,
    normalize_balances,
    response_data,
    text_id,
)
from freqtrade.strategy.trx_phase1.account_savings import overview
from freqtrade.strategy.trx_phase1.adoption_preview import adoption_preview
from freqtrade.strategy.trx_phase1.ledger import money
from freqtrade.strategy.trx_phase1.live_broker import ExecutionError
from freqtrade.strategy.trx_phase1.request_limits import RequestLimiter


def capture(account, service, *, current_positions_only=False):
    observed = (
        account.refresh(current_positions_only=True)
        if current_positions_only
        else account.refresh()
    )
    _, account_id = account.allocation_context()
    preview = adoption_preview(observed, now_ms=service.clock())
    if not account_id or not preview["reviewable"]:
        raise ExecutionError("fresh_complete_adoption_snapshot_required")
    identity = service.credential_identity()
    settings, _, complete = load_settings(account.config_path)
    if not complete or settings.account_type != "classic":
        raise ExecutionError("enabled_classic_credentials_required")
    client = account.client_factory(
        {
            "apiKey": settings.api_key.get_secret_value(),
            "secret": settings.secret_key.get_secret_value(),
            "password": settings.passphrase.get_secret_value(),
            "enableRateLimit": True,
            "timeout": 8000,
            "verbose": False,
            "options": {"defaultType": "spot", "adjustForTimeDifference": True},
        }
    )
    client._shared_limiter = RequestLimiter(service.root / "request-limits.sqlite")
    try:
        client.load_time_difference()
        info = response_data(client.privateSpotGetV2SpotAccountInfo())
        if hashlib.sha256(text_id(info["userId"]).encode()).hexdigest() != account_id:
            raise ExecutionError("account_binding_mismatch")
        reader = AccountReader(client, settings, service.clock(), service.clock)
        balances = normalize_balances(
            reader._read(client.privateSpotGetV2SpotAccountAssets, {"assetType": "all"})
        )
        snapshot = deepcopy(observed["snapshot"])
        before = {r["coin"]: r for r in snapshot["balances"]}
        after = {r["coin"]: r for r in balances}
        for coin in ("TRX", "USDGO", "USDT"):
            if (
                coin not in before
                or coin not in after
                or any(
                    money(before[coin][key]) != money(after[coin][key])
                    for key in ("available", "frozen", "locked", "restricted_available")
                )
            ):
                raise ExecutionError("account_changed_during_adoption")
        totals = overview(reader)
        for row in snapshot["savings"]["coverage"]["reconciliation"]:
            if totals[row["coin"]] != money(row["holdings_quantity"]):
                raise ExecutionError("account_changed_during_adoption")
        snapshot.update(
            balances=balances,
            orders=reader.recent_trading(),
            conversion_orders=reader.recent_trading(symbol="USDGOUSDT"),
        )
        # Quotes are deliberately the final exchange reads before the local commit.
        snapshot["prices"] = reader.prices()
        now = service.clock()
        if any(not -5000 <= now - q["timestamp"] <= 10000 for q in snapshot["prices"].values()):
            raise ExecutionError("adoption_quote_stale")
        snapshot["completed_at"] = now
        observed = {**observed, "snapshot": snapshot, "stale": False}
        result = adoption_preview(observed, now_ms=now)
        if not result["reviewable"] or identity != service.credential_identity():
            raise ExecutionError("fresh_complete_adoption_snapshot_required")
        result["opening_quotes"] = deepcopy(snapshot["prices"])
        return result, account_id, identity, now
    finally:
        client.close()
