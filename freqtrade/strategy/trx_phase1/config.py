from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Independent mechanisms, in (thresholds, quantities, periods) order.
MECHANISMS = {
    "fixed": (False, False, False),
    "dynamic_threshold": (True, False, False),
    "dynamic_quantity": (False, True, False),
    "dynamic_period_only": (False, False, True),
    "dynamic_size": (True, True, False),
    "dynamic_threshold_period": (True, False, True),
    "dynamic_quantity_period": (False, True, True),
    "dynamic_period": (True, True, True),
}


class ResearchConfig(BaseModel):
    """Versioned research assumptions, not optimized parameters or live authorization."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)
    version: Literal[
        "trx-phase1-v0.1",
        "trx-phase1-v0.2",
        "trx-phase1-v0.3",
        "trx-phase1-v0.4",
        "trx-phase1-v0.5",
        "trx-phase1-v0.6",
        "trx-phase1-v0.7",
        "trx-phase1-v0.8",
    ] = "trx-phase1-v0.4"
    initial_cash: float = Field(default=1000, gt=0)
    startup: Literal["wait", "small_buy", "half_position", "allocated_position"] = "wait"
    variant: Literal[
        "fixed",
        "dynamic_threshold",
        "dynamic_quantity",
        "dynamic_period_only",
        "dynamic_size",
        "dynamic_threshold_period",
        "dynamic_quantity_period",
        "dynamic_period",
    ] = "dynamic_period"
    periods: list[int] = Field(default_factory=lambda: [5, 15, 60, 240])
    volatility_window: int = Field(default=120, ge=20, le=10080)
    volatility_high: float = Field(default=0.012, gt=0)
    buy_threshold: float = Field(default=0.012, gt=0, lt=1)
    sell_threshold: float = Field(default=0.012, gt=0, lt=1)
    minimum_threshold: float = Field(default=0.004, gt=0, lt=1)
    maximum_threshold: float = Field(default=0.08, gt=0, lt=1)
    volatility_multiplier: float = Field(default=0.5, ge=0)
    buy_occupation_multiplier: float = Field(default=0.02, ge=0)
    sell_occupation_multiplier: float = Field(default=0.006, ge=0)
    buy_fraction: float = Field(default=0.12, gt=0, le=1)
    sell_fraction: float = Field(default=0.3, gt=0, le=1)
    initial_buy_fraction: float = Field(default=0.02, gt=0, le=1)
    buy_quantity_dynamic: bool | None = None
    sell_quantity_dynamic: bool | None = None
    price_position_enabled: bool = False
    price_position_lookback_days: int = Field(default=365, ge=1, le=3650)
    price_position_quantile: float = Field(default=0.75, gt=0, lt=1)
    price_position_buy_multiplier: float = Field(default=0.5, ge=0, le=1)
    price_position_sell_multiplier: float = Field(default=1.5, ge=1, le=10)
    sell_rebound_multiplier: float = Field(default=0.2, ge=0, le=1)
    minimum_period_hold: int = Field(default=3, ge=1, le=100)
    buy_fee: float = Field(default=0.001, ge=0, lt=0.1)
    sell_fee: float = Field(default=0.001, ge=0, lt=0.1)
    fee_rebate_ratio: float = Field(default=0.5, ge=0, le=1)
    rebate_settlement_hour: int = Field(default=0, ge=0, le=23)
    rebate_utc_offset_minutes: int = Field(default=480, ge=-840, le=840)
    slippage: float = Field(default=0.0005, ge=0, lt=0.1)
    min_notional: float = Field(default=5, gt=0)
    min_quantity: float = Field(default=0, ge=0)
    quantity_decimals: int = Field(default=6, ge=0, le=12)
    execution_margin: float = Field(default=0.0005, ge=0, lt=0.1)
    maximum_execution_deviation: float = Field(default=0.01, ge=0, lt=0.5)
    tail_completion: bool = True
    execution_preflight: bool = False
    buy_reference_recovery: bool = False
    recovery_stagnation_days: int = Field(default=30, ge=1, le=365)
    recovery_valid_days: int = Field(default=7, ge=1, le=30)
    recovery_cooldown_days: int = Field(default=7, ge=1, le=90)
    recovery_buy_fraction: float = Field(default=0.02, gt=0, le=0.1)
    sell_quantity_basis: Literal["principal", "profitable"] = "principal"
    sell_mode: Literal["proportional", "recent_lot_then_core"] = "proportional"
    sell_fraction_cap: float = Field(default=1, gt=0, le=1)
    earn_model: Literal["none", "trx_hourly", "simplified_hourly", "net_benefit_hourly"] = "none"
    trx_apr: float = Field(default=0.10, ge=0, le=1)
    usdgo_apr: float = Field(default=0.12, ge=0, le=1)
    usdt_apr: float = Field(default=0, ge=0, le=1)
    trx_apr_tiers: list[tuple[float, float]] = Field(default_factory=list)
    usdgo_apr_tiers: list[tuple[float, float]] = Field(default_factory=list)
    usdt_apr_tiers: list[tuple[float, float]] = Field(default_factory=list)
    apr_source: str = ""
    apr_observed_at_utc: str = ""
    apr_scenario_scale: float = Field(default=1, ge=0, le=1)
    cash_forecast_hours: int = Field(default=24, ge=1, le=720)
    cash_min_switch_value: float = Field(default=5, ge=0)
    cash_switch_margin: float = Field(default=0, ge=0)
    cash_min_hold_minutes: int = Field(default=60, ge=0, le=43200)
    fx_rate: float = Field(default=1, gt=0)
    fx_spread: float = Field(default=0, ge=0, lt=0.1)
    fx_fee: float = Field(default=0, ge=0, lt=0.1)
    fx_slippage: float = Field(default=0, ge=0, lt=0.1)
    fx_quantity_decimals: int = Field(default=12, ge=0, le=18)
    fx_deviation_limit: float = Field(default=0.01, ge=0, lt=0.5)
    redemption_delay_minutes: int = Field(default=0, ge=0, le=1440)
    trx_redemption_delay_minutes: int | None = Field(default=None, ge=0, le=1440)
    usdt_redemption_delay_minutes: int = Field(default=0, ge=0, le=1440)
    fx_quote_available: bool = True
    fx_shock_after_minutes: int = Field(default=0, ge=0)
    fx_shock_rate: float = Field(default=0, ge=0)
    trx_product_available: bool = True
    usdgo_product_available: bool = True
    trx_earn_cap: float = Field(default=0, ge=0)
    usdgo_earn_cap: float = Field(default=0, ge=0)

    @property
    def trx_redemption_delay(self):
        return (
            self.redemption_delay_minutes
            if self.trx_redemption_delay_minutes is None
            else self.trx_redemption_delay_minutes
        )

    @property
    def mechanisms(self):
        return dict(
            zip(("thresholds", "quantities", "periods"), MECHANISMS[self.variant], strict=True)
        )

    def sizing_enabled(self, side):
        override = getattr(self, side + "_quantity_dynamic")
        return self.mechanisms["quantities"] if override is None else override

    @model_validator(mode="after")
    def check_periods(self):
        if self.sell_mode == "recent_lot_then_core" and self.startup != "half_position":
            raise ValueError("Recent-lot sales require an explicit half-position core allocation")
        if self.startup == "half_position" and self.initial_buy_fraction != 0.5:
            raise ValueError("Half-position startup requires initial_buy_fraction=0.5")
        if self.startup in ("half_position", "allocated_position"):
            if self.earn_model != "net_benefit_hourly":
                raise ValueError("First-open allocation requires the historical cash/FX model")
            if self.initial_buy_fraction >= 1:
                raise ValueError("First-open allocation requires a positive remaining USDGO budget")
        if not self.periods or self.periods != sorted(set(self.periods)):
            raise ValueError("periods must be unique and increasing")
        if any(p < 1 or p > 1440 or 1440 % p for p in self.periods):
            raise ValueError("periods must divide a UTC day and be between 1 and 1440 minutes")
        if self.minimum_threshold > self.maximum_threshold:
            raise ValueError("minimum threshold exceeds maximum threshold")
        for coin in ("trx", "usdgo", "usdt"):
            previous = 0
            for cap, rate in getattr(self, coin + "_apr_tiers"):
                if cap <= previous or not 0 <= rate <= 1:
                    raise ValueError(
                        "APR tiers require increasing positive caps and rates between 0 and 1"
                    )
                previous = cap
        return self

    def rebate_due_at(self, timestamp):
        day = 86400000
        offset = self.rebate_utc_offset_minutes * 60000
        hour = self.rebate_settlement_hour * 3600000
        return ((timestamp + offset - hour) // day + 1) * day + hour - offset

    @classmethod
    def from_saved(cls, values):
        # Old reports/states never promised a refund. Do not invent historical income.
        if values.get("version") == "trx-phase1-v0.1" and "fee_rebate_ratio" not in values:
            values = {**values, "fee_rebate_ratio": 0}
        # A historical zero-Earn report never acquires simulated income on reload.
        if "earn_model" not in values:
            values = {**values, "earn_model": "none"}
        if (
            values.get("version", "trx-phase1-v0.3")
            not in (
                "trx-phase1-v0.4",
                "trx-phase1-v0.5",
                "trx-phase1-v0.6",
                "trx-phase1-v0.7",
                "trx-phase1-v0.8",
            )
            and "sell_rebound_multiplier" not in values
        ):
            values = {**values, "sell_rebound_multiplier": 0}
        return cls(**values)


ASSUMPTIONS = {
    "confirmed_anchor": "Shared actual-fill anchor, updated on completion, retained after flat.",
    "candidate_rules": [
        "D02: one plan per new close; expire at the next confirmation close.",
        "D03: market/cost occupation, period hysteresis, separate buy/sell periods.",
        "D04: bounded fractional purchases/sales; occupation slows additions.",
        "D05: finish a tradable, entirely profitable tail; otherwise retain dust.",
        "D07: unreliable data, unknown fees or unavailable balances pause affected actions.",
        "D08: stopping suspends decisions without liquidating principal or interest.",
    ],
    "parameters": "Numerical defaults are research inputs, not optimized or live-approved.",
    "execution": "Decide at close, fill at next open with costs, retain ending inventory.",
    "interest": (
        "Historical research supports user-defined hourly TRX/USDGO scenarios. "
        "earn_model=none and old reports retain zero Earn. "
        "Real account income still requires actual credited records."
    ),
    "fee_rebate": (
        "User-confirmed 50% rebate with daily research settlement. Default: next day "
        "00:00 UTC+8 in USDT. Pending refunds are excluded from NAV and spendable cash."
    ),
    "authorization": "Research only; account operations require separate activation.",
}
