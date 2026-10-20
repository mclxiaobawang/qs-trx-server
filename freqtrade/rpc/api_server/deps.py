"""Dependencies of the standalone TRX API."""

from fastapi import Depends, Request

from freqtrade.strategy.trx_phase1.shadow_market import PublicShadowMarket


def get_config(request: Request):
    return request.app.state.config


def get_api_config(config=Depends(get_config)):
    return config["api_server"]


def get_exchange(config):
    # Paper execution reads public TRX data through the shared request budget.
    return PublicShadowMarket()
