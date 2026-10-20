from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server import api_trx
from freqtrade.rpc.api_server.app import create_app, load_config
from freqtrade.strategy.trx_phase1.timerange import TimeRange


@pytest.fixture
def config(tmp_path):
    return {
        "user_data_dir": str(tmp_path),
        "api_server": {
            "username": "operator",
            "password": "fixture-password",
            "jwt_secret_key": "fixture-secret-which-is-at-least-32-characters",
        },
    }


def test_server_login_refresh_and_trx_authentication(config, mocker):
    restore = mocker.patch.object(api_trx, "start_live_restore")
    shutdown = mocker.patch.object(api_trx, "shutdown_live_services")
    account = mocker.Mock()
    account.status.return_value = {"mode": "fixture"}
    app = create_app(config)
    app.dependency_overrides[api_trx.get_account_service] = lambda: account
    with TestClient(app) as client:
        assert client.get("/api/v1/ping").json() == {"status": "pong"}
        assert client.get("/api/v1/trx/account").status_code == 401
        assert client.post("/api/v1/token/login", auth=("operator", "wrong")).status_code == 401
        login = client.post("/api/v1/token/login", auth=("operator", "fixture-password")).json()
        headers = {"Authorization": "Bearer " + login["access_token"]}
        assert client.get("/api/v1/trx/account", headers=headers).json() == {"mode": "fixture"}
        assert client.post("/api/v1/token/refresh", headers=headers).status_code == 401
        refresh = {"Authorization": "Bearer " + login["refresh_token"]}
        assert client.get("/api/v1/trx/account", headers=refresh).status_code == 401
        renewed = client.post("/api/v1/token/refresh", headers=refresh)
        assert renewed.status_code == 200 and renewed.json()["access_token"]
        for path in ("/api/v1/trades", "/api/v1/backtest", "/api/v1/freqaimodels", "/docs"):
            assert client.get(path, headers=headers).status_code == 404
        restore.assert_called_once()
    shutdown.assert_called_once()


@pytest.mark.parametrize(
    "claims",
    [
        {"identity": {"u": "someone-else"}},
        {"identity": ["operator"]},
        {"identity": {"u": "operator"}, "exp": 1},
        {"identity": {"u": "operator"}, "type": "refresh"},
        {"identity": {"u": "operator"}, "exp": None},
    ],
)
def test_invalid_tokens_never_reach_account_service(config, mocker, claims):
    account = mocker.Mock()
    app = create_app(config)
    app.dependency_overrides[api_trx.get_account_service] = lambda: account
    now = datetime.now(UTC)
    token = jwt.encode(
        {"iat": now, "exp": now + timedelta(minutes=1), "type": "access", **claims},
        config["api_server"]["jwt_secret_key"],
        algorithm="HS256",
    )
    client = TestClient(app)
    assert (
        client.get("/api/v1/trx/account", headers={"Authorization": "Bearer " + token}).status_code
        == 401
    )
    account.status.assert_not_called()


def test_config_environment_overrides_and_does_not_load_exchange_credentials(tmp_path, monkeypatch):
    path = tmp_path / "server.json"
    path.write_text('{"api_server": {"username": "old"}}')
    monkeypatch.setenv("TRX_API_USERNAME", "operator")
    monkeypatch.setenv("TRX_API_PASSWORD", "fixture-password")
    monkeypatch.setenv("TRX_API_JWT_SECRET_KEY", "fixture-secret-which-is-at-least-32-characters")
    config = load_config(path)
    app = create_app(config)
    assert app.state.config["api_server"]["username"] == "operator"
    assert app.state.config["exchange"] == {"name": "bitget"}
    assert "old" in path.read_text()


@pytest.mark.parametrize(
    "change",
    [
        {"api_server": {}},
        {"exchange": {"name": "binance"}},
        {"trading_mode": "futures"},
        {"api_server": {"username": "u", "password": "p", "jwt_secret_key": "short"}},
    ],
)
def test_invalid_server_configuration_is_rejected(config, change):
    with pytest.raises(ValueError):
        create_app({**config, **change})


@pytest.mark.parametrize(
    "window",
    ["20260101-20260102", "1767225600-1767312000000", "20260101T000000-20260102T0000"],
)
def test_research_date_windows_keep_utc_and_exclusive_end(window):
    parsed = TimeRange.parse_timerange(window)
    assert parsed.startdt == datetime(2026, 1, 1, tzinfo=UTC)
    assert parsed.stopdt == datetime(2026, 1, 2, tzinfo=UTC)


@pytest.mark.parametrize("window", ["-", "nonsense", "20260230-", "20260102-20260101"])
def test_invalid_research_date_windows_are_rejected(window):
    with pytest.raises(ValueError):
        TimeRange.parse_timerange(window)
