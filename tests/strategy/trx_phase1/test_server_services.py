"""The deployed frontend uses one login across trading and independent readers."""

import json
from copy import deepcopy

from fastapi.testclient import TestClient

from freqtrade.rpc.api_server import api_trx
from freqtrade.rpc.api_server.app import create_app
from freqtrade.strategy.trx_phase1.live_history import LiveHistory
from freqtrade.strategy.trx_phase1.live_history_app import create_app as history_app
from freqtrade.strategy.trx_phase1.live_ops import LiveView, Operations
from freqtrade.strategy.trx_phase1.live_ops_app import create_app as observer_app
from freqtrade.strategy.trx_phase1.live_restore import ExistingLiveRestore
from freqtrade.strategy.trx_phase1.live_strategy import LiveStrategy
from tests.strategy.trx_phase1.test_live_restore import completed, service  # noqa: F401


def test_existing_account_restores_and_frontend_reads_share_tokens_without_financial_writes(
    completed,  # noqa: F811 - imported pytest fixture
    tmp_path,
    monkeypatch,
    mocker,
):
    svc, broker, clock, _, _ = completed
    runtime = svc.get_runtime(svc.load())
    runtime.capture_observation(clock[0])
    runtime.persist()
    source = runtime.execution.root
    original = {p.name: p.read_bytes() for p in source.glob("*.sqlite")}
    book = deepcopy(runtime.execution.state["book"])
    writes = deepcopy(broker.state()["writes"])
    start = mocker.patch.object(LiveStrategy, "start")
    monkeypatch.setattr(ExistingLiveRestore, "start", lambda self: self.once())
    monkeypatch.setattr(api_trx, "_live_services", {tmp_path: svc})
    monkeypatch.setattr(api_trx, "_live_restorer", None)
    credentials = {
        "username": "fixture-operator",
        "password": "fixture-password",
        "jwt_secret_key": "fixture-secret-which-is-at-least-32-characters",
    }
    config = {
        "user_data_dir": str(tmp_path),
        "api_server": credentials,
        "trx_phase1": {"existing_live_only": True, "auto_resume_live": True},
    }
    app = create_app(config)
    account = mocker.Mock()
    account.allocation_context.return_value = ({}, None)
    app.dependency_overrides[api_trx.get_account_service] = lambda: account
    view = LiveView(source, runtime.execution.account_id, clock=lambda: clock[0])
    ops = Operations(view, tmp_path / "ops.sqlite", tmp_path / "backup.json", tmp_path)
    history = LiveHistory(
        source, tmp_path / "archive/history.sqlite", view.account_id, clock=lambda: clock[0]
    )
    reader_config = {**credentials, "account_id": view.account_id}
    with (
        TestClient(app) as main_client,
        TestClient(observer_app(reader_config, view, ops)) as observer,
        TestClient(history_app(reader_config, history)) as archive,
    ):
        start.assert_called_once()
        login = main_client.post(
            "/api/v1/token/login", auth=(credentials["username"], credentials["password"])
        ).json()
        headers = {"Authorization": "Bearer " + login["access_token"]}
        full = main_client.get("/api/v1/trx/live", headers=headers)
        light = observer.get("/api/v1/trx/live/view", headers=headers)
        events = archive.get(
            "/api/v1/trx/live/history", params={"account_id": view.account_id}, headers=headers
        )
        assert full.status_code == light.status_code == events.status_code == 200
        assert full.json()["prepared"]["account_id"] == light.json()["account_id"]
        assert (
            full.json()["runtime"]["summary"]["principal_trx"]
            == light.json()["runtime"]["summary"]["principal_trx"]
        )
        assert events.json()["rows"]
        assert observer.get("/api/v1/trx/live/ops", headers=headers).status_code == 200
        refreshed = main_client.post(
            "/api/v1/token/refresh",
            headers={"Authorization": "Bearer " + login["refresh_token"]},
        ).json()
        assert (
            observer.get(
                "/api/v1/trx/live/view",
                headers={"Authorization": "Bearer " + refreshed["access_token"]},
            ).status_code
            == 200
        )
        assert observer.get("/api/v1/trx/live/view").status_code == 401
        assert archive.post("/api/v1/trx/live/history", headers=headers).status_code == 405
        assert "fixture-secret" not in json.dumps([full.json(), light.json(), events.json()])
    assert {p.name: p.read_bytes() for p in source.glob("*.sqlite")} == original
    assert runtime.execution.state["book"] == book
    assert broker.state()["writes"] == writes
    assert runtime.state["desired"] == "running"
