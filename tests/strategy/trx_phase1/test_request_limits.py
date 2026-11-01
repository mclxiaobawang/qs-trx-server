import hashlib
import multiprocessing
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import ccxt
import pytest

from freqtrade.strategy.trx_phase1.account import AccountSettings, ReadOnlyBitget
from freqtrade.strategy.trx_phase1.live_broker import (
    ClassicBroker,
    ClassicTransport,
    ExecutionError,
    WriteNotSent,
    WriteUnknown,
)
from freqtrade.strategy.trx_phase1.request_limits import RateLimitWait, RequestLimiter
from freqtrade.strategy.trx_phase1.shadow_market import PublicBitget


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def limiter(tmp_path):
    clock = Clock()
    return RequestLimiter(
        tmp_path / "limits.sqlite", clock=clock, sleep=clock.sleep, jitter=lambda: 0
    )


def make_transport(limiter, mocker):
    settings = AccountSettings(api_key="fixture", secret_key="secret", passphrase="pass")
    session = mocker.Mock()
    session.request.return_value = SimpleNamespace(
        status_code=200, json=lambda: {"code": "00000", "data": {"userId": "123"}}
    )
    return ClassicTransport(settings, writes_enabled=True, session=session, limiter=limiter)


def _process_requests(path, output):
    limiter = RequestLimiter(path)
    for _ in range(4):
        limiter.acquire("v2/spot/account/assets", "credential")
        output.put(time.time())


def test_two_processes_share_pacing_and_leave_no_queued_reservations(tmp_path):
    path = tmp_path / "limits.sqlite"
    RequestLimiter(path)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [context.Process(target=_process_requests, args=(path, output)) for _ in range(2)]
    for process in processes:
        process.start()
    stamps = sorted(output.get(timeout=20) for _ in range(8))
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert stamps[-1] - stamps[0] >= 0.85
    assert RequestLimiter.status(path)["queued"] == 0
    assert RequestLimiter.status(path)["requests"] == 8
    assert path.stat().st_mode & 0o077 == 0


def test_private_uid_budget_is_shared_across_credentials_but_public_is_separate(limiter):
    limiter.bind("key-a", "same-uid-hash")
    limiter.bind("key-b", "same-uid-hash")
    limiter.acquire("assets", "key-a", max_wait=0)
    with pytest.raises(RateLimitWait):
        limiter.acquire("fills", "key-b", max_wait=0)
    limiter.acquire("candles", max_wait=0)


def test_redemption_cooldown_survives_restart_and_does_not_block_reads(limiter):
    limiter.acquire("v2/earn/savings/redeem", "key", write=True)
    restarted = RequestLimiter(limiter.path, clock=limiter.clock, sleep=limiter.sleep)
    limiter.clock.sleep(1)
    before = limiter.clock()
    with pytest.raises(RateLimitWait):
        restarted.acquire("v2/earn/savings/redeem", "key", write=True)
    assert limiter.clock() == before
    restarted.acquire("v2/spot/account/assets", "key")
    limiter.clock.sleep(60)
    restarted.acquire("v2/earn/savings/redeem", "key", write=True)


def test_reconciliation_overtakes_queued_ui_reads(tmp_path):
    limiter = RequestLimiter(tmp_path / "limits.sqlite")
    limiter.acquire("assets", "key")
    order = []
    with limiter.db() as db:
        db.execute("UPDATE buckets SET ready=?", (time.time() + 0.4,))

    def wait(priority):
        limiter.acquire("fills", "key", priority=priority)
        order.append(priority)

    with ThreadPoolExecutor(max_workers=2) as pool:
        ui = pool.submit(wait, 2)
        deadline = time.monotonic() + 2
        while RequestLimiter.status(limiter.path)["queued"] != 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        receipt = pool.submit(wait, 0)
        ui.result(timeout=3)
        receipt.result(timeout=3)
    assert order == [0, 2]


def test_identity_cache_expires_and_credentials_do_not_reuse_it(limiter, mocker):
    transport = make_transport(limiter, mocker)
    broker = ClassicBroker(transport)
    account = hashlib.sha256(b"123").hexdigest()
    assert broker.account_id() == broker.account_id() == account
    assert transport.session.request.call_count == 1
    limiter.clock.sleep(30)
    assert broker.account_id() == account
    assert transport.session.request.call_count == 2
    assert limiter.identity("different-credentials") is None
    with sqlite3.connect(limiter.path) as db:
        rows = str(db.execute("SELECT * FROM identities").fetchall())
        assert "fixture" not in rows and "secret" not in rows


def test_identity_endpoint_keeps_stricter_interval(limiter):
    path = "v2/spot/account/info"
    limiter.acquire(path, "key")
    limiter.clock.sleep(1)
    with pytest.raises(RateLimitWait):
        limiter.acquire(path, "key", max_wait=0)
    limiter.clock.sleep(0.2)
    limiter.acquire(path, "key", max_wait=0)


@pytest.mark.parametrize("json_error", [False, True])
def test_read_429_retries_after_shared_cooldown_and_signs_after_wait(limiter, mocker, json_error):
    transport = make_transport(limiter, mocker)
    blocked = SimpleNamespace(
        status_code=200 if json_error else 429,
        headers={"Retry-After": "2"},
        json=lambda: {"code": "429"} if json_error else (_ for _ in ()).throw(ValueError()),
    )
    transport.session.request.side_effect = [blocked, transport.session.request.return_value]
    mocker.patch("freqtrade.strategy.trx_phase1.live_broker.time.time", limiter.clock)
    transport.call("GET", "v2/spot/account/assets")
    stamps = [
        int(c.kwargs["headers"]["ACCESS-TIMESTAMP"])
        for c in transport.session.request.call_args_list
    ]
    assert stamps[1] - stamps[0] >= 2000
    assert RequestLimiter.status(limiter.path)["responses_429"] == 1


def test_post_429_never_retries_and_blocks_other_clients(limiter, mocker):
    transport = make_transport(limiter, mocker)
    transport.session.request.return_value = SimpleNamespace(
        status_code=429,
        headers={"Retry-After": "90"},
        json=lambda: (_ for _ in ()).throw(ValueError()),
    )
    with pytest.raises(WriteUnknown, match="exchange_write_429"):
        transport.call(
            "POST",
            "v2/earn/savings/subscribe",
            {"productId": "p", "periodType": "flexible", "amount": "1"},
        )
    assert transport.session.request.call_count == 1
    with pytest.raises(RateLimitWait):
        RequestLimiter(limiter.path, clock=limiter.clock).acquire("candles", max_wait=0)


def test_slow_durable_journal_admits_once_before_one_submission(limiter, mocker):
    transport = make_transport(limiter, mocker)
    path = "v2/earn/savings/subscribe"
    transport.reserve_write(path, 1)
    limiter.clock.sleep(0.1)
    transport.call("POST", path, {"productId": "p", "periodType": "flexible", "amount": "1"})
    assert transport.session.request.call_count == 1
    with sqlite3.connect(limiter.path) as db:
        assert db.execute("SELECT COUNT(*) FROM admissions").fetchone()[0] == 1


def test_original_write_deadline_still_prevents_submission(limiter, mocker):
    transport = make_transport(limiter, mocker)
    path = "v2/earn/savings/subscribe"
    transport.reserve_write(path, 0.1)
    limiter.clock.sleep(0.2)
    with pytest.raises(WriteNotSent):
        transport.call("POST", path, {"productId": "p", "periodType": "flexible", "amount": "1"})
    transport.session.request.assert_not_called()


def test_submission_never_bypasses_shared_cooldown(limiter, mocker):
    transport = make_transport(limiter, mocker)
    path = "v2/earn/savings/subscribe"
    transport.reserve_write(path, 10)
    limiter.clock.sleep(0.1)
    limiter.limited({"Retry-After": "60"})
    with pytest.raises(WriteNotSent):
        transport.call("POST", path, {"productId": "p", "periodType": "flexible", "amount": "1"})
    transport.session.request.assert_not_called()


def test_repeated_429_reads_are_bounded(limiter, mocker):
    transport = make_transport(limiter, mocker)
    transport.session.request.return_value = SimpleNamespace(status_code=429, headers={})
    with pytest.raises(ExecutionError, match="request_rate_wait"):
        transport.call("GET", "v2/spot/account/assets")
    assert transport.session.request.call_count == 4
    assert limiter.clock() >= 1007


@pytest.mark.parametrize("kind", [ReadOnlyBitget, PublicBitget])
def test_ccxt_non_json_429_retries_via_shared_scheduler(kind, limiter, mocker):
    client = kind({"apiKey": "fixture", "secret": "secret", "password": "pass"})
    client._shared_limiter = limiter
    calls = []

    def send(*args):
        calls.append(limiter.clock())
        if len(calls) == 1:
            client.on_rest_response(
                429, "limited", "url", "GET", {"Retry-After": "3"}, "<html>429</html>", {}, None
            )
        return {"code": "00000", "data": {"serverTime": "1"}}

    mocker.patch.object(ccxt.bitget, "request", side_effect=send)
    assert client.request("v2/public/time")["code"] == "00000"
    assert len(calls) == 2 and calls[1] - calls[0] >= 3
    assert RequestLimiter.status(limiter.path)["responses_429"] == 1


def test_identical_identity_reads_coalesce_but_balances_remain_fresh(limiter, mocker):
    clients = [ReadOnlyBitget({"apiKey": "key", "secret": "s", "password": "p"}) for _ in range(2)]
    for client in clients:
        client._shared_limiter = limiter
    send = mocker.patch.object(
        ccxt.bitget, "request", return_value={"code": "00000", "data": {"userId": "123"}}
    )
    clients[0].request("v2/spot/account/info", "privateSpot")
    clients[1].request("v2/spot/account/info", "privateSpot")
    assert send.call_count == 1
    clients[0].request("v2/spot/account/assets", "privateSpot")
    clients[1].request("v2/spot/account/assets", "privateSpot")
    assert send.call_count == 3
    limiter.clock.sleep(30)
    clients[1].request("v2/spot/account/info", "privateSpot")
    assert send.call_count == 4


def test_public_candle_observations_are_never_coalesced(limiter, mocker):
    client = PublicBitget()
    client._shared_limiter = limiter
    send = mocker.patch.object(ccxt.bitget, "request", return_value={"code": "00000", "data": []})
    for _ in range(2):
        client.request("v2/spot/market/candles")
    assert send.call_count == 2


def test_symbol_cache_keeps_classic_and_ccxt_response_shapes_separate(limiter, mocker):
    transport = make_transport(limiter, mocker)
    rows = [{"symbol": "TRXUSDT"}]
    transport.session.request.return_value = SimpleNamespace(
        status_code=200, json=lambda: {"code": "00000", "data": rows}
    )
    client = PublicBitget()
    client._shared_limiter = limiter
    send = mocker.patch.object(ccxt.bitget, "request", return_value={"code": "00000", "data": rows})
    for _ in range(2):
        assert transport.call("GET", "v2/spot/public/symbols") == rows
        assert client.request("v2/spot/public/symbols")["data"] == rows
    assert transport.session.request.call_count == send.call_count == 1


def test_long_retry_after_defers_reads_with_a_bounded_wait(limiter, mocker):
    transport = make_transport(limiter, mocker)
    transport.session.request.return_value = SimpleNamespace(
        status_code=429, headers={"Retry-After": "3600"}
    )
    with pytest.raises(ExecutionError, match="request_rate_wait"):
        transport.call("GET", "v2/spot/account/assets")
    assert transport.session.request.call_count == 1
    assert limiter.clock() - 1000 <= 15.01


def test_corrupt_metrics_do_not_hide_the_rest_of_the_status(tmp_path):
    path = tmp_path / "limits.sqlite"
    path.write_bytes(b"corrupt")
    assert RequestLimiter.status(path)["error"] == "request_limit_storage_unavailable"
