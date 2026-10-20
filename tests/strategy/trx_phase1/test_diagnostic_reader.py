import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from freqtrade.rpc.api_server import trx_diagnostic_reader
from freqtrade.rpc.api_server.trx_diagnostic_reader import DiagnosticReader


def test_concurrent_reads_share_work_and_saved_file_changes_invalidate_cache(tmp_path, monkeypatch):
    path = tmp_path / "saved.sqlite"
    path.write_bytes(b"first saved result")
    submitted = threading.Event()
    futures = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, saved_path, timezone):
            future = Future()
            futures.append((saved_path, timezone, future))
            submitted.set()
            return future

    monkeypatch.setattr(trx_diagnostic_reader, "ProcessPoolExecutor", Pool)
    reader = DiagnosticReader()
    with ThreadPoolExecutor(max_workers=2) as clients:
        first = clients.submit(reader.read, path, "Asia/Shanghai")
        assert submitted.wait(2)
        second = clients.submit(reader.read, path, "Asia/Shanghai")
        futures[0][2].set_result({"collected": True, "signals": 80})
        assert first.result(2) == second.result(2) == {"collected": True, "signals": 80}
    assert len(futures) == 1
    assert reader.read(path, "Asia/Shanghai")["signals"] == 80
    with ThreadPoolExecutor(max_workers=1) as client:
        submitted.clear()
        utc = client.submit(reader.read, path, "UTC")
        assert submitted.wait(2)
        futures[-1][2].set_result({"timezone": "UTC"})
        assert utc.result(2) == {"timezone": "UTC"}
        saved = path.stat()
        os.utime(path, ns=(saved.st_atime_ns, saved.st_mtime_ns + 1_000_000))
        submitted.clear()
        changed = client.submit(reader.read, path, "Asia/Shanghai")
        assert submitted.wait(2)
        futures[-1][2].set_result({"signals": 81})
        assert changed.result(2) == {"signals": 81}
    assert len(futures) == 3


def test_failed_read_can_retry_and_shutdown_releases_worker(tmp_path, monkeypatch):
    path = tmp_path / "saved.sqlite"
    path.touch()
    calls, closed = [], []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args):
            calls.append(args)
            future = Future()
            if len(calls) == 1:
                future.set_exception(ValueError("cannot read saved data"))
            else:
                future.set_result({"collected": True})
            return future

        def shutdown(self, **kwargs):
            closed.append(kwargs)

    monkeypatch.setattr(trx_diagnostic_reader, "ProcessPoolExecutor", Pool)
    reader = DiagnosticReader()
    with pytest.raises(ValueError, match="cannot read"):
        reader.read(path, "UTC")
    assert reader.read(path, "UTC") == {"collected": True}
    reader.close()
    assert closed == [{"wait": False, "cancel_futures": True}]
    assert not reader.pending
