"""Keep saved diagnostic aggregation responsive while a replay is running."""

import multiprocessing
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from freqtrade.strategy.trx_phase1.diagnostics import overview


class DiagnosticReader:
    def __init__(self):
        self.lock = threading.Lock()
        self.pool = None
        self.pending = OrderedDict()

    def read(self, path, timezone):
        path = Path(path).resolve()
        saved = path.stat()
        key = (str(path), saved.st_mtime_ns, saved.st_ctime_ns, saved.st_size, timezone)
        with self.lock:
            if key not in self.pending:
                if self.pool is None:
                    # A Python replay competes with SQLite row iteration for the GIL.
                    # Spawn a read-only worker instead of scanning in the API thread.
                    self.pool = ProcessPoolExecutor(
                        max_workers=1, mp_context=multiprocessing.get_context("spawn")
                    )
                self.pending[key] = self.pool.submit(overview, path, timezone)
            future = self.pending[key]
            self.pending.move_to_end(key)
            for old_key, old_future in list(self.pending.items()):
                if len(self.pending) <= 8:
                    break
                if old_key != key and old_future.done():
                    del self.pending[old_key]
        try:
            return future.result(timeout=120)
        except TimeoutError:
            # A retry shares the existing calculation instead of adding another scan.
            raise
        except Exception as exc:
            with self.lock:
                if self.pending.get(key) is future:
                    del self.pending[key]
                if isinstance(exc, BrokenProcessPool) and self.pool is not None:
                    self.pool.shutdown(wait=False, cancel_futures=True)
                    self.pool = None
                    self.pending.clear()
            raise

    def close(self):
        with self.lock:
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
                self.pool = None
            self.pending.clear()


diagnostic_reader = DiagnosticReader()
