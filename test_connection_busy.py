"""when_connection_free: a second call waits out a query on the same connection."""

import threading
import time
import unittest

import duckdb
import polars as pl

from mlBridge.mlBridgePostmortemLib import when_connection_free

# About a second of work on one connection.
_SLOW_SQL = "SELECT COUNT(*) FROM range(150000000) t(i) WHERE i % 7 = 3"


class WhenConnectionFreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect()
        self.worker = threading.Thread(target=lambda: self.con.sql(_SLOW_SQL).fetchall())
        self.worker.start()
        self.addCleanup(self.worker.join)
        # Wait until the worker's query holds the connection.
        deadline = time.monotonic() + 10
        while self.worker.is_alive() and time.monotonic() < deadline:
            try:
                self.con.execute("SELECT 1").fetchall()
            except duckdb.ResourceInUseException:
                return
            time.sleep(0.001)
        self.skipTest("the slow query finished before the connection was seen busy")

    def test_a_bare_call_on_a_busy_connection_is_refused(self) -> None:
        with self.assertRaises(duckdb.ResourceInUseException):
            self.con.register("frame", pl.DataFrame({"a": [1]}))

    def test_register_and_query_wait_for_the_running_query(self) -> None:
        when_connection_free(lambda: self.con.register("frame", pl.DataFrame({"a": [1, 2]})))
        self.assertFalse(self.worker.is_alive())
        rows = when_connection_free(lambda: self.con.sql("SELECT SUM(a) AS total FROM frame").pl())
        self.assertEqual(rows["total"].to_list(), [3])

    def test_gives_up_after_the_timeout(self) -> None:
        started = time.monotonic()
        with self.assertRaises(duckdb.ResourceInUseException):
            when_connection_free(lambda: self.con.execute("SELECT 1").fetchall(), timeout_s=0.05)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_other_errors_are_not_retried(self) -> None:
        self.worker.join()
        calls = []

        def failing():
            calls.append(1)
            return self.con.sql("SELECT nope FROM missing_table").pl()

        with self.assertRaises(duckdb.Error):
            when_connection_free(failing)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
