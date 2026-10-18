"""Ranking-row clicks keep the hourglass until the follow-up grids paint."""

from __future__ import annotations

import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mlBridge.postmortem_section_help import section_waits_on_row_click

HARNESS = (
    Path(__file__).resolve().parents[1]
    / "elo"
    / "streamlitlib"
    / "tests"
    / "ranking_row_hourglass.html"
)


class SectionWaitTests(unittest.TestCase):
    def test_ranking_sections_wait_on_row_click(self) -> None:
        for favorite_id in (
            "Par_Rankings",
            "Declarer_Rankings",
            "Overall_Rankings_Predictions",
            "Rankings_Without_Gifts",
        ):
            self.assertTrue(section_waits_on_row_click(favorite_id), favorite_id)

    def test_other_sections_do_not_wait_on_row_click(self) -> None:
        for favorite_id in ("Boards_I_Declared", "Summarize", "Successful_Sacrifices"):
            self.assertFalse(section_waits_on_row_click(favorite_id), favorite_id)


class _SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return


class RankingHourglassHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise unittest.SkipTest("playwright is required") from exc
        cls._playwright = sync_playwright
        root = HARNESS.parents[1]
        handler = partial(_SilentHandler, directory=str(root))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        server = getattr(cls, "server", None)
        if server is None:
            return
        server.shutdown()
        server.server_close()
        cls.thread.join(timeout=5)

    def test_hourglass_lasts_until_grids_paint(self) -> None:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(f"{self.base}/tests/ranking_row_hourglass.html")
            page.wait_for_function("window.__hourglass")
            result = page.evaluate("window.__hourglass")
            browser.close()
        self.assertEqual(result["before"], "wait", result)
        self.assertEqual(result["cellCursor"], "wait", result)
        self.assertEqual(result["during"], "wait", result)
        self.assertEqual(result["stillWaiting"], "wait", result)
        self.assertEqual(result["after"], "auto", result)
        self.assertFalse(result["classAfter"], result)
