from __future__ import annotations

import os
import unittest
from pathlib import Path

from mlBridge import mlBridgeApiCommon as api_common  # src/ on PYTHONPATH


class ApiCommonTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = os.environ.pop(api_common.DATA_ROOT_ENV, None)

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop(api_common.DATA_ROOT_ENV, None)
        else:
            os.environ[api_common.DATA_ROOT_ENV] = self._previous

    def test_prepare_sql_rejects_forbidden_and_qualifies_from(self) -> None:
        sql, limit = api_common.prepare_sql("SELECT 1", "club_board_results", 99999)
        self.assertEqual(sql, "FROM self SELECT 1")
        self.assertEqual(limit, api_common.MAX_SQL_ROW_LIMIT)
        with self.assertRaises(ValueError):
            api_common.prepare_sql("COPY self TO 'x'", "club_board_results")

    def test_favorites_and_health(self) -> None:
        payload = {
            "Buttons": {"b1": {"prompts": ["@fav1"]}},
            "SelectBoxes": {
                "Vetted_Prompts": {
                    "fav1": {
                        "title": "One",
                        "prompts": [{"prompt": "rows", "sql": "SELECT 1"}],
                    }
                }
            },
        }
        favorites = api_common.flatten_favorites_payload(payload)
        listed = api_common.select_favorites(favorites, "fav1")
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["favorites"][0]["buttons"], ["b1"])
        with self.assertRaises(KeyError):
            api_common.select_favorites(favorites, "missing")
        health = api_common.health_payload(
            {"data_path": "/data"},
            service="acbl-stats-api",
            api_version="1.0.0",
            build_tag="test",
        )
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["service"], "acbl-stats-api")

    def test_data_root_search_paths(self) -> None:
        os.environ[api_common.DATA_ROOT_ENV] = "/data"
        paths = api_common.data_root_search_paths("acbl")
        self.assertEqual(paths[0], Path("/data/stats/acbl"))
        self.assertEqual(
            paths[1], Path("/data/_wslc_host/acbl-stage/club_results_parquet")
        )


if __name__ == "__main__":
    unittest.main()
