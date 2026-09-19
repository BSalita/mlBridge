import unittest

import polars as pl

from mlBridge.mlBridgeAugmentLib import (
    extract_declarer_from_contract,
    normalize_contract_columns,
)
from mlBridge.mlBridgeFFLib import (
    _lancelot_contract_result,
    _lancelot_score_field_kind,
    _lancelot_signed_ns_score,
    lancelot_lineup_player_name_expr,
    lancelot_seat_person,
)


class LancelotScoreTests(unittest.TestCase):
    def test_banded_percent_is_not_a_trick_score(self):
        self.assertEqual(_lancelot_score_field_kind("60%+"), "percent")
        self.assertIsNone(_lancelot_signed_ns_score("60%+", "60%+"))
        self.assertIsNone(_lancelot_signed_ns_score("60%+", "40%-"))
        self.assertIsNone(_lancelot_signed_ns_score("%Tournoi", "%Tournoi"))

    def test_numeric_scores_keep_direction(self):
        self.assertEqual(_lancelot_signed_ns_score("140", ""), 140)
        self.assertEqual(_lancelot_signed_ns_score("", "140"), -140)
        self.assertEqual(_lancelot_signed_ns_score("PASSE", ""), 0)

    def test_conflicting_numeric_scores_still_raise(self):
        with self.assertRaisesRegex(ValueError, "both populated"):
            _lancelot_signed_ns_score("140", "140")

    def test_lone_dash_is_missing_contract_result(self):
        self.assertIsNone(_lancelot_contract_result("-"))
        self.assertIsNone(_lancelot_contract_result(" - "))
        self.assertEqual(_lancelot_contract_result("+1"), 1)
        self.assertEqual(_lancelot_contract_result("-2"), -2)
        self.assertEqual(_lancelot_contract_result("="), 0)

    def test_ouest_declarer_maps_to_west(self):
        out = extract_declarer_from_contract(
            pl.DataFrame({"Contract": ["1SO", "2HN", "PASS", "1SX"]})
        )
        self.assertEqual(out["Declarer_Direction"].to_list(), ["W", "N", None, None])

    def test_normalize_contract_accepts_ouest_without_strict_replace(self):
        out = normalize_contract_columns(
            pl.DataFrame(
                {
                    "Contract": ["1SO", "PASS"],
                    "Player_Name_N": ["A", "B"],
                    "Player_Name_E": ["C", "D"],
                    "Player_Name_S": ["E", "F"],
                    "Player_Name_W": ["G", "H"],
                    "Vul_NS": [True, False],
                    "Vul_EW": [False, True],
                }
            )
        )
        self.assertEqual(out["Declarer_Direction"].to_list(), ["W", None])
        self.assertEqual(out["LHO_Direction"].to_list(), ["N", None])
        self.assertEqual(out["Dummy_Direction"].to_list(), ["E", None])
        self.assertEqual(out["RHO_Direction"].to_list(), ["S", None])

    def test_visitor_name_string_is_kept_as_seat_person(self):
        self.assertEqual(
            lancelot_seat_person("MADAR ."),
            {"firstName": "", "lastName": "MADAR ."},
        )
        frame = pl.DataFrame(
            {
                "lineup_northPlayer": ["MADAR ."],
                "lineup_northPlayer_firstName": [None],
                "lineup_northPlayer_lastName": [None],
                "lineup_southPlayer_firstName": ["Juliette"],
                "lineup_southPlayer_lastName": ["SYMCHOWICZ"],
            }
        )
        out = frame.select(
            lancelot_lineup_player_name_expr(frame, "lineup_northPlayer").alias("N"),
            lancelot_lineup_player_name_expr(frame, "lineup_southPlayer").alias("S"),
        )
        self.assertEqual(out["N"][0], "MADAR .")
        self.assertEqual(out["S"][0], "Juliette SYMCHOWICZ")


if __name__ == "__main__":
    unittest.main()
