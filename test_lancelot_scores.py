import unittest

from mlBridge.mlBridgeFFLib import (
    _lancelot_contract_result,
    _lancelot_score_field_kind,
    _lancelot_signed_ns_score,
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


if __name__ == "__main__":
    unittest.main()
