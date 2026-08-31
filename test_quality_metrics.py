import unittest

import polars as pl

from mlBridge.mlBridgeAugmentLib import (
    add_position_role_info,
    create_score_diff_columns,
)


class QualityMetricTests(unittest.TestCase):
    def test_position_roles_emit_correct_quality_semantics(self):
        par_contracts_dtype = pl.List(
            pl.Struct(
                {
                    "Level": pl.String,
                    "Strain": pl.String,
                    "Doubled": pl.String,
                    "Pair_Direction": pl.String,
                    "Result": pl.Int16,
                }
            )
        )
        df = pl.DataFrame(
            {
                "Declarer_Direction": ["N", "E", "S", None],
                "Declarer_Pair_Direction": ["NS", "EW", "NS", None],
                "Player_ID_N": ["n"] * 4,
                "Player_ID_E": ["e"] * 4,
                "Player_ID_S": ["s"] * 4,
                "Player_ID_W": ["w"] * 4,
                "BidSuit": ["S", "H", "D", None],
                "BidLvl": [4, 5, 3, None],
                "Vul_Declarer": [False, True, False, None],
                "Contract": ["4S", "5HX", "3D", "PASS"],
                "Score_NS": [620, 100, 110, 0],
                "Score_EW": [-620, -100, -110, 0],
                "Par_NS": [620, 100, 100, None],
                "Par_EW": [-620, -100, -100, None],
                "DD_Score_NS": [620, 100, 90, None],
                "DD_Score_EW": [-620, -100, -90, None],
                "DD_Score_Declarer": [620, -100, 90, None],
                "ParContracts": pl.Series(
                    [
                        [
                            {
                                "Level": "4",
                                "Strain": "S",
                                "Doubled": "",
                                "Pair_Direction": "NS",
                                "Result": 0,
                            }
                        ],
                        [
                            {
                                "Level": "5",
                                "Strain": "C",
                                "Doubled": "X",
                                "Pair_Direction": "EW",
                                "Result": -1,
                            }
                        ],
                        [],
                        None,
                    ],
                    dtype=par_contracts_dtype,
                ),
            }
        )

        result = add_position_role_info(df)

        self.assertEqual(result["Par_Contract_NS"].to_list(), [1, 1, -1, None])
        self.assertEqual(result["Par_Contract_EW"].to_list(), [1, 1, 1, None])
        self.assertEqual(result["Is_Par_Contract"].to_list(), [True, True, False, None])
        self.assertEqual(result["Is_Par_Suit"].to_list(), [True, False, False, None])
        self.assertEqual(
            result["Is_Sacrifice_Opportunity"].to_list(),
            [False, True, False, None],
        )
        self.assertEqual(result["Is_Sacrifice"].to_list(), [False, True, False, None])

    def test_t_minus_dd_remains_declarer_trick_difference(self):
        df = pl.DataFrame(
            {
                "Score_NS": [420, -50],
                "Score_EW": [-420, 50],
                "Par_NS": [420, 100],
                "Par_EW": [-420, -100],
                "EV_Max_NS": [400.0, 90.0],
                "EV_Max_EW": [-400.0, -90.0],
                "Tricks": [10, 7],
                "DD_Tricks": [9, 8],
            }
        )

        result = create_score_diff_columns(df)

        self.assertEqual(result["DD_Tricks_Diff"].to_list(), [1, -1])


if __name__ == "__main__":
    unittest.main()
