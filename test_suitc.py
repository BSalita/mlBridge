import tempfile
import unittest
from pathlib import Path

import polars as pl

from mlBridge import mlBridgeSuitCLib as sc

# SuitC's own documented example (README): AQ2 opposite 543.
AQ2_OUT = """
	min number of tricks: 1
	max number of tricks: 2
	number of plays     : 2

The optimal results
===================

	2 tricks:  50.4845
	Result  :   1.5048

       Probabilities    West       East       tricks
     ===============    ====       ====       ======
     1 x  0.26086957    KJT9876    -           2 - 2
     6 x  0.48447205    KJT987     6           1 - 2
    15 x  0.72670807    KJT98      76          1 - 2
    20 x  0.88819876    KJT9       876         1 - 2
    15 x  0.88819876    KJT        9876        1 - 2
     6 x  0.72670807    KJ         T9876       2 - 2
     1 x  0.48447205    K          JT9876      2 - 2
     1 x  0.48447205    JT9876     K           2 - 2
     6 x  0.72670807    JT987      K6          1 - 2
    15 x  0.88819876    JT98       K76         1 - 1
    20 x  0.88819876    JT9        K876        1 - 1
    15 x  0.72670807    JT         K9876       1 - 1
     6 x  0.48447205    J          KT9876      1 - 1
     1 x  0.26086957    -          KJT9876     1 - 1

The optimum play for: MAX N2
============================

	2 tricks:  50.4845
	1 tricks:  49.5155
	Result  :   1.5048
"""


class ParseTests(unittest.TestCase):
    def test_optimal_results(self) -> None:
        et, probs = sc._optimal(AQ2_OUT)
        self.assertAlmostEqual(et, 1.5048)
        self.assertEqual(probs[0], 1.0)  # below SuitC's minimum: certain
        self.assertAlmostEqual(probs[1], 0.504845)
        self.assertEqual(probs[2:], [0.0] * 11)  # above its maximum

    def test_dd_expectation(self) -> None:
        # Splits where the best line takes 2 (upper end of each range).
        expected = sum(n * p * hi for n, p, hi in [
            (1, .26086957, 2), (6, .48447205, 2), (15, .72670807, 2), (20, .88819876, 2), (15, .88819876, 2),
            (6, .72670807, 2), (1, .48447205, 2), (1, .48447205, 2), (6, .72670807, 2), (15, .88819876, 1),
            (20, .88819876, 1), (15, .72670807, 1), (6, .48447205, 1), (1, .26086957, 1)]) / 100
        self.assertAlmostEqual(sc._dd_expectation(AQ2_OUT), expected, places=6)

    def test_holding_key_is_order_free(self) -> None:
        self.assertEqual(sc.holding_key("543", "AQ2"), sc.holding_key("AQ2", "543"))


@unittest.skipUnless(sc.DEFAULT_SUITC_EXE.is_file(), "SuitC.exe not built")
class SuitCRunTests(unittest.TestCase):
    def test_ten_card_and_single_line_outputs(self) -> None:
        # "10 -10" trick ranges, and output with no "optimal results" section.
        for a, b, et in (("AKT9876543", "QJ", 10.0), ("7", "AQJT985432", 9.52)):
            r = sc.analyse_holding(a, b)
            self.assertEqual(r["Status"], "ok", (a, b))
            self.assertAlmostEqual(r["ET"], et, places=4)

    def test_error_rows_are_retried_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = sc.SuitCCache(Path(tmp), workers=2)
            cache._pending = [dict(sc.analyse_holding("AQ2", "543"), Status="error", ET=None)]
            cache._flush()
            self.assertEqual(cache.ensure([("AQ2", "543")], show_progress=False), 1)
            rows = cache.table().filter((pl.col("A") == "AQ2") & (pl.col("B") == "543"))
            self.assertEqual(rows.height, 1)
            self.assertEqual(rows["Status"][0], "ok")
            self.assertEqual(sc.SuitCCache(Path(tmp)).table().height, 1)  # reload keeps the last row

    def test_holdings(self) -> None:
        r = sc.analyse_holding("AQ2", "543")
        self.assertEqual(r["Status"], "ok")
        self.assertAlmostEqual(r["ET"], 1.5048, places=4)
        self.assertGreaterEqual(r["DDx"], r["ET"])
        self.assertIsNone(r["Stop"])  # defenders hold fewer than twice the cards
        self.assertEqual(sc.analyse_holding("AK", "765432")["Block"], 4.6391 - 2.0)  # no entry to the long hand
        self.assertAlmostEqual(sc.analyse_holding("K2", "3")["Stop"], 0.500054, places=5)
        void = sc.analyse_holding("", "")
        self.assertEqual((void["ET"], void["DDx"], void["Stop"]), (0.0, 0.0, 0.0))
        all13 = sc.analyse_holding("AKQJT98", "765432")
        self.assertEqual((all13["ET"], all13["P"][6], all13["P"][7]), (7.0, 1.0, 0.0))

    def test_features_and_declarer_columns(self) -> None:
        pbn = "N:AQ2.KJ9.AKQ.5432 KJT9876.AQ.876.T 543.432.JT9.AKQJ .T8765.5432.9876"
        with tempfile.TemporaryDirectory() as tmp:
            cache = sc.SuitCCache(Path(tmp), workers=4)
            frame = pl.DataFrame({"PBN": [pbn, pbn], "Declarer_Direction": ["S", "E"], "DD_Tricks_Diff": [0, -1]})
            out = sc.add_suitc_features(frame, cache, show_progress=False)
            self.assertEqual(out.height, 2)
            self.assertEqual(out.columns[:3], frame.columns)
            self.assertEqual(set(sc.sc_columns()) - set(out.columns), set())
            self.assertAlmostEqual(out["SC_ET_NS_S"][0], 1.5048, places=4)  # AQ2 / 543
            self.assertAlmostEqual(out["SC_ET_NS_H"][0], 1.0412, places=4)  # KJ9 / 432
            total = sum(out[f"SC_ET_NS_{s}"][0] for s in "SHDC")
            self.assertAlmostEqual(out["SC_ET_NS"][0], total, places=4)
            # A second cache over the same folder reads the saved part: nothing new to run.
            self.assertEqual(sc.SuitCCache(Path(tmp)).ensure(sc._suit_holdings(frame).select("A", "B").rows()), 0)
            dec = sc.add_suitc_declarer_columns(out)
            self.assertAlmostEqual(dec["SC_Guess_Declarer"][0], out["SC_Guess_NS"][0], places=5)
            self.assertAlmostEqual(dec["SC_Guess_Declarer"][1], out["SC_Guess_EW"][1], places=5)
            self.assertAlmostEqual(dec["DD_Tricks_Diff_SC"][1], -1 + out["SC_Guess_EW"][1], places=5)


if __name__ == "__main__":
    unittest.main()
