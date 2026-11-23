import unittest

import polars as pl

from mlBridge.mlBridgeAcblLib import normalize_brs_expr
from mlBridge.mlBridgeLib import brs_to_pbn

# Club session 1549918, board 1, as ACBL sends it from about May 2026 on.
SPACED = ("S Q 6 3H Q J 4 2D K 5C Q J 9 3S K TH T 9D A Q T 7 6C A T 8 2"
          "S A J 8 7H K 8 7 5 3D J 4C 7 6S 9 5 4 2H A 6D 9 8 3 2C K 5 4")
# The same deal in the older form.
OLDER = ("SQ63HQJ42DK5CQJ93SK10H109DAQ1076CA1082"
         "SAJ87HK8753DJ4C76S9542HA6D9832CK54")


class NormalizeBrsTests(unittest.TestCase):
    def _norm(self, values):
        return pl.DataFrame({'board_record_string': values}).select(
            normalize_brs_expr())['board_record_string'].to_list()

    def test_spaced_form_becomes_the_older_form(self) -> None:
        self.assertEqual(self._norm([SPACED]), [OLDER])

    def test_older_form_and_nulls_are_unchanged(self) -> None:
        self.assertEqual(self._norm([OLDER, None]), [OLDER, None])

    def test_normalized_string_gives_the_same_pbn(self) -> None:
        with self.assertRaises(RuntimeError):
            brs_to_pbn(SPACED)
        self.assertEqual(
            brs_to_pbn(self._norm([SPACED])[0]),
            "N:Q63.QJ42.K5.QJ93 AJ87.K8753.J4.76 9542.A6.9832.K54 KT.T9.AQT76.AT82")


if __name__ == "__main__":
    unittest.main()
