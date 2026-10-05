import unittest

import polars as pl

from mlBridge.mlBridgeAugmentLib import player_id_for_direction


class PlayerIdForDirectionTests(unittest.TestCase):
    def test_integer_and_temporary_ids_become_text(self) -> None:
        frame = pl.DataFrame(
            {
                "Declarer_Direction": ["N", "E", None, "W"],
                "Player_ID_N": pl.Series([5798205, 1, 2, 3], dtype=pl.Int32),
                "Player_ID_E": pl.Series([4, 8, 9, 10], dtype=pl.Int32),
                "Player_ID_S": pl.Series([11, 12, 13, 14], dtype=pl.Int32),
                "Player_ID_W": ["#155", "16", None, "7022131"],
            }
        )
        parsed = frame.select(player_id_for_direction("Declarer_Direction").alias("Declarer"))
        self.assertEqual(parsed["Declarer"].dtype, pl.String)
        self.assertEqual(parsed["Declarer"].to_list(), ["5798205", "8", None, "7022131"])


if __name__ == "__main__":
    unittest.main()
