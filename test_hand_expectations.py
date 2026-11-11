import unittest

import polars as pl
from endplay.types import Deal

from mlBridge.mlBridgeAugmentLib import (
    estimate_hand_trick_expectations,
    generate_unconstrained_deal_pbns,
    hand_trick_expectations_for_boards,
    solve_dd_for_pbns,
)
from mlBridge.mlBridgePostmortemLib import HAND_EXPECTATIONS_TABLE, query_reads_own_table

DEAL = "N:J76.A9642.KJ.AT7 .KT875.AQ3.QJ984 AKQ98543.J3.T7.5 T2.Q.986542.K632"
# Each hand holds one whole suit: N spades, E hearts, S diamonds, W clubs.
ONE_SUIT_HANDS = "N:AKQJT98765432... .AKQJT98765432.. ..AKQJT98765432. ...AKQJT98765432"
PROB_COLUMNS = [f"Prob_Taking_{n}" for n in range(14)]


class HandTrickExpectationTests(unittest.TestCase):
    def test_one_row_per_deal_direction_and_strain(self) -> None:
        df = estimate_hand_trick_expectations([DEAL, ONE_SUIT_HANDS, DEAL], produce=3)
        self.assertEqual(df.columns, ["PBN", "Direction", "Strain", *PROB_COLUMNS, "Shuffles"])
        self.assertEqual(df.height, 2 * 4 * 5)
        self.assertEqual(df["PBN"].unique(maintain_order=True).to_list(), [DEAL, ONE_SUIT_HANDS])
        first = df.filter(pl.col("PBN") == DEAL)
        self.assertEqual(first["Direction"].to_list(), [d for d in "NESW" for _ in range(5)])
        self.assertEqual(first["Strain"].to_list(), list("SHDCN") * 4)
        self.assertEqual(df["Shuffles"].unique().to_list(), [3])

    def test_probabilities_are_frequencies_that_sum_to_one(self) -> None:
        df = estimate_hand_trick_expectations([DEAL], produce=4)
        for row in df.select(PROB_COLUMNS).iter_rows():
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            for prob in row:
                self.assertAlmostEqual(prob * 4, round(prob * 4), places=5)

    def test_a_hand_holding_a_whole_suit_takes_13_tricks_in_it(self) -> None:
        df = estimate_hand_trick_expectations([ONE_SUIT_HANDS], produce=5)
        for direction, strain in zip("NESW", "SHDC"):
            row = df.filter((pl.col("Direction") == direction) & (pl.col("Strain") == strain))
            self.assertEqual(row["Prob_Taking_13"].item(), 1.0, f"{direction} {strain}")

    def test_matches_solving_the_same_shuffles_directly(self) -> None:
        produce = 6
        df = estimate_hand_trick_expectations([DEAL], produce=produce, seed=7)
        hands = DEAL[2:].split()
        # East keeps its hand; hearts is row 1 of the SHDCN table, East column 1 of NESW.
        shuffles = generate_unconstrained_deal_pbns(
            Deal("N:" + " ".join(["...", hands[1], "...", "..."])), produce, seed=7
        )
        for pbn in shuffles:
            self.assertEqual(pbn[2:].split()[1], hands[1])
        tricks = [table.to_list()[1][1] for table in solve_dd_for_pbns(shuffles)]
        expected = [tricks.count(n) / produce for n in range(14)]
        got = df.filter((pl.col("Direction") == "E") & (pl.col("Strain") == "H")).select(PROB_COLUMNS).row(0)
        for want, have in zip(expected, got):
            self.assertAlmostEqual(want, have, places=5)

    def test_same_seed_gives_the_same_table(self) -> None:
        first = estimate_hand_trick_expectations([DEAL], produce=3, seed=11)
        second = estimate_hand_trick_expectations([DEAL], produce=3, seed=11)
        self.assertTrue(first.equals(second))


class BoardExpectationTests(unittest.TestCase):
    def test_boards_are_solved_once_and_sorted(self) -> None:
        session = pl.DataFrame(
            {
                "Board": [2, 1, 2, 1, None],
                "PBN": [ONE_SUIT_HANDS, DEAL, ONE_SUIT_HANDS, DEAL, None],
                "Score_NS": [100, 620, -50, 650, 0],
            }
        )
        df = hand_trick_expectations_for_boards(session, produce=2)
        self.assertEqual(df.columns, ["Board", "Direction", "Strain", *PROB_COLUMNS, "Shuffles", "PBN"])
        self.assertEqual(df.height, 2 * 4 * 5)
        self.assertEqual(df["Board"].to_list(), [1] * 20 + [2] * 20)
        self.assertEqual(df["Direction"].to_list()[:20], [d for d in "NESW" for _ in range(5)])
        self.assertEqual(df["Strain"].to_list()[:10], list("SHDCN") * 2)
        self.assertEqual(df.filter(pl.col("Board") == 2)["PBN"].unique().to_list(), [ONE_SUIT_HANDS])


class OwnTableQueryTests(unittest.TestCase):
    def test_only_a_query_naming_the_table_is_left_without_self(self) -> None:
        self.assertTrue(query_reads_own_table(f"SELECT Board FROM {HAND_EXPECTATIONS_TABLE} ORDER BY Board"))
        self.assertTrue(query_reads_own_table(f"select * from {HAND_EXPECTATIONS_TABLE}"))
        self.assertFalse(query_reads_own_table("SELECT Board FROM self"))
        self.assertFalse(query_reads_own_table("SELECT Board, Contract ORDER BY Board"))


if __name__ == "__main__":
    unittest.main()
