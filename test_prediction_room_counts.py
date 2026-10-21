import unittest

import polars as pl

from mlBridge.ffbridge_prediction_features import attach_room_counts


class RoomCountTests(unittest.TestCase):
    def test_counts_the_tables_that_played_each_rows_predicted_contract_and_seat(self) -> None:
        boards = pl.DataFrame(
            {
                "Board": [14, 14, 14, 14, 14, 3, 3],
                "Contract": ["4SW", "4SW", "4SXW", "PASS", "3NE", "PASS", None],
                "Declarer_Direction": ["W", "W", "W", None, "E", None, None],
                "Contract_Pred": ["4SW", "4SW", "3NE", "4SW", None, "PASS", "PASS"],
                "Declarer_Direction_Pred": ["W", "W", "E", "W", None, "NULL", "NULL"],
            }
        )
        counted = attach_room_counts(boards, ("Board",))
        self.assertEqual(counted["Contract"].to_list(), boards["Contract"].to_list())
        self.assertEqual(counted["Board_Tables"].to_list(), [5, 5, 5, 5, 5, 1, 1])
        # The doubled 4SXW is a different contract from 4SW, but the same seat.
        self.assertEqual(counted["Contract_Pred_Tables"].to_list(), [2, 2, 1, 2, None, 1, 1])
        self.assertEqual(counted["Declarer_Direction_Pred_Tables"].to_list(), [3, 3, 1, 3, None, 1, 1])
        self.assertEqual(attach_room_counts(counted, ("Board",)).columns, counted.columns)

    def test_a_null_predicted_seat_on_a_pass_out_counts_the_passed_out_tables(self) -> None:
        boards = pl.DataFrame(
            {
                "Board": [5, 5, 5],
                "Contract": ["PASS", "PASS", "2SS"],
                "Declarer_Direction": pl.Series([None, None, "S"], dtype=pl.Categorical),
                "Contract_Pred": ["PASS", "PASS", "PASS"],
                "Declarer_Direction_Pred": [None, None, None],
            }
        )
        counted = attach_room_counts(boards, ("Board",))
        self.assertEqual(counted["Contract_Pred_Tables"].to_list(), [2, 2, 2])
        self.assertEqual(counted["Declarer_Direction_Pred_Tables"].to_list(), [2, 2, 2])

    def test_two_deals_under_one_board_number_are_counted_apart(self) -> None:
        boards = pl.DataFrame(
            {
                "Board": [1, 1, 1],
                "PBN": ["N:deal-a", "N:deal-a", "N:deal-b"],
                "Contract": ["3NN", "3NN", "3NN"],
                "Declarer_Direction": ["N", "N", "N"],
                "Contract_Pred": ["3NN", "3NN", "3NN"],
                "Declarer_Direction_Pred": ["N", "N", "N"],
            }
        )
        counted = attach_room_counts(boards, ("Board", "PBN"))
        self.assertEqual(counted["Board_Tables"].to_list(), [2, 2, 1])
        self.assertEqual(counted["Contract_Pred_Tables"].to_list(), [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
