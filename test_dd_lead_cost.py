import unittest

import polars as pl
from endplay.dds import solve_board
from endplay.types import Deal, Denom, Player

from mlBridge import dds_ddss
from mlBridge.mlBridgeAugmentLib import add_dd_lead_cost

PBN = "N:AKQ2.T98.765.432 JT9.AKQ.432.8765 876.765.AKQ.AKQJ 543.J432.JT98.T9"
DENOM = {"S": Denom.spades, "H": Denom.hearts, "D": Denom.diamonds, "C": Denom.clubs, "N": Denom.nt}
SEAT = {"N": Player.north, "E": Player.east, "S": Player.south, "W": Player.west}


def endplay_lead_tricks(strain: str, leader: str) -> dict:
    """Defenders' tricks per lead from endplay's own DDS (one card per equal run)."""
    deal = Deal(PBN)
    deal.trump = DENOM[strain]
    deal.first = SEAT[leader]
    return {"SHDC"[card.suit.value] + card.rank.abbr: tricks for card, tricks in solve_board(deal)}


@unittest.skipUnless(dds_ddss.DDSS_AVAILABLE, "ddss DLL not available")
class SolveOpeningLeadsTests(unittest.TestCase):
    def test_matches_endplay_and_scores_every_card(self) -> None:
        positions = [(PBN, strain, leader) for strain in "SHDCN" for leader in "NESW"]
        solved = dds_ddss.solve_opening_leads(positions, chunk_size=7)  # several chunks
        tables = dds_ddss.calc_all_tables_pbnx([PBN])[0]
        for (pbn, strain, leader), tricks in zip(positions, solved):
            with self.subTest(strain=strain, leader=leader):
                self.assertEqual(len(tricks), 13)
                expected = endplay_lead_tricks(strain, leader)
                self.assertEqual({card: tricks[card] for card in expected}, expected)
                # The best lead holds declarer (leader's right-hand opponent) to DD tricks.
                declarer = "NESW"[("NESW".index(leader) + 3) % 4]
                dd = tables["SHDCN".index(strain), "NESW".index(declarer)]
                self.assertEqual(max(tricks.values()), 13 - dd)

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            dds_ddss.solve_opening_leads([(PBN, "X", "W")])
        with self.assertRaises(ValueError):
            dds_ddss.solve_opening_leads([(PBN, "N", "Z")])


@unittest.skipUnless(dds_ddss.DDSS_AVAILABLE, "ddss DLL not available")
class AddDdLeadCostTests(unittest.TestCase):
    def test_costs_nulls_and_row_order(self) -> None:
        # South declares notrump, so West leads. West: S 543  H J432  D JT98  C T9.
        ref = endplay_lead_tricks("N", "W")
        best = max(ref.values())
        frame = pl.DataFrame(
            {
                "PBN": [PBN] * 7,
                "BidSuit": ["N", "N", "N", "N", None, "N", "N"],
                "Declarer_Direction": ["S", "S", "S", "S", "S", "S", "S"],
                "Lead": ["DJ", "HJ", "c10", "SA", "DJ", None, " "],
                "row": list(range(7)),
            }
        )
        out = add_dd_lead_cost(frame)
        self.assertEqual(out["row"].to_list(), list(range(7)))
        self.assertEqual(out.schema["DD_Lead_Cost"], pl.UInt8)
        self.assertEqual(out.schema["Lead_Is_DD_Best"], pl.Boolean)
        costs = out["DD_Lead_Cost"].to_list()
        self.assertEqual(costs[0], best - ref["DJ"])
        self.assertEqual(costs[1], best - ref["HJ"])
        self.assertEqual(costs[2], best - ref.get("CT", ref.get("C9")))  # "c10" read as CT
        self.assertEqual(costs[3:], [None, None, None, None])  # not held, passed out, no lead, blank
        self.assertEqual(out["Lead_Is_DD_Best"].to_list()[:3], [c == 0 for c in costs[:3]])
        self.assertEqual(out["Lead_Is_DD_Best"].to_list()[3:], [None] * 4)

    def test_sd_lead_score(self) -> None:
        from mlBridge.mlBridgeAugmentLib import SDLeadCache, add_sd_lead_score, sample_lead_layouts

        frame = pl.DataFrame(
            {
                "PBN": [PBN] * 4,
                "BidSuit": ["N", "N", "H", "N"],
                "BidLvl": [3, 1, 4, 3],
                "Declarer_Direction": ["S", "S", "N", "S"],
                "Lead": ["DJ", "HJ", "C5", "SA"],  # SA: West does not hold it
            }
        )
        frame = add_dd_lead_cost(frame)
        frame = frame.with_columns(pl.Series("DD_Tricks", [8, 8, 10, 8], dtype=pl.UInt8))
        cache = SDLeadCache(samples=12, mode="contract")
        out = add_sd_lead_score(frame, cache)
        self.assertEqual(out.height, 4)
        scored = out.head(3)
        self.assertTrue((scored["SD_Lead_Cost"] >= 0).all())
        self.assertTrue((scored["SD_Lead_Set_Pct"].is_between(0, 1)).all())
        self.assertTrue((scored["SD_Lead_Samples"] == 12).all())
        self.assertEqual(out.row(3, named=True)["SD_Lead_Cost"], None)
        # Rows 0 and 1 are the same position (South in notrump): same layouts, so the
        # same best lead; 1N needs 7 defensive tricks to set, 3N only 5.
        self.assertEqual(scored["SD_Lead_Best_Card"][0], scored["SD_Lead_Best_Card"][1])
        self.assertEqual(scored["SD_Lead_Best_Tricks"][0], scored["SD_Lead_Best_Tricks"][1])
        self.assertLessEqual(scored["SD_Lead_Best_Set_Pct"][1], scored["SD_Lead_Best_Set_Pct"][0])
        # Reproducible, and a second call solves nothing new.
        again = add_sd_lead_score(frame, cache)
        self.assertTrue(again.equals(out))
        # Contract mode keeps the declaring side's HCP within 2 and its trump length.
        layouts = sample_lead_layouts(PBN, "H", "E", 12, "contract")
        self.assertEqual(len(layouts), 12)
        hcp = {"A": 4, "K": 3, "Q": 2, "J": 1}
        def side(pbn: str) -> tuple:
            hands = pbn[2:].split()
            ns = hands[0].split(".") + hands[2].split(".")
            return sum(hcp.get(c, 0) for s in ns for c in s), len(ns[1]) + len(ns[5])
        real = side(PBN)
        for layout in layouts:
            self.assertTrue(layout.split()[1] == PBN.split()[1])  # East's hand is kept
            got = side(layout)
            self.assertLessEqual(abs(got[0] - real[0]), 2)
            self.assertEqual(got[1], real[1])

    def test_existing_columns_rejected(self) -> None:
        frame = pl.DataFrame({"PBN": [PBN], "BidSuit": ["N"], "Declarer_Direction": ["S"],
                              "Lead": ["DJ"], "DD_Lead_Cost": [0]})
        with self.assertRaises(AssertionError):
            add_dd_lead_cost(frame)


if __name__ == "__main__":
    unittest.main()
