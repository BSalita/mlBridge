import unittest
from unittest import mock

import polars as pl
from endplay.dealer import generate_deals
from endplay.types import Deal

from mlBridge import mlBridgeAugmentLib as augment_lib
from mlBridge.mlBridgeAugmentLib import (
    deal_generation_constraints,
    estimate_sd_trick_distributions,
    estimate_sd_trick_distributions_for_df,
    generate_unconstrained_deal_pbns,
)

DEALS = (
    "N:J76.A9642.KJ.AT7 .KT875.AQ3.QJ984 AKQ98543.J3.T7.5 T2.Q.986542.K632",
    "N:Q42.2.A8642.A753 JT9.J54.KJ9.K864 AK63.QT9873.Q3.J 875.AK6.T75.QT92",
)


def _predeals(pbn: str) -> list[str]:
    hands = pbn[2:].split()
    ns = ["N:" + " ".join([hands[0], "...", hands[2], "..."])]
    ew = ["N:" + " ".join(["...", hands[1], "...", hands[3]])]
    return ns + ew


class UnconstrainedDealPbnTests(unittest.TestCase):
    def test_matches_endplay_generate_deals(self) -> None:
        for pbn in DEALS:
            for predeal_string in _predeals(pbn):
                predeal = Deal(predeal_string)
                expected = [
                    deal.to_pbn()
                    for deal in generate_deals(
                        deal_generation_constraints,
                        predeal=predeal,
                        swapping=0,
                        produce=25,
                        seed=42,
                        strict=True,
                    )
                ]
                got = generate_unconstrained_deal_pbns(Deal(predeal_string), 25, seed=42)
                self.assertEqual(got, expected)

    def test_pbns_are_canonical_and_keep_predealt_hands(self) -> None:
        predeal_string = _predeals(DEALS[0])[0]
        fixed = predeal_string[2:].split()
        for pbn in generate_unconstrained_deal_pbns(Deal(predeal_string), 10, seed=7):
            self.assertEqual(pbn, Deal(pbn).to_pbn())
            hands = pbn[2:].split()
            self.assertEqual(hands[0], fixed[0])
            self.assertEqual(hands[2], fixed[2])


class EstimatorSeedTests(unittest.TestCase):
    def test_default_seed_is_42(self) -> None:
        default, _ = estimate_sd_trick_distributions(DEALS[0], 5)
        explicit, _ = estimate_sd_trick_distributions(DEALS[0], 5, seed=42)
        for side in ("NS", "EW"):
            self.assertEqual(default[side]["SD_Deal"].to_list(), explicit[side]["SD_Deal"].to_list())

    def test_seed_changes_the_deals_and_repeats_them(self) -> None:
        first, _ = estimate_sd_trick_distributions(DEALS[0], 5, seed=7)
        again, _ = estimate_sd_trick_distributions(DEALS[0], 5, seed=7)
        other, _ = estimate_sd_trick_distributions(DEALS[0], 5, seed=8)
        for side in ("NS", "EW"):
            self.assertEqual(first[side]["SD_Deal"].to_list(), again[side]["SD_Deal"].to_list())
            self.assertNotEqual(first[side]["SD_Deal"].to_list(), other[side]["SD_Deal"].to_list())

    def test_for_df_passes_its_seed_to_deal_generation(self) -> None:
        seeds = []

        class Stop(Exception):
            pass

        def record(predeal, produce, seed):
            seeds.append(seed)
            raise Stop

        df = pl.DataFrame({"PBN": [DEALS[0]], "Dealer": ["N"], "Vul": ["None"]})
        cache = pl.DataFrame(
            schema={"PBN": pl.String, "Dealer": pl.String, "Vul": pl.String, "Probs_Trials": pl.UInt16}
        )
        with mock.patch.object(augment_lib, "generate_unconstrained_deal_pbns", record):
            for kwargs, want in (({}, 42), ({"seed": 9}, 9)):
                seeds.clear()
                with self.assertRaises(Stop):
                    estimate_sd_trick_distributions_for_df(df, cache, 5, **kwargs)
                self.assertEqual(seeds, [want])


if __name__ == "__main__":
    unittest.main()
