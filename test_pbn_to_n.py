import unittest

from endplay.types import Deal

from mlBridge.mlBridgeFFLib import PbnToN


def _normalized(pbn: str) -> str:
    return Deal(pbn).to_pbn()


class PbnToNTests(unittest.TestCase):
    def test_north_is_identity(self) -> None:
        raw = (
            "N:J76.A9642.KJ.AT7 .KT875.AQ3.QJ984 AKQ98543.J3.T7.5 T2.Q.986542.K632"
        )
        self.assertEqual(PbnToN(raw), _normalized(raw))

    def test_east_lists_clockwise_to_north_first(self) -> None:
        raw = (
            "E:.KT875.AQ3.QJ984 AKQ98543.J3.T7.5 T2.Q.986542.K632 J76.A9642.KJ.AT7"
        )
        expected = _normalized(
            "N:J76.A9642.KJ.AT7 .KT875.AQ3.QJ984 AKQ98543.J3.T7.5 T2.Q.986542.K632"
        )
        self.assertEqual(PbnToN(raw), expected)

    def test_west_lists_clockwise_to_north_first(self) -> None:
        raw = (
            "W:875.AK6.T75.QT92 Q42.2.A8642.A753 JT9.J54.KJ9.K864 AK63.QT9873.Q3.J"
        )
        expected = _normalized(
            "N:Q42.2.A8642.A753 JT9.J54.KJ9.K864 AK63.QT9873.Q3.J 875.AK6.T75.QT92"
        )
        self.assertEqual(PbnToN(raw), expected)

    def test_south_lists_clockwise_to_north_first(self) -> None:
        raw = (
            "S:T2.Q.986542.K632 J76.A9642.KJ.AT7 .KT875.AQ3.QJ984 AKQ98543.J3.T7.5"
        )
        expected = _normalized(
            "N:.KT875.AQ3.QJ984 AKQ98543.J3.T7.5 T2.Q.986542.K632 J76.A9642.KJ.AT7"
        )
        self.assertEqual(PbnToN(raw), expected)

    def test_east_is_not_the_old_180_rotation(self) -> None:
        raw = (
            "E:.KT875.AQ3.QJ984 AKQ98543.J3.T7.5 T2.Q.986542.K632 J76.A9642.KJ.AT7"
        )
        old_wrong = _normalized(
            "N:AKQ98543.J3.T7.5 T2.Q.986542.K632 J76.A9642.KJ.AT7 .KT875.AQ3.QJ984"
        )
        self.assertNotEqual(PbnToN(raw), old_wrong)


if __name__ == "__main__":
    unittest.main()
