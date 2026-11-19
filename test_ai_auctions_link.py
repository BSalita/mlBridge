"""The AI Auctions link: boards packed by mlBridgePostmortemLib are read back unchanged by the explorer's rule.

  python -m pytest mlBridge/test_ai_auctions_link.py     (from src, with src on PYTHONPATH)
"""
import base64
from urllib.parse import parse_qs, urlsplit

import polars as pl
import pytest

from mlBridge.mlBridgePostmortemLib import AI_AUCTIONS_URL, ai_auctions_url, pack_boards

SEATS = "NESW"
RANKS = "AKQJT98765432"
VULS = ("None", "NS", "EW", "Both")
NORTH_FIRST = "N:AQJ9652.64.T3.82 K8.KQ973.A85.K93 T7.AJ5.KQ962.AQ4 43.T82.J74.JT765"
EAST_FIRST = "E:K8.KQ973.A85.K93 T7.AJ5.KQ962.AQ4 43.T82.J74.JT765 AQJ9652.64.T3.82"
HANDS = dict(zip(SEATS, NORTH_FIRST[2:].split()))


def unpack(text):
    """The explorer's reading of ?boards= (game_boards in streamlit_app_auction_explorer.py)."""
    assert text[0] == "1"
    raw = base64.urlsafe_b64decode(text[1:] + "=" * (-(len(text) - 1) % 4))
    assert raw and len(raw) % 16 == 0
    boards = []
    for at in range(0, len(raw), 16):
        chunk = raw[at:at + 16]
        holders = int.from_bytes(chunk[3:], "big")
        hands = [["", "", "", ""] for _ in SEATS]
        for card in range(52):
            hands[(holders >> 2 * (51 - card)) & 3][card // 13] += RANKS[card % 13]
        boards.append((int.from_bytes(chunk[:2], "big"), SEATS[chunk[2] & 3], VULS[chunk[2] >> 2 & 3],
                       {s: ".".join(hands[i]) for i, s in enumerate(SEATS)}))
    return boards


def test_round_trip_any_first_seat_and_vulnerability_spelling():
    packed = pack_boards([(1, NORTH_FIRST, "N", "None"), (17, EAST_FIRST, "W", "E_W"), (300, NORTH_FIRST, "S", "Both")])
    assert unpack(packed) == [(1, "N", "None", HANDS), (17, "W", "EW", HANDS), (300, "S", "Both", HANDS)]
    assert len(packed) == 1 + 64                      # 22 characters a board


def test_a_bad_deal_is_refused():
    with pytest.raises(ValueError, match="different cards"):
        pack_boards([(1, NORTH_FIRST.replace("AQJ9652", "AQJ965K"), "N", "None")])
    with pytest.raises(ValueError, match="vulnerability"):
        pack_boards([(1, NORTH_FIRST, "N", "Red")])
    with pytest.raises(ValueError):
        pack_boards([(1, NORTH_FIRST[2:], "N", "None")])


def test_url_holds_each_deal_once_and_nothing_personal():
    df = pl.DataFrame({
        "Board": [2, 1, 1, 2], "PBN": [EAST_FIRST, NORTH_FIRST, NORTH_FIRST, EAST_FIRST],
        "Dealer": ["E", "N", "N", "E"], "Vul": ["N_S", "None", "None", "N_S"],
        "Player_Name_N": ["Someone"] * 4, "Score_NS": [420, 50, -100, 170],
    })
    url = ai_auctions_url(df, "acbl", "E")
    assert url.startswith(AI_AUCTIONS_URL + "/?")
    asked = parse_qs(urlsplit(url).query)
    assert set(asked) == {"from", "boards", "seat"} and asked["from"] == ["acbl"] and asked["seat"] == ["E"]
    assert [(b[0], b[1], b[2]) for b in unpack(asked["boards"][0])] == [(1, "N", "None"), (2, "E", "NS")]


def test_played_contract_of_the_players_table_only():
    third = "S:T7.AJ5.KQ962.AQ4 43.T82.J74.JT765 AQJ9652.64.T3.82 K8.KQ973.A85.K93"
    df = pl.DataFrame({
        "Board": [1, 1, 2, 2, 3, 3, 4],
        "PBN": [NORTH_FIRST, NORTH_FIRST, EAST_FIRST, EAST_FIRST, third, third, NORTH_FIRST],
        "Dealer": ["N", "N", "E", "E", "S", "S", "W"], "Vul": ["None", "None", "N_S", "N_S", "E_W", "E_W", "Both"],
        "Contract": ["4SXE", "3NN", "PASS", "2HS", "3NN", "3NS", "A+"],
        "Tricks": [9, 10, None, 8, 9, 9, None],
        # board 1: the player's table; 2: passed out there; 3: not played by the player; 4: an adjusted score
        "Boards_I_Played": [True, False, True, False, False, False, True],
    })
    asked = parse_qs(urlsplit(ai_auctions_url(df, "ffbridge", "E")).query, keep_blank_values=True)
    assert asked["played"] == ["4SXE-9.PASS.."]
    assert len(unpack(asked["boards"][0])) == 4

    two_tables = df.with_columns(pl.lit(True).alias("Boards_I_Played")).filter(pl.col("Board") == 1)
    assert "played" not in parse_qs(urlsplit(ai_auctions_url(two_tables, "pbn")).query)


def test_no_deals_is_an_error():
    df = pl.DataFrame({"Board": [1], "PBN": [None], "Dealer": ["N"], "Vul": ["None"]}, schema_overrides={"PBN": pl.String})
    with pytest.raises(ValueError, match="no deals"):
        ai_auctions_url(df, "pbn")
