import pathlib
import tempfile
import unittest

import polars as pl

import mlBridge.mlBridgeFFIndexLib as indexlib


def _ranking_results() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tournament_id": ["300001", "300002"],
            "tournament_name": ["Session One", "Session Two"],
            "date": ["2025-01-02T14:00:00+01:00", "2025-02-03"],
            "series_id": [5, 604],
            "team_id": ["10", "20"],
            "club_id": ["A", "B"],
            "club_name": ["Club A", "Club B"],
            "player1_name": ["Guy Laumond", "Other Player"],
            "player2_name": ["Partner One", "Guy Laumond"],
            "player1_lancelot_id": ["136662", "999"],
            "player2_lancelot_id": ["111", "136662"],
            "player1_classic_person_id": ["322582", "888"],
            "player2_classic_person_id": ["101", "322582"],
            "player1_license_number": ["4958370", "9999999"],
            "player2_license_number": ["1111111", "4958370"],
        }
    )


class PlayerSessionIndexTests(unittest.TestCase):
    def test_builds_person_aliases_and_player_sessions(self):
        persons, sessions = indexlib.build_index_frames(_ranking_results())

        guy = indexlib.lookup_person(persons, "license:4958370")
        self.assertEqual(guy["lancelot_person_id"], "136662")
        self.assertEqual(guy["classic_person_id"], "322582")
        self.assertEqual(guy["display_name"], "Guy Laumond")

        history = indexlib.lookup_sessions(
            sessions,
            "136662",
            date_from="2025-02-01",
        )
        self.assertEqual(history["session_id"].to_list(), ["300002"])

    def test_every_identifier_namespace_resolves_same_person(self):
        persons, _ = indexlib.build_index_frames(_ranking_results())

        identifiers = [
            "136662",
            "lancelot:136662",
            "322582",
            "classic:322582",
            "migration:322582",
            "4958370",
            "license:4958370",
            "ffb:4958370",
        ]
        resolved = {
            indexlib.lookup_person(persons, identifier)["lancelot_person_id"]
            for identifier in identifiers
        }
        self.assertEqual(resolved, {"136662"})

    def test_bare_numeric_collision_requires_explicit_namespace(self):
        persons, _ = indexlib.build_index_frames(_ranking_results())
        collision = pl.DataFrame(
            {
                "lancelot_person_id": ["322582"],
                "classic_person_id": ["777"],
                "license_number": ["123"],
                "display_name": ["Collision"],
                "first_session_date": ["2025-01-01"],
                "last_session_date": ["2025-01-01"],
            }
        )
        persons = pl.concat([persons, collision])

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            indexlib.lookup_person(persons, "322582")
        self.assertEqual(
            indexlib.lookup_person(persons, "classic:322582")[
                "lancelot_person_id"
            ],
            "136662",
        )
        self.assertEqual(
            indexlib.lookup_person(persons, "lancelot:322582")[
                "lancelot_person_id"
            ],
            "322582",
        )

    def test_round_trip_validates_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            metadata = indexlib.build_and_write_index(
                _ranking_results(),
                index_dir=directory,
            )
            persons, sessions = indexlib.load_index(directory)

        self.assertEqual(metadata["persons_rows"], persons.height)
        self.assertEqual(metadata["player_session_rows"], sessions.height)


if __name__ == "__main__":
    unittest.main()
