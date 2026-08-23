"""Shared persisted Lancelot person and player-session index.

The index is deliberately independent of Elo calculations.  Its producer may
reuse the same public ranking downloads, but consumers depend only on this
canonical schema.
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import polars as pl


INDEX_SCHEMA_VERSION = 1
PERSONS_FILENAME = "lancelot_persons.parquet"
SESSIONS_FILENAME = "lancelot_player_sessions.parquet"
METADATA_FILENAME = "lancelot_player_session_index.meta.json"

IDENTIFIER_COLUMNS = {
    "lancelot": "lancelot_person_id",
    "person": "lancelot_person_id",
    "classic": "classic_person_id",
    "migration": "classic_person_id",
    "license": "license_number",
    "ffb": "license_number",
}

_REQUIRED_RESULT_COLUMNS = {
    "tournament_id",
    "tournament_name",
    "date",
    "series_id",
    "team_id",
    "club_id",
    "club_name",
    "player1_name",
    "player2_name",
    "player1_lancelot_id",
    "player2_lancelot_id",
    "player1_classic_person_id",
    "player2_classic_person_id",
    "player1_license_number",
    "player2_license_number",
}


def default_index_dir() -> pathlib.Path:
    configured = os.environ.get("FFBRIDGE_PLAYER_SESSION_INDEX_DIR", "").strip()
    if configured:
        return pathlib.Path(configured)
    cache_root = os.environ.get("FFBRIDGE_CACHE_DIR", "").strip()
    if cache_root:
        return pathlib.Path(cache_root) / "player_session_index"
    production_root = pathlib.Path("/data/ffbridge")
    if production_root.is_dir():
        return production_root / "player_session_index"
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "Elo_Ratings"
        / "data"
        / "ffbridge"
        / "player_session_index"
    )


def index_paths(
    index_dir: Optional[pathlib.Path] = None,
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    directory = pathlib.Path(index_dir) if index_dir is not None else default_index_dir()
    return (
        directory / PERSONS_FILENAME,
        directory / SESSIONS_FILENAME,
        directory / METADATA_FILENAME,
    )


def _clean_string(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String, strict=False).str.strip_chars()
    return pl.when(value == "").then(None).otherwise(value)


def _player_appearances(results_df: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(_REQUIRED_RESULT_COLUMNS - set(results_df.columns))
    if missing:
        raise ValueError(f"Lancelot ranking results lack index columns: {missing}")

    club_id = _clean_string("club_id")
    if "club_code" in results_df.columns:
        club_id = pl.coalesce(club_id, _clean_string("club_code"))
    common = [
        _clean_string("tournament_id").alias("session_id"),
        _clean_string("tournament_name").alias("session_label"),
        _clean_string("date").alias("raw_date"),
        _clean_string("date").str.slice(0, 10).alias("session_date"),
        pl.col("series_id").cast(pl.Int64, strict=False).alias("series_id"),
        _clean_string("team_id").alias("team_id"),
        club_id.alias("club_id"),
        _clean_string("club_name").alias("club_name"),
    ]
    frames = []
    for player_number in (1, 2):
        frames.append(
            results_df.select(
                *common,
                _clean_string(f"player{player_number}_lancelot_id").alias(
                    "lancelot_person_id"
                ),
                _clean_string(f"player{player_number}_classic_person_id").alias(
                    "classic_person_id"
                ),
                _clean_string(f"player{player_number}_license_number").alias(
                    "license_number"
                ),
                _clean_string(f"player{player_number}_name").alias("display_name"),
            )
        )
    appearances = pl.concat(frames, how="vertical")
    return appearances.filter(
        pl.col("lancelot_person_id").is_not_null()
        & pl.col("session_id").is_not_null()
    )


def build_index_frames(results_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Build canonical person-alias and player-session frames from rankings."""
    appearances = _player_appearances(results_df).sort(
        ["session_date", "session_id"], nulls_last=True
    )

    persons = (
        appearances.group_by("lancelot_person_id", maintain_order=True)
        .agg(
            pl.col("classic_person_id").drop_nulls().last(),
            pl.col("license_number").drop_nulls().last(),
            pl.col("display_name").drop_nulls().last(),
            pl.col("session_date").drop_nulls().min().alias("first_session_date"),
            pl.col("session_date").drop_nulls().max().alias("last_session_date"),
        )
        .sort("lancelot_person_id")
    )

    sessions = (
        appearances.unique(
            subset=["lancelot_person_id", "session_id"], keep="last", maintain_order=True
        )
        .select(
            "lancelot_person_id",
            "session_id",
            "team_id",
            "session_date",
            "raw_date",
            "series_id",
            "session_label",
            "club_id",
            "club_name",
        )
        .sort(
            ["lancelot_person_id", "session_date", "session_id"],
            descending=[False, True, True],
        )
    )
    return persons, sessions


def _atomic_write_parquet(frame: pl.DataFrame, path: pathlib.Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def write_index(
    persons: pl.DataFrame,
    sessions: pl.DataFrame,
    *,
    index_dir: Optional[pathlib.Path] = None,
) -> dict[str, Any]:
    persons_path, sessions_path, metadata_path = index_paths(index_dir)
    persons_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(persons, persons_path)
    _atomic_write_parquet(sessions, sessions_path)

    metadata = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "persons_rows": persons.height,
        "player_session_rows": sessions.height,
        "persons_file": persons_path.name,
        "sessions_file": sessions_path.name,
    }
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    return metadata


def build_and_write_index(
    results_df: pl.DataFrame,
    *,
    index_dir: Optional[pathlib.Path] = None,
) -> dict[str, Any]:
    persons, sessions = build_index_frames(results_df)
    return write_index(persons, sessions, index_dir=index_dir)


def validate_index(index_dir: Optional[pathlib.Path] = None) -> dict[str, Any]:
    persons_path, sessions_path, metadata_path = index_paths(index_dir)
    missing = [
        str(path)
        for path in (persons_path, sessions_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Lancelot player-session index is incomplete; missing: " + ", ".join(missing)
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Lancelot player-session index schema version "
            f"{metadata.get('schema_version')!r}; expected {INDEX_SCHEMA_VERSION}."
        )
    return metadata


def load_index(
    index_dir: Optional[pathlib.Path] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    validate_index(index_dir)
    persons_path, sessions_path, _ = index_paths(index_dir)
    return pl.read_parquet(persons_path), pl.read_parquet(sessions_path)


def load_persons(index_dir: Optional[pathlib.Path] = None) -> pl.DataFrame:
    validate_index(index_dir)
    persons_path, _, _ = index_paths(index_dir)
    return pl.read_parquet(persons_path)


def normalize_identifier(value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("player identifier is required")
    if normalized.isdigit():
        return normalized.lstrip("0") or "0"
    return normalized


def lookup_person(persons: pl.DataFrame, identifier: str) -> Optional[dict[str, Any]]:
    """Resolve a bare or ``kind:value`` identifier, failing on collisions."""
    raw = str(identifier or "").strip()
    if not raw:
        raise ValueError("player identifier is required")

    kind = None
    value = raw
    if ":" in raw:
        prefix, candidate = raw.split(":", 1)
        kind = prefix.strip().lower()
        if kind not in IDENTIFIER_COLUMNS:
            raise ValueError(
                f"Unknown player identifier type {prefix!r}; expected one of "
                f"{sorted(IDENTIFIER_COLUMNS)}."
            )
        value = candidate
    value = normalize_identifier(value)

    columns = (
        [IDENTIFIER_COLUMNS[kind]]
        if kind is not None
        else ["lancelot_person_id", "classic_person_id", "license_number"]
    )
    predicate = None
    for column in columns:
        candidate = _clean_string(column) == value
        predicate = candidate if predicate is None else predicate | candidate
    matches = persons.filter(predicate).unique(subset=["lancelot_person_id"])
    if matches.height == 0:
        return None
    if matches.height > 1:
        ids = matches["lancelot_person_id"].to_list()
        raise ValueError(
            f"Player identifier {identifier!r} is ambiguous across Lancelot IDs {ids}; "
            "use an explicit license:, classic:, or lancelot: prefix."
        )
    return matches.to_dicts()[0]


def lookup_sessions(
    sessions: pl.DataFrame,
    lancelot_person_id: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pl.DataFrame:
    query = sessions.lazy().filter(
        pl.col("lancelot_person_id").cast(pl.String)
        == normalize_identifier(lancelot_person_id)
    )
    if date_from:
        query = query.filter(pl.col("session_date") >= date_from)
    if date_to:
        query = query.filter(pl.col("session_date") <= date_to)
    return query.sort("session_date", descending=True).collect()


def query_index_sessions(
    lancelot_person_id: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    index_dir: Optional[pathlib.Path] = None,
) -> pl.DataFrame:
    _, sessions_path, _ = index_paths(index_dir)
    if not sessions_path.is_file():
        raise FileNotFoundError(
            f"Lancelot player-session index not found: {sessions_path}"
        )
    query = pl.scan_parquet(sessions_path).filter(
        pl.col("lancelot_person_id").cast(pl.String)
        == normalize_identifier(lancelot_person_id)
    )
    if date_from:
        query = query.filter(pl.col("session_date") >= date_from)
    if date_to:
        query = query.filter(pl.col("session_date") <= date_to)
    return query.sort("session_date", descending=True).collect()
