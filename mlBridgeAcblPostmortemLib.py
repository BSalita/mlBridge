"""Headless ACBL postmortem builders shared by APIs and user interfaces."""

from __future__ import annotations

from typing import Any, Dict, Optional

import polars as pl

from mlBridge.mlBridgeAcblLib import (
    merge_clean_augment_club_dfs,
    merge_clean_augment_tournament_dfs,
)
from mlBridge.mlBridgeAugmentLib import AllAugmentations


def augment_postmortem_dataframe(
    df: pl.DataFrame,
    *,
    single_dummy_sample_count: int = 40,
) -> pl.DataFrame:
    """Apply the full postmortem augmentation pipeline without Streamlit."""
    augmenter = AllAugmentations(
        df,
        None,
        sd_productions=single_dummy_sample_count,
        output_progress=False,
        progress=None,
        lock_func=None,
    )
    augmented, _ = augmenter.perform_all_augmentations()
    return augmented.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def build_club_postmortem(
    frames: Dict[str, pl.DataFrame],
    player_id: str,
    *,
    single_dummy_sample_count: int = 40,
) -> pl.DataFrame:
    """Build a fully augmented club postmortem from flat session frames."""
    merged = merge_clean_augment_club_dfs(frames, {}, str(player_id))
    if merged is None:
        raise ValueError("Club session could not be normalized")
    return augment_postmortem_dataframe(
        merged, single_dummy_sample_count=single_dummy_sample_count)


def tournament_section_for_player(
    session_data: Dict[str, Any], player_id: str
) -> Optional[Dict[str, Any]]:
    """Return the tournament section containing the requested player."""
    pid = str(player_id).strip()
    for section in session_data.get("sections") or []:
        for result in section.get("board_results") or []:
            if any(str(value).strip() == pid for value in result.get("pair_acbl") or []):
                return section
    return None


def build_tournament_postmortem(
    session_data: Dict[str, Any],
    player_id: str,
    *,
    single_dummy_sample_count: int = 40,
) -> pl.DataFrame:
    """Build a fully augmented tournament postmortem from official API JSON."""
    section = tournament_section_for_player(session_data, player_id)
    if section is None:
        raise ValueError(
            f"Player {player_id} was not found in the tournament session")
    frames = {
        "event": session_data.get("event") or {},
        "score_score_type": section.get("scoring_type"),
        "session": session_data,
        "section": section.get("section_label"),
    }
    merged = merge_clean_augment_tournament_dfs(
        frames, session_data, {}, str(player_id))
    if merged is None:
        raise ValueError("Tournament session could not be normalized")
    event = session_data.get("event") or {}
    tournament = session_data.get("tournament") or {}
    metadata = {
        "event_name": event.get("name"),
        "event_type": event.get("game_type"),
        "board_scoring_method": section.get("scoring_type"),
        "tournament_name": tournament.get("name"),
    }
    merged = merged.with_columns(
        pl.lit(value).alias(name)
        for name, value in metadata.items()
        if name not in merged.columns
    )
    return augment_postmortem_dataframe(
        merged, single_dummy_sample_count=single_dummy_sample_count)
