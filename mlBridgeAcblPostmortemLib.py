"""Headless ACBL postmortem builders shared by APIs and user interfaces."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any, Dict, Optional

import polars as pl

from mlBridge.mlBridgeAcblLib import (
    merge_clean_augment_club_dfs,
    merge_clean_augment_tournament_dfs,
)
from mlBridge.mlBridgeAugmentLib import AllAugmentations

logger = logging.getLogger(__name__)

# The ACBL pipeline (acbl_hand_records_augment.py, ACBL_SUITC=1) adds the SC_*
# SuitC single-suit columns to every deal, and step 5a lists them as model
# inputs, so live postmortems add them too. The cache is the shared one
# (SUITC_CACHE_DIR; /data/suitc/combos in the postmortem container); SuitC
# runs only for holdings it lacks. ACBL_POSTMORTEM_SUITC=0 turns this off.
_suitc_cache: Any = None
_suitc_checked = False
_suitc_init_lock = threading.Lock()
# SuitCCache is not thread-safe: augmentations that use it run one at a time.
_suitc_augment_lock = threading.Lock()


def postmortem_suitc_cache() -> Any:
    """The process's shared mlBridgeSuitCLib.SuitCCache, or None when SuitC is off.

    Off when ACBL_POSTMORTEM_SUITC is 0/false/off, or when the SuitC program is
    missing (holdings absent from the cache would be stored as errors).
    """
    global _suitc_cache, _suitc_checked
    with _suitc_init_lock:
        if not _suitc_checked:
            _suitc_checked = True
            if os.environ.get("ACBL_POSTMORTEM_SUITC", "1").strip().lower() in ("0", "false", "off", "no"):
                logger.info("SuitC columns off (ACBL_POSTMORTEM_SUITC)")
            else:
                from mlBridge import mlBridgeSuitCLib
                try:
                    exe = mlBridgeSuitCLib.suitc_exe()
                except FileNotFoundError as exc:
                    logger.warning(f"SuitC columns off: {exc}")
                else:
                    _suitc_cache = mlBridgeSuitCLib.SuitCCache()
                    if not _suitc_cache.directory.is_dir():
                        logger.warning(
                            f"SuitC cache directory not found: {_suitc_cache.directory}; "
                            "every holding runs SuitC until it fills")
                    logger.info(f"SuitC columns on: {exe}, cache {_suitc_cache.directory}")
        return _suitc_cache


def augment_postmortem_dataframe(
    df: pl.DataFrame,
    *,
    single_dummy_sample_count: int = 40,
    suitc: Any = "auto",
    output_progress: bool = False,
    progress: Any = None,
    lock_func: Any = None,
) -> pl.DataFrame:
    """Apply the full postmortem augmentation pipeline without Streamlit.

    suitc: a mlBridgeSuitCLib.SuitCCache, None for no SC_* columns, or "auto"
    for postmortem_suitc_cache().
    """
    if isinstance(suitc, str) and suitc == "auto":
        suitc = postmortem_suitc_cache()
    augmenter = AllAugmentations(
        df,
        None,
        sd_productions=single_dummy_sample_count,
        output_progress=output_progress,
        progress=progress,
        lock_func=lock_func,
        suitc=suitc,
    )
    with (_suitc_augment_lock if suitc is not None else contextlib.nullcontext()):
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
