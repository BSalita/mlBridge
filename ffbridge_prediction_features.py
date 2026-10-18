"""FFBridge prediction features shared by training and the postmortem.

Hand shape for all four seats comes from the PBN deal. Player and partnership
history comes from earlier club boards: every lookup is as of the day before
the session, so a session's own boards never describe its players.

History tables hold running totals through each date. Seat ids must be in the
same namespace as the board results used to build them (Classic person ids).
"""

from __future__ import annotations

import numpy as np
import polars as pl

SEATS = ("N", "E", "S", "W")
SUITS = ("S", "H", "D", "C")
PAIRS = (("NS", "N", "S"), ("EW", "E", "W"))

SHAPE_FEATURES = tuple(
    [f"SL_{seat}_{suit}" for seat in SEATS for suit in SUITS]
    + [f"Fit_{pair}_{suit}" for pair, _, _ in PAIRS for suit in SUITS]
)
HISTORY_FEATURES = tuple(
    [f"Hist_{stat}_{seat}" for stat in ("Pct", "Decl_Pct", "Def_Pct", "Boards") for seat in SEATS]
    + [f"Hist_Partner_{stat}_{pair}" for stat in ("Sessions", "Pct") for pair, _, _ in PAIRS]
    + ["Hist_Pct_Gap"]
)

# Prior weight, in boards, pulling a short record toward 50%.
PLAYER_PRIOR_BOARDS = 60.0
DECLARER_PRIOR_BOARDS = 15.0
DEFENDER_PRIOR_BOARDS = 30.0
PARTNER_PRIOR_BOARDS = 60.0

PLAYER_TOTALS = ("boards", "side_sum", "decl_n", "decl_sum", "def_n", "def_sum")
PAIR_TOTALS = ("sessions", "boards", "side_sum")


def id_token(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8, strict=False).str.strip_chars().str.replace(r"\.0$", "")


def pair_key(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    a, b = id_token(left), id_token(right)
    return (
        pl.when(a.is_null() | b.is_null() | (a == "") | (b == ""))
        .then(None)
        .when(a <= b)
        .then(a + pl.lit("|") + b)
        .otherwise(b + pl.lit("|") + a)
    )


def shape_features(frame: pl.DataFrame, pbn: str = "PBN") -> pl.DataFrame:
    """Suit lengths for every seat and each side's combined length per suit."""
    deal = pl.col(pbn).cast(pl.Utf8)
    bad = frame.filter(~deal.str.starts_with("N:")).height
    if bad:
        raise ValueError(f"{bad} PBN deals do not start with 'N:'")
    hands = deal.str.slice(2).str.split(" ")
    exprs = []
    for seat_index, seat in enumerate(SEATS):
        suits = hands.list.get(seat_index).str.split(".")
        for suit_index, suit in enumerate(SUITS):
            exprs.append(
                suits.list.get(suit_index).str.len_chars().cast(pl.Float32).alias(f"SL_{seat}_{suit}")
            )
    out = frame.with_columns(exprs)
    return out.with_columns(
        (pl.col(f"SL_{left}_{suit}") + pl.col(f"SL_{right}_{suit}")).alias(f"Fit_{pair}_{suit}")
        for pair, left, right in PAIRS
        for suit in SUITS
    )


def _side_pct_ns(boards: pl.DataFrame) -> pl.Expr:
    direction = pl.col("Declarer_Direction").cast(pl.Utf8)
    pct = pl.col("Declarer_Pct").cast(pl.Float64, strict=False)
    return (
        pl.when(direction.is_in(["N", "S"])).then(pct)
        .when(direction.is_in(["E", "W"])).then(1.0 - pct)
        .otherwise(None)
    )


def build_history_tables(boards: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Running per-player and per-partnership totals through each date.

    ``boards`` needs Date, session_id, Declarer_Direction, Declarer_Pct and
    Player_ID_N/E/S/W.
    """
    base = boards.select(
        pl.col("Date").cast(pl.Date),
        pl.col("session_id").cast(pl.Utf8),
        pl.col("Declarer_Direction").cast(pl.Utf8),
        pl.col("Declarer_Pct").cast(pl.Float64, strict=False),
        *[id_token(pl.col(f"Player_ID_{seat}")).alias(seat) for seat in SEATS],
    ).filter(pl.col("Declarer_Pct").is_not_null() & pl.col("Declarer_Pct").is_finite())
    base = base.with_columns(_side_pct_ns(base).alias("pct_ns")).filter(pl.col("pct_ns").is_not_null())

    seat_rows = []
    for seat in SEATS:
        ns = seat in ("N", "S")
        declarer_side = ["N", "S"] if ns else ["E", "W"]
        is_declarer = pl.col("Declarer_Direction") == seat
        is_defender = ~pl.col("Declarer_Direction").is_in(declarer_side)
        seat_rows.append(
            base.select(
                "Date",
                pl.col(seat).alias("player_id"),
                (pl.col("pct_ns") if ns else 1.0 - pl.col("pct_ns")).alias("side"),
                is_declarer.cast(pl.Float64).alias("decl"),
                pl.when(is_declarer).then(pl.col("Declarer_Pct")).otherwise(0.0).alias("decl_pct"),
                is_defender.cast(pl.Float64).alias("def"),
                pl.when(is_defender).then(1.0 - pl.col("Declarer_Pct")).otherwise(0.0).alias("def_pct"),
            )
        )
    players = (
        pl.concat(seat_rows)
        .filter(pl.col("player_id").is_not_null() & (pl.col("player_id") != ""))
        .group_by("player_id", "Date")
        .agg(
            pl.len().cast(pl.Float64).alias("boards"),
            pl.col("side").sum().alias("side_sum"),
            pl.col("decl").sum().alias("decl_n"),
            pl.col("decl_pct").sum().alias("decl_sum"),
            pl.col("def").sum().alias("def_n"),
            pl.col("def_pct").sum().alias("def_sum"),
        )
        .sort("player_id", "Date")
        .with_columns(pl.col(name).cum_sum().over("player_id") for name in PLAYER_TOTALS)
        .collect()
    )

    pair_rows = []
    for _, left, right in PAIRS:
        ns = left == "N"
        pair_rows.append(
            base.select(
                "Date",
                "session_id",
                pair_key(pl.col(left), pl.col(right)).alias("pair_key"),
                (pl.col("pct_ns") if ns else 1.0 - pl.col("pct_ns")).alias("side"),
            )
        )
    pairs = (
        pl.concat(pair_rows)
        .filter(pl.col("pair_key").is_not_null())
        .group_by("pair_key", "Date")
        .agg(
            pl.col("session_id").n_unique().cast(pl.Float64).alias("sessions"),
            pl.len().cast(pl.Float64).alias("boards"),
            pl.col("side").sum().alias("side_sum"),
        )
        .sort("pair_key", "Date")
        .with_columns(pl.col(name).cum_sum().over("pair_key") for name in PAIR_TOTALS)
        .collect()
    )
    return players, pairs


def _shrunk(total: pl.Expr, count: pl.Expr, prior: float) -> pl.Expr:
    return (total.fill_null(0.0) + prior * 0.5) / (count.fill_null(0.0) + prior)


def attach_history(
    frame: pl.DataFrame,
    players: pl.DataFrame,
    pairs: pl.DataFrame,
    seat_ids: dict[str, str],
    date: str = "Date",
) -> pl.DataFrame:
    """Add HISTORY_FEATURES from totals strictly before each row's date.

    ``seat_ids`` maps N/E/S/W to the frame columns holding ids in the history
    namespace. Players with no earlier boards get the 50% prior.
    """
    session_date = pl.col(date)
    if frame.schema[date] in (pl.Utf8, pl.String):
        session_date = session_date.str.to_date(strict=True)
    out = frame.with_row_index("_hist_row").with_columns(session_date.cast(pl.Date).alias("_hist_date"))
    keys = out.select("_hist_row", "_hist_date", *[id_token(pl.col(col)).alias(f"_id_{seat}") for seat, col in seat_ids.items()])
    history = players.rename({"Date": "_hist_date"}).sort("_hist_date")

    features = keys.select("_hist_row")
    for seat in SEATS:
        looked_up = (
            keys.select("_hist_row", "_hist_date", pl.col(f"_id_{seat}").alias("player_id"))
            .sort("_hist_date")
            .join_asof(
                history, on="_hist_date", by="player_id", strategy="backward",
                allow_exact_matches=False, check_sortedness=False,
            )
            .select(
                "_hist_row",
                _shrunk(pl.col("side_sum"), pl.col("boards"), PLAYER_PRIOR_BOARDS).alias(f"Hist_Pct_{seat}"),
                _shrunk(pl.col("decl_sum"), pl.col("decl_n"), DECLARER_PRIOR_BOARDS).alias(f"Hist_Decl_Pct_{seat}"),
                _shrunk(pl.col("def_sum"), pl.col("def_n"), DEFENDER_PRIOR_BOARDS).alias(f"Hist_Def_Pct_{seat}"),
                pl.col("boards").fill_null(0.0).log1p().alias(f"Hist_Boards_{seat}"),
            )
        )
        features = features.join(looked_up, on="_hist_row", how="left")

    pair_history = pairs.rename({"Date": "_hist_date"}).sort("_hist_date")
    for pair, left, right in PAIRS:
        looked_up = (
            keys.select(
                "_hist_row",
                "_hist_date",
                pair_key(pl.col(f"_id_{left}"), pl.col(f"_id_{right}")).alias("pair_key"),
            )
            .sort("_hist_date")
            .join_asof(
                pair_history, on="_hist_date", by="pair_key", strategy="backward",
                allow_exact_matches=False, check_sortedness=False,
            )
            .select(
                "_hist_row",
                pl.col("sessions").fill_null(0.0).log1p().alias(f"Hist_Partner_Sessions_{pair}"),
                _shrunk(pl.col("side_sum"), pl.col("boards"), PARTNER_PRIOR_BOARDS).alias(f"Hist_Partner_Pct_{pair}"),
            )
        )
        features = features.join(looked_up, on="_hist_row", how="left")

    features = features.with_columns(
        (
            (pl.col("Hist_Pct_N") + pl.col("Hist_Pct_S")) / 2.0
            - (pl.col("Hist_Pct_E") + pl.col("Hist_Pct_W")) / 2.0
        ).alias("Hist_Pct_Gap")
    )
    out = out.join(features, on="_hist_row", how="left").sort("_hist_row")
    if out.height != frame.height:
        raise RuntimeError("History join changed the row count")
    return out.drop("_hist_row", "_hist_date").with_columns(
        pl.col(name).cast(pl.Float32) for name in HISTORY_FEATURES
    )


def contract_declarer(contract: str) -> str:
    return "NULL" if contract == "PASS" else contract[-1]


def joint_contract_and_declarer(
    contract_classes: list[str],
    contract_probs: np.ndarray,
    direction_classes: list[str],
    direction_probs: np.ndarray,
) -> tuple[list[str], list[str]]:
    """One contract and its declarer from both models' probabilities.

    Each contract is scored by its own probability times the declarer model's
    probability for that contract's declarer (``NULL`` for a pass-out).
    Probability arrays are rows x classes in schema class order.
    """
    contract_probs = np.asarray(contract_probs, dtype=np.float64)
    direction_probs = np.asarray(direction_probs, dtype=np.float64)
    direction_index = {name: index for index, name in enumerate(direction_classes)}
    missing = sorted({contract_declarer(name) for name in contract_classes} - set(direction_index))
    if missing:
        raise ValueError(f"Declarer model has no class for {missing}")
    seat = np.array([direction_index[contract_declarer(name)] for name in contract_classes])
    with np.errstate(divide="ignore"):
        score = np.log(contract_probs) + np.log(direction_probs[:, seat])
    best = score.argmax(axis=1)
    contracts = [contract_classes[index] for index in best]
    return contracts, [contract_declarer(name) for name in contracts]
