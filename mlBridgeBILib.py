"""BridgeInterNet client for Octopus and Simultanet results.

Streamlit-free on purpose. Parses the public HTML tables at
http://www.bridgeinter.net for the three games that publish both Scratch and
Handicap percentages:

* Monday Octopus   series 386  ``loYYMMDD``  ``octopus_l/``
* Thursday Octopus series 386  ``joYYMMDD``  ``octopus_j/``
* Friday Simultanet series 384  ``viYYMMDD``  ``simultanet/``

Club percentages come from each club's ``restotal.php`` page (every pair at
that club, with a local rank). The national ``resseance*.php`` page is only
the top 100 and is a different column.

The Thursday history index (``SeancesPrecedantes_j.php``) currently 500s;
session URLs are still deterministic from the calendar date.
"""

from __future__ import annotations

import html
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import polars as pl
import requests
from tqdm import tqdm


BRIDGEINTER_BASE_URL = "http://www.bridgeinter.net/"
OCTOPUS_SERIES_ID = 386
SIMULTANET_SERIES_ID = 384
BI_SERIES_IDS = frozenset({OCTOPUS_SERIES_ID, SIMULTANET_SERIES_ID})

GetText = Callable[[str], str]


@dataclass(frozen=True)
class GameSpec:
    key: str
    series_id: int
    weekday: int
    prefix: str
    directory: str
    ranking_page: str
    history_page: str


GAMES: Dict[str, GameSpec] = {
    "octopus_monday": GameSpec(
        key="octopus_monday",
        series_id=OCTOPUS_SERIES_ID,
        weekday=0,
        prefix="lo",
        directory="octopus_l",
        ranking_page="resseance_l.php",
        history_page="SeancesPrecedantes_l.php",
    ),
    "octopus_thursday": GameSpec(
        key="octopus_thursday",
        series_id=OCTOPUS_SERIES_ID,
        weekday=3,
        prefix="jo",
        directory="octopus_j",
        ranking_page="resseance_j.php",
        history_page="SeancesPrecedantes_j.php",
    ),
    "simultanet": GameSpec(
        key="simultanet",
        series_id=SIMULTANET_SERIES_ID,
        weekday=4,
        prefix="vi",
        directory="simultanet",
        ranking_page="resseance.php",
        history_page="seancesprecedentes_vi.php",
    ),
}
GAMES_BY_WEEKDAY = {spec.weekday: spec for spec in GAMES.values()}
GAMES_BY_PREFIX = {spec.prefix: spec for spec in GAMES.values()}

_ROW_RE = re.compile(
    r"<tr\s+class=['\"]?text_res['\"]?[^>]*>(.*?)</tr>",
    flags=re.IGNORECASE | re.DOTALL,
)
_TD_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_PCT_RE = re.compile(r"(\d{1,2}\.\d{2})\s*%")
_PAIR_ID_RE = re.compile(r"v_numpaire=([^&\"'\s>]+)", flags=re.IGNORECASE)
_CLUB_LINK_RE = re.compile(
    r"""href\s*=\s*["']?([^"'\s>]*restotal\.php\?[^"'\s>]+)["']?[^>]*>([^<]+)""",
    flags=re.IGNORECASE,
)
_HISTORY_LINK_RE = re.compile(
    r"resseance(?:_[lj])?\.php\?v_codeseance=([a-z]{2}\d{6})&(?:amp;)?v_type_classement=([sh])",
    flags=re.IGNORECASE,
)
_SESSION_CODE_RE = re.compile(r"^([a-z]{2})(\d{6})$", flags=re.IGNORECASE)


def _parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", html.unescape(value or ""))
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_text.upper()).strip()


def game_for_date(value: str | date | datetime) -> Optional[GameSpec]:
    return GAMES_BY_WEEKDAY.get(_parse_date(value).weekday())


def session_code_for_date(value: str | date | datetime) -> Optional[str]:
    spec = game_for_date(value)
    if spec is None:
        return None
    return f"{spec.prefix}{_parse_date(value).strftime('%y%m%d')}"


def date_from_session_code(session_code: str) -> date:
    match = _SESSION_CODE_RE.match(session_code.strip())
    if match is None:
        raise ValueError(f"Unrecognized BridgeInterNet session code: {session_code!r}")
    prefix, yymmdd = match.group(1).lower(), match.group(2)
    if prefix not in GAMES_BY_PREFIX:
        raise ValueError(f"Unknown BridgeInterNet game prefix: {prefix!r}")
    parsed = datetime.strptime(yymmdd, "%y%m%d").date()
    spec = GAMES_BY_PREFIX[prefix]
    if parsed.weekday() != spec.weekday:
        raise ValueError(
            f"Session {session_code} is a {parsed:%A}, expected {spec.key}"
        )
    return parsed


def ranking_url(
    value: str | date | datetime,
    classement: str,
) -> Optional[str]:
    spec = game_for_date(value)
    code = session_code_for_date(value)
    score_code = _score_code(classement)
    if spec is None or code is None:
        return None
    return (
        f"{BRIDGEINTER_BASE_URL}{spec.directory}/{spec.ranking_page}"
        f"?v_codeseance={code}&v_type_classement={score_code}"
    )


def club_url(
    value: str | date | datetime,
    club_code: str,
    classement: str,
) -> Optional[str]:
    spec = game_for_date(value)
    code = session_code_for_date(value)
    if spec is None or code is None:
        return None
    return (
        f"{BRIDGEINTER_BASE_URL}{spec.directory}/restotal.php"
        f"?v_codeclub={club_code}&v_type_classement={_score_code(classement)}"
        f"&v_codeseance={code}"
    )


def history_url(game: str) -> str:
    spec = GAMES[game]
    return f"{BRIDGEINTER_BASE_URL}{spec.directory}/{spec.history_page}"


def dates_for_game(game: str, start: str | date, end: str | date) -> list[date]:
    spec = GAMES[game]
    cursor = _parse_date(start)
    last = _parse_date(end)
    if last < cursor:
        raise ValueError(f"end {last} is before start {cursor}")
    dates: list[date] = []
    while cursor <= last:
        if cursor.weekday() == spec.weekday:
            dates.append(cursor)
        cursor += timedelta(days=1)
    return dates


def _score_code(classement: str) -> str:
    normalized = str(classement).strip().lower()
    if normalized in {"s", "scratch"}:
        return "s"
    if normalized in {"h", "handicap"}:
        return "h"
    raise ValueError(f"classement must be scratch/s or handicap/h, got {classement!r}")


def _classement_name(score_code: str) -> str:
    return "scratch" if score_code == "s" else "handicap"


def default_get_text(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    encoding = response.encoding or "utf-8"
    return response.content.decode(encoding, errors="replace")


def _cell_text(td_html: str) -> str:
    return normalize_text(_TAG_RE.sub(" ", td_html))


def parse_club_links(page_html: str, page_url: str) -> list[dict[str, str]]:
    clubs: list[dict[str, str]] = []
    seen: set[str] = set()
    for relative_url, label in _CLUB_LINK_RE.findall(page_html):
        absolute = urljoin(page_url, html.unescape(relative_url))
        query = parse_qs(urlparse(absolute).query)
        club_code = (query.get("v_codeclub") or [""])[0].strip().upper()
        if not club_code or club_code in seen:
            continue
        seen.add(club_code)
        clubs.append(
            {
                "club_code": club_code,
                "club_name": html.unescape(label).strip(),
                "url": absolute,
            }
        )
    return clubs


def parse_history(page_html: str) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for session_code, score_code in _HISTORY_LINK_RE.findall(page_html):
        code = session_code.lower()
        ranking = score_code.lower()
        key = (code, ranking)
        if key in seen:
            continue
        seen.add(key)
        sessions.append(
            {
                "session_code": code,
                "date": date_from_session_code(code),
                "classement": _classement_name(ranking),
                "game": GAMES_BY_PREFIX[code[:2]].key,
            }
        )
    sessions.sort(key=lambda row: (row["date"], row["classement"]), reverse=True)
    return sessions


def parse_result_rows(
    page_html: str,
    *,
    page_kind: str,
    club_code: str = "",
    club_name: str = "",
) -> list[dict[str, Any]]:
    """Parse ``<tr class=text_res>`` ranking rows.

    ``page_kind`` is ``club`` (restotal.php: local, global, theo, direction,
    names, pct) or ``national`` (resseance: global, theo, direction, names,
    pct, club).
    """
    if page_kind not in {"club", "national"}:
        raise ValueError(f"page_kind must be club or national, got {page_kind!r}")

    rows: list[dict[str, Any]] = []
    for row_html in _ROW_RE.findall(page_html):
        cells = _TD_RE.findall(row_html)
        if len(cells) < 6:
            continue
        texts = [_cell_text(cell) for cell in cells]
        pct_index = next(
            (i for i, text in enumerate(texts) if _PCT_RE.search(text)),
            None,
        )
        if pct_index is None or pct_index < 2:
            continue
        pct_match = _PCT_RE.search(texts[pct_index])
        if pct_match is None:
            continue
        player1 = texts[pct_index - 2]
        player2 = texts[pct_index - 1]
        if not player1 or not player2:
            continue
        direction = ""
        direction_index = pct_index - 3
        if direction_index >= 0 and texts[direction_index] in {"NS", "EO", ""}:
            direction = texts[direction_index]

        local_rank = None
        global_rank = None
        theoretical_rank = None
        row_club_name = club_name
        if page_kind == "club":
            local_rank = _optional_int(texts[0] if texts else None)
            global_rank = _optional_int(texts[1] if len(texts) > 1 else None)
            theoretical_rank = _optional_int(texts[2] if len(texts) > 2 else None)
        else:
            global_rank = _optional_int(texts[0] if texts else None)
            theoretical_rank = _optional_int(texts[1] if len(texts) > 1 else None)
            if cells and cells[-1] and not _PCT_RE.search(_cell_text(cells[-1])):
                row_club_name = html.unescape(_TAG_RE.sub(" ", cells[-1])).strip()

        pair_id = None
        for cell in cells[max(0, pct_index - 2) : pct_index]:
            pair_match = _PAIR_ID_RE.search(cell)
            if pair_match:
                pair_id = pair_match.group(1)
                break

        rows.append(
            {
                "player1_name": html.unescape(_TAG_RE.sub("", cells[pct_index - 2])).strip(),
                "player2_name": html.unescape(_TAG_RE.sub("", cells[pct_index - 1])).strip(),
                "percentage": float(pct_match.group(1)),
                "local_rank": local_rank,
                "global_rank": global_rank,
                "theoretical_rank": theoretical_rank,
                "direction": direction,
                "club_code": club_code,
                "club_name": row_club_name,
                "pair_id": pair_id,
            }
        )
    return rows


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", value)
    if not digits:
        return None
    return int(digits)


def _surname_token(full_name: str) -> str:
    parts = normalize_text(full_name).split()
    return parts[-1] if parts else ""


def _lancelot_surnames(row: Dict[str, Any]) -> tuple[str, str]:
    team = row.get("team") if isinstance(row.get("team"), dict) else {}
    player1 = team.get("player1") if isinstance(team.get("player1"), dict) else {}
    player2 = team.get("player2") if isinstance(team.get("player2"), dict) else {}
    last1 = str(player1.get("lastName") or "")
    last2 = str(player2.get("lastName") or "")
    if last1 and last2:
        return normalize_text(last1), normalize_text(last2)
    return (
        _surname_token(str(row.get("player1_name") or "")),
        _surname_token(str(row.get("player2_name") or "")),
    )


def _pair_matches(player1_name: str, player2_name: str, surname1: str, surname2: str) -> bool:
    if not surname1 or not surname2:
        return False
    haystack = f"{normalize_text(player1_name)} {normalize_text(player2_name)}"
    if surname1 not in haystack or surname2 not in haystack:
        return False
    if surname1 == surname2:
        return haystack.count(surname1) >= 2
    return True


def match_unique_row(
    rows: Iterable[Dict[str, Any]],
    surname1: str,
    surname2: str,
) -> Optional[Dict[str, Any]]:
    matches = [
        row
        for row in rows
        if _pair_matches(
            str(row.get("player1_name") or ""),
            str(row.get("player2_name") or ""),
            surname1,
            surname2,
        )
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def fetch_session_tables(
    value: str | date | datetime,
    *,
    get_text: Optional[GetText] = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Download national + club Scratch/Handicap tables for one BI session."""
    day = _parse_date(value)
    spec = game_for_date(day)
    if spec is None:
        raise ValueError(f"{day.isoformat()} is not a BridgeInterNet game day")
    fetch_text = get_text or default_get_text
    scratch_url = ranking_url(day, "scratch")
    handicap_url = ranking_url(day, "handicap")
    assert scratch_url is not None and handicap_url is not None

    scratch_html = fetch_text(scratch_url)
    handicap_html = fetch_text(handicap_url)
    clubs = parse_club_links(scratch_html, scratch_url)
    if not clubs:
        clubs = parse_club_links(handicap_html, handicap_url)

    def _club_page(club: dict[str, str], classement: str) -> tuple[dict[str, str], str, str]:
        url = club_url(day, club["club_code"], classement)
        assert url is not None
        return club, classement, fetch_text(url)

    club_jobs = [(club, classement) for club in clubs for classement in ("scratch", "handicap")]
    club_pages: list[tuple[dict[str, str], str, str]] = []
    if club_jobs:
        with ThreadPoolExecutor(max_workers=min(8, len(club_jobs))) as executor:
            futures = [
                executor.submit(_club_page, club, classement)
                for club, classement in club_jobs
            ]
            iterator = futures
            if show_progress:
                iterator = tqdm(
                    futures,
                    desc=f"BI {spec.key} {day.isoformat()} clubs",
                    total=len(futures),
                )
            club_pages = [future.result() for future in iterator]

    national_scratch = parse_result_rows(scratch_html, page_kind="national")
    national_handicap = parse_result_rows(handicap_html, page_kind="national")
    club_scratch: list[dict[str, Any]] = []
    club_handicap: list[dict[str, Any]] = []
    for club, classement, page_html in club_pages:
        parsed = parse_result_rows(
            page_html,
            page_kind="club",
            club_code=club["club_code"],
            club_name=club["club_name"],
        )
        if classement == "scratch":
            club_scratch.extend(parsed)
        else:
            club_handicap.extend(parsed)

    return {
        "game": spec.key,
        "series_id": spec.series_id,
        "date": day.isoformat(),
        "session_code": session_code_for_date(day),
        "scratch_url": scratch_url,
        "handicap_url": handicap_url,
        "clubs": clubs,
        "national_scratch": national_scratch,
        "national_handicap": national_handicap,
        "club_scratch": club_scratch,
        "club_handicap": club_handicap,
    }


def match_ranking_to_session(
    ranking: Iterable[Dict[str, Any]],
    session: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Map Lancelot ranking rows onto BI national/club percentages by surname."""
    scores: Dict[str, Dict[str, Any]] = {}
    for row in ranking:
        if not isinstance(row, dict):
            continue
        team = row.get("team") if isinstance(row.get("team"), dict) else {}
        team_id = str(team.get("id") or row.get("team_id") or row.get("pair_id") or "")
        if not team_id:
            continue
        surname1, surname2 = _lancelot_surnames(row)
        national_scratch = match_unique_row(session["national_scratch"], surname1, surname2)
        national_handicap = match_unique_row(session["national_handicap"], surname1, surname2)
        club_scratch = match_unique_row(session["club_scratch"], surname1, surname2)
        club_handicap = match_unique_row(session["club_handicap"], surname1, surname2)
        scores[team_id] = {
            "scratch_percentage": _row_pct(club_scratch) or _row_pct(national_scratch),
            "handicap_percentage": _row_pct(club_handicap) or _row_pct(national_handicap),
            "national_scratch_percentage": _row_pct(national_scratch),
            "national_handicap_percentage": _row_pct(national_handicap),
            "club_scratch_percentage": _row_pct(club_scratch),
            "club_handicap_percentage": _row_pct(club_handicap),
            "club_scratch_rank": None if club_scratch is None else club_scratch.get("local_rank"),
            "club_handicap_rank": None if club_handicap is None else club_handicap.get("local_rank"),
            "theoretical_rank": (
                _row_rank(club_handicap)
                or _row_rank(club_scratch)
                or _row_rank(national_handicap)
                or _row_rank(national_scratch)
            ),
            "scratch_url": session.get("scratch_url"),
            "handicap_url": session.get("handicap_url"),
        }
    return scores


def _row_rank(row: Optional[Dict[str, Any]], field: str = "theoretical_rank") -> Optional[int]:
    if row is None:
        return None
    value = row.get(field)
    if value is None or value == "":
        return None
    return int(value)


def _row_pct(row: Optional[Dict[str, Any]]) -> Optional[float]:
    if row is None:
        return None
    value = row.get("percentage")
    return float(value) if value is not None else None


def fetch_session_pair_scores(
    ranking: Iterable[Dict[str, Any]],
    tournament_date: str,
    series_id: Optional[Any] = None,
    *,
    get_text: Optional[GetText] = None,
) -> Dict[str, Dict[str, Any]]:
    """Elo-facing helper: BI scores keyed by Lancelot team id."""
    try:
        normalized_series_id = int(series_id) if series_id is not None else None
    except (TypeError, ValueError):
        normalized_series_id = None
    if normalized_series_id not in BI_SERIES_IDS:
        return {}
    spec = game_for_date(tournament_date)
    if spec is None or spec.series_id != normalized_series_id:
        return {}
    try:
        session = fetch_session_tables(tournament_date, get_text=get_text)
    except requests.RequestException as exc:
        print(f"[bridgeinter] fetch failed for {tournament_date}: {exc}", flush=True)
        return {}
    return match_ranking_to_session(ranking, session)


def fill_missing_club_pcts(
    results_df: pl.DataFrame,
    *,
    get_text: Optional[GetText] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    series_ids: Iterable[int] = BI_SERIES_IDS,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Fill null club percentages and Theoretical_Rank from BridgeInterNet tables.

    Only rows whose date falls on a BI game day and whose series is 384 or 386
    are considered. Existing non-null values are left unchanged. Ambiguous
    surname matches stay null. Lancelot did not publish theoreticalRank before
    about 2026-07-01; organizer pages still have a Théo column for older dates.
    """
    required = {"date", "series_id", "player1_name", "player2_name"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {sorted(missing)}")
    for column in (
        "Club_Scratch_Pct",
        "Club_Handicap_Pct",
        "Club_Scratch_Rank",
        "Club_Handicap_Rank",
        "Theoretical_Rank",
    ):
        if column not in results_df.columns:
            dtype = pl.Float64 if column.endswith("_Pct") else pl.Int64
            results_df = results_df.with_columns(pl.lit(None).cast(dtype).alias(column))

    work = results_df.with_columns(
        pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("_bi_date"),
        pl.col("series_id").cast(pl.Int64, strict=False).alias("_bi_series"),
    )
    series_set = {int(value) for value in series_ids}
    candidates = work.filter(pl.col("_bi_series").is_in(list(series_set)))
    if date_from:
        candidates = candidates.filter(pl.col("_bi_date") >= date_from[:10])
    if date_to:
        candidates = candidates.filter(pl.col("_bi_date") <= date_to[:10])
    candidates = candidates.filter(
        pl.col("Club_Scratch_Pct").is_null()
        | pl.col("Club_Handicap_Pct").is_null()
        | pl.col("Theoretical_Rank").is_null()
    )
    dates = sorted({row["_bi_date"] for row in candidates.select("_bi_date").unique().to_dicts()})
    bi_dates = [day for day in dates if game_for_date(day) is not None]
    if not bi_dates:
        return results_df.drop([col for col in ("_bi_date", "_bi_series") if col in results_df.columns])

    started = datetime.now()
    print(f"[bridgeinter] backfill start {started.isoformat(timespec='seconds')} dates={len(bi_dates)}", flush=True)

    scratch_pct = results_df["Club_Scratch_Pct"].to_list()
    handicap_pct = results_df["Club_Handicap_Pct"].to_list()
    scratch_rank = results_df["Club_Scratch_Rank"].to_list()
    handicap_rank = results_df["Club_Handicap_Rank"].to_list()
    theoretical_rank = results_df["Theoretical_Rank"].to_list()
    has_lowercase_theo = "theoretical_rank" in results_df.columns
    lowercase_theo = (
        results_df["theoretical_rank"].to_list() if has_lowercase_theo else None
    )
    date_values = results_df["date"].cast(pl.Utf8).str.slice(0, 10).to_list()
    series_values = results_df["series_id"].to_list()
    p1_values = results_df["player1_name"].to_list()
    p2_values = results_df["player2_name"].to_list()

    date_iter = tqdm(bi_dates, desc="BI sessions") if show_progress else bi_dates
    filled = 0
    for day in date_iter:
        spec = game_for_date(day)
        assert spec is not None
        session = fetch_session_tables(day, get_text=get_text, show_progress=False)
        for index, (row_date, series_id, name1, name2) in enumerate(
            zip(date_values, series_values, p1_values, p2_values)
        ):
            if str(row_date)[:10] != day:
                continue
            try:
                row_series = int(series_id)
            except (TypeError, ValueError):
                continue
            if row_series != spec.series_id:
                continue
            surname1 = _surname_token(str(name1 or ""))
            surname2 = _surname_token(str(name2 or ""))
            club_scratch = match_unique_row(session["club_scratch"], surname1, surname2)
            club_handicap = match_unique_row(session["club_handicap"], surname1, surname2)
            national_scratch = match_unique_row(
                session["national_scratch"], surname1, surname2
            )
            national_handicap = match_unique_row(
                session["national_handicap"], surname1, surname2
            )
            if club_scratch is not None and scratch_pct[index] is None:
                scratch_pct[index] = club_scratch["percentage"]
                if scratch_rank[index] is None:
                    scratch_rank[index] = club_scratch.get("local_rank")
                filled += 1
            if club_handicap is not None and handicap_pct[index] is None:
                handicap_pct[index] = club_handicap["percentage"]
                if handicap_rank[index] is None:
                    handicap_rank[index] = club_handicap.get("local_rank")
                filled += 1
            if theoretical_rank[index] is None:
                theo = (
                    _row_rank(club_handicap)
                    or _row_rank(club_scratch)
                    or _row_rank(national_handicap)
                    or _row_rank(national_scratch)
                )
                if theo is not None:
                    theoretical_rank[index] = theo
                    if lowercase_theo is not None and lowercase_theo[index] is None:
                        lowercase_theo[index] = theo
                    filled += 1

    elapsed = (datetime.now() - started).total_seconds()
    print(
        f"[bridgeinter] backfill end {datetime.now().isoformat(timespec='seconds')} "
        f"filled_values={filled} elapsed={elapsed:.1f}s",
        flush=True,
    )
    updated = {
        "Club_Scratch_Pct": scratch_pct,
        "Club_Handicap_Pct": handicap_pct,
        "Club_Scratch_Rank": scratch_rank,
        "Club_Handicap_Rank": handicap_rank,
        "Theoretical_Rank": theoretical_rank,
    }
    if lowercase_theo is not None:
        updated["theoretical_rank"] = lowercase_theo
    return results_df.with_columns(
        [pl.Series(name, values) for name, values in updated.items()]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch one BridgeInterNet session")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    started = datetime.now()
    print(f"start {started.isoformat(timespec='seconds')}", flush=True)
    session = fetch_session_tables(args.date, show_progress=True)
    print(
        f"{session['game']} {session['session_code']} "
        f"clubs={len(session['clubs'])} "
        f"club_scratch={len(session['club_scratch'])} "
        f"club_handicap={len(session['club_handicap'])} "
        f"national_scratch={len(session['national_scratch'])}"
    )
    elapsed = (datetime.now() - started).total_seconds()
    if elapsed >= 30:
        print(f"elapsed {elapsed:.1f}s", flush=True)
