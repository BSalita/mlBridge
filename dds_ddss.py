"""
dds_ddss.py

Thin ctypes wrapper around the ddss fork's CalcAllTablesPBNx API.
Provides a drop-in replacement for endplay's calc_all_tables that supports
arbitrary batch sizes (the library chunks internally at 1000 deals).

If the ddss DLL is not found, DDSS_AVAILABLE is set to False and the module
can be imported without error -- callers should check DDSS_AVAILABLE before
calling any functions.
"""

import ctypes
import os
import pathlib
import sys
import threading
from typing import List, Optional

import endplay._dds as endplay_dds

# TODO: how can ddss dll be made available? Does it ddss need to be made available as a package like endplay?
_WIN_DLL_PATH = pathlib.Path(r"C:\sw\bridge\ML-Contract-Bridge\src\ddss\build-cmake\Release\dds.dll")
_LINUX_SO_PATH = pathlib.Path("/usr/local/lib/libdds.so")

DDSS_AVAILABLE = False
_dll = None
_DLL_PATH = None


def _dll_candidates() -> List[pathlib.Path]:
    """Search order: DDSS_DLL_PATH, then the platform install / build paths."""
    paths: List[pathlib.Path] = []
    env = os.environ.get("DDSS_DLL_PATH")
    if env:
        paths.append(pathlib.Path(env))
    if sys.platform == "win32":
        paths.append(_WIN_DLL_PATH)
    else:
        paths.append(_LINUX_SO_PATH)
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        paths.append(repo_root / "ddss" / "dist" / "linux-x86_64" / "libdds.so")
        paths.append(repo_root / "ddss" / "build-linux" / "libdds.so")
    # Preserve order, drop duplicates.
    seen = set()
    unique: List[pathlib.Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


class ddTableDealPBN(ctypes.Structure):
    _fields_ = [("cards", ctypes.c_char * 80)]


class _DdssDDSInfo(ctypes.Structure):
    """ctypes layout for ddss GetDDSInfo (include/dll.h)."""

    _fields_ = [
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("patch", ctypes.c_int),
        ("versionString", ctypes.c_char * 10),
        ("system", ctypes.c_int),
        ("numBits", ctypes.c_int),
        ("compiler", ctypes.c_int),
        ("constructor", ctypes.c_int),
        ("numCores", ctypes.c_int),
        ("threading", ctypes.c_int),
        ("noOfThreads", ctypes.c_int),
        ("threadSizes", ctypes.c_char * 128),
        ("systemString", ctypes.c_char * 1024),
    ]


class ddTableResults(ctypes.Structure):
    _fields_ = [("resTable", (ctypes.c_int * 4) * 5)]  # [strain][hand]


class parResults(ctypes.Structure):
    _fields_ = [
        ("parScore", (ctypes.c_char * 16) * 2),
        ("parContractsString", (ctypes.c_char * 128) * 2),
    ]


# SolveAllBoards structs (include/dll.h). ddss raises MAXNOOFBOARDS to
# MAXNOOFTABLES * DDS_STRAINS = 5000, so these are not upstream-dds sized.
MAXNOOFBOARDS = 5000


class dealPBN(ctypes.Structure):
    _fields_ = [
        ("trump", ctypes.c_int),
        ("first", ctypes.c_int),
        ("currentTrickSuit", ctypes.c_int * 3),
        ("currentTrickRank", ctypes.c_int * 3),
        ("remainCards", ctypes.c_char * 80),
    ]


class boardsPBN(ctypes.Structure):
    _fields_ = [
        ("noOfBoards", ctypes.c_int),
        ("deals", dealPBN * MAXNOOFBOARDS),
        ("target", ctypes.c_int * MAXNOOFBOARDS),
        ("solutions", ctypes.c_int * MAXNOOFBOARDS),
        ("mode", ctypes.c_int * MAXNOOFBOARDS),
    ]


class futureTricks(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_int),
        ("cards", ctypes.c_int),
        ("suit", ctypes.c_int * 13),
        ("rank", ctypes.c_int * 13),
        ("equals", ctypes.c_int * 13),
        ("score", ctypes.c_int * 13),
    ]


class solvedBoards(ctypes.Structure):
    _fields_ = [
        ("noOfBoards", ctypes.c_int),
        ("solvedBoard", futureTricks * MAXNOOFBOARDS),
    ]


class DDSSResult:
    """Mimics endplay's DDTable interface so downstream code works unchanged.

    resTable layout: [strain][hand] where
        strain: 0=S, 1=H, 2=D, 3=C, 4=N
        hand:   0=N, 1=E, 2=S, 3=W
    """

    def __init__(self, res: ddTableResults):
        self._res = res
        # Some endplay helpers such as `par()` reach into DDTable._data directly
        # instead of using only the public indexing API. Mirror the ctypes result
        # into a real endplay ddTableResults object so those callers keep working.
        self._data = endplay_dds.ddTableResults()
        for strain in range(5):
            for hand in range(4):
                self._data.resTable[strain][hand] = res.resTable[strain][hand]

    def __getitem__(self, cell):
        strain, hand = cell
        return self._data.resTable[int(strain)][int(hand)]

    def to_list(self, player_major: bool = False) -> list:
        if player_major:
            return [
                [self._data.resTable[d][p] for d in range(5)]
                for p in range(4)
            ]
        return [
            [self._data.resTable[d][p] for p in range(4)]
            for d in range(5)
        ]

    def pprint(self, stream=sys.stdout) -> None:
        """Print the double-dummy table in endplay's grid format."""
        denoms = "CDHSN"
        players = "NSEW"
        print("   ", " ".join(denom.rjust(2) for denom in denoms), file=stream)
        for player in players:
            print(player.rjust(3), end="", file=stream)
            for denom in denoms:
                d = denoms.index(denom)
                p = players.index(player)
                print(str(self._data.resTable[d][p]).rjust(3), end="", file=stream)
            print(file=stream)

    def __str__(self) -> str:
        denoms = "CDHSN"
        players = "NSEW"
        return (
            ",".join(denoms)
            + ";"
            + ";".join(
                player + ":" + ",".join(str(self._data.resTable[d][p]) for d in range(5))
                for p, player in enumerate(players)
            )
        )


def _load_dll() -> bool:
    global _dll, DDSS_AVAILABLE, _DLL_PATH
    candidates = _dll_candidates()
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        tried = ", ".join(str(path) for path in candidates)
        print(f"ddss DLL not found (tried {tried}). Falling back to endplay DDS.")
        return False
    last_error = None
    for path in existing:
        try:
            if sys.platform == "win32":
                loaded = ctypes.WinDLL(str(path))
            else:
                loaded = ctypes.CDLL(str(path))

            loaded.CalcAllTablesPBNx.restype = ctypes.c_int
            loaded.CalcAllTablesPBNx.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ddTableDealPBN),
                ctypes.c_int,
                ctypes.c_int * 5,
                ctypes.POINTER(ddTableResults),
                ctypes.POINTER(parResults),
            ]

            loaded.SolveAllBoards.restype = ctypes.c_int
            loaded.SolveAllBoards.argtypes = [
                ctypes.POINTER(boardsPBN),
                ctypes.POINTER(solvedBoards),
            ]

            loaded.SetMaxThreads.restype = None
            loaded.SetMaxThreads.argtypes = [ctypes.c_int]
            # 0 = every core. Multi-process callers set DDSS_MAX_THREADS so
            # their workers don't each claim every core.
            loaded.SetMaxThreads(int(os.environ.get("DDSS_MAX_THREADS", "0")))

            loaded.ErrorMessage.restype = None
            loaded.ErrorMessage.argtypes = [ctypes.c_int, ctypes.c_char_p]

            _dll = loaded
            _DLL_PATH = path
            DDSS_AVAILABLE = True
            print(f"Loaded ddss DLL from {path}")
            return True
        except OSError as e:
            last_error = e
    print(f"Failed to load ddss DLL: {last_error}. Falling back to endplay DDS.")
    return False


_load_dll()


def _dds_version_from_get_info(dll, info_cls) -> Optional[str]:
    """Read DDS versionString via GetDDSInfo from a loaded dds/ddss DLL."""
    try:
        get_info = dll.GetDDSInfo
        get_info.argtypes = [ctypes.POINTER(info_cls)]
        get_info.restype = None
        info = info_cls()
        get_info(ctypes.byref(info))
        version = info.versionString.decode("ascii", errors="replace").strip("\x00").strip()
        if version:
            return version
        return f"{info.major}.{info.minor}.{info.patch}"
    except Exception:
        return None


def _ddss_dds_version() -> Optional[str]:
    if not DDSS_AVAILABLE or _dll is None:
        return None
    return _dds_version_from_get_info(_dll, _DdssDDSInfo)


_BSC_PREFIX = "bridge-solver-cpp dds-api"


def _loaded_library() -> tuple[str, Optional[str]]:
    """Which DDS-compatible library the loaded DLL is, and its revision.

    bridge-solver-cpp's library (BSalita/bridge-solver, dds-api) has ddss's API
    and version fields, so it is told apart by the first line of GetDDSInfo's
    systemString: "bridge-solver-cpp dds-api <revision>". Anything else is ddss.
    """
    try:
        get_info = _dll.GetDDSInfo
        get_info.argtypes = [ctypes.POINTER(_DdssDDSInfo)]
        get_info.restype = None
        info = _DdssDDSInfo()
        get_info(ctypes.byref(info))
        first = info.systemString.decode("ascii", errors="replace").split("\n", 1)[0].strip()
    except Exception:
        return "ddss", None
    if first.startswith(_BSC_PREFIX):
        revision = first[len(_BSC_PREFIX):].strip() or None
        return "bridge-solver-cpp", revision
    return "ddss", None


def _endplay_dds_version() -> Optional[str]:
    try:
        import endplay._dds as ep_dds
    except Exception:
        return None
    return _dds_version_from_get_info(ep_dds._dll, ep_dds.DDSInfo)


def engine_info() -> dict:
    """Return the live DD engine used by solve_dd_for_deals()."""
    if DDSS_AVAILABLE and _DLL_PATH is not None:
        library, revision = _loaded_library()
        return {
            # "ddss" for either library: both are loaded and called the same way.
            "engine": "ddss",
            "library": library,  # "ddss" or "bridge-solver-cpp"
            "revision": revision,
            "path": str(_DLL_PATH),
            "dds_version": _ddss_dds_version(),
        }
    endplay_version = None
    try:
        import endplay

        endplay_version = endplay.__version__
    except Exception:
        pass
    return {
        "engine": "endplay",
        "path": None,
        "dds_version": _endplay_dds_version(),
        "endplay_version": endplay_version,
    }


def engine_caption() -> str:
    """One-line label: which DD library is loaded and its version."""
    info = engine_info()
    dds_version = info.get("dds_version") or "?"
    if info["engine"] == "ddss":
        if info.get("library") == "bridge-solver-cpp":
            return f"DDS: bridge-solver-cpp dds-api {info.get('revision') or '?'} (DDS {dds_version} API)"
        return f"DDS: ddss {dds_version}"
    endplay_version = info.get("endplay_version") or "?"
    return f"DDS: endplay dds {dds_version} (endplay {endplay_version})"


def calc_all_tables_pbnx(
    pbn_strings: List[str],
    mode: int = -1,
    trump_filter: Optional[List[int]] = None,
) -> List[DDSSResult]:
    """Compute double-dummy tables for a list of PBN deal strings.

    Parameters:
        pbn_strings: PBN deal strings (e.g. "N:AK.QJ.T9.8765 ...").
        mode: Par vulnerability mode. -1 = no par, 0-3 = vulnerability.
        trump_filter: 5-element list; 0 = solve strain, 1 = skip.
                      Default solves all strains.

    Returns:
        List of DDSSResult objects (same length as pbn_strings).

    Raises:
        RuntimeError: If DDSS_AVAILABLE is False or the DLL returns an error.
        ValueError: If pbn_strings has negative length (invalid).
    """
    if not DDSS_AVAILABLE:
        raise RuntimeError("ddss DLL is not available")

    n = len(pbn_strings)
    if n == 0:
        return []

    if trump_filter is None:
        trump_filter = [0, 0, 0, 0, 0]

    DealArray = ddTableDealPBN * n
    ResultArray = ddTableResults * n

    deals = DealArray()
    results = ResultArray()
    tf = (ctypes.c_int * 5)(*trump_filter)

    for i, pbn in enumerate(pbn_strings):
        deals[i].cards = pbn.encode("ascii") if isinstance(pbn, str) else pbn

    par_array = None
    par_ptr = None
    if mode >= 0:
        ParArray = parResults * n
        par_array = ParArray()
        par_ptr = ctypes.cast(par_array, ctypes.POINTER(parResults))

    ret = _dll.CalcAllTablesPBNx(
        n,
        deals,
        mode,
        tf,
        results,
        par_ptr,
    )

    if ret != 1:
        buf = ctypes.create_string_buffer(80)
        _dll.ErrorMessage(ret, buf)
        raise RuntimeError(f"ddss CalcAllTablesPBNx error {ret}: {buf.value.decode()}")

    return [DDSSResult(results[i]) for i in range(n)]


_STRAIN_INDEX = {"S": 0, "H": 1, "D": 2, "C": 3, "N": 4}
_SEAT_INDEX = {"N": 0, "E": 1, "S": 2, "W": 3}
_SUIT_CHARS = "SHDC"
_RANK_CHARS = "--23456789TJQKA"  # DDS ranks are 2..14


def solve_opening_leads(
    positions: List[tuple],
    chunk_size: int = MAXNOOFBOARDS,
) -> List[dict]:
    """Double-dummy tricks for every opening lead.

    Parameters:
        positions: (pbn, strain, leader) tuples. pbn is a full "N:..." deal,
                   strain is one of SHDCN, leader is the seat on lead (NESW).
        chunk_size: boards per SolveAllBoards call (at most MAXNOOFBOARDS).

    Returns:
        One dict per position, mapping each card the leader holds ("SA",
        "HT", ...) to the tricks the defending side takes after that lead.

    Raises:
        RuntimeError: If ddss is not loaded or SolveAllBoards fails.
        ValueError: On a bad strain, leader or chunk size.
    """
    if not DDSS_AVAILABLE:
        raise RuntimeError("ddss DLL is not available")
    if not 1 <= chunk_size <= MAXNOOFBOARDS:
        raise ValueError(f"chunk_size must be 1..{MAXNOOFBOARDS}, got {chunk_size}")

    out: List[dict] = []
    bo = boardsPBN()
    solved = solvedBoards()
    for start in range(0, len(positions), chunk_size):
        chunk = positions[start : start + chunk_size]
        bo.noOfBoards = len(chunk)
        for i, (pbn, strain, leader) in enumerate(chunk):
            if strain not in _STRAIN_INDEX:
                raise ValueError(f"bad strain {strain!r} for {pbn}")
            if leader not in _SEAT_INDEX:
                raise ValueError(f"bad leader {leader!r} for {pbn}")
            d = bo.deals[i]
            d.trump = _STRAIN_INDEX[strain]
            d.first = _SEAT_INDEX[leader]
            for j in range(3):
                d.currentTrickSuit[j] = 0
                d.currentTrickRank[j] = 0
            d.remainCards = pbn.encode("ascii")
            bo.target[i] = -1  # find the maximum
            bo.solutions[i] = 3  # score every legal card
            bo.mode[i] = 1
        ret = _dll.SolveAllBoards(ctypes.byref(bo), ctypes.byref(solved))
        if ret != 1:
            buf = ctypes.create_string_buffer(80)
            _dll.ErrorMessage(ret, buf)
            raise RuntimeError(f"ddss SolveAllBoards error {ret}: {buf.value.decode()}")
        for i in range(len(chunk)):
            fut = solved.solvedBoard[i]
            tricks: dict = {}
            for c in range(fut.cards):
                suit = _SUIT_CHARS[fut.suit[c]]
                score = fut.score[c]
                tricks[suit + _RANK_CHARS[fut.rank[c]]] = score
                # Lower cards of the same suit that play the same.
                equals = fut.equals[c]
                for rank in range(2, 15):
                    if equals & (1 << rank):
                        tricks[suit + _RANK_CHARS[rank]] = score
            out.append(tricks)
    return out


def get_par_results(
    pbn_strings: List[str],
    mode: int = 0,
    trump_filter: Optional[List[int]] = None,
) -> List[dict]:
    """Compute DD tables AND par results.

    Parameters:
        pbn_strings: PBN deal strings.
        mode: Vulnerability (0=None, 1=Both, 2=NS, 3=EW).
        trump_filter: 5-element list; 0 = solve strain, 1 = skip.

    Returns:
        List of dicts with 'dd' (DDSSResult) and 'par' (parResults) keys.
    """
    if not DDSS_AVAILABLE:
        raise RuntimeError("ddss DLL is not available")

    n = len(pbn_strings)
    if n == 0:
        return []

    if trump_filter is None:
        trump_filter = [0, 0, 0, 0, 0]

    DealArray = ddTableDealPBN * n
    ResultArray = ddTableResults * n
    ParArray = parResults * n

    deals = DealArray()
    results = ResultArray()
    par = ParArray()
    tf = (ctypes.c_int * 5)(*trump_filter)

    for i, pbn in enumerate(pbn_strings):
        deals[i].cards = pbn.encode("ascii") if isinstance(pbn, str) else pbn

    ret = _dll.CalcAllTablesPBNx(
        n, deals, mode, tf, results,
        ctypes.cast(par, ctypes.POINTER(parResults)),
    )

    if ret != 1:
        buf = ctypes.create_string_buffer(80)
        _dll.ErrorMessage(ret, buf)
        raise RuntimeError(f"ddss CalcAllTablesPBNx error {ret}: {buf.value.decode()}")

    return [
        {
            "dd": DDSSResult(results[i]),
            "par_score": [par[i].parScore[s].value.decode() for s in range(2)],
            "par_contracts": [par[i].parContractsString[s].value.decode() for s in range(2)],
        }
        for i in range(n)
    ]


# ---- Claims (CheckClaimPBN: bridge-solver-cpp's library and ddss, include/dll.h)


class playTracePBN(ctypes.Structure):
    _fields_ = [("number", ctypes.c_int), ("cards", ctypes.c_char * 106)]


class claimResult(ctypes.Structure):
    _fields_ = [
        ("verdict", ctypes.c_int),
        ("guaranteed", ctypes.c_int),
        ("undecided", ctypes.c_int),
        ("tricksWon", ctypes.c_int),
        ("tricksLeft", ctypes.c_int),
        ("nodes", ctypes.c_longlong),
    ]


CLAIM_DEFAULT_BUDGET = 2_000_000
_CLAIM_VERDICTS = {0: "invalid", 1: "valid", 2: "undecided"}
_CARD_CHARS = "23456789TJQKA"
# ddss keeps search memory per thread index, so calls with one index must not
# overlap; bridge-solver-cpp's library ignores the index.
_claim_lock = threading.Lock()


def claim_supported() -> bool:
    """Whether the loaded library has the claim checker (CheckClaimPBN)."""
    if not DDSS_AVAILABLE or _dll is None:
        return False
    try:
        _dll.CheckClaimPBN
    except AttributeError:
        return False
    return True


def _normalize_card(card: str) -> str:
    text = card.strip().upper().replace("10", "T")
    if len(text) != 2 or text[0] not in _SUIT_CHARS or text[1] not in _CARD_CHARS:
        raise ValueError(f"card {card!r} is not a suit then a rank, like SA or HT")
    return text


def check_claim(
    pbn: str,
    strain: str,
    declarer: str,
    plays: List[str],
    claimer: Optional[str] = None,
    total_tricks: Optional[int] = None,
    leader: Optional[str] = None,
    budget: int = CLAIM_DEFAULT_BUDGET,
) -> dict:
    """Whether a claim holds however the cards the claimer cannot see lie.

    The claimer sees their own hand and dummy; the other two hands may be split
    any way consistent with the play (hand sizes and show-outs) and are played
    by an adversary. A defender's claim must work whatever partner holds and
    however partner plays. This is the model BBO's claim checker is described
    as using; the search is bridge-solver-rs's claim checker, ported.

    Parameters:
        pbn: the deal at the start of play, "N:..." (a full deal or an ending
             with four equal hands).
        strain: SHDC or N (NT).
        declarer: NESW.
        plays: cards played so far ("SA", "HT", ...), from the opening lead;
               may end mid-trick.
        claimer: the claiming seat; defaults to declarer. Dummy is declarer.
        total_tricks: the claiming side's total for the hand, tricks already
             won included (what BBO shows, LIN's mc). None asks for the
             largest valid claim instead.
        leader: who leads the first card; defaults to declarer's left.
        budget: search nodes; a claim that runs out is "undecided", which
             callers should treat as rejected, as BBO does.

    Returns:
        {"verdict": "valid" | "invalid" | "undecided"} for total_tricks, or
        {"guaranteed": int, "undecided": bool} without it (totals for the
        hand), plus tricks_won, tricks_left and nodes.

    Raises:
        RuntimeError: If the library is not loaded or has no claim checker.
        ValueError: On bad input, with the library's error text.
    """
    if not claim_supported():
        raise RuntimeError(
            "the loaded DDS library has no claim checker (CheckClaimPBN); "
            "it needs bridge-solver-cpp's dds-api or ddss with claims"
        )
    strain = strain.strip().upper()
    if strain == "NT":
        strain = "N"
    if strain not in _STRAIN_INDEX:
        raise ValueError(f"strain {strain!r} is not one of S, H, D, C, N")
    seats = {}
    for name, seat in (("declarer", declarer), ("claimer", claimer or declarer)):
        seat = seat.strip().upper()
        if seat not in _SEAT_INDEX:
            raise ValueError(f"{name} {seat!r} is not one of N, E, S, W")
        seats[name] = _SEAT_INDEX[seat]
    if leader is None:
        first = (seats["declarer"] + 1) % 4
    else:
        if leader.strip().upper() not in _SEAT_INDEX:
            raise ValueError(f"leader {leader!r} is not one of N, E, S, W")
        first = _SEAT_INDEX[leader.strip().upper()]
    cards = [_normalize_card(card) for card in plays]
    if len(cards) > 52:
        raise ValueError("more than 52 cards played")
    if total_tricks is not None and not 0 <= total_tricks <= 13:
        raise ValueError(f"total_tricks {total_tricks} is not 0 .. 13")

    deal = dealPBN()
    deal.trump = _STRAIN_INDEX[strain]
    deal.first = first
    for j in range(3):
        deal.currentTrickSuit[j] = 0
        deal.currentTrickRank[j] = 0
    deal.remainCards = pbn.strip().encode("ascii")
    trace = playTracePBN()
    trace.number = len(cards)
    trace.cards = "".join(cards).encode("ascii")
    result = claimResult()

    fn = _dll.CheckClaimPBN
    fn.restype = ctypes.c_int
    fn.argtypes = [
        dealPBN, playTracePBN, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.POINTER(claimResult), ctypes.c_int,
    ]
    with _claim_lock:
        ret = fn(
            deal, trace, seats["declarer"], seats["claimer"],
            -1 if total_tricks is None else int(total_tricks),
            int(budget), ctypes.byref(result), 0,
        )
    if ret != 1:
        buf = ctypes.create_string_buffer(80)
        _dll.ErrorMessage(ret, buf)
        hint = {
            -98: "a played card is not in the hand of the seat to play, or fails to follow suit",
            -3: "the claim is for more tricks than remain",
            -14: "the four hands are not the same size",
            -99: "the deal is not PBN, N:<N> <E> <S> <W> with suits S.H.D.C",
        }.get(ret)
        detail = f"{hint} ({buf.value.decode()})" if hint else buf.value.decode()
        raise ValueError(f"CheckClaimPBN error {ret}: {detail}")
    out: dict = {}
    if total_tricks is None:
        out["guaranteed"] = result.guaranteed
        out["undecided"] = bool(result.undecided)
    else:
        out["verdict"] = _CLAIM_VERDICTS[result.verdict]
    out["tricks_won"] = result.tricksWon
    out["tricks_left"] = result.tricksLeft
    out["nodes"] = result.nodes
    return out
