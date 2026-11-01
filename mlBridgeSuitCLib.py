"""Single-suit, single-dummy analysis with Gerrit van der Velde's SuitC.

For every deal and partnership, each suit is analysed on its own: declarer sees
only the partnership's cards, the defenders defend perfectly, and every split of
the defenders' cards is weighted by its a-priori probability. SuitC assumes
North-South can lead from either hand whenever they like (unlimited entries)
unless told otherwise.

Per partnership P (NS, EW) and suit s (S, H, D, C) the deal-level columns are:
- SC_ET_{P}_{s}      expected tricks with the best line (SuitC "Result")
- SC_P{k}_{P}_{s}    best chance (0..1) of taking at least k tricks, k = 1..13
- SC_DDx_{P}_{s}     expected tricks if declarer could see the defenders' cards
                     (the best line on each split, weighted by its probability)
- SC_Guess_{P}_{s}   SC_DDx - SC_ET: tricks lost to not seeing the cards
- SC_Stop_{P}_{s}    chance (0..1) that P keeps at least one trick when the
                     opponents lead the suit every time (SuitC -D); null unless
                     the opponents hold at least twice P's cards
- SC_Block_{P}_{s}   SC_ET minus the expected tricks when play starts in the
                     short hand and the long hand has no outside entry; null
                     when either hand is void
plus SC_ET_{P} and SC_Guess_{P}, the sums over the four suits.

Results are cached per holding in SC_CACHE_DIR (parquet parts, shared by the
ACBL and FFBridge pipelines). There are only about 800,000 distinct holdings,
so after one backfill (``ffbridge``/``acbl`` runs or suitc_backfill.py) new
deals rarely need SuitC at all.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid
from typing import Iterable, Optional

import polars as pl

logger = logging.getLogger(__name__)

RANKS = 'AKQJT98765432'
SUITS = 'SHDC'
PAIRS = {'NS': ('N', 'S'), 'EW': ('E', 'W')}
SEATS = 'NESW'
MAX_TRICKS = 13

# -c020k caps quirk-tree merging (Gerrit used it for his full database). On 600
# real holdings it changed one safety-play percentage by 0.4 and no expected
# trick count, and it stops the rare combinations that would run for hours.
SUITC_OPTIONS = ('-r', '-c020k')
SUITC_TIMEOUT_S = 120
# Bump when the analysis or its parsing changes; older cache rows are ignored.
SC_CACHE_VERSION = 1

_REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SUITC_EXE = (_REPO / 'SuitC' / 'build' / 'Release' / 'SuitC.exe') if sys.platform == 'win32' \
    else (_REPO / 'SuitC' / 'build' / 'SuitC')
DEFAULT_CACHE_DIR = pathlib.Path(os.environ.get('SUITC_CACHE_DIR', r'E:\bridge\data\suitc\combos'))

_CACHE_SCHEMA = {
    'A': pl.String, 'B': pl.String, 'Version': pl.UInt16, 'Status': pl.String,
    'ET': pl.Float32, 'P': pl.List(pl.Float32), 'DDx': pl.Float32,
    'Stop': pl.Float32, 'Block': pl.Float32, 'Seconds': pl.Float32,
}

_RESULT_RE = re.compile(r'Result\s*:\s*([\d.]+)')
_TRICKS_RE = re.compile(r'^\s*(\d+) tricks:\s*([\d.]+)', re.M)
_MINMAX_RE = re.compile(r'min number of tricks:\s*(\d+)\s+max number of tricks:\s*(\d+)')
# Two-digit trick counts print as "10 -10", one-digit as "2 - 3".
_BOARD_RE = re.compile(r'^\s*(\d+) x\s+([\d.]+)\s+\S+\s+\S+\s+(\d+)\s*-\s*(\d+)\s*$', re.M)


def suitc_exe() -> pathlib.Path:
    exe = pathlib.Path(os.environ.get('SUITC_EXE', DEFAULT_SUITC_EXE))
    if not exe.is_file():
        raise FileNotFoundError(f"SuitC program not found: {exe} (build SuitC with CMake or set SUITC_EXE)")
    return exe


def holding_key(a: str, b: str) -> tuple[str, str]:
    """Cache key for one partnership's holding: the two hands in sorted order.

    Without a forced lead SuitC tries every lead, so swapping the hands gives
    the same answers.
    """
    return (a, b) if a <= b else (b, a)


def _defenders(a: str, b: str) -> str:
    return ''.join(r for r in RANKS if r not in a and r not in b)


def _run(args: list[str], timeout: float) -> str:
    """One SuitC run. Reads the output while it runs: a large table blocks on a full pipe."""
    done = subprocess.run([str(suitc_exe()), *args], capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0 or 'Result' not in done.stdout:
        raise RuntimeError(f"SuitC {' '.join(args)} failed (rc {done.returncode}): {(done.stdout + done.stderr)[-300:]}")
    return done.stdout


def _optimal(out: str) -> tuple[float, list[float]]:
    """Expected tricks and P(>= k), k = 1..13, from SuitC's "optimal results".

    SuitC lists only the trick counts that are not certain; below its minimum
    the chance is 1, above its maximum 0. When one line is best for every
    objective SuitC prints no "optimal results", only that line, whose
    exact-k chances then give P(>= k) by summing from the top.
    """
    lo, hi = (int(v) for v in _MINMAX_RE.search(out).groups())
    if 'The optimal results' in out:
        section = out.split('The optimal results', 1)[1].split('The optimum play', 1)[0]
        listed = {int(k): float(p) / 100.0 for k, p in _TRICKS_RE.findall(section)}
    else:
        section = out.split('The optimum play', 1)[1].split('The optimum play', 1)[0]
        exact = {int(k): float(p) / 100.0 for k, p in _TRICKS_RE.findall(section)}
        listed = {k: sum(p for j, p in exact.items() if j >= k) for k in exact}
    probs = [1.0 if k <= lo else listed.get(k, 0.0 if k > hi else 1.0) for k in range(1, MAX_TRICKS + 1)]
    return float(_RESULT_RE.search(section).group(1)), probs


def _dd_expectation(out: str) -> float:
    """Sum over the defenders' splits of probability x the best line's tricks (SuitC -b)."""
    rows = _BOARD_RE.findall(out)
    if not rows:
        raise RuntimeError("SuitC -b output has no distribution table")
    weight = sum(int(n) * float(p) for n, p, _lo, _hi in rows)
    if abs(weight - 100.0) > 0.01:
        raise RuntimeError(f"SuitC distribution probabilities sum to {weight}, not 100")
    return sum(int(n) * float(p) * int(hi) for n, p, _lo, hi in rows) / 100.0


def analyse_holding(a: str, b: str, timeout: float = SUITC_TIMEOUT_S) -> dict:
    """SuitC analysis of one partnership holding (a, b: e.g. "AQ2", "543"; "" = void)."""
    started = time.perf_counter()
    ew = _defenders(a, b)
    row = {'A': a, 'B': b, 'Version': SC_CACHE_VERSION, 'Status': 'ok',
           'ET': None, 'P': None, 'DDx': None, 'Stop': None, 'Block': None}
    try:
        ns_len, ew_len = len(a) + len(b), len(ew)
        if ns_len == 0:
            row.update(ET=0.0, P=[0.0] * MAX_TRICKS, DDx=0.0, Stop=0.0)
        elif ew_len == 0:  # all 13 cards: run the longer hand
            n = max(len(a), len(b))
            row.update(ET=float(n), P=[1.0 if k <= n else 0.0 for k in range(1, MAX_TRICKS + 1)], DDx=float(n))
            if a and b:
                row['Block'] = 0.0
        else:
            hands = [a or '.', b or '.', ew]
            out = _run([*SUITC_OPTIONS, '-b', *hands], timeout)
            et, probs = _optimal(out)
            row.update(ET=et, P=probs, DDx=_dd_expectation(out))
            if ew_len >= 2 * ns_len:
                defence = _run([*SUITC_OPTIONS, '-D', *hands], timeout)
                row['Stop'] = _optimal(defence)[1][0]
            if a and b:
                # Start in the short hand; the long hand has no outside entry.
                short, long_ = ('n', 's') if (len(a), a) < (len(b), b) else ('s', 'n')
                blocked = _run([*SUITC_OPTIONS, f'-L{short}', f'-l{long_}0', *hands], timeout)
                row['Block'] = et - _optimal(blocked)[0]
    except subprocess.TimeoutExpired:
        row.update(Status='timeout', ET=None, P=None, DDx=None, Stop=None, Block=None)
    except Exception as exc:  # recorded so a bad holding is not retried every run
        logger.warning(f"SuitC failed for {a!r} / {b!r}: {exc}")
        row.update(Status='error', ET=None, P=None, DDx=None, Stop=None, Block=None)
    row['Seconds'] = time.perf_counter() - started
    return row


class SuitCCache:
    """SuitC results per holding, kept as parquet parts in ``directory``.

    Parts are loaded once; ensure() runs SuitC for missing holdings in parallel
    and writes a new part every ``flush_every`` results, so a long backfill can
    stop and resume. Rows from another SC_CACHE_VERSION are ignored.
    """

    def __init__(self, directory: Optional[pathlib.Path] = None, workers: Optional[int] = None,
                 flush_every: int = 20_000):
        self.directory = pathlib.Path(directory) if directory is not None else DEFAULT_CACHE_DIR
        self.workers = workers or os.cpu_count() or 8
        self.flush_every = flush_every
        self._table: Optional[pl.DataFrame] = None
        self._pending: list[dict] = []

    def table(self) -> pl.DataFrame:
        if self._table is None:
            parts = sorted(self.directory.glob('part-*.parquet')) if self.directory.is_dir() else []
            if parts:
                t = time.time()
                self._table = (pl.scan_parquet(parts).filter(pl.col('Version') == SC_CACHE_VERSION)
                               .collect().unique(subset=['A', 'B'], keep='last'))
                logger.info(f"SuitC cache: {self._table.height} holdings from {len(parts)} parts ({time.time() - t:.0f}s)")
            else:
                self._table = pl.DataFrame(schema=_CACHE_SCHEMA)
        return self._table

    def _flush(self) -> None:
        if not self._pending:
            return
        part = pl.DataFrame(self._pending, schema=_CACHE_SCHEMA)
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"part-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = self.directory / f".{name}.tmp"
        part.write_parquet(tmp)
        os.replace(tmp, self.directory / name)
        # A retried holding replaces its earlier row.
        self._table = pl.concat([self.table(), part], how='vertical_relaxed').unique(
            subset=['A', 'B'], keep='last', maintain_order=True)
        self._pending = []

    def ensure(self, keys: Iterable[tuple[str, str]], show_progress: bool = True) -> int:
        """Run SuitC for every (A, B) holding not yet cached. Returns how many were run.

        Rows cached as 'error' are run again (a parsing fix then takes effect);
        'timeout' rows are kept.
        """
        have = self.table().filter(pl.col('Status') != 'error').select('A', 'B')
        wanted = pl.DataFrame(list(set(keys)), schema={'A': pl.String, 'B': pl.String}, orient='row')
        todo = wanted.join(have, on=['A', 'B'], how='anti').rows()
        if not todo:
            return 0
        logger.info(f"SuitC: {len(todo)} new holdings ({self.workers} workers)")
        from tqdm import tqdm
        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            futures = [pool.submit(analyse_holding, a, b) for a, b in todo]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc='SuitC holdings', disable=not show_progress):
                self._pending.append(future.result())
                if len(self._pending) >= self.flush_every:
                    self._flush()
        self._flush()
        bad = self.table().filter(pl.col('Status') != 'ok').height
        logger.info(f"SuitC: {len(todo)} holdings in {time.time() - started:.0f}s; {bad} cached as timeout/error")
        return len(todo)


def _suit_holdings(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (input row, partnership, suit) with the two hands' cards."""
    hands = df.select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias('_row'),
        pl.col('PBN').str.slice(2).str.split(' ').alias('_h'),
    ).select('_row', *[pl.col('_h').list.get(i).str.split('.').alias(SEATS[i]) for i in range(4)])
    parts = []
    for pair, (x, y) in PAIRS.items():
        for si, suit in enumerate(SUITS):
            parts.append(hands.select(
                '_row', pl.lit(pair).alias('_pair'), pl.lit(suit).alias('_suit'),
                pl.col(x).list.get(si).alias('_x'), pl.col(y).list.get(si).alias('_y')))
    return pl.concat(parts).with_columns(
        pl.min_horizontal('_x', '_y').alias('A'), pl.max_horizontal('_x', '_y').alias('B'))


SC_SUIT_STATS = ('ET', 'DDx', 'Guess', 'Stop', 'Block')


def sc_columns() -> list[str]:
    """Names of the deal-level SC_* columns, in output order."""
    cols = []
    for pair in PAIRS:
        for suit in SUITS:
            cols += [f'SC_{s}_{pair}_{suit}' for s in SC_SUIT_STATS]
            cols += [f'SC_P{k}_{pair}_{suit}' for k in range(1, MAX_TRICKS + 1)]
        cols += [f'SC_ET_{pair}', f'SC_Guess_{pair}']
    return cols


def add_suitc_features(df: pl.DataFrame, cache: SuitCCache, show_progress: bool = True) -> pl.DataFrame:
    """Add the deal-level SC_* columns (see module docstring) from 'PBN' ("N:..." deals).

    Holdings SuitC could not analyse (timeout/error) give nulls for that suit
    and null partnership sums.
    """
    assert 'PBN' in df.columns, "Required column 'PBN' not found in DataFrame"
    assert 'SC_ET_NS' not in df.columns, "SC_* columns already exist"
    bad = df.filter(~pl.col('PBN').str.starts_with('N:'))
    if bad.height:
        raise ValueError(f"PBN must start with 'N:' ({bad.height} rows, e.g. {bad['PBN'][0]!r})")
    t = time.time()
    long = _suit_holdings(df)
    cache.ensure(long.select('A', 'B').unique().rows(), show_progress=show_progress)
    stats = cache.table().select(
        'A', 'B', 'ET', 'DDx', (pl.col('DDx') - pl.col('ET')).alias('Guess'), 'Stop', 'Block',
        *[pl.col('P').list.get(k - 1).alias(f'P{k}') for k in range(1, MAX_TRICKS + 1)])
    long = long.join(stats, on=['A', 'B'], how='left').with_columns(
        (pl.col('_pair') + '_' + pl.col('_suit')).alias('_ps'))
    values = [*SC_SUIT_STATS, *[f'P{k}' for k in range(1, MAX_TRICKS + 1)]]
    wide = long.pivot(on='_ps', index='_row', values=values, separator='|')
    wide = wide.rename({c: f"SC_{c.split('|')[0]}_{c.split('|')[1]}" for c in wide.columns if c != '_row'})
    wide = wide.with_columns(
        *[pl.sum_horizontal([pl.col(f'SC_ET_{p}_{s}') for s in SUITS]).alias(f'SC_ET_{p}') for p in PAIRS],
        *[pl.sum_horizontal([pl.col(f'SC_Guess_{p}_{s}') for s in SUITS]).alias(f'SC_Guess_{p}') for p in PAIRS],
    )
    # sum_horizontal skips nulls; a partnership total is only meaningful when every suit was analysed.
    wide = wide.with_columns(
        *[pl.when(pl.all_horizontal([pl.col(f'SC_{stat}_{p}_{s}').is_not_null() for s in SUITS]))
          .then(pl.col(f'SC_{stat}_{p}')).otherwise(None).alias(f'SC_{stat}_{p}')
          for p in PAIRS for stat in ('ET', 'Guess')])
    out = df.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias('_row')).join(
        wide, on='_row', how='left', maintain_order='left').drop('_row')
    out = out.with_columns(pl.col(c).cast(pl.Float32) for c in sc_columns())
    logger.info(f"SuitC features for {df.height} rows in {time.time() - t:.0f}s")
    return out.select(*df.columns, *sc_columns())


def add_suitc_declarer_columns(df: pl.DataFrame, diff_col: str = 'DD_Tricks_Diff') -> pl.DataFrame:
    """Board-result columns from the deal-level SC_* columns.

    - SC_Guess_Declarer: SC_Guess for the declaring partnership (all four suits)
    - DD_Tricks_Diff_SC: DD_Tricks_Diff + SC_Guess_Declarer, i.e. tricks taken
      against what a declarer who cannot see the defenders' cards should expect.
      Positive means better than that.

    diff_col: the tricks-minus-DD column (FFBridge's embedded path calls it DDTricks_Diff).
    """
    for col in ('Declarer_Direction', diff_col, 'SC_Guess_NS', 'SC_Guess_EW'):
        assert col in df.columns, f"Required column '{col}' not found in DataFrame"
    guess = (pl.when(pl.col('Declarer_Direction').is_in(['N', 'S'])).then(pl.col('SC_Guess_NS'))
             .when(pl.col('Declarer_Direction').is_in(['E', 'W'])).then(pl.col('SC_Guess_EW'))
             .otherwise(None))
    return df.with_columns(guess.cast(pl.Float32).alias('SC_Guess_Declarer')).with_columns(
        (pl.col(diff_col).cast(pl.Float32) + pl.col('SC_Guess_Declarer')).alias('DD_Tricks_Diff_SC'))
