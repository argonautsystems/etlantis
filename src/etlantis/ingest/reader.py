# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""etlantis.ingest.reader — Polars-native CSV/Excel/Parquet reader with encoding fallback.

Generalized from the RiskyEats cleanroom `E1_ingest_vectorized.py`. The
cleanroom version was pandas-based with multiprocessing for parallelism
and DBPR-specific filename handling. This version is:

  * Polars-native — returns `pl.DataFrame`, never pandas.
  * Format-dispatching — reads CSV/TSV/Excel/Parquet by file extension,
    not by filename pattern. Apps that need pattern-based dispatch wrap
    this with their own routing.
  * Encoding-resilient — UTF-8 first, then cp1252 → latin-1 fallback for
    CSVs (the public-records reality: gov sites still serve windows-1252
    bytes labelled "text/csv"). Polars itself only handles UTF-8 / UTF-8
    lossy natively, so the fallback path decodes bytes manually and feeds
    the re-encoded UTF-8 back through Polars.
  * Strict by default — `pl.read_csv` is invoked with its native strict
    settings. Lossy parsing (`ignore_errors`, `truncate_ragged_lines`)
    must be opted in via `csv_kwargs`. Silent data loss caused enough
    pain in the cleanroom era that we now require the caller to ask
    for it.

Lifecycle:

    from etlantis.ingest.reader import read_table, ReadResult

    result = read_table(Path("data/inspections.csv"))
    if result.df is not None:
        print(result.df.shape)

    # Lossy parse for known-messy gov data:
    result = read_table(
        Path("data/disciplinary.csv"),
        csv_kwargs={"ignore_errors": True, "truncate_ragged_lines": True},
    )
"""

from __future__ import annotations

import io
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# Default encoding fallback chain for CSV. UTF-8 first because that's what
# we hope for; latin-1 is the lossless safety net (every byte sequence is
# valid latin-1); cp1252 is between because some sources mix Windows-1252
# punctuation into otherwise-ASCII payloads.
_DEFAULT_CSV_ENCODINGS: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")

# Extensions we know how to dispatch. Lowercased; compared after suffix.
_CSV_EXTS = {".csv", ".tsv", ".txt"}
_EXCEL_EXTS = {".xlsx", ".xls"}
_PARQUET_EXTS = {".parquet", ".pq"}


@dataclass(frozen=True)
class ReadResult:
    """Outcome of a single read_table() call.

    Read failures don't raise — they return a ReadResult with `df=None`
    and `error` populated. This keeps the call site free of try/except
    boilerplate and lets pipelines decide per-file how to handle failures
    (skip / abort / fallback to a backup file).
    """

    path: Path
    """The file we attempted to read."""

    df: pl.DataFrame | None
    """The parsed DataFrame, or None if every fallback failed."""

    encoding: str | None = None
    """For CSVs: the encoding that successfully parsed. None for non-CSV
    formats and on failure."""

    rows: int = 0
    """Row count, 0 on failure."""

    error: str | None = None
    """Human-readable failure description, None on success."""

    warnings: tuple[str, ...] = ()
    """Non-fatal observations (e.g. high-null-percentage suggesting parse
    issues). Populated only on success. Tuple (not list) so the frozen
    dataclass actually IS immutable through this attribute too."""


def read_table(
    path: Path | str,
    *,
    encodings: tuple[str, ...] | None = None,
    null_warn_threshold: float = 0.5,
    csv_kwargs: Mapping[str, Any] | None = None,
    excel_kwargs: Mapping[str, Any] | None = None,
    parquet_kwargs: Mapping[str, Any] | None = None,
) -> ReadResult:
    """Read a single tabular file as a Polars DataFrame.

    Format dispatch by extension: .csv/.tsv/.txt → CSV reader (with
    encoding fallback); .xlsx/.xls → Excel reader; .parquet/.pq → Parquet.

    Args:
        path: File to read.
        encodings: Override the default UTF-8/cp1252/latin-1 fallback
            chain for CSV. Ignored for non-CSV formats.
        null_warn_threshold: For CSV — if the parsed frame has more than
            this fraction of nulls across all cells (range 0.0–1.0), add
            a warning to the result. Default 0.5 mirrors cleanroom
            heuristic.
        csv_kwargs / excel_kwargs / parquet_kwargs: Extra keyword args
            forwarded to the underlying Polars reader. The encoding-
            fallback path overrides csv_kwargs["encoding"] internally.

    Returns:
        ReadResult. Never raises on parse failure — `df=None` + `error`
        set instead. Other exceptions (FileNotFoundError, etc.) DO
        propagate; check existence at the call site if needed.
    """
    path = Path(path)
    if not path.exists():
        return ReadResult(path=path, df=None, error=f"file not found: {path}")
    suffix = path.suffix.lower()

    if suffix in _CSV_EXTS:
        return _read_csv(
            path,
            encodings=encodings or _DEFAULT_CSV_ENCODINGS,
            null_warn_threshold=null_warn_threshold,
            extra_kwargs=dict(csv_kwargs or {}),
        )
    if suffix in _EXCEL_EXTS:
        return _read_excel(path, extra_kwargs=dict(excel_kwargs or {}))
    if suffix in _PARQUET_EXTS:
        return _read_parquet(path, extra_kwargs=dict(parquet_kwargs or {}))

    return ReadResult(path=path, df=None, error=f"unsupported extension: {suffix!r}")


def _read_csv(
    path: Path,
    *,
    encodings: tuple[str, ...],
    null_warn_threshold: float,
    extra_kwargs: dict[str, Any],
) -> ReadResult:
    """CSV reader with encoding fallback.

    Polars supports `encoding="utf8"` and `encoding="utf8-lossy"` natively.
    For non-UTF-8 sources we read raw bytes, decode with the candidate
    encoding, re-encode as UTF-8, and feed the buffer back to Polars.
    This is slower than direct read but only fires when UTF-8 fails, and
    correctness beats speed for one-off ingestion.

    Strict-by-default:
      - infer_schema_length is lifted to 10000 (polars default 100 misses
        gov-data type surprises hundreds of rows in). Lifting the schema
        window does NOT discard data; it only changes the type-inference
        heuristic.
      - `ignore_errors` and `truncate_ragged_lines` are NOT enabled by
        default. Set them via `csv_kwargs` when you want lenient parsing
        of known-messy sources, and accept that you're trading data
        fidelity for survival. Default behavior raises on malformed rows
        so silent data loss can't sneak past code review.

    The `encoding` kwarg is reserved by this function and will be popped
    out of `extra_kwargs` if the caller supplied one (with a warning) —
    the encoding-fallback chain is what we control here, not the caller.
    """
    last_error: str | None = None
    defaults: dict[str, Any] = {
        "infer_schema_length": 10000,
    }
    defaults.update(extra_kwargs)
    # The encoding fallback chain is this function's contract; a caller-
    # supplied `encoding` would either silently override or — worse —
    # cause a TypeError ("multiple values for keyword argument 'encoding'")
    # when we pass our own. Strip and warn.
    if "encoding" in defaults:
        logger.warning(
            "[reader] %s: csv_kwargs['encoding']=%r ignored — "
            "encoding is governed by the fallback chain",
            path.name,
            defaults.pop("encoding"),
        )

    for encoding in encodings:
        try:
            df = _read_csv_with_encoding(path, encoding, defaults)
        except (pl.exceptions.ComputeError, pl.exceptions.PolarsError) as exc:
            last_error = f"{encoding}: {exc}"
            logger.debug("[reader] %s: %s parse failed: %s", path.name, encoding, exc)
            continue
        except UnicodeDecodeError as exc:
            last_error = f"{encoding}: {exc}"
            logger.debug("[reader] %s: %s decode failed: %s", path.name, encoding, exc)
            continue

        warnings: list[str] = []
        rows = df.height
        if rows == 0:
            warnings.append("file parsed but is empty")
        else:
            null_pct = _null_fraction(df)
            if null_pct > null_warn_threshold:
                warnings.append(f"{null_pct:.1%} null cells — possible delimiter/encoding mismatch")

        if encoding != "utf-8":
            logger.info(
                "[reader] %s: parsed via fallback encoding %s (%d rows)",
                path.name,
                encoding,
                rows,
            )
        return ReadResult(
            path=path,
            df=df,
            encoding=encoding,
            rows=rows,
            warnings=tuple(warnings),
        )

    return ReadResult(
        path=path,
        df=None,
        error=f"all encodings failed (last: {last_error})",
    )


def _read_csv_with_encoding(path: Path, encoding: str, defaults: dict[str, Any]) -> pl.DataFrame:
    """Read a CSV at `path` under `encoding`, going through Polars.

    UTF-8 reads go directly through pl.read_csv. Non-UTF-8 reads decode
    bytes into a Python str via the requested encoding (raises
    UnicodeDecodeError if the byte sequence isn't valid for that
    encoding), re-encode as UTF-8, and hand the buffer to pl.read_csv.

    Latin-1 has the unique property that every byte sequence is a valid
    decode — so the latin-1 step never raises UnicodeDecodeError. This is
    why latin-1 sits last in the fallback chain: it always "succeeds" at
    decoding, but may produce nonsense for cp1252-encoded source bytes
    (mojibake on smart-quote characters). cp1252 sits ahead of latin-1
    so that valid Windows-1252 sources resolve to clean text first.
    """
    if encoding == "utf-8":
        return pl.read_csv(path, encoding="utf8", **defaults)
    raw = path.read_bytes()
    decoded = raw.decode(encoding)
    buf = io.BytesIO(decoded.encode("utf-8"))
    return pl.read_csv(buf, encoding="utf8", **defaults)


def _read_excel(path: Path, *, extra_kwargs: dict[str, Any]) -> ReadResult:
    """Excel reader. Polars dispatches to xlsx2csv / openpyxl / calamine
    depending on what's installed; we don't pin a backend so the user's
    environment chooses.

    Multi-sheet workbooks: by default `pl.read_excel` returns the first
    sheet. Apps wanting all sheets should pass `sheet_id=None` via
    `extra_kwargs` and handle the dict-of-DataFrames result themselves
    (out of scope here — keep the contract simple).
    """
    try:
        df = pl.read_excel(path, **extra_kwargs)
    except Exception as exc:  # noqa: BLE001 — Polars wraps backend errors here
        return ReadResult(path=path, df=None, error=f"excel parse failed: {exc}")

    # pl.read_excel may return a dict if multi-sheet; refuse to guess.
    if not isinstance(df, pl.DataFrame):
        return ReadResult(
            path=path,
            df=None,
            error=(
                f"expected a single DataFrame from {path.name}; got "
                f"{type(df).__name__}. Pass sheet_name= via excel_kwargs to "
                "select one sheet."
            ),
        )
    return ReadResult(path=path, df=df, rows=df.height)


def _read_parquet(path: Path, *, extra_kwargs: dict[str, Any]) -> ReadResult:
    """Parquet is the well-behaved sibling: native Polars, no encoding
    games, no schema drift."""
    try:
        df = pl.read_parquet(path, **extra_kwargs)
    except Exception as exc:  # noqa: BLE001 — pyarrow/polars surface multiple types
        return ReadResult(path=path, df=None, error=f"parquet parse failed: {exc}")
    return ReadResult(path=path, df=df, rows=df.height)


def _null_fraction(df: pl.DataFrame) -> float:
    """Fraction of null cells across the whole frame (range 0.0–1.0).

    Used as a heuristic for "this CSV parsed but probably wrong" — when
    the delimiter or encoding is off, polars often returns a frame with
    everything in the first column and the rest as nulls.
    """
    total_cells = df.height * df.width
    if total_cells == 0:
        return 0.0
    null_total = sum(df.null_count().row(0))
    return null_total / total_cells


def read_many(
    paths: list[Path | str],
    **kwargs: Any,
) -> list[ReadResult]:
    """Read multiple tables sequentially. Same kwargs as `read_table()`.

    Sequential by design at v0.1: parallelism (threads vs processes vs
    polars' internal scan parallelism) interacts badly with file-handle
    limits, encoding-fallback retries, and downstream memory pressure.
    Apps that need parallelism should compose with their own pool — the
    primitives here stay synchronous and predictable.
    """
    return [read_table(p, **kwargs) for p in paths]


__all__ = ["ReadResult", "read_table", "read_many"]
