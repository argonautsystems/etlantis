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

"""etlantis.transform.consolidate — concat + dedupe + normalize column names.

Generalized from cleanroom `P0_consolidate_vectorized.py`. The cleanroom
version was DBPR-specific (hardcoded inspections/licenses/disciplinary
buckets) and pandas-based. This version is:

  * Polars-native — operates on `pl.DataFrame`.
  * Generic — `concat_frames()` is the workhorse; apps assemble bucket
    routing themselves.
  * Schema-resilient — normalizes column names (strips whitespace,
    deduplicates collisions) BEFORE concat, since gov-CSV column-naming
    is the #1 source of "two columns that should match but don't" bugs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsolidationResult:
    """Outcome of a `concat_frames()` call.

    Carries the consolidated frame plus enough telemetry to surface
    schema-drift surprises and silent dedupe loss.
    """

    df: pl.DataFrame
    """The consolidated DataFrame. Empty (zero rows, zero cols) if the
    input list was empty."""

    input_frames: int
    """How many frames went in (after empty-frame filtering)."""

    rows_in: int
    """Sum of input frame heights — what we'd have without dedupe."""

    rows_out: int
    """Final frame height — equals `df.height`."""

    duplicates_removed: int
    """`rows_in - rows_out`. Zero when dedupe wasn't requested."""

    column_renames: tuple[dict[str, str], ...]
    """Per-frame rename maps in input order. `column_renames[i]` describes
    the renames applied to frame `i` (after empty-frame filtering). Empty
    inner dicts mean no rewriting was needed for that frame. Per-frame
    rather than flat so two frames that rewrote the same original column
    to different targets don't overwrite each other in the audit trail.
    """


def concat_frames(
    frames: list[pl.DataFrame],
    *,
    dedupe_subset: list[str] | None = None,
    sort_by: list[str] | None = None,
    sort_descending: bool = False,
    normalize_column_names: bool = True,
) -> ConsolidationResult:
    """Vertically concatenate a list of DataFrames with optional dedupe + sort.

    Args:
        frames: Polars DataFrames to concatenate. Empty list returns an
            empty result. `None` entries and zero-height frames are
            filtered out before concat.
        dedupe_subset: Columns to deduplicate on. `None` skips dedupe;
            `[]` deduplicates on all columns (i.e. drop fully-duplicate
            rows). The first occurrence wins, mirroring polars'
            `unique(keep="first")`.
        sort_by: Columns to sort by after concat (and after dedupe).
            `None` skips sort.
        sort_descending: When `sort_by` is set, sort descending. Default
            ascending. Single bool applies to all sort columns; pass
            multi-direction sort kwargs by post-processing if you need it.
        normalize_column_names: Strip leading/trailing whitespace from
            column names, then disambiguate collisions with `_1`, `_2`
            suffixes. Default True — public-records CSVs are notorious
            for ' Name' / 'Name ' / 'Name' all coexisting in one schema.

    Returns:
        ConsolidationResult. `df` is always a valid DataFrame (possibly
        empty). Raises only if input frames have incompatible schemas
        AFTER name normalization (Polars surfaces that as a SchemaError).
    """
    real_frames = [f for f in frames if f is not None and f.height > 0]
    rows_in = sum(f.height for f in real_frames)

    if not real_frames:
        return ConsolidationResult(
            df=pl.DataFrame(),
            input_frames=0,
            rows_in=0,
            rows_out=0,
            duplicates_removed=0,
            column_renames=(),
        )

    per_frame_renames: tuple[dict[str, str], ...]
    if normalize_column_names:
        real_frames, per_frame_renames = _normalize_columns(real_frames)
    else:
        per_frame_renames = tuple({} for _ in real_frames)

    # Validate dedupe_subset is present in every input frame BEFORE concat.
    # Without this, `diagonal_relaxed` would null-fill the missing key in
    # frames that don't have it, then `unique()` would collapse all of
    # those rows into a single null-keyed row — silent, severe data loss.
    if dedupe_subset:
        for idx, frame in enumerate(real_frames):
            missing = [c for c in dedupe_subset if c not in frame.columns]
            if missing:
                raise ValueError(
                    f"dedupe_subset columns {missing} missing from input frame {idx} "
                    f"(after column normalization). dedupe across heterogeneous "
                    f"schemas would silently null-key those rows and collapse them. "
                    f"Available columns in this frame: {frame.columns}"
                )

    df = pl.concat(real_frames, how="diagonal_relaxed")

    if dedupe_subset is not None:
        before = df.height
        df = df.unique(subset=dedupe_subset or None, keep="first", maintain_order=True)
        duplicates_removed = before - df.height
    else:
        duplicates_removed = 0

    if sort_by:
        df = df.sort(sort_by, descending=sort_descending)

    return ConsolidationResult(
        df=df,
        input_frames=len(real_frames),
        rows_in=rows_in,
        rows_out=df.height,
        duplicates_removed=duplicates_removed,
        column_renames=per_frame_renames,
    )


def _normalize_columns(
    frames: list[pl.DataFrame],
) -> tuple[list[pl.DataFrame], tuple[dict[str, str], ...]]:
    """Strip whitespace from column names and disambiguate collisions.

    Public-records reality: a CSV may carry both ' license_number' and
    'license_number' as distinct columns due to inspector-tool export
    quirks. After stripping, both become 'license_number' — Polars'
    concat would error or merge them depending on the strategy. We
    deterministically rename collisions to `name`, `name_1`, `name_2`,
    etc. and report a per-frame rename map.

    Canonical-ownership rule: when both ' name' and 'name' are present,
    the already-clean 'name' keeps the canonical slot and ' name' becomes
    'name_1'. Any downstream manifest expecting `name` to refer to the
    clean column is therefore correct without further reasoning. We
    achieve this with a two-pass scan: pass 1 reserves slots for already-
    clean column names; pass 2 assigns canonical (or _1/_2/_3 collision-
    suffixed) targets to the remaining columns in order.

    Returns the renamed frames and a tuple of per-frame rename dicts. The
    tuple length equals the input frame count.
    """
    out_frames: list[pl.DataFrame] = []
    out_renames: list[dict[str, str]] = []
    for frame in frames:
        # Group columns by their stripped form, preserving input order
        # so dict iteration is deterministic.
        groups: dict[str, list[str]] = {}
        for col in frame.columns:
            groups.setdefault(col.strip(), []).append(col)

        rename_map: dict[str, str] = {}
        for stripped, members in groups.items():
            # Pick the canonical winner: prefer an already-clean member
            # (so manifest code expecting `name` keeps pointing at the
            # actually-clean column). If none is already clean, take the
            # first in input order. The remaining members get _1, _2, …
            # in input order.
            winner = next((c for c in members if c == stripped), members[0])
            if winner != stripped:
                rename_map[winner] = stripped
            suffix_index = 0
            for col in members:
                if col == winner:
                    continue
                suffix_index += 1
                rename_map[col] = f"{stripped}_{suffix_index}"

        out_frames.append(frame.rename(rename_map) if rename_map else frame)
        out_renames.append(rename_map)
    return out_frames, tuple(out_renames)


__all__ = ["concat_frames", "ConsolidationResult"]
