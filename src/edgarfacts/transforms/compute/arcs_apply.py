# src/edgarfacts/transforms/compute/arcs_apply.py
"""
Apply US-GAAP calculation arcs to a figures table.

This module computes missing (adsh, tag) figures by aggregating weighted
source tags according to calculation linkbase arcs.

Contract (important)
--------------------
Input figures_df MUST have (wide) columns:
- adsh (int-like)
- tag  (str/category)
- reported_figure
- quarterly_figure
- reported_figure_py
- quarterly_figure_py
- is_computed (bool)

Input arcs_df MUST have columns (single taxonomy version):
- seq (int)
- from (str)
- to (str)
- weight (float)

Output is figures_df with additional computed rows and/or filled values.
Existing reported values are not overwritten unless explicitly requested.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


_VALUE_COLS = [
    "reported_figure",
    "quarterly_figure",
    "reported_figure_py",
    "quarterly_figure_py",
]


def _ensure_required_columns(figures_df: pd.DataFrame) -> None:
    missing = [c for c in (["adsh", "tag", "is_computed"] + _VALUE_COLS) if c not in figures_df.columns]
    if missing:
        raise ValueError(f"figures_df is missing required columns: {missing}")


def _ensure_arcs_columns(arcs_df: pd.DataFrame) -> None:
    missing = [c for c in ["seq", "from", "to", "weight"] if c not in arcs_df.columns]
    if missing:
        raise ValueError(f"arcs_df is missing required columns: {missing}")


def apply_arcs_to_figures(
    figures_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    *,
    keep_original_first: bool = True,
    max_passes: int = 1,
) -> pd.DataFrame:
    """
    Apply arcs to compute missing figures.

    Parameters
    ----------
    figures_df:
        Wide figures table (see module docstring).
    arcs_df:
        Calculation arcs for a single taxonomy year.
    keep_original_first:
        If True, existing rows for a (adsh, tag) take precedence over computed ones.
        If False, computed rows override existing ones where both exist.
    max_passes:
        Number of full passes over seq levels. Usually 1 is sufficient if seq ordering
        is adequate. Increase if you want to allow deeper propagation.

    Returns
    -------
    pd.DataFrame
        Same schema as figures_df, with additional computed rows for missing tags.
    """
    _ensure_required_columns(figures_df)
    _ensure_arcs_columns(arcs_df)

    # Work on copies; keep deterministic order
    out = figures_df.copy()
    arcs = arcs_df.copy()

    # Normalize types for joins
    arcs["weight"] = arcs["weight"].astype(float)

    # Ensure tag is string-like for matching 'from'/'to'
    # (We won't force category here; caller can re-cast later.)
    out["tag"] = out["tag"].astype(str)
    arcs["from"] = arcs["from"].astype(str)
    arcs["to"] = arcs["to"].astype(str)

    # Optional: restrict arcs to tags that exist in the universe of tags we're tracking
    # (union of current tags and arc targets/sources)
    # Keep as-is; downstream may do stronger filtering.

    for _ in range(max_passes):
        for s in np.sort(arcs["seq"].unique()):
            a = arcs[arcs["seq"] == s][["from", "to", "weight"]].copy()
            if a.empty:
                continue

            # Join existing figures on arc "from" as source tag
            src = out.merge(a, how="inner", left_on="tag", right_on="from")

            if src.empty:
                continue

            # Weighted contribution to the target tag
            src = src.drop(columns=["tag", "from"]).rename(columns={"to": "tag"})
            for c in _VALUE_COLS:
                src[c] = src[c] * src["weight"]

            src = src.drop(columns=["weight"])

            # Aggregate contributions per (adsh, tag)
            comp = (
                src.groupby(["adsh", "tag"], as_index=False, sort=False)
                .sum(min_count=1)
                .assign(is_computed=True)
            )

            # Combine with existing, resolving collisions deterministically
            if keep_original_first:
                out = (
                    pd.concat([out.assign(_prio=1), comp.assign(_prio=2)], ignore_index=True)
                    .sort_values(["adsh", "tag", "_prio"], kind="mergesort")
                    .drop_duplicates(subset=["adsh", "tag"], keep="first")
                    .drop(columns="_prio")
                )
            else:
                out = (
                    pd.concat([out.assign(_prio=2), comp.assign(_prio=1)], ignore_index=True)
                    .sort_values(["adsh", "tag", "_prio"], kind="mergesort")
                    .drop_duplicates(subset=["adsh", "tag"], keep="first")
                    .drop(columns="_prio")
                )

    return out


def apply_arcs_by_version(
    figures_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    arcs_all_years_df: pd.DataFrame,
    logger,
    *,
    keep_original_first: bool = True,
) -> pd.DataFrame:
    """
    Apply calculation arcs by taxonomy version.

    Parameters
    ----------
    figures_df:
        Wide figures table.
    sub_df:
        Submissions table providing (adsh -> version). Must include columns ['adsh','version'].
    arcs_all_years_df:
        Arcs for multiple years. Must include columns ['version','seq','from','to','weight'].
    logger:
        Logger for progress reporting.
    keep_original_first:
        Collision policy, passed to apply_arcs_to_figures.

    Returns
    -------
    pd.DataFrame
        figures_df with computed tags added per version.
    """
    _ensure_required_columns(figures_df)
    if "version" not in sub_df.columns:
        raise ValueError("sub_df must contain a 'version' column")
    if "version" not in arcs_all_years_df.columns:
        raise ValueError("arcs_all_years_df must contain a 'version' column")

    # Attach version to each adsh in figures_df
    vmap = sub_df[["adsh", "version"]].drop_duplicates()
    work = figures_df.merge(vmap, how="inner", on="adsh")

    out_frames = []
    for v in np.sort(work["version"].unique()):
        logger.info(f"Applying taxonomy arcs for version {int(v)}")
        f_v = work[work["version"] == v].drop(columns="version")
        a_v = arcs_all_years_df[arcs_all_years_df["version"] == v].drop(columns="version")
        out_frames.append(
            apply_arcs_to_figures(f_v, a_v, keep_original_first=keep_original_first)
        )

    out = pd.concat(out_frames, ignore_index=True)

    # Preserve original ordering / tag dtype (caller may re-cast to category)
    return out
