# src/edgarfacts/transforms/compute/companyfacts.py
"""
High-level computation of missing figures from raw company facts using taxonomy arcs.

This module is responsible for:
- Filtering arcs to the tag universe used in the facts
- Attaching taxonomy version to each (adsh) via submissions
- Computing *wide* figures (reported/quarterly/prior-year) first (done elsewhere)
- Applying arcs per version to compute missing tags (via arcs_apply)

In the refactor plan, this module mainly provides a convenience wrapper that:
- Ensures consistent tag dtype/categories
- Applies arcs per taxonomy version
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .arcs_apply import apply_arcs_by_version


def filter_arcs_to_tags(arcs_df: pd.DataFrame, tag_categories: pd.Index) -> pd.DataFrame:
    """
    Filter arcs to only those where both endpoints are in the known tag universe.

    Parameters
    ----------
    arcs_df:
        Must include ['from','to'] plus arc columns.
    tag_categories:
        The tag universe (typically facts_df['tag'].cat.categories).

    Returns
    -------
    Filtered arcs_df (copy).
    """
    if "from" not in arcs_df.columns or "to" not in arcs_df.columns:
        raise ValueError("arcs_df must contain 'from' and 'to' columns")

    tags = set(map(str, list(tag_categories)))
    m = arcs_df["from"].astype(str).isin(tags) & arcs_df["to"].astype(str).isin(tags)
    return arcs_df.loc[m].copy()


def align_arc_tag_dtype(arcs_df: pd.DataFrame, tag_categories: pd.Index) -> pd.DataFrame:
    """
    Optionally cast arc endpoints to pandas Categorical using the provided categories.
    This is beneficial for memory and join performance.

    Returns a copy.
    """
    arcs = arcs_df.copy()
    cats = list(map(str, list(tag_categories)))
    arcs["from"] = pd.Categorical(arcs["from"].astype(str), categories=cats)
    arcs["to"] = pd.Categorical(arcs["to"].astype(str), categories=cats)
    return arcs


def compute_missing_with_arcs(
    logger,
    figures_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    arcs_all_years_df: pd.DataFrame,
    *,
    keep_original_first: bool = True,
    enforce_tag_category: bool = True,
) -> pd.DataFrame:
    """
    Compute missing figures using US-GAAP calculation arcs.

    Parameters
    ----------
    figures_df:
        Wide figures table with columns:
        ['adsh','tag','reported_figure','quarterly_figure','reported_figure_py',
         'quarterly_figure_py','is_computed'].
    sub_df:
        Submissions with at least ['adsh','version'].
    arcs_all_years_df:
        Arcs for multiple years with at least
        ['version','seq','from','to','weight'].
    keep_original_first:
        If True, never overwrite existing values with computed ones.
    enforce_tag_category:
        If True and figures_df['tag'] is categorical, filter arcs to those categories
        and optionally cast arc endpoints to matching categoricals.

    Returns
    -------
    pd.DataFrame
        Wide figures_df augmented with computed rows where possible.
    """
    arcs = arcs_all_years_df

    if enforce_tag_category and "tag" in figures_df.columns and pd.api.types.is_categorical_dtype(figures_df["tag"]):
        cats = figures_df["tag"].cat.categories
        arcs = filter_arcs_to_tags(arcs, cats)
        arcs = align_arc_tag_dtype(arcs, cats)

    out = apply_arcs_by_version(
        figures_df=figures_df,
        sub_df=sub_df,
        arcs_all_years_df=arcs,
        logger=logger,
        keep_original_first=keep_original_first,
    )

    # Restore tag categorical dtype if present on input
    if enforce_tag_category and pd.api.types.is_categorical_dtype(figures_df["tag"]):
        out["tag"] = pd.Categorical(out["tag"].astype(str), categories=list(figures_df["tag"].cat.categories))

    return out
