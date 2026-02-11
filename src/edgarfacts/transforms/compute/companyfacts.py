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

from .arcs_apply import apply_arcs_by_version, filter_unreliable_arcs, expand_arcs_with_single_variable_rearrangements


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

    validate_arcs: bool = True,
    expand_arcs: bool = True,
    revalidate_after_expand: bool = True,

    reliability_value_col: str = "reported_figure",
    min_tests_per_equation: int = 5,
    max_fail_rate: float = 0.1,
    rtol: float = 1e-5,
    atol: float = 1e-4,
    require_full_sources: bool = True,

    pivot_weights: tuple[float, float] = (1.0, -1.0),
    enforce_signature_unique_seq: bool = True,
    renumber_seq: bool = True,
) -> pd.DataFrame:
    """
    Compute missing figures using US-GAAP calculation arcs.

    Pipeline (optional, controlled by flags):
    1) Reliability filter of input arcs (per (version, statement, from)).
    2) Expand arcs with single-variable rearrangements (artificial equations).
    3) Reliability filter again after expansion (recommended).
    4) Apply arcs by taxonomy version.

    Notes
    -----
    - keep_original_first should remain True for semantics-preserving behavior.
    - This function assumes arcs_df uses XBRL calc semantics:
        FROM (total) = Σ weight * TO (component)
    """
    arcs = arcs_all_years_df

    # Ensure arc endpoints are within the known tag universe (optional but recommended).
    # This also prevents the reliability checker from creating large string objects.
    if enforce_tag_category and "tag" in figures_df.columns and isinstance(figures_df["tag"].dtype, CategoricalDtype):
        cats = figures_df["tag"].cat.categories
        arcs = filter_arcs_to_tags(arcs, cats)
        arcs = align_arc_tag_dtype(arcs, cats)  # keeps from/to categorical and aligned

    # --- Pass 1: reliability filter on original arcs ---
    if validate_arcs:
        arcs, stats1 = filter_unreliable_arcs(
            figures_df=figures_df,
            sub_df=sub_df,
            arcs_all_years_df=arcs,
            value_col=reliability_value_col,
            min_tests_per_equation=min_tests_per_equation,
            max_fail_rate=max_fail_rate,
            rtol=rtol,
            atol=atol,
            require_full_sources=require_full_sources,
            logger=logger,
        )
        if logger is not None:
            logger.info(
                f"Arc reliability pass1: kept_arcs={len(arcs)}, "
                f"equations_tested={len(stats1)}"
            )

    # --- Expansion: generate rearranged equations (artificial arcs) ---
    if expand_arcs:
        # Expansion expects statement to exist (you confirmed it does in source arcs).
        # If the caller passed arcs without statement, fail early (better than silently mis-grouping).
        if "statement" not in arcs.columns:
            raise ValueError("arcs dataframe must include 'statement' column for expansion grouping")

        arcs = expand_arcs_with_single_variable_rearrangements(
            arcs,
            pivot_weights=pivot_weights,
            enforce_signature_unique_seq=enforce_signature_unique_seq,
            renumber_seq=renumber_seq,
        )
        if logger is not None:
            logger.info(f"Arc expansion: expanded_arcs={len(arcs)}")

    # --- Pass 2: reliability filter on expanded arcs ---
    if expand_arcs and revalidate_after_expand:
        arcs, stats2 = filter_unreliable_arcs(
            figures_df=figures_df,
            sub_df=sub_df,
            arcs_all_years_df=arcs,
            value_col=reliability_value_col,
            min_tests_per_equation=min_tests_per_equation,
            max_fail_rate=max_fail_rate,
            rtol=rtol,
            atol=atol,
            require_full_sources=require_full_sources,
            logger=logger,
        )
        if logger is not None:
            logger.info(
                f"Arc reliability pass2: kept_arcs={len(arcs)}, "
                f"equations_tested={len(stats2)}"
            )

    # --- Apply arcs (already implemented) ---
    out = apply_arcs_by_version(
        figures_df=figures_df,
        sub_df=sub_df,
        arcs_df=arcs,
        logger=logger,
        keep_original_first=keep_original_first,
    )

    # Restore tag categorical dtype if present on input
    if enforce_tag_category and isinstance(figures_df["tag"].dtype, CategoricalDtype):
        out["tag"] = pd.Categorical(out["tag"].astype(str), categories=list(figures_df["tag"].cat.categories))

    return out
