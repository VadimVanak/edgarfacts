"""
Build the base "wide" figures table from raw SEC company facts.

This module converts raw facts:

    facts_df: ['adsh','tag','start','end','value']

into a wide table:

    figures_df: ['adsh','tag',
                 'reported_figure','quarterly_figure',
                 'reported_figure_py','quarterly_figure_py',
                 'is_computed']

and enriches submissions with inferred reporting windows:

    sub_enriched_df: sub_df + ['start_rep','end_rep','start_q','end_q',
                              'start_rep_py','end_rep_py','start_q_py','end_q_py']

Pipeline (high level)
---------------------
1) Canonicalize amendments and dedupe so amended values override originals
2) Remove contradictory duplicates (same key, different value)
3) Correct common scaling outliers (per CIK)
4) Remove implausibly huge values except a whitelist of tags
5) Infer reporting windows and compute period values:
   - non-instant values -> value1..value4
   - instant values     -> value1..value4 at window end dates
6) Combine period + instant with deterministic priority
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from edgarfacts.transforms import config
from .amendments import canonicalize_and_merge_amendments
from .outliers import remove_outliers_parallel
from .periods import infer_reporting_windows, compute_period_values, compute_instant_period_values

from edgarfacts.transforms.taxonomy.reader import read_taxonomy_arcs_many
from edgarfacts.transforms.compute.arcs_apply import apply_arcs_by_version


def _ensure_facts_schema(facts_df: pd.DataFrame) -> None:
    req = {"adsh", "tag", "start", "end", "value"}
    missing = sorted(req - set(facts_df.columns))
    if missing:
        raise ValueError(f"facts_df missing required columns: {missing}")


def _ensure_sub_schema(sub_df: pd.DataFrame) -> None:
    req = {"adsh", "cik", "period", "accepted", "amendment_adsh", "is_amended"}
    missing = sorted(req - set(sub_df.columns))
    if missing:
        raise ValueError(f"sub_df missing required columns: {missing}")


def _normalize_fact_dtypes(facts_df: pd.DataFrame) -> pd.DataFrame:
    df = facts_df.copy()
    df["adsh"] = pd.to_numeric(df["adsh"], errors="raise").astype("int64")
    # tag kept as-is (often category); avoid forcing to str here
    df["start"] = df["start"].astype(config.DATETIME_DTYPE)
    df["end"] = df["end"].astype(config.DATETIME_DTYPE)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    return df


def _remove_contradicting_values(facts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the same (adsh, tag, start, end) appears with multiple values.
    Keep exact duplicates; drop conflicting duplicates entirely.
    """
    df = facts_df.copy()

    # Keep one copy of identical rows
    df = df.drop_duplicates(subset=["adsh", "tag", "start", "end", "value"])

    # Drop any key that appears with different values (keep=False drops all)
    df = df.drop_duplicates(subset=["adsh", "tag", "start", "end"], keep=False)

    return df


def _remove_huge_values(facts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter extreme absolute values except for a whitelist of tags.

    Memory notes:
    - NEVER cast 'tag' to str for the full column.
    - If tag is categorical, isin() stays compact.
    """
    df = facts_df

    # fast numeric mask; uses ndarray, minimal overhead
    v = df["value"].to_numpy(copy=False)
    mask_value_ok = np.abs(v) <= config.MAX_ABS_FIGURE_VALUE

    # whitelist mask without exploding categories into object strings
    tag_col = df["tag"]
    if pd.api.types.is_categorical_dtype(tag_col):
        # ensure whitelist values are in the same "representation" as categories
        # (categories are strings in your pipeline)
        mask_whitelist = tag_col.isin(config.LARGE_VALUE_TAG_WHITELIST)
    else:
        # do NOT astype(str); just compare directly
        mask_whitelist = tag_col.isin(config.LARGE_VALUE_TAG_WHITELIST)

    keep = mask_value_ok | mask_whitelist.to_numpy(copy=False)

    # boolean indexing returns a view-like slice; copy to detach downstream if needed
    return df.loc[keep].copy()


def build_base_figures(
    logger,
    facts_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    *,
    outlier_workers: Optional[int] = None,
    use_process_pool: bool = True,
    apply_arcs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build base figures (reported/quarterly/prior-year) from raw facts and submissions.

    Returns
    -------
    figures_df:
        ['adsh','tag','reported_figure','quarterly_figure','reported_figure_py','quarterly_figure_py','is_computed']
    sub_enriched_df:
        sub_df + reporting window columns (start/end for 4 enums)
    """
    _ensure_facts_schema(facts_df)
    _ensure_sub_schema(sub_df)

    # Preserve tag categories if categorical
    tag_categories = None
    if "tag" in facts_df.columns and pd.api.types.is_categorical_dtype(facts_df["tag"]):
        tag_categories = facts_df["tag"].cat.categories

    # 0) Normalize dtypes
    facts = _normalize_fact_dtypes(facts_df)

    # 1) Amendments: canonicalize + dedupe to allow amended override
    facts = canonicalize_and_merge_amendments(facts, sub_df)

    # 2) Remove contradictory duplicates
    facts = _remove_contradicting_values(facts)

    # 3) Outlier correction (requires cik in sub_df)
    facts, n_out = remove_outliers_parallel(
        facts, sub_df, logger, workers=outlier_workers, use_process_pool=use_process_pool
    )

    # 4) Remove huge values
    facts = _remove_huge_values(facts)
    logger.info(f"Huge values removed")

    # 5) Infer reporting windows
    windows = infer_reporting_windows(facts, sub_df)

    # 6) Compute non-instant period values and enrich sub
    period_values, sub_enriched = compute_period_values(facts, sub_df, windows)

    # 7) Compute instant period values (start==end) using window end dates
    inst_values = compute_instant_period_values(facts, windows)
    logger.info(f"Period values computed")

    # 8) Combine with deterministic priority: non-instant first, then instants
    # (mirrors original approach)
    combined = (
        pd.concat([period_values.assign(_prio=1), inst_values.assign(_prio=2)], ignore_index=True)
        .sort_values(["adsh", "tag", "_prio"], kind="mergesort")
        .drop_duplicates(subset=["adsh", "tag"], keep="first")
        .drop(columns="_prio")
    )
    logger.info(f"Combine with deterministic priority: non-instant first, then instants - done")

    # 9) Rename to final wide schema + is_computed flag
    figures = combined.rename(
        columns={
            "value1": "reported_figure",
            "value2": "quarterly_figure",
            "value3": "reported_figure_py",
            "value4": "quarterly_figure_py",
        }
    ).copy()

    figures["is_computed"] = False

    # Enforce dtypes
    figures["adsh"] = pd.to_numeric(figures["adsh"], errors="raise").astype("int64")
    figures["reported_figure"] = pd.to_numeric(figures["reported_figure"], errors="coerce").astype("float64")
    figures["quarterly_figure"] = pd.to_numeric(figures["quarterly_figure"], errors="coerce").astype("float64")
    figures["reported_figure_py"] = pd.to_numeric(figures["reported_figure_py"], errors="coerce").astype("float64")
    figures["quarterly_figure_py"] = pd.to_numeric(figures["quarterly_figure_py"], errors="coerce").astype("float64")

    # Ensure window datetimes in sub_enriched are seconds
    for c in [
        "start_rep",
        "end_rep",
        "start_q",
        "end_q",
        "start_rep_py",
        "end_rep_py",
        "start_q_py",
        "end_q_py",
    ]:
        if c in sub_enriched.columns:
            sub_enriched[c] = pd.to_datetime(sub_enriched[c]).astype(config.DATETIME_DTYPE)

    # 10) Apply calculation arcs at the end (fills missing tags, marks is_computed=True)
    if apply_arcs:
        logger.info("Loading US-GAAP calculation arcs")
        arcs = read_taxonomy_arcs_many([2008, 2009] + list(range(2011, 2026)))

        logger.info("Applying calculation arcs")
        figures = apply_arcs_by_version(
            figures_df=figures,
            sub_df=sub_enriched,   # needed for adsh->version mapping
            arcs_df=arcs,
        )
    
    return figures, sub_enriched
