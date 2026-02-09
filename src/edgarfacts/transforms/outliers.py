# src/edgarfacts/transforms/compute/outliers.py
"""
Outlier detection and correction for raw SEC company facts.

Purpose
-------
The SEC company facts feed occasionally contains values that are incorrectly
scaled (typically because filers misuse XBRL attributes like `decimals` as if
it controlled scaling). This module detects common scale mistakes by comparing
nearby values for the same (cik, tag) and corrects values by multiplying them
by one of {1e-6, 1e-3, 1e3, 1e6}.

Design
------
- Operates on the raw facts schema: ['adsh','tag','start','end','value']
- Uses submissions to attach CIK: sub_df[['adsh','cik']]
- Computes helper column `value_adj` for comparability across different period lengths
- Runs correction per CIK, optionally in parallel
- Returns a corrected facts DataFrame with the SAME schema as the input facts
  (no extra columns are kept)

Notes
-----
This logic is intentionally close to the original script to preserve behavior,
but it is structured for testability and explicit contracts.
"""

from __future__ import annotations

from functools import partial
from multiprocessing.pool import Pool
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from edgarfacts.transforms import config


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------


def attach_cik(facts_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach `cik` to facts_df via adsh.

    Inputs
    ------
    facts_df: columns ['adsh','tag','start','end','value']
    sub_df: must include ['adsh','cik']

    Output
    ------
    facts_df with extra column 'cik' (int64)
    """
    df = facts_df[["tag", "adsh", "start", "end", "value"]].merge(
        sub_df[["adsh", "cik"]], how="inner", on="adsh"
    )
    df["cik"] = pd.to_numeric(df["cik"], errors="raise").astype("int64")
    return df


def compute_value_adj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute value_adj used for outlier detection.

    value_adj approximates a quarterly-scale magnitude:
    - instant values (end==start) keep original magnitude
    - otherwise normalize to 90 days, but never normalize with fewer than 90 days

    Adds column: 'value_adj' (float64)
    """
    df = df.copy()
    days = (df["end"] - df["start"]).dt.days

    denom = np.where(days > config.MIN_PERIOD_DAYS_FOR_NORMALIZATION, days, config.MIN_PERIOD_DAYS_FOR_NORMALIZATION)
    df["value_adj"] = np.abs(
        np.where(
            df["end"] == df["start"],
            df["value"],
            df["value"] / denom * config.MIN_PERIOD_DAYS_FOR_NORMALIZATION,
        )
    ).astype("float64")
    return df


# ---------------------------------------------------------------------
# Core outlier logic (per CIK)
# ---------------------------------------------------------------------


def _remove_outliers_one_group(inp: Tuple[int, pd.DataFrame], logger) -> Tuple[pd.DataFrame, int]:
    """
    Correct outliers for one CIK group.

    Input
    -----
    (i, df_cik):
        i: group index for logging
        df_cik: columns ['adsh','tag','start','end','value','cik','value_adj']

    Returns
    -------
    (df_fixed, n_outliers)
        df_fixed: same columns as df_cik (value/value_adj corrected)
        n_outliers: total corrected rows (counted as number of matched outlier rows)
    """
    i, df = inp
    if i % 100 == 0:
        logger.info(f"Processing ticker #{i}")

    n_outliers = 0

    # iterative correction: once corrected, neighbors can become detectable
    while True:
        # Median value_adj per (cik, tag) used for robustness
        med = (
            df.groupby(["cik", "tag"], observed=True, as_index=False)["value_adj"]
            .median()
            .rename(columns={"value_adj": "median"})
        )
        df1 = df.merge(med, how="left", on=["cik", "tag"])

        # Pairwise comparisons for same (cik, tag)
        df2 = df1.merge(
            df[["tag", "cik", "end", "value_adj"]],
            on=["cik", "tag"],
            suffixes=["", "_y"],
        )

        # Only compare within 120 days
        df2 = df2[np.abs((df2["end_y"] - df2["end"]).dt.days) <= 120]

        # Common scaling mistakes:
        # multiplied by 1e3 / 1e6 or divided by 1e3 / 1e6
        v = df2["value"].astype("float64")
        m = df2["median"].astype("float64")
        vy = df2["value_adj_y"].astype("float64")

        is_outlier1 = (
            (v != 0)
            & (np.abs(v) > np.abs(m) * 100)
            & (np.abs(v) > np.abs(vy) * 965)
            & (np.abs(v) < np.abs(vy) * 1036)
        )
        is_outlier1 |= (
            (v != 0)
            & (np.abs(v) > np.abs(m) * 100)
            & (np.abs(v) > np.abs(vy) * 950)
            & (np.abs(v) < np.abs(vy) * 1053)
            & (np.abs(v) > 3.2e8)
        )
        is_outlier1 |= (
            (v != 0)
            & (np.abs(v) > np.abs(m) * 100)
            & (np.abs(v) > np.abs(vy) * 750)
            & (np.abs(v) < np.abs(vy) * 1333)
            & (np.abs(v) > 63e9)
        )

        is_outlier2 = (
            (v != 0)
            & (np.abs(v) > np.abs(m) * 10000)
            & (np.abs(v) > np.abs(vy) * 750000)
            & (np.abs(v) < np.abs(vy) * 1333000)
        )

        is_outlier3 = (
            (v != 0)
            & (np.abs(v) < np.abs(m) / 100)
            & (np.abs(v) > np.abs(vy) / 1036)
            & (np.abs(v) < np.abs(vy) / 965)
        )
        is_outlier3 |= (
            (v != 0)
            & (np.abs(v) < np.abs(m) / 100)
            & (np.abs(v) > np.abs(vy) / 1053)
            & (np.abs(v) < np.abs(vy) / 950)
            & (np.abs(vy) > 3.2e8)
        )
        is_outlier3 |= (
            (v != 0)
            & (np.abs(v) < np.abs(m) / 100)
            & (np.abs(v) > np.abs(vy) / 1333)
            & (np.abs(v) < np.abs(vy) / 750)
            & (np.abs(vy) > 63e9)
        )

        is_outlier4 = (
            (v != 0)
            & (np.abs(v) < np.abs(m) / 10000)
            & (np.abs(v) > np.abs(vy) / 1333000)
            & (np.abs(v) < np.abs(vy) / 750000)
        )

        # Remove conflicts
        is_outlier1 = is_outlier1 & ~is_outlier2
        is_outlier3 = is_outlier3 & ~is_outlier4

        # Build correction table
        corr = pd.concat(
            (
                df2[is_outlier1][df.columns].drop_duplicates().assign(mult=1e-3),
                df2[is_outlier2][df.columns].drop_duplicates().assign(mult=1e-6),
                df2[is_outlier3][df.columns].drop_duplicates().assign(mult=1e3),
                df2[is_outlier4][df.columns].drop_duplicates().assign(mult=1e6),
            ),
            ignore_index=True,
        )

        if corr.empty:
            return df, n_outliers

        n_outliers += len(corr)

        # Apply correction
        df = df.merge(corr, how="left")
        df["mult"] = df["mult"].fillna(1.0)
        df["value"] = df["value"].astype("float64") * df["mult"]
        df["value_adj"] = df["value_adj"].astype("float64") * df["mult"]
        df = df.drop(columns="mult")


def remove_outliers_parallel(
    facts_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    logger,
    *,
    workers: Optional[int] = None,
    use_process_pool: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove/correct outliers in raw facts.

    Inputs
    ------
    facts_df: columns ['adsh','tag','start','end','value']
    sub_df: must include ['adsh','cik']
    logger: logger instance
    workers: process pool size
    use_process_pool: if False, runs sequentially

    Outputs
    -------
    (facts_fixed, n_outliers)
    - facts_fixed: same schema as facts_df (no extra columns)
    - n_outliers: number of corrected rows (sum across CIK groups)
    """
    # Prepare working frame
    df = attach_cik(facts_df, sub_df)
    df = compute_value_adj(df)

    # Group by cik
    groups = [(i, g.copy()) for i, (_, g) in enumerate(df.groupby("cik", sort=False))]

    if not use_process_pool or len(groups) <= 1:
        results = [_remove_outliers_one_group(g, logger=logger) for g in groups]
    else:
        pool = Pool(processes=workers)
        try:
            fn = partial(_remove_outliers_one_group, logger=logger)
            results = pool.map(fn, groups)
        finally:
            pool.close()
            pool.join()

    n_outliers = int(sum(r[1] for r in results))
    logger.info(f"{n_outliers} outliers removed")

    fixed = pd.concat((r[0] for r in results), ignore_index=True)

    # Return to raw facts schema (drop helper cols)
    fixed = fixed.drop(columns=["cik", "value_adj"], errors="ignore")

    # Keep deterministic dtypes/precision
    fixed["adsh"] = pd.to_numeric(fixed["adsh"], errors="raise").astype("int64")
    fixed["value"] = pd.to_numeric(fixed["value"], errors="coerce").astype("float64")
    fixed["start"] = fixed["start"].astype(config.DATETIME_DTYPE)
    fixed["end"] = fixed["end"].astype(config.DATETIME_DTYPE)

    return fixed, n_outliers
