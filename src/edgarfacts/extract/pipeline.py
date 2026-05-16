# src/edgarfacts/extract/pipeline.py
"""
Main extraction pipeline for edgarfacts.

Public entry point:
- extract_submissions_and_facts(logger, debug_mode=False, prev_df=None, prev_sub=None)

This orchestrates:
- ticker mapping (ticker.txt)
- periods (financial-statement-data-sets page)
- tag list (FASB us-gaap taxonomy packages)
- facts from companyfacts.zip
- submissions from quarterly FSD zips + bulk submissions.zip
- version enrichment + amendment flags
- version repair
- fallback extraction of missing figures from individual filings

Important:
- Do NOT change output dataframe schemas.
- Use datetime64[s] everywhere.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from edgarfacts.fetching import URLFetcher
from edgarfacts.extract.tickers import read_tickers
from edgarfacts.extract.periods import read_periods
from edgarfacts.extract.tags import read_tags
from edgarfacts.extract.facts_companyfacts import load_facts
from edgarfacts.extract.submissions_fsd import read_submissions_parallel
from edgarfacts.extract.submissions_bulk import (
    read_submissions_2,
    update_version_info,
    set_amended_flag,
    read_missing_submissions,
    repair_version,
)
from edgarfacts.extract.missing_figures import read_missing_figures, read_missing_figures_2


def determine_delta_first_period(prev_sub: pd.DataFrame) -> tuple[int, int]:
    """Return the first quarter delta mode should consider from prior submissions.

    Delta extraction re-processes a small lookback window relative to the most
    recent known filing acceptance timestamp.  The first supported quarter is
    two quarters before that latest accepted date.
    """
    accepted = pd.to_datetime(prev_sub["accepted"], errors="coerce")
    latest_accepted = accepted.max()

    if pd.isna(latest_accepted):
        raise ValueError("prev_sub must contain at least one valid accepted timestamp")

    first_period_date = pd.Timestamp(latest_accepted) - pd.DateOffset(months=6)
    first_quarter = (int(first_period_date.month) - 1) // 3 + 1

    return int(first_period_date.year), int(first_quarter)


def normalize_submission_dtypes(sub: pd.DataFrame) -> pd.DataFrame:
    """Normalize submission columns to the stable full-pipeline dtypes."""
    sub = sub.copy()

    sub["cik"] = pd.to_numeric(sub["cik"], errors="raise").astype("int64")
    sub["sic"] = pd.to_numeric(sub["sic"], errors="raise").astype("int64")
    sub["adsh"] = pd.to_numeric(sub["adsh"], errors="raise").astype("int64")
    sub["version"] = pd.to_numeric(sub["version"], errors="raise").astype("int64")

    if "amendment_adsh" in sub.columns:
        sub["amendment_adsh"] = (
            pd.to_numeric(sub["amendment_adsh"], errors="raise")
            .fillna(0)
            .astype("int64")
        )

    if "ticker" in sub.columns:
        sub["ticker"] = sub["ticker"].astype("category")

    if sub["period"].dtype != np.dtype("datetime64[s]"):
        sub["period"] = sub["period"].astype("datetime64[s]")

    if sub["accepted"].dtype != np.dtype("datetime64[s]"):
        sub["accepted"] = sub["accepted"].astype("datetime64[s]")

    return sub


def normalize_facts_dtypes(df: pd.DataFrame, tag_list: np.ndarray) -> pd.DataFrame:
    """Normalize facts columns to the stable full-pipeline dtypes."""
    df = df.copy()

    df["adsh"] = pd.to_numeric(df["adsh"], errors="raise").astype("int64")
    df["tag"] = pd.Categorical(df["tag"], categories=tag_list)

    if df["start"].dtype != np.dtype("datetime64[s]"):
        df["start"] = df["start"].astype("datetime64[s]")

    if df["end"].dtype != np.dtype("datetime64[s]"):
        df["end"] = df["end"].astype("datetime64[s]")

    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)

    return df


def extract_submissions_and_facts_internal(fetcher: URLFetcher, logger, debug_mode: bool = False):
    # 1) Tickers
    tickers = read_tickers(fetcher)
    logger.info(f"{len(tickers)} tickers loaded")

    # 2) Periods
    period_arr = read_periods(fetcher)
    logger.info(f"Last available period is {period_arr[-1]}")

    # DEBUG MODE (kept as in original script)
    if debug_mode:
        tickers = tickers.query("ticker=='msft' or ticker=='nvda'")
        period_arr = period_arr[-2:]
    # END DEBUG MODE

    # 3) Tags
    tag_list = read_tags(fetcher)

    # 4) Valid CIKs
    valid_ciks = tickers.cik.unique()

    # 5) Facts
    df = load_facts(valid_ciks, tag_list, fetcher, logger)
    logger.info("Company facts loaded")

    # 6) Submissions from quarterly FSD zips
    sub = read_submissions_parallel(period_arr, fetcher, valid_ciks, logger)

    # 7) Submissions from bulk submissions.zip (more recent / daily updated)
    sub2 = read_submissions_2(valid_ciks, fetcher, logger)

    # Remove from sub2 those already having version info in sub
    sub2 = sub2[~sub2["adsh"].isin(sub[(sub["version"] != 0)]["adsh"])]
    logger.info("Submissions loaded")

    # Combine with version-less rows from sub (for later version enrichment)
    sub2 = pd.concat(
        (
            sub2,
            sub[(sub["version"] == 0) & ~sub["adsh"].isin(sub2["adsh"])].drop(columns="version"),
        )
    )

    # Keep only those not already covered by versioned sub and present in facts
    sub2 = sub2[
        ~sub2["adsh"].isin(sub[sub["version"] != 0]["adsh"])
        & sub2["adsh"].isin(df["adsh"].unique())
    ]

    # Some facts may have no submissions metadata in either source (rare)
    missing_adsh = np.setdiff1d(df["adsh"], np.union1d(sub["adsh"], sub2["adsh"]))
    sub3 = read_missing_submissions(missing_adsh, fetcher)
    if sub3 is not None:
        sub2 = pd.concat((sub2, sub3), ignore_index=True)

    # Enrich version info by scanning primary document content
    sub2 = update_version_info(sub2, fetcher=fetcher, logger=logger)

    # Remove any overlaps and drop 'file' from quarterly submissions
    sub = sub[~sub["adsh"].isin(sub2["adsh"])].drop(columns="file")
    logger.info("Version information loaded")

    # 8) Combine, amendment flags, join tickers
    sub = (
        pd.concat((sub, sub2), ignore_index=True)
        .pipe(set_amended_flag)
        .merge(tickers, how="inner", on="cik")
    )

    # Enforce numeric dtypes (prevent object drift from concatenation)
    sub["cik"] = pd.to_numeric(sub["cik"], errors="raise").astype("int64")
    sub["sic"] = pd.to_numeric(sub["sic"], errors="raise").astype("int64")
    sub["adsh"] = pd.to_numeric(sub["adsh"], errors="raise").astype("int64")
    sub["version"] = pd.to_numeric(sub["version"], errors="raise").astype("int64")
    sub["amendment_adsh"] = pd.to_numeric(sub["amendment_adsh"], errors="raise").astype("int64")

    # Ensure datetime64[s] (defensive; should already be)
    if sub["period"].dtype != np.dtype("datetime64[s]"):
        sub["period"] = sub["period"].astype("datetime64[s]")
    if sub["accepted"].dtype != np.dtype("datetime64[s]"):
        sub["accepted"] = sub["accepted"].astype("datetime64[s]")

    # Keep only submissions that actually have facts.
    sub = normalize_submission_dtypes(sub)
    
    return df, sub


def extract_submissions_and_facts_delta(
    fetcher: URLFetcher,
    logger,
    prev_df: pd.DataFrame,
    prev_sub: pd.DataFrame,
    debug_mode: bool = False,
):
    """Run an incremental extraction using previous facts and submissions."""
    logger.info("Running extraction pipeline in delta mode")
    logger.info(f"{len(prev_sub)} previous submissions provided")
    logger.info(f"{len(prev_df)} previous facts provided")

    # Defensive copies keep caller-owned frames untouched.
    prev_df = prev_df.copy()
    prev_sub = prev_sub.copy()

    # 1) Tickers
    tickers = read_tickers(fetcher)
    logger.info(f"{len(tickers)} tickers loaded")

    if debug_mode:
        tickers = tickers.query("ticker=='msft' or ticker=='nvda'").copy()

    # 2) Tags
    tag_list = read_tags(fetcher)

    # Normalize previous data before comparing ADSH values and period bounds.
    prev_df = normalize_facts_dtypes(prev_df, tag_list)
    prev_sub = normalize_submission_dtypes(prev_sub)

    # 3) Valid CIKs
    valid_ciks = tickers.cik.unique()

    # 4) Current bulk submissions from submissions.zip. This is much cheaper than
    # full companyfacts plus quarterly FSD loading.
    current_bulk_sub = read_submissions_2(valid_ciks, fetcher, logger)
    logger.info(f"{len(current_bulk_sub)} current bulk submissions loaded")

    # Delta mode uses a rolling historical coverage boundary based on the
    # newest previously accepted submission. Look back two quarters so late
    # amendments and recently added submissions are evaluated against a small
    # stable overlap window.
    first_year, first_quarter = determine_delta_first_period(prev_sub)

    # Convert first supported quarter into timestamp.
    # Example:
    # (2008, 1) -> 2008-01-01
    # (2008, 2) -> 2008-04-01
    min_supported_period = pd.Timestamp(
        year=int(first_year),
        month=(int(first_quarter) - 1) * 3 + 1,
        day=1,
    ).to_datetime64()

    logger.info(
        f"Delta minimum supported period from latest previous accepted date: "
        f"{min_supported_period}"
    )

    logger.info(
        f"Previous submission period range: "
        f"{prev_sub['period'].min()} -> {prev_sub['period'].max()}"
    )

    logger.info(
        "Bulk submission period range before filter: "
        f"{current_bulk_sub['period'].min()} -> "
        f"{current_bulk_sub['period'].max()}"
    )

    logger.info(f"Bulk submissions before historical-universe filter: {len(current_bulk_sub)}")

    # Remove submissions older than the delta lookback boundary.
    current_bulk_sub = current_bulk_sub[current_bulk_sub["period"] >= min_supported_period].copy()

    logger.info(
        "Bulk submission period range after filter: "
        f"{current_bulk_sub['period'].min()} -> "
        f"{current_bulk_sub['period'].max()}"
    )

    logger.info(f"Bulk submissions after historical-universe filter: {len(current_bulk_sub)}")

    # 5) Find accession numbers not already present in historical facts or submissions.
    known_adsh = np.union1d(
        prev_df["adsh"].dropna().astype("int64").unique(),
        prev_sub["adsh"].dropna().astype("int64").unique(),
    )

    delta_sub_raw = current_bulk_sub[~current_bulk_sub["adsh"].isin(known_adsh)].copy()
    logger.info(f"{len(delta_sub_raw)} new submissions found")

    if len(delta_sub_raw) == 0:
        logger.info("No new submissions found. Returning previous data.")
        sub = repair_version(set_amended_flag(prev_sub))
        sub = normalize_submission_dtypes(sub)
        logger.info(f"Final submissions count: {len(sub)}")
        logger.info(f"Final facts count: {len(prev_df)}")
        return prev_df.reset_index(drop=True), sub.reset_index(drop=True)

    # 6) Enrich version info only for new submissions.
    delta_sub = update_version_info(delta_sub_raw, fetcher=fetcher, logger=logger)

    # 7) Join ticker information.
    delta_sub = delta_sub.merge(tickers, how="inner", on="cik")

    # 8) Normalize dtypes for new submissions before fact extraction.
    delta_sub = normalize_submission_dtypes(delta_sub)

    logger.info(
        f"Delta submissions with unresolved version before fact extraction: "
        f"{(delta_sub['version'] == 0).sum()}"
    )

    # 9) Extract facts only for the newly discovered submissions. Do not call
    # read_missing_figures_2 here because it scans all historical submissions
    # without facts.
    delta_df = read_missing_figures(
        sub=delta_sub,
        tag_list=tag_list,
        fetcher=fetcher,
        logger=logger,
    )

    # 10) Keep only delta submissions that actually produced facts.
    if delta_df is not None and len(delta_df) > 0:
        logger.info(f"{len(delta_df)} new facts extracted")

        valid_delta_adsh = delta_df["adsh"].dropna().astype("int64").unique()

        before_fact_filter = len(delta_sub)
        delta_sub = delta_sub[delta_sub["adsh"].isin(valid_delta_adsh)].copy()
        after_fact_filter = len(delta_sub)

        logger.info(
            f"Delta submissions after fact filter: "
            f"{after_fact_filter} of {before_fact_filter}"
        )

        df = pd.concat([prev_df, delta_df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["adsh", "tag", "start", "end", "value"],
            keep="last",
        )
        df = normalize_facts_dtypes(df, tag_list)
    else:
        logger.info("No new facts extracted for delta submissions")
        delta_sub = delta_sub.iloc[0:0].copy()
        df = normalize_facts_dtypes(prev_df, tag_list)

    # 11) Combine with previous submissions only after removing delta submissions
    # that did not produce facts.
    sub = pd.concat([prev_sub, delta_sub], ignore_index=True)
    sub = sub.drop_duplicates(subset=["adsh"], keep="last")

    # Recompute global amendment/version state because a new amendment can affect old rows.
    sub = set_amended_flag(sub)
    sub = repair_version(sub)
    sub = normalize_submission_dtypes(sub)

    logger.info(f"Final facts count: {len(df)}")
    logger.info(f"Final submissions count before final fact filter: {len(sub)}")

    # Keep only submissions that actually have facts.
    sub = normalize_submission_dtypes(sub)

    logger.info(f"Final submissions count after final fact filter: {len(sub)}")

    return df.reset_index(drop=True), sub.reset_index(drop=True)


def extract_submissions_and_facts(
    logger,
    debug_mode: bool = False,
    prev_df: Optional[pd.DataFrame] = None,
    prev_sub: Optional[pd.DataFrame] = None,
):
    """
    Public pipeline entry point.

    Parameters
    ----------
    logger:
        Logger instance (use edgarfacts.get_logger()).
    debug_mode:
        If True, run a reduced extraction for development/testing.
    prev_df:
        Previous facts dataframe. If provided with ``prev_sub``, delta mode is used.
    prev_sub:
        Previous submissions dataframe. If provided with ``prev_df``, delta mode is used.

    Returns
    -------
    (df, sub)
        df: facts dataframe (adsh, tag, start, end, value)
        sub: submissions dataframe (10 columns, incl. ticker categorical)
    """
    fetcher = URLFetcher(logger)

    if prev_df is None or prev_sub is None:
        logger.info("Running extraction pipeline in full mode")
        df, sub = extract_submissions_and_facts_internal(fetcher, logger, debug_mode=debug_mode)

        # Repair versions and pull missing figures (fallback)
        sub = repair_version(sub)
        df = read_missing_figures_2(fetcher, logger, df, sub).reset_index(drop=True)
        logger.info(f"Final submissions count: {len(sub)}")
        logger.info(f"Final facts count: {len(df)}")

        return df, sub

    return extract_submissions_and_facts_delta(
        fetcher=fetcher,
        logger=logger,
        prev_df=prev_df,
        prev_sub=prev_sub,
        debug_mode=debug_mode,
    )
