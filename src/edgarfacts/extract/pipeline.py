# src/edgarfacts/extract/pipeline.py
"""
Main extraction pipeline for edgarfacts.

Public entry point:
- extract_submissions_and_facts(logger, debug_mode=False)

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
from edgarfacts.extract.missing_figures import read_missing_figures_2


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

    # Ensure datetime64[s] (defensive; should already be)
    if sub["period"].dtype != np.dtype("datetime64[s]"):
        sub["period"] = sub["period"].astype("datetime64[s]")
    if sub["accepted"].dtype != np.dtype("datetime64[s]"):
        sub["accepted"] = sub["accepted"].astype("datetime64[s]")

    return df, sub


def extract_submissions_and_facts(logger, debug_mode: bool = False):
    """
    Public pipeline entry point.

    Parameters
    ----------
    logger:
        Logger instance (use edgarfacts.get_logger()).
    debug_mode:
        If True, run a reduced extraction for development/testing.

    Returns
    -------
    (df, sub)
        df: facts dataframe (adsh, tag, start, end, value)
        sub: submissions dataframe (10 columns, incl. ticker categorical)
    """
    fetcher = URLFetcher(logger)

    df, sub = extract_submissions_and_facts_internal(fetcher, logger, debug_mode=debug_mode)

    # Repair versions and pull missing figures (fallback)
    sub = repair_version(sub)
    df = read_missing_figures_2(fetcher, logger, df, sub).reset_index(drop=True)

    return df, sub
