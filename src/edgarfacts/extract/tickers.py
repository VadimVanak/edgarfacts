# src/edgarfacts/extract/tickers.py
"""
Ticker ↔ CIK mapping utilities.

This module downloads and prepares the SEC-maintained mapping
between ticker symbols and Central Index Keys (CIKs).
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from edgarfacts.fetching import URLFetcher


_TICKER_URL = "https://www.sec.gov/include/ticker.txt"


def read_tickers(fetcher: URLFetcher) -> pd.DataFrame:
    """
    Download and return the ticker-to-CIK mapping from sec.gov.

    Notes
    -----
    - Each CIK is unique; duplicate tickers are dropped by CIK.
    - `cik` is returned as uint32 for memory efficiency.
    - `ticker` is returned as pandas Categorical.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - ticker (category)
        - cik (uint32)
    """
    with fetcher.fetch(_TICKER_URL) as resp:
        tickers = pd.read_csv(
            BytesIO(resp.read()),
            sep="\t",
            names=["ticker", "cik"],
        )

    # Drop missing tickers and ensure unique CIKs
    tickers = (
        tickers[tickers["ticker"].notnull()]
        .drop_duplicates(subset="cik")
        .reset_index(drop=True)
    )

    # Enforce dtypes (must not change output structure)
    tickers["cik"] = tickers["cik"].astype(np.uint32)
    tickers["ticker"] = pd.Categorical(tickers["ticker"])

    return tickers
