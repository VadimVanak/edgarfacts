# src/edgarfacts/extract/submissions_fsd.py
"""
Submission extraction from SEC "Financial Statement Data Sets" (FSD).

This corresponds to the quarterly ZIP files at:
https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{quarter}.zip

We read:
- sub.txt  (submission metadata)
- num.txt  (taxonomy version; used here only to extract us-gaap year)

The functions here intentionally preserve the output DataFrame structure and
datetime64[s] precision used in the original script.
"""

from __future__ import annotations

from functools import partial
from io import BytesIO
from multiprocessing.pool import Pool
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from zipfile import ZipFile

from edgarfacts.fetching import URLFetcher


_FSD_ZIP_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{quarter}.zip"
_ALLOWED_FORMS = ["10-Q", "10-K", "10-Q/A", "10-K/A"]


def read_submissions(
    period: Tuple[int, int],
    fetcher: URLFetcher,
    valid_ciks: np.ndarray,
    logger,
) -> pd.DataFrame:
    """
    Retrieve and process submission information (sub.txt + num.txt) for a specific (year, quarter).

    Parameters
    ----------
    period:
        Tuple (year, quarter).
    fetcher:
        URLFetcher instance.
    valid_ciks:
        Array of CIKs to keep.
    logger:
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Columns (same as original code):
        - adsh (int)
        - cik (int)
        - sic (int)
        - form (object)
        - period (datetime64[s])
        - accepted (datetime64[s])
        - file (object)   [renamed from instance]
        - version (int)   [us-gaap year, 0 if unknown]
    """
    year, quarter = period
    logger.info(f"Loading year {year} quarter {quarter}")

    with fetcher.fetch(_FSD_ZIP_URL.format(year=year, quarter=quarter)) as resp:
        zf = ZipFile(BytesIO(resp.read()))

    sub = pd.read_csv(
        zf.open("sub.txt"),
        sep="\t",
        usecols=["adsh", "cik", "sic", "form", "period", "accepted", "instance"],
    )

    sub = sub[
        sub["form"].isin(_ALLOWED_FORMS)
        & sub["period"].notnull()
        & sub["cik"].isin(valid_ciks)
    ].rename(columns={"instance": "file"})

    sub["adsh"] = sub["adsh"].str.replace("-", "").astype(int)
    sub["sic"] = sub["sic"].astype(float).fillna(0).astype(int)

    num_df = pd.read_csv(zf.open("num.txt"), sep="\t", usecols=["adsh", "version"])
    num_df = num_df[num_df["version"].str.startswith("us-gaap")]
    num_df["version"] = num_df["version"].str.split("/").str[1].str[0:4].astype(int)
    num_df["adsh"] = num_df["adsh"].str.replace("-", "").astype(int)
    num_df = num_df.drop_duplicates()

    sub = sub.merge(num_df, how="left", on="adsh")
    sub["version"] = sub["version"].fillna(0).astype(int)

    # Keep datetime64[s] as requested.
    sub["period"] = pd.to_datetime(sub["period"].astype(int), format="%Y%m%d").astype("datetime64[s]")
    sub["accepted"] = sub["accepted"].astype("datetime64[s]")

    return sub


def read_submissions_parallel(
    period_arr: Iterable[Tuple[int, int]],
    fetcher: URLFetcher,
    valid_ciks: np.ndarray,
    logger,
) -> pd.DataFrame:
    """
    Read submissions for multiple periods in parallel.

    Notes
    -----
    This mirrors the original implementation using multiprocessing.Pool.
    Be aware that in some environments (e.g., Windows spawn) passing complex
    objects between processes can be problematic. We keep this behavior unchanged
    to avoid altering extraction semantics.

    Returns
    -------
    pandas.DataFrame
        Concatenated results of `read_submissions` over all periods.
    """
    partial_read = partial(
        read_submissions,
        fetcher=fetcher,
        valid_ciks=valid_ciks,
        logger=logger,
    )

    pool = Pool()
    try:
        results: List[pd.DataFrame] = pool.map(partial_read, list(period_arr))
    finally:
        pool.close()
        pool.join()

    return pd.concat(results, ignore_index=False)
