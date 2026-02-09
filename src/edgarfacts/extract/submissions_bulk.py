# src/edgarfacts/extract/submissions_bulk.py
"""
Submission extraction from SEC bulk submissions dataset (submissions.zip)
and version enrichment by reading individual submission files.

This module mirrors the original script behavior and preserves the output
DataFrame structure and datetime64[s] precision.

Key responsibilities:
- Read bulk submissions metadata (submissions.zip)
- Update/repair US-GAAP version info by scanning individual filings
- Identify amended filings and create amendment flags
- Read missing submissions for a set of accession numbers (ADSH)
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import List, Optional

import numpy as np
import pandas as pd
from zipfile import ZipFile
import msgspec

from edgarfacts.fetching import URLFetcher


_ALLOWED_FORMS = ["10-Q", "10-K", "10-Q/A", "10-K/A"]
_SUBMISSIONS_ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"


# -----------------------------
# msgspec structures (unchanged)
# -----------------------------

class Recent(msgspec.Struct):
    form: List[str] = []
    accessionNumber: List[str] = []
    reportDate: List[str] = []
    primaryDocument: List[str] = []
    acceptanceDateTime: List[str] = []


class Filings(msgspec.Struct):
    recent: Recent


class Submissions(msgspec.Struct):
    cik: Optional[str] = None
    sic: Optional[str] = None
    filings: Optional[Filings] = None

    def to_dataframe(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "adsh": np.char.replace(self.filings.recent.accessionNumber, "-", "").astype(
                        "int64"
                    ),
                    "cik": int(self.cik),
                    "sic": 0 if (self.sic is None or len(self.sic) == 0) else int(self.sic),
                    "form": self.filings.recent.form,
                    "period": np.array(self.filings.recent.reportDate).astype("datetime64[D]"),
                    "accepted": np.char.replace(
                        self.filings.recent.acceptanceDateTime, ".000Z", ""
                    ).astype("datetime64[s]"),
                    "file": self.filings.recent.primaryDocument,
                }
            )
            .query("form.isin(@_ALLOWED_FORMS)")
            .copy()
        )


# -----------------------------
# Public-ish helpers used by pipeline
# -----------------------------

def read_submissions_2(valid_ciks: np.ndarray, fetcher: URLFetcher, logger) -> pd.DataFrame:
    """
    Read and process submissions for a list of valid CIKs from submissions.zip.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - adsh (int64)
        - cik (int)
        - sic (int)
        - form
        - period (datetime64[D])
        - accepted (datetime64[s])
        - file
    """
    decoder = msgspec.json.Decoder(Submissions)
    df_array: List[pd.DataFrame] = []

    with fetcher.fetch(_SUBMISSIONS_ZIP_URL) as resp:
        zf = ZipFile(BytesIO(resp.read()))

    nlist = [n for n in zf.namelist() if n[3:13].isdigit() and int(n[3:13]) in valid_ciks]

    for i, file in enumerate(nlist):
        if i == 0 or i % 1000 == 999:
            logger.info(f"Processing file {i+1} of {len(nlist)}")

        with zf.open(file) as f:
            d = decoder.decode(f.read())

        try:
            _ = len(d.filings.recent.accessionNumber)
        except Exception:
            continue

        df_array.append(d.to_dataframe())

    return pd.concat(df_array, ignore_index=True)


def update_version_info(sub2: pd.DataFrame, fetcher: URLFetcher, logger) -> pd.DataFrame:
    """
    For each row in sub2, fetch the primary document and attempt to find "us-gaap/20YY"
    in a streaming manner. Produces a new column 'version' and drops 'file'.

    Returns
    -------
    pandas.DataFrame
        Same as input sub2 without 'file', plus:
        - version (int)
    """
    sub2 = sub2.copy().reset_index(drop=True)
    sub2["version"] = 0

    for index, row in sub2.iterrows():
        if index == 0 or index % 100 == 99:
            logger.info(f"Loading version info {index+1} of {len(sub2)}")

        response = fetcher.fetch(
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{row['cik']}/{row['adsh']:018d}/{row['file']}",
            ignore_exceptions=True,
        )
        if response is None:
            continue

        version = 0
        previous_chunk = ""
        while True:
            chunk = response.read(4096).decode("utf-8")
            if not chunk:
                response.close()
                break

            combined_chunk = previous_chunk + chunk
            match = re.search(r"us-gaap\/(20\d{2})", combined_chunk)
            if match:
                response.close()
                version = int(match.group(1))
                break

            previous_chunk = chunk

        sub2.loc[index, "version"] = version

    return sub2.drop(columns="file")


def set_amended_flag(sub: pd.DataFrame) -> pd.DataFrame:
    """
    For submissions that have been amended by a later submission, set:
    - amendment_adsh: 0 for the latest submission, else latest accession number
    - is_amended: amendment_adsh != 0
    """
    sub = sub.copy()
    sub["amendment_adsh"] = (
        sub.sort_values(by=["cik", "period", "accepted"])
        .groupby(["cik", "period"])["adsh"]
        .transform(lambda x: np.where(x == x.iloc[-1], 0, x.iloc[-1]))
    )
    sub["is_amended"] = sub["amendment_adsh"] != 0
    return sub


def read_missing_submissions(missing_adsh: np.ndarray, fetcher: URLFetcher) -> Optional[pd.DataFrame]:
    """
    Attempt to find missing submission metadata for a list of accession numbers (ADSH)
    by pulling JSON submissions for plausible CIKs.

    Returns
    -------
    pandas.DataFrame | None
        DataFrame of matching rows, or None if nothing found.
    """
    ciks = np.unique(missing_adsh // 100_000_000)
    decoder = msgspec.json.Decoder(Submissions)
    df_list: List[pd.DataFrame] = []

    for c in ciks:
        response = fetcher.fetch(
            f"https://data.sec.gov/submissions/CIK{c:010d}.json",
            ignore_exceptions=True,
        )
        if response is None:
            continue

        d = decoder.decode(response.read())
        response.close()
        if d.filings is None:
            continue
        df_list.append(d.to_dataframe())

    if len(df_list) == 0:
        return None

    out = pd.concat(df_list, ignore_index=True)
    return out.query("adsh.isin(@missing_adsh)").copy()


def repair_version(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Repair missing (0) version values using:
    1) amendment map (amendment_adsh -> version)
    2) forward fill within cik over acceptance date
    3) backward fill within cik over acceptance date

    Keeps output structure the same (only modifies 'version').
    """
    sub = sub.copy()

    version_map = (
        sub[["amendment_adsh", "version"]]
        .query("version > 0 and amendment_adsh > 0")
        .set_index("amendment_adsh")
        .to_dict(orient="dict")["version"]
    )

    sub.loc[sub["version"] == 0, "version"] = (
        sub.loc[sub["version"] == 0, "adsh"].map(version_map).fillna(0).astype(int)
    )

    sub.sort_values(by=["cik", "accepted"], inplace=True)
    sub["version"] = sub["version"].replace(0, np.nan)
    sub["version"] = sub.groupby("cik", as_index=False)["version"].ffill()

    sub.sort_values(by=["cik", "accepted"], ascending=[True, False], inplace=True)
    sub["version"] = sub.groupby("cik", as_index=False)["version"].ffill().astype(int)

    return sub
