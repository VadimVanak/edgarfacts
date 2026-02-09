# src/edgarfacts/extract/facts_companyfacts.py
"""
Company facts extraction from SEC companyfacts.zip.

This module mirrors the original script behavior:
- Builds a msgspec decoder for a dynamic set of US-GAAP tags
- Parses SEC companyfacts JSON into a single facts DataFrame

Output DataFrame structure (must not change):
Columns: ["adsh", "tag", "start", "end", "value"]
Dtypes:
- adsh: int64
- tag: pandas Categorical (categories = tag_list)
- start/end: datetime64[s]
- value: float
"""

from __future__ import annotations

from io import BytesIO
from typing import List, Optional
from zipfile import ZipFile

import msgspec
import numpy as np
import pandas as pd

from edgarfacts.fetching import URLFetcher


# -----------------------------
# msgspec structures (unchanged)
# -----------------------------

class TagItem(msgspec.Struct):
    end: str
    accn: str
    val: float
    form: str
    start: Optional[str] = None

    def to_tuple(self, tag: str):
        return (
            int(self.accn.replace("-", "", 2)),
            tag,
            (
                self.end
                if self.start is None
                else "1900-01-01"
                if self.start[0:4] < "1900"
                else self.start
            ),
            self.end,
            self.val,
        )


class Units(msgspec.Struct):
    USD: List[TagItem] = None
    shares: List[TagItem] = None

    def to_list(self, tag: str):
        l = []
        if self.USD is not None:
            l += [
                t.to_tuple(tag)
                for t in self.USD
                if t.form in ["10-K", "10-Q", "10-K/A", "10-Q/A"]
            ]
        if self.shares is not None:
            l += [
                t.to_tuple(tag)
                for t in self.shares
                if t.form in ["10-K", "10-Q", "10-K/A", "10-Q/A"]
            ]
        return l


class Tag(msgspec.Struct):
    units: Units


class Dei(msgspec.Struct):
    EntityCommonStockSharesOutstanding: Optional[Tag] = None
    EntityPublicFloat: Optional[Tag] = None

    def to_list(self):
        l = []
        if self.EntityCommonStockSharesOutstanding is not None:
            l += self.EntityCommonStockSharesOutstanding.units.to_list(
                "EntityCommonStockSharesOutstanding"
            )
        if self.EntityPublicFloat is not None:
            l += self.EntityPublicFloat.units.to_list("EntityPublicFloat")
        return l


def get_decoder(tag_list: np.ndarray) -> msgspec.json.Decoder:
    """
    Create and return a msgspec decoder for SEC companyfacts JSON, restricted to tag_list.

    Parameters
    ----------
    tag_list:
        Iterable/array of tags to decode under the "us-gaap" section.

    Returns
    -------
    msgspec.json.Decoder
        Decoder for the top-level Figures struct.
    """
    # msgspec defstruct expects a list of (name, type, default)
    tag_types = [(str(t), Tag | None, None) for t in tag_list]
    UsGaap = msgspec.defstruct("UsGaap", tag_types)

    class Facts(msgspec.Struct):
        us_gaap: Optional[UsGaap] = msgspec.field(name="us-gaap", default=None)
        dei: Optional[Dei] = None

        def to_dataframe(self) -> pd.DataFrame:
            l = []
            if self.us_gaap is not None:
                attr = [a for a in dir(self.us_gaap) if not a.startswith("__")]
                for a in attr:
                    item = getattr(self.us_gaap, a)
                    if item is not None:
                        l += item.units.to_list(a)
            if self.dei is not None:
                l += self.dei.to_list()

            df = pd.DataFrame(l, columns=["adsh", "tag", "start", "end", "value"])
            df["adsh"] = df["adsh"].astype(np.int64)
            df["tag"] = pd.Categorical(df["tag"], categories=self._tag_list)
            df["start"] = df["start"].astype("datetime64[s]")
            df["end"] = df["end"].astype("datetime64[s]")
            return df

    Facts._tag_list = tag_list

    class Figures(msgspec.Struct):
        facts: Facts = None

        def to_dataframe(self) -> pd.DataFrame:
            return self.facts.to_dataframe()

    return msgspec.json.Decoder(Figures)


def load_facts(
    valid_ciks: np.ndarray,
    tag_list: np.ndarray,
    fetcher: URLFetcher,
    logger,
) -> pd.DataFrame:
    """
    Extract facts from SEC companyfacts.zip for the given CIKs and tags.

    Parameters
    ----------
    valid_ciks:
        Array of CIKs to include.
    tag_list:
        Array of tag names to decode and keep.
    fetcher:
        URLFetcher instance.
    logger:
        Logger instance.

    Returns
    -------
    pandas.DataFrame
        Columns: ["adsh", "tag", "start", "end", "value"]
        with datetime64[s] for start/end and categorical tag.
    """
    decoder = get_decoder(tag_list)
    df_array: List[pd.DataFrame] = []

    with fetcher.fetch("https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip") as resp:
        zf = ZipFile(BytesIO(resp.read()))

    nlist = [n for n in zf.namelist() if int(n[3:13]) in valid_ciks]

    for i, file in enumerate(nlist):
        if i == 0 or i % 1000 == 999:
            logger.info(f"Processing file {i+1} of {len(nlist)}")
        with zf.open(file) as f:
            content = f.read()

        d = decoder.decode(content)
        if d.facts is not None:
            df = d.to_dataframe()
            if len(df) > 0:
                df_array.append(df)

    return pd.concat(df_array, ignore_index=True)
