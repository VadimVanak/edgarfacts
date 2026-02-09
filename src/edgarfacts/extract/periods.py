# src/edgarfacts/extract/periods.py
"""
Period discovery utilities.

This module scrapes the SEC page listing available Financial Statement Data Sets
and extracts all available (year, quarter) tuples.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

from edgarfacts.fetching import URLFetcher


_PERIODS_PAGE = "https://www.sec.gov/dera/data/financial-statement-data-sets"


def read_periods(fetcher: URLFetcher) -> List[Tuple[int, int]]:
    """
    Download the SEC financial statement dataset listing page and extract all available periods.

    The SEC hosts datasets as:
      /files/dera/data/financial-statement-data-sets/YYYYqQ.zip

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of (year, quarter) tuples.
        Example: [(2009, 1), (2009, 2), ...]
    """
    with fetcher.fetch(_PERIODS_PAGE) as resp:
        page = resp.read().decode("utf-8")

    matches = re.findall(
        r"\/files\/dera\/data\/financial-statement-data-sets\/(\d{4}q[1-4])\.zip",
        page,
    )
    # np.sort for deterministic order, then convert "YYYYqQ" -> (YYYY, Q)
    return [tuple(map(int, m.split("q"))) for m in np.sort(matches)]
