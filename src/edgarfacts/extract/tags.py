# src/edgarfacts/extract/tags.py
"""
US-GAAP / DEI tag discovery utilities.

This module builds a list of XBRL element names (tags) from FASB's
US-GAAP taxonomy packages, restricted to monetary and shares items.

The returned list is used to configure the companyfacts JSON decoder.
"""

from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from edgarfacts.fetching import URLFetcher


def read_tags(fetcher: URLFetcher) -> np.ndarray:
    """
    Read and return the set of XBRL tag names to extract.

    Implementation details mirror the original script:
    - Iterate years 2012..2024 inclusive
    - For years < 2022, taxonomy zip has "-01-31" suffix, else no suffix
    - Parse XSD and collect <xs:element> names where:
        type in {"xbrli:monetaryItemType", "xbrli:sharesItemType"}
        and abstract is null
    - Always include DEI tags:
        "EntityCommonStockSharesOutstanding", "EntityPublicFloat"

    Returns
    -------
    numpy.ndarray
        Unique sorted array of tag names (dtype typically '<U...').
    """
    tag_list = np.array([])

    for year in range(2012, 2025):
        suffix = "-01-31" if year < 2022 else ""
        zip_path = f"https://xbrl.fasb.org/us-gaap/{year}/us-gaap-{year}{suffix}.zip"
        xsd_path = f"us-gaap-{year}{suffix}/elts/us-gaap-{year}{suffix}.xsd"

        with fetcher.fetch(zip_path) as resp:
            zf = ZipFile(BytesIO(resp.read()))

        with zf.open(xsd_path) as x:
            root = ElementTree.parse(x).getroot()

        tag_df = (
            pd.DataFrame(
                [r.attrib for r in root.iter("{http://www.w3.org/2001/XMLSchema}element")]
            )
            .query(
                "type.isin(['xbrli:monetaryItemType','xbrli:sharesItemType']) and abstract.isnull()"
            )
        )

        tag_list = np.union1d(tag_list, tag_df["name"])

    return np.union1d(tag_list, ["EntityCommonStockSharesOutstanding", "EntityPublicFloat"])
