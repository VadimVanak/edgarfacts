# src/edgarfacts/extract/missing_figures.py
"""
Fallback extraction for missing figures by downloading and parsing individual filings.

This path is used when:
- submission metadata exists but no facts were found in companyfacts.zip for that adsh
- or when recent reports are present in submissions.zip but not in quarterly FSD zips

We:
1) Read FilingSummary.xml to find the primary filing document and relevant taxonomies
2) Download the primary XBRL instance (.xml)
3) Parse contexts/units and extract numeric facts for supported tags
4) Return a facts DataFrame with the same structure as companyfacts loader:
   Columns: ["adsh", "tag", "start", "end", "value"]
   start/end are datetime64[s], tag is categorical, value is float

NOTE: Implementation intentionally mirrors the original script to avoid changing
output structure or semantics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from edgarfacts.fetching import URLFetcher
from edgarfacts.extract.tags import read_tags


def get_submission_attrib(adsh: int, cik: int, fetcher: URLFetcher) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Retrieve primary document filename and relevant taxonomy namespace URLs from FilingSummary.xml.

    Returns
    -------
    (file, ns_list)
        file: primary document name (10-Q / 10-K / amendments), or None
        ns_list: list of taxonomy namespace strings (for DEI / US-GAAP), or None
    """
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh:018d}/FilingSummary.xml"
    resp = fetcher.fetch(url, ignore_exceptions=True)
    if resp is None:
        return None, None

    root = ElementTree.parse(resp).getroot()
    resp.close()

    file = None
    for f in root.findall("InputFiles/File"):
        if "doctype" in f.attrib and f.attrib["doctype"] in ["10-Q", "10-Q/A", "10-K", "10-K/A"]:
            file = f.text
            break

    ns_list: List[str] = []
    for ns in root.findall("BaseTaxonomies/BaseTaxonomy"):
        if "/dei/" in ns.text or "/us-gaap/" in ns.text:
            ns_list.append(ns.text)

    return file, ns_list


def get_context(root: ElementTree.Element, cik: int) -> pd.DataFrame:
    xbrl_ns = {"xbrl": "http://www.xbrl.org/2003/instance"}
    context = root.findall("xbrl:context", namespaces=xbrl_ns)
    cnt_list = []

    for c in context:
        cnt = c.attrib["id"]
        entity_id = int(
            c.find("xbrl:entity", namespaces=xbrl_ns)
            .find("xbrl:identifier", namespaces=xbrl_ns)
            .text
        )
        try:
            start_date = (
                c.find("xbrl:period", namespaces=xbrl_ns)
                .find("xbrl:startDate", namespaces=xbrl_ns)
                .text
            )
            end_date = (
                c.find("xbrl:period", namespaces=xbrl_ns)
                .find("xbrl:endDate", namespaces=xbrl_ns)
                .text
            )
        except Exception:
            end_date = (
                c.find("xbrl:period", namespaces=xbrl_ns)
                .find("xbrl:instant", namespaces=xbrl_ns)
                .text
            )
            start_date = end_date

        cnt_list.append(
            {"contextRef": cnt, "entity": entity_id, "start": start_date, "end": end_date}
        )

    return pd.DataFrame(cnt_list).query("entity==@cik")


def get_units(root: ElementTree.Element) -> List[str]:
    xbrl_ns = {"xbrl": "http://www.xbrl.org/2003/instance"}
    units = root.findall("xbrl:unit", namespaces=xbrl_ns)
    id_list: List[str] = []

    for u in units:
        unit_id = u.attrib["id"]
        try:
            measure = u.find("xbrl:measure", namespaces=xbrl_ns).text
        except Exception:
            continue

        if (
            measure in ["shares", "USD"]
            or measure.endswith(":USD")
            or measure.endswith(":shares")
        ):
            id_list.append(unit_id)

    return id_list


def get_submission(root: ElementTree.Element, cik: int, ns_list: List[str]) -> pd.DataFrame:
    df_list: List[pd.DataFrame] = []

    for ns in ns_list:
        df = pd.DataFrame(
            [
                r.attrib | {"tag": r.tag} | {"value": r.text}
                for r in root.findall("ns:*", namespaces={"ns": ns})
            ]
        )
        if len(df) == 0:
            continue
        df["tag"] = df["tag"].replace("{" + ns + "}", "", regex=True)
        df_list.append(df)

    df = pd.concat(df_list)
    df["value"] = df["value"].fillna(0.0)

    if "unitRef" in df.columns:
        df = df[df["unitRef"].isin(get_units(root))]

    return df.merge(get_context(root, cik))[["tag", "start", "end", "value"]].copy()


def read_missing_figures(
    sub: pd.DataFrame,
    tag_list: np.ndarray,
    fetcher: URLFetcher,
    logger,
) -> Optional[pd.DataFrame]:
    """
    Extract missing figures for each submission in `sub`.

    Returns
    -------
    pandas.DataFrame | None
        Columns: ["tag", "start", "end", "value", "adsh"] (then normalized later)
    """
    df_list: List[pd.DataFrame] = []

    for index, row in sub.iterrows():
        cik, adsh = int(row["cik"]), int(row["adsh"])
        file, ns_list = get_submission_attrib(adsh, cik, fetcher)

        if file is None:
            logger.warning(f"No main submission file found for submission {adsh}")
            continue

        if not file.endswith(".xml"):
            file = "_htm".join(file.rsplit(".htm", 1)) + ".xml"

        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh:018d}/{file}"
        resp = fetcher.fetch(url, ignore_exceptions=True)
        if resp is None:
            continue

        root = ElementTree.parse(resp).getroot()
        resp.close()

        df = get_submission(root, cik, ns_list).assign(adsh=adsh)
        df_list.append(df)

    if len(df_list) == 0:
        return None

    df = pd.concat(df_list, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[~df["value"].isna() & df["tag"].isin(tag_list)]
    df["start"] = df["start"].astype("datetime64[s]")
    df["end"] = df["end"].astype("datetime64[s]")
    df["tag"] = pd.Categorical(df["tag"], categories=tag_list)

    # Reorder to match the standard facts schema
    return df[["adsh", "tag", "start", "end", "value"]].copy()


def read_missing_figures_2(fetcher: URLFetcher, logger, df: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """
    Find submissions without figures in `df` and attempt to extract them from individual filings.

    Returns
    -------
    pandas.DataFrame
        Concatenation of df and newly extracted facts (if any).
    """
    tag_list = read_tags(fetcher)

    sub_no_figures = sub[~sub["adsh"].isin(df["adsh"])]
    logger.info(f"There are {len(sub_no_figures)} reports without figures.")
    logger.info("Loading missing figures.")

    df2 = read_missing_figures(sub_no_figures, tag_list, fetcher, logger)
    logger.info("Additional figures loaded.")

    return pd.concat((df, df2), ignore_index=True) if (df2 is not None) else df
