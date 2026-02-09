"""
Download and parse US-GAAP taxonomy calculation linkbases into arc tables.

This module focuses on calculation linkbases ("cal") for selected statements
and returns a single normalized arcs DataFrame.

Output schema (stable for internal use):
- version: int (taxonomy year)
- statement: str (statement role code)
- seq: int (topological-ish ordering for iterative computation)
- from: str (source tag)
- to: str (target tag)
- weight: float
"""

from __future__ import annotations

from io import BytesIO
from multiprocessing.pool import Pool
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from urllib.request import Request, urlopen
from zipfile import ZipFile
from xml.etree import ElementTree

from .paths import construct_paths


_STATEMENTS_DEFAULT: List[str] = ["scf-indir", "sfp-cls", "soi", "sheci", "soc"]


def calculation_to_dataframe(root: ElementTree.Element) -> pd.DataFrame:
    """
    Convert a calculation linkbase XML root into a DataFrame of arcs.

    Returns columns: ['weight','from','to'] with cleaned identifiers.
    """
    return (
        pd.DataFrame(
            [r.attrib for r in root.iter("{http://www.xbrl.org/2003/linkbase}calculationArc")]
        )
        .rename(
            columns={
                "{http://www.w3.org/1999/xlink}from": "from",
                "{http://www.w3.org/1999/xlink}to": "to",
            }
        )[["weight", "from", "to"]]
        .replace(
            [
                "loc_",
                "http://fasb.org/us-gaap/role/statement/",
                "http://xbrl.us/us-gaap/role/statement/",
            ],
            "",
            regex=True,
        )
        .assign(weight=lambda x: x["weight"].astype("float"))
    )


def assign_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a sequence number to arcs to enable iterative computation.

    This is a heuristic ordering based on dependencies between 'from' and 'to'.
    """
    df = df.copy()
    df["seq"] = 0
    for seq in range(100):
        missing_tags = df[df["to"].isin(df.query("seq>=@seq")["from"])]["from"].unique()
        if len(missing_tags) == 0:
            break
        df.loc[df["from"].isin(missing_tags), "seq"] += 1
    return df


def read_taxonomy_arcs(year: int, statements: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Download and parse taxonomy calculation arcs for a given taxonomy year.

    Parameters
    ----------
    year:
        Taxonomy year.
    statements:
        List of statement role codes to include. Defaults to a standard set.

    Returns
    -------
    arcs_df:
        Columns: ['weight','from','to','statement','seq','version']
    """
    statements = _STATEMENTS_DEFAULT if statements is None else statements

    zip_path, xml_path_template = construct_paths(year)

    # Download taxonomy zip
    resp = urlopen(Request(zip_path))
    zf = ZipFile(BytesIO(resp.read()))

    arc_list: List[pd.DataFrame] = []
    for stmt in statements:
        # Read calculation linkbase for the statement
        with zf.open(xml_path_template % (stmt, "cal")) as x:
            root = ElementTree.parse(x).getroot()

        df = calculation_to_dataframe(root).assign(statement=stmt).pipe(assign_sequence)
        arc_list.append(df)

    arcs = pd.concat(arc_list, ignore_index=True).assign(version=year)
    return arcs[["version", "statement", "seq", "from", "to", "weight"]]


def _read_taxonomy_arcs_safe(args):
    year, statements = args
    # Keep wrapper top-level for multiprocessing pickling
    return read_taxonomy_arcs(year, statements=statements)


def read_taxonomy_arcs_many(
    years: Iterable[int],
    *,
    statements: Optional[List[str]] = None,
    workers: Optional[int] = None,
    use_process_pool: bool = True,
) -> pd.DataFrame:
    """
    Download and parse taxonomy arcs for multiple years.

    Parameters
    ----------
    years:
        Iterable of years.
    statements:
        Statement role codes to include.
    workers:
        Pool size (None = default).
    use_process_pool:
        If True, use multiprocessing Pool. If False, run sequentially.

    Returns
    -------
    arcs_all:
        Concatenated arcs DataFrame with columns:
        ['version','statement','seq','from','to','weight']
    """
    years = list(years)

    if not use_process_pool or len(years) <= 1:
        frames = [read_taxonomy_arcs(y, statements=statements) for y in years]
        return pd.concat(frames, ignore_index=True)

    pool = Pool(processes=workers)
    try:
        results = pool.map(_read_taxonomy_arcs_safe, [(y, statements) for y in years])
    finally:
        pool.close()
        pool.join()

    return pd.concat(results, ignore_index=True)
