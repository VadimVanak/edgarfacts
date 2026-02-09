"""
Calculation linkbase parsing utilities.

These functions are used to convert XBRL calculation linkbase XML files into a
normalized arc table and to assign a sequence value that approximates a valid
calculation order.

The sequence ordering is heuristic: it attempts to ensure that prerequisites
appear earlier when iteratively computing derived tags from base tags.
"""

from __future__ import annotations

import pandas as pd
from xml.etree import ElementTree


def calculation_to_dataframe(root: ElementTree.Element) -> pd.DataFrame:
    """
    Convert a calculation linkbase XML root into a DataFrame of arcs.

    Parameters
    ----------
    root:
        Root element of the parsed XML calculation linkbase.

    Returns
    -------
    DataFrame with columns:
        - weight: float
        - from: str
        - to: str

    Notes
    -----
    This extracts `<link:calculationArc>` elements and normalizes xlink
    attributes to plain 'from'/'to'. It also removes common prefixes used
    in loc identifiers and statement role URIs.
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


def assign_sequence(df: pd.DataFrame, *, max_iter: int = 100) -> pd.DataFrame:
    """
    Assign a sequence number to arcs to allow iterative computation.

    Parameters
    ----------
    df:
        DataFrame containing at least columns ['from','to'].
    max_iter:
        Maximum number of propagation iterations.

    Returns
    -------
    df_out:
        Copy of df with an added integer column 'seq'.

    Method
    ------
    Initialize seq=0 for all arcs. Then repeatedly bump seq for arcs whose
    'from' tag is required by any arc's 'to' tag at the current or later
    sequence level. This creates a coarse ordering usable for forward passes.
    """
    df_out = df.copy()
    df_out["seq"] = 0

    for seq in range(max_iter):
        # Tags that appear as 'from' for any arc in seq>=current and are required as 'to'
        missing_tags = df_out[df_out["to"].isin(df_out.query("seq>=@seq")["from"])]["from"].unique()
        if len(missing_tags) == 0:
            break
        df_out.loc[df_out["from"].isin(missing_tags), "seq"] += 1

    return df_out
