# src/edgarfacts/transforms/compute/periods.py
"""
Reporting period inference and quarterly figure derivation.

Goal
----
Given raw facts (adsh, tag, start, end, value) and submissions metadata,
derive a wide table with four values per (adsh, tag):

1) value1: current year-to-date (YTD) reported value (reporting year)
2) value2: current quarter value (reporting period)
3) value3: prior-year YTD value (previous reporting year)
4) value4: prior-year quarter value (previous reporting period)

Additionally, enrich the submissions dataframe with the inferred start/end
dates for these 4 windows for each submission.

Important constraints
---------------------
- All datetime columns should be datetime64[s]
- We do not change the raw extraction schema; this module produces NEW frames:
  - period_values_df (adsh, tag, value1..value4)
  - sub_enriched_df (sub + start/end window columns)

Implementation notes
--------------------
The logic is intentionally close to the original script, but:
- period windows are computed independently (no hidden dependency on prior mutation)
- amendments are assumed to be handled upstream (canonical adsh, deduped facts)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from edgarfacts.transforms import config


def infer_reporting_windows(facts_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer up to four reporting windows (enum 1..4) per adsh.

    Inputs
    ------
    facts_df: columns ['adsh','tag','start','end','value'] (canonicalized, deduped)
    sub_df: must include ['adsh','period'] with period as datetime64[s] or compatible

    Output
    ------
    windows_df: columns ['adsh','enum','start','end'] with enum in {1,2,3,4}
    """
    req_f = {"adsh", "end", "start", "tag"}
    req_s = {"adsh", "period"}
    if not req_f.issubset(facts_df.columns):
        raise ValueError(f"facts_df must contain columns {sorted(req_f)}")
    if not req_s.issubset(sub_df.columns):
        raise ValueError(f"sub_df must contain columns {sorted(req_s)}")

    subp = sub_df[["adsh", "period"]].drop_duplicates().copy()
    subp["period"] = pd.to_datetime(subp["period"]).astype(config.DATETIME_DTYPE)

    # Count occurrences of end dates within each report and join to submission period.
    df1 = (
        facts_df[["adsh", "end", "tag"]]
        .groupby(["adsh", "end"], as_index=False)
        .count()
        .merge(subp, how="left", on="adsh")
        .query("end.dt.year<=2100")  # defensive: ignore junk far-future instants
    )

    tol = config.PERIOD_MATCH_TOLERANCE_DAYS

    # Current reporting end (end date close to submission period)
    df2 = (
        df1[np.abs((df1["period"] - df1["end"]).dt.days) < tol][["adsh", "end", "tag"]]
        .sort_values(by=["adsh", "tag"], ascending=[True, False])
        .drop_duplicates(subset=["adsh"], keep="first")[["adsh", "end"]]
    )

    # Prior-year reporting end (end date close to period shifted by +1y)
    df3 = (
        df1[np.abs((df1["period"] - (df1["end"] + pd.DateOffset(years=1))).dt.days) < tol][["adsh", "end", "tag"]]
        .sort_values(by=["adsh", "tag"], ascending=[True, False])
        .drop_duplicates(subset=["adsh"], keep="first")[["adsh", "end"]]
    )

    # For current end, identify two most frequent (start,end) windows (enum 1 and 2)
    df4 = (
        facts_df[["adsh", "start", "end", "tag"]]
        .merge(df2, how="inner", on=["adsh", "end"])
        .groupby(["adsh", "end", "start"], as_index=False, sort=True)
        .count()
    )

    # Filter valid durations (exclude instants and insane ranges)
    df4 = df4[
        (df4["start"] != df4["end"])
        & (df4["start"] + pd.DateOffset(years=1, days=10) > df4["end"])
    ].copy()

    df4.sort_values(by=["adsh", "tag"], ascending=[True, False], inplace=True)
    df4["enum"] = df4.groupby(["adsh", "end"]).cumcount() + 1
    df4 = (
        df4[df4["enum"] <= 2][["adsh", "start", "end", "enum"]]
        .sort_values(by=["adsh", "end", "start"])
        .copy()
    )

    # Align prior-year windows for enum 3 and 4 (matching starts shifted by -1y)
    py_tol = config.PRIOR_YEAR_ALIGNMENT_TOLERANCE_DAYS

    # enum 3 matches enum 1 shifted by -1y
    df5 = (
        facts_df[["adsh", "start", "end", "tag"]]
        .merge(df3, how="inner", on=["adsh", "end"])
        .groupby(["adsh", "end", "start"], as_index=False, sort=True)
        .count()
        .merge(
            df4[df4["enum"] == 1][["adsh", "start"]].rename(columns={"start": "start_ref"}),
            how="left",
            on="adsh",
        )
    )
    df5 = (
        df5[np.abs((df5["start"] - (df5["start_ref"] - pd.DateOffset(years=1))).dt.days) < py_tol]
        .sort_values(by=["adsh", "tag"], ascending=[True, False])
        .drop_duplicates(subset=["adsh"], keep="first")[["adsh", "start", "end"]]
        .assign(enum=3)
    )

    # enum 4 matches enum 2 shifted by -1y
    df6 = (
        facts_df[["adsh", "start", "end", "tag"]]
        .merge(df3, how="inner", on=["adsh", "end"])
        .groupby(["adsh", "end", "start"], as_index=False, sort=True)
        .count()
        .merge(
            df4[df4["enum"] == 2][["adsh", "start"]].rename(columns={"start": "start_ref"}),
            how="left",
            on="adsh",
        )
    )
    df6 = (
        df6[np.abs((df6["start"] - (df6["start_ref"] - pd.DateOffset(years=1))).dt.days) < py_tol]
        .sort_values(by=["adsh", "tag"], ascending=[True, False])
        .drop_duplicates(subset=["adsh"], keep="first")[["adsh", "start", "end"]]
        .assign(enum=4)
    )

    windows = pd.concat(
        (
            df4[["adsh", "start", "end", "enum"]],
            df5[["adsh", "start", "end", "enum"]],
            df6[["adsh", "start", "end", "enum"]],
        ),
        ignore_index=True,
    )

    windows["start"] = windows["start"].astype(config.DATETIME_DTYPE)
    windows["end"] = windows["end"].astype(config.DATETIME_DTYPE)
    windows["enum"] = windows["enum"].astype("uint8")
    windows["adsh"] = pd.to_numeric(windows["adsh"], errors="raise").astype("int64")

    return windows.rename(columns={"enum": "enum"})


def _previous_non_amended_adsh(sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map each (adsh) to previous non-amended adsh for the same cik, ordered by period.
    """
    req = {"adsh", "cik", "period", "is_amended"}
    if not req.issubset(sub_df.columns):
        raise ValueError(f"sub_df must contain columns {sorted(req)}")

    df2 = (
        sub_df[~sub_df["is_amended"]][["adsh", "cik", "period"]]
        .drop_duplicates()
        .sort_values(by=["cik", "period"])
        .copy()
    )
    df2["adsh_y"] = df2.groupby("cik")["adsh"].shift(1, fill_value=0)
    return df2[["adsh", "adsh_y"]]


def compute_period_values(
    facts_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute value1..value4 for non-instant facts.

    Inputs
    ------
    facts_df: canonicalized, deduped facts (start!=end rows are used)
    sub_df: submissions
    windows_df: output of infer_reporting_windows

    Outputs
    -------
    period_values_df: columns ['adsh','tag','value1','value2','value3','value4']
    sub_enriched_df: sub_df with start/end columns for enums 1..4
    """
    # Ensure datetimes
    df = facts_df.copy()
    df["start"] = df["start"].astype(config.DATETIME_DTYPE)
    df["end"] = df["end"].astype(config.DATETIME_DTYPE)

    # Exclude instants
    df = df[df["start"] != df["end"]].copy()

    # Join windows to facts
    dfw = df.merge(windows_df, how="inner", on=["adsh", "start", "end"])

    # Split by enum and join into wide form for values
    e1 = dfw[dfw["enum"] == 1][["adsh", "tag", "value", "start", "end"]].rename(columns={"value": "value1", "start": "start1", "end": "end1"})
    e2 = dfw[dfw["enum"] == 2][["adsh", "tag", "value", "start", "end"]].rename(columns={"value": "value2", "start": "start2", "end": "end2"})
    e3 = dfw[dfw["enum"] == 3][["adsh", "tag", "value", "start", "end"]].rename(columns={"value": "value3", "start": "start3", "end": "end3"})
    e4 = dfw[dfw["enum"] == 4][["adsh", "tag", "value", "start", "end"]].rename(columns={"value": "value4", "start": "start4", "end": "end4"})

    wide = (
        e1.merge(e2, how="left", on=["adsh", "tag"])
          .merge(e3, how="left", on=["adsh", "tag"])
          .merge(e4, how="left", on=["adsh", "tag"])
          .reset_index(drop=True)
    )

    # Attempt to back out quarterly figures if value2/value4 missing and value1/value3 are accumulated.
    # This mirrors your original logic by looking at previous non-amended report for the same CIK.
    prev_map = _previous_non_amended_adsh(sub_df)
    wide = wide.merge(prev_map, how="left", on="adsh")

    prev = wide.rename(columns={"adsh": "adsh_y"}).merge(
        wide.rename(columns={"adsh": "adsh_y"}),
        on=["adsh_y", "tag"],
        how="left",
        suffixes=["", "_prev"],
    )
    # NOTE: The above is intentionally minimal; in the full refactor we would compute
    # quarter subtraction using the exact window relationships. For now we keep the
    # explicit API contract and leave the detailed subtraction logic to be implemented.

    # Enrich sub_df with the window columns
    sub_enriched = sub_df.copy()
    for i, (sname, ename) in enumerate(
        [("start_rep", "end_rep"), ("start_q", "end_q"), ("start_rep_py", "end_rep_py"), ("start_q_py", "end_q_py")],
        start=1,
    ):
        wi = windows_df[windows_df["enum"] == i][["adsh", "start", "end"]].rename(
            columns={"start": sname, "end": ename}
        )
        sub_enriched = sub_enriched.merge(wi, how="left", on="adsh")

    # Return only required columns
    period_values = wide[["adsh", "tag", "value1", "value2", "value3", "value4"]].copy()
    return period_values, sub_enriched


def compute_instant_period_values(
    facts_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute value1..value4 for instant facts (start==end) based on window end dates.

    Inputs
    ------
    facts_df: canonicalized, deduped facts
    windows_df: output of infer_reporting_windows

    Output
    ------
    inst_values_df: columns ['adsh','tag','value1','value2','value3','value4']
    """
    df = facts_df.copy()
    df["start"] = df["start"].astype(config.DATETIME_DTYPE)
    df["end"] = df["end"].astype(config.DATETIME_DTYPE)

    inst = df[df["start"] == df["end"]][["adsh", "tag", "end", "value"]].copy()

    # Build end-date lookup per enum
    e = windows_df[["adsh", "enum", "end"]].copy()
    e = e.pivot(index="adsh", columns="enum", values="end").reset_index()
    e.columns = ["adsh"] + [f"end_{i}" for i in range(1, len(e.columns))]

    out = inst.merge(e, how="left", on="adsh")

    # For each enum end date, fetch the instant value at that end date
    def _merge_end(col_end: str, out_col: str, base: pd.DataFrame) -> pd.DataFrame:
        return base.merge(
            inst.rename(columns={"end": col_end, "value": out_col})[["adsh", "tag", col_end, out_col]],
            how="left",
            on=["adsh", "tag", col_end],
        )

    out = _merge_end("end_1", "value1", out)
    out = _merge_end("end_2", "value2", out)
    out = _merge_end("end_3", "value3", out)
    out = _merge_end("end_4", "value4", out)

    return out[["adsh", "tag", "value1", "value2", "value3", "value4"]].copy()
