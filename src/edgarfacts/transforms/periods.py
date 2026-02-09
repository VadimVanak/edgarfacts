"""
Reporting period inference and quarterly figure derivation.

This module is optimized for memory:
- avoids wide merges over the full facts table where possible
- uses size() counts instead of counting tags
- removes accidental self-merge blowups
- uses datetime64[s] consistently
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from edgarfacts.transforms import config


def infer_reporting_windows(facts_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer up to four reporting windows (enum 1..4) per adsh.

    Output
    ------
    windows_df: columns ['adsh','enum','start','end'] with enum in {1,2,3,4}
    """
    req_f = {"adsh", "end", "start"}
    req_s = {"adsh", "period"}
    if not req_f.issubset(facts_df.columns):
        raise ValueError(f"facts_df must contain columns {sorted(req_f)}")
    if not req_s.issubset(sub_df.columns):
        raise ValueError(f"sub_df must contain columns {sorted(req_s)}")

    # Minimize columns early
    f = facts_df[["adsh", "start", "end"]].copy()
    f["adsh"] = pd.to_numeric(f["adsh"], errors="raise").astype("int64")
    f["start"] = f["start"].astype(config.DATETIME_DTYPE)
    f["end"] = f["end"].astype(config.DATETIME_DTYPE)

    subp = sub_df[["adsh", "period"]].drop_duplicates().copy()
    subp["adsh"] = pd.to_numeric(subp["adsh"], errors="raise").astype("int64")
    subp["period"] = pd.to_datetime(subp["period"]).astype(config.DATETIME_DTYPE)

    # Count rows per (adsh, end) without dragging 'tag' around
    df1 = (
        f[["adsh", "end"]]
        .groupby(["adsh", "end"], sort=False, observed=True)
        .size()
        .reset_index(name="n")
        .merge(subp, how="left", on="adsh", sort=False)
        .query("end.dt.year<=2100")
    )

    tol = config.PERIOD_MATCH_TOLERANCE_DAYS

    # Pick an end date close to submission period; among candidates pick the most common one (largest n)
    df2 = (
        df1[np.abs((df1["period"] - df1["end"]).dt.days) < tol][["adsh", "end", "n"]]
        .sort_values(by=["adsh", "n"], ascending=[True, False], kind="mergesort")
        .drop_duplicates(subset=["adsh"], keep="first")[["adsh", "end"]]
    )

    # Prior-year end: period close to end shifted by +1 year
    df3 = (
        df1[np.abs((df1["period"] - (df1["end"] + pd.DateOffset(years=1))).dt.days) < tol][["adsh", "end", "n"]]
        .sort_values(by=["adsh", "n"], ascending=[True, False], kind="mergesort")
        .drop_duplicates(subset=["adsh"], keep="first")[["adsh", "end"]]
    )

    # For current end: find the two most frequent (start,end) ranges (enum 1 & 2)
    df4 = (
        f.merge(df2, how="inner", on=["adsh", "end"], sort=False)
        .groupby(["adsh", "end", "start"], sort=False, observed=True)
        .size()
        .reset_index(name="n")
    )

    # Filter valid durations (exclude instants and insane ranges)
    df4 = df4[
        (df4["start"] != df4["end"])
        & (df4["start"] + pd.DateOffset(years=1, days=10) > df4["end"])
    ].copy()

    df4 = df4.sort_values(by=["adsh", "end", "n"], ascending=[True, True, False], kind="mergesort")
    df4["enum"] = df4.groupby(["adsh", "end"], sort=False).cumcount() + 1
    df4 = df4[df4["enum"] <= 2][["adsh", "start", "end", "enum"]].copy()

    # Align prior-year windows (enum 3/4) based on start dates shifted by -1 year
    py_tol = config.PRIOR_YEAR_ALIGNMENT_TOLERANCE_DAYS

    # Precompute per-adsh reference starts
    ref1 = df4[df4["enum"] == 1][["adsh", "start"]].rename(columns={"start": "start_ref"}).copy()
    ref2 = df4[df4["enum"] == 2][["adsh", "start"]].rename(columns={"start": "start_ref"}).copy()

    # Candidate prior-year ranges for (adsh, end) in df3
    py = (
        f.merge(df3, how="inner", on=["adsh", "end"], sort=False)
        .groupby(["adsh", "end", "start"], sort=False, observed=True)
        .size()
        .reset_index(name="n")
    )

    # enum 3: match enum 1 shifted by -1y
    df5 = (
        py.merge(ref1, how="left", on="adsh", sort=False)
        .loc[lambda x: x["start_ref"].notna()]
        .loc[lambda x: np.abs((x["start"] - (x["start_ref"] - pd.DateOffset(years=1))).dt.days) < py_tol]
        .sort_values(by=["adsh", "n"], ascending=[True, False], kind="mergesort")
        .drop_duplicates(subset=["adsh"], keep="first")[["adsh", "start", "end"]]
        .assign(enum=3)
    )

    # enum 4: match enum 2 shifted by -1y
    df6 = (
        py.merge(ref2, how="left", on="adsh", sort=False)
        .loc[lambda x: x["start_ref"].notna()]
        .loc[lambda x: np.abs((x["start"] - (x["start_ref"] - pd.DateOffset(years=1))).dt.days) < py_tol]
        .sort_values(by=["adsh", "n"], ascending=[True, False], kind="mergesort")
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

    windows["adsh"] = windows["adsh"].astype("int64")
    windows["start"] = windows["start"].astype(config.DATETIME_DTYPE)
    windows["end"] = windows["end"].astype(config.DATETIME_DTYPE)
    windows["enum"] = windows["enum"].astype("uint8")

    return windows


def compute_period_values(
    facts_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute value1..value4 for non-instant facts.

    Output
    ------
    period_values_df: ['adsh','tag','value1','value2','value3','value4']
    sub_enriched_df: sub_df + window columns
    """
    # Keep only necessary columns; avoid copying huge df unless needed
    df = facts_df[["adsh", "tag", "start", "end", "value"]].copy()
    df["adsh"] = pd.to_numeric(df["adsh"], errors="raise").astype("int64")
    df["start"] = df["start"].astype(config.DATETIME_DTYPE)
    df["end"] = df["end"].astype(config.DATETIME_DTYPE)

    # Exclude instants early
    df = df[df["start"] != df["end"]]

    # --- Memory-optimized join: semi-join by MultiIndex intersection ---
    # Build windows lookup keyed by (adsh,start,end) -> enum
    w = windows_df[["adsh", "start", "end", "enum"]].copy()
    w["adsh"] = pd.to_numeric(w["adsh"], errors="raise").astype("int64")
    w["start"] = w["start"].astype(config.DATETIME_DTYPE)
    w["end"] = w["end"].astype(config.DATETIME_DTYPE)

    w_idx = w.set_index(["adsh", "start", "end"])
    df_idx = df.set_index(["adsh", "start", "end"])

    common = df_idx.index.intersection(w_idx.index)
    if len(common) == 0:
        # No windows found -> empty values frame + enriched sub with NaNs
        period_values = df.iloc[0:0][["adsh", "tag"]].assign(value1=np.nan, value2=np.nan, value3=np.nan, value4=np.nan)
        sub_enriched = _enrich_sub_with_windows(sub_df, windows_df)
        return period_values[["adsh", "tag", "value1", "value2", "value3", "value4"]], sub_enriched

    dfw = df_idx.loc[common].reset_index()
    dfw = dfw.merge(w.reset_index(drop=True), how="left", on=["adsh", "start", "end"], sort=False)

    # Now dfw is much smaller than a full merge over df
    # Split by enum and create wide values
    e1 = dfw[dfw["enum"] == 1][["adsh", "tag", "value"]].rename(columns={"value": "value1"})
    e2 = dfw[dfw["enum"] == 2][["adsh", "tag", "value"]].rename(columns={"value": "value2"})
    e3 = dfw[dfw["enum"] == 3][["adsh", "tag", "value"]].rename(columns={"value": "value3"})
    e4 = dfw[dfw["enum"] == 4][["adsh", "tag", "value"]].rename(columns={"value": "value4"})

    wide = (
        e1.merge(e2, how="left", on=["adsh", "tag"], sort=False)
          .merge(e3, how="left", on=["adsh", "tag"], sort=False)
          .merge(e4, how="left", on=["adsh", "tag"], sort=False)
          .reset_index(drop=True)
    )

    # NOTE: removed the previous self-merge placeholder.
    # Quarter subtraction logic should be implemented later in a controlled way.

    sub_enriched = _enrich_sub_with_windows(sub_df, windows_df)

    period_values = wide[["adsh", "tag", "value1", "value2", "value3", "value4"]].copy()
    period_values["adsh"] = period_values["adsh"].astype("int64")
    return period_values, sub_enriched


def compute_instant_period_values(
    facts_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute value1..value4 for instant facts (start==end) based on window end dates.

    Output
    ------
    inst_values_df: ['adsh','tag','value1','value2','value3','value4']
    """
    df = facts_df[["adsh", "tag", "start", "end", "value"]].copy()
    df["adsh"] = pd.to_numeric(df["adsh"], errors="raise").astype("int64")
    df["start"] = df["start"].astype(config.DATETIME_DTYPE)
    df["end"] = df["end"].astype(config.DATETIME_DTYPE)

    inst = df[df["start"] == df["end"]][["adsh", "tag", "end", "value"]].copy()

    # Build per-enum end tables without pivot (less memory, avoids wide sparse frame)
    ends = (
        windows_df[["adsh", "enum", "end"]]
        .drop_duplicates()
        .copy()
    )
    ends["adsh"] = pd.to_numeric(ends["adsh"], errors="raise").astype("int64")
    ends["end"] = ends["end"].astype(config.DATETIME_DTYPE)

    end1 = ends[ends["enum"] == 1][["adsh", "end"]].rename(columns={"end": "end_1"})
    end2 = ends[ends["enum"] == 2][["adsh", "end"]].rename(columns={"end": "end_2"})
    end3 = ends[ends["enum"] == 3][["adsh", "end"]].rename(columns={"end": "end_3"})
    end4 = ends[ends["enum"] == 4][["adsh", "end"]].rename(columns={"end": "end_4"})

    out = inst.merge(end1, how="left", on="adsh", sort=False)
    out = out.merge(end2, how="left", on="adsh", sort=False)
    out = out.merge(end3, how="left", on="adsh", sort=False)
    out = out.merge(end4, how="left", on="adsh", sort=False)

    # Join instant values at each end date
    inst_base = inst[["adsh", "tag", "end", "value"]].copy()

    out = out.merge(
        inst_base.rename(columns={"end": "end_1", "value": "value1"}),
        how="left",
        on=["adsh", "tag", "end_1"],
        sort=False,
    )
    out = out.merge(
        inst_base.rename(columns={"end": "end_2", "value": "value2"}),
        how="left",
        on=["adsh", "tag", "end_2"],
        sort=False,
    )
    out = out.merge(
        inst_base.rename(columns={"end": "end_3", "value": "value3"}),
        how="left",
        on=["adsh", "tag", "end_3"],
        sort=False,
    )
    out = out.merge(
        inst_base.rename(columns={"end": "end_4", "value": "value4"}),
        how="left",
        on=["adsh", "tag", "end_4"],
        sort=False,
    )

    return out[["adsh", "tag", "value1", "value2", "value3", "value4"]].copy()


def _enrich_sub_with_windows(sub_df: pd.DataFrame, windows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add start/end columns for enums 1..4 to submissions.
    """
    sub_enriched = sub_df.copy()
    sub_enriched["adsh"] = pd.to_numeric(sub_enriched["adsh"], errors="raise").astype("int64")

    mapping = [
        (1, "start_rep", "end_rep"),
        (2, "start_q", "end_q"),
        (3, "start_rep_py", "end_rep_py"),
        (4, "start_q_py", "end_q_py"),
    ]
    for enum, sname, ename in mapping:
        wi = (
            windows_df[windows_df["enum"] == enum][["adsh", "start", "end"]]
            .drop_duplicates()
            .rename(columns={"start": sname, "end": ename})
            .copy()
        )
        wi["adsh"] = pd.to_numeric(wi["adsh"], errors="raise").astype("int64")
        wi[sname] = wi[sname].astype(config.DATETIME_DTYPE)
        wi[ename] = wi[ename].astype(config.DATETIME_DTYPE)
        sub_enriched = sub_enriched.merge(wi, how="left", on="adsh", sort=False)

    return sub_enriched
