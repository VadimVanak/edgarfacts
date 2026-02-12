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
import gc

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

    # df4 currently has columns: adsh, end, start, n
    # 1) Rank candidates by frequency (n) within each (adsh, end)
    df4 = df4.sort_values(by=["adsh", "end", "n"], ascending=[True, True, False], kind="mergesort")
    df4["rk_n"] = df4.groupby(["adsh", "end"], sort=False).cumcount() + 1
    
    top = df4[df4["rk_n"] == 1][["adsh", "end", "start", "n"]].rename(columns={"start": "start1", "n": "n1"})
    sec = df4[df4["rk_n"] == 2][["adsh", "end", "start", "n"]].rename(columns={"start": "start2", "n": "n2"})
    
    # 2) Decide whether to keep the 2nd window:
    #    keep if it is not "too small" vs the top window
    #    condition to DROP: n2 < 0.3 * n1  (top exceeds second by >70%)
    pairs = top.merge(sec, how="left", on=["adsh", "end"], sort=False)
    
    pairs["keep2"] = pairs["n2"].notna() & (pairs["n2"] >= 0.3 * pairs["n1"])
    
    # Reconstruct the kept candidates
    cand1 = pairs[["adsh", "end", "start1"]].rename(columns={"start1": "start"})
    cand2 = pairs[pairs["keep2"]][["adsh", "end", "start2"]].rename(columns={"start2": "start"})
    
    df4 = pd.concat([cand1, cand2], ignore_index=True)
    
    # 3) Assign enum by start ordering (enum=1 earliest start, enum=2 later start)
    df4 = df4.sort_values(by=["adsh", "end", "start"], ascending=[True, True, True], kind="mergesort")
    df4["enum"] = df4.groupby(["adsh", "end"], sort=False).cumcount() + 1
    df4 = df4[["adsh", "start", "end", "enum"]]
    df4["enum"] = df4["enum"].astype("uint8")

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


def compute_instant_period_values_chunked(
    facts_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    adsh_chunk_size: int = 200_000,   # tune: 100k–500k depending on RAM
) -> pd.DataFrame:
    # Instants only
    inst = facts_df.loc[:, ["adsh", "tag", "start", "end", "value"]]
    inst = inst[inst["start"] == inst["end"]].loc[:, ["adsh", "tag", "end", "value"]]
    if inst.empty:
        return pd.DataFrame(columns=["adsh","tag","value1","value2","value3","value4"])

    # Normalize minimal dtypes
    inst = inst.copy()
    inst["adsh"] = pd.to_numeric(inst["adsh"], errors="raise").astype("int64")
    inst["end"] = pd.to_datetime(inst["end"]).astype(config.DATETIME_DTYPE)
    inst["value"] = pd.to_numeric(inst["value"], errors="coerce").astype("float64")

    # Dedup for smaller index
    inst.sort_values(["adsh","tag","end"], kind="mergesort", inplace=True)
    inst.drop_duplicates(["adsh","tag","end"], keep="last", inplace=True)

    # Build end_1..end_4 mapping (small)
    w = windows_df.loc[:, ["adsh","enum","end"]].copy()
    w["adsh"] = pd.to_numeric(w["adsh"], errors="raise").astype("int64")
    w["end"] = pd.to_datetime(w["end"]).astype(config.DATETIME_DTYPE)
    w["enum"] = pd.to_numeric(w["enum"], errors="raise").astype("uint8")

    ends = w.pivot(index="adsh", columns="enum", values="end")
    for i in (1,2,3,4):
        if i not in ends.columns:
            ends[i] = pd.NaT
    ends = ends[[1,2,3,4]].rename(columns={1:"end_1",2:"end_2",3:"end_3",4:"end_4"}).reset_index()

    # We will iterate over adsh present in inst (not all ends)
    adsh_all = inst["adsh"].unique()
    adsh_all.sort()

    out_chunks = []
    for lo in range(0, len(adsh_all), adsh_chunk_size):
        adsh_chunk = adsh_all[lo:lo+adsh_chunk_size]

        inst_c = inst[inst["adsh"].isin(adsh_chunk)]
        if inst_c.empty:
            continue

        # base rows for this chunk
        base = inst_c.loc[:, ["adsh","tag"]].drop_duplicates()
        base = base.merge(ends, how="left", on="adsh", sort=False)

        # Build chunk index once
        inst_idx = inst_c.set_index(["adsh","tag","end"])["value"].sort_index()

        # Lookup each end_i within chunk
        for i in (1,2,3,4):
            e = base[f"end_{i}"]
            keys = pd.MultiIndex.from_arrays(
                [base["adsh"].to_numpy(), base["tag"].to_numpy(), e.to_numpy()]
            )
            base[f"value{i}"] = inst_idx.reindex(keys).to_numpy()
            del keys
            gc.collect()

        base.drop(columns=["end_1","end_2","end_3","end_4"], inplace=True, errors="ignore")
        out_chunks.append(base.loc[:, ["adsh","tag","value1","value2","value3","value4"]])

        # Free chunk objects
        del inst_c, base, inst_idx
        gc.collect()

    return pd.concat(out_chunks, ignore_index=True) if out_chunks else pd.DataFrame(
        columns=["adsh","tag","value1","value2","value3","value4"]
    )


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
