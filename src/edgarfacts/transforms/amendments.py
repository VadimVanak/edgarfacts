"""
Amendment handling for SEC submissions and facts.

Problem
-------
If a submission is amended, the amendment report may contain only the changed
figures. For downstream analysis we want a canonical "latest" report that
combines:
- original report figures, plus
- amended figures overriding originals where present.

Approach
--------
1) Build a canonical ADSH mapping:
   - if amendment_adsh == 0 -> canonical = adsh
   - else canonical = amendment_adsh
   - also resolve chains (A -> B -> C) so canonical points to the terminal amendment

2) Canonicalize facts by rewriting adsh to adsh_canonical.

3) Deduplicate by keeping the latest accepted record when multiple rows exist
   for the same (canonical adsh, tag, start, end). This ensures that amended
   values override older ones.

Contracts
---------
- Input facts schema is preserved: ['adsh','tag','start','end','value']
- Output facts schema is identical.
- No mutation of input frames.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def build_canonical_adsh_map(sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build mapping from original ADSH to canonical ADSH (latest amendment ADSH).

    Inputs
    ------
    sub_df: must include columns ['adsh','amendment_adsh'] (ints)

    Output
    ------
    mapping_df with columns:
      - adsh_original (int64)
      - adsh_canonical (int64)

    Notes
    -----
    - Resolves amendment chains: if A amended by B and B amended by C,
      then A -> C, B -> C, C -> C.
    - If amendment_adsh is 0, the canonical is the adsh itself.
    """
    req = {"adsh", "amendment_adsh"}
    missing = sorted(req - set(sub_df.columns))
    if missing:
        raise ValueError(f"sub_df missing required columns: {missing}")

    df = sub_df[["adsh", "amendment_adsh"]].drop_duplicates().copy()
    df["adsh"] = pd.to_numeric(df["adsh"], errors="raise").astype("int64")
    df["amendment_adsh"] = pd.to_numeric(df["amendment_adsh"], errors="raise").astype("int64")

    # initial canonical: self if 0 else amendment_adsh
    df["adsh_canonical"] = np.where(df["amendment_adsh"] == 0, df["adsh"], df["amendment_adsh"]).astype("int64")

    # resolve chains by iterative pointer chasing
    # build dict of direct pointers adsh -> adsh_canonical
    pointer = df.set_index("adsh")["adsh_canonical"].to_dict()

    def resolve(x: int) -> int:
        seen = set()
        cur = int(x)
        while True:
            nxt = int(pointer.get(cur, cur))
            if nxt == cur:
                return cur
            if nxt in seen:
                # cycle protection: return current best effort
                return nxt
            seen.add(cur)
            cur = nxt

    # Vectorized-ish resolution via map
    df["adsh_canonical"] = df["adsh"].map(resolve).astype("int64")

    return df.rename(columns={"adsh": "adsh_original"})[["adsh_original", "adsh_canonical"]]


def apply_canonical_adsh(facts_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rewrite facts_df.adsh to the canonical ADSH.

    Inputs
    ------
    facts_df: columns ['adsh','tag','start','end','value']
    mapping_df: output of build_canonical_adsh_map

    Output
    ------
    facts_df_canon: same schema as facts_df with adsh rewritten where mapping exists.

    Notes
    -----
    - If an adsh in facts_df is not present in mapping_df, it is kept unchanged.
      (Should be rare if sub_df covers all adsh in facts_df.)
    """
    req_f = {"adsh", "tag", "start", "end", "value"}
    req_m = {"adsh_original", "adsh_canonical"}
    if not req_f.issubset(facts_df.columns):
        raise ValueError(f"facts_df must contain columns {sorted(req_f)}")
    if not req_m.issubset(mapping_df.columns):
        raise ValueError(f"mapping_df must contain columns {sorted(req_m)}")

    df = facts_df.copy()
    df["adsh"] = pd.to_numeric(df["adsh"], errors="raise").astype("int64")

    m = mapping_df.drop_duplicates().copy()
    m["adsh_original"] = pd.to_numeric(m["adsh_original"], errors="raise").astype("int64")
    m["adsh_canonical"] = pd.to_numeric(m["adsh_canonical"], errors="raise").astype("int64")

    df = df.merge(m, how="left", left_on="adsh", right_on="adsh_original")
    df["adsh"] = df["adsh_canonical"].fillna(df["adsh"]).astype("int64")
    df = df.drop(columns=["adsh_original", "adsh_canonical"])

    return df


def dedupe_latest_by_acceptance(
    facts_df_canon: pd.DataFrame,
    sub_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deduplicate canonical facts so amended values override originals.

    Inputs
    ------
    facts_df_canon: canonicalized facts ['adsh','tag','start','end','value']
    sub_df: must include ['adsh','accepted','amendment_adsh'] (and typically is_amended)

    Output
    ------
    facts_dedup: same schema as facts_df_canon, duplicates removed.

    Method
    ------
    - Attach accepted timestamp from the *original* adsh rows by expanding
      each original adsh into its canonical adsh:
        original adsh -> canonical adsh
    - For each (canonical adsh, tag, start, end), keep the row with
      latest accepted timestamp.
    """
    req_f = {"adsh", "tag", "start", "end", "value"}
    req_s = {"adsh", "accepted", "amendment_adsh"}
    if not req_f.issubset(facts_df_canon.columns):
        raise ValueError(f"facts_df_canon must contain columns {sorted(req_f)}")
    if not req_s.issubset(sub_df.columns):
        raise ValueError(f"sub_df must contain columns {sorted(req_s)}")

    # Build canonical mapping from sub_df
    mapping = build_canonical_adsh_map(sub_df)

    # Map original adsh -> canonical, and carry accepted
    sub_min = sub_df[["adsh", "accepted"]].drop_duplicates().copy()
    sub_min["adsh"] = pd.to_numeric(sub_min["adsh"], errors="raise").astype("int64")
    sub_min = sub_min.merge(mapping, how="left", left_on="adsh", right_on="adsh_original")
    sub_min["adsh_canonical"] = sub_min["adsh_canonical"].fillna(sub_min["adsh"]).astype("int64")

    # Attach accepted to facts rows by matching on the original adsh.
    # But facts are already canonical, so we need accepted per canonical.
    # Use the maximum accepted among all originals mapping to the canonical.
    accepted_by_canon = (
        sub_min.groupby("adsh_canonical", as_index=False)["accepted"]
        .max()
        .rename(columns={"adsh_canonical": "adsh"})
    )

    df = facts_df_canon.copy()
    df["adsh"] = pd.to_numeric(df["adsh"], errors="raise").astype("int64")

    df = df.merge(accepted_by_canon, how="left", on="adsh")

    # If accepted is missing, keep row but place it at the end of sorting
    # (very rare; indicates sub_df didn't cover that adsh)
    df["_accepted_rank"] = df["accepted"].fillna(pd.Timestamp.min)

    df = (
        df.sort_values(by=["adsh", "tag", "start", "end", "_accepted_rank"], ascending=[True, True, True, True, False])
        .drop_duplicates(subset=["adsh", "tag", "start", "end"], keep="first")
        .drop(columns=["accepted", "_accepted_rank"])
    )

    return df


def canonicalize_and_merge_amendments(
    facts_df: pd.DataFrame,
    sub_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge amended submissions with their originals without destroying originals.

    Behavior
    --------
    For each amendment edge ``source_adsh -> amended_adsh`` (where
    ``sub_df.amendment_adsh != 0``), copy only missing fact keys from source to
    amended submission:

    - key = (tag, start, end)
    - if amended already has the key, amended value is kept
    - if amended does not have the key, source row is copied with ``adsh`` set
      to amended adsh

    Original submission rows are preserved unchanged.

    Chains are handled iteratively in target-acceptance order, so inherited
    missing values can flow through multiple amendment generations.

    Returns
    -------
    facts_df_out: same schema as facts_df
    """
    req_f = {"adsh", "tag", "start", "end", "value"}
    req_s = {"adsh", "amendment_adsh", "accepted"}
    if not req_f.issubset(facts_df.columns):
        raise ValueError(f"facts_df must contain columns {sorted(req_f)}")
    if not req_s.issubset(sub_df.columns):
        raise ValueError(f"sub_df must contain columns {sorted(req_s)}")

<<<<<<< codex/write-unit-tests-for-taxonomy-transforms-b0o2og
    # Single defensive copy at function boundary (keep caller input immutable).
    base = facts_df.copy()
    base["adsh"] = pd.to_numeric(base["adsh"], errors="raise").astype("int64")

    edges = sub_df[["adsh", "amendment_adsh", "accepted"]].drop_duplicates()
=======
    base = facts_df.copy()
    base["adsh"] = pd.to_numeric(base["adsh"], errors="raise").astype("int64")

    edges = sub_df[["adsh", "amendment_adsh", "accepted"]].drop_duplicates().copy()
>>>>>>> main
    edges["adsh"] = pd.to_numeric(edges["adsh"], errors="raise").astype("int64")
    edges["amendment_adsh"] = pd.to_numeric(edges["amendment_adsh"], errors="coerce").fillna(0).astype("int64")
    edges["accepted"] = pd.to_datetime(edges["accepted"], errors="coerce")

    # Keep only actual amendment links: original/submitted report -> amended report
    edges = edges[edges["amendment_adsh"] != 0].rename(
        columns={"adsh": "source_adsh", "amendment_adsh": "target_adsh"}
    )
    if edges.empty:
        return base

    # Deterministic processing; allows chain propagation (A->B->C).
    edges = edges.sort_values(["accepted", "target_adsh", "source_adsh"], kind="mergesort")

    key_cols = ["tag", "start", "end"]

    # Work only on involved submissions (small amended subset).
    target_adshs = edges["target_adsh"].drop_duplicates().astype("int64")
    source_adshs = edges["source_adsh"].drop_duplicates().astype("int64")
    involved_adshs = pd.Index(source_adshs).union(pd.Index(target_adshs))

<<<<<<< codex/write-unit-tests-for-taxonomy-transforms-b0o2og
    involved = base[base["adsh"].isin(involved_adshs)]
    non_amended = base[~base["adsh"].isin(target_adshs)]

    data_by_adsh: dict[int, pd.DataFrame] = {}
    keyset_by_adsh: dict[int, set] = {}
    for adsh, g in involved.groupby("adsh", sort=False):
        adsh_i = int(adsh)
        data_by_adsh[adsh_i] = g
        keyset_by_adsh[adsh_i] = set(zip(g["tag"], g["start"], g["end"]))
=======
    involved = base[base["adsh"].isin(involved_adshs)].copy()
    non_amended = base[~base["adsh"].isin(target_adshs)].copy()

    data_by_adsh: dict[int, pd.DataFrame] = {
        int(adsh): g.copy() for adsh, g in involved.groupby("adsh", sort=False)
    }
    keyset_by_adsh: dict[int, set] = {
        int(adsh): set(zip(g["tag"], g["start"], g["end"])) for adsh, g in involved.groupby("adsh", sort=False)
    }
>>>>>>> main

    for e in edges.itertuples(index=False):
        source_adsh = int(e.source_adsh)
        target_adsh = int(e.target_adsh)

        src = data_by_adsh.get(source_adsh)
        if src is None or src.empty:
            continue

        tgt_keys = keyset_by_adsh.get(target_adsh)
        if tgt_keys is None:
            tgt_keys = set()
            keyset_by_adsh[target_adsh] = tgt_keys

        src_keys = list(zip(src["tag"], src["start"], src["end"]))
        missing_mask = np.fromiter((k not in tgt_keys for k in src_keys), dtype=bool, count=len(src_keys))
        if not missing_mask.any():
            continue

        to_add = src.loc[missing_mask].copy()
        to_add["adsh"] = np.int64(target_adsh)

        existing_target = data_by_adsh.get(target_adsh)
        if existing_target is None or existing_target.empty:
            data_by_adsh[target_adsh] = to_add
        else:
            data_by_adsh[target_adsh] = pd.concat([existing_target, to_add], ignore_index=True)

        tgt_keys.update(k for i, k in enumerate(src_keys) if missing_mask[i])

<<<<<<< codex/write-unit-tests-for-taxonomy-transforms-b0o2og
    empty = base.iloc[0:0]
    amended_out_frames = [data_by_adsh.get(int(adsh), empty) for adsh in target_adshs.tolist()]
    amended_out = pd.concat(amended_out_frames, ignore_index=True) if amended_out_frames else empty
=======
    amended_out_frames = [data_by_adsh.get(int(adsh), base.iloc[0:0].copy()) for adsh in target_adshs.tolist()]
    amended_out = pd.concat(amended_out_frames, ignore_index=True) if amended_out_frames else base.iloc[0:0].copy()
>>>>>>> main

    out = pd.concat([non_amended, amended_out], ignore_index=True)

    # Remove exact duplicates only; do not collapse differing values.
<<<<<<< codex/write-unit-tests-for-taxonomy-transforms-b0o2og
    out = out.drop_duplicates(subset=["adsh", "tag", "start", "end", "value"], keep="first")
=======
    out = out.drop_duplicates(subset=["adsh", "tag", "start", "end", "value"], keep="first").copy()
>>>>>>> main
    out["adsh"] = pd.to_numeric(out["adsh"], errors="raise").astype("int64")
    return out
