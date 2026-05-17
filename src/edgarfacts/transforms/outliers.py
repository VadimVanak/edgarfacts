# src/edgarfacts/transforms/compute/outliers.py
"""
Outlier detection, correction, and feature building for SEC company facts.

The correction API preserves the raw facts contract while the feature-building
API exposes non-mutating signals for manual or ML-assisted outlier review.
"""

from __future__ import annotations

import gc
from functools import partial
from multiprocessing.pool import Pool
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from edgarfacts.transforms import config

# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------


def attach_cik(facts_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach `cik` to facts_df via adsh.

    Inputs
    ------
    facts_df: columns ['adsh','tag','start','end','value']
    sub_df: must include ['adsh','cik']

    Output
    ------
    facts_df with extra column 'cik' (int64)
    """
    df = facts_df[["tag", "adsh", "start", "end", "value"]].merge(
        sub_df[["adsh", "cik"]], how="inner", on="adsh"
    )
    df["cik"] = pd.to_numeric(df["cik"], errors="raise").astype("int64")
    return df


def compute_value_adj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute value_adj used for outlier detection.

    value_adj approximates a quarterly-scale magnitude:
    - instant values (end==start) keep original magnitude
    - otherwise normalize to 90 days, but never normalize with fewer than 90 days

    Adds column: 'value_adj' (float64)
    """
    df = df.copy()
    days = (df["end"] - df["start"]).dt.days

    denom = np.where(
        days > config.MIN_PERIOD_DAYS_FOR_NORMALIZATION,
        days,
        config.MIN_PERIOD_DAYS_FOR_NORMALIZATION,
    )
    df["value_adj"] = np.abs(
        np.where(
            df["end"] == df["start"],
            df["value"],
            df["value"] / denom * config.MIN_PERIOD_DAYS_FOR_NORMALIZATION,
        )
    ).astype("float64")
    return df


def classify_outlier_multiplier(
    features: pd.DataFrame,
    *,
    logger=None,
    default_class: float = 1.0,
) -> pd.Series:
    """Return an index-aligned initial scale-error multiplier without mutating values."""
    _log(logger, f"classifying outlier multipliers for {len(features)} rows")
    if features.empty:
        return pd.Series(
            default_class, index=features.index, dtype="float64", name="outlier_multiplier"
        )

    required = {"cik", "tag", "end", "value", "value_adj"}
    missing = required.difference(features.columns)
    if missing:
        raise KeyError(f"features missing required columns: {sorted(missing)}")

    work = features[["cik", "tag", "end", "value", "value_adj"]].copy()
    work["_row_id"] = np.arange(len(work), dtype=np.int64)

    med = (
        work.groupby(["cik", "tag"], observed=True, as_index=False)["value_adj"]
        .median()
        .rename(columns={"value_adj": "median"})
    )
    left = work.merge(med, how="left", on=["cik", "tag"])
    pairs = left.merge(
        work[["tag", "cik", "end", "value_adj"]],
        on=["cik", "tag"],
        suffixes=["", "_y"],
    )
    pairs = pairs[np.abs((pairs["end_y"] - pairs["end"]).dt.days) <= 120]
    pair_mult = _classify_pairwise_scale_matches(pairs)

    classified = pairs.loc[pair_mult != 1.0, ["_row_id"]].copy()
    classified["_priority"] = pair_mult[pair_mult != 1.0].map(_OUTLIER_MULTIPLIER_PRIORITY)
    classified["outlier_multiplier"] = pair_mult[pair_mult != 1.0].to_numpy(dtype="float64")

    out = np.full(len(work), float(default_class), dtype="float64")
    if not classified.empty:
        best = classified.sort_values(["_row_id", "_priority"]).drop_duplicates("_row_id")
        out[best["_row_id"].to_numpy(dtype=np.int64)] = best["outlier_multiplier"].to_numpy(
            dtype="float64"
        )

    return pd.Series(out, index=features.index, name="outlier_multiplier")


def build_outlier_dataset(
    facts: pd.DataFrame,
    submissions: pd.DataFrame,
    arcs: pd.DataFrame,
    logger,
    *,
    overlap_window: int = 3,
    rolling_window: int = 9,
    min_abs_for_log: float = 1e-12,
    max_records: int = 1,
) -> pd.DataFrame:
    """Build an extended feature frame for manual/ML-assisted outlier review."""
    _log(logger, "starting build_outlier_dataset")
    if "version" not in submissions.columns:
        raise KeyError("submissions must include a 'version' column")

    versions = list(pd.Series(submissions["version"].dropna().unique()).sort_values())
    version_groups = _group_versions_by_record_count(facts, submissions, versions, max_records)
    _log(logger, f"build_outlier_dataset processing {len(versions)} versions: {versions}")
    _log(
        logger,
        f"build_outlier_dataset split versions into {len(version_groups)} groups "
        f"with max_records={max_records}: {version_groups}",
    )

    results: list[pd.DataFrame] = []
    for group_idx, version_group in enumerate(version_groups, start=1):
        _log(logger, f"starting outlier dataset version_group={version_group}")
        sub_v = submissions.loc[submissions["version"].isin(version_group)].copy()
        arcs_v = (
            arcs.loc[arcs["version"].isin(version_group)].copy()
            if "version" in arcs.columns
            else arcs.iloc[0:0].copy()
        )
        adsh = sub_v["adsh"].drop_duplicates()
        facts_v = facts.loc[facts["adsh"].isin(adsh)].copy()
        _log(
            logger,
            f"version_group={group_idx}/{len(version_groups)} {version_group}: "
            f"submissions={len(sub_v)} arcs={len(arcs_v)} facts={len(facts_v)}",
        )

        out_v = _build_outlier_dataset_one_chunk(
            facts_v,
            sub_v,
            arcs_v,
            logger,
            overlap_window=overlap_window,
            rolling_window=rolling_window,
            min_abs_for_log=min_abs_for_log,
        )
        out_v["outlier_multiplier"] = classify_outlier_multiplier(out_v, logger=logger).astype(
            "float64"
        )
        _log(
            logger,
            f"version_group={version_group}: after initial classification shape={out_v.shape}",
        )
        results.append(out_v)
        _log(logger, f"finished outlier dataset version_group={version_group} shape={out_v.shape}")

        del sub_v, arcs_v, adsh, facts_v, out_v
        gc.collect()

    if results:
        combined = pd.concat(results, ignore_index=True)
    else:
        combined = _empty_outlier_dataset_frame()
    _log(logger, f"finished build_outlier_dataset combined shape={combined.shape}")
    return combined


def _group_versions_by_record_count(
    facts: pd.DataFrame,
    submissions: pd.DataFrame,
    versions: list,
    max_records: int,
) -> list[list]:
    """Group sorted versions so each chunk has at most max_records fact rows when possible."""
    if max_records < 1:
        raise ValueError("max_records must be at least 1")

    record_counts = (
        facts[["adsh"]]
        .merge(submissions[["adsh", "version"]], how="inner", on="adsh")
        .groupby("version", observed=True)
        .size()
        .reindex(versions, fill_value=0)
    )

    groups: list[list] = []
    current_group: list = []
    current_records = 0
    for version, n_records in record_counts.items():
        n_records = int(n_records)
        if n_records > max_records:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_records = 0
            groups.append([version])
            continue

        if current_group and current_records + n_records > max_records:
            groups.append(current_group)
            current_group = []
            current_records = 0

        current_group.append(version)
        current_records += n_records

    if current_group:
        groups.append(current_group)
    return groups


# ---------------------------------------------------------------------
# Shared outlier-rule logic
# ---------------------------------------------------------------------


_OUTLIER_MULTIPLIER_PRIORITY = {1e-6: 0, 1e-3: 1, 1e6: 2, 1e3: 3}


def _compute_outlier_rule_masks(df2: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute legacy scale-error masks for pairwise same-cik/tag comparisons."""
    v = df2["value"].astype("float64")
    m = df2["median"].astype("float64")
    vy = df2["value_adj_y"].astype("float64")

    is_outlier1 = (
        (v != 0)
        & (np.abs(v) > np.abs(m) * 100)
        & (np.abs(v) > np.abs(vy) * 965)
        & (np.abs(v) < np.abs(vy) * 1036)
    )
    is_outlier1 |= (
        (v != 0)
        & (np.abs(v) > np.abs(m) * 100)
        & (np.abs(v) > np.abs(vy) * 950)
        & (np.abs(v) < np.abs(vy) * 1053)
        & (np.abs(v) > 3.2e8)
    )
    is_outlier1 |= (
        (v != 0)
        & (np.abs(v) > np.abs(m) * 100)
        & (np.abs(v) > np.abs(vy) * 750)
        & (np.abs(v) < np.abs(vy) * 1333)
        & (np.abs(v) > 63e9)
    )

    is_outlier2 = (
        (v != 0)
        & (np.abs(v) > np.abs(m) * 10000)
        & (np.abs(v) > np.abs(vy) * 750000)
        & (np.abs(v) < np.abs(vy) * 1333000)
    )

    is_outlier3 = (
        (v != 0)
        & (np.abs(v) < np.abs(m) / 100)
        & (np.abs(v) > np.abs(vy) / 1036)
        & (np.abs(v) < np.abs(vy) / 965)
    )
    is_outlier3 |= (
        (v != 0)
        & (np.abs(v) < np.abs(m) / 100)
        & (np.abs(v) > np.abs(vy) / 1053)
        & (np.abs(v) < np.abs(vy) / 950)
        & (np.abs(vy) > 3.2e8)
    )
    is_outlier3 |= (
        (v != 0)
        & (np.abs(v) < np.abs(m) / 100)
        & (np.abs(v) > np.abs(vy) / 1333)
        & (np.abs(v) < np.abs(vy) / 750)
        & (np.abs(vy) > 63e9)
    )

    is_outlier4 = (
        (v != 0)
        & (np.abs(v) < np.abs(m) / 10000)
        & (np.abs(v) > np.abs(vy) / 1333000)
        & (np.abs(v) < np.abs(vy) / 750000)
    )

    is_outlier1 = is_outlier1 & ~is_outlier2
    is_outlier3 = is_outlier3 & ~is_outlier4
    return {
        "too_large_1e3": is_outlier1,
        "too_large_1e6": is_outlier2,
        "too_small_1e3": is_outlier3,
        "too_small_1e6": is_outlier4,
    }


def _classify_pairwise_scale_matches(df2: pd.DataFrame) -> pd.Series:
    """Map legacy pairwise scale-error masks to correction multipliers."""
    mult = pd.Series(1.0, index=df2.index, dtype="float64")
    masks = _compute_outlier_rule_masks(df2)
    mult.loc[masks["too_large_1e3"]] = 1e-3
    mult.loc[masks["too_large_1e6"]] = 1e-6
    mult.loc[masks["too_small_1e3"]] = 1e3
    mult.loc[masks["too_small_1e6"]] = 1e6
    return mult


# ---------------------------------------------------------------------
# Feature-building internals
# ---------------------------------------------------------------------


def _build_outlier_dataset_one_chunk(
    facts: pd.DataFrame,
    submissions: pd.DataFrame,
    arcs: pd.DataFrame,
    logger,
    *,
    overlap_window: int,
    rolling_window: int,
    min_abs_for_log: float,
) -> pd.DataFrame:
    """Build features for one version chunk, which may contain one or more taxonomy versions."""
    base = attach_cik(facts, submissions)
    base["start"] = pd.to_datetime(base["start"]).astype(config.DATETIME_DTYPE)
    base["end"] = pd.to_datetime(base["end"]).astype(config.DATETIME_DTYPE)
    base["value"] = pd.to_numeric(base["value"], errors="coerce").astype("float64")
    base = compute_value_adj(base)
    base = base.merge(_submission_metadata(submissions), how="left", on=["adsh", "cik"])
    base["duration_days"] = (base["end"] - base["start"]).dt.days.astype("int32")
    base["is_instant"] = base["start"] == base["end"]
    base["abs_value_log10"] = _safe_log10(base["value_adj"], min_abs_for_log).astype("float32")
    base["sign"] = np.sign(base["value"].fillna(0)).astype("int8")
    _log(
        logger,
        f"version={_version_label(submissions)}: after base frame creation shape={base.shape}",
    )

    _add_tag_stats(base, min_abs_for_log)
    _log(logger, f"version={_version_label(submissions)}: after tag stats")

    _add_submission_stats(base)
    _log(logger, f"version={_version_label(submissions)}: after submission stats")

    _add_prev_next_rolling_stats(base, rolling_window)
    _log(logger, f"version={_version_label(submissions)}: after previous/next/rolling stats")

    _add_duplicate_features(base)
    _log(logger, f"version={_version_label(submissions)}: after duplicate features")

    _add_best_overlap_value(base, overlap_window=overlap_window)
    _log(logger, f"version={_version_label(submissions)}: after best-overlap features")

    parent_map = _build_top_parent_map(arcs)
    base = _add_parent_values(base, parent_map)
    _log(logger, f"version={_version_label(submissions)}: after parent features")

    return _compact_outlier_dataset(base)


def _submission_metadata(submissions: pd.DataFrame) -> pd.DataFrame:
    meta = submissions[
        [
            c
            for c in [
                "adsh",
                "cik",
                "version",
                "sic",
                "form",
                "accepted",
                "is_amended",
                "amendment_adsh",
            ]
            if c in submissions.columns
        ]
    ].copy()
    if "accepted" in meta.columns:
        meta["accepted_year"] = pd.to_datetime(meta["accepted"], errors="coerce").dt.year.astype(
            "float32"
        )
        meta = meta.drop(columns="accepted")
    else:
        meta["accepted_year"] = np.nan
    if "is_amended" not in meta.columns:
        if "form" in meta.columns:
            meta["is_amended"] = meta["form"].astype("string").str.endswith("/A", na=False)
        elif "amendment_adsh" in meta.columns:
            meta["is_amended"] = (
                pd.to_numeric(meta["amendment_adsh"], errors="coerce").fillna(0).ne(0)
            )
        else:
            meta["is_amended"] = False
    meta = meta.drop(columns=["amendment_adsh"], errors="ignore")
    for col in ["version", "sic", "form"]:
        if col not in meta.columns:
            meta[col] = pd.NA
    return meta.drop_duplicates(["adsh", "cik"])


def _add_tag_stats(df: pd.DataFrame, min_abs_for_log: float) -> None:
    grp = df.groupby("tag", observed=True)
    df["tag_occurrence_count"] = grp["value"].transform("size").astype("int32")
    df["tag_company_count"] = grp["cik"].transform("nunique").astype("int32")
    df["tag_global_median_log10"] = grp["abs_value_log10"].transform("median").astype("float32")
    df["tag_global_mad_log10"] = grp["abs_value_log10"].transform(_mad).astype("float32")


def _add_submission_stats(df: pd.DataFrame) -> None:
    """
    Add submission-level statistics using approximate q25/q75 from median and MAD.

    Approximation:
        q25 ≈ median - 0.6745 * MAD
        q75 ≈ median + 0.6745 * MAD

    Mutates df in place.
    """
    grp = df.groupby("adsh", observed=True, sort=False)

    stats = (
        grp["abs_value_log10"]
        .agg(
            submission_median_log10="median",
            submission_mad_log10=_mad,
        )
        .reset_index()
    )

    size = (
        grp["value"]
        .size()
        .rename("submission_size")
        .reset_index()
    )

    stats = stats.merge(size, on="adsh", how="left", copy=False)

    stats["submission_q25_log10"] = (
        stats["submission_median_log10"] - 0.6745 * stats["submission_mad_log10"]
    )
    stats["submission_q75_log10"] = (
        stats["submission_median_log10"] + 0.6745 * stats["submission_mad_log10"]
    )

    # Majority rounded log10 scale per submission.
    scale = df[["adsh", "abs_value_log10"]].copy()
    scale["_scale"] = scale["abs_value_log10"].round().astype("float32")
    scale = scale.dropna(subset=["_scale"])

    if scale.empty:
        stats["submission_majority_scale"] = np.nan
    else:
        scale_counts = (
            scale.groupby(["adsh", "_scale"], observed=True, sort=False)
            .size()
            .rename("_n")
            .reset_index()
        )

        majority_scale = (
            scale_counts.sort_values(
                ["adsh", "_n", "_scale"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            .drop_duplicates("adsh")
            [["adsh", "_scale"]]
            .rename(columns={"_scale": "submission_majority_scale"})
        )

        stats = stats.merge(majority_scale, on="adsh", how="left", copy=False)

        del scale_counts, majority_scale

    stats["submission_size"] = stats["submission_size"].astype("int32")

    for col in [
        "submission_median_log10",
        "submission_mad_log10",
        "submission_q25_log10",
        "submission_q75_log10",
        "submission_majority_scale",
    ]:
        stats[col] = pd.to_numeric(stats[col], errors="coerce").astype("float32")

    merged = df[["adsh"]].merge(stats, on="adsh", how="left", copy=False)

    for col in [
        "submission_size",
        "submission_median_log10",
        "submission_mad_log10",
        "submission_q25_log10",
        "submission_q75_log10",
        "submission_majority_scale",
    ]:
        df[col] = merged[col].to_numpy()

    del grp, stats, size, scale, merged
    gc.collect()


def _add_prev_next_rolling_stats(df: pd.DataFrame, rolling_window: int) -> None:
    df.sort_values(["cik", "tag", "end", "start", "adsh"], inplace=True, kind="mergesort")
    grp = df.groupby(["cik", "tag"], observed=True, sort=False)
    df["prev_value"] = grp["value"].shift(1).astype("float64")
    df["next_value"] = grp["value"].shift(-1).astype("float64")
    df["n_prior_observations"] = grp.cumcount().astype("int32")
    sizes = grp["value"].transform("size").astype("int32")
    df["n_future_observations"] = (sizes - df["n_prior_observations"] - 1).astype("int32")

    rolling_median = pd.Series(np.nan, index=df.index, dtype="float32")
    rolling_mad = pd.Series(np.nan, index=df.index, dtype="float32")
    rolling_q25 = pd.Series(np.nan, index=df.index, dtype="float32")
    rolling_q75 = pd.Series(np.nan, index=df.index, dtype="float32")

    for _, group in grp:
        shifted = group["abs_value_log10"].shift(1)
        roll = shifted.rolling(rolling_window, min_periods=1)
        rolling_median.loc[group.index] = roll.median().astype("float32")
        rolling_mad.loc[group.index] = roll.apply(_mad, raw=False).astype("float32")
        rolling_q25.loc[group.index] = roll.quantile(0.25).astype("float32")
        rolling_q75.loc[group.index] = roll.quantile(0.75).astype("float32")

    df["rolling_median_log10"] = rolling_median
    df["rolling_mad_log10"] = rolling_mad
    df["rolling_q25_log10"] = rolling_q25
    df["rolling_q75_log10"] = rolling_q75


def _add_duplicate_features(df: pd.DataFrame) -> None:
    keys = ["cik", "tag", "start", "end"]
    grp = df.groupby(keys, observed=True)["value"]
    df["duplicate_value_count"] = grp.transform("size").astype("int32")
    df["duplicate_unique_value_count"] = grp.transform("nunique").astype("int32")
    df["duplicate_majority_value"] = np.where(
        df["duplicate_unique_value_count"].eq(1), df["value"], np.nan
    ).astype("float64")


def _add_best_overlap_value(df: pd.DataFrame, *, overlap_window: int) -> None:
    """Add nearest overlapping cik/tag value by scanning only neighboring rows."""
    n = len(df)
    best_values = np.full(n, np.nan, dtype="float64")
    best_fracs = np.zeros(n, dtype="float32")
    if n == 0:
        df["best_overlap_value"] = best_values
        df["best_overlap_fraction"] = best_fracs
        return

    ordered = df.reset_index(drop=False).sort_values(
        ["cik", "tag", "start", "end", "adsh"], kind="mergesort"
    )
    index_pos = {idx: pos for pos, idx in enumerate(df.index)}

    for _, group in ordered.groupby(["cik", "tag"], observed=True, sort=False):
        starts = group["start"].to_numpy(dtype="datetime64[D]")
        ends = group["end"].to_numpy(dtype="datetime64[D]")
        values = group["value"].to_numpy(dtype="float64")
        orig_indexes = group["index"].to_numpy()
        m = len(group)
        for j in range(m):
            best_frac = 0.0
            best_value = np.nan
            lo = max(0, j - overlap_window)
            hi = min(m, j + overlap_window + 1)
            for k in range(lo, hi):
                if k == j:
                    continue
                overlap = (
                    (min(ends[j], ends[k]) - max(starts[j], starts[k]))
                    .astype("timedelta64[D]")
                    .astype(int)
                )
                if overlap < 0:
                    continue
                dur_j = max((ends[j] - starts[j]).astype("timedelta64[D]").astype(int), 1)
                dur_k = max((ends[k] - starts[k]).astype("timedelta64[D]").astype(int), 1)
                frac = float(overlap + 1) / float(max(min(dur_j, dur_k) + 1, 1))
                if frac > best_frac:
                    best_frac = frac
                    best_value = values[k]
            pos = index_pos[orig_indexes[j]]
            best_fracs[pos] = best_frac
            best_values[pos] = best_value

    df["best_overlap_value"] = best_values
    df["best_overlap_fraction"] = best_fracs


def _build_top_parent_map(arcs: pd.DataFrame) -> pd.DataFrame:
    """Return top two most frequent parent tags for each (version, child tag)."""
    columns = [
        "version",
        "tag",
        "parent_rank",
        "parent_tag",
        "parent_weight",
        "parent_frequency",
    ]
    required = {"version", "from", "to", "weight"}
    if arcs.empty or not required.issubset(arcs.columns):
        return pd.DataFrame(columns=columns)

    counts = (
        arcs.groupby(["version", "to", "from"], observed=True)
        .agg(parent_frequency=("from", "size"), parent_weight=("weight", "median"))
        .reset_index()
        .rename(columns={"to": "tag", "from": "parent_tag"})
    )
    counts = counts.sort_values(
        ["version", "tag", "parent_frequency", "parent_tag"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    counts["parent_rank"] = counts.groupby(["version", "tag"], observed=True).cumcount() + 1
    counts = counts.loc[counts["parent_rank"].le(2), columns]
    counts["parent_frequency"] = counts["parent_frequency"].astype("int32")
    counts["parent_weight"] = pd.to_numeric(counts["parent_weight"], errors="coerce").astype(
        "float32"
    )
    return counts


def _add_parent_values(df: pd.DataFrame, parent_map: pd.DataFrame) -> pd.DataFrame:
    """Add top parent values by looking up parent tags inside the same submission."""
    out = df.copy()
    for rank in (1, 2):
        out[f"parent{rank}_value"] = np.nan
        out[f"parent{rank}_weight"] = np.nan
        out[f"parent{rank}_frequency"] = np.int32(0)

    if parent_map.empty or df.empty:
        return out

    values = (
        df.groupby(["adsh", "tag"], observed=True, as_index=False)["value"]
        .median()
        .rename(columns={"tag": "parent_tag", "value": "_parent_value"})
    )
    for rank in (1, 2):
        pm = parent_map.loc[parent_map["parent_rank"].eq(rank)].drop(columns="parent_rank")
        if pm.empty:
            continue
        tmp = out[["adsh", "version", "tag"]].merge(pm, how="left", on=["version", "tag"])
        tmp = tmp.merge(values, how="left", on=["adsh", "parent_tag"])
        out[f"parent{rank}_value"] = tmp["_parent_value"].to_numpy(dtype="float64")
        out[f"parent{rank}_weight"] = tmp["parent_weight"].to_numpy(dtype="float32")
        out[f"parent{rank}_frequency"] = (
            tmp["parent_frequency"].fillna(0).astype("int32").to_numpy()
        )
    return out


def _compact_outlier_dataset(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["version", "sic", "form", "tag"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in [
        "abs_value_log10",
        "rolling_median_log10",
        "rolling_mad_log10",
        "rolling_q25_log10",
        "rolling_q75_log10",
        "submission_median_log10",
        "submission_mad_log10",
        "submission_q25_log10",
        "submission_q75_log10",
        "submission_majority_scale",
        "tag_global_median_log10",
        "tag_global_mad_log10",
        "best_overlap_fraction",
        "parent1_weight",
        "parent2_weight",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in [
        "duration_days",
        "duplicate_value_count",
        "duplicate_unique_value_count",
        "submission_size",
        "tag_occurrence_count",
        "tag_company_count",
        "n_prior_observations",
        "n_future_observations",
        "parent1_frequency",
        "parent2_frequency",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("int32")
    return df[_OUTLIER_DATASET_COLUMNS]


def _empty_outlier_dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_OUTLIER_DATASET_COLUMNS, "outlier_multiplier"])


def _safe_log10(values: pd.Series, min_abs_for_log: float) -> pd.Series:
    return np.log10(np.maximum(np.abs(pd.to_numeric(values, errors="coerce")), min_abs_for_log))


def _mad(s: pd.Series) -> float:
    med = s.median()
    return float((s - med).abs().median())


def _version_label(submissions: pd.DataFrame):
    if "version" not in submissions.columns or submissions.empty:
        return "<unknown>"

    versions = sorted(pd.Series(submissions["version"].dropna().unique()).tolist())

    if not versions:
        return "<unknown>"

    if len(versions) == 1:
        return versions[0]

    return versions


def _log(logger, message: str) -> None:
    if logger is not None:
        logger.info(message)


_OUTLIER_DATASET_COLUMNS = [
    "adsh",
    "cik",
    "tag",
    "start",
    "end",
    "duration_days",
    "is_instant",
    "accepted_year",
    "version",
    "sic",
    "form",
    "is_amended",
    "value",
    "value_adj",
    "abs_value_log10",
    "sign",
    "prev_value",
    "next_value",
    "best_overlap_value",
    "best_overlap_fraction",
    "duplicate_value_count",
    "duplicate_unique_value_count",
    "duplicate_majority_value",
    "rolling_median_log10",
    "rolling_mad_log10",
    "rolling_q25_log10",
    "rolling_q75_log10",
    "n_prior_observations",
    "n_future_observations",
    "submission_size",
    "submission_median_log10",
    "submission_mad_log10",
    "submission_q25_log10",
    "submission_q75_log10",
    "submission_majority_scale",
    "tag_occurrence_count",
    "tag_company_count",
    "tag_global_median_log10",
    "tag_global_mad_log10",
    "parent1_value",
    "parent1_weight",
    "parent1_frequency",
    "parent2_value",
    "parent2_weight",
    "parent2_frequency",
]


# ---------------------------------------------------------------------
# Core outlier logic (per CIK)
# ---------------------------------------------------------------------


def _remove_outliers_one_group(inp: Tuple[int, pd.DataFrame], logger) -> Tuple[pd.DataFrame, int]:
    """
    Correct outliers for one CIK group.

    Input
    -----
    (i, df_cik):
        i: group index for logging
        df_cik: columns ['adsh','tag','start','end','value','cik','value_adj']

    Returns
    -------
    (df_fixed, n_outliers)
        df_fixed: same columns as df_cik (value/value_adj corrected)
        n_outliers: total corrected rows (counted as number of matched outlier rows)
    """
    i, df = inp
    if i % 100 == 0:
        logger.info(f"Processing ticker #{i}")

    n_outliers = 0

    # iterative correction: once corrected, neighbors can become detectable
    while True:
        # Median value_adj per (cik, tag) used for robustness
        med = (
            df.groupby(["cik", "tag"], observed=True, as_index=False)["value_adj"]
            .median()
            .rename(columns={"value_adj": "median"})
        )
        df1 = df.merge(med, how="left", on=["cik", "tag"])

        # Pairwise comparisons for same (cik, tag)
        df2 = df1.merge(
            df[["tag", "cik", "end", "value_adj"]],
            on=["cik", "tag"],
            suffixes=["", "_y"],
        )

        # Only compare within 120 days
        df2 = df2[np.abs((df2["end_y"] - df2["end"]).dt.days) <= 120]
        pair_mult = _classify_pairwise_scale_matches(df2)

        # Build correction table
        corr = pd.concat(
            (
                df2[pair_mult.eq(1e-3)][df.columns].drop_duplicates().assign(mult=1e-3),
                df2[pair_mult.eq(1e-6)][df.columns].drop_duplicates().assign(mult=1e-6),
                df2[pair_mult.eq(1e3)][df.columns].drop_duplicates().assign(mult=1e3),
                df2[pair_mult.eq(1e6)][df.columns].drop_duplicates().assign(mult=1e6),
            ),
            ignore_index=True,
        )

        if corr.empty:
            return df, n_outliers

        n_outliers += len(corr)

        # Apply correction
        df = df.merge(corr, how="left")
        df["mult"] = df["mult"].fillna(1.0)
        df["value"] = df["value"].astype("float64") * df["mult"]
        df["value_adj"] = df["value_adj"].astype("float64") * df["mult"]
        df = df.drop(columns="mult")


def remove_outliers_parallel(
    facts_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    logger,
    *,
    workers: Optional[int] = None,
    use_process_pool: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove/correct outliers in raw facts.

    Inputs
    ------
    facts_df: columns ['adsh','tag','start','end','value']
    sub_df: must include ['adsh','cik']
    logger: logger instance
    workers: process pool size
    use_process_pool: if False, runs sequentially

    Outputs
    -------
    (facts_fixed, n_outliers)
    - facts_fixed: same schema as facts_df (no extra columns)
    - n_outliers: number of corrected rows (sum across CIK groups)
    """
    # Prepare working frame
    df = attach_cik(facts_df, sub_df)
    df = compute_value_adj(df)

    # Group by cik
    groups = [(i, g.copy()) for i, (_, g) in enumerate(df.groupby("cik", sort=False))]

    if not use_process_pool or len(groups) <= 1:
        results = [_remove_outliers_one_group(g, logger=logger) for g in groups]
    else:
        pool = Pool(processes=workers)
        try:
            fn = partial(_remove_outliers_one_group, logger=logger)
            results = pool.map(fn, groups)
        finally:
            pool.close()
            pool.join()

    n_outliers = int(sum(r[1] for r in results))
    logger.info(f"{n_outliers} outliers removed")

    fixed = pd.concat((r[0] for r in results), ignore_index=True)

    # Return to raw facts schema (drop helper cols)
    fixed = fixed.drop(columns=["cik", "value_adj"], errors="ignore")
    fixed = fixed[list(facts_df.columns)]

    # Keep deterministic dtypes/precision
    fixed["adsh"] = pd.to_numeric(fixed["adsh"], errors="raise").astype("int64")
    fixed["value"] = pd.to_numeric(fixed["value"], errors="coerce").astype("float64")
    fixed["start"] = fixed["start"].astype(config.DATETIME_DTYPE)
    fixed["end"] = fixed["end"].astype(config.DATETIME_DTYPE)

    return fixed, n_outliers
