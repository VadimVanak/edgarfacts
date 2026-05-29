# src/edgarfacts/transforms/compute/outliers.py
"""
Outlier detection, correction, and feature building for SEC company facts.

The correction API preserves the raw facts contract while the feature-building
API exposes non-mutating signals for manual or ML-assisted outlier review.
"""

from __future__ import annotations

import gc
from pathlib import Path
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
    cik_bundle_size: int = 100,
) -> pd.Series:
    """
    Return an index-aligned initial scale-error multiplier without mutating values.

    Uses bundled CIK processing to reduce memory versus full-frame pairwise join,
    while avoiding the overhead of one join per CIK.
    """
    _log(logger, f"classifying outlier multipliers for {len(features):,} rows")

    if cik_bundle_size < 1:
        raise ValueError("cik_bundle_size must be at least 1")

    if features.empty:
        return pd.Series(
            default_class,
            index=features.index,
            dtype="float64",
            name="outlier_multiplier",
        )

    required = {"cik", "tag", "end", "value", "value_adj"}
    missing = required.difference(features.columns)
    if missing:
        raise KeyError(f"features missing required columns: {sorted(missing)}")

    out = pd.Series(
        float(default_class),
        index=features.index,
        dtype="float64",
        name="outlier_multiplier",
    )

    work = features[["cik", "tag", "end", "value", "value_adj"]].copy()
    work["_orig_index"] = features.index

    ciks = pd.Series(work["cik"].dropna().unique()).sort_values().to_numpy()
    n_bundles = int(np.ceil(len(ciks) / cik_bundle_size))

    for bundle_idx, start in enumerate(range(0, len(ciks), cik_bundle_size), start=1):
        cik_bundle = ciks[start : start + cik_bundle_size]

        if logger is not None:
            logger.info(
                f"classifying outlier multipliers: "
                f"CIK bundle {bundle_idx:,}/{n_bundles:,}, "
                f"ciks={len(cik_bundle):,}"
            )

        bundle = work.loc[work["cik"].isin(cik_bundle)]

        cls = _classify_outlier_multiplier_one_bundle(
            bundle,
            default_class=default_class,
        )

        mask = cls.ne(default_class)
        if mask.any():
            out.loc[cls.index[mask]] = cls.loc[mask].to_numpy(dtype="float64")

        del cik_bundle, bundle, cls, mask
        gc.collect()

    _log(
        logger,
        f"classified {int((out != default_class).sum()):,} candidate outliers",
    )

    del work
    gc.collect()

    return out


def _classify_outlier_multiplier_one_bundle(
    df: pd.DataFrame,
    *,
    default_class: float = 1.0,
) -> pd.Series:
    """
    Classify one bundle of CIKs using legacy pairwise scale-error logic.

    Comparisons remain restricted to same (cik, tag), so bundling CIKs does
    not change the classification logic.
    """
    result = pd.Series(
        float(default_class),
        index=df["_orig_index"].to_numpy(),
        dtype="float64",
        name="outlier_multiplier",
    )

    if len(df) <= 1:
        return result

    med = (
        df.groupby(["cik", "tag"], observed=True, as_index=False)["value_adj"]
        .median()
        .rename(columns={"value_adj": "median"})
    )

    left = df.merge(med, how="left", on=["cik", "tag"], copy=False)

    del med
    gc.collect()

    pairs = left.merge(
        df[["tag", "cik", "end", "value_adj"]],
        on=["cik", "tag"],
        suffixes=["", "_y"],
        copy=False,
    )

    del left
    gc.collect()

    pairs = pairs[np.abs((pairs["end_y"] - pairs["end"]).dt.days) <= 120]

    if pairs.empty:
        return result

    pair_mult = _classify_pairwise_scale_matches(pairs)
    mask = pair_mult.ne(default_class)

    if not mask.any():
        del pairs, pair_mult, mask
        gc.collect()
        return result

    classified = pairs.loc[mask, ["_orig_index"]].copy()
    classified["_priority"] = pair_mult.loc[mask].map(_OUTLIER_MULTIPLIER_PRIORITY)
    classified["outlier_multiplier"] = pair_mult.loc[mask].to_numpy(dtype="float64")

    best = classified.sort_values(["_orig_index", "_priority"], kind="mergesort").drop_duplicates(
        "_orig_index"
    )

    result.loc[best["_orig_index"].to_numpy()] = best["outlier_multiplier"].to_numpy(
        dtype="float64"
    )

    del pairs, pair_mult, mask, classified, best
    gc.collect()

    return result


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
    target_path: str | Path = "data/outlier_dataset_chunks",
) -> list[Path]:
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

    target_dir = Path(target_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for group_idx, version_group in enumerate(version_groups, start=1):
        _log(logger, f"starting outlier dataset version_group={version_group}")
        sub_v = submissions.loc[submissions["version"].isin(version_group)].copy()
        adsh = sub_v["adsh"].drop_duplicates()
        facts_v = facts.loc[facts["adsh"].isin(adsh)].copy()
        _log(
            logger,
            f"version_group={group_idx}/{len(version_groups)} {version_group}: "
            f"submissions={len(sub_v)} facts={len(facts_v)}",
        )

        out_v = _build_outlier_dataset_one_chunk(
            facts_v,
            sub_v,
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
        out_v = _compact_outlier_dataset(out_v)
        chunk_file = target_dir / f"outlier_dataset_chunk_{group_idx:04d}.parquet"
        out_v.to_parquet(chunk_file, index=False)
        saved_files.append(chunk_file)
        _log(logger, f"saved outlier dataset chunk to {chunk_file}")
        _log(logger, f"finished outlier dataset version_group={version_group} shape={out_v.shape}")

        del sub_v, adsh, facts_v, out_v
        gc.collect()

    _log(logger, f"finished build_outlier_dataset saved {len(saved_files)} files to {target_dir}")
    return saved_files


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
    base["duration_days"] = (base["end"] - base["start"]).dt.days.astype("int32")
    base["abs_value_log10"] = _safe_log10(base["value_adj"], min_abs_for_log).astype("float32")
    _log(
        logger,
        f"version={_version_label(submissions)}: after base frame creation shape={base.shape}",
    )

    _add_tag_stats(base)
    _log(logger, f"version={_version_label(submissions)}: after tag stats")

    _add_rolling_stats(base, rolling_window)
    _log(logger, f"version={_version_label(submissions)}: after rolling stats")

    _add_duplicate_features(base)
    _log(logger, f"version={_version_label(submissions)}: after duplicate features")

    _add_best_overlap_value(base, overlap_window=overlap_window)
    _log(logger, f"version={_version_label(submissions)}: after best-overlap features")

    return base


def _add_tag_stats(df: pd.DataFrame) -> None:
    grp = df.groupby("tag", observed=True)
    df["tag_global_median_log10"] = grp["abs_value_log10"].transform("median").astype("float32")

    tag_global_mad = pd.Series(np.nan, index=df.index, dtype="float32")
    for _, group in grp:
        tag_global_mad.loc[group.index] = _mad(group["abs_value_log10"])
    df["tag_global_mad_log10"] = tag_global_mad


def _add_rolling_stats(df: pd.DataFrame, rolling_window: int) -> None:
    df.sort_values(["cik", "tag", "end", "start", "adsh"], inplace=True, kind="mergesort")
    grp = df.groupby(["cik", "tag"], observed=True, sort=False)
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
    unique_value_count = df.groupby(keys, observed=True)["value"].transform("nunique")
    df["duplicate_majority_value"] = np.where(unique_value_count.eq(1), df["value"], np.nan).astype(
        "float64"
    )


def _add_best_overlap_value(df: pd.DataFrame, *, overlap_window: int) -> None:
    """
    Fast approximate best-overlap value using vectorized offset comparisons.

    Avoids:
    - groupby loops
    - nested Python loops
    - MultiIndex factorization

    Mutates df in place.
    """
    if df.empty:
        df["best_overlap_value"] = np.nan
        return

    ordered = df.sort_values(["cik", "tag", "start", "end", "adsh"], kind="mergesort")

    n = len(ordered)

    best_value = np.full(n, np.nan, dtype="float64")
    best_frac = np.zeros(n, dtype="float32")

    starts = ordered["start"].to_numpy(dtype="datetime64[D]")
    ends = ordered["end"].to_numpy(dtype="datetime64[D]")
    values = ordered["value"].to_numpy(dtype="float64")

    cik = ordered["cik"].to_numpy()
    tag_codes = pd.factorize(ordered["tag"], sort=False)[0].astype("int32")

    dur = (ends - starts).astype("timedelta64[D]").astype("int32")
    dur = np.maximum(dur, 0)

    for k in range(1, overlap_window + 1):
        same_group = (cik[:-k] == cik[k:]) & (tag_codes[:-k] == tag_codes[k:])

        if not same_group.any():
            continue

        left_idx = np.nonzero(same_group)[0]
        right_idx = left_idx + k

        overlap_start = np.maximum(starts[left_idx], starts[right_idx])
        overlap_end = np.minimum(ends[left_idx], ends[right_idx])

        overlap_days = (overlap_end - overlap_start).astype("timedelta64[D]").astype("int32") + 1
        overlap_days = np.maximum(overlap_days, 0)

        denom = np.minimum(
            np.maximum(dur[left_idx] + 1, 1),
            np.maximum(dur[right_idx] + 1, 1),
        )

        frac = (overlap_days / denom).astype("float32")

        # left -> right update
        better_left = frac > best_frac[left_idx]
        if better_left.any():
            upd = left_idx[better_left]
            src = right_idx[better_left]
            best_frac[upd] = frac[better_left]
            best_value[upd] = values[src]

        # right -> left update
        better_right = frac > best_frac[right_idx]
        if better_right.any():
            upd = right_idx[better_right]
            src = left_idx[better_right]
            best_frac[upd] = frac[better_right]
            best_value[upd] = values[src]

        del same_group, left_idx, right_idx, overlap_start, overlap_end
        del overlap_days, denom, frac
        gc.collect()

    result = pd.DataFrame(
        {
            "_orig_index": ordered.index.to_numpy(),
            "best_overlap_value": best_value,
        }
    ).set_index("_orig_index")

    df["best_overlap_value"] = result.loc[df.index, "best_overlap_value"].to_numpy(dtype="float64")

    del ordered, result, best_value, best_frac, starts, ends, values, cik, tag_codes, dur
    gc.collect()


def _compact_outlier_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if "tag" in df.columns:
        df["tag"] = df["tag"].astype("category")
    for col in [
        "abs_value_log10",
        "rolling_median_log10",
        "rolling_mad_log10",
        "rolling_q25_log10",
        "rolling_q75_log10",
        "tag_global_median_log10",
        "tag_global_mad_log10",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in [
        "duration_days",
        "n_prior_observations",
        "n_future_observations",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("int32")
    columns = [*_OUTLIER_DATASET_COLUMNS]
    if "outlier_multiplier" in df.columns:
        columns.append("outlier_multiplier")
    return df[columns]


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
    "tag",
    "start",
    "end",
    "duration_days",
    "value",
    "value_adj",
    "abs_value_log10",
    "best_overlap_value",
    "duplicate_majority_value",
    "rolling_median_log10",
    "rolling_mad_log10",
    "rolling_q25_log10",
    "rolling_q75_log10",
    "n_prior_observations",
    "n_future_observations",
    "tag_global_median_log10",
    "tag_global_mad_log10",
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
