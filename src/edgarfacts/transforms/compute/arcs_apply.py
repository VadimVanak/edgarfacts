"""
Apply US-GAAP calculation arcs to a figures table.

This module computes missing (adsh, tag) figures by aggregating weighted
source tags according to calculation linkbase arcs. Note that XBRL taxonomies 
uses field 'to' as SOURCE.

Important performance correction
--------------------------------
We DO NOT join on raw XBRL tag strings. Tags can be long and explode memory.
Instead we:
- encode tags to int32 ids (tag_id) using the figures_df tag vocabulary
  (preferably figures_df['tag'] categorical categories)
- encode arcs 'from'/'to' to int32 ids (from_id/to_id) using the same vocabulary
- perform all joins/groupbys on integer ids
- restore the original tag dtype on output

Contract (important)
--------------------
Input figures_df MUST have (wide) columns:
- adsh (int-like)
- tag  (str/category)
- reported_figure
- quarterly_figure
- reported_figure_py
- quarterly_figure_py
- is_computed (bool)

Input arcs_df MUST have columns (single taxonomy version):
- seq (int)
- from (str)   [XBRL tag name]
- to (str)     [XBRL tag name]
- weight (float)

Output is figures_df with additional computed rows and/or filled values.
Existing reported values are not overwritten unless explicitly requested.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


_VALUE_COLS = [
    "reported_figure",
    "quarterly_figure",
    "reported_figure_py",
    "quarterly_figure_py",
]


def _ensure_required_columns(figures_df: pd.DataFrame) -> None:
    missing = [c for c in (["adsh", "tag", "is_computed"] + _VALUE_COLS) if c not in figures_df.columns]
    if missing:
        raise ValueError(f"figures_df is missing required columns: {missing}")


def _ensure_arcs_columns(arcs_df: pd.DataFrame) -> None:
    missing = [c for c in ["seq", "from", "to", "weight"] if c not in arcs_df.columns]
    if missing:
        raise ValueError(f"arcs_df is missing required columns: {missing}")


def _build_tag_vocab(figures_df: pd.DataFrame) -> Tuple[pd.Index, bool]:
    """
    Returns
    -------
    vocab : pd.Index
        Tag vocabulary. If figures_df['tag'] is categorical, this is its categories.
        Otherwise it is unique string values observed in figures_df['tag'].
    is_categorical : bool
        Whether figures_df['tag'] was categorical.
    """
    tag = figures_df["tag"]
    if pd.api.types.is_categorical_dtype(tag):
        return tag.cat.categories, True
    # If not categorical, build a deterministic vocabulary from observed tags
    # (avoid depending on python hash order)
    vals = pd.Index(pd.unique(tag.astype(str)))
    vals = vals.sort_values()
    return vals, False


def _encode_figures_tags(figures_df: pd.DataFrame, vocab: pd.Index, is_categorical: bool) -> pd.DataFrame:
    """
    Adds int32 tag_id column, without forcing tag to become long strings.
    """
    out = figures_df.copy()

    out["adsh"] = pd.to_numeric(out["adsh"], errors="raise").astype("int64")

    # Normalize numeric cols (kept float64 for reproducibility)
    for c in _VALUE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["is_computed"] = out["is_computed"].astype(bool)

    if is_categorical:
        # Fast: categorical codes
        out["tag_id"] = out["tag"].cat.codes.astype("int32")
        # cat.codes uses -1 for NaN; treat as invalid
        if (out["tag_id"] < 0).any():
            out = out[out["tag_id"] >= 0].copy()
    else:
        # Map via vocab -> ids
        tag_to_id = pd.Series(np.arange(len(vocab), dtype="int32"), index=vocab)
        # Ensure string matching only for mapping step (does not store extra copies long-term)
        out["tag_id"] = out["tag"].astype(str).map(tag_to_id).astype("int32")
        # Drop rows with tags not in vocab (should not happen)
        out = out[out["tag_id"].notna()].copy()

    return out


def _encode_arcs(arcs_df: pd.DataFrame, vocab: pd.Index) -> pd.DataFrame:
    """
    Convert arcs 'from'/'to' to int32 ids using the figures tag vocab.
    Drops string columns immediately to keep memory low.
    """
    arcs = arcs_df[["seq", "from", "to", "weight"]].copy()

    arcs["seq"] = pd.to_numeric(arcs["seq"], errors="raise").astype("int32")
    arcs["weight"] = pd.to_numeric(arcs["weight"], errors="raise").astype("float64")

    # Build mapping once
    tag_to_id = pd.Series(np.arange(len(vocab), dtype="int32"), index=vocab)

    # Map; unknowns -> -1; then filter out
    from_id = arcs["from"].astype(str).map(tag_to_id)
    to_id = arcs["to"].astype(str).map(tag_to_id)

    arcs["from_id"] = from_id.fillna(-1).astype("int32")
    arcs["to_id"] = to_id.fillna(-1).astype("int32")

    arcs = arcs[(arcs["from_id"] >= 0) & (arcs["to_id"] >= 0)].copy()

    # Drop long strings ASAP
    arcs = arcs.drop(columns=["from", "to"])

    return arcs


def _restore_tag_dtype(df: pd.DataFrame, vocab: pd.Index, was_categorical: bool) -> pd.DataFrame:
    """
    Restore 'tag' column and drop tag_id.
    """
    out = df.copy()
    if "tag_id" not in out.columns:
        return out

    # Always restore as category if original was category; otherwise restore as string.
    if was_categorical:
        out["tag"] = pd.Categorical.from_codes(out["tag_id"].astype("int32"), categories=vocab)
    else:
        out["tag"] = vocab.take(out["tag_id"].astype("int32")).astype(str)
    out = out.drop(columns=["tag_id"])
    return out


def apply_arcs_to_figures(
    figures_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    *,
    keep_original_first: bool = True,
    max_passes: int = 1,
) -> pd.DataFrame:
    """
    Apply arcs to compute missing figures (single taxonomy version).

    Notes
    -----
    - Joins are performed on int32 ids (tag_id/from_id/to_id), not strings.
    - Existing (adsh, tag) rows are not overwritten when keep_original_first=True.

    Returns
    -------
    pd.DataFrame
        Same schema as figures_df, with additional computed rows for missing tags.
    """
    _ensure_required_columns(figures_df)
    _ensure_arcs_columns(arcs_df)

    vocab, was_categorical = _build_tag_vocab(figures_df)

    # Encode once
    out = _encode_figures_tags(figures_df, vocab=vocab, is_categorical=was_categorical)
    arcs = _encode_arcs(arcs_df, vocab=vocab)

    # Work only with needed columns during computation
    out = out[["adsh", "tag_id", "is_computed"] + _VALUE_COLS].copy()

    # Deterministic iteration over seq
    seqs = np.sort(arcs["seq"].unique())

    for _ in range(max_passes):
        for s in seqs:
            a = arcs[arcs["seq"] == s][["from_id", "to_id", "weight"]].copy()
            if a.empty:
                continue

            # Join on integer tag ids (component tag = to_id)
            src = out.merge(a, how="inner", left_on="tag_id", right_on="to_id", sort=False)
            if src.empty:
                continue
            
            # Weighted contribution to total tag_id (from_id)
            # Drop component tag_id/to_id, rename from_id -> tag_id
            src = src.drop(columns=["tag_id", "to_id"]).rename(columns={"from_id": "tag_id"})
            for c in _VALUE_COLS:
                src[c] = src[c] * src["weight"]
            src = src.drop(columns=["weight"])
          
            # Aggregate contributions per (adsh, tag_id)
            comp = (
                src.groupby(["adsh", "tag_id"], as_index=False, sort=False)
                .sum(min_count=1)
                .assign(is_computed=True)
            )

            # Combine with existing, resolving collisions deterministically
            if keep_original_first:
                out = (
                    pd.concat([out.assign(_prio=1), comp.assign(_prio=2)], ignore_index=True)
                    .sort_values(["adsh", "tag_id", "_prio"], kind="mergesort")
                    .drop_duplicates(subset=["adsh", "tag_id"], keep="first")
                    .drop(columns="_prio")
                )
            else:
                out = (
                    pd.concat([out.assign(_prio=2), comp.assign(_prio=1)], ignore_index=True)
                    .sort_values(["adsh", "tag_id", "_prio"], kind="mergesort")
                    .drop_duplicates(subset=["adsh", "tag_id"], keep="first")
                    .drop(columns="_prio")
                )

    # Restore original schema/dtypes
    out = out.rename(columns={"tag_id": "tag_id"})  # no-op, for clarity
    out = _restore_tag_dtype(out, vocab=vocab, was_categorical=was_categorical)

    # Ensure column order similar to input (keep extra cols last if any)
    cols = ["adsh", "tag"] + _VALUE_COLS + ["is_computed"]
    # Some callers may keep additional columns; preserve them after required cols
    rest = [c for c in out.columns if c not in cols]
    out = out[cols + rest]

    return out


def apply_arcs_by_version(
    figures_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    logger,
    *,
    keep_original_first: bool = True,
) -> pd.DataFrame:
    """
    Apply calculation arcs by taxonomy version.

    Parameters
    ----------
    figures_df:
        Wide figures table.
    sub_df:
        Submissions table providing (adsh -> version). Must include columns ['adsh','version'].
    arcs_df:
        Arcs for multiple years. Must include columns ['version','seq','from','to','weight'].
        (May include additional columns; they are ignored.)
    logger:
        Logger for progress reporting.
    keep_original_first:
        Collision policy, passed to apply_arcs_to_figures.

    Returns
    -------
    pd.DataFrame
        figures_df with computed tags added per version.
    """
    _ensure_required_columns(figures_df)
    if "version" not in sub_df.columns:
        raise ValueError("sub_df must contain a 'version' column")
    if "version" not in arcs_df.columns:
        raise ValueError("arcs_df must contain a 'version' column")

    # Preserve original tag dtype
    vocab, was_categorical = _build_tag_vocab(figures_df)

    # Attach version to each adsh in figures_df
    vmap = sub_df[["adsh", "version"]].drop_duplicates().copy()
    vmap["adsh"] = pd.to_numeric(vmap["adsh"], errors="raise").astype("int64")
    vmap["version"] = pd.to_numeric(vmap["version"], errors="coerce").fillna(0).astype("int32")

    work = figures_df.merge(vmap, how="inner", on="adsh", sort=False)

    # Encode arcs for all years once (drop strings)
    arcs_all = arcs_df[["version", "seq", "from", "to", "weight"]].copy()
    arcs_all["version"] = pd.to_numeric(arcs_all["version"], errors="coerce").fillna(0).astype("int32")

    # Encode arcs against figures vocab (unknown tags dropped)
    # Do NOT encode per version repeatedly.
    arcs_enc = arcs_all.copy()
    tag_to_id = pd.Series(np.arange(len(vocab), dtype="int32"), index=vocab)
    arcs_enc["from_id"] = arcs_enc["from"].astype(str).map(tag_to_id).fillna(-1).astype("int32")
    arcs_enc["to_id"] = arcs_enc["to"].astype(str).map(tag_to_id).fillna(-1).astype("int32")
    arcs_enc["seq"] = pd.to_numeric(arcs_enc["seq"], errors="raise").astype("int32")
    arcs_enc["weight"] = pd.to_numeric(arcs_enc["weight"], errors="raise").astype("float64")
    arcs_enc = arcs_enc[(arcs_enc["from_id"] >= 0) & (arcs_enc["to_id"] >= 0)].copy()
    arcs_enc = arcs_enc.drop(columns=["from", "to"])

    out_frames = []
    for v in np.sort(work["version"].unique()):
        logger.info(f"Applying taxonomy arcs for version {int(v)}")
        f_v = work[work["version"] == v].drop(columns="version")

        a_v = arcs_enc[arcs_enc["version"] == v][["seq", "from_id", "to_id", "weight"]].copy()
        if a_v.empty:
            out_frames.append(f_v)
            continue

        # Apply using the encoded-ids path but without re-encoding arcs
        # We reuse apply_arcs_to_figures by reconstructing a string-ish arcs_df is undesirable,
        # so we run the same computation inline on ids.

        # Encode figures tags
        f_enc = _encode_figures_tags(f_v, vocab=vocab, is_categorical=was_categorical)
        f_enc = f_enc[["adsh", "tag_id", "is_computed"] + _VALUE_COLS].copy()

        seqs = np.sort(a_v["seq"].unique())
        out_enc = f_enc
        for s in seqs:
            a = a_v[a_v["seq"] == s][["from_id", "to_id", "weight"]]
            src = out_enc.merge(a, how="inner", left_on="tag_id", right_on="to_id", sort=False)
            if src.empty:
                continue
            src = src.drop(columns=["tag_id", "to_id"]).rename(columns={"from_id": "tag_id"})
            for c in _VALUE_COLS:
                src[c] = src[c] * src["weight"]
            src = src.drop(columns=["weight"])

            comp = (
                src.groupby(["adsh", "tag_id"], as_index=False, sort=False)
                .sum(min_count=1)
                .assign(is_computed=True)
            )
            if keep_original_first:
                out_enc = (
                    pd.concat([out_enc.assign(_prio=1), comp.assign(_prio=2)], ignore_index=True)
                    .sort_values(["adsh", "tag_id", "_prio"], kind="mergesort")
                    .drop_duplicates(subset=["adsh", "tag_id"], keep="first")
                    .drop(columns="_prio")
                )
            else:
                out_enc = (
                    pd.concat([out_enc.assign(_prio=2), comp.assign(_prio=1)], ignore_index=True)
                    .sort_values(["adsh", "tag_id", "_prio"], kind="mergesort")
                    .drop_duplicates(subset=["adsh", "tag_id"], keep="first")
                    .drop(columns="_prio")
                )

        out_v = _restore_tag_dtype(out_enc, vocab=vocab, was_categorical=was_categorical)

        cols = ["adsh", "tag"] + _VALUE_COLS + ["is_computed"]
        rest = [c for c in out_v.columns if c not in cols]
        out_v = out_v[cols + rest]

        out_frames.append(out_v)

    out = pd.concat(out_frames, ignore_index=True)
    return out


def filter_unreliable_arcs(
    figures_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    arcs_all_years_df: pd.DataFrame,
    *,
    value_col: str = "reported_figure",
    min_tests_per_equation: int = 50,
    max_fail_rate: float = 0.05,
    rtol: float = 1e-5,
    atol: float = 1e-4,
    require_full_sources: bool = True,
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out unreliable XBRL calculation equations based on observed filings.

    XBRL calculationArc semantics:
        FROM (summation/total) = Σ weight * TO (component)

    Reliability is computed per equation keyed by (version, statement, from).
    If an equation fails too often on filings where it is fully testable, all its arcs are dropped.

    Parameters
    ----------
    figures_df:
        Must contain ['adsh','tag', value_col]. 'tag' must be categorical (memory safety).
    sub_df:
        Must contain ['adsh','version'].
    arcs_all_years_df:
        Must contain ['version','statement','from','to','weight'] at minimum.
        Extra columns are preserved in the returned arcs_filtered.
    value_col:
        Which figures column to validate against.
    min_tests_per_equation:
        Minimum number of testable (adsh, equation) observations required to judge an equation.
    max_fail_rate:
        Drop equations where fail_rate > max_fail_rate.
    rtol, atol:
        Tolerance for equality check: abs(pred - y) <= atol + rtol * max(1, abs(y)).
    require_full_sources:
        If True, only test rows where *all* TO components are present (non-NA).
    logger:
        Optional logger for progress reporting.

    Returns
    -------
    arcs_filtered : pd.DataFrame
        arcs_all_years_df filtered to reliable equations. If no equations qualify as reliable
        (e.g., too few tests everywhere), returns arcs_all_years_df unchanged (safe default).
    stats_df : pd.DataFrame
        Per-equation diagnostics with columns:
        ['version','statement','from','n_terms','n_tests','n_fail','fail_rate'].
    """
    # ---- validation ----
    if "tag" not in figures_df.columns or "adsh" not in figures_df.columns:
        raise ValueError("figures_df must contain columns ['adsh','tag']")
    if value_col not in figures_df.columns:
        raise ValueError(f"value_col='{value_col}' not present in figures_df")
    if not {"adsh", "version"}.issubset(sub_df.columns):
        raise ValueError("sub_df must contain columns ['adsh','version']")

    required_arcs = {"version", "statement", "from", "to", "weight"}
    missing = sorted(required_arcs.difference(arcs_all_years_df.columns))
    if missing:
        raise ValueError(f"arcs_all_years_df missing required columns: {missing}")

    if not isinstance(figures_df["tag"].dtype, CategoricalDtype):
        raise ValueError("figures_df['tag'] must be categorical for memory-safe filtering")

    cats = figures_df["tag"].cat.categories

    # ---- attach version to figures ----
    vmap = sub_df[["adsh", "version"]].drop_duplicates().copy()
    vmap["adsh"] = pd.to_numeric(vmap["adsh"], errors="raise").astype("int64")
    vmap["version"] = pd.to_numeric(vmap["version"], errors="coerce").fillna(0).astype("int32")

    work = figures_df[["adsh", "tag", value_col]].merge(vmap, on="adsh", how="inner", sort=False)

    # ---- normalize arcs and align endpoints to tag universe ----
    arcs = arcs_all_years_df.copy()
    arcs["version"] = pd.to_numeric(arcs["version"], errors="coerce").fillna(0).astype("int32")
    arcs["weight"] = pd.to_numeric(arcs["weight"], errors="raise").astype("float64")

    # Keep statement as string-ish (often object); drop missing.
    arcs = arcs.dropna(subset=["statement", "from", "to"])
    arcs["statement"] = arcs["statement"].astype(str)

    # Align from/to to figures tag universe (categorical => codes, no long-string joins)
    arcs["from"] = pd.Categorical(arcs["from"].astype(str), categories=cats)
    arcs["to"] = pd.Categorical(arcs["to"].astype(str), categories=cats)

    arcs = arcs[arcs["from"].notna() & arcs["to"].notna()].copy()

    stats_frames = []
    keep_frames = []

    # ---- per-version loop (bounded memory) ----
    for v in np.sort(work["version"].unique()):
        f_v = work[work["version"] == v][["adsh", "tag", value_col]]
        a_v = arcs[arcs["version"] == v][["statement", "from", "to", "weight"]]
        if f_v.empty or a_v.empty:
            continue

        # n_terms per equation (version, statement, from)
        terms = (
            a_v.groupby(["statement", "from"], observed=True, sort=False)
            .size()
            .rename("n_terms")
            .reset_index()
        )

        # Join component facts to arcs on TO (component). Statement stays from arcs side.
        src = f_v.merge(a_v, left_on="tag", right_on="to", how="inner", sort=False)
        if src.empty:
            continue

        x = pd.to_numeric(src[value_col], errors="coerce").astype("float64")
        contrib = x * src["weight"].astype("float64")

        # Predict totals per (adsh, statement, from)
        pred = (
            pd.DataFrame(
                {
                    "adsh": src["adsh"].to_numpy(),
                    "statement": src["statement"].to_numpy(),
                    "from": src["from"].to_numpy(),
                    "pred_part": contrib.to_numpy(),
                    "present": x.notna().to_numpy(),
                }
            )
            .groupby(["adsh", "statement", "from"], observed=True, sort=False, as_index=False)
            .agg(
                pred=("pred_part", lambda s: s.sum(min_count=1)),
                n_present=("present", "sum"),
            )
        )

        # Actual totals are just facts for the FROM tag; statement is not present in figures,
        # so we cross with equations by joining on (adsh, from) and then later split by statement.
        from_vals = terms["from"]
        tgt = (
            f_v[f_v["tag"].isin(from_vals)]
            .rename(columns={"tag": "from", value_col: "y"})
            .copy()
        )

        test = pred.merge(tgt, on=["adsh", "from"], how="inner", sort=False)
        if test.empty:
            continue

        test = test.merge(terms, on=["statement", "from"], how="left", sort=False)

        m = test["y"].notna() & test["pred"].notna()
        if require_full_sources:
            m = m & (test["n_present"] == test["n_terms"])
        test = test[m]
        if test.empty:
            continue

        diff = (test["pred"] - test["y"]).abs()
        tol = atol + rtol * np.maximum(1.0, test["y"].abs())
        ok = diff <= tol

        eq_stats = (
            pd.DataFrame(
                {
                    "statement": test["statement"].to_numpy(),
                    "from": test["from"].to_numpy(),
                    "ok": ok.to_numpy(),
                }
            )
            .groupby(["statement", "from"], observed=True, sort=False)
            .agg(
                n_tests=("ok", "size"),
                n_fail=("ok", lambda s: (~s).sum()),
            )
            .reset_index()
        )
        eq_stats.insert(0, "version", v)
        eq_stats = eq_stats.merge(terms, on=["statement", "from"], how="left", sort=False)
        eq_stats["fail_rate"] = eq_stats["n_fail"] / eq_stats["n_tests"]
        stats_frames.append(eq_stats)

        reliable = eq_stats[
            (eq_stats["n_tests"] >= min_tests_per_equation)
            & (eq_stats["fail_rate"] <= max_fail_rate)
        ][["version", "statement", "from"]]
        keep_frames.append(reliable)

        if logger is not None:
            logger.info(
                f"Arc reliability v={int(v)}: equations={len(eq_stats)}, reliable={len(reliable)}"
            )

    stats_df = (
        pd.concat(stats_frames, ignore_index=True)
        if stats_frames
        else pd.DataFrame(columns=["version", "statement", "from", "n_terms", "n_tests", "n_fail", "fail_rate"])
    )

    keep = (
        pd.concat(keep_frames, ignore_index=True)
        if keep_frames
        else pd.DataFrame(columns=["version", "statement", "from"])
    )

    # Safe default: if nothing qualifies as reliable, do not drop anything.
    if keep.empty:
        return arcs_all_years_df, stats_df

    # Filter arcs by reliable equations (version, statement, from)
    arcs_filtered = arcs.merge(keep, on=["version", "statement", "from"], how="inner", sort=False).copy()

    # Restore schema consistency: from/to back to str (common downstream expectation)
    arcs_filtered["from"] = arcs_filtered["from"].astype(str)
    arcs_filtered["to"] = arcs_filtered["to"].astype(str)

    # Preserve extra columns from original arcs_all_years_df:
    # We filtered from a copy of arcs_all_years_df, so extras are already present.
    return arcs_filtered, stats_df
