from __future__ import annotations

import numpy as np
import pandas as pd
import gc


def build_prev_10k_mapping(
    submissions: pd.DataFrame,
    *,
    tol_days: int = 10,
    use_end_col: str = "end_rep",
    exclude_amended_right: bool = True,
    amended_flag_col: str = "is_amended",
) -> pd.DataFrame:
    req = {"adsh", "cik", "form", use_end_col}
    if exclude_amended_right:
        req |= {amended_flag_col}
    missing = req - set(submissions.columns)
    if missing:
        raise ValueError(f"submissions missing columns: {sorted(missing)}")

    sub = submissions[["adsh", "cik", "form", use_end_col] + ([amended_flag_col] if exclude_amended_right else [])].copy()
    sub[use_end_col] = pd.to_datetime(sub[use_end_col])

    left = sub.rename(columns={"adsh": "_adsh", use_end_col: "_end"}).copy()

    right = sub.loc[sub["form"].astype("string").str.startswith("10-K")].rename(
        columns={"adsh": "_cand_adsh", use_end_col: "_cand_end"}
    )[["_cand_adsh", "cik", "_cand_end"] + ([amended_flag_col] if exclude_amended_right else [])].copy()

    if exclude_amended_right:
        right = right.loc[~right[amended_flag_col].fillna(False)].copy()

    merged = left.merge(right, on="cik", how="left")
    merged["_lag_days"] = (merged["_end"] - merged["_cand_end"]).dt.days
    max_lag = 365 + int(tol_days)

    merged = merged.loc[
        merged["_lag_days"].notna()
        & (merged["_lag_days"] > 0)
        & (merged["_lag_days"] <= max_lag)
    ]

    if merged.empty:
        out = submissions[["adsh"]].copy()
        out["prev_10k_adsh"] = np.int64(0)
        out["prev_10k_diff_days"] = np.int16(0)
        out["adsh"] = out["adsh"].astype("int64")
        out["prev_10k_adsh"] = out["prev_10k_adsh"].astype("int64")
        out["prev_10k_diff_days"] = out["prev_10k_diff_days"].astype("int16")
        return out

    merged = merged.sort_values(["_adsh", "_cand_end", "_cand_adsh"], ascending=[True, False, False])
    best = merged.groupby("_adsh", as_index=False).head(1)

    out = best[["_adsh", "_cand_adsh", "_lag_days"]].rename(
        columns={"_adsh": "adsh", "_cand_adsh": "prev_10k_adsh", "_lag_days": "prev_10k_diff_days"}
    )

    out = submissions[["adsh"]].merge(out, on="adsh", how="left")
    out["adsh"] = out["adsh"].astype("int64")
    out["prev_10k_adsh"] = out["prev_10k_adsh"].fillna(0).astype("int64")
    out["prev_10k_diff_days"] = out["prev_10k_diff_days"].fillna(0).astype("int16")
    return out


def build_adsh_shifted_by_one_year_mapping(
    submissions: pd.DataFrame,
    *,
    tol_days: int = 10,
    match_form_family: bool = True,
    exclude_amended_right: bool = True,
    amended_flag_col: str = "is_amended",
) -> pd.DataFrame:
    req = {"adsh", "cik", "form", "end_rep"}
    if exclude_amended_right:
        req |= {amended_flag_col}
    missing = req - set(submissions.columns)
    if missing:
        raise ValueError(f"submissions missing columns: {sorted(missing)}")

    # LEFT: all
    left = submissions[["adsh", "cik", "form", "end_rep"]].copy()
    left["end_rep"] = pd.to_datetime(left["end_rep"])

    if match_form_family:
        f = left["form"].astype("string")
        left["form_family"] = np.where(
            f.str.startswith("10-K"), "10-K",
            np.where(f.str.startswith("10-Q"), "10-Q", "OTHER")
        )
    else:
        left["form_family"] = "ANY"

    left = left.rename(columns={"adsh": "_adsh", "end_rep": "_end_rep"}).copy()
    left["_target_end_rep"] = left["_end_rep"] - pd.DateOffset(years=1)

    # RIGHT: candidates (exclude amended)
    right = submissions[["adsh", "cik", "form", "end_rep"] + ([amended_flag_col] if exclude_amended_right else [])].copy()
    right["end_rep"] = pd.to_datetime(right["end_rep"])

    if match_form_family:
        fr = right["form"].astype("string")
        right["form_family"] = np.where(
            fr.str.startswith("10-K"), "10-K",
            np.where(fr.str.startswith("10-Q"), "10-Q", "OTHER")
        )
    else:
        right["form_family"] = "ANY"

    if exclude_amended_right:
        right = right.loc[~right[amended_flag_col].fillna(False)].copy()

    right = right.rename(columns={"adsh": "_cand_adsh", "end_rep": "_cand_end_rep", "form": "_cand_form"}).copy()

    keys = ["cik", "form_family"] if match_form_family else ["cik"]

    merged = left.merge(right, on=keys, how="left")
    merged["_diff_days"] = (merged["_cand_end_rep"] - merged["_target_end_rep"]).dt.days
    merged = merged.loc[merged["_diff_days"].notna() & (merged["_diff_days"].abs() <= tol_days)]

    if merged.empty:
        out = submissions[["adsh"]].copy()
        out["adsh_py"] = np.int64(0)
        out["py_diff_days"] = np.int16(0)
        out["adsh"] = out["adsh"].astype("int64")
        out["adsh_py"] = out["adsh_py"].astype("int64")
        out["py_diff_days"] = out["py_diff_days"].astype("int16")
        return out

    merged["_abs_diff"] = merged["_diff_days"].abs()
    merged = merged.sort_values(
        ["_adsh", "_abs_diff", "_cand_end_rep", "_cand_adsh"],
        ascending=[True, True, False, False],
    )
    best = merged.groupby("_adsh", as_index=False).head(1)

    out = best[["_adsh", "_cand_adsh", "_diff_days"]].rename(
        columns={"_adsh": "adsh", "_cand_adsh": "adsh_py", "_diff_days": "py_diff_days"}
    )

    out = submissions[["adsh"]].merge(out, on="adsh", how="left")
    out["adsh"] = out["adsh"].astype("int64")
    out["adsh_py"] = out["adsh_py"].fillna(0).astype("int64")
    out["py_diff_days"] = out["py_diff_days"].fillna(0).astype("int16")
    return out


def build_prev_adsh_mapping(
    submissions: pd.DataFrame,
    *,
    left_start_col: str = "start_rep",
    left_end_col: str = "end_rep",
    right_start_col: str = "start_rep",
    right_end_col: str = "end_rep",
    out_prev_adsh_col: str = "prev_adsh",
    out_prev_form_col: str = "prev_form",
    exclude_amended_right: bool = True,
    amended_flag_col: str = "is_amended",
) -> pd.DataFrame:
    req = {"adsh", "cik", "form", left_start_col, left_end_col, right_start_col, right_end_col}
    if exclude_amended_right:
        req |= {amended_flag_col}
    missing = req - set(submissions.columns)
    if missing:
        raise ValueError(f"submissions missing columns: {sorted(missing)}")

    left = submissions[["adsh", "cik", left_start_col, left_end_col]].copy()

    right_cols = ["adsh", "cik", "form", right_start_col, right_end_col]
    if exclude_amended_right:
        right_cols.append(amended_flag_col)
    right = submissions[right_cols].copy()

    if exclude_amended_right:
        right = right.loc[~right[amended_flag_col].fillna(False)].copy()

    left = left.rename(columns={"adsh": "_adsh_left", left_start_col: "_start_left", left_end_col: "_end_left"})
    right = right.rename(columns={
        "adsh": "_adsh_right",
        "form": "_form_right",
        right_start_col: "_start_right",
        right_end_col: "_end_right",
    })

    merged = left.merge(
        right,
        left_on=["cik", "_start_left"],
        right_on=["cik", "_start_right"],
        how="left",
    )

    merged = merged.loc[merged["_end_right"].notna() & (merged["_end_right"] < merged["_end_left"])]

    if merged.empty:
        out = submissions[["adsh"]].copy()
        out[out_prev_adsh_col] = np.int64(0)
        out[out_prev_form_col] = ""
        out[out_prev_adsh_col] = out[out_prev_adsh_col].astype("int64")
        out[out_prev_form_col] = out[out_prev_form_col].astype("string")
        return out

    merged = merged.sort_values(["_adsh_left", "_end_right"])
    best = merged.groupby("_adsh_left", as_index=False).tail(1)

    out = best[["_adsh_left", "_adsh_right", "_form_right"]].rename(
        columns={"_adsh_left": "adsh", "_adsh_right": out_prev_adsh_col, "_form_right": out_prev_form_col}
    )

    out = submissions[["adsh"]].merge(out, on="adsh", how="left")
    out[out_prev_adsh_col] = out[out_prev_adsh_col].fillna(0).astype("int64")
    out[out_prev_form_col] = out[out_prev_form_col].fillna("").astype("string")
    return out


def fill_missing_quarterly_figures(
    figures: pd.DataFrame,
    submissions: pd.DataFrame,
    *,
    keep_existing: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Fill missing quarterly_figure and quarterly_figure_py.

    Previous report is defined ONLY by submissions (same cik + same start_rep; choose latest earlier end_rep).

    CURRENT YEAR:
      - if prev_adsh == 0 OR prev_form is 10-K => quarterly = reported_figure
      - else => quarterly = reported_figure(current) - reported_figure(prev)

    PREVIOUS YEAR:
      - uses THE SAME prev_adsh mapping (same fiscal-year sequencing)
      - if prev_adsh == 0 OR prev_form is 10-K => quarterly_py = reported_figure_py
      - else => quarterly_py = reported_figure_py(current) - reported_figure_py(prev)

    (This avoids the incorrect “previous-of-previous” effect.)
    """
    req_fig = {"adsh", "tag", "reported_figure", "quarterly_figure", "reported_figure_py", "quarterly_figure_py"}
    missing = req_fig - set(figures.columns)
    if missing:
        raise ValueError(f"figures missing columns: {sorted(missing)}")

    req_sub = {"adsh", "cik", "form", "start_rep", "end_rep"}
    missing = req_sub - set(submissions.columns)
    if missing:
        raise ValueError(f"submissions missing columns: {sorted(missing)}")

    figs = figures.copy()
    for c in ["reported_figure", "quarterly_figure", "reported_figure_py", "quarterly_figure_py"]:
        figs[c] = pd.to_numeric(figs[c], errors="coerce").astype("float64")

    # stable row identity for alignment
    figs["__rowid"] = figs.index.to_numpy()

    # attach minimal submissions columns needed for mapping join
    sub_cols = ["adsh", "cik", "start_rep", "end_rep"]
    fmeta = figs[[
        "__rowid", "adsh", "tag",
        "reported_figure", "quarterly_figure",
        "reported_figure_py", "quarterly_figure_py",
    ]].merge(
        submissions[sub_cols],
        on="adsh",
        how="left",
        validate="many_to_one",
    )

    if fmeta["cik"].isna().any():
        bad = fmeta.loc[fmeta["cik"].isna(), "adsh"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Some Figures.adsh missing in Submissions. Example adsh: {bad}")

    # single prev mapping
    prev_map = build_prev_adsh_mapping(
        submissions,
        left_start_col="start_rep",
        left_end_col="end_rep",
        right_start_col="start_rep",
        right_end_col="end_rep",
        out_prev_adsh_col="prev_adsh",
        out_prev_form_col="prev_form",
    )

    fmeta = fmeta.merge(prev_map, on="adsh", how="left", validate="many_to_one")
    fmeta["prev_adsh"] = pd.to_numeric(fmeta["prev_adsh"], errors="coerce").fillna(0).astype("int64")
    fmeta["prev_form"] = fmeta["prev_form"].fillna("").astype("string")

    # index by rowid (alignment guarantee)
    fmeta = fmeta.set_index("__rowid", drop=True)
    figs = figs.set_index("__rowid", drop=True)

    # lookup prev reported values by (prev_adsh, tag)
    idx = pd.MultiIndex.from_frame(figs[["adsh", "tag"]])

    ytd_cur = pd.Series(figs["reported_figure"].to_numpy(), index=idx, dtype="float64")
    ytd_py = pd.Series(figs["reported_figure_py"].to_numpy(), index=idx, dtype="float64")

    prev_idx = pd.MultiIndex.from_arrays([fmeta["prev_adsh"].to_numpy(), fmeta["tag"].to_numpy()])

    fmeta["prev_reported_figure"] = ytd_cur.reindex(prev_idx).to_numpy(dtype="float64", na_value=np.nan)
    # IMPORTANT: prev_reported_figure_py is the previous report’s reported_figure_py (same prev_adsh)
    fmeta["prev_reported_figure_py"] = ytd_py.reindex(prev_idx).to_numpy(dtype="float64", na_value=np.nan)

    # compute quarterlies
    is_first = (fmeta["prev_adsh"] == 0) | fmeta["prev_form"].str.startswith("10-K")

    q_calc = pd.Series(np.nan, index=fmeta.index, dtype="float64")
    q_calc.loc[is_first] = fmeta.loc[is_first, "reported_figure"]
    q_calc.loc[~is_first] = fmeta.loc[~is_first, "reported_figure"] - fmeta.loc[~is_first, "prev_reported_figure"]

    q_py_calc = pd.Series(np.nan, index=fmeta.index, dtype="float64")
    q_py_calc.loc[is_first] = fmeta.loc[is_first, "reported_figure_py"]
    q_py_calc.loc[~is_first] = (
        fmeta.loc[~is_first, "reported_figure_py"] - fmeta.loc[~is_first, "prev_reported_figure_py"]
    )

    # preserve existing if requested
    if keep_existing:
        q_final = fmeta["quarterly_figure"].where(fmeta["quarterly_figure"].notna(), q_calc)
        q_py_final = fmeta["quarterly_figure_py"].where(fmeta["quarterly_figure_py"].notna(), q_py_calc)
    else:
        q_final, q_py_final = q_calc, q_py_calc

    # assign back by index (no numpy; no reorder bugs)
    out = figures.copy()
    out.loc[fmeta.index, "quarterly_figure"] = q_final
    out.loc[fmeta.index, "quarterly_figure_py"] = q_py_final

    if debug:
        out.loc[fmeta.index, "prev_adsh"] = fmeta["prev_adsh"].astype("int64")
        out.loc[fmeta.index, "prev_form"] = fmeta["prev_form"]
        out.loc[fmeta.index, "prev_reported_figure"] = fmeta["prev_reported_figure"]
        out.loc[fmeta.index, "prev_reported_figure_py"] = fmeta["prev_reported_figure_py"]

        # enforce integer dtype (avoid float upcast in debug columns)
        out["prev_adsh"] = pd.to_numeric(out["prev_adsh"], errors="coerce").fillna(0).astype("int64")
        out["prev_form"] = out["prev_form"].fillna("").astype("string")

    return out


def fill_missing_py_from_shifted_reports(
    figures: pd.DataFrame,
    submissions: pd.DataFrame,
    *,
    tol_days: int = 10,
    match_form_family: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Fill missing reported_figure_py / quarterly_figure_py by copying from a matched prior-year report:

      (adsh_py, tag).reported_figure   -> reported_figure_py
      (adsh_py, tag).quarterly_figure  -> quarterly_figure_py

    Only fills when target is NaN. Matching is done by end_rep ~ end_rep-1y (±tol_days).
    """
    req_fig = {"adsh", "tag", "reported_figure", "quarterly_figure", "reported_figure_py", "quarterly_figure_py"}
    missing = req_fig - set(figures.columns)
    if missing:
        raise ValueError(f"figures missing columns: {sorted(missing)}")

    out = figures.copy()
    for c in ["reported_figure", "quarterly_figure", "reported_figure_py", "quarterly_figure_py"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    mapping = build_adsh_shifted_by_one_year_mapping(
        submissions,
        tol_days=tol_days,
        match_form_family=match_form_family,
    )

    # attach mapping to each row (preserve original index)
    tmp = out[["adsh", "tag"]].copy()
    tmp["__rowid"] = out.index.to_numpy()
    tmp = tmp.merge(mapping, on="adsh", how="left", validate="many_to_one")
    tmp["adsh_py"] = pd.to_numeric(tmp["adsh_py"], errors="coerce").fillna(0).astype("int64")
    tmp = tmp.set_index("__rowid", drop=True)

    # lookup sources keyed by (adsh, tag)
    idx = pd.MultiIndex.from_frame(out[["adsh", "tag"]])
    src_rep = pd.Series(out["reported_figure"].to_numpy(), index=idx, dtype="float64")
    src_q = pd.Series(out["quarterly_figure"].to_numpy(), index=idx, dtype="float64")

    src_idx = pd.MultiIndex.from_arrays([tmp["adsh_py"].to_numpy(), tmp["tag"].to_numpy()])
    src_rep_vals = src_rep.reindex(src_idx).to_numpy(dtype="float64", na_value=np.nan)
    src_q_vals = src_q.reindex(src_idx).to_numpy(dtype="float64", na_value=np.nan)

    # numpy fill masks (fast, avoids index pitfalls)
    rep_py = out["reported_figure_py"].to_numpy(dtype="float64", copy=True)
    q_py = out["quarterly_figure_py"].to_numpy(dtype="float64", copy=True)

    has_match = tmp["adsh_py"].to_numpy() != 0

    fill_rep = has_match & np.isnan(rep_py) & ~np.isnan(src_rep_vals)
    fill_q = has_match & np.isnan(q_py) & ~np.isnan(src_q_vals)

    rep_py[fill_rep] = src_rep_vals[fill_rep]
    q_py[fill_q] = src_q_vals[fill_q]

    out["reported_figure_py"] = rep_py.astype("float64")
    out["quarterly_figure_py"] = q_py.astype("float64")

    if debug:
        out["adsh_py_match"] = tmp["adsh_py"].astype("int64")
        out["py_diff_days"] = tmp["py_diff_days"].astype("int16")

    return out


def compute_annual_figures_current_year(
    figures: pd.DataFrame,
    submissions: pd.DataFrame,
    *,
    tol_days: int = 10,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Compute annual_figure (derived) WITHOUT overwriting source columns.

    Rules:
      - 10-K*: annual_figure = reported_figure
      - otherwise:
          annual_figure = reported_figure + prev_10k_reported_figure - reported_figure_py

    Output:
      Adds column:
        - annual_figure (float64)

      If debug=True adds:
        - prev_10k_adsh (int64), prev_10k_diff_days (int16), prev_10k_reported_figure (float64)
    """
    req_fig = {"adsh", "tag", "reported_figure", "reported_figure_py"}
    missing = req_fig - set(figures.columns)
    if missing:
        raise ValueError(f"figures missing columns: {sorted(missing)}")

    req_sub = {"adsh", "cik", "form", "end_rep"}
    missing = req_sub - set(submissions.columns)
    if missing:
        raise ValueError(f"submissions missing columns: {sorted(missing)}")

    out = figures.copy()

    # Work on renamed numeric views (do not overwrite originals)
    ytd_current = pd.to_numeric(out["reported_figure"], errors="coerce").astype("float64")
    ytd_py = pd.to_numeric(out["reported_figure_py"], errors="coerce").astype("float64")

    tmp = pd.DataFrame({
        "__rowid": out.index.to_numpy(),
        "adsh": out["adsh"].to_numpy(),
        "tag": out["tag"].to_numpy(),
        "ytd_current": ytd_current.to_numpy(),
        "ytd_py": ytd_py.to_numpy(),
    })

    meta = submissions[["adsh", "form"]].copy()
    meta["form"] = meta["form"].astype("string")
    tmp = tmp.merge(meta, on="adsh", how="left", validate="many_to_one")
    if tmp["form"].isna().any():
        bad = tmp.loc[tmp["form"].isna(), "adsh"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Some Figures.adsh missing in Submissions. Example adsh: {bad}")

    is_10k = tmp["form"].str.startswith("10-K")

    m10k = build_prev_10k_mapping(submissions, tol_days=tol_days, use_end_col="end_rep")
    tmp = tmp.merge(m10k, on="adsh", how="left", validate="many_to_one")
    tmp["prev_10k_adsh"] = pd.to_numeric(tmp["prev_10k_adsh"], errors="coerce").fillna(0).astype("int64")

    tmp = tmp.set_index("__rowid", drop=True)

    # Lookup prev 10-K ytd by (prev_10k_adsh, tag) using ytd_current series from ORIGINAL reported_figure
    idx = pd.MultiIndex.from_frame(out[["adsh", "tag"]])
    ytd_lut = pd.Series(ytd_current.to_numpy(), index=idx, dtype="float64")

    prev_idx = pd.MultiIndex.from_arrays([tmp["prev_10k_adsh"].to_numpy(), tmp["tag"].to_numpy()])
    prev_10k_ytd = ytd_lut.reindex(prev_idx).to_numpy(dtype="float64", na_value=np.nan)
    tmp["prev_10k_ytd"] = prev_10k_ytd

    annual = pd.Series(np.nan, index=tmp.index, dtype="float64")
    annual.loc[is_10k.to_numpy()] = tmp.loc[is_10k.to_numpy(), "ytd_current"]

    mask = ~is_10k.to_numpy()
    annual.loc[mask] = (
        tmp.loc[mask, "ytd_current"].astype("float64")
        + tmp.loc[mask, "prev_10k_ytd"].astype("float64")
        - tmp.loc[mask, "ytd_py"].astype("float64")
    )

    out["annual_figure"] = annual.astype("float64")

    if debug:
        out["prev_10k_adsh"] = tmp["prev_10k_adsh"].astype("int64")
        out["prev_10k_diff_days"] = tmp["prev_10k_diff_days"].astype("int16")
        out["prev_10k_reported_figure"] = tmp["prev_10k_ytd"].astype("float64")

    return out

def add_annual_figure_py_from_shifted_reports(
    figures: pd.DataFrame,
    submissions: pd.DataFrame,
    *,
    tol_days: int = 10,
    match_form_family: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add annual_figure_py by copying annual_figure from a submission shifted by ~1 year.

    Mapping:
      adsh  ->  adsh_py   (candidate ~ 1 year earlier, ±tol_days), using
      build_adsh_shifted_by_one_year_mapping(submissions, ...)

    Value:
      annual_figure_py(adsh, tag) = annual_figure(adsh_py, tag)

    Notes:
      - Does NOT overwrite any existing columns.
      - If annual_figure does not exist in `figures`, raises ValueError.
      - If there is no match (adsh_py == 0) or source annual_figure is NaN, result is NaN (OK).
      - If debug=True, adds adsh_py_match and py_diff_days.
    """
    req_fig = {"adsh", "tag", "annual_figure"}
    missing = req_fig - set(figures.columns)
    if missing:
        raise ValueError(f"figures missing columns: {sorted(missing)}")

    out = figures.copy()
    out["annual_figure"] = pd.to_numeric(out["annual_figure"], errors="coerce").astype("float64")

    mapping = build_adsh_shifted_by_one_year_mapping(
        submissions,
        tol_days=tol_days,
        match_form_family=match_form_family,
    )

    tmp = out[["adsh", "tag"]].copy()
    tmp["__rowid"] = out.index.to_numpy()
    tmp = tmp.merge(mapping, on="adsh", how="left", validate="many_to_one")
    tmp["adsh_py"] = pd.to_numeric(tmp["adsh_py"], errors="coerce").fillna(0).astype("int64")
    tmp = tmp.set_index("__rowid", drop=True)

    # source lookup: (adsh, tag) -> annual_figure
    idx = pd.MultiIndex.from_frame(out[["adsh", "tag"]])
    src_annual = pd.Series(out["annual_figure"].to_numpy(), index=idx, dtype="float64")

    src_idx = pd.MultiIndex.from_arrays([tmp["adsh_py"].to_numpy(), tmp["tag"].to_numpy()])
    annual_py_vals = src_annual.reindex(src_idx).to_numpy(dtype="float64", na_value=np.nan)

    out["annual_figure_py"] = annual_py_vals.astype("float64")

    if debug:
        out["adsh_py_match"] = tmp["adsh_py"].astype("int64")
        out["py_diff_days"] = tmp["py_diff_days"].astype("int16")

    return out


def remove_infrequent_figures(df):
    stat = (
        df.reset_index()[["tag", "reported_figure"]]
        .groupby("tag", observed=True, as_index=False)
        .size()
        .sort_values(by="size", ascending=False)
    )
    stat["rank"] = range(len(stat))
    informative_tags = stat[stat["size"] > stat["rank"]]["tag"].values

    df.reset_index(inplace=True)
    df = df[df['tag'].isin(informative_tags)&~df['reported_figure'].isna()].copy()
    return df
    

def transform_and_pivot_figures(
    figures: pd.DataFrame,
    submissions: pd.DataFrame,
    *,
    tol_days: int = 10,
    match_form_family: bool = True,
) -> pd.DataFrame:
    """
    Pipeline:

    Figures (4 transforms):
      1) fill_missing_quarterly_figures
      2) fill_missing_py_from_shifted_reports
      3) compute_annual_figures_current_year
      4) add_annual_figure_py_from_shifted_reports

    Submissions (intervals):
      - annual:
          end_a    = end_rep
          end_a_py = end_rep_py
          start_a:
              10-K -> start_rep
              else -> start_rep of prev 10-K (within 365+tol_days)
          start_a_py: start_rep of submission shifted by ~1 year (adsh_py), else NaT

      - quarterly:
          end_q: if NaT -> end_rep
          start_q: if NaT ->
              prev_adsh==0 -> start_rep
              else -> (prev end_q if available else prev end_rep) + 1 day

          end_q_py: if NaT -> end_rep_py
          start_q_py/end_q_py: if still missing and adsh_py found -> copy (start_q/end_q) from adsh_py

      - then drop start_rep,end_rep,start_rep_py,end_rep_py from submissions.

    Output:
      - Pivot figures with index='adsh', columns='tag', values=[q, q_py, a, a_py]
      - Flatten columns to "{tag}{suffix}", suffix mapping:
          quarterly_figure     -> _q
          quarterly_figure_py  -> _q_py
          annual_figure        -> _a
          annual_figure_py     -> _a_py
      - Merge pivoted figures with modified submissions on adsh.
    """
    submissions = submissions.drop_duplicates(subset="adsh")
    # ---- 1) figures transforms ----
    df = remove_infrequent_figures(figures)
    # Free memory
    del figures
    gc.collect()
    df = fill_missing_quarterly_figures(df, submissions, keep_existing=True, debug=False)
    df = fill_missing_py_from_shifted_reports(
        df, submissions, tol_days=tol_days, match_form_family=match_form_family, debug=False
    )
    df = compute_annual_figures_current_year(df, submissions, tol_days=tol_days, debug=False)
    df = add_annual_figure_py_from_shifted_reports(
        df, submissions, tol_days=tol_days, match_form_family=match_form_family, debug=False
    )

    # ---- 2) submissions: annual + quarterly intervals (with imputation) ----
    sub_req = {
        "adsh", "cik", "form",
        "start_rep", "end_rep", "start_rep_py", "end_rep_py",
        "start_q", "end_q", "start_q_py", "end_q_py",
        "is_amended",
    }
    missing = sub_req - set(submissions.columns)
    if missing:
        raise ValueError(f"submissions missing columns: {sorted(missing)}")

    sub = submissions.copy()
    sub["form"] = sub["form"].astype("string")

    # normalize datetime columns we touch
    for c in ["start_rep", "end_rep", "start_rep_py", "end_rep_py", "start_q", "end_q", "start_q_py", "end_q_py"]:
        sub[c] = pd.to_datetime(sub[c])

    # --- annual ends (trivial) ---
    sub["end_a"] = sub["end_rep"]
    sub["end_a_py"] = sub["end_rep_py"]

    # start_a: 10-K -> own start_rep, else prev 10-K start_rep
    is_10k = sub["form"].str.startswith("10-K")

    m10k = build_prev_10k_mapping(sub, tol_days=tol_days, use_end_col="end_rep", exclude_amended_right=True)
    start_rep_by_adsh = pd.Series(sub["start_rep"].to_numpy(), index=sub["adsh"].astype("int64").to_numpy())
    prev_10k_start = start_rep_by_adsh.reindex(m10k["prev_10k_adsh"].to_numpy()).to_numpy(dtype="datetime64[ns]")

    sub["start_a"] = pd.NaT
    sub.loc[is_10k, "start_a"] = sub.loc[is_10k, "start_rep"]
    sub.loc[~is_10k, "start_a"] = prev_10k_start[~is_10k.to_numpy()]

    # start_a_py: from shifted submission's start_rep
    mpy = build_adsh_shifted_by_one_year_mapping(
        sub, tol_days=tol_days, match_form_family=match_form_family, exclude_amended_right=True
    )
    adsh_py = mpy["adsh_py"].to_numpy()
    start_a_py_vals = start_rep_by_adsh.reindex(adsh_py).to_numpy(dtype="datetime64[ns]")
    start_a_py_vals[adsh_py == 0] = np.datetime64("NaT")
    sub["start_a_py"] = start_a_py_vals

    # --- quarterly ends fallback ---
    # If end_q is missing, use end_rep (report end)
    sub["end_q"] = sub["end_q"].where(sub["end_q"].notna(), sub["end_rep"])
    # If end_q_py is missing, use end_rep_py (prior-year report end)
    sub["end_q_py"] = sub["end_q_py"].where(sub["end_q_py"].notna(), sub["end_rep_py"])

    # build prev_adsh mapping (exclude amended on RIGHT) to impute start_q
    prev_map = build_prev_adsh_mapping(
        sub,
        left_start_col="start_rep",
        left_end_col="end_rep",
        right_start_col="start_rep",
        right_end_col="end_rep",
        out_prev_adsh_col="prev_adsh",
        out_prev_form_col="prev_form",
        exclude_amended_right=True,
    )

    prev_adsh = prev_map["prev_adsh"].to_numpy(dtype="int64")

    # lookup previous submission's end_q (already fallback-filled) and end_rep
    end_q_by_adsh = pd.Series(sub["end_q"].to_numpy(), index=sub["adsh"].astype("int64").to_numpy())
    end_rep_by_adsh = pd.Series(sub["end_rep"].to_numpy(), index=sub["adsh"].astype("int64").to_numpy())

    prev_end_q = end_q_by_adsh.reindex(prev_adsh).to_numpy(dtype="datetime64[ns]")
    prev_end_rep = end_rep_by_adsh.reindex(prev_adsh).to_numpy(dtype="datetime64[ns]")

    # compute start_q only where missing
    start_q = sub["start_q"].to_numpy(dtype="datetime64[ns]", copy=True)
    need_sq = pd.isna(start_q)

    is_first = prev_adsh == 0
    # if prev exists: prefer prev_end_q, else prev_end_rep; then +1 day
    base_prev_end = prev_end_q.copy()
    base_prev_end[pd.isna(base_prev_end)] = prev_end_rep[pd.isna(base_prev_end)]

    plus_one = base_prev_end + np.timedelta64(1, "D")

    # fill rules
    start_q[need_sq & is_first] = sub["start_rep"].to_numpy(dtype="datetime64[ns]")[need_sq & is_first]
    start_q[need_sq & ~is_first] = plus_one[need_sq & ~is_first]
    sub["start_q"] = start_q

    # --- quarterly PY imputation using shifted submission where possible ---
    # We already have end_q_py fallbacked to end_rep_py, but we now want to
    # impute missing (start_q_py and/or end_q_py) from the shifted submission's (start_q, end_q).
    start_q_by_adsh = pd.Series(sub["start_q"].to_numpy(), index=sub["adsh"].astype("int64").to_numpy())
    end_q_by_adsh = pd.Series(sub["end_q"].to_numpy(), index=sub["adsh"].astype("int64").to_numpy())

    shifted_start_q = start_q_by_adsh.reindex(adsh_py).to_numpy(dtype="datetime64[ns]")
    shifted_end_q = end_q_by_adsh.reindex(adsh_py).to_numpy(dtype="datetime64[ns]")

    # fill missing start_q_py from shifted_start_q
    sq_py = sub["start_q_py"].to_numpy(dtype="datetime64[ns]", copy=True)
    need_sq_py = pd.isna(sq_py) & (adsh_py != 0) & ~pd.isna(shifted_start_q)
    sq_py[need_sq_py] = shifted_start_q[need_sq_py]
    sub["start_q_py"] = sq_py

    # fill missing end_q_py from shifted_end_q (if still missing after end_rep_py fallback)
    eq_py = sub["end_q_py"].to_numpy(dtype="datetime64[ns]", copy=True)
    need_eq_py = pd.isna(eq_py) & (adsh_py != 0) & ~pd.isna(shifted_end_q)
    eq_py[need_eq_py] = shifted_end_q[need_eq_py]
    sub["end_q_py"] = eq_py

    # drop annual window anchors as requested
    sub = sub.drop(columns=["start_rep", "end_rep", "start_rep_py", "end_rep_py"])

    # ---- 3) pivot figures ----
    needed_fig_cols = {
        "adsh", "tag",
        "quarterly_figure", "quarterly_figure_py",
        "annual_figure", "annual_figure_py",
    }
    missing = needed_fig_cols - set(df.columns)
    if missing:
        raise ValueError(f"transformed figures missing columns: {sorted(missing)}")

    value_cols = ["quarterly_figure", "quarterly_figure_py", "annual_figure", "annual_figure_py"]
    df.set_index(["adsh", "tag"], inplace=True)
    df.drop(columns = np.setdiff1d(df.columns, value_cols + ["adsh", "tag"]), inplace=True)
    wide = df.unstack("tag")
    # Free memory
    del df
    gc.collect()

    # ---- 4) flatten columns: "{tag}{suffix}" ----
    suffix = {
        "quarterly_figure": "_q",
        "quarterly_figure_py": "_q_py",
        "annual_figure": "_a",
        "annual_figure_py": "_a_py",
    }
    wide.columns = [f"{tag}{suffix[val]}" for (val, tag) in wide.columns]
  
    # ---- 5) merge ----
    sub_aligned = (
        sub[sub['adsh'].isin(wide.index)]
        .set_index("adsh")
        .reindex(wide.index)
    ).copy()
    wide[sub_aligned.columns]=sub_aligned.to_numpy()
    return wide
