# edgarfacts Public API Reference

This document describes every function exposed at the top-level package import path:

```python
import edgarfacts
```

Specifically, these names are exported by `src/edgarfacts/__init__.py`:

- `get_logger`
- `extract_submissions_and_facts`
- `check_submissions_and_facts`
- `check_build_base_figures_selected_results`
- `build_base_figures`
- `transform_and_pivot_figures`
- `check_pivot_figures`

---

## Quick import example

```python
from edgarfacts import (
    get_logger,
    extract_submissions_and_facts,
    check_submissions_and_facts,
    check_build_base_figures_selected_results,
    build_base_figures,
    transform_and_pivot_figures,
    check_pivot_figures,
)
```

---

## 1) `get_logger(...)`

**Source module:** `edgarfacts.logging_utils`

```python
def get_logger(
    name: str = "edgarfacts",
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    fmt: str = "%(message)s",
    datefmt: str | None = None,
    propagate: bool = False,
) -> logging.Logger
```

Creates and returns a configured `logging.Logger`.

### What it does

- Adds exactly one `StreamHandler` for the target stream (prevents duplicate handlers on repeated calls).
- Defaults to `sys.stdout`, which is notebook/job-runner friendly.
- Accepts logging level as int or string (for example, `"INFO"`).
- Sets `logger.propagate` (default `False`) to reduce duplicate root-logger output.

### Typical usage

```python
logger = get_logger(level="INFO")
logger.info("Pipeline started")
```

---

## 2) `extract_submissions_and_facts(...)`

**Source module:** `edgarfacts.extract.pipeline`

```python
def extract_submissions_and_facts(logger, debug_mode: bool = False)
```

Main extraction entry point. Downloads/parses SEC EDGAR sources and returns the two core datasets.

### Parameters

- `logger`: logger instance (typically from `get_logger()`).
- `debug_mode`: when `True`, runs a reduced extraction intended for development/testing.

### Returns

Tuple `(df, sub)`:

- `df`: facts dataframe (core columns: `adsh`, `tag`, `start`, `end`, `value`).
- `sub`: submissions dataframe with filing metadata (including version/amendment information).

### Notes

- Internally creates a `URLFetcher`.
- Performs version repair and fallback extraction of missing figures before returning.

---

## 3) `check_submissions_and_facts(...)`

**Source module:** `edgarfacts.validation.checks`

```python
def check_submissions_and_facts(logger, df: pd.DataFrame, sub: pd.DataFrame) -> None
```

Runs the package’s production integrity checks (acts as an emergency brake).

### Parameters

- `logger`: logger used by check routines.
- `df`: facts dataframe.
- `sub`: submissions dataframe.

### Behavior

- Delegates to internal submission checks and figure checks.
- Raises `AssertionError` on integrity failures.

---

## 4) `check_build_base_figures_selected_results(...)`

**Source module:** `edgarfacts.validation.checks`

```python
def check_build_base_figures_selected_results(df: pd.DataFrame, sub: pd.DataFrame) -> None
```

Deterministic validation for outputs of `build_base_figures`.

### Parameters

- `df`: figures dataframe output from `build_base_figures`.
  - Supports either:
    - legacy columns: `value1..value4`
    - current columns: `reported_figure`, `quarterly_figure`, `reported_figure_py`, `quarterly_figure_py`
- `sub`: enriched submissions output from `build_base_figures` (must include reporting-window fields used by the checks).

### Behavior

- Verifies known reference rows are present.
- Raises:
  - `ValueError` if expected figure columns are missing.
  - `AssertionError` if expected records are not found.

---

## 5) `build_base_figures(...)`

**Source module:** `edgarfacts.transforms.figures`

```python
def build_base_figures(
    logger,
    facts_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    *,
    outlier_workers: int | None = None,
    use_process_pool: bool = True,
    apply_arcs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]
```

Builds the base figures table from raw facts and enriches submissions with inferred reporting windows.

### Parameters

- `logger`: logger instance.
- `facts_df`: raw facts table, requires columns `adsh`, `tag`, `start`, `end`, `value`.
- `sub_df`: submissions table, requires at least `adsh`, `cik`, `period`, `accepted`, `amendment_adsh`, `is_amended`.
- `outlier_workers`: optional worker count for outlier correction.
- `use_process_pool`: whether outlier correction uses process-based parallelism.
- `apply_arcs`: whether to apply US-GAAP calculation arcs to fill computable missing tags.

### Returns

Tuple `(figures_df, sub_enriched_df)`:

- `figures_df` columns include:
  - `adsh`, `tag`
  - `reported_figure`, `quarterly_figure`
  - `reported_figure_py`, `quarterly_figure_py`
  - `is_computed`
- `sub_enriched_df` = submissions + inferred windows:
  - `start_rep`, `end_rep`, `start_q`, `end_q`
  - `start_rep_py`, `end_rep_py`, `start_q_py`, `end_q_py`

### Pipeline highlights

- Normalizes dtypes.
- Canonicalizes/merges amendments.
- Removes contradictory duplicates.
- Corrects outliers and removes extreme values.
- Computes non-instant and instant period values.
- Optionally applies taxonomy arcs at the end.

---

## 6) `transform_and_pivot_figures(...)`

**Source module:** `edgarfacts.transforms.pivotize.pivotize`

```python
def transform_and_pivot_figures(
    figures: pd.DataFrame,
    submissions: pd.DataFrame,
    *,
    tol_days: int = 10,
    match_form_family: bool = True,
) -> pd.DataFrame
```

High-level transformation + pivot pipeline producing a wide model-ready dataset.

### Parameters

- `figures`: base figures dataframe.
- `submissions`: enriched submissions dataframe.
- `tol_days`: tolerance window (days) when matching prior-year/related filings.
- `match_form_family`: if `True`, prior-year matching prefers same SEC form family (e.g., 10-Q↔10-Q, 10-K↔10-K).

### Returns

A pivoted dataframe merged with interval-enriched submissions metadata.

### Output characteristics

- Pivots `tag` values by `adsh` across value families:
  - quarterly current (`_q`)
  - quarterly prior-year (`_q_py`)
  - annual current (`_a`)
  - annual prior-year (`_a_py`)
- Produces flattened column names like `{tag}_q`, `{tag}_a`, etc.

---

## 7) `check_pivot_figures(...)`

**Source module:** `edgarfacts.validation.checks`

```python
def check_pivot_figures(df: pd.DataFrame) -> None
```

Sanity-checks a pivoted figures dataframe using known reference values.

### Parameters

- `df`: pivoted dataframe expected to be indexed by `adsh` and to contain tag-suffixed columns (for example `Revenues_a`).

### Behavior

- Validates specific reference values from known filings.
- Raises `AssertionError` if any expected value is missing or mismatched.

---

## Suggested workflow

```python
from edgarfacts import (
    get_logger,
    extract_submissions_and_facts,
    check_submissions_and_facts,
    build_base_figures,
    check_build_base_figures_selected_results,
    transform_and_pivot_figures,
    check_pivot_figures,
)

logger = get_logger()

# 1) Extract raw core datasets
facts_df, sub_df = extract_submissions_and_facts(logger)
check_submissions_and_facts(logger, facts_df, sub_df)

# 2) Build base figure dataset + enriched windows
figures_df, sub_enriched_df = build_base_figures(logger, facts_df, sub_df)
check_build_base_figures_selected_results(figures_df, sub_enriched_df)

# 3) Produce pivoted wide dataset
pivot_df = transform_and_pivot_figures(figures_df, sub_enriched_df)
check_pivot_figures(pivot_df)
```
