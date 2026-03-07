# edgarfacts Public API Reference

This document describes every function exposed at the top-level package import path:

```python
import edgarfacts
```

Specifically, these names are exported by `src/edgarfacts/__init__.py`:

* `get_logger`
* `extract_submissions_and_facts`
* `check_submissions_and_facts`
* `check_build_base_figures_selected_results`
* `build_base_figures`
* `transform_and_pivot_figures`
* `check_pivot_figures`

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

* Adds exactly one `StreamHandler` for the target stream.
* Defaults to `sys.stdout`, which is suitable for notebooks and batch jobs.
* Accepts logging level as int or string (for example, `"INFO"`).
* Sets `logger.propagate` (default `False`) to reduce duplicated root-logger output.

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

* `logger`: logger instance, typically from `get_logger()`.
* `debug_mode`: when `True`, runs a reduced extraction intended for development or testing.

### Returns

Tuple `(df, sub)`:

* `df`: facts dataframe with core columns such as:

  * `adsh`, `tag`, `start`, `end`, `value`
* `sub`: submissions dataframe with filing metadata, including versioning and amendment information

### Notes

* Internally creates a `URLFetcher`.
* Performs version repair and fallback extraction of missing figures before returning.

---

## 3) `check_submissions_and_facts(...)`

**Source module:** `edgarfacts.validation.checks`

```python
def check_submissions_and_facts(logger, df: pd.DataFrame, sub: pd.DataFrame) -> None
```

Runs the package’s production integrity checks on extracted raw data.

### Parameters

* `logger`: logger used by check routines.
* `df`: facts dataframe.
* `sub`: submissions dataframe.

### Behavior

* Delegates to internal submission checks and figure checks.
* Raises `AssertionError` on integrity failures.

---

## 4) `check_build_base_figures_selected_results(...)`

**Source module:** `edgarfacts.validation.checks`

```python
def check_build_base_figures_selected_results(df: pd.DataFrame, sub: pd.DataFrame) -> None
```

Deterministic validation for outputs of `build_base_figures`.

### Parameters

* `df`: figures dataframe output from `build_base_figures`.

  * Supports either:

    * legacy columns: `value1..value4`
    * current columns: `reported_figure`, `quarterly_figure`, `reported_figure_py`, `quarterly_figure_py`
* `sub`: enriched submissions output from `build_base_figures`, including reporting-window fields used by the checks

### Behavior

* Verifies known reference rows are present.
* Raises:

  * `ValueError` if expected figure columns are missing
  * `AssertionError` if expected records are not found

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

Builds the canonical base figures table from raw facts and enriches submissions with inferred reporting windows.

### Parameters

* `logger`: logger instance.
* `facts_df`: raw facts table, requiring columns:

  * `adsh`, `tag`, `start`, `end`, `value`
* `sub_df`: submissions table, requiring at least:

  * `adsh`, `cik`, `period`, `accepted`, `amendment_adsh`, `is_amended`
* `outlier_workers`: optional worker count for outlier correction.
* `use_process_pool`: whether outlier correction uses process-based parallelism.
* `apply_arcs`: whether to apply US-GAAP calculation arcs to fill computable missing tags.

### Returns

Tuple `(figures_df, sub_enriched_df)`:

#### `figures_df`

Columns include:

* `adsh`, `tag`
* `reported_figure`, `quarterly_figure`
* `reported_figure_py`, `quarterly_figure_py`
* `is_computed`
* `is_instant`

`is_instant` is a boolean flag indicating whether the row originated from:

* an **instant** fact (`True`), meaning `start == end`
* a **duration** fact (`False`), meaning `start != end`

This distinction is important because downstream transformations apply arithmetic reconstruction only to duration facts.

#### `sub_enriched_df`

Submissions plus inferred reporting windows:

* `start_rep`, `end_rep`, `start_q`, `end_q`
* `start_rep_py`, `end_rep_py`, `start_q_py`, `end_q_py`

### Pipeline highlights

* Normalizes dtypes.
* Canonicalizes and merges amendments.
* Removes contradictory duplicates.
* Corrects outliers and removes extreme values.
* Separates instant and non-instant facts.
* Computes non-instant and instant period values.
* Combines both into one canonical figures table.
* Stamps each output row with `is_instant`.
* Optionally applies taxonomy arcs at the end.

### Important semantics

* **Duration facts** can later be arithmetically transformed into quarterly and annual values.
* **Instant facts** are not arithmetically reconstructed in the same way; they are primarily copied or matched across filings.

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

High-level transformation and pivot pipeline producing a wide, model-ready dataset.

### Parameters

* `figures`: base figures dataframe from `build_base_figures`
* `submissions`: enriched submissions dataframe from `build_base_figures`
* `tol_days`: tolerance window in days when matching prior-year or related filings
* `match_form_family`: if `True`, prior-year matching prefers the same SEC form family, for example:

  * `10-Q ↔ 10-Q`
  * `10-K ↔ 10-K`

### Returns

A pivoted dataframe merged with interval-enriched submissions metadata.

### What it does

It performs four figure transformations before pivoting:

1. Fill missing quarterly figures where possible
2. Fill missing prior-year values from shifted prior-year filings
3. Compute current-year annual values
4. Add prior-year annual values from shifted filings

Then it enriches submissions with annual and quarterly interval metadata and pivots figures wide by tag.

### Duration-aware behavior

This function is **type-aware** through `figures.is_instant`:

#### For duration rows (`is_instant == False`)

* Quarterly values may be reconstructed using filing sequence arithmetic.
* Current-year annual values may be reconstructed using:

  * current reported value
  * previous 10-K
  * previous-year corresponding value

#### For instant rows (`is_instant == True`)

* No subtraction/addition-based quarterly reconstruction is performed.
* Current-year annual values are only set directly for 10-K filings.
* Prior-year values may still be copied from shifted filings when available.

This prevents invalid arithmetic on balance-sheet-like point-in-time figures.

### Output characteristics

The pivoted output contains flattened tag-value columns with suffixes:

* quarterly current: `_q`
* quarterly prior-year: `_q_py`
* annual current: `_a`
* annual prior-year: `_a_py`

Examples:

* `Revenues_q`
* `Revenues_a`
* `Assets_a`
* `CashAndCashEquivalentsAtCarryingValue_q_py`

### Submission interval enrichment

The returned dataframe also includes interval metadata derived from submissions, including:

* annual interval fields:

  * `start_a`, `end_a`
  * `start_a_py`, `end_a_py`
* quarterly interval fields:

  * `start_q`, `end_q`
  * `start_q_py`, `end_q_py`

These intervals may be imputed when missing, using filing relationships established during transformation.

---

## 7) `check_pivot_figures(...)`

**Source module:** `edgarfacts.validation.checks`

```python
def check_pivot_figures(df: pd.DataFrame) -> None
```

Sanity-checks a pivoted figures dataframe using known reference values.

### Parameters

* `df`: pivoted dataframe expected to contain tag-suffixed columns such as `Revenues_a`

### Behavior

* Validates specific reference values from known filings.
* Raises `AssertionError` if any expected value is missing or mismatched.

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

# 1) Extract raw datasets
facts_df, sub_df = extract_submissions_and_facts(logger)
check_submissions_and_facts(logger, facts_df, sub_df)

# 2) Build canonical base figures + enriched reporting windows
figures_df, sub_enriched_df = build_base_figures(logger, facts_df, sub_df)
check_build_base_figures_selected_results(figures_df, sub_enriched_df)

# 3) Produce pivoted wide dataset
pivot_df = transform_and_pivot_figures(figures_df, sub_enriched_df)
check_pivot_figures(pivot_df)
```

---

## Summary of core data contracts

### Base figures contract

`build_base_figures(...)` returns rows keyed by `(adsh, tag)` with numeric values in:

* `reported_figure`
* `quarterly_figure`
* `reported_figure_py`
* `quarterly_figure_py`

and flags:

* `is_computed`
* `is_instant`

### Pivot contract

`transform_and_pivot_figures(...)` returns one wide row per `adsh`, merging:

* submission metadata
* inferred annual/quarterly interval bounds
* pivoted tag-value features with suffixes `_q`, `_q_py`, `_a`, `_a_py`

### Modeling note

For downstream modeling, `is_instant` matters conceptually even though it is not preserved as a per-tag wide feature in the pivoted output. The transformation pipeline already uses it internally to avoid invalid arithmetic on instant facts.
