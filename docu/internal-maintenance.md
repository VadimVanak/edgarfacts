# edgarfacts Internal Maintenance Guide

> Audience: maintainers and contributors who need to modify extraction, transforms, validation, or release behavior safely.

---

## 1) Repository maintenance goals

This project has a strict operational goal:

1. Keep extraction reproducible despite SEC source drift.
2. Preserve stable dataframe contracts for downstream users.
3. Fail loudly when data integrity changes.

The code is intentionally split into **extract**, **transform**, and **validation** layers so maintenance changes can be scoped and tested in isolation.

---

## 2) High-level architecture

Top-level public exports live in `src/edgarfacts/__init__.py` and map to a 3-stage pipeline:

1. **Extraction** (`extract_submissions_and_facts`)  
   Pull SEC sources and return core `facts` + `submissions` datasets.
2. **Transform** (`build_base_figures`, `transform_and_pivot_figures`)  
   Infer reporting windows, compute quarterly/annual values, and pivot.
3. **Validation** (`check_*`)  
   Production guardrails that catch silent failures and schema regressions.

### Execution flow used in production

```text
get_logger
  -> extract_submissions_and_facts
      -> check_submissions_and_facts
          -> build_base_figures
              -> check_build_base_figures_selected_results
                  -> transform_and_pivot_figures
                      -> check_pivot_figures
```

---

## 3) Module ownership map (where to edit)

### Core API boundary

- `src/edgarfacts/__init__.py`
  - Keep `__all__` and imports synchronized when changing public API.

### Extraction (raw input normalization)

- `src/edgarfacts/extract/pipeline.py`
  - Primary orchestration logic and final dtype enforcement.
- `src/edgarfacts/extract/tickers.py`
  - `ticker.txt` ingestion.
- `src/edgarfacts/extract/periods.py`
  - Financial Statement Data Set period discovery.
- `src/edgarfacts/extract/tags.py`
  - US-GAAP tag universe discovery.
- `src/edgarfacts/extract/facts_companyfacts.py`
  - `companyfacts.zip` facts parsing.
- `src/edgarfacts/extract/submissions_fsd.py`
  - quarterly FSD submissions extraction.
- `src/edgarfacts/extract/submissions_bulk.py`
  - bulk `submissions.zip` handling, version updates, amendment flags.
- `src/edgarfacts/extract/missing_figures.py`
  - filing-level fallback when facts are missing.

### Transform (feature construction)

- `src/edgarfacts/transforms/figures.py`
  - base figures and window enrichment pipeline.
- `src/edgarfacts/transforms/periods.py`
  - reporting-window inference and value computation primitives.
- `src/edgarfacts/transforms/amendments.py`
  - amendment canonicalization/merge behavior.
- `src/edgarfacts/transforms/outliers.py`
  - outlier correction strategy.
- `src/edgarfacts/transforms/compute/arcs_apply.py`
  - taxonomy arc application.
- `src/edgarfacts/transforms/pivotize/pivotize.py`
  - high-level pivot + annual/quarterly interval filling.

### Validation and logging

- `src/edgarfacts/validation/checks.py`
  - all emergency brake checks.
- `src/edgarfacts/logging_utils.py`
  - logger initialization and handler de-duplication.

---

## 4) Data contracts and invariants you must preserve

## 4.1 Core extracted outputs

### Facts dataframe (`df`)

Expected logical schema:

- `adsh` (`int64`)
- `tag` (typically categorical/string-like)
- `start` (`datetime64[s]`)
- `end` (`datetime64[s]`)
- `value` (`float64`)

### Submissions dataframe (`sub`)

Expected logical schema:

- `adsh` (`int64`)
- `cik` (`int64`)
- `sic` (`int64`)
- `form` (string/object)
- `period` (`datetime64[s]`)
- `accepted` (`datetime64[s]`)
- `version` (`int64`)
- `amendment_adsh` (`int64`)
- `is_amended` (`bool`)
- `ticker` (categorical)

### Mandatory invariants

- `adsh` must remain numeric and stable across joins.
- Datetime precision should stay at seconds where enforced.
- Merge logic must not silently duplicate filings.
- Amendment handling must keep deterministic precedence.

---

## 4.2 Base figures outputs

`build_base_figures` returns:

1. `figures_df` with:
   - `adsh`, `tag`
   - `reported_figure`, `quarterly_figure`
   - `reported_figure_py`, `quarterly_figure_py`
   - `is_computed`
2. `sub_enriched_df` including interval columns:
   - `start_rep`, `end_rep`, `start_q`, `end_q`
   - `start_rep_py`, `end_rep_py`, `start_q_py`, `end_q_py`

Do not rename these columns without coordinated API/version changes.

---

## 5) Safe change strategy by layer

## 5.1 Extraction changes

Use this checklist before merging:

- Confirm source endpoint/format updates in SEC files.
- Keep `extract_submissions_and_facts` return schema stable.
- Verify version repair still runs after source adjustments.
- Verify fallback missing-figure extraction still executes.
- Re-run validation checks for known reference filings.

Common failure mode: source format changes produce object dtypes after concat.  
Mitigation: explicit `pd.to_numeric(...).astype(...)` enforcement as done in pipeline.

## 5.2 Transform changes

When changing period logic or imputation:

- Ensure non-instant and instant value priority remains deterministic.
- Validate amendment-adjusted values are not overwritten by stale originals.
- Confirm yearly shift mapping (`adsh_py`) still follows tolerance semantics.
- Preserve output suffix semantics in pivot stage (`_q`, `_q_py`, `_a`, `_a_py`).

Common failure mode: subtle off-by-one window boundaries.  
Mitigation: regression checks against existing hard-coded reference rows.

## 5.3 Validation changes

Only relax checks when source behavior legitimately changed and you can justify it.

- Prefer adding reference cases over deleting old ones.
- Keep assertions explicit and actionable.
- If a known SEC edge case is introduced, document why in comments and commit message.

---

## 6) Operational runbook

## 6.1 Standard local smoke run

```python
from edgarfacts import (
    get_logger,
    extract_submissions_and_facts,
    check_submissions_and_facts,
)

logger = get_logger()
df, sub = extract_submissions_and_facts(logger, debug_mode=True)
check_submissions_and_facts(logger, df, sub)
```

Use `debug_mode=True` for quick iteration and CI smoke.

## 6.2 Extended transform smoke run

```python
from edgarfacts import (
    get_logger,
    extract_submissions_and_facts,
    build_base_figures,
    check_build_base_figures_selected_results,
    transform_and_pivot_figures,
    check_pivot_figures,
)

logger = get_logger()
facts, sub = extract_submissions_and_facts(logger, debug_mode=True)
figures, sub_enriched = build_base_figures(logger, facts, sub)
check_build_base_figures_selected_results(figures, sub_enriched)
pivot = transform_and_pivot_figures(figures, sub_enriched)
check_pivot_figures(pivot)
```

---

## 7) Troubleshooting guide

## 7.1 Extraction returns too few rows

Likely causes:

- SEC source unavailable or partially downloaded.
- Parsing drift in one ingestion path (`companyfacts`, FSD, or bulk submissions).
- Over-aggressive filtering due to changed tag universe.

Actions:

1. Enable INFO logs and inspect stage-by-stage row counts.
2. Compare per-stage counts against previous successful run.
3. Confirm each source branch contributes expected records.
4. Run `check_submissions_and_facts` to locate first failing invariant.

## 7.2 Many missing quarterly figures

Likely causes:

- reporting-window inference drift
- previous-filing mapping mismatch (`prev_adsh`)
- form-family matching too strict for source changes

Actions:

1. Inspect `start_rep/end_rep/start_q/end_q` in `sub_enriched`.
2. Validate `build_prev_adsh_mapping` output cardinality.
3. Compare output with `keep_existing=True` path in quarterly fill.

## 7.3 Pivot columns missing expected tags

Likely causes:

- tag filtering removed too many rows
- infrequent-figure pruning too aggressive
- arc application not run or taxonomy version mismatch

Actions:

1. Check whether the tag exists pre-pivot in `figures_df`.
2. Confirm arc loading/apply steps completed.
3. Run `check_pivot_figures` for explicit mismatch details.

---

## 8) Performance and memory maintenance notes

- Prefer vectorized operations; avoid Python loops on full frames.
- Keep `tag` categorical where possible to reduce memory.
- Avoid unnecessary `.astype(str)` on large columns.
- Free large intermediates when no longer needed (`del`, `gc.collect()` used in pipeline).
- Keep joins narrow (`only needed columns`) to avoid accidental frame bloat.

---

## 9) API compatibility policy for maintainers

Treat these as compatibility-sensitive:

- public names exported from `__init__.py`
- documented return tuple structure
- stable column names used by validation and transforms

If you must break compatibility:

1. Update docs and README.
2. Update checks and migration guidance.
3. Use a clear breaking-change commit/PR note.

---

## 10) Release checklist (maintenance focused)

Before tagging/releasing:

- [ ] Public API exports reviewed (`__init__.py`).
- [ ] Core extraction checks pass.
- [ ] Base-figures selected-result checks pass.
- [ ] Pivot checks pass.
- [ ] Documentation updated for any changed behavior.
- [ ] Changelog/release note includes operational impact.

---

## 11) Contributor conventions for this repository

- Keep changes minimal and local to the affected layer.
- Preserve deterministic behavior in dedupe/priority logic.
- Avoid introducing silent fallback behavior without logging.
- Prefer explicit errors over hidden coercions when schema assumptions are broken.

---

## 12) Where to add future maintenance docs

Use the `docu/` directory for maintainers docs:

- `docu/public-api.md` → external API reference
- `docu/internal-maintenance.md` → this operations/maintenance runbook

If maintenance docs grow further, split by concern:

- `docu/maintenance-extract.md`
- `docu/maintenance-transforms.md`
- `docu/maintenance-validation.md`

