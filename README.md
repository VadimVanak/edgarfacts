# edgarfacts

**edgarfacts** is a Python package for extracting **SEC EDGAR XBRL company facts** and **filing submissions metadata** into clean, analysis-ready pandas DataFrames.

It is designed for **large-scale, reproducible data ingestion** and includes strict production-grade validation checks to detect silent extraction failures.

---

## Features

- Download and parse:
  - `companyfacts.zip` (XBRL facts)
  - quarterly Financial Statement Data Sets (`sub.txt`, `num.txt`)
  - bulk `submissions.zip`
  - individual filings as a fallback for missing facts
- Robust handling of:
  - US-GAAP taxonomy versions
  - amendments and amended filings
  - missing or late submissions
- Outputs **two stable pandas DataFrames**:
  - `df` — financial facts (`adsh`, `tag`, `start`, `end`, `value`)
  - `sub` — submission metadata (CIK, ticker, form, period, version, amendment flags)
- Uses `datetime64[s]` consistently (no unnecessary precision)
- Includes a strict **“emergency brake”** validation layer for production use

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/VadimVanak/edgarfacts.git
````
Python ≥ **3.9** is required.

---

## Basic Usage

```python
from edgarfacts import get_logger, extract_submissions_and_facts, check_submissions_and_facts

logger = get_logger()

# Run full extraction (can take significant time)
df, sub = extract_submissions_and_facts(logger)

# Run strict production checks (emergency brake)
check_submissions_and_facts(logger, df, sub)
```

If any integrity condition fails, an `AssertionError` is raised immediately.

---

## Debug / Testing Mode

For unit tests and development, extraction can be run in a **short, fast mode**
(limited to a small historical period, e.g. 2009Q1):

```python
df, sub = extract_submissions_and_facts(logger, debug_mode=True)
```

This mode is intended for:

* unit tests (using `unittest`)
* CI smoke tests
* rapid iteration

---

## Output DataFrames (Stable Contract)

### `df` — Facts

| column | dtype         |
| ------ | ------------- |
| adsh   | int64         |
| tag    | category      |
| start  | datetime64[s] |
| end    | datetime64[s] |
| value  | float64       |

### `sub` — Submissions

| column         | dtype         |
| -------------- | ------------- |
| adsh           | int64         |
| cik            | int64         |
| sic            | int64         |
| form           | object        |
| period         | datetime64[s] |
| accepted       | datetime64[s] |
| version        | int64         |
| amendment_adsh | int64         |
| is_amended     | bool          |
| ticker         | category      |

**These schemas are guaranteed not to change without a major version bump.**

---

## Validation Philosophy

Even with unit tests, real-world EDGAR extraction can fail due to:

* partial downloads
* upstream format changes
* silent parsing issues

For this reason, `check_submissions_and_facts` is intended for **production use** and acts as an **emergency brake**:

* size checks
* dtype checks
* range checks
* known reference filings
* known SEC edge cases

If something goes wrong, the pipeline stops loudly.

---

## Transforms (Future Work)

The `edgarfacts.transforms` namespace is reserved for **post-extraction transformations** such as:

* aggregation
* reshaping
* feature engineering
* filtering

Transforms are intentionally kept **independent** from extraction and validation logic.

---

## License

MIT License.

---

## Disclaimer

This project is not affiliated with or endorsed by the U.S. Securities and Exchange Commission (SEC).
Users are responsible for complying with SEC fair-access policies, including proper User-Agent identification and rate limits.

```
