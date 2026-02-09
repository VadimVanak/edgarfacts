# src/edgarfacts/transforms/__init__.py
"""
Transformations for edgarfacts.

This subpackage contains post-extraction transformations applied to the
(raw) facts and submissions dataframes produced by
`extract_submissions_and_facts`.

Design principles:
- Transformations are logically independent from extraction and validation.
- Input and output dataframe schemas are explicit and stable.
- Heavy I/O (e.g. taxonomy downloads) is isolated from pure dataframe logic.
- Functions are written to be unit-testable with small synthetic datasets.

Public API:
- transform_figures: end-to-end transformation pipeline producing
  cleaned, period-aligned, and taxonomy-completed figures.
"""

from .figures import build_base_figures

__all__ = ["build_base_figures"]
