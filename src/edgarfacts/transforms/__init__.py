# src/edgarfacts/transforms/__init__.py
"""
Transformations subpackage for edgarfacts.

This namespace is reserved for post-extraction transformations of the
datasets produced by `extract_submissions_and_facts`, such as:

- normalization / reshaping
- aggregation across periods
- feature engineering
- filtering and enrichment

Transformations are intentionally kept independent from the extraction
and validation logic to ensure a stable and reproducible ingestion layer.
"""

__all__ = []
