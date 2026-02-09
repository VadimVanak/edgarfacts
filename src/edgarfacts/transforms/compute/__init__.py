# src/edgarfacts/transforms/compute/__init__.py
"""
Figure computation utilities for edgarfacts.

This subpackage contains logic to:
- remove and correct outliers in raw facts
- merge amended and original submissions
- derive quarterly / period figures from accumulated values
- compute missing figures using US-GAAP calculation arcs

The code here operates purely on pandas DataFrames and is independent of:
- SEC data extraction
- validation / emergency checks
- taxonomy download mechanics (consumed as inputs)

Public API
----------
- transform_figures

Internal modules are intentionally fine-grained to allow unit testing
on small synthetic datasets.
"""

from .pipeline import transform_figures

__all__ = ["transform_figures"]
