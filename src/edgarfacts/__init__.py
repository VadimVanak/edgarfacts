# src/edgarfacts/__init__.py
"""
edgarfacts

A Python package for extracting SEC EDGAR XBRL company facts and
filing submissions into analysis-ready pandas DataFrames.

Public API:
- get_logger
- extract_submissions_and_facts
- check_submissions_and_facts
"""

from __future__ import annotations

# Public logging utility
from .logging_utils import get_logger

# Public extraction pipeline
from .extract.pipeline import extract_submissions_and_facts

# Public validation / emergency-break checks
from .validation.checks import check_submissions_and_facts

__all__ = [
    "get_logger",
    "extract_submissions_and_facts",
    "check_submissions_and_facts",
    "transform_figures",
]
