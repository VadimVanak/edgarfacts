# src/edgarfacts/extract/__init__.py
"""
Extraction subpackage for edgarfacts.

This package contains all logic related to downloading, parsing,
and assembling SEC EDGAR datasets (companyfacts, submissions,
XBRL metadata, and related helpers).

End users should not import from submodules directly.
The stable public entry point is:

    edgarfacts.extract_submissions_and_facts
"""

# Re-export only the pipeline entry point for internal use
from .pipeline import extract_submissions_and_facts

__all__ = ["extract_submissions_and_facts"]
