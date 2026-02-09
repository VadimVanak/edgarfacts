# src/edgarfacts/validation/__init__.py
"""
Validation utilities for edgarfacts.

This subpackage contains integrity and consistency checks intended
for *production use* to detect silent extraction failures and trigger
an emergency stop.

Public entry point:
- check_submissions_and_facts
"""

from .checks import check_submissions_and_facts

__all__ = ["check_submissions_and_facts"]
