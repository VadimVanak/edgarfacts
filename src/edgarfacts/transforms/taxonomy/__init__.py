# src/edgarfacts/transforms/taxonomy/__init__.py
"""
US-GAAP taxonomy utilities used by edgarfacts transforms.

This subpackage provides functionality to:
- Construct download URLs and internal ZIP paths for US-GAAP taxonomy packages
- Parse calculation linkbases into arc tables
- Download and assemble arc tables across multiple taxonomy years

Public API
----------
- read_taxonomy_arcs
- read_taxonomy_arcs_many
"""

from .reader import read_taxonomy_arcs, read_taxonomy_arcs_many

__all__ = ["read_taxonomy_arcs", "read_taxonomy_arcs_many"]
