"""
pivotize package

High-level pipeline for transforming SEC Figures and Submissions data into
a pivotized wide dataset with computed quarterly and annual values and
imputed reporting intervals.

Public API:
    - transform_and_pivot_figures_with_intervals

Utility functions (advanced use):
    - fill_missing_quarterly_figures
    - fill_missing_py_from_shifted_reports
    - compute_annual_figures_current_year
    - add_annual_figure_py_from_shifted_reports
    - build_prev_adsh_mapping
    - build_adsh_shifted_by_one_year_mapping
    - build_prev_10k_mapping
"""

from .pivotize import transform_and_pivot_figures

__all__ = [
    # main entry point
    "transform_and_pivot_figures",
]
