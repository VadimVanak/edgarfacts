import unittest

import numpy as np
import pandas as pd

from edgarfacts.transforms.pivotize import transform_and_pivot_figures
from edgarfacts.transforms.pivotize.pivotize import (
    add_annual_figure_py_from_shifted_reports,
    build_adsh_shifted_by_one_year_mapping,
    build_prev_10k_mapping,
    build_prev_adsh_mapping,
    compute_annual_figures_current_year,
    fill_missing_py_from_shifted_reports,
    fill_missing_quarterly_figures,
)


class PivotizeTests(unittest.TestCase):
    def test_build_prev_10k_mapping_excludes_amended_candidates(self):
        sub = pd.DataFrame(
            {
                "adsh": [1, 2, 3],
                "cik": [10, 10, 10],
                "form": ["10-K/A", "10-K", "10-Q"],
                "end_rep": pd.to_datetime(["2022-12-31", "2023-12-31", "2024-03-31"]),
                "is_amended": [True, False, False],
            }
        )

        out = build_prev_10k_mapping(sub)
        mapped = out.loc[out["adsh"] == 3].iloc[0]

        self.assertEqual(mapped["prev_10k_adsh"], 2)
        self.assertGreater(mapped["prev_10k_diff_days"], 0)

    def test_build_adsh_shifted_by_one_year_mapping_prefers_same_form_family(self):
        sub = pd.DataFrame(
            {
                "adsh": [10, 11, 20],
                "cik": [1, 1, 1],
                "form": ["10-K", "10-Q", "10-Q"],
                "end_rep": pd.to_datetime(["2022-12-31", "2023-03-31", "2024-03-31"]),
                "is_amended": [False, False, False],
            }
        )

        out = build_adsh_shifted_by_one_year_mapping(sub, match_form_family=True)
        self.assertEqual(out.loc[out["adsh"] == 20, "adsh_py"].iloc[0], 11)

    def test_build_prev_adsh_mapping_selects_latest_previous_in_same_start(self):
        sub = pd.DataFrame(
            {
                "adsh": [100, 101, 102],
                "cik": [7, 7, 7],
                "form": ["10-Q", "10-Q", "10-Q"],
                "start_rep": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
                "end_rep": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30"]),
                "is_amended": [False, False, False],
            }
        )

        out = build_prev_adsh_mapping(sub)
        self.assertEqual(out.loc[out["adsh"] == 102, "prev_adsh"].iloc[0], 101)

    def test_fill_missing_quarterly_figures_computes_and_preserves(self):
        figures = pd.DataFrame(
            {
                "adsh": [100, 101],
                "tag": ["Revenue", "Revenue"],
                "reported_figure": [30.0, 70.0],
                "quarterly_figure": [30.0, np.nan],
                "reported_figure_py": [20.0, 40.0],
                "quarterly_figure_py": [20.0, np.nan],
            }
        )
        sub = pd.DataFrame(
            {
                "adsh": [100, 101],
                "cik": [5, 5],
                "form": ["10-Q", "10-Q"],
                "start_rep": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "end_rep": pd.to_datetime(["2024-03-31", "2024-06-30"]),
                "is_amended": [False, False],
            }
        )

        out = fill_missing_quarterly_figures(figures, sub, keep_existing=True)
        self.assertEqual(out.loc[out["adsh"] == 100, "quarterly_figure"].iloc[0], 30.0)
        self.assertEqual(out.loc[out["adsh"] == 101, "quarterly_figure"].iloc[0], 40.0)
        self.assertEqual(out.loc[out["adsh"] == 101, "quarterly_figure_py"].iloc[0], 20.0)

    def test_fill_missing_py_from_shifted_reports(self):
        figures = pd.DataFrame(
            {
                "adsh": [11, 20],
                "tag": ["Revenue", "Revenue"],
                "reported_figure": [30.0, 40.0],
                "quarterly_figure": [30.0, 40.0],
                "reported_figure_py": [np.nan, np.nan],
                "quarterly_figure_py": [np.nan, np.nan],
            }
        )
        sub = pd.DataFrame(
            {
                "adsh": [11, 20],
                "cik": [1, 1],
                "form": ["10-Q", "10-Q"],
                "end_rep": pd.to_datetime(["2023-03-31", "2024-03-31"]),
                "is_amended": [False, False],
            }
        )

        out = fill_missing_py_from_shifted_reports(figures, sub)
        self.assertEqual(out.loc[out["adsh"] == 20, "reported_figure_py"].iloc[0], 30.0)
        self.assertEqual(out.loc[out["adsh"] == 20, "quarterly_figure_py"].iloc[0], 30.0)

    def test_compute_annual_and_add_annual_py(self):
        figures = pd.DataFrame(
            {
                "adsh": [10, 20, 30],
                "tag": ["Revenue", "Revenue", "Revenue"],
                "reported_figure": [100.0, 40.0, 50.0],
                "quarterly_figure": [100.0, 40.0, 50.0],
                "reported_figure_py": [80.0, 20.0, 30.0],
                "quarterly_figure_py": [80.0, 20.0, 30.0],
            }
        )
        sub = pd.DataFrame(
            {
                "adsh": [10, 20, 30],
                "cik": [9, 9, 9],
                "form": ["10-K", "10-Q", "10-Q"],
                "end_rep": pd.to_datetime(["2023-12-31", "2024-03-31", "2025-03-31"]),
                "is_amended": [False, False, False],
            }
        )

        with_annual = compute_annual_figures_current_year(figures, sub)
        # 10-Q annual = reported + prev_10k - reported_py
        self.assertEqual(with_annual.loc[with_annual["adsh"] == 20, "annual_figure"].iloc[0], 120.0)

        with_py = add_annual_figure_py_from_shifted_reports(with_annual, sub)
        self.assertEqual(with_py.loc[with_py["adsh"] == 30, "annual_figure_py"].iloc[0], 120.0)

    def test_transform_and_pivot_figures_end_to_end(self):
        figures = pd.DataFrame(
            {
                "adsh": [100, 101, 200],
                "tag": ["Revenue", "Revenue", "Revenue"],
                "reported_figure": [30.0, 70.0, 40.0],
                "quarterly_figure": [30.0, np.nan, np.nan],
                "reported_figure_py": [20.0, np.nan, np.nan],
                "quarterly_figure_py": [20.0, np.nan, np.nan],
            }
        )
        submissions = pd.DataFrame(
            {
                "adsh": [100, 101, 200],
                "cik": [1, 1, 1],
                "form": ["10-Q", "10-Q", "10-Q"],
                "start_rep": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01"]),
                "end_rep": pd.to_datetime(["2024-03-31", "2024-06-30", "2025-03-31"]),
                "start_rep_py": pd.to_datetime(["2023-01-01", "2023-01-01", "2024-01-01"]),
                "end_rep_py": pd.to_datetime(["2023-03-31", "2023-06-30", "2024-03-31"]),
                "start_q": [pd.NaT, pd.NaT, pd.NaT],
                "end_q": [pd.NaT, pd.NaT, pd.NaT],
                "start_q_py": [pd.NaT, pd.NaT, pd.NaT],
                "end_q_py": [pd.NaT, pd.NaT, pd.NaT],
                "is_amended": [False, False, False],
            }
        )

        out = transform_and_pivot_figures(figures, submissions)

        self.assertIn("Revenue_q", out.columns)
        self.assertIn("Revenue_q_py", out.columns)
        self.assertIn("Revenue_a", out.columns)
        self.assertIn("Revenue_a_py", out.columns)
        self.assertNotIn("start_rep", out.columns)
        self.assertNotIn("end_rep", out.columns)


if __name__ == "__main__":
    unittest.main()
