import unittest

import pandas as pd

from pandas.api.types import CategoricalDtype

from edgarfacts.transforms.compute.arcs_apply import (
    apply_arcs_by_version,
    apply_arcs_to_figures,
    expand_arcs_with_single_variable_rearrangements,
    filter_unreliable_arcs,
)
from edgarfacts.transforms.compute.companyfacts import (
    align_arc_tag_dtype,
    compute_missing_with_arcs,
    filter_arcs_to_tags,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


def _base_figures(tags, categories=None):
    tag_col = pd.Categorical(tags, categories=categories) if categories is not None else tags
    return pd.DataFrame(
        {
            "adsh": [1 for _ in tags],
            "tag": tag_col,
            "reported_figure": [40.0 if t == "B" else 60.0 if t == "C" else None for t in tags],
            "quarterly_figure": [40.0 if t == "B" else 60.0 if t == "C" else None for t in tags],
            "reported_figure_py": [4.0 if t == "B" else 6.0 if t == "C" else None for t in tags],
            "quarterly_figure_py": [4.0 if t == "B" else 6.0 if t == "C" else None for t in tags],
            "is_computed": [False for _ in tags],
        }
    )


class ComputeArcsApplyTests(unittest.TestCase):
    def test_apply_arcs_to_figures_computes_missing_tag(self):
        figures = _base_figures(["A", "B", "C"], categories=["A", "B", "C"])
        arcs = pd.DataFrame(
            {
                "seq": [0, 0],
                "from": ["A", "A"],
                "to": ["B", "C"],
                "weight": [1.0, 1.0],
            }
        )

        result = apply_arcs_to_figures(figures, arcs, keep_original_first=False)
        a_row = result[result["tag"] == "A"].iloc[0]

        self.assertEqual(a_row["reported_figure"], 100.0)
        self.assertTrue(bool(a_row["is_computed"]))

    def test_apply_arcs_to_figures_honors_keep_original_first(self):
        figures = _base_figures(["A", "B", "C"], categories=["A", "B", "C"])
        figures.loc[figures["tag"] == "A", [
            "reported_figure",
            "quarterly_figure",
            "reported_figure_py",
            "quarterly_figure_py",
        ]] = [999.0, 999.0, 99.0, 99.0]

        arcs = pd.DataFrame(
            {
                "seq": [0, 0],
                "from": ["A", "A"],
                "to": ["B", "C"],
                "weight": [1.0, 1.0],
            }
        )

        kept = apply_arcs_to_figures(figures, arcs, keep_original_first=True)
        replaced = apply_arcs_to_figures(figures, arcs, keep_original_first=False)

        self.assertEqual(kept.loc[kept["tag"] == "A", "reported_figure"].iloc[0], 999.0)
        self.assertEqual(replaced.loc[replaced["tag"] == "A", "reported_figure"].iloc[0], 100.0)

    def test_apply_arcs_by_version_uses_submission_taxonomy_version(self):
        figures = pd.DataFrame(
            {
                "adsh": [1, 1, 2, 2],
                "tag": pd.Categorical(["B", "C", "B", "C"], categories=["A", "B", "C", "D"]),
                "reported_figure": [10.0, 5.0, 7.0, 3.0],
                "quarterly_figure": [10.0, 5.0, 7.0, 3.0],
                "reported_figure_py": [1.0, 0.5, 0.7, 0.3],
                "quarterly_figure_py": [1.0, 0.5, 0.7, 0.3],
                "is_computed": [False, False, False, False],
            }
        )
        sub_df = pd.DataFrame({"adsh": [1, 2], "version": [2022, 2023]})
        arcs = pd.DataFrame(
            {
                "version": [2022, 2022, 2023, 2023],
                "seq": [0, 0, 0, 0],
                "from": ["A", "A", "D", "D"],
                "to": ["B", "C", "B", "C"],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )

        result = apply_arcs_by_version(figures, sub_df, arcs, logger=_Logger())

        adsh1_tags = set(result[result["adsh"] == 1]["tag"].astype(str))
        adsh2_tags = set(result[result["adsh"] == 2]["tag"].astype(str))
        self.assertIn("A", adsh1_tags)
        self.assertNotIn("D", adsh1_tags)
        self.assertIn("D", adsh2_tags)
        self.assertNotIn("A", adsh2_tags)

    def test_filter_unreliable_arcs_safe_default_when_none_qualify(self):
        figures = pd.DataFrame(
            {
                "adsh": [1, 1, 1],
                "tag": pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
                "reported_figure": [100.0, 40.0, 60.0],
            }
        )
        sub_df = pd.DataFrame({"adsh": [1], "version": [2022]})
        arcs = pd.DataFrame(
            {
                "version": [2022, 2022],
                "statement": ["sfp-cls", "sfp-cls"],
                "seq": [0, 0],
                "from": ["A", "A"],
                "to": ["B", "C"],
                "weight": [1.0, 1.0],
            }
        )

        filtered, stats = filter_unreliable_arcs(
            figures_df=figures,
            sub_df=sub_df,
            arcs_all_years_df=arcs,
            min_tests_per_equation=10,
            logger=_Logger(),
        )

        pd.testing.assert_frame_equal(filtered.reset_index(drop=True), arcs.reset_index(drop=True))
        self.assertIn("n_tests", stats.columns)

    def test_expand_arcs_with_single_variable_rearrangements(self):
        arcs = pd.DataFrame(
            {
                "version": [2022, 2022],
                "statement": ["sfp-cls", "sfp-cls"],
                "seq": [0, 0],
                "from": ["Total", "Total"],
                "to": ["X", "Y"],
                "weight": [1.0, -1.0],
            }
        )

        expanded = expand_arcs_with_single_variable_rearrangements(
            arcs, renumber_seq=False, enforce_signature_unique_seq=False
        )

        self.assertGreater(len(expanded), len(arcs))
        x_eq = expanded[(expanded["from"] == "X") & (expanded["seq"] == 1)]
        y_eq = expanded[(expanded["from"] == "Y") & (expanded["seq"] == 1)]
        self.assertEqual(len(x_eq), 2)
        self.assertEqual(len(y_eq), 2)


class ComputeCompanyfactsTests(unittest.TestCase):
    def test_filter_arcs_to_tags_and_align_arc_tag_dtype(self):
        arcs = pd.DataFrame(
            {
                "from": ["A", "A", "Z"],
                "to": ["B", "C", "B"],
                "weight": [1.0, 1.0, 1.0],
            }
        )
        cats = pd.Index(["A", "B", "C"])

        filtered = filter_arcs_to_tags(arcs, cats)
        aligned = align_arc_tag_dtype(filtered, cats)

        self.assertEqual(len(filtered), 2)
        self.assertTrue(isinstance(aligned["from"].dtype, CategoricalDtype))
        self.assertTrue(isinstance(aligned["to"].dtype, CategoricalDtype))

    def test_compute_missing_with_arcs_end_to_end_minimal(self):
        cats = pd.Categorical(["B", "C"], categories=["A", "B", "C"])
        figures = pd.DataFrame(
            {
                "adsh": [1, 1],
                "tag": cats,
                "reported_figure": [40.0, 60.0],
                "quarterly_figure": [40.0, 60.0],
                "reported_figure_py": [4.0, 6.0],
                "quarterly_figure_py": [4.0, 6.0],
                "is_computed": [False, False],
            }
        )
        sub_df = pd.DataFrame({"adsh": [1], "version": [2022]})
        arcs = pd.DataFrame(
            {
                "version": [2022, 2022],
                "statement": ["sfp-cls", "sfp-cls"],
                "seq": [0, 0],
                "from": ["A", "A"],
                "to": ["B", "C"],
                "weight": [1.0, 1.0],
            }
        )

        out = compute_missing_with_arcs(
            logger=_Logger(),
            figures_df=figures,
            sub_df=sub_df,
            arcs_all_years_df=arcs,
            validate_arcs=False,
            expand_arcs=False,
            enforce_tag_category=False,
        )

        a_val = out.loc[out["tag"].astype(str) == "A", "reported_figure"].iloc[0]
        self.assertEqual(a_val, 100.0)
        self.assertIn("A", set(out["tag"].astype(str)))


if __name__ == "__main__":
    unittest.main()
