import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from edgarfacts.transforms.outliers import (
    attach_cik,
    build_outlier_dataset,
    classify_outlier_multiplier,
    compute_value_adj,
    remove_outliers_parallel,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


def _scaled_error_facts():
    return pd.DataFrame(
        {
            "adsh": [1, 2, 3, 4],
            "tag": ["Revenue"] * 4,
            "start": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]),
            "end": pd.to_datetime(["2020-03-30", "2020-06-29", "2020-09-28", "2020-12-29"]),
            "value": [100.0, 120.0, 120000.0, 120.0],
        }
    )


class OutlierTests(unittest.TestCase):
    def test_remove_outliers_parallel_preserves_schema_and_corrects_1e3(self):
        facts = _scaled_error_facts()
        sub = pd.DataFrame({"adsh": [1, 2, 3, 4], "cik": [10, 10, 10, 10]})

        fixed, n_outliers = remove_outliers_parallel(facts, sub, _Logger(), use_process_pool=False)

        self.assertListEqual(fixed.columns.tolist(), facts.columns.tolist())
        self.assertGreaterEqual(n_outliers, 1)
        self.assertEqual(float(fixed.loc[fixed["adsh"].eq(3), "value"].iloc[0]), 120.0)

    def test_classify_outlier_multiplier_flags_without_mutating_value(self):
        facts = _scaled_error_facts()
        sub = pd.DataFrame({"adsh": [1, 2, 3, 4], "cik": [10, 10, 10, 10]})
        features = compute_value_adj(attach_cik(facts, sub))
        original = features["value"].copy()

        mult = classify_outlier_multiplier(features)

        self.assertEqual(float(mult.loc[features["adsh"].eq(3)].iloc[0]), 1e-3)
        pd.testing.assert_series_equal(features["value"], original)

    def test_build_outlier_dataset_two_versions_duplicates_and_parent_schema(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 1, 2, 3, 4, 5],
                "tag": ["Revenue", "Revenue", "Assets", "Revenue", "Revenue", "Assets"],
                "start": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-01",
                        "2020-01-01",
                        "2020-04-01",
                        "2021-01-01",
                        "2021-01-01",
                    ]
                ),
                "end": pd.to_datetime(
                    [
                        "2020-03-30",
                        "2020-03-30",
                        "2020-03-30",
                        "2020-06-29",
                        "2021-03-30",
                        "2021-03-30",
                    ]
                ),
                "value": [100.0, 101.0, 1000.0, 110.0, 120000.0, 2000.0],
            }
        )
        submissions = pd.DataFrame(
            {
                "adsh": [1, 2, 3, 4, 5],
                "cik": [10, 10, 10, 20, 20],
                "version": ["v1", "v1", "v1", "v2", "v2"],
                "accepted": pd.to_datetime(["2021-01-01"] * 5),
                "sic": ["1234"] * 5,
                "form": ["10-K", "10-K", "10-Q", "10-K/A", "10-K"],
            }
        )
        arcs = pd.DataFrame(
            {
                "version": ["v1"],
                "statement": ["IS"],
                "seq": [1],
                "from": ["Assets"],
                "to": ["Revenue"],
                "weight": [1.0],
            }
        )

        with TemporaryDirectory() as td:
            paths = build_outlier_dataset(facts, submissions, arcs, _Logger(), target_path=td)
            out = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

        removed_columns = {
            "cik",
            "is_instant",
            "accepted_year",
            "version",
            "sic",
            "form",
            "is_amended",
            "sign",
            "best_overlap_fraction",
            "duplicate_value_count",
            "duplicate_unique_value_count",
            "submission_size",
            "submission_median_log10",
            "submission_mad_log10",
            "submission_q25_log10",
            "submission_q75_log10",
            "submission_majority_scale",
            "tag_global_median_log10",
            "tag_global_mad_log10",
            "parent1_value",
            "parent1_weight",
            "parent1_frequency",
            "parent2_value",
            "parent2_weight",
            "parent2_frequency",
        }
        self.assertTrue(removed_columns.isdisjoint(out.columns))
        self.assertIn("outlier_multiplier", out.columns)
        dup = out[(out["adsh"].eq(1)) & (out["tag"].astype(str).eq("Revenue"))]
        self.assertTrue(dup["duplicate_majority_value"].isna().all())

    def test_build_outlier_dataset_groups_versions_up_to_max_records(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 2, 3],
                "tag": ["Revenue", "Revenue", "Revenue"],
                "start": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"]),
                "end": pd.to_datetime(["2020-03-30", "2020-06-29", "2020-09-28"]),
                "value": [100.0, 110.0, 120.0],
            }
        )
        submissions = pd.DataFrame(
            {
                "adsh": [1, 2, 3],
                "cik": [10, 10, 10],
                "version": ["v1", "v2", "v3"],
            }
        )
        arcs = pd.DataFrame(columns=["version", "statement", "seq", "from", "to", "weight"])

        with TemporaryDirectory() as td:
            paths = build_outlier_dataset(
                facts, submissions, arcs, _Logger(), max_records=2, target_path=td
            )
            first_group = pd.read_parquet(paths[0])
            second_group = pd.read_parquet(paths[1])

        self.assertTrue((first_group["tag_occurrence_count"] == 2).all())
        self.assertTrue((second_group["tag_occurrence_count"] == 1).all())

    def test_build_outlier_dataset_keeps_oversized_version_as_own_group(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 1, 2],
                "tag": ["Revenue", "Assets", "Revenue"],
                "start": pd.to_datetime(["2020-01-01"] * 3),
                "end": pd.to_datetime(["2020-03-30"] * 3),
                "value": [100.0, 1000.0, 110.0],
            }
        )
        submissions = pd.DataFrame(
            {
                "adsh": [1, 2],
                "cik": [10, 10],
                "version": ["v1", "v2"],
            }
        )
        arcs = pd.DataFrame(columns=["version", "statement", "seq", "from", "to", "weight"])

        with TemporaryDirectory() as td:
            paths = build_outlier_dataset(
                facts, submissions, arcs, _Logger(), max_records=1, target_path=td
            )
            first_group = pd.read_parquet(paths[0])
            second_group = pd.read_parquet(paths[1])

        first_revenue = first_group[first_group["tag"].astype(str).eq("Revenue")]
        second_revenue = second_group[second_group["tag"].astype(str).eq("Revenue")]
        self.assertTrue((first_revenue["tag_occurrence_count"] == 1).all())
        self.assertTrue((second_revenue["tag_occurrence_count"] == 1).all())

    def test_build_outlier_dataset_empty_arcs_omits_parent_columns(self):
        facts = pd.DataFrame(
            {
                "adsh": [1],
                "tag": ["Revenue"],
                "start": pd.to_datetime(["2020-01-01"]),
                "end": pd.to_datetime(["2020-03-30"]),
                "value": [100.0],
            }
        )
        submissions = pd.DataFrame({"adsh": [1], "cik": [10], "version": ["v1"]})
        arcs = pd.DataFrame(columns=["version", "statement", "seq", "from", "to", "weight"])

        with TemporaryDirectory() as td:
            paths = build_outlier_dataset(facts, submissions, arcs, _Logger(), target_path=td)
            out = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

        self.assertNotIn("parent1_value", out.columns)
        self.assertNotIn("parent2_value", out.columns)

    def test_build_outlier_dataset_saves_chunk_files(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 2],
                "tag": ["Revenue", "Revenue"],
                "start": pd.to_datetime(["2020-01-01", "2020-04-01"]),
                "end": pd.to_datetime(["2020-03-30", "2020-06-29"]),
                "value": [100.0, 110.0],
            }
        )
        submissions = pd.DataFrame({"adsh": [1, 2], "cik": [10, 10], "version": ["v1", "v2"]})
        arcs = pd.DataFrame(columns=["version", "statement", "seq", "from", "to", "weight"])

        with TemporaryDirectory() as td:
            paths = build_outlier_dataset(
                facts, submissions, arcs, _Logger(), max_records=1, target_path=td
            )
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(p).exists() for p in paths))


if __name__ == "__main__":
    unittest.main()
