import unittest

import pandas as pd

from edgarfacts.transforms import __all__ as transforms_all
from edgarfacts.transforms.amendments import (
    apply_canonical_adsh,
    build_canonical_adsh_map,
    canonicalize_and_merge_amendments,
    dedupe_latest_by_acceptance,
)
from edgarfacts.transforms.config import (
    ARC_MAX_PASSES,
    DATETIME_DTYPE,
    MAX_ABS_FIGURE_VALUE,
    PERIOD_MATCH_TOLERANCE_DAYS,
)
from edgarfacts.transforms.figures import (
    _normalize_fact_dtypes,
    _remove_contradicting_values,
    _remove_huge_values,
)
from edgarfacts.transforms.outliers import attach_cik, compute_value_adj, remove_outliers_parallel
from edgarfacts.transforms.periods import (
    _enrich_sub_with_windows,
    compute_instant_period_values,
    compute_period_values,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


class AmendmentsTests(unittest.TestCase):
    def test_build_canonical_adsh_map_resolves_chain(self):
        sub = pd.DataFrame({"adsh": [1, 2, 3], "amendment_adsh": [2, 3, 0]})

        mapping = build_canonical_adsh_map(sub)
        got = dict(zip(mapping["adsh_original"], mapping["adsh_canonical"]))

        self.assertEqual(got[1], 3)
        self.assertEqual(got[2], 3)
        self.assertEqual(got[3], 3)

    def test_apply_canonical_adsh_rewrites_known_and_keeps_unknown(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 9],
                "tag": ["Revenue", "Revenue"],
                "start": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "end": pd.to_datetime(["2023-12-31", "2023-12-31"]),
                "value": [10.0, 20.0],
            }
        )
        mapping = pd.DataFrame({"adsh_original": [1], "adsh_canonical": [3]})

        out = apply_canonical_adsh(facts, mapping)

        self.assertListEqual(out["adsh"].tolist(), [3, 9])

    def test_dedupe_latest_by_acceptance_keeps_latest(self):
        facts = pd.DataFrame(
            {
                "adsh": [3, 3],
                "tag": ["Revenue", "Revenue"],
                "start": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "end": pd.to_datetime(["2023-12-31", "2023-12-31"]),
                "value": [10.0, 11.0],
            }
        )
        sub = pd.DataFrame(
            {
                "adsh": [1, 3],
                "amendment_adsh": [3, 0],
                "accepted": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            }
        )

        out = dedupe_latest_by_acceptance(facts, sub)

        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["value"], 10.0)

    def test_canonicalize_and_merge_amendments_integration(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 2],
                "tag": ["Revenue", "Revenue"],
                "start": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "end": pd.to_datetime(["2023-12-31", "2023-12-31"]),
                "value": [10.0, 20.0],
            }
        )
        sub = pd.DataFrame(
            {
                "adsh": [1, 2],
                "amendment_adsh": [2, 0],
                "accepted": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            }
        )

        out = canonicalize_and_merge_amendments(facts, sub)
        # Original remains; amended keeps own value and does not get overwritten.
        self.assertEqual(len(out), 2)
        v1 = out.loc[out["adsh"] == 1, "value"].iloc[0]
        v2 = out.loc[out["adsh"] == 2, "value"].iloc[0]
        self.assertEqual(v1, 10.0)
        self.assertEqual(v2, 20.0)

    def test_canonicalize_and_merge_amendments_copies_only_missing_keys_to_amended(self):
        facts = pd.DataFrame(
            {
                "adsh": [894717000137, 894717000137, 894718000047],
                "tag": ["Revenue", "CostOfRevenue", "Revenue"],
                "start": pd.to_datetime(["2017-03-01", "2017-03-01", "2017-03-01"]),
                "end": pd.to_datetime(["2017-08-31", "2017-08-31", "2017-08-31"]),
                "value": [100.0, 60.0, 120.0],
            }
        )
        # Matches the user-described direction: original adsh points to amendment adsh.
        sub = pd.DataFrame(
            {
                "adsh": [894718000047, 894717000137],
                "amendment_adsh": [0, 894718000047],
                "accepted": pd.to_datetime(["2018-04-19 06:24:00", "2017-10-03 06:32:00"]),
            }
        )

        out = canonicalize_and_merge_amendments(facts, sub)

        # Original adsh keeps all its rows
        orig = out[out["adsh"] == 894717000137]
        self.assertSetEqual(set(orig["tag"].tolist()), {"Revenue", "CostOfRevenue"})

        # Amended keeps its own Revenue value and receives missing CostOfRevenue only.
        amended = out[out["adsh"] == 894718000047]
        self.assertSetEqual(set(amended["tag"].tolist()), {"Revenue", "CostOfRevenue"})
        self.assertEqual(amended.loc[amended["tag"] == "Revenue", "value"].iloc[0], 120.0)
        self.assertEqual(amended.loc[amended["tag"] == "CostOfRevenue", "value"].iloc[0], 60.0)


class OutliersTests(unittest.TestCase):
    def test_attach_cik_and_compute_value_adj(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 1],
                "tag": ["Revenue", "Cash"],
                "start": pd.to_datetime(["2023-01-01", "2023-12-31"]),
                "end": pd.to_datetime(["2023-12-31", "2023-12-31"]),
                "value": [365.0, 50.0],
            }
        )
        sub = pd.DataFrame({"adsh": [1], "cik": [100]})

        merged = attach_cik(facts, sub)
        out = compute_value_adj(merged)

        self.assertIn("cik", merged.columns)
        self.assertAlmostEqual(out.loc[out["tag"] == "Revenue", "value_adj"].iloc[0], 90.24725274725274)
        self.assertEqual(out.loc[out["tag"] == "Cash", "value_adj"].iloc[0], 50.0)

    def test_remove_outliers_parallel_preserves_schema(self):
        facts = pd.DataFrame(
            {
                "adsh": [1],
                "tag": ["Revenue"],
                "start": pd.to_datetime(["2023-01-01"]),
                "end": pd.to_datetime(["2023-12-31"]),
                "value": [100.0],
            }
        )
        sub = pd.DataFrame({"adsh": [1], "cik": [100]})

        out, n_out = remove_outliers_parallel(facts, sub, logger=_Logger(), use_process_pool=False)

        self.assertEqual(n_out, 0)
        self.assertListEqual(list(out.columns), ["tag", "adsh", "start", "end", "value"])


class PeriodsTests(unittest.TestCase):
    def test_compute_period_values_returns_empty_when_no_window_overlap(self):
        facts = pd.DataFrame(
            {
                "adsh": [1],
                "tag": ["Revenue"],
                "start": pd.to_datetime(["2023-01-01"]),
                "end": pd.to_datetime(["2023-12-31"]),
                "value": [100.0],
            }
        )
        windows = pd.DataFrame(
            {
                "adsh": [1],
                "enum": [1],
                "start": pd.to_datetime(["2022-01-01"]),
                "end": pd.to_datetime(["2022-12-31"]),
            }
        )
        sub = pd.DataFrame({"adsh": [1]})

        values, enriched = compute_period_values(facts, sub, windows)

        self.assertEqual(len(values), 0)
        self.assertIn("start_rep", enriched.columns)

    def test_compute_instant_period_values_maps_window_ends(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 1],
                "tag": ["Cash", "Cash"],
                "start": pd.to_datetime(["2023-12-31", "2022-12-31"]),
                "end": pd.to_datetime(["2023-12-31", "2022-12-31"]),
                "value": [50.0, 40.0],
            }
        )
        windows = pd.DataFrame(
            {
                "adsh": [1, 1],
                "enum": [1, 3],
                "start": pd.to_datetime(["2023-01-01", "2022-01-01"]),
                "end": pd.to_datetime(["2023-12-31", "2022-12-31"]),
            }
        )

        out = compute_instant_period_values(facts, windows)

        self.assertEqual(out.iloc[0]["value1"], 50.0)
        self.assertEqual(out.iloc[0]["value3"], 40.0)

    def test_enrich_sub_with_windows_adds_expected_columns(self):
        sub = pd.DataFrame({"adsh": [1]})
        windows = pd.DataFrame(
            {
                "adsh": [1],
                "enum": [1],
                "start": pd.to_datetime(["2023-01-01"]),
                "end": pd.to_datetime(["2023-12-31"]),
            }
        )

        out = _enrich_sub_with_windows(sub, windows)

        self.assertIn("start_rep", out.columns)
        self.assertIn("end_rep", out.columns)


class FiguresAndConfigTests(unittest.TestCase):
    def test_normalize_remove_conflicts_and_huge_values(self):
        facts = pd.DataFrame(
            {
                "adsh": [1, 1, 1],
                "tag": pd.Categorical(["Revenue", "Revenue", "NotionalAmountOfDerivatives"]),
                "start": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"]),
                "end": pd.to_datetime(["2023-12-31", "2023-12-31", "2023-12-31"]),
                "value": [10.0, 11.0, MAX_ABS_FIGURE_VALUE * 10],
            }
        )

        norm = _normalize_fact_dtypes(facts)
        no_conflicts = _remove_contradicting_values(norm)
        no_huge = _remove_huge_values(norm)

        self.assertEqual(str(norm["start"].dtype), DATETIME_DTYPE)
        self.assertEqual(len(no_conflicts), 1)  # conflicting Revenue rows removed entirely
        self.assertEqual(len(no_huge), 3)  # huge whitelisted tag retained; normal rows remain

    def test_public_api_and_config_sanity(self):
        self.assertIn("build_base_figures", transforms_all)
        self.assertGreaterEqual(ARC_MAX_PASSES, 1)
        self.assertGreater(PERIOD_MATCH_TOLERANCE_DAYS, 0)


if __name__ == "__main__":
    unittest.main()
