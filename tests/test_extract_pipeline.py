import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from edgarfacts.extract.pipeline import (
    determine_delta_first_period,
    extract_submissions_and_facts_delta,
    normalize_submission_dtypes,
)
from edgarfacts.extract.submissions_bulk import repair_version


class ExtractPipelineDeltaTests(unittest.TestCase):
    def test_determine_delta_first_period_uses_two_quarter_lookback(self):
        prev_sub = pd.DataFrame(
            {
                "accepted": pd.to_datetime(
                    [
                        "2026-01-10 12:00:00",
                        "2026-05-16 08:30:00",
                        "2025-12-31 23:59:59",
                    ]
                )
            }
        )

        self.assertEqual(determine_delta_first_period(prev_sub), (2025, 4))

    def test_normalize_submission_dtypes_converts_ticker_to_category(self):
        sub = pd.DataFrame(
            {
                "cik": [1],
                "sic": [1234],
                "adsh": [1001],
                "version": [2024],
                "amendment_adsh": [0],
                "period": pd.to_datetime(["2026-03-31"]),
                "accepted": pd.to_datetime(["2026-05-16 08:30:00"]),
                "ticker": ["TEST"],
            }
        )

        normalized = normalize_submission_dtypes(sub)

        self.assertEqual(str(normalized["ticker"].dtype), "category")

    def test_delta_filters_new_submissions_to_rows_with_extracted_facts(self):
        class Logger:
            def info(self, message):
                pass

        prev_df = pd.DataFrame(
            {
                "adsh": [1001],
                "tag": ["Revenue"],
                "start": pd.to_datetime(["2025-01-01"]),
                "end": pd.to_datetime(["2025-12-31"]),
                "value": [10.0],
            }
        )
        prev_sub = pd.DataFrame(
            {
                "cik": [1],
                "sic": [1234],
                "adsh": [1001],
                "version": [2024],
                "amendment_adsh": [0],
                "is_amended": [False],
                "period": pd.to_datetime(["2025-12-31"]),
                "accepted": pd.to_datetime(["2026-05-16 08:30:00"]),
                "form": ["10-K"],
                "ticker": ["OLD"],
            }
        )
        tickers = pd.DataFrame({"cik": [1, 2], "ticker": ["OLD", "NEW"]})
        current_bulk_sub = pd.DataFrame(
            {
                "cik": [1, 2],
                "sic": [1234, 5678],
                "adsh": [2001, 2002],
                "period": pd.to_datetime(["2026-03-31", "2026-03-31"]),
                "accepted": pd.to_datetime(["2026-05-17", "2026-05-18"]),
                "form": ["10-Q", "10-Q"],
                "file": ["a.htm", "b.htm"],
            }
        )
        updated_delta_sub = current_bulk_sub.drop(columns="file").copy()
        updated_delta_sub["version"] = [2025, 0]
        delta_df = pd.DataFrame(
            {
                "adsh": [2001],
                "tag": ["Revenue"],
                "start": pd.to_datetime(["2026-01-01"]),
                "end": pd.to_datetime(["2026-03-31"]),
                "value": [20.0],
            }
        )

        with (
            patch("edgarfacts.extract.pipeline.read_tickers", return_value=tickers),
            patch("edgarfacts.extract.pipeline.read_tags", return_value=np.array(["Revenue"])),
            patch("edgarfacts.extract.pipeline.read_submissions_2", return_value=current_bulk_sub),
            patch("edgarfacts.extract.pipeline.update_version_info", return_value=updated_delta_sub),
            patch("edgarfacts.extract.pipeline.read_missing_figures", return_value=delta_df),
        ):
            df, sub = extract_submissions_and_facts_delta(
                fetcher=None,
                logger=Logger(),
                prev_df=prev_df,
                prev_sub=prev_sub,
            )

        self.assertEqual(set(df["adsh"].tolist()), {1001, 2001})
        self.assertEqual(set(sub["adsh"].tolist()), {1001, 2001})
        self.assertNotIn(2002, sub["adsh"].tolist())
        self.assertEqual(str(sub["ticker"].dtype), "category")

    def test_repair_version_keeps_zero_when_cik_has_no_known_version(self):
        sub = pd.DataFrame(
            {
                "cik": [1, 1],
                "accepted": pd.to_datetime(["2026-05-16", "2026-02-16"]),
                "adsh": [1001, 1002],
                "amendment_adsh": [0, 0],
                "version": [0, 0],
            }
        )

        repaired = repair_version(sub)

        self.assertEqual(repaired["version"].tolist(), [0, 0])
        self.assertEqual(str(repaired["version"].dtype), "int64")


if __name__ == "__main__":
    unittest.main()
