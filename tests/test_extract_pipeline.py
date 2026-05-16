import unittest

import pandas as pd

from edgarfacts.extract.pipeline import determine_delta_first_period
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
