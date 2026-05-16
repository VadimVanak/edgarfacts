import pandas as pd

from edgarfacts.extract import extract_fsd_submissions
import edgarfacts.extract.pipeline as pipeline


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_extract_fsd_submissions_reads_all_periods_without_previous_submissions(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        pipeline,
        "read_tickers",
        lambda fetcher: pd.DataFrame({"ticker": ["msft", "nvda"], "cik": [1, 2]}),
    )
    monkeypatch.setattr(pipeline, "read_periods", lambda fetcher: [(2024, 4), (2025, 1)])

    def fake_read_submissions_parallel(period_arr, fetcher, valid_ciks, logger):
        calls["period_arr"] = list(period_arr)
        calls["valid_ciks"] = list(valid_ciks)
        return pd.DataFrame(
            {
                "adsh": [10, 20],
                "cik": [1, 2],
                "accepted": pd.to_datetime(["2024-11-15", "2025-02-15"]),
            }
        )

    monkeypatch.setattr(pipeline, "read_submissions_parallel", fake_read_submissions_parallel)

    sub = extract_fsd_submissions(object(), DummyLogger())

    assert calls["period_arr"] == [(2024, 4), (2025, 1)]
    assert calls["valid_ciks"] == [1, 2]
    assert sub["adsh"].tolist() == [10, 20]


def test_extract_fsd_submissions_only_reads_periods_after_previous_submission(monkeypatch):
    calls = {}
    previous_sub = pd.DataFrame(
        {
            "adsh": [1, 2],
            "cik": [1, 1],
            "accepted": pd.to_datetime(["2024-11-15", "2025-02-01"]),
        }
    )

    monkeypatch.setattr(
        pipeline,
        "read_tickers",
        lambda fetcher: pd.DataFrame({"ticker": ["msft"], "cik": [1]}),
    )
    monkeypatch.setattr(
        pipeline,
        "read_periods",
        lambda fetcher: [(2024, 4), (2025, 1), (2025, 2)],
    )

    def fake_read_submissions_parallel(period_arr, fetcher, valid_ciks, logger):
        calls["period_arr"] = list(period_arr)
        return pd.DataFrame(
            {
                "adsh": [3],
                "cik": [1],
                "accepted": pd.to_datetime(["2025-05-01"]),
            }
        )

    monkeypatch.setattr(pipeline, "read_submissions_parallel", fake_read_submissions_parallel)

    sub = extract_fsd_submissions(object(), DummyLogger(), prev_sub=previous_sub)

    assert calls["period_arr"] == [(2025, 2)]
    assert sub["adsh"].tolist() == [1, 2, 3]


def test_extract_fsd_submissions_returns_previous_submissions_when_no_new_periods(monkeypatch):
    previous_sub = pd.DataFrame(
        {
            "adsh": [1],
            "cik": [1],
            "accepted": pd.to_datetime(["2025-02-01"]),
        }
    )

    monkeypatch.setattr(
        pipeline,
        "read_tickers",
        lambda fetcher: pd.DataFrame({"ticker": ["msft"], "cik": [1]}),
    )
    monkeypatch.setattr(pipeline, "read_periods", lambda fetcher: [(2024, 4), (2025, 1)])
    monkeypatch.setattr(
        pipeline,
        "read_submissions_parallel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not read")),
    )

    sub = extract_fsd_submissions(object(), DummyLogger(), prev_sub=previous_sub)

    assert sub is previous_sub
