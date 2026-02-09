# src/edgarfacts/validation/checks.py
"""
Production integrity checks ("emergency brake") for edgarfacts outputs.

These checks are intentionally strict and designed to detect silent extraction failures
(e.g., partial downloads, parsing drift, upstream format changes).

They are *not* meant to guarantee extraction always succeeds—rather, they provide a
high-confidence way to fail fast when something went wrong.

Important invariants (must not change):
- DataFrame column sets and ordering
- dtypes (we standardize on datetime64[s])
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_subs(logger, sub: pd.DataFrame) -> None:
    """
    Validate the submissions DataFrame.
    Raises AssertionError on failure.
    """
    # Basic size sanity (production-scale)
    assert len(sub) > 200_000, "Too few lines in submissions"

    # Exact column count expected
    assert len(sub.columns) == 10, "Submissions should have exactly 10 columns"

    # Expected schema (datetime64[s] everywhere)
    expected = {
        "adsh": np.dtype("int64"),
        "cik": np.dtype("int64"),
        "sic": np.dtype("int64"),
        "form": np.dtype("O"),
        "period": np.dtype("datetime64[s]"),
        "accepted": np.dtype("datetime64[s]"),
        "version": np.dtype("int64"),
        "amendment_adsh": np.dtype("int64"),
        "is_amended": np.dtype("bool"),
        # ticker handled separately because dtype object differs by pandas version,
        # but must be categorical.
        "ticker": "category",
    }

    for c in sub.columns:
        if c == "ticker":
            assert isinstance(sub["ticker"].dtype, pd.CategoricalDtype), "ticker must be categorical"
        else:
            assert sub[c].dtype == expected[c], f"Wrong type for column {c}: {sub[c].dtype} != {expected[c]}"

    # Range checks
    assert sub["adsh"].max() < 1e16, "Some ADSH values are too high"
    assert sub["sic"].max() < 10000, "Some SIC values are too high"
    assert sub["period"].min().year > 2003, "Some periods are too early"
    assert sub["period"].max().year < 2030, "Some periods are too late"
    assert sub["accepted"].min().year > 2003, "Some acceptance dates are too early"
    assert sub["accepted"].max().year < 2030, "Some acceptance dates are too late"

    # amendment_adsh consistency
    assert np.isin(np.setdiff1d(sub["amendment_adsh"], [0]), sub["adsh"].unique()).all(), (
        "Some amendment_adsh are not in ADSH list"
    )
    assert len(sub.query("amendment_adsh>0 and not is_amended")) == 0, (
        "amendment_adsh!=0, but is_amended flag is False"
    )
    assert len(sub.query("amendment_adsh==0 and is_amended")) == 0, (
        "amendment_adsh=0, but is_amended flag is True"
    )

    # Version plausibility checks (drift-resistant)
    assert (sub["version"] == 0).sum() == 0, "There are reports without versions"
    
    min_allowed = 2008
    max_year_in_data = int(max(sub["accepted"].dt.year.max(), sub["period"].dt.year.max()))
    max_allowed = max_year_in_data + 2
    
    bad = ~sub["version"].between(min_allowed, max_allowed)
    assert bad.sum() == 0, (
        f"Found {bad.sum()} rows with implausible version years "
        f"(allowed [{min_allowed}, {max_allowed}])"
    )
    
    ahead = sub["version"] > (sub["accepted"].dt.year + 3)
    assert ahead.sum() == 0, "Some versions are far in the future vs acceptance year"
    
    # Forms sanity
    assert (~sub["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])).sum() == 0, "There are unknown forms"

    # Selective check (exact match for known examples)
    subextr = sub[(sub.adsh == 119312522137021) | (sub.adsh == 110465923054237)].reset_index(drop=True)
    subextr["ticker"] = subextr["ticker"].astype(str)

    data = {
        "adsh": [119312522137021, 110465923054237],
        "cik": [1878897, 1902700],
        "sic": [6531, 2833],
        "form": ["10-K/A", "10-K/A"],
        "period": [
            pd.to_datetime("2021-12-31").to_datetime64().astype("datetime64[s]"),
            pd.to_datetime("2022-12-31").to_datetime64().astype("datetime64[s]"),
        ],
        "accepted": [
            pd.to_datetime("2022-05-02 21:18:57").to_datetime64().astype("datetime64[s]"),
            pd.to_datetime("2023-05-01 15:10:29").to_datetime64().astype("datetime64[s]"),
        ],
        "version": [2021, 2021],
        "amendment_adsh": [0, 141057823001429],
        "is_amended": [False, True],
        "ticker": pd.Categorical(["doug", "pgff"]),
    }
    expected_df = pd.DataFrame(data)

    # Ensure same categorical dtype behavior for compare
    expected_df["ticker"] = expected_df["ticker"].astype(str)

    assert len(subextr.compare(expected_df)) == 0, "Selective check failed"

    # Known edge cases: few reports where period is after acceptance date.
    future_periods = [
        88616314000119,
        72174814000106,
        109690621001302,
        156276213000125,
        147124212001044,
        149315224018671,
        162528517000005,
        159991622000122,
        183568124000069,
    ]
    assert len(sub[(sub["accepted"] < sub["period"]) & ~sub["adsh"].isin(future_periods)]) == 0, (
        "There are reports with future periods"
    )

    assert len(sub.query("version==0")) == 0, "There are reports without versions"

    logger.info("Submissions checks OK")


def check_figures(logger, df: pd.DataFrame, sub: pd.DataFrame) -> None:
    """
    Validate the facts DataFrame.
    Raises AssertionError on failure.
    """
    assert len(df) > 50_000_000, "Too few lines in facts"
    assert len(df.columns) == 5, "Facts should have exactly 5 columns"

    expected = {
        "adsh": np.dtype("int64"),
        "tag": "category",
        "start": np.dtype("datetime64[s]"),
        "end": np.dtype("datetime64[s]"),
        "value": np.dtype("float64"),
    }

    for c in df.columns:
        if c == "tag":
            assert isinstance(df["tag"].dtype, pd.CategoricalDtype), "tag must be categorical"
        else:
            assert df[c].dtype == expected[c], f"Wrong type for column {c}: {df[c].dtype} != {expected[c]}"

    assert (df["start"] <= df["end"]).all(), "Start date after end date"

    # Selective check: known sample for specific ADSH/tags
    figsample = df[
        (df.adsh == 156459021039151)
        & df["tag"].isin(
            [
                "AdvertisingExpense",
                "AllocatedShareBasedCompensationExpense",
                "AmortizationOfIntangibleAssets",
            ]
        )
    ].reset_index(drop=True)

    data = {
        "adsh": [156459021039151] * 9,
        "tag": ["AdvertisingExpense"] * 3
        + ["AllocatedShareBasedCompensationExpense"] * 3
        + ["AmortizationOfIntangibleAssets"] * 3,
        "start": pd.to_datetime(
            [
                "2018-07-01",
                "2019-07-01",
                "2020-07-01",
                "2018-07-01",
                "2019-07-01",
                "2020-07-01",
                "2018-07-01",
                "2019-07-01",
                "2020-07-01",
            ]
        ).astype("datetime64[s]"),
        "end": pd.to_datetime(
            [
                "2019-06-30",
                "2020-06-30",
                "2021-06-30",
                "2019-06-30",
                "2020-06-30",
                "2021-06-30",
                "2019-06-30",
                "2020-06-30",
                "2021-06-30",
            ]
        ).astype("datetime64[s]"),
        "value": [
            1.6e9,
            1.6e9,
            1.5e9,
            4.652e9,
            5.289e9,
            6.118e9,
            1.9e9,
            1.6e9,
            1.6e9,
        ],
    }
    expected_df = pd.DataFrame(data)
    expected_df["tag"] = pd.Categorical(expected_df["tag"], categories=df["tag"].cat.categories)

    assert len(figsample.compare(expected_df)) == 0, "Selective check failed"

    # Facts not assigned to a record in sub (tolerate a small number)
    assert len(df[(~df["adsh"].isin(sub["adsh"]))]["adsh"].unique()) < 55, (
        "There are figures not assigned to a report"
    )

    # Known reports without figures (ignore these, but check all others)
    sub_no_data = [
        114420411065305,
        114420411053088,
        104746909007400,
        110852420000029,
        128703223000355,
        155837022007084,
        119312511051841,
        138713117005367,
        138713117003911,
        138713117002568,
        119312512266785,
        107878220001010,
        144530513002979,
        158798723000073,
        119312519209634,
        160971115000009,
        165365316000008,
    ]
    assert len(sub[~sub["adsh"].isin(df["adsh"]) & ~sub["adsh"].isin(sub_no_data)]) == 0, (
        "There are reports without figures"
    )

    logger.info("Figures checks OK")


def check_submissions_and_facts(logger, df: pd.DataFrame, sub: pd.DataFrame) -> None:
    """
    Run all production checks. Intended as an emergency brake.
    """
    check_subs(logger, sub)
    check_figures(logger, df, sub)
