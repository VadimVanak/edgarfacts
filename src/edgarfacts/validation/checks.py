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



def check_build_base_figures_selected_results(df: pd.DataFrame, sub: pd.DataFrame) -> None:
    """
    Run selected deterministic checks for outputs of ``build_base_figures``.

    Parameters
    ----------
    df:
        Figures output from ``build_base_figures``. Supports either:
        - ['adsh','tag','value1','value2','value3','value4']
        - ['adsh','tag','reported_figure','quarterly_figure','reported_figure_py','quarterly_figure_py']
    sub:
        Enriched submissions output from ``build_base_figures`` containing
        reporting-window columns.

    Raises
    ------
    AssertionError
        If any expected reference rows are missing.
    """
    if {"reported_figure", "quarterly_figure", "reported_figure_py", "quarterly_figure_py"}.issubset(df.columns):
        figure_cols = {
            "value1": "reported_figure",
            "value2": "quarterly_figure",
            "value3": "reported_figure_py",
            "value4": "quarterly_figure_py",
        }
    elif {"value1", "value2", "value3", "value4"}.issubset(df.columns):
        figure_cols = {"value1": "value1", "value2": "value2", "value3": "value3", "value4": "value4"}
    else:
        raise ValueError(
            "df must contain either value1..value4 or reported_figure/quarterly_figure/reported_figure_py/quarterly_figure_py"
        )

    expected_rows_1 = pd.concat(
        (
            pd.DataFrame(
                {
                    "adsh": [
                        95012309025033, 708410000050, 95012309072007,
                        95012310069540, 103764612000027, 412714000046, 104746915006136,
                        156459020037281, 165495421005555, 126493121000056, 126493121000056,
                        110465922081498, 110465921094125, 126493124000014,
                        490410000112, 628119000013, 4055414000023, 86766516000218,
                        114420417033785, 95017022008943, 184078023000022,
                    ],
                    "tag": [
                        "EffectOfExchangeRateOnCashAndCashEquivalents",
                        "OtherComprehensiveIncomeDefinedBenefitPlansAdjustmentNetOfTaxPortionAttributableToParent",
                        "CostOfRevenue",
                        "EquityMethodInvestmentDividendsOrDistributions",
                        "TreasuryStockValueAcquiredCostMethod",
                        "PaymentsOfDividendsCommonStock",
                        "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
                        "ProceedsFromLinesOfCredit",
                        "WeightedAverageNumberOfSharesOutstandingBasic",
                        "IncreaseDecreaseInAccruedLiabilities",
                        "NetCashProvidedByUsedInFinancingActivities",
                        "RevenueRemainingPerformanceObligation",
                        "RevenueRemainingPerformanceObligation",
                        "AccruedLiabilitiesCurrent",
                        "EntityCommonStockSharesOutstanding",
                        "DeferredIncomeTaxAssetsNet",
                        "UndistributedEarningsOfForeignSubsidiaries",
                        "DebtCurrent",
                        "LeaseholdImprovementsGross",
                        "PrepaidExpenseAndOtherAssetsCurrent",
                        "AccountsNotesAndLoansReceivableNetCurrent",
                    ],
                    "value1": [
                        5.675000e06, -1.300000e07, 3.582802e09, 3.425000e06, 6.372100e07,
                        4.170000e01, 6.470000e07, 1.105670e08, 1.2517412e07, -3.500000e03,
                        4.276910e05, 8.500000e08, 7.500000e08, 1.688026e06, 4.79437027e08,
                        1.634719e09, 1.100000e08, 1.313000e06, 6.926800e04, 1.897020e08, 2.610000e06,
                    ],
                    "value2": [
                        None, None, None, None, None, 2.090000e07, 9.720000e07, 2.504800e07,
                        None, None, None, None, None, 1.688026e06, 4.79437027e08, None,
                        1.100000e08, 1.313000e06, None, None, None,
                    ],
                    "value3": [
                        3.137400e07, -8.000000e06, 4.686412e09, 1.740000e06, 5.720000e07,
                        None, 5.700000e06, 8.343800e07, 1.2555108e07, -3.539000e03, -3.539000e06,
                        None, None, 1.607245e06, None, None, 1.080000e11, None, None, None, None,
                    ],
                    "value4": [
                        None, None, None, None, None, None, 2.200000e06, 1.980100e07,
                        None, None, None, None, None, None, None, None, 1.080000e11,
                        None, None, None, None,
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "adsh": [894718000047, 143774923020794],
                    "tag": ["SalesRevenueNet", "AccountsReceivableNetCurrent"],
                    "value1": [401612000.0, 1165000.0],
                    "value2": [196329000.0, None],
                    "value3": [451156000.0, 1294000.0],
                    "value4": [200790000.0, None],
                }
            ),
            pd.DataFrame(
                {
                    "adsh": [
                        184545923000015, 184545923000015, 184545923000015, 184545923000015, 184545923000015,
                        95017023054855, 95017023054855, 95017024008814, 95017024008814,
                    ],
                    "tag": [
                        "AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities",
                        "AdjustmentsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities",
                        "GainLossOnSaleOfDerivatives",
                        "GainsLossesOnSalesOfAssets",
                        "IncreaseDecreaseInOperatingCapital",
                        "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "Revenues",
                        "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "Revenues",
                    ],
                    "value1": [
                        29158000.0, 29967000.0, -24475000.0, -24475000.0, -809000.0,
                        5.6517e10, 5.6517e10, 1.18537e11, 1.18537e11,
                    ],
                    "value2": [
                        24775000.0, 24775000.0, -24475000.0, -24475000.0, 2.395187e06,
                        np.nan, np.nan, 6.202e10, 6.202e10,
                    ],
                    "value3": [
                        2952000.0, 2627000.0, 0.0, 0.0, 325000.0,
                        5.0122e10, 5.0122e10, 1.02869e11, 1.02869e11,
                    ],
                    "value4": [
                        300000.0, 300000.0, 0.0, 0.0, 1.617396e06,
                        np.nan, np.nan, 5.2747e10, 5.2747e10,
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "adsh": [95017023054855, 95017024008814, 95017024048288, 95017024087843],
                    "tag": [
                        "CostOfGoodsAndServicesSold",
                        "CostOfGoodsAndServicesSold",
                        "CostOfGoodsAndServicesSold",
                        "CostOfGoodsAndServicesSold",
                    ],
                    "value1": [1.6302e10, 3.5925e10, 5.443e10, 7.4114e10],
                    "value2": [np.nan, 1.9623e10, 1.8505e10, 1.9684e10],
                    "value3": [1.5452e10, 3.294e10, 4.9068e10, 6.5863e10],
                    "value4": [np.nan, 1.7488e10, 1.6128e10, 1.6795e10],
                }
            ),
        ),
        ignore_index=True,
    ).rename(columns=figure_cols)

    figure_join_cols = ["adsh", "tag", figure_cols["value1"], figure_cols["value2"], figure_cols["value3"], figure_cols["value4"]]
    missing_fig_rows = len(expected_rows_1) - len(df.merge(expected_rows_1, on=figure_join_cols, how="inner"))
    assert missing_fig_rows == 0, f"{missing_fig_rows} lines were not found in the dataframe"

    check_sub = pd.DataFrame(
        {
            "adsh": [95017024087843, 95017024048288, 95017024008814, 95017023054855],
            "cik": [789019, 789019, 789019, 789019],
            "sic": [7372, 7372, 7372, 7372],
            "form": ["10-K", "10-Q", "10-Q", "10-Q"],
            "period": pd.to_datetime(["2024-06-30", "2024-03-31", "2023-12-31", "2023-09-30"]),
            "accepted": pd.to_datetime(["2024-07-30 16:06:00", "2024-04-25 16:06:00", "2024-01-30 16:06:00", "2023-10-24 16:08:00"]),
            "version": [2023, 2023, 2023, 2023],
            "amendment_adsh": [0, 0, 0, 0],
            "is_amended": [False, False, False, False],
            "ticker": ["msft", "msft", "msft", "msft"],
            "start_rep": pd.to_datetime(["2023-07-01", "2023-07-01", "2023-07-01", "2023-07-01"]),
            "end_rep": pd.to_datetime(["2024-06-30", "2024-03-31", "2023-12-31", "2023-09-30"]),
            "start_q": pd.to_datetime([None, "2024-01-01", "2023-10-01", None]),
            "end_q": pd.to_datetime([None, "2024-03-31", "2023-12-31", None]),
            "start_rep_py": pd.to_datetime(["2022-07-01", "2022-07-01", "2022-07-01", "2022-07-01"]),
            "end_rep_py": pd.to_datetime(["2023-06-30", "2023-03-31", "2022-12-31", "2022-09-30"]),
            "start_q_py": pd.to_datetime([None, "2023-01-01", "2022-10-01", None]),
            "end_q_py": pd.to_datetime([None, "2023-03-31", "2022-12-31", None]),
        }
    )

    sub_join_cols = [c for c in check_sub.columns if c in sub.columns]
    missing_sub_rows = len(check_sub) - len(sub.merge(check_sub[sub_join_cols], on=sub_join_cols, how="inner"))
    assert missing_sub_rows == 0, f"{missing_sub_rows} lines were not found in the dataframe"


def check_submissions_and_facts(logger, df: pd.DataFrame, sub: pd.DataFrame) -> None:
    """
    Run all production checks. Intended as an emergency brake.
    """
    check_subs(logger, sub)
    check_figures(logger, df, sub)
