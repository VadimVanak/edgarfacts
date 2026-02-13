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

    expected_rows_1 = pd.DataFrame(
        {
            "adsh": [
                412714000046,
                490410000112,
                628119000013,
                708410000050,
                894718000047,
                4055414000023,
                86766516000218,
                95012309025033,
                95012309072007,
                95012310069540,
                95017022008943,
                95017023054855,
                95017023054855,
                95017023054855,
                95017024008814,
                95017024008814,
                95017024008814,
                95017024048288,
                95017024087843,
                103764612000027,
                104746915006136,
                110465921094125,
                110465922081498,
                114420417033785,
                126493121000056,
                126493124000014,
                143774923020794,
                156459020037281,
                165495421005555,
                184078023000022,
                184545923000015,
                184545923000015,
                184545923000015,
                184545923000015,
                184545923000015,
            ],
            "tag": [
                "PaymentsOfDividendsCommonStock",
                "EntityCommonStockSharesOutstanding",
                "DeferredIncomeTaxAssetsNet",
                "OtherComprehensiveIncomeDefinedBenefitPlansAdjustmentNetOfTaxPortionAttributableToParent",
                "SalesRevenueNet",
                "UndistributedEarningsOfForeignSubsidiaries",
                "DebtCurrent",
                "EffectOfExchangeRateOnCashAndCashEquivalents",
                "CostOfRevenue",
                "EquityMethodInvestmentDividendsOrDistributions",
                "PrepaidExpenseAndOtherAssetsCurrent",
                "CostOfGoodsAndServicesSold",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "CostOfGoodsAndServicesSold",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "CostOfGoodsAndServicesSold",
                "CostOfGoodsAndServicesSold",
                "TreasuryStockValueAcquiredCostMethod",
                "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
                "RevenueRemainingPerformanceObligation",
                "RevenueRemainingPerformanceObligation",
                "LeaseholdImprovementsGross",
                "IncreaseDecreaseInAccruedLiabilities",
                "AccruedLiabilitiesCurrent",
                "AccountsReceivableNetCurrent",
                "ProceedsFromLinesOfCredit",
                "WeightedAverageNumberOfSharesOutstandingBasic",
                "AccountsNotesAndLoansReceivableNetCurrent",
                "AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities",
                "AdjustmentsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities",
                "GainLossOnSaleOfDerivatives",
                "GainsLossesOnSalesOfAssets",
                "IncreaseDecreaseInOperatingCapital",
            ],
            "reported_figure": [
                41.7,
                479437027.0,
                1634719000.0,
                -13000000.0,
                401612000.0,
                110000000.0,
                1313000.0,
                5675000.0,
                3582802000.0,
                3425000.0,
                189702000.0,
                16302000000.0,
                56517000000.0,
                56517000000.0,
                35925000000.0,
                118537000000.0,
                118537000000.0,
                54430000000.0,
                74114000000.0,
                63721000.0,
                64700000.0,
                750000000.0,
                850000000.0,
                69268.0,
                -3500.0,
                1688026.0,
                1165000.0,
                110567000.0,
                12517412.0,
                2610000.0,
                29158000.0,
                29967000.0,
                -24475000.0,
                -24475000.0,
                -809000.0,
            ],
            "quarterly_figure": [
                None,
                479437027.0,
                None,
                None,
                196329000.0,
                None,
                1313000.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                19623000000.0,
                62020000000.0,
                62020000000.0,
                18505000000.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                24475000.0,
                24475000.0,
                -24475000.0,
                -24475000.0,
                None,
            ],
            "reported_figure_py": [
                None,
                None,
                None,
                -8000000.0,
                451156000.0,
                108000000000.0,
                None,
                31374000.0,
                4686412000.0,
                1740000.0,
                None,
                15452000000.0,
                50122000000.0,
                50122000000.0,
                32940000000.0,
                102869000000.0,
                102869000000.0,
                49068000000.0,
                65863000000.0,
                57200000.0,
                5700000.0,
                None,
                None,
                None,
                -3539.0,
                1607245.0,
                1294000.0,
                83438000.0,
                12555108.0,
                None,
                2952000.0,
                2627000.0,
                0.0,
                0.0,
                325000.0,
            ],
            "quarterly_figure_py": [
                None,
                None,
                None,
                None,
                200790000.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                17488000000.0,
                52747000000.0,
                52747000000.0,
                16128000000.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0.0,
                0.0,
                0.0,
                0.0,
                None,
            ],
            "is_computed": [
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
            ],
        }
    )

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
