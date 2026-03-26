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


def check_pivot_figures(df: pd.DataFrame) -> None:
    check_list = [
        {
            "adsh": 119312526027207,
            "tag": "Revenues_a",
            "value": 305453000000.0,
        },
        {
            "adsh": 119312526027207,
            "tag": "CostOfGoodsAndServicesSold_a",
            "value": 95954000000.0,
        },
        {
            "adsh": 119312526027207,
            "tag": "GrossProfit_a",
            "value": 209499000000.0,
        },
        {
            "adsh": 733213000004,
            "tag": "OtherComprehensiveIncomeUnrealizedGainLossOnDerivativesArisingDuringPeriodTax_q",
            "value": 23700000.0,
        },
        {"adsh": 894718000045, "tag": "Assets_q_py", "value": 988201000.0},
        {"adsh": 894718000046, "tag": "InventoryNet_q", "value": 101026000.0},
        {"adsh": 894718000047, "tag": "ProfitLoss_q_py", "value": 10159000.0},
        {
            "adsh": 894718000047,
            "tag": "InventoryGross_a",
            "value": 108453000.0,
        },
        {
            "adsh": 1849813000040,
            "tag": "DeferredTaxAssetsLiabilitiesNetNoncurrent_a_py",
            "value": 21057000.0,
        },
        {
            "adsh": 1961712000262,
            "tag": "IncreaseDecreaseInOperatingCapital_q",
            "value": 1159000000.0,
        },
        {
            "adsh": 3062517000115,
            "tag": "OtherLiabilitiesNoncurrent_a",
            "value": 44911000.0,
        },
        {
            "adsh": 4721723000075,
            "tag": "OtherComprehensiveIncomeLossNetOfTaxPortionAttributableToParent_a",
            "value": 530000000.0,
        },
        {
            "adsh": 5697818000053,
            "tag": "DeferredTaxAssetsGross_q_py",
            "value": 46326000.0,
        },
        {
            "adsh": 5955823000019,
            "tag": "DeferredTaxLiabilitiesOther_q",
            "value": 267000000.0,
        },
        {
            "adsh": 5955823000019,
            "tag": "StockholdersEquity_a",
            "value": 4569000000.0,
        },
        {
            "adsh": 5955823000019,
            "tag": "NetCashProvidedByUsedInOperatingActivities_q",
            "value": 642000000.0,
        },
        {
            "adsh": 5955823000019,
            "tag": "GoodwillGross_q",
            "value": 3932000000.0,
        },
        {
            "adsh": 5955823000019,
            "tag": "DerivativeAssetCollateralObligationToReturnCashOffset_q",
            "value": 3273000000.0,
        },
        {
            "adsh": 6329615000058,
            "tag": "IncreaseDecreaseInAccountsAndNotesReceivable_a",
            "value": 13492000.0,
        },
        {
            "adsh": 6329615000058,
            "tag": "NetIncomeLossAvailableToCommonStockholdersDiluted_a",
            "value": 42504000.0,
        },
        {
            "adsh": 6329615000058,
            "tag": "LaborAndRelatedExpense_a",
            "value": 6812000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsNotesAndLoansReceivableNetCurrent_q",
            "value": 21837000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsAndShortTermInvestments_q",
            "value": 67150000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CostOfRevenue_q",
            "value": 48482000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments_q",
            "value": 28058000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OperatingCostsAndExpenses_q",
            "value": 7903000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ReceivablesNetCurrent_q",
            "value": 41150000000.0,
        },
        {"adsh": 32019324000069, "tag": "Revenues_q", "value": 90753000000.0},
        {
            "adsh": 32019324000069,
            "tag": "ShortTermBorrowings_q",
            "value": 1997000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NoninterestIncome_q",
            "value": 90753000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ShortTermInvestments_q",
            "value": 34455000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermInvestments_q",
            "value": 95187000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermInvestmentsAndReceivablesNet_q",
            "value": 95187000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "DeferredRevenueCurrent_q",
            "value": 8012000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccumulatedOtherComprehensiveIncomeLossNetOfTax_q",
            "value": -8960000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeTaxExpenseBenefit_q",
            "value": 4422000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStockSharesIssued_q",
            "value": 15337686000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsReceivableNetCurrent_q",
            "value": 21837000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ComprehensiveIncomeNetOfTax_q",
            "value": 24054000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashAndCashEquivalentsAtCarryingValue_q",
            "value": 32695000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "GrossProfit_q",
            "value": 42271000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AssetsNoncurrent_q",
            "value": 208995000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment_q_py",
            "value": 69668000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment_q",
            "value": 71697000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AssetsCurrent_q",
            "value": 128416000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherAssetsCurrent_q",
            "value": 13884000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ResearchAndDevelopmentExpense_q",
            "value": 7903000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "MarketableSecuritiesNoncurrent_q",
            "value": 95187000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CostOfGoodsAndServicesSold_q_py",
            "value": 52860000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "GrossProfit_q_py",
            "value": 41976000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeTaxExpenseBenefit_q_py",
            "value": 4222000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetIncomeLoss_q_py",
            "value": 24160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OperatingExpenses_q_py",
            "value": 13658000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OperatingIncomeLoss_q_py",
            "value": 28318000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "Revenues_q_py",
            "value": 94836000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "SellingGeneralAndAdministrativeExpense_q_py",
            "value": 6201000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NonoperatingIncomeExpense_q_py",
            "value": 64000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ResearchAndDevelopmentExpense_q_py",
            "value": 7457000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest_q_py",
            "value": 28382000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsReceivableNetCurrent_a",
            "value": 21837000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PropertyPlantAndEquipmentNet_a",
            "value": 43546000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "MarketableSecuritiesCurrent_a",
            "value": 34455000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NontradeReceivablesCurrent_a",
            "value": 19313000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings_a",
            "value": 6232000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AssetsCurrent_q_py",
            "value": 112913000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AssetsNoncurrent_q_py",
            "value": 219247000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashAndCashEquivalentsAtCarryingValue_q_py",
            "value": 24687000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStockSharesIssued_q_py",
            "value": 15723406000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "InventoryNet_q_py",
            "value": 7482000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "Liabilities_q_py",
            "value": 270002000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LiabilitiesAndStockholdersEquity_q_py",
            "value": 332160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LiabilitiesCurrent_q_py",
            "value": 120075000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AssetsCurrent_a_py",
            "value": 112913000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AssetsNoncurrent_a_py",
            "value": 219247000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashAndCashEquivalentsAtCarryingValue_a_py",
            "value": 24687000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStockSharesIssued_a_py",
            "value": 15723406000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "InventoryNet_a_py",
            "value": 7482000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "Liabilities_a_py",
            "value": 270002000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LiabilitiesAndStockholdersEquity_a_py",
            "value": 332160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LiabilitiesCurrent_a_py",
            "value": 120075000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsNotesAndLoansReceivableNetCurrent_q_py",
            "value": 17936000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsPayableAndAccruedLiabilitiesCurrent_q_py",
            "value": 42945000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsPayableCurrent_q_py",
            "value": 42945000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsReceivableNetCurrent_q_py",
            "value": 17936000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccumulatedOtherComprehensiveIncomeLossNetOfTax_q_py",
            "value": -11746000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities_q_py",
            "value": 4169000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AdjustmentsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities_q_py",
            "value": 4400000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AllocatedShareBasedCompensationExpense_q_py",
            "value": 2686000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "Assets_q_py",
            "value": 332160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AvailableForSaleSecuritiesDebtMaturitiesRollingAfterYearTenFairValue_q_py",
            "value": 17181000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AvailableForSaleSecuritiesDebtMaturitiesRollingYearSixThroughTenFairValue_q_py",
            "value": 11928000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AvailableForSaleSecuritiesDebtMaturitiesRollingYearTwoThroughFiveFairValue_q_py",
            "value": 81352000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AvailableForSaleSecuritiesDebtMaturitiesSingleMaturityDate_q_py",
            "value": 110461000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsAndShortTermInvestments_q_py",
            "value": 55872000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents_q_py",
            "value": 27129000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect_q_py",
            "value": 5155000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect_q_py",
            "value": 5155000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommercialPaper_q_py",
            "value": 1996000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStockSharesAuthorized_q_py",
            "value": 50400000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStockSharesOutstanding_q_py",
            "value": 15723406000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStocksIncludingAdditionalPaidInCapital_q_py",
            "value": 69568000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ComprehensiveIncomeNetOfTax_q_py",
            "value": 25326000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ContractWithCustomerLiability_q_py",
            "value": 12500000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ContractWithCustomerLiabilityCurrent_q_py",
            "value": 8131000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ContractWithCustomerLiabilityRevenueRecognized_q_py",
            "value": 3500000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CostOfRevenue_q_py",
            "value": 52860000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "DebtCurrent_q_py",
            "value": 12574000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "DeferredRevenueCurrent_q_py",
            "value": 8131000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "DepreciationDepletionAndAmortization_q_py",
            "value": 2898000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "EmployeeBenefitsAndShareBasedCompensationNoncash_q_py",
            "value": 2686000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense_q_py",
            "value": 620000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "GeneralAndAdministrativeExpense_q_py",
            "value": 2686000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest_q_py",
            "value": 24160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeTaxesPaidNet_q_py",
            "value": 4066000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInAccountsAndNotesReceivable_q_py",
            "value": -5321000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInAccountsPayable_q_py",
            "value": -14689000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities_q_py",
            "value": -14689000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInAccountsReceivable_q_py",
            "value": -5321000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInInventories_q_py",
            "value": 741000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOperatingAssets_q_py",
            "value": -17052000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOperatingCapital_q_py",
            "value": -231000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOperatingLiabilities_q_py",
            "value": -16821000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOtherOperatingAssets_q_py",
            "value": -7000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOtherReceivables_q_py",
            "value": -12465000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInReceivables_q_py",
            "value": -17786000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings_q_py",
            "value": 7482000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LiabilitiesNoncurrent_q_py",
            "value": 149927000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LiabilitiesOtherThanLongtermDebtNoncurrent_q_py",
            "value": 52886000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebt_q_py",
            "value": 107600000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebtAndCapitalLeaseObligations_q_py",
            "value": 97041000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebtAndCapitalLeaseObligationsCurrent_q_py",
            "value": 10578000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebtCurrent_q_py",
            "value": 10578000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebtNoncurrent_q_py",
            "value": 97041000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermInvestments_q_py",
            "value": 110461000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermInvestmentsAndReceivablesNet_q_py",
            "value": 110461000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "MarketableSecuritiesCurrent_q_py",
            "value": 31185000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "MarketableSecuritiesNoncurrent_q_py",
            "value": 110461000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInFinancingActivities_q_py",
            "value": -25724000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations_q_py",
            "value": -25724000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInInvestingActivities_q_py",
            "value": 2319000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations_q_py",
            "value": 2319000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInOperatingActivities_q_py",
            "value": 28560000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetIncomeLossAvailableToCommonStockholdersBasic_q_py",
            "value": 24160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetIncomeLossAvailableToCommonStockholdersDiluted_q_py",
            "value": 24160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NoninterestIncome_q_py",
            "value": 94836000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NontradeReceivablesCurrent_q_py",
            "value": 17963000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OperatingCostsAndExpenses_q_py",
            "value": 7457000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherAssetsCurrent_q_py",
            "value": 13660000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherAssetsNoncurrent_q_py",
            "value": 65388000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeLossAvailableForSaleSecuritiesAdjustmentNetOfTax_q_py",
            "value": 1465000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeLossForeignCurrencyTransactionAndTranslationAdjustmentNetOfTax_q_py",
            "value": -95000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeLossNetOfTaxPortionAttributableToParent_q_py",
            "value": 1166000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeLossReclassificationAdjustmentFromAOCIForSaleOfSecuritiesNetOfTax_q_py",
            "value": -62000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeUnrealizedHoldingGainLossOnSecuritiesArisingDuringPeriodNetOfTax_q_py",
            "value": 1403000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherLiabilitiesCurrent_q_py",
            "value": 56425000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherLiabilitiesNoncurrent_q_py",
            "value": 52886000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherNoncashIncomeExpense_q_py",
            "value": 1415000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForProceedsFromInvestments_q_py",
            "value": -5341000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForProceedsFromOtherInvestingActivities_q_py",
            "value": 106000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForProceedsFromProductiveAssets_q_py",
            "value": 2916000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForRepurchaseOfCommonStock_q_py",
            "value": 19594000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForRepurchaseOfEquity_q_py",
            "value": 19594000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsOfDividends_q_py",
            "value": 3650000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsRelatedToTaxWithholdingForShareBasedCompensation_q_py",
            "value": 418000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsToAcquireAvailableForSaleSecuritiesDebt_q_py",
            "value": 6044000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsToAcquireInvestments_q_py",
            "value": 6044000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsToAcquireProductiveAssets_q_py",
            "value": 2916000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsToAcquirePropertyPlantAndEquipment_q_py",
            "value": 2916000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities_q_py",
            "value": 9997000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromPaymentsForOtherFinancingActivities_q_py",
            "value": -66000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfCommercialPaper_q_py",
            "value": 254000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfDebt_q_py",
            "value": -1996000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfLongTermDebtAndCapitalSecurities_q_py",
            "value": -2250000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfShortTermDebt_q_py",
            "value": 254000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess_q_py",
            "value": 254000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepurchaseOfEquity_q_py",
            "value": -19594000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromSaleAndMaturityOfAvailableForSaleSecurities_q_py",
            "value": 11385000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromSaleMaturityAndCollectionsOfInvestments_q_py",
            "value": 11385000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromSaleOfAvailableForSaleSecuritiesDebt_q_py",
            "value": 1388000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProfitLoss_q_py",
            "value": 24160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PropertyPlantAndEquipmentGross_q_py",
            "value": 113066000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PropertyPlantAndEquipmentNet_q_py",
            "value": 43398000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ReceivablesNetCurrent_q_py",
            "value": 35899000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "RepaymentsOfLongTermDebt_q_py",
            "value": 2250000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "RepaymentsOfLongTermDebtAndCapitalSecurities_q_py",
            "value": 2250000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "RetainedEarningsAccumulatedDeficit_q_py",
            "value": 4336000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q_py",
            "value": 94836000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ShareBasedCompensation_q_py",
            "value": 2686000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ShortTermBorrowings_q_py",
            "value": 1996000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ShortTermInvestments_q_py",
            "value": 31185000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockIssuedDuringPeriodSharesPeriodIncreaseDecrease_q_py",
            "value": -129000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockRepurchasedAndRetiredDuringPeriodShares_q_py",
            "value": 129000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockRepurchasedAndRetiredDuringPeriodValue_q_py",
            "value": 19100000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockholdersEquity_q_py",
            "value": 62158000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest_q_py",
            "value": 62158000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockholdersEquityPeriodIncreaseDecrease_q_py",
            "value": -19100000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "WeightedAverageNumberDilutedSharesOutstandingAdjustment_q_py",
            "value": 59896000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "WeightedAverageNumberOfDilutedSharesOutstanding_q_py",
            "value": 15847050000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "WeightedAverageNumberOfSharesOutstandingBasic_q_py",
            "value": 15787154000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsNotesAndLoansReceivableNetCurrent_a_py",
            "value": 17936000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsPayableAndAccruedLiabilitiesCurrent_a_py",
            "value": 42945000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsPayableCurrent_a_py",
            "value": 42945000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccountsReceivableNetCurrent_a_py",
            "value": 17936000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment_a_py",
            "value": 69668000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AccumulatedOtherComprehensiveIncomeLossNetOfTax_a_py",
            "value": -11746000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AdjustmentsNoncashItemsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities_a_py",
            "value": 19802000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AdjustmentsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities_a_py",
            "value": 15263000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AllocatedShareBasedCompensationExpense_a_py",
            "value": 10112000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "Assets_a_py",
            "value": 332160000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AvailableForSaleSecuritiesDebtMaturitiesRollingAfterYearTenFairValue_a_py",
            "value": 17181000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AvailableForSaleSecuritiesDebtMaturitiesRollingYearSixThroughTenFairValue_a_py",
            "value": 11928000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AvailableForSaleSecuritiesDebtMaturitiesRollingYearTwoThroughFiveFairValue_a_py",
            "value": 81352000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "AvailableForSaleSecuritiesDebtMaturitiesSingleMaturityDate_a_py",
            "value": 110461000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsAndShortTermInvestments_a_py",
            "value": 55872000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents_a_py",
            "value": 27129000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect_a_py",
            "value": -2051000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect_a_py",
            "value": -2051000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommercialPaper_a_py",
            "value": 1996000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStockSharesAuthorized_a_py",
            "value": 50400000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStockSharesOutstanding_a_py",
            "value": 15723406000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CommonStocksIncludingAdditionalPaidInCapital_a_py",
            "value": 69568000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ComprehensiveIncomeNetOfTax_a_py",
            "value": 89069000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ContractWithCustomerLiability_a_py",
            "value": 12500000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ContractWithCustomerLiabilityCurrent_a_py",
            "value": 8131000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ContractWithCustomerLiabilityRevenueRecognized_a_py",
            "value": 8200000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CostOfGoodsAndServicesSold_a_py",
            "value": 218807000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "CostOfRevenue_a_py",
            "value": 218807000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "DebtCurrent_a_py",
            "value": 12574000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "DeferredRevenueCurrent_a_py",
            "value": 8131000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "DepreciationDepletionAndAmortization_a_py",
            "value": 11484000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "EmployeeBenefitsAndShareBasedCompensationNoncash_a_py",
            "value": 10112000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense_a_py",
            "value": 3615000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "GeneralAndAdministrativeExpense_a_py",
            "value": 10112000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "GrossProfit_a_py",
            "value": 166288000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest_a_py",
            "value": 111728000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments_a_py",
            "value": 108249000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest_a_py",
            "value": 94321000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeTaxExpenseBenefit_a_py",
            "value": 17407000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncomeTaxesPaidNet_a_py",
            "value": 15166000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInAccountsAndNotesReceivable_a_py",
            "value": -2231000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInAccountsPayable_a_py",
            "value": -9566000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities_a_py",
            "value": -9566000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInAccountsReceivable_a_py",
            "value": -2231000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInInventories_a_py",
            "value": 2129000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOperatingAssets_a_py",
            "value": 325000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOperatingCapital_a_py",
            "value": 4539000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOperatingLiabilities_a_py",
            "value": -4214000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOtherOperatingAssets_a_py",
            "value": 7049000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOtherOperatingLiabilities_a_py",
            "value": 4874000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInOtherReceivables_a_py",
            "value": -6622000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "IncreaseDecreaseInReceivables_a_py",
            "value": -8853000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings_a_py",
            "value": 7482000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LiabilitiesNoncurrent_a_py",
            "value": 149927000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LiabilitiesOtherThanLongtermDebtNoncurrent_a_py",
            "value": 52886000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebt_a_py",
            "value": 107600000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebtAndCapitalLeaseObligations_a_py",
            "value": 97041000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebtAndCapitalLeaseObligationsCurrent_a_py",
            "value": 10578000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebtCurrent_a_py",
            "value": 10578000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermDebtNoncurrent_a_py",
            "value": 97041000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermInvestments_a_py",
            "value": 110461000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "LongTermInvestmentsAndReceivablesNet_a_py",
            "value": 110461000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "MarketableSecuritiesCurrent_a_py",
            "value": 31185000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "MarketableSecuritiesNoncurrent_a_py",
            "value": 110461000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInFinancingActivities_a_py",
            "value": -115526000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations_a_py",
            "value": -115526000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInInvestingActivities_a_py",
            "value": 3891000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations_a_py",
            "value": 3891000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetCashProvidedByUsedInOperatingActivities_a_py",
            "value": 109584000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetIncomeLoss_a_py",
            "value": 94321000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetIncomeLossAvailableToCommonStockholdersBasic_a_py",
            "value": 94321000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NetIncomeLossAvailableToCommonStockholdersDiluted_a_py",
            "value": 94321000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NoninterestIncome_a_py",
            "value": 385095000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NonoperatingIncomeExpense_a_py",
            "value": -576000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "NontradeReceivablesCurrent_a_py",
            "value": 17963000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OperatingCostsAndExpenses_a_py",
            "value": 28724000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OperatingExpenses_a_py",
            "value": 53984000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OperatingIncomeLoss_a_py",
            "value": 112304000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherAssetsCurrent_a_py",
            "value": 13660000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherAssetsNoncurrent_a_py",
            "value": 65388000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeLossAvailableForSaleSecuritiesAdjustmentNetOfTax_a_py",
            "value": -2705000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeLossForeignCurrencyTransactionAndTranslationAdjustmentNetOfTax_a_py",
            "value": -1239000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeLossNetOfTaxPortionAttributableToParent_a_py",
            "value": -5252000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeLossReclassificationAdjustmentFromAOCIForSaleOfSecuritiesNetOfTax_a_py",
            "value": -287000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherComprehensiveIncomeUnrealizedHoldingGainLossOnSecuritiesArisingDuringPeriodNetOfTax_a_py",
            "value": -2992000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherLiabilitiesCurrent_a_py",
            "value": 56425000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherLiabilitiesNoncurrent_a_py",
            "value": 52886000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "OtherNoncashIncomeExpense_a_py",
            "value": 2689000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForProceedsFromInvestments_a_py",
            "value": -17583000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForProceedsFromOtherInvestingActivities_a_py",
            "value": 1292000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForProceedsFromProductiveAssets_a_py",
            "value": 12094000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForRepurchaseOfCommonStock_a_py",
            "value": 85362000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsForRepurchaseOfEquity_a_py",
            "value": 85362000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsOfDividends_a_py",
            "value": 14932000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsRelatedToTaxWithholdingForShareBasedCompensation_a_py",
            "value": 5739000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsToAcquireAvailableForSaleSecuritiesDebt_a_py",
            "value": 26133000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsToAcquireInvestments_a_py",
            "value": 26133000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsToAcquireProductiveAssets_a_py",
            "value": 12094000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PaymentsToAcquirePropertyPlantAndEquipment_a_py",
            "value": 12094000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities_a_py",
            "value": 29041000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromPaymentsForOtherFinancingActivities_a_py",
            "value": -510000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfCommercialPaper_a_py",
            "value": -5004000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfDebt_a_py",
            "value": -8983000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfLongTermDebtAndCapitalSecurities_a_py",
            "value": -3979000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfShortTermDebt_a_py",
            "value": -5004000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess_a_py",
            "value": -5003000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromRepurchaseOfEquity_a_py",
            "value": -85362000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromSaleAndMaturityOfAvailableForSaleSecurities_a_py",
            "value": 43716000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromSaleMaturityAndCollectionsOfInvestments_a_py",
            "value": 43716000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProceedsFromSaleOfAvailableForSaleSecuritiesDebt_a_py",
            "value": 14675000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ProfitLoss_a_py",
            "value": 94321000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PropertyPlantAndEquipmentGross_a_py",
            "value": 113066000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "PropertyPlantAndEquipmentNet_a_py",
            "value": 43398000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ReceivablesNetCurrent_a_py",
            "value": 35899000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "RepaymentsOfLongTermDebt_a_py",
            "value": 9444000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "RepaymentsOfLongTermDebtAndCapitalSecurities_a_py",
            "value": 9444000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ResearchAndDevelopmentExpense_a_py",
            "value": 28724000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "RetainedEarningsAccumulatedDeficit_a_py",
            "value": 4336000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a_py",
            "value": 385095000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "Revenues_a_py",
            "value": 385095000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "SellingGeneralAndAdministrativeExpense_a_py",
            "value": 25260000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ShareBasedCompensation_a_py",
            "value": 10112000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ShortTermBorrowings_a_py",
            "value": 1996000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "ShortTermInvestments_a_py",
            "value": 31185000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockIssuedDuringPeriodSharesPeriodIncreaseDecrease_a_py",
            "value": -565000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockRepurchasedAndRetiredDuringPeriodShares_a_py",
            "value": 565000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockRepurchasedAndRetiredDuringPeriodValue_a_py",
            "value": 85000000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockholdersEquity_a_py",
            "value": 62158000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest_a_py",
            "value": 62158000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "StockholdersEquityPeriodIncreaseDecrease_a_py",
            "value": -85000000000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "WeightedAverageNumberDilutedSharesOutstandingAdjustment_a_py",
            "value": 45260000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "WeightedAverageNumberOfDilutedSharesOutstanding_a_py",
            "value": 15765899000.0,
        },
        {
            "adsh": 32019324000069,
            "tag": "WeightedAverageNumberOfSharesOutstandingBasic_a_py",
            "value": 15720639000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q",
            "value": 64727000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "CostOfGoodsAndServicesSold_q",
            "value": 19684000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "GrossProfit_q",
            "value": 45043000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "ResearchAndDevelopmentExpense_q",
            "value": 8056000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "SellingAndMarketingExpense_q",
            "value": 6816000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "GeneralAndAdministrativeExpense_q",
            "value": 2246000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "OperatingIncomeLoss_q",
            "value": 27925000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "NonoperatingIncomeExpense_q",
            "value": -675000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a",
            "value": 245122000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "CostOfGoodsAndServicesSold_a",
            "value": 74114000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "GrossProfit_a",
            "value": 171008000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "ResearchAndDevelopmentExpense_a",
            "value": 29510000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "SellingAndMarketingExpense_a",
            "value": 24456000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "GeneralAndAdministrativeExpense_a",
            "value": 7609000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "OperatingIncomeLoss_a",
            "value": 109433000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "NonoperatingIncomeExpense_a",
            "value": -1646000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q_py",
            "value": 56189000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "CostOfGoodsAndServicesSold_q_py",
            "value": 16795000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "GrossProfit_q_py",
            "value": 39394000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "ResearchAndDevelopmentExpense_q_py",
            "value": 6739000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "SellingAndMarketingExpense_q_py",
            "value": 6204000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "GeneralAndAdministrativeExpense_q_py",
            "value": 2197000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "OperatingIncomeLoss_q_py",
            "value": 24254000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "NonoperatingIncomeExpense_q_py",
            "value": 473000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a_py",
            "value": 211915000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "CostOfGoodsAndServicesSold_a_py",
            "value": 65863000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "GrossProfit_a_py",
            "value": 146052000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "ResearchAndDevelopmentExpense_a_py",
            "value": 27195000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "SellingAndMarketingExpense_a_py",
            "value": 22759000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "GeneralAndAdministrativeExpense_a_py",
            "value": 7575000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "OperatingIncomeLoss_a_py",
            "value": 88523000000.0,
        },
        {
            "adsh": 95017024087843,
            "tag": "NonoperatingIncomeExpense_a_py",
            "value": 788000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q",
            "value": 61858000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "CostOfGoodsAndServicesSold_q",
            "value": 18505000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "GrossProfit_q",
            "value": 43353000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "ResearchAndDevelopmentExpense_q",
            "value": 7653000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "SellingAndMarketingExpense_q",
            "value": 6207000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "GeneralAndAdministrativeExpense_q",
            "value": 1912000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "OperatingIncomeLoss_q",
            "value": 27581000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "NonoperatingIncomeExpense_q",
            "value": -854000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a",
            "value": 236584000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "CostOfGoodsAndServicesSold_a",
            "value": 71225000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "GrossProfit_a",
            "value": 165359000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "ResearchAndDevelopmentExpense_a",
            "value": 28193000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "SellingAndMarketingExpense_a",
            "value": 23844000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "GeneralAndAdministrativeExpense_a",
            "value": 7560000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "OperatingIncomeLoss_a",
            "value": 105762000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "NonoperatingIncomeExpense_a",
            "value": -498000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q_py",
            "value": 52857000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "CostOfGoodsAndServicesSold_q_py",
            "value": 16128000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "GrossProfit_q_py",
            "value": 36729000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "ResearchAndDevelopmentExpense_q_py",
            "value": 6984000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "SellingAndMarketingExpense_q_py",
            "value": 5750000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "GeneralAndAdministrativeExpense_q_py",
            "value": 1643000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "OperatingIncomeLoss_q_py",
            "value": 22352000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "NonoperatingIncomeExpense_q_py",
            "value": 321000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a_py",
            "value": 207591000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "CostOfGoodsAndServicesSold_a_py",
            "value": 65497000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "GrossProfit_a_py",
            "value": 142094000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "ResearchAndDevelopmentExpense_a_py",
            "value": 27305000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "SellingAndMarketingExpense_a_py",
            "value": 22859000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "GeneralAndAdministrativeExpense_a_py",
            "value": 7127000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "OperatingIncomeLoss_a_py",
            "value": 84803000000.0,
        },
        {
            "adsh": 95017024048288,
            "tag": "NonoperatingIncomeExpense_a_py",
            "value": 268000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q",
            "value": 62020000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "CostOfGoodsAndServicesSold_q",
            "value": 19623000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "GrossProfit_q",
            "value": 42397000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "ResearchAndDevelopmentExpense_q",
            "value": 7142000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "SellingAndMarketingExpense_q",
            "value": 6246000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "GeneralAndAdministrativeExpense_q",
            "value": 1977000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "OperatingIncomeLoss_q",
            "value": 27032000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "NonoperatingIncomeExpense_q",
            "value": -506000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a",
            "value": 227583000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "CostOfGoodsAndServicesSold_a",
            "value": 68848000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "GrossProfit_a",
            "value": 158735000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "ResearchAndDevelopmentExpense_a",
            "value": 27524000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "SellingAndMarketingExpense_a",
            "value": 23387000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "GeneralAndAdministrativeExpense_a",
            "value": 7291000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "OperatingIncomeLoss_a",
            "value": 100533000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "NonoperatingIncomeExpense_a",
            "value": 677000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q_py",
            "value": 52747000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "CostOfGoodsAndServicesSold_q_py",
            "value": 17488000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "GrossProfit_q_py",
            "value": 35259000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "ResearchAndDevelopmentExpense_q_py",
            "value": 6844000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "SellingAndMarketingExpense_q_py",
            "value": 5679000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "GeneralAndAdministrativeExpense_q_py",
            "value": 2337000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "OperatingIncomeLoss_q_py",
            "value": 20399000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "NonoperatingIncomeExpense_q_py",
            "value": -60000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a_py",
            "value": 204094000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "CostOfGoodsAndServicesSold_a_py",
            "value": 64984000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "GrossProfit_a_py",
            "value": 139110000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "ResearchAndDevelopmentExpense_a_py",
            "value": 26627000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "SellingAndMarketingExpense_a_py",
            "value": 22704000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "GeneralAndAdministrativeExpense_a_py",
            "value": 6964000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "OperatingIncomeLoss_a_py",
            "value": 82815000000.0,
        },
        {
            "adsh": 95017024008814,
            "tag": "NonoperatingIncomeExpense_a_py",
            "value": -227000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q",
            "value": 56517000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "CostOfGoodsAndServicesSold_q",
            "value": 16302000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "GrossProfit_q",
            "value": 40215000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "ResearchAndDevelopmentExpense_q",
            "value": 6659000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "SellingAndMarketingExpense_q",
            "value": 5187000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "GeneralAndAdministrativeExpense_q",
            "value": 1474000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "OperatingIncomeLoss_q",
            "value": 26895000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "NonoperatingIncomeExpense_q",
            "value": 389000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a",
            "value": 218310000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "CostOfGoodsAndServicesSold_a",
            "value": 66713000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "GrossProfit_a",
            "value": 151597000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "ResearchAndDevelopmentExpense_a",
            "value": 27226000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "SellingAndMarketingExpense_a",
            "value": 22820000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "GeneralAndAdministrativeExpense_a",
            "value": 7651000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "OperatingIncomeLoss_a",
            "value": 93900000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "NonoperatingIncomeExpense_a",
            "value": 1123000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q_py",
            "value": 50122000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "CostOfGoodsAndServicesSold_q_py",
            "value": 15452000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "GrossProfit_q_py",
            "value": 34670000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "ResearchAndDevelopmentExpense_q_py",
            "value": 6628000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "SellingAndMarketingExpense_q_py",
            "value": 5126000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "GeneralAndAdministrativeExpense_q_py",
            "value": 1398000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "OperatingIncomeLoss_q_py",
            "value": 21518000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "NonoperatingIncomeExpense_q_py",
            "value": 54000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a_py",
            "value": 203075000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "CostOfGoodsAndServicesSold_a_py",
            "value": 64456000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "GrossProfit_a_py",
            "value": 138619000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "ResearchAndDevelopmentExpense_a_py",
            "value": 25541000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "SellingAndMarketingExpense_a_py",
            "value": 22404000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "GeneralAndAdministrativeExpense_a_py",
            "value": 6011000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "OperatingIncomeLoss_a_py",
            "value": 84663000000.0,
        },
        {
            "adsh": 95017023054855,
            "tag": "NonoperatingIncomeExpense_a_py",
            "value": 101000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q",
            "value": 56189000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "CostOfGoodsAndServicesSold_q",
            "value": 16795000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "GrossProfit_q",
            "value": 39394000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "ResearchAndDevelopmentExpense_q",
            "value": 6739000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "SellingAndMarketingExpense_q",
            "value": 6204000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "GeneralAndAdministrativeExpense_q",
            "value": 2197000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "OperatingIncomeLoss_q",
            "value": 24254000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "NonoperatingIncomeExpense_q",
            "value": 473000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a",
            "value": 211915000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "CostOfGoodsAndServicesSold_a",
            "value": 65863000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "GrossProfit_a",
            "value": 146052000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "ResearchAndDevelopmentExpense_a",
            "value": 27195000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "SellingAndMarketingExpense_a",
            "value": 22759000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "GeneralAndAdministrativeExpense_a",
            "value": 7575000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "OperatingIncomeLoss_a",
            "value": 88523000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "NonoperatingIncomeExpense_a",
            "value": 788000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_q_py",
            "value": 51865000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "CostOfGoodsAndServicesSold_q_py",
            "value": 16429000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "GrossProfit_q_py",
            "value": 35436000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "ResearchAndDevelopmentExpense_q_py",
            "value": 6849000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "SellingAndMarketingExpense_q_py",
            "value": 6304000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "GeneralAndAdministrativeExpense_q_py",
            "value": 1749000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "OperatingIncomeLoss_q_py",
            "value": 20534000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "NonoperatingIncomeExpense_q_py",
            "value": -47000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "RevenueFromContractWithCustomerExcludingAssessedTax_a_py",
            "value": 198270000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "CostOfGoodsAndServicesSold_a_py",
            "value": 62650000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "GrossProfit_a_py",
            "value": 135620000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "ResearchAndDevelopmentExpense_a_py",
            "value": 24512000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "SellingAndMarketingExpense_a_py",
            "value": 21825000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "GeneralAndAdministrativeExpense_a_py",
            "value": 5900000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "OperatingIncomeLoss_a_py",
            "value": 83383000000.0,
        },
        {
            "adsh": 95017023035122,
            "tag": "NonoperatingIncomeExpense_a_py",
            "value": 333000000.0,
        },
    ]

    for check in check_list:
        assert (
            df.loc[df.index == check["adsh"], check["tag"]].values[0]
            == check["value"]
        )
