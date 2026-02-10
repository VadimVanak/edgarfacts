# src/edgarfacts/transforms/config.py
"""
Global configuration for edgarfacts transformations.

This module defines *policy-level* constants used across transformation
pipelines (outlier handling, arc application, reporting-period inference).

These values are intentionally centralized to:
- make heuristics explicit and auditable
- avoid hard-coded magic numbers
- allow future user overrides if needed

This module MUST NOT contain any computation logic.
"""

from __future__ import annotations


# =============================================================================
# Outlier detection / correction
# =============================================================================

# Maximum absolute value allowed for most figures
MAX_ABS_FIGURE_VALUE = 5e12

# Tags allowed to exceed MAX_ABS_FIGURE_VALUE
LARGE_VALUE_TAG_WHITELIST = {
    "NotionalAmountOfDerivatives",
    "DerivativeNotionalAmount",
    "DerivativeAssetNotionalAmount",
    "DerivativeLiabilityNotionalAmount",
}

# Minimum number of days assumed when normalizing accumulated values
MIN_PERIOD_DAYS_FOR_NORMALIZATION = 90


# =============================================================================
# Arc-based computation
# =============================================================================

# Maximum number of full passes over arc sequences
# (normally 1 is sufficient if seq ordering is correct)
ARC_MAX_PASSES = 1

# Collision policy when computed and reported figures coexist
# True  -> keep reported values, only fill missing ones
# False -> allow computed values to override reported ones
ARC_KEEP_ORIGINAL_FIRST = True


# =============================================================================
# Reporting-period inference
# =============================================================================

# Max day difference between filing period and figure end date
PERIOD_MATCH_TOLERANCE_DAYS = 30

# Tolerance when aligning prior-year periods
PRIOR_YEAR_ALIGNMENT_TOLERANCE_DAYS = 10


# =============================================================================
# Datetime policy
# =============================================================================

# Canonical datetime precision for all transformed outputs
DATETIME_DTYPE = "datetime64[s]"


# =============================================================================
# Tags requires special handling
# =============================================================================

SPECIAL_INSTANT_TAGS = {
    "EntityCommonStockSharesOutstanding",
    "EntityPublicFloat",
}

SPECIAL_INSTANT_TOL_DAYS = 30

