# Refactoring Guide -- Capital-Cost Refactor + Main Restructure (Session B, 07 July 2026)

This guide is organized **by file**. For each file, the changes are listed **in
top-to-bottom order** so you can work straight down the file. Every change gives
a **BEFORE** block (the exact text on your base branch,
`update-data-and-projections-aeo2026-cambium2024`) and an **AFTER** block (paste
this in to replace it), plus the approximate base-branch line number and a short
WHY.

**How to use:** open the file, find the BEFORE block (line numbers are from the
base branch and drift as you edit downward), select it, paste the AFTER block.
Work one file at a time, top to bottom.

**Files changed in this session**
1. `cmu_tare_model/utils/remdb_v4_installed_cost_utils.py` (15 changes)
2. `cmu_tare_model/constants.py` (1 change)
3. `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py` (3 changes)
4. `cmu_tare_model/private_impact/data_processing/determine_rebate_eligibility_and_amount.py` (1 change)
5. `cmu_tare_model/tare_model_main_v2_3_EXPORT_5July2026.py` (1 change)
6. `CLAUDE.md` (5 changes -- documentation)
7. `scripts/capture_capital_cost_baseline.py` (new file)
8. `baseline_capture/` (generated parquet oracle -- no manual action)

**Context / caveats** (full detail at the bottom): the real EUSS stock is offline
(Zenodo), so verification used a synthetic path-covering oracle plus the real
REMDB v4 cost table. Model-level token migration and notebook cell transport that
cannot be run offline were **deliberately not made** and are listed under
"Deferred / hand-migration" at the end. Pre-existing test bar (unchanged by this
work): `33 failed, 236 passed, 24 errors` + 1 collection-error module.

---

# File 1 -- `cmu_tare_model/utils/remdb_v4_installed_cost_utils.py`

This is the core file. 15 changes, all either value-identical (Task 2) or the
authorized clamping removal (Task 4). After applying all of them, the module is
byte-identical to the baseline oracle except for the intended capacity-clamping
removal (Change 1.11 / 1.15).

## Change 1.1 -- Module header (base lines 1-34)

WHY: the old header claimed clamping was removed while the clamping steps still
ran; rewritten to describe the actual pipeline.

**BEFORE**
```python
"""
========================================================================================================================================================================
REMDB v4 Installed Cost Utilities (SIMPLIFIED)
========================================================================================================================================================================

This module prepares equipment metrics for REMDB v4 cost calculations.

Key Functions:
- add_remdb_metrics(): Unified function for both replacement and upgrade metrics

Features:
- Percentile-based filtering to exclude capacity outliers before processing
- Unit conversion based on REMDB specifications
- NaN values propagate naturally for homes with invalid fuel/technology types

Refactored: January 2026
- Combined duplicate replacement/upgrade logic into single function
- Added percentile filtering for capacity values
- Removed dead/commented code
- Simplified column management

UPDATED: January 12, 2026
- Fixed argument order in _convert_pm1() call (pm1_metric_col and pm1_unit_col were swapped)
- Added defensive df.copy() at start of all functions to prevent mutation on re-execution
- Fixed duplicate column issue in output concatenation
- Removed _fill_missing_from_bounds() - NaN values propagate naturally per validation framework
- Removed _check_out_of_bounds() - 95% CI percentile filter handles outliers
- Removed out_of_bound_method parameter - no longer needed
- Removed legacy code for clamping and keeping as is - now handled via filtering

NO LONGER USING THE SUM OF THE HEATING AND COOLING LOADS FOR SYSTEM SIZE AND COST ESTIMATION
- The supplemental heating (electric strip heat) is implicitly included in the REMDB v4 costs
- Also, the primary system size is the same for both heating and cooling. You wouldn't have two different ASHP tonnages.
"""
```

**AFTER**
```python
"""REMDB v4 installed-cost metric preparation.

This module turns raw EUSS equipment fields into the performance metrics that
the REMDB v4 cost regression consumes. The single public entry point is
add_remdb_metrics(), which handles both replacement (counterfactual, like-for-
like) and upgrade (heat pump) metrics.

What the pipeline does, in order (see add_remdb_metrics):
  1. Optional percentile filtering of capacity outliers.
  2. Assign a REMDB row_id from the baseline (replacement) or heat-pump
     (upgrade) equipment type.
  3. Map the REMDB regression coefficients and unit specs onto each home.
  4. Convert capacity (pm1) and efficiency (pm2) into the units the regression
     expects.
  5. Replacement only: raise below-floor efficiencies (pm2) up to the minimum
     efficiency equipment sold today, preserving the raw value in a
     ``{pm2_col}_original`` column.
  6. Report diagnostics (including any capacity values outside the REMDB
     training bounds) and return a summary frame plus a detailed frame.

Capacity outliers are reported but never modified: pm1 is used as converted.
Homes far outside the training range are handled by the upstream percentile
filter and by NaN propagation, not by clamping.

NaN handling: homes with invalid fuel/technology types resolve to row_id
'unknown' and carry NaN metrics, which propagate to NaN costs downstream. This
is intentional and matches the validation framework's masking.

System sizing: a single primary system size drives both the heating and cooling
cost; supplemental electric-strip heat is already priced into the REMDB v4
figures, so heating and cooling loads are not summed into one larger system.
"""
```

## Change 1.2 -- typing import (base line ~39)

WHY: `_apply_efficiency_floor` gains a `Dict[str, float]` type hint (Change 1.9).

**BEFORE**
```python
from typing import Optional, Tuple, Literal
```

**AFTER**
```python
from typing import Dict, Optional, Tuple, Literal
```

## Change 1.3 -- constants import (base lines 43-47)

WHY: `CAPACITY_BOUND_CLAMPING_TOLERANCE` is removed (Task 4).

**BEFORE**
```python
from cmu_tare_model.constants import (
    EQUIPMENT_SPECS,
    EFFICIENCY_FLOORS_PM2,
    CAPACITY_BOUND_CLAMPING_TOLERANCE,
)
```

**AFTER**
```python
from cmu_tare_model.constants import (
    EQUIPMENT_SPECS,
    EFFICIENCY_FLOORS_PM2,
)
```

## Change 1.4 -- dead block in `_assign_replacement_row_id` (base lines ~144-150)

WHY: remove dead commented "NON-HVAC END USES" block. Delete it; keep the
`df_copy[row_id_col] = np.select(...)` line above and `return df_copy` below.

**BEFORE**
```python
        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')

    # =========================================
    # DELETE FOR NOW - NON-HVAC END USES
    # =========================================

    # else:
    #     # raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking'")
    #     raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling'")

    return df_copy
```

**AFTER**
```python
        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')

    return df_copy
```

## Change 1.5 -- dead block in `_assign_upgrade_row_id` (base lines ~186-194)

WHY: same dead block; replace the stray one-line comment + dead block with a
clear explanatory comment.

**BEFORE**
```python
        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')

    # Cooling only considered as replacement cost because heat pumps are the upgrade option for both heating and cooling

    # =========================================
    # DELETE FOR NOW - NON-HVAC END USES
    # =========================================

    # else:
    #     # raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking'")
    #     raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling'")

    return df_copy
```

**AFTER**
```python
        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')

    # Cooling is priced only as a replacement cost: the heat pump upgrade already
    # serves both the heating and the cooling load, so there is no separate
    # cooling upgrade row_id.

    return df_copy
```

## Change 1.6 -- `_convert_pm1` docstring ASCII (base lines ~247-248)

WHY: ASCII-only rule. The two bullet lines use the division/multiplication signs.

**BEFORE**
```python
    - "Tons": kBtu/h ÷ 12 (heat pumps, central ACs)
    - "BTU/hr": kBtu/h × 1000 (furnaces, boilers, baseboard)
```

**AFTER**
```python
    - "Tons": kBtu/h / 12 (heat pumps, central ACs)
    - "BTU/hr": kBtu/h x 1000 (furnaces, boilers, baseboard)
```

## Change 1.7 -- `_convert_pm1` inline comments ASCII (base lines ~275, ~281)

WHY: ASCII-only rule (arrows and division/multiplication signs). These two
comment lines are a few lines apart -- edit each in place.

**BEFORE (line ~275)**
```python
    # Tons (heat pumps, ACs): kBtu/h → Tons (÷12)
```
**AFTER**
```python
    # Tons (heat pumps, ACs): kBtu/h -> Tons (/12)
```

**BEFORE (line ~281)**
```python
    # BTU/hr (furnaces, boilers, baseboard): kBtu/h → BTU/hr (×1000)
```
**AFTER**
```python
    # BTU/hr (furnaces, boilers, baseboard): kBtu/h -> BTU/hr (x1000)
```

## Change 1.8 -- `_convert_pm2` docstring ASCII (base line ~310)

WHY: ASCII-only rule (arrow).

**BEFORE**
```python
    - AFUE: Extract numeric, divide by 100 (e.g., "80% AFUE" → 0.80)
```
**AFTER**
```python
    - AFUE: Extract numeric, divide by 100 (e.g., "80% AFUE" -> 0.80)
```

## Change 1.9 -- `_apply_efficiency_floor` signature type hint (base line ~363)

WHY: type-hint standard.

**BEFORE**
```python
    row_id_col: str,
    pm2_col: str,
    efficiency_floors: dict,
    verbose: bool = False
) -> pd.DataFrame:
```
**AFTER**
```python
    row_id_col: str,
    pm2_col: str,
    efficiency_floors: Dict[str, float],
    verbose: bool = False
) -> pd.DataFrame:
```

## Change 1.10 -- `_apply_efficiency_floor` body vectorized (base lines ~374-406)

WHY: replace the per-row_id Python loop with a single map + `Series.clip`.
**Verified byte-identical** to the loop (below-floor -> floor; at/above ->
unchanged; NaN floor or NaN pm2 -> unchanged; `_original` still written first).
This is the body from `df_out = df.copy()` down to just before `return df_out`.

**BEFORE**
```python
    df_out = df.copy()
    
    # Preserve the raw EUSS efficiency before any modification
    original_col = f'{pm2_col}_original'
    df_out[original_col] = df_out[pm2_col].copy()
    
    total_clamped = 0

    for row_id, floor in efficiency_floors.items():
        mask = (df_out[row_id_col] == row_id) & df_out[pm2_col].notna()
        if not mask.any():
            continue

        below_floor = mask & (df_out[pm2_col] < floor)
        n_below = below_floor.sum()

        if n_below > 0:
            original_values = df_out.loc[below_floor, pm2_col]
            df_out.loc[below_floor, pm2_col] = floor
            total_clamped += n_below

            if verbose:
                n_total = mask.sum()
                print(f"    {row_id}: clamped {n_below:,}/{n_total:,} homes "
                      f"from [{original_values.min():.2f}–{original_values.max():.2f}] "
                      f"→ floor {floor}")

    if verbose:
        if total_clamped == 0:
            print(f"    No homes required efficiency floor clamping.")
        else:
            print(f"    Total clamped: {total_clamped:,} homes")
            print(f"    Original values preserved in: {original_col}")

    return df_out
```

**AFTER**
```python
    df_out = df.copy()

    # Preserve the raw EUSS efficiency before any flooring is applied.
    original_col = f'{pm2_col}_original'
    df_out[original_col] = df_out[pm2_col].copy()

    # Map each home's row_id to its floor. Rows whose row_id is not in the
    # floors dict get NaN; Series.clip treats a NaN lower bound as "do not
    # clip", so those rows (and rows whose pm2 is NaN) pass through unchanged.
    # This raises every below-floor value up to its floor in one vectorized
    # step, exactly as the previous per-row_id loop did.
    floor_by_row = df_out[row_id_col].map(efficiency_floors)
    df_out[pm2_col] = df_out[pm2_col].clip(lower=floor_by_row)

    if verbose:
        # A home was raised when its floored pm2 differs from the original and
        # the original was present (NaN originals are never touched).
        raised = (
            (df_out[pm2_col] != df_out[original_col])
            & df_out[original_col].notna()
        )
        total_clamped = int(raised.sum())
        if total_clamped == 0:
            print("    No homes required efficiency floor clamping.")
        else:
            print(f"    Total clamped: {total_clamped:,} homes")
            print(f"    Original values preserved in: {original_col}")

    return df_out
```

## Change 1.11 -- DELETE `_log_capacity_clamp` and `_apply_capacity_clamping` (base lines ~430-547) [Task 4, value-moving]

WHY: capacity clamping removed. **Delete both functions entirely.** They sit
between the end of `_apply_efficiency_floor` (`return df_out`) and the start of
`def _report_bounds_comparison(`. After deletion there should be exactly two
blank lines between `_apply_efficiency_floor`'s `return df_out` and
`def _report_bounds_comparison(`.

**BEFORE (delete this whole block)**
```python
def _log_capacity_clamp(
    mask: pd.Series,
    df: pd.DataFrame,
    row_id_col: str,
    pm1: pd.Series,
    bound: pd.Series,
    was_clamped: bool,
    direction: str,
    tolerance: float,
) -> None:
    """Print per-row_id diagnostics for capacity clamping.

    Args:
        mask: Boolean mask of affected homes.
        df: DataFrame for row_id lookup.
        row_id_col: Column with REMDB row_id.
        pm1: Original pm1 values (before clamping).
        bound: Bound values aligned to pm1.
        was_clamped: True if homes were clamped, False if left unchanged.
        direction: 'below' or 'above' (relative to bound).
        tolerance: Tolerance fraction for context in message.
    """
    for rid in df.loc[mask, row_id_col].unique():
        m = mask & (df[row_id_col] == rid)
        n = int(m.sum())
        vals = pm1[m]
        bnd = bound[m].iloc[0]
        pcts = ((vals - bnd).abs() / bnd * 100)

        if was_clamped:
            clamp_dir = "UP" if direction == "below" else "DOWN"
            print(f"    {rid}: clamped {n:,} homes {clamp_dir} to bound "
                  f"{bnd:.2f} (from [{vals.min():.2f}–{vals.max():.2f}])")
        else:
            print(f"    {rid}: {n:,} homes NOT clamped "
                  f"(>{tolerance*100:.0f}% {direction} bound {bnd:.2f}; "
                  f"range [{vals.min():.2f}–{vals.max():.2f}], "
                  f"{pcts.min():.0f}%–{pcts.max():.0f}% away)")


def _apply_capacity_clamping(
    df: pd.DataFrame,
    row_id_col: str,
    pm1_col: str,
    pm1_lower_bound_col: str,
    pm1_upper_bound_col: str,
    tolerance: float = CAPACITY_BOUND_CLAMPING_TOLERANCE,
    verbose: bool = False
) -> pd.DataFrame:
    """Clamp pm1 (capacity) to REMDB training bounds where within tolerance.

    For replacement cost estimation, some EUSS capacity values fall slightly
    outside the REMDB v4 regression's training range.  This function clamps
    pm1 values to the nearest training bound, but ONLY when the value is
    within *tolerance* (fractional) of that bound.

    Values far outside the bounds (> tolerance) are left unchanged so they
    can be handled separately (e.g., sq-ft-based NaN-masking per protocol).

    Args:
        df: DataFrame with pm1 values already converted by _convert_pm1().
        row_id_col: Column containing the REMDB row_id.
        pm1_col: Column containing the converted pm1 values.
        pm1_lower_bound_col: Column with REMDB lower training bound.
        pm1_upper_bound_col: Column with REMDB upper training bound.
        tolerance: Maximum fractional distance from bound for clamping.
        verbose: If True, print diagnostic info about clamped homes.

    Returns:
        DataFrame with pm1 values clamped where applicable.
    """
    df_out = df.copy()

    pm1 = pd.to_numeric(df_out[pm1_col], errors='coerce')
    lower = pd.to_numeric(df_out[pm1_lower_bound_col], errors='coerce')
    upper = pd.to_numeric(df_out[pm1_upper_bound_col], errors='coerce')

    valid = pm1.notna()
    total_clamped = 0

    # --- Lower-bound clamping ---
    below_lower = valid & lower.notna() & (pm1 < lower)
    if below_lower.any():
        frac_below = (lower - pm1) / lower
        within_tol = below_lower & (frac_below <= tolerance)
        beyond_tol = below_lower & (frac_below > tolerance)

        if within_tol.any():
            df_out.loc[within_tol, pm1_col] = lower[within_tol]
            total_clamped += int(within_tol.sum())
            if verbose:
                _log_capacity_clamp(within_tol, df_out, row_id_col, pm1, lower,
                                    was_clamped=True, direction="below", tolerance=tolerance)
        if verbose and beyond_tol.any():
            _log_capacity_clamp(beyond_tol, df_out, row_id_col, pm1, lower,
                                was_clamped=False, direction="below", tolerance=tolerance)

    # --- Upper-bound clamping ---
    above_upper = valid & upper.notna() & (pm1 > upper)
    if above_upper.any():
        frac_above = (pm1 - upper) / upper
        within_tol = above_upper & (frac_above <= tolerance)
        beyond_tol = above_upper & (frac_above > tolerance)

        if within_tol.any():
            df_out.loc[within_tol, pm1_col] = upper[within_tol]
            total_clamped += int(within_tol.sum())
            if verbose:
                _log_capacity_clamp(within_tol, df_out, row_id_col, pm1, upper,
                                    was_clamped=True, direction="above", tolerance=tolerance)
        if verbose and beyond_tol.any():
            _log_capacity_clamp(beyond_tol, df_out, row_id_col, pm1, upper,
                                was_clamped=False, direction="above", tolerance=tolerance)

    if verbose and total_clamped == 0:
        print(f"    No homes required capacity bound clamping.")

    return df_out


def _report_bounds_comparison(
```

**AFTER (the block collapses to just the next function's def)**
```python
def _report_bounds_comparison(
```

## Change 1.12 -- `filter_by_percentile` print ASCII (base line ~695)

WHY: ASCII-only rule (arrow).

**BEFORE**
```python
        print(f"   Rows: {n_original:,} → {n_filtered:,} (removed {n_removed:,}, {pct_removed:.2f}%)")
```
**AFTER**
```python
        print(f"   Rows: {n_original:,} -> {n_filtered:,} (removed {n_removed:,}, {pct_removed:.2f}%)")
```

## Change 1.13 -- `add_remdb_metrics` return annotation (base line ~713)

WHY: remove the stray Unicode return-marker comment.

**BEFORE**
```python
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # ← Always return tuple
```
**AFTER**
```python
) -> Tuple[pd.DataFrame, pd.DataFrame]:
```

## Change 1.14 -- Step 4.5a comment ASCII (base line ~898)

WHY: ASCII-only rule (arrows). This is inside `add_remdb_metrics`, in the
efficiency-floor step comment.

**BEFORE**
```python
    #   SEER 8 → SEER 15, AFUE 60% → AFUE 80%, etc.
```
**AFTER**
```python
    #   SEER 8 -> SEER 15, AFUE 60% -> AFUE 80%, etc.
```

## Change 1.15 -- DELETE Step 4.5b block in `add_remdb_metrics` (base lines ~912-932) [Task 4, value-moving]

WHY: capacity clamping removed. Replace the whole Step 4.5b block with a short
comment. It sits between the end of the Step 4.5a `_apply_efficiency_floor(...)`
call and the `# STEP 5: Report bounds comparison` banner.

**BEFORE**
```python
    # =========================================================================
    # STEP 4.5b: Clamp capacity to REMDB training bounds (replacement only)
    # =========================================================================
    # For replacement costs only: clamp pm1 toward the REMDB training-data
    # bounds when the value is within TOLERANCE (default 10%) of a bound.
    #   - Slightly below lower bound  → clamp UP   to the lower bound
    #   - Slightly above upper bound  → clamp DOWN to the upper bound
    #   - Far outside bounds           → leave unchanged
    if metric_type == 'replacement':
        if verbose:
            print(f"\n  Step 4.5b: Clamping pm1 (capacity) to REMDB bounds "
                  f"(±{CAPACITY_BOUND_CLAMPING_TOLERANCE*100:.0f}% tolerance, replacement only)")
        df_copy = _apply_capacity_clamping(
            df=df_copy,
            row_id_col=row_id_col,
            pm1_col=pm1_col,
            pm1_lower_bound_col=f'{prefix}pm1_lower_bound',
            pm1_upper_bound_col=f'{prefix}pm1_upper_bound',
            tolerance=CAPACITY_BOUND_CLAMPING_TOLERANCE,
            verbose=verbose
        )
```

**AFTER**
```python
    # Capacity (pm1) is used exactly as converted. Values outside the REMDB
    # training bounds are reported in Step 5 but never modified; the upstream
    # percentile filter and NaN propagation handle genuine outliers.
```

---

# File 2 -- `cmu_tare_model/constants.py`

## Change 2.1 -- Remove `CAPACITY_BOUND_CLAMPING_TOLERANCE` (base lines ~259-278) [Task 4]

WHY: the constant and its clamping step are removed; replace the whole block
(header + comment + assignment) with a rationale-only note. It sits between the
`EFFICIENCY_FLOORS_PM2 = { ... }` block and the
`# CONSTANTS: BSQ / EUSS TIMESERIES COLUMN NAMES` banner.

**BEFORE**
```python
# =============================================================
# CONSTANTS: CAPACITY BOUND CLAMPING FOR REPLACEMENT COSTS
# =============================================================
# Applied to pm1 (capacity) values BEFORE the REMDB v4 regression,
# only for replacement costs.
#
# Approach: Use the REMDB v4 training-data lower/upper bounds and
# only clamp values that are within TOLERANCE of a bound.
#
#   - Values slightly below the lower bound (within TOLERANCE) are
#     clamped UP to the lower bound.
#   - Values slightly above the upper bound (within TOLERANCE) are
#     clamped DOWN to the upper bound.
#   - Values far outside the bounds (> TOLERANCE) are left unchanged.
#
# Example (lower bound = 1.5 tons, tolerance = 0.10):
#   1.4 tons → clamp to 1.5  (6.7% below, within 10%)
#   1.0 tons → leave as 1.0  (33% below, beyond 10%)
# =============================================================
CAPACITY_BOUND_CLAMPING_TOLERANCE = 0.10  # 10%
```

**AFTER**
```python
# =============================================================
# CONSTANTS: CAPACITY BOUNDS FOR REPLACEMENT COSTS
# =============================================================
# Capacity (pm1) is fed to the REMDB v4 regression exactly as converted from
# the EUSS size fields -- it is never clamped to the training bounds. Values
# outside the bounds are reported for diagnostics only (see
# _report_bounds_comparison in remdb_v4_installed_cost_utils.py); genuine
# outliers are handled by the upstream capacity percentile filter and by NaN
# propagation. The former CAPACITY_BOUND_CLAMPING_TOLERANCE and the tolerance-
# based clamping step were removed on 07 July 2026 because clamping silently
# moved a small number of homes' capacities (and therefore their capital
# costs) with no methodological basis over the plain converted value.
```

---

# File 3 -- `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py`

All three changes are inside `calculate_capital_costs`.

## Change 3.1 -- docstring ASCII: net formula (base lines ~472-473)

WHY: ASCII-only rule (minus signs).

**BEFORE**
```python
    - 'heating': net = total − heating replacement cost  (Case A)
    - 'heating_and_cooling': net = total − (heating + cooling replacement cost)  (Case B)
```
**AFTER**
```python
    - 'heating': net = total - heating replacement cost  (Case A)
    - 'heating_and_cooling': net = total - (heating + cooling replacement cost)  (Case B)
```

## Change 3.2 -- docstring ASCII: scenario args (base lines ~486-487)

WHY: ASCII-only rule (em dashes).

**BEFORE**
```python
            'heating' (default, Case A) — only heating replacement cost subtracted.
            'heating_and_cooling' (Case B) — heating + cooling replacement cost subtracted.
```
**AFTER**
```python
            'heating' (default, Case A) -- only heating replacement cost subtracted.
            'heating_and_cooling' (Case B) -- heating + cooling replacement cost subtracted.
```

## Change 3.3 -- fillna(0) audit WHY comment (base line ~546) [Task 3]

WHY: document the pre-mask `.fillna(0)` audit. Insert the comment block
immediately before the existing
`# Single policy scenario ('2025 Reference Case'): IRA rebates always apply.`
line (which is followed by `if category == 'heating':`).

**BEFORE**
```python
    # Single policy scenario ('2025 Reference Case'): IRA rebates always apply.
    if category == 'heating':
```
**AFTER**
```python
    # The .fillna(0) on each cost/rebate column below runs BEFORE the valid_mask
    # is applied at the end of this function. That ordering is safe only if no
    # home inside valid_mask can carry a NaN in a required cost/rebate column --
    # otherwise a valid home would silently read as cost 0 (or an un-rebated
    # cost). By construction the upgrade/replacement/rebate columns are written
    # for exactly the valid homes and NaN'd for the rest, so a valid home should
    # always have a real value here. That guarantee has NOT been confirmed
    # empirically on the full EUSS stock in this session (the stock and income
    # data are offline), so the audit is recorded but the silent fill is left
    # unchanged. TODO (researcher): on a full run, count valid_mask homes with a
    # NaN in any required column; if that count is zero, replace these .fillna(0)
    # calls with a fail-loud check per the fail-fast standard.
    #
    # Single policy scenario ('2025 Reference Case'): IRA rebates always apply.
    if category == 'heating':
```

---

# File 4 -- `cmu_tare_model/private_impact/data_processing/determine_rebate_eligibility_and_amount.py`

## Change 4.1 -- comment ASCII (base line ~168)

WHY: ASCII-only rule (em dash). Inside `calculate_percent_AMI`.

**BEFORE**
```python
    # being processed — critical for MP4 vs MP8 result consistency.
```
**AFTER**
```python
    # being processed -- critical for MP4 vs MP8 result consistency.
```

---

# File 5 -- `cmu_tare_model/tare_model_main_v2_3_EXPORT_5July2026.py`

## Change 5.1 -- Define `GRID_IMPACT_ANALYSIS` run control (base line ~101) [Task 5]

WHY: the file references `GRID_IMPACT_ANALYSIS` in two `if GRID_IMPACT_ANALYSIS:`
gates but never defines it (latent `NameError`). Insert a new run-control cell
right after the setup print block (which ends with `""")`) and before the
`# %%` / `# Select whether to begin new run...` cell.

> This is an export of `tare_model_main_v2_3.ipynb`. After pasting, **backport
> the same new cell into the `.ipynb`** (see backport list at the end).

**BEFORE**
```python
""")

# %%
# Select whether to begin new run or visualize existing model outputs
while True:
```
**AFTER**
```python
""")

# %%
# =============================================================================
# ANALYSIS RUN CONTROLS
# =============================================================================
# GRID_IMPACT_ANALYSIS gates the grid-impact section further down. That section
# runs the peak-demand notebook via %run and needs live model outputs plus AWS
# access, so it is off by default; set it to True only in an environment that
# has those. It is defined here (rather than inline) so the toggle lives with
# the other run controls and the `if GRID_IMPACT_ANALYSIS:` gates below always
# resolve to a defined name.
GRID_IMPACT_ANALYSIS = False

# %%
# Select whether to begin new run or visualize existing model outputs
while True:
```

---

# File 6 -- `CLAUDE.md` (documentation)

Five edits align the reference doc to the six-case scheme. Base line numbers in
parentheses.

## Change 6.1 -- NPV cases + adopter blocks (base lines ~124-139)

**BEFORE**
````
**NPV cases (three per MP, as of Session A refactor):**
```
ref2025_mp{mp}_heatingSavings_coolingLCC_private_npv_{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingSavings_private_npv_{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingLCC_private_npv_{method_suffix}
```
- `LCC` = that end-use's avoided-replacement capital is credited in the NPV
- `Savings` = only operating savings credited for that end-use
- All three cases include BOTH heating and cooling operating savings

**Economic adopter columns (three per MP, as of Session A refactor):**
```
ref2025_mp{mp}_heatingSavings_coolingLCC_econ_adopter_{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingSavings_econ_adopter_{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingLCC_econ_adopter_{method_suffix}
```
````

**AFTER**
````
**NPV cases (six per MP, as of the 6 July 2026 session):**
```
ref2025_mp{mp}_heatingSavings_coolingLCC_sub_private_npv{method_suffix}
ref2025_mp{mp}_heatingSavings_coolingLCC_unsub_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingSavings_sub_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingSavings_unsub_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingLCC_sub_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_private_npv{method_suffix}
```
- Build with `create_npv_case_col(scenario_prefix, npv_case, method_suffix)`;
  `npv_case` must be one of `NPV_CASE_CATEGORIES` (column_names.py). Note there is
  no cost-scenario token and no WTP token in these names.
- `LCC` = that end-use's avoided-replacement capital is credited in the NPV
- `Savings` = only operating savings credited for that end-use
- `_sub` = subsidized (IRA rebate applied); `_unsub` = unsubsidized companion
- All six cases include BOTH heating and cooling operating savings
- `{method_suffix}` already carries its own leading underscore (e.g. `_fixed_base`)

**Economic adopter columns (six per MP, as of the 6 July 2026 session):**
```
ref2025_mp{mp}_{npv_case}_econ_adopter{method_suffix}
```
one per `npv_case` in `NPV_CASE_CATEGORIES` (the same six tokens listed above,
including `_sub`/`_unsub`).
````

## Change 6.2 -- Sensitivity table NPV-scope row (base line ~158)

**BEFORE**
```
| NPV scope | `heatingSavings_coolingLCC` \| `heatingLCC_coolingSavings` \| `heatingLCC_coolingLCC` |
```
**AFTER**
```
| NPV scope | `heatingSavings_coolingLCC` \| `heatingLCC_coolingSavings` \| `heatingLCC_coolingLCC`, each x `_sub` / `_unsub` (six cases; see `NPV_CASE_CATEGORIES`) |
| Subsidy split | `_sub` (IRA rebate applied) \| `_unsub` (unsubsidized companion) |
```

## Change 6.3 -- Golden-values PENDING row (base line ~229, after the ACS-2024 LMI row)

**AFTER (add this new row directly beneath the `LMI ... 62.4% ... Session 1e` row)**
```
| Mean economic adoption rate (six NPV cases, `_sub`/`_unsub`) | PENDING | PENDING | AEO2026/Cambium2024 | To be re-derived; the six-case scheme replaced the retired heating-only case, so no golden value exists yet. Do not backfill without a full model run. |
```

## Change 6.4 / 6.5 -- Session log rows (base line ~244, after the `Session A` row)

**AFTER (add these two rows directly beneath the `Session A` row)**
```
| 6 July 2026 | 6 Jul 2026 | Six NPV/adopter cases: each of the three scope tokens split into `_sub`/`_unsub`; `create_npv_case_col` added; `peak_load_functions` defaults to `heatingLCC_coolingSavings_sub`; Option A dotplot plots subsidized adoption with unsubsidized deltas. Loose ends closed in Session B below. |
| Session B | 7 Jul 2026 | Capital-cost refactor + baseline oracle (`scripts/capture_capital_cost_baseline.py`). CLAUDE.md updated to six-case naming. Old-token sweep: `create_npv_col` (moreWTP/lessWTP) still coexists with `create_npv_case_col`; notebook exports carry `moreWTP`/`iraRef`/`preIRA` stragglers plus half-migrated `create_npv_case_col(..., wtp=..., cost_scenario=...)` calls that raise `TypeError`. Flagged for hand-migration. Propagation verified PASS for `compute_adoption_rate`, `visuals_adoption_potential`, `visuals_adoption_dotplot`. See `REFACTORING_GUIDE_07July2026.md`. |
```

---

# File 7 -- `scripts/capture_capital_cost_baseline.py` (NEW)

No BEFORE -- this is a new file (the functional-equivalence oracle). Nothing to
replace; it is committed in full. It builds a deterministic synthetic home set,
runs the real `add_remdb_metrics` for the three pipeline combos, and dumps
`df_main`, `df_detailed`, the v4 regression cost per percentile, and a hashed
manifest under `baseline_capture/`. To re-run:
`PYTHONPATH=. python scripts/capture_capital_cost_baseline.py`. Requires the
REMDB v4 CSV at `cmu_tare_model/data/retrofit_costs/` (git-ignored, local only).

---

# Verification performed

- After File 1 Changes 1.1-1.14 (before removing clamping): harness output
  (`df_main`, `df_detailed`, v4 regression costs at all percentiles) **byte-
  identical** to `baseline_capture/` via
  `pandas.testing.assert_frame_equal(check_exact=True)`;
  `test_efficiency_floor_refactoring.py` 5/5 unmodified.
- After File 1 Changes 1.11/1.15 + File 2 (clamping removed): only replacement
  pm1/costs moved (see impact table below); pm2 and `pm2_..._original`
  byte-identical.
- Files 3, 4, 6: comment/docstring only -- value-identical.
- Full suite after all changes: `33 failed, 236 passed, 24 errors` (== base bar,
  no new failures).

## Task 4 capacity-clamping impact (from the frozen `baseline_capture/`)

Replacement metrics only (upgrade unaffected); same 3 synthetic homes in heating
and cooling replacement:

| Home | row_id | pm1 with clamp | pm1 without clamp | delta restored |
|---|---|---|---|---|
| 10 | air_source_heat_pump_non_ducted_multi_zone | 5.000 | 5.500 | +0.500 |
| 13 | air_source_heat_pump_centrally_ducted | 1.500 | 1.425 | -0.075 |
| 14 | air_source_heat_pump_centrally_ducted | 5.000 | 5.250 | +0.250 |

v4 mid installed cost, home 10: **$25,069.90 -> $27,417.46**. (low: 15,041.94 ->
16,450.48; high: 35,097.86 -> 38,384.44.)

---

# Deferred / hand-migration (NOT changed this session -- do these yourself)

These were skipped on purpose because they cannot be run or verified offline and
involve non-mechanical judgment (Decision Rules: skip + document, don't guess).

### D1 -- Old-token migration in the orchestration / notebook-export layer
`moreWTP` / `lessWTP` / `iraRef` / `preIRA` and old case tokens still appear in:
`utils/column_names.py` (`create_npv_col` + docstrings),
`private_impact/calculate_lifetime_private_impact.py`
(`calculate_and_update_npv` builds `create_npv_col(... wtp=... )`),
`model_scenarios/tare_run_simulation_v2_3_EXPORT_28June2026.py`,
`model_scenarios/tare_scenarios_v2_3_EXPORT_28June2026.py`,
both `adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_*.py`,
both `tare_model_main_v2_3_EXPORT_*.py`.
**Broken calls to fix first:** `create_npv_case_col(scenario_prefix, npv_case,
wtp='moreWTP', cost_scenario=..., method_suffix=...)` -- e.g.
`tare_run_simulation_v2_3_EXPORT_28June2026.py:259` -- raises `TypeError`
because `create_npv_case_col` takes only `(scenario_prefix, npv_case,
method_suffix)`. Decide whether `create_npv_col` / `calculate_and_update_npv`
are retired; repoint live call sites to `create_npv_case_col` with the correct
six-case token (the old `moreWTP/lessWTP x heating/heating_and_cooling` -> six
`sub/unsub x LCC` mapping is a research call), then re-export the notebooks.

### D2 -- `calculate_capital_costs` fillna(0) -> fail-loud
On a full run, count `valid_mask` homes with a NaN in any required cost/rebate
column. If zero, replace the `.fillna(0)` calls with a raise per the fail-fast
standard (the WHY comment from Change 3.3 marks the spot).

### D3 -- `tare_model_main_v2_3_EXPORT_5July2026.py` restructure
Consolidate the two near-duplicate MP8 climate-SCC histogram cells (both assign
`fig_heating_climate_scc_FIXED_BASE`, ~lines 366 and 476); migrate their
`create_npv_col(..., 'moreWTP', ...)` strings (lines 354, 464) to
`create_npv_case_col`; transport the adoption summary, dotplot maps, and Option A
simplified dotplot from the 5 July `bill_savings` EXPORT into the
`ECONOMIC ADOPTION POTENTIAL` / `PLACEHOLDER` cells.

### D4 -- `validate_capital_costs.py`
No change needed. Its `_build_clamping_summary` reports the **efficiency floor**
(pm2), which is preserved -- not capacity clamping. Capacity out-of-bounds
reporting already exists via `_report_bounds_comparison`.

---

# `.ipynb` backport list (never edit `.ipynb` from a `.py` export blindly)

| Change | `.py` export edited | Backport into |
|---|---|---|
| 5.1 -- `GRID_IMPACT_ANALYSIS = False` run-control cell | `tare_model_main_v2_3_EXPORT_5July2026.py` | `cmu_tare_model/tare_model_main_v2_3.ipynb` |
| D3 (when done by hand) | same export | same `.ipynb` |

Files 1-4 and 6 are plain `.py`/`.md` -- no notebook backport needed.

---

# Environment notes (why verification looks the way it does)

- The ~331k-home EUSS stock and the BLS/ACS income inputs are on Zenodo and
  git-ignored, so the full model / rebate / NPV pipeline cannot run here.
- The REMDB v4 cost table was supplied and placed at
  `cmu_tare_model/data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv`
  (git-ignored, local only). This makes the oracle and the efficiency-floor test
  runnable.
- Equivalence was therefore proven on a deterministic synthetic input built to
  hit every code path the refactor touches -- a valid refactor-equivalence proof
  for that input.
- Branch: all work is on `claude/tare-capital-cost-refactor-754hua`, which is at
  the same base commit as `update-data-and-projections-aeo2026-cambium2024`; the
  prompt's branch name differs but the base and content are correct.
