# VSL Adjustment Factor Correction - October 2025

## Summary

Corrected the VSL (Value of Statistical Life) adjustment factors to properly align with the marginal social cost (MSC) data. The primary issue was a base year normalization mismatch, resulting in health impacts being underestimated by approximately 1.0% across all years.

## Issue Identified

### Original Configuration
- **MSC Data**: Based on EPA VSL guidance of 11.3M USD2021, inflated to 12.71M USD2023
  - Files: `rcm_msc_county_vsl1271_usd2023_*.csv`
  - CPI ratio (2023/2021): 1.1244861054729305

- **VSL Adjustment Factors**: Based on EPA VSL guidance of 11.0M USD2022, inflated to 11.45M USD2023
  - File: `vsl_adjustment_factor_2023-2050.xlsx`
  - CPI ratio (2023/2022): 1.04116451111376
  - **Base year normalization: 2024** (adj_factor_2024 = 1.0)

### The Problem

The MSC data is expressed in 2023 dollars, but the adjustment factors were normalized to 2024 as the base year. This caused a 1-year offset error:
- For calculations in 2025, the old adjustment factor was 1.01 (relative to 2024)
- But it should have been 1.0201 (relative to 2023)
- This 1% difference compounds across all years

**Note**: While the VSL base values differed (12.71M vs 11.45M), this 11% difference in absolute VSL level does NOT directly cause an error because:
1. The MSC data already has the VSL=12.71M baked into the $/ton values
2. Adjustment factors only capture temporal growth (1% annually), not absolute levels
3. The real issue was the base year mismatch, not the VSL source difference

## Changes Made

### 1. Updated VSL Adjustment Factor File
**File**: `cmu_tare_model/data/marginal_social_costs/vsl_adjustment_factor_2023-2050.xlsx`

**New Configuration**:
- Base VSL: 12.71M USD2023 (from 11.3M USD2021 × 1.1244861)
- Base year: **2023** (adj_factor_2023 = 1.0)
- Growth rate: 1.01 (1% annually per HHS guidance)
- Years covered: 2023-2050

**Adjustment Factor Comparison**:
```
Year    Old Factor    New Factor    Change
2023    0.990099      1.000000      +1.0%
2024    1.000000      1.010000      +1.0%
2025    1.010000      1.020100      +1.0%
2026    1.020100      1.030301      +1.0%
...     ...           ...           +1.0%
```

**Backup**: Original file saved as `vsl_adjustment_factor_2023-2050_BACKUP_old_vsl11.45.xlsx`

### 2. Updated Documentation
**File**: `cmu_tare_model/public_impact/data_processing/create_lookup_health_vsl_adjustment.py`

Corrected the docstring to:
- Reflect the correct VSL calculation: 11.3M USD2021 → 12.71M USD2023
- Update base year from 2024 to 2023
- Add correction notes explaining the changes made

## Impact on Results

### Direct Impact
- All health impact calculations will increase by approximately **+1.0%**
- This applies uniformly to all years (2023-2050)
- Impact is due to fixing the base year normalization, not the VSL level difference

### Example Calculation
For health damage in 2025:
- **Old (Incorrect)**: MSC_2023 × 1.01 = $101,000 per ton
- **New (Correct)**: MSC_2023 × 1.0201 = $102,010 per ton
- **Difference**: +1.0%

### Affected Outputs
All results from `calculate_lifetime_health_impacts_sensitivity.py` that use:
- `lookup_health_vsl_adjustment` dictionary
- Fossil fuel MSC adjustments (line 382-383)
- Electricity MSC adjustments (line 421-422)

## Verification Steps

1. **Check adjustment factor loading**:
   ```python
   from cmu_tare_model.public_impact.data_processing.create_lookup_health_vsl_adjustment import lookup_health_vsl_adjustment
   print(lookup_health_vsl_adjustment[2023])  # Should be 1.0
   print(lookup_health_vsl_adjustment[2024])  # Should be 1.01
   ```

2. **Verify MSC consistency**:
   - MSC data files still use VSL=12.71M (correct, no changes needed)
   - Adjustment factors now properly aligned with 2023 base year

3. **Re-run health impact calculations**:
   - All health benefits should increase by ~1.0%
   - Time series patterns should remain the same (same growth rate)

## Files Changed

1. `cmu_tare_model/data/marginal_social_costs/vsl_adjustment_factor_2023-2050.xlsx`
   - Regenerated with correct base year (2023) and VSL source documentation

2. `cmu_tare_model/public_impact/data_processing/create_lookup_health_vsl_adjustment.py`
   - Updated docstring with correct VSL calculation and correction notes

3. `cmu_tare_model/data/marginal_social_costs/vsl_adjustment_factor_2023-2050_BACKUP_old_vsl11.45.xlsx`
   - Backup of original file for reference

## Next Steps

1. **Re-run model**: Execute `calculate_lifetime_health_impacts_sensitivity.py` with corrected factors
2. **Update results**: All health impact outputs should be regenerated
3. **Verify**: Check that health impacts increased by ~1% across all scenarios
4. **Documentation**: Update any reports or papers citing the previous health impact values

## Technical Details

### VSL Calculation Chain

**Correct (New)**:
```
EPA VSL (2021) → Inflate to 2023 → Apply growth
11.3M USD2021 → 12.71M USD2023 → 12.71 × 1.01^(year-2023)
```

**Previous (Old)**:
```
EPA VSL (2022) → Inflate to 2023 → Apply growth
11.0M USD2022 → 11.45M USD2023 → 11.45 × 1.01^(year-2024)
```

### Why Base Year Matters

The MSC data represents health damages in 2023 dollars. When projecting to future years, we need to account for VSL growth from 2023 onward, not from 2024 onward. The old approach effectively "lost" one year of growth by normalizing to 2024 instead of 2023.

### Why VSL Level Doesn't Matter (for this issue)

The MSC data already incorporates the VSL level (12.71M) into the $/ton values. Adjustment factors are multiplicative scalars representing growth over time, not absolute VSL levels. As long as the growth rate is correct (1% annually) and the base year matches the MSC data year (2023), the absolute VSL value used doesn't affect the calculation.

## References

- EPA VSL Guidance (2021): 11.3M USD2021
- EPA VSL Guidance (2024): 11.0M USD2022
- HHS VSL Guidelines: 1% annual real income growth rate
- CPI data: Bureau of Labor Statistics annual averages
- Consumer Product Safety Commission (CPSC) 2024 Notice of Availability of Final Guidance for Estimating Value per Statistical Life

---

**Date**: October 24, 2025
**Author**: Investigation prompted by user review of VSL consistency
**Status**: Corrected and ready for model re-runs
