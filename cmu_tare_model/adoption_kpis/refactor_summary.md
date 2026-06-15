# Adoption KPIs Refactoring Summary

*Last updated: 2026-05-30*

---

## Objective

Replace the multi-tier, damage-inclusive adoption metric with a single, intuitive economic test:

> **A home is an economic adopter if its incremental private NPV (`moreWTP`) ≥ 0.**  
> The heat pump pays for its extra cost over the baseline from bill savings alone. Break-even counts as adoption. No climate/health damage value included.

Simplify the *decision logic* and make the result novice-readable. Keep all existing correctness machinery (validation framework, NaN masking, column naming conventions).

---

## Locked Parameters

| Parameter | Value |
|---|---|
| Cost scenario | `v4MID` |
| WTP variant | `moreWTP` |
| Discount rate | `fixed_base` |
| Discount rate column | `private_discount_rate_fixed_base` |
| Policy scenario (IRA) | `'AEO2023 Reference Case'` → prefix `iraRef_mp{mp}_` |
| Policy scenario (Pre-IRA) | `'No Inflation Reduction Act'` → prefix `preIRA_mp{mp}_` |
| HVAC scenario (primary) | `'heating'` (Case A) |

---

## Workflow

All notebook edits are made to the **EXPORT.py** file first. The user copy-pastes changed cells into the live notebook and shares kernel outputs for verification. This avoids a VS Code in-memory cache issue where direct `.ipynb` JSON edits do not persist.

**Key files:**
- `cmu_tare_model/adoption_potential/determine_economic_adoption_potential.py` — economic decision logic (Phase 1)
- `cmu_tare_model/adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings_EXPORT.py` — working copy for all notebook cell edits
- `cmu_tare_model/adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings.ipynb` — live notebook (user-managed)

---

## Non-Negotiable Rules

- Always use `>= 0` (not `> 0`) — break-even is adoption.
- Always use `moreWTP` / `v4MID` — never `lessWTP` / `v3`.
- Use `0.0` / `1.0` (not `False` / `True`) in float64 columns — avoids pandas FutureWarning.
- Derive all column prefixes via `define_scenario_params(mp, policy_scenario)[0]` — never hardcode `'preIRA_mp3_'` etc.
- Never edit cells 1–12 of the notebook (preserved region with golden values).
- Archive = prepend header comment + keep code; never delete.
- Never modify the validation framework (`utils/validation_framework.py`).

---

## Progress by Phase

### Phase 1 — Simplify `economic_adoption_decision` ✅ COMPLETE

**File:** `cmu_tare_model/adoption_potential/determine_economic_adoption_potential.py`

Changes made:
- Replaced multi-tier, damage-inclusive decision tree with a single `moreWTP >= 0` test.
- Removed `lessWTP` logic entirely.
- Fixed FutureWarning: column initialized as `float64`; valid non-adopters set to `0.0`, adopters to `1.0` (not bool).
- Excluded homes (outside valid mask) remain `NaN` as required by the validation framework.

Key code (lines ~138–156):
```python
df_new_columns[economic_adopter_col_name] = create_retrofit_only_series(df_copy, valid_mask)
df_new_columns.loc[valid_mask, economic_adopter_col_name] = 0.0

economic_adopter_mask = (
    valid_mask
    & df_copy[moreWTP_col].notna()
    & (df_copy[moreWTP_col] >= 0)
)
df_new_columns.loc[economic_adopter_mask, economic_adopter_col_name] = 1.0
```

---

### Phase 2 — Rewrite Notebook Adoption Section ✅ COMPLETE (gate confirmed)

**File:** EXPORT.py (cells 18, 20, 21 + archive headers)

Changes made:

**Cell 18** — replaced single-scenario call with a dual HVAC loop:
```python
for mp in selected_mps:
    for hvac_scenario in ['heating', 'heating_and_cooling']:
        df_econ = economic_adoption_decision(...)
        # writes new columns directly into DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']
```

**Cell 20** — updated to read econ adopter column from `inmap` using `define_scenario_params` prefix.

**Cell 21 (choropleth)** — updated title and colorbar label:
- Title: `'Economic Adoption Potential — MP{mp}\n(Incremental Cost Recovered, IRA-Ref)'`
- Colorbar: `'Economic Adopters — Incremental Cost Recovered (%)'`

**Archived cells** — prepended archive header to:
- Multi-tier dot plot (superseded by Phase 3)
- NPV histogram (damage-inclusive, superseded)

**Gate confirmed via kernel output:**
- ~20% mean economic adoption nationally
- 3,098/3,098 counties mapped
- No FutureWarning
- Correct choropleth titles/labels

---

### Phase 3 — Economic Adopter Dot Plot ✅ CODE WRITTEN, awaiting gate check

**File:** EXPORT.py (two new cells inserted after archived dotplot, before diagnostic cells)

What was added:
- **Markdown cell** — explains the plot (single economic test, open circle = Pre-IRA, filled = IRA-Ref).
- **Python cell** — self-contained dot plot (does not reuse the multi-tier `plot_adoption_panel` infrastructure):
  - Generates `preIRA_*` econ adopter columns on demand using `'No Inflation Reduction Act'` policy (idempotent).
  - Groups homes by `base_heating_fuel` × `lmi_or_mui` matching `GROUPING_ORDER`.
  - Prints text summary table (preIRA % / IRA-Ref % / delta in pp).
  - Plots open circle (Pre-IRA) + filled circle (IRA-Ref) + connector line per row.
  - Annotation above each IRA-Ref dot: `X% (+Ypp)`.
  - Saves to `figures/figure_econ_adoption_dotplot_mp{mp}_{location_id}.{png,pdf}`.

**Column names generated:**
```
preIRA_mp{mp}_heating_econ_adopter_moreWTP_v4MID_fixed_base
iraRef_mp{mp}_heating_econ_adopter_moreWTP_v4MID_fixed_base
```

**Gate check criteria (run cell and share output):**
1. `preIRA_*` columns generated without error.
2. Text summary prints without NaN values.
3. IRA-Ref adoption ≥ Pre-IRA adoption for most groups (rebates shift right).
4. National % matches choropleth mean (~20%).
5. No FutureWarning.
6. Files saved to `figures/`.

---

## Remaining Work

| Item | Status |
|---|---|
| Phase 3 gate check — run Phase 3 cell, share output | ❌ Pending |
| Confirm Pre-IRA vs IRA-Ref shift is directionally correct | ❌ Pending (after gate) |
| Copy Phase 3 cells from EXPORT.py into live notebook | ❌ Pending (after gate) |

---

## Key Column Naming Reference

```
{policy_prefix}heating_econ_adopter_moreWTP_v4MID_fixed_base
{policy_prefix}heating_and_cooling_econ_adopter_moreWTP_v4MID_fixed_base
```

Where `policy_prefix = define_scenario_params(mp, policy_scenario)[0]`:
- IRA-Ref → `iraRef_mp3_` / `iraRef_mp4_`
- Pre-IRA → `preIRA_mp3_` / `preIRA_mp4_`

Columns are written into `DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']` in-place.

---

## Earlier Work (Pre-Refactor Sessions)

*(Retained for context — superseded by the phases above)*

- Initial implementation of `economic_adoption_decision` flagging `moreWTP > 0` (strict, since relaxed to `>= 0`).
- Equivalence diagnostic between econ adopter signal and legacy Tier 1 + Tier 2 adoption tiers.
- Identified `lessWTP == -0.0` boundary edge case in MP4 (no longer relevant — `lessWTP` path removed).
- Confirmed `compute_adoption_rate(..., adopter_tiers=[True])` as the correct aggregation path for boolean econ columns (no longer used — econ adopter column is now `float64` 0.0/1.0, aggregated via `.mean()`).
