# Local-Rebate Sensitivity -- Implementation Plan (plain-language outline)

**What we are building, in one sentence:** a new "plus local rebate" version of the model's
economics that stacks Colorado / Longmont / Boulder heat-pump rebates on top of the June 2026
federal rebate, so we can see how many more homes become economic adopters once local money is
counted.

This document is the plain language companion to the Claude Code session prompt. The session prompt
tells the agent *what to do and in what order*; this file explains *why*, and pins down the *rebate
conditional logic* in full (section 4) so the module can be built without reference to any external
source.

---

## 1. Overview

Each home already has a June 2026 NPV -- the lifetime savings minus the net cost of the heat pump
after the federal rebate. A local rebate lowers the capital cost, but never changes the energy
savings.

So:

```
NPV_plusLocal = NPV_sub_june2026 + local_adder
```

In other words, we do not need to recompute savings or net capital from scratch. We compute one new
dollar amount per home (the local adder from the local rebate) and shift the existing NPV by it. The
new adoption flag is then:

```
adopter_plusLocal = 1 if NPV_plusLocal >= 0 else 0
```

Incorporating the local rebates adds the `sub_june2026_plusLocal` case and increases the number of
cases from **9 to 12**. This is three NPV cost offsets (which avoided-replacement cost is credited
back) times four rebate options (`unsub`, `sub`, `sub_june2026`, and the new
`sub_june2026_plusLocal`). Only the June 2026 base gets a local variant, because local rebates stack
on the current/proposed federal program, not the 2024 one.

The three cost offsets are:

| Case name | What cost it credits back |
|---|---|
| `heatingSavings_coolingLCC` | the air conditioner the home did not have to replace |
| `heatingLCC_coolingSavings` | the heating system the home did not have to replace |
| `heatingLCC_coolingLCC` | both of the above |

All three count the same energy bill savings. They differ only in which piece of avoided equipment
cost is credited back. Every one of them gets its own local-rebate version, which is why three new
cases are added and not one.

The nine existing cases are not changed by any of this -- we are only adding three new ones
alongside them.

---

## 2. Avoiding double counting

Some of the Colorado rebates in this list are not really local money -- they are federal money that
Colorado hands out. The model already counts those federal dollars in the `sub_june2026` case. If we
also count them as local rebates, every one of those homes gets the same dollars twice and looks
more affordable than it is. The table below sorts which is which.

| Rebate | Funder | In `plusLocal`? | Why |
|---|---|---|---|
| State of CO Heat Pump Discount | CO state | **YES** | State tax-credit / upfront discount, separate pot from IRA. ~$333 net to customer. |
| Efficiency Works (all tiers) | PRPA / LPC utility | **YES** | Ratepayer-funded utility rebate; Longmont territory only. |
| EnergySmart YES | Boulder County | **YES\*** | County-funded; Boulder County + under 100% AMI only; combined cap <= 50% of project cost. |
| CO Home Efficiency Rebates (HER 1-4) | CO (IRA-funded) | **NO** | This *is* the federal HOMES program the model already scores in `sub_june2026`. Adding it double-counts. |
| CO Home Energy Rebate (HEAR) | CO (IRA-funded) | **NO** | Colorado pass-through of federal HEEHR/HEAR -- already in the model's HEEHR arm. Retired anyway. |

\* EnergySmart rarely fires for modeled homes: most ResStock retrofit homes sit above 100% AMI.
Include the column; expect mostly $0.

---

## 3. Home metadata columns required for the rebate conditional logic

The local module reads these straight from the model's in-memory household frame during a run
(the same `DATAFRAMES_BY_MP[mp]` frame the rest of the private-impact math uses) -- not from any
exported CSV.

| What we need | Household-frame column | Used for |
|---|---|---|
| Ducted or not | `hvac_has_ducts` | ducted -> per-unit rate; non-ducted -> per-ton rate |
| Baseline fuel | `base_heating_fuel` | electric-baseboard-replacement detection |
| Heating type | `heating_type` | electric-baseboard-replacement detection |
| System size | `size_heating_system_primary_k_btu_h` | tons = size / 12 |
| Income | `percent_AMI` | EnergySmart under-100%-AMI check |
| Utility territory | `utility_territory` (**may need creating -- see sec. 7**) | Efficiency Works, Longmont only |
| County | `county` | EnergySmart Boulder-County check |
| Project cost | `mp{mp}_heating_upgrade_installed_cost_v4MID` | EnergySmart combined cap, optional overall cap |
| Federal June 2026 rebate | `mp{mp}_heating_rebate_amount_june2026_v4MID` | FYI / optional overall cap |
| June 2026 NPV (per cost offset) | `ref2025_mp{mp}_<case>_sub_june2026_private_npv_fixed_base` | the base the adder shifts |

The measure package number (3 or 4) is not a column -- it is passed into the function as an
argument, the same way the existing rebate functions receive it.

These names are shown in full for readability only. In the code they are built with the naming
helpers (`create_cost_col`, `create_rebate_col`, `create_npv_case_col`), never typed out.

---

## 4. Rebate conditional logic (pseudocode)

It is written one home at a time for readability. The actual code module uses vectorized operations
for efficiency. Constants are put in the constants.py file so that they exist in a single spot and
can be easily updated.

### 4a. Constants (the editable levers)

```
# Efficiency Works rate table. basis "unit" -> flat per system; "ton" -> multiply by tons.
EFFICIENCY_WORKS_RATES = {
    "EW_Ducted_T1":           (basis="unit", rate=1000),   # ducted, standard   (proxy: MP3)
    "EW_Ducted_T2":           (basis="unit", rate=2000),   # ducted, high-eff   (proxy: MP4)
    "EW_NonDucted_Std":       (basis="ton",  rate=500),    # non-ducted, standard
    "EW_NonDucted_Baseboard": (basis="ton",  rate=1000),   # non-ducted, replacing electric baseboard
}

CO_HEATPUMP_DISCOUNT   = 333     # flat, net-to-customer (contractor keeps 66% of $1,000)
ENERGYSMART_MAX        = 2000    # Boulder County, under 100% AMI
COMBINED_CAP_FRAC      = 0.50    # local rebates combined may not exceed 50% of project cost

EFFICIENCY_WORKS_TERRITORY = "Longmont-EffWorks"   # the only territory Efficiency Works pays in
ENERGYSMART_COUNTY         = "Boulder"
```

### 4b. Per-home local adder

```
function local_rebate_adder(home):

    # --- Step 1: system size in tons (12 kBtu/h = 1 ton) ---
    system_tons = home.size_heating_system_primary_k_btu_h / 12

    # --- Step 2: is this a full electric-baseboard replacement? ---
    # true only when the existing system is electric AND its type name contains "baseboard".
    # Match case-insensitively so "Electric Baseboard" and "electric baseboard" both hit.
    is_baseboard = (lower(home.base_heating_fuel) == "electricity"
                    AND "baseboard" in lower(home.heating_type))

    # --- Step 3: pick the Efficiency Works tier ---
    if home.hvac_has_ducts:
        tier_key = "EW_Ducted_T2" if home.menu_mp == 4 else "EW_Ducted_T1"
    else:
        tier_key = "EW_NonDucted_Baseboard" if is_baseboard else "EW_NonDucted_Std"

    basis, rate = EFFICIENCY_WORKS_RATES[tier_key]

    # --- Step 4: Efficiency Works amount, only applies in Longmont territory ---
    # unit basis -> flat rate; ton basis -> rate per ton. Any other territory (e.g. Xcel) -> $0.
    if home.utility_territory == EFFICIENCY_WORKS_TERRITORY:
        ew_amount = rate if basis == "unit" else rate * system_tons
    else:
        ew_amount = 0

    # --- Step 5: CO state heat-pump discount (flat, applies to all CO homes) ---
    co_discount = CO_HEATPUMP_DISCOUNT

    # --- Step 6: EnergySmart, only for Boulder County homes under 100% AMI ---
    if home.county == ENERGYSMART_COUNTY AND home.percent_AMI < 100:
        energysmart_raw = ENERGYSMART_MAX
    else:
        energysmart_raw = 0

    # --- Step 7: EnergySmart combined cap ---
    # EnergySmart tops up only to the point where local rebates reach 50% of project cost.
    # room_left = 50% of project cost, minus what EW and the CO discount already used.
    room_left   = max(0, COMBINED_CAP_FRAC * home.project_cost - ew_amount - co_discount)
    energysmart = min(energysmart_raw, room_left)

    # --- Step 8: the local adder ---
    local_adder = ew_amount + co_discount + energysmart
    return local_adder
```

### 4c. From adder to NPV and adoption (per cost offset, per home)

```
NPV_plusLocal     = NPV_sub_june2026 + local_adder          # the equation in section 1
adopter_plusLocal = 1 if NPV_plusLocal >= 0 else 0
```

### 4d. Optional overall cap (OPEN DECISION -- see section 7)

If we decide federal + local may not exceed the installed cost on small systems, limit the *total*
rebate, then reduce the local amount to fit under that limit before shifting NPV:

```
total_rebate = min(project_cost, federal_june2026 + local_adder)
local_adder  = max(0, total_rebate - federal_june2026)   # local can only fill the gap up to cost
```

Leaving this off keeps the section-1 equation exact; turning it on makes it conditional on the cap
not binding. Recommendation: enforce it but log where it applies, so we can see if it ever mattered.

---

## 5. How it plugs into the model

The order matters because one file writes a column the next one reads.

1. **`constants.py`** -- add the rate table and levers from 4a, a `REBATE_GUIDANCE_LOCAL` naming
   token, and the three new case names to `NPV_CASE_CATEGORIES`, appended *after* the existing nine.

   *Why here first:* the export and adoption modules loop over `NPV_CASE_CATEGORIES`, so adding the
   names here is what makes those two pick up the new case automatically.

2. **New local-rebate module** (a new function kept *separate* from the income-based federal one) --
   implements section 4 and writes one new per-home column, the local rebate amount, named through the
   `create_rebate_col(..., guidance=REBATE_GUIDANCE_LOCAL)` helper function.

   *Why separate:* the federal path (`calculate_rebate_program`) is driven by income (HEEHR vs
   HOMES). The local rebate is driven by geography and system type -- a different shape.

3. **`calculate_lifetime_private_impact.py`** -- read the new local column (copy the existing
   `rebate_june2026_col` block, including its "if column exists" guard so a run with no local dollars
   safely gives back the plain June 2026 answer), compute the three:

   `net_capital_<case>_sub_june2026_plusLocal = net_capital_<case>_unsub - rebate_june2026 -
   rebate_local`

   Then add three entries to the hardcoded `npv_case_inputs` dict.

   **The constant and this dict must be extended together:** If we extend the constant but forget
   this dict, the export asks for three columns that were never written and crashes.

4. **Economic-adoption module** -- confirm it builds adopter flags by looping `NPV_CASE_CATEGORIES`;
   if so, the three new flags appear for free. If it hardcodes cases anywhere, extend it the same
   way.

5. **The household CSV export (`export_tepper_csv.py`) + its data dictionary**

   - This module is called inside the main notebook. Add the local *amount* column to the rebate
     group by hand, and bump the data dictionary from nine cases to twelve (and its "27 nine-case
     columns" check to 36).
   - Because this module builds its column lists by looping over the case names, the new NPV,
     capital-cost, and adopter columns appear on their own once step 1 is done. Only the local rebate
     dollar amount has to be added by hand.

---

## 6. Worked examples for the local rebate logic

These three homes exercise every branch and are the acceptance test for the module.

| Field | Home 1 | Home 2 | Home 3 |
|---|---|---|---|
| menu_mp | 4 | 3 | 4 |
| hvac_has_ducts | TRUE | FALSE | TRUE |
| base_heating_fuel | Electricity | Electricity | Natural Gas |
| heating_type | Electric Furnace | Electric Baseboard | Natural Gas Furnace |
| size (kBtu/h) | 24 | 18 | 30 |
| percent_AMI | 90 | 70 | 200 |
| utility_territory | Longmont-EffWorks | Longmont-EffWorks | Xcel |
| county | Boulder | Boulder | Boulder |
| project cost | 14,000 | 6,000 | 16,000 |
| NPV_sub_june2026 | -800 | 200 | -3,000 |
| **tier_key** | EW_Ducted_T2 | EW_NonDucted_Baseboard | EW_Ducted_T2 |
| **ew_amount** | 2,000 (unit) | 1,500 (1000 x 1.5 ton) | 0 (Xcel) |
| **co_discount** | 333 | 333 | 333 |
| **energysmart** | 2,000 | 1,167 (cap applies) | 0 (AMI check) |
| **LOCAL ADDER** | **4,333** | **3,000** | **333** |
| **NPV_plusLocal** | 3,533 | 3,200 | -2,667 |
| **outcome** | **FLIPPED to adopter** | adopter (already was) | non-adopter |

- Home 1 is a marginal Longmont home that **flips** on the local stack.
- Home 2 shows the per-ton rate and Boulder County's **50%-of-project combined cap**
  (EnergySmart raw $2,000 trimmed to $1,167, because 0.5 x 6,000 - 1,500 - 333 = 1,167).
- Home 3 (Xcel + gas + 200% AMI) gets **no** Efficiency Works and **no** EnergySmart, only the flat
  state discount, and stays out.

---

## 7. Open questions and known gaps

### Questions we need answered before building (Tamar please review)

1. **Overall cap for federal and local rebates (section 4d)**
   - Should we add the `MIN(project_cost, federal + local)` limit or not? This changes the equation
     in section 1.
   - Related: we have assumed a local rebate does not reduce or otherwise affect the federal rebate
     cap. Please confirm that is right, because the whole "just add the local adder" approach depends
     on it.
2. **Do federal rebate dollars count toward the EnergySmart 50% cap?**
   - As specified in Step 7, the cap currently only counts *local* rebates (EW + CO discount) against
     the 50%-of-cost limit. If the county intends the 50% cap to include federal dollars too, the
     formula in Step 7 must be updated. Confirm with the program.

### Known gaps, already planned for

1. **Utility-territory column**
   - The logic needs a per-home `utility_territory`. Confirm it exists in the loaded frame; if not,
     create it from `county` / `city` (Longmont -> Efficiency Works, most of the rest of Boulder
     County -> Xcel). This is a data-availability item for the audit.
2. **Xcel-territory amounts**
   - Xcel has its own ASHP rebates that are **not yet sourced**. For now Xcel homes earn $0
     Efficiency Works (only the flat CO discount). Leave a documented placeholder -- do not invent
     values.
3. **Geography of the run**
   - A CO/Boulder run (set the geographic filter and re-run) is needed before any of these local
     dollars actually fire.
   - The existing PA applicable-home denominators (78.4% heating / 82.7% cooling) must be recomputed
     for CO.

---

## 8. Build order checklist

- [ ] Task 1 -- Audit only: find `NPV_CASE_CATEGORIES` iterators, confirm the hardcoded
      `npv_case_inputs`, locate the adoption module, check the `utility_territory` column, list the
      open cap questions. No edits.
- [ ] Task 2 -- `constants.py`: rates, levers, `REBATE_GUIDANCE_LOCAL`, three new cases appended.
- [ ] Task 3 -- new local-rebate module (section 4); verify against the three test homes.
- [ ] Task 4 -- wire into private NPV; verify the nine existing cases produce exactly the same values
      as before, and that the equation holds.
- [ ] Task 5 -- confirm/extend adoption; 12 adopter flags.
- [ ] Task 6 -- export + data dictionary to twelve cases.
- [ ] Task 7 -- tests: three test homes, double-count exclusion, Longmont-only Efficiency Works,
      EnergySmart cap, a check that the nine existing cases produce the same numbers as before when
      no local rebates apply, and the section-1 equation. Run the suite.

Each step ends with a check and a stop, per the repo's audit-first / one-diff-per-gate rule.
