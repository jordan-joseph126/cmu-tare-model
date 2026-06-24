# CLAUDE.md — TARE Model / Joseph et al. 2026
# Heat-Pump Electrification Economics (ResStock 2022.1.1 / EUSS)
# Last updated: 23 June 2026 -- coding standards expanded; profile preferences integrated

> This file is read by Claude Code at the start of every session. It is the authoritative
> source of truth for project architecture, naming conventions, and permanent constraints.
> Session-specific prompts take precedence over this file when there is a conflict.

---

## Project at a Glance

**Research question:** Economics of heat-pump electrification across U.S. counties
**Data:** ~331,531 baseline homes | 331,526 applicable | 3,098 counties (ResStock 2022.1.1 EUSS)
**Heat-pump models:** MP3 (standard ASHP, 15 SEER1, 9 HSPF1) | MP4 (high-efficiency ASHP, 24–29.3 SEER1, 13–14 HSPF1)
**Policy scenario:** Single — `'2025 Reference Case'` (see Canonical Values below)
**Adoption metric:** `moreWTP >= 0` — economic payback only; no climate/health damages in the adoption decision

---

## Critical Rules — Read First

These apply to every session, every task, without exception.

### Files that must NEVER be edited

| File | Reason |
|---|---|
| `utils/validation_framework.py` | Core validation logic — never touch |
| Any `.ipynb` file | VSCode in-memory cache causes changes not to persist; backport manually |
| `fetch_aeo_data_and_project_EXPORT_14June2026.py` | EIA API scenario string must match API identifier — do NOT rename |
| TARE/EUSS load cells (preserved region) | Upstream data source — do not modify |
| Demand computation cells (preserved region) | Preserve original computation — do not modify |
| `utils/validation_framework.py` | Repeated for emphasis — never, ever touch |

### One-edit-per-stop-gate rule

Before applying any edit, show the researcher the exact diff (old -> new) with
3-5 lines of context above and below the change, and wait for explicit approval.
Only call the Edit tool after approval is given. Do not batch edits across files
or functions.

### Audit before every edit

Read the actual current file state before proposing any change. Do not assume what a previous session did. Previous sessions sometimes ended mid-task with unknown final state.

---

## Canonical Values (hard-coded knowledge)

```python
SCENARIO_STRING = '2025 Reference Case'   # exact string — must byte-match CSV policy_scenario column
COLUMN_PREFIX   = 'ref2025_mp{mp}_'       # always derived — never hardcoded as 'ref2025_mp3_'
ANCHOR_YEAR     = 2025                    # fuel prices and degree-day factors base year
LIFETIME_YEARS  = 15                      # NPV calculation horizon
```

**Do NOT use these strings in model code** — they are retired:
- `'AEO2026 Counterfactual Baseline'` (was renamed in Session 1 — model code only; fetch script keeps it)
- `'AEO2023 Reference Case'`
- `'No Inflation Reduction Act'`
- `preIRA`, `iraRef` as column prefixes

---

## Data Sources (current state as of Session 1)

| Dataset | File | Notes |
|---|---|---|
| Fuel prices | `eia_fuel_price_data_2025_usd2025.csv` | Already USD2025/kWh — no CPI deflation needed |
| Fuel price factors | `aeo2026_fuel_price_factors_2025_2050.csv` | 40 rows; all 2025 values = 1.0 |
| Degree-day factors | `aeo2026_degree_day_factors_2025_2050.csv` | 20 rows; year columns MUST be cast to int on read |
| ResStock source | ResStock 2022.1.1 EUSS | Do not update to ResStock 2025.1 |

**Degree-day read pattern (mandatory):**
```python
df = pd.read_csv(PATH)
df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]  # MUST cast to int
```
Skipping the int cast causes year lookups to silently return 1.0 (no projection applied).

**State key format:** Two-letter abbreviation (`'PA'`, `'TX'`), NOT full state name.
A wrong key returns silently as zero — no error, just wrong output.

---

## File Architecture

### Editable modules (Claude Code may edit these)

| File | Role |
|---|---|
| `cmu_tare_model/utils/degree_day_consumption_utils.py` | HDD + CDD-adjusted consumption; use this, not hdd_consumption_utils |
| `cmu_tare_model/private_impact/data_processing/create_lookup_fuel_prices.py` | Fuel price lookup |
| `cmu_tare_model/private_impact/calculate_lifetime_fuel_costs.py` | Lifetime fuel cost computation |
| `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py` | NPV computation |
| `cmu_tare_model/utils/modeling_params.py` | Scenario parameters |
| `cmu_tare_model/utils/calculation_utils.py` | Shared calculation helpers |
| `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | Data loading |
| `cmu_tare_model/constants.py` | EQUIPMENT_SPECS, VALID_CATEGORIES, REBATE_MAPPING |
| `determine_economic_adoption_potential.py` | Economic adoption framework (active) |
| `determine_adoption_potential_sensitivity.py` | Tiered adoption (DEPRECATED — header only, no logic changes) |
| `visualize_geospatial_data.py` | Choropleth / map rendering |
| `visuals_adoption_dotplot.py` | Economic dot plot |
| `calculate_postTARE_am_kpis_*_EXPORTED_*.py` | Main analysis notebook exports |

### Deprecated (do not import from; add header comment only)

| File | Status |
|---|---|
| `hdd_consumption_utils.py` | Superseded by `degree_day_consumption_utils.py` — does not handle cooling |
| `determine_adoption_potential_sensitivity.py` | Superseded by `determine_economic_adoption_potential.py` |

---

## Column Naming Conventions

**Always derive via helpers — never hardcode:**
```python
col_base = define_scenario_params(mp, policy)[0]   # → 'ref2025_mp3_'
mp_str   = f'mp{mp}'                               # '3' or '4' — never 'mp3' literal
```

**NPV cases (three per MP, as of Session 2):**
```
ref2025_mp{mp}_npv_heating_only
ref2025_mp{mp}_npv_heating_and_cooling_savings
ref2025_mp{mp}_npv_heating_and_cooling_full
```

**Economic adopter columns (three per MP, as of Session 2):**
```
ref2025_mp{mp}_heating_only_econ_adopter_moreWTP_v4MID_fixed_base
ref2025_mp{mp}_heating_and_cooling_savings_econ_adopter_moreWTP_v4MID_fixed_base
ref2025_mp{mp}_heating_and_cooling_full_econ_adopter_moreWTP_v4MID_fixed_base
```

**Canonical variable suffixes:** `v4MID` | `moreWTP` | `fixed_base` | `central`
Never use: `v3`, `lessWTP`, `iraRef_mp{mp}_`, `preIRA_mp{mp}_`, `aeo2026_mp{mp}_`

> **Note:** The `CODEBASE_MASTER_REFERENCE.md` documents older column naming with `preIRA`/`iraRef`
> prefixes and four columns per MP. That predates the Session 1 scenario consolidation.
> The naming above is current. If you see old-style column names in existing code, flag them —
> they are the old architecture to be replaced.

---

## Sensitivity Dimensions

| Dimension | Active values |
|---|---|
| RCM models (health damage) | `ap2`, `easiur`, `inmap` |
| Private discount rates | `fixed_low` (3%) \| `fixed_base` (7%) \| `fixed_high` (10%) \| `variable` (Ramsey) |
| Policy scenario | Single: `'2025 Reference Case'` — no IRA/pre-IRA split |
| NPV scope | `heating_only` \| `heating_and_cooling_savings` \| `heating_and_cooling_full` |

---

## Masking and Validation Rules

**Heating:** `include_heating = valid_fuel_heating AND valid_tech_heating`
- `valid_fuel_heating`: fuel ∈ {Electricity, Natural Gas, Propane, Fuel Oil}
- `valid_tech_heating`: technology ∈ `ALLOWED_TECHNOLOGIES['heating']`
- Homes with `in.heating_fuel = 'None'` are automatically excluded

**Cooling:** `include_cooling = valid_fuel_cooling AND valid_tech_cooling`
- `valid_fuel_cooling`: hardcoded True (cooling is always electric) — this flag is a no-op
- `valid_tech_cooling`: technology ∈ {Central AC, Room AC} — this is the ONLY cooling filter
- Homes with no AC (`'None'`) or evaporative coolers are excluded here

**Cooling in NPV:**
For homes where `include_cooling = False`: cooling savings = 0, cooling capital = 0.
The NPV ordering checks still hold: NPV2 = NPV1 and NPV3 = NPV1 for these homes.

**Existing-ASHP homes — RESOLVED: exclude.**
`'Electricity ASHP'` (and any variant) must NOT appear in `EQUIPMENT_SPECS` or
`ALLOWED_TECHNOLOGIES['heating']`. The study models fossil-fuel-to-ASHP transitions;
a home that already has an ASHP has no counterfactual fossil fuel system to replace.
If an existing-ASHP entry is found in `constants.py`, flag it and remove it.

---

## NPV Ordering Checks (enforce in verification)

Per home:
- `NPV2 ≥ NPV1` (Case 2 adds cooling savings ≥ 0)
- `NPV2 ≥ NPV3` (Case 3 adds cooling capital cost, reducing NPV)

Per county (means):
- Adoption rate Case 2 ≥ Case 1
- Adoption rate Case 2 ≥ Case 3

---

## Golden Values

These were established under the pre-Session-1 data (old fuel prices, old degree-day factors).
They will change when Session 2 NPV runs with new data. Never silently overwrite —
add a new row marked "supersedes" and keep the old row.

| Quantity | MP3 | MP4 | Data vintage | Session |
|---|---|---|---|---|
| Operating-cost % change, county median | −38.5% | −60.6% | Pre-AEO2026 | Round 3 |
| Total electricity demand change (GWh) | +427,043.7 | +30,618.4 | Pre-AEO2026 | Round 3 |
| Median demand % change | +22.5% | −8.1% | Pre-AEO2026 | Round 3 |
| Mean economic adoption rate (heating only) | 20.8% | 20.5% | Pre-AEO2026 | Round 3 |
| Operating-cost % symmetric norm | ±81.4% | (shared) | Pre-AEO2026 | Round 3 |
| Demand GWh symmetric norm | ±1038.3 GWh | (shared) | Pre-AEO2026 | Round 3 |

---

## Session Log (brief)

| Session | Date | Key outcomes |
|---|---|---|
| Round 1–2 | (dates TBD) | See SESSION_CHANGELOG.md |
| Round 3 | 10 Jun 2026 | moreWTP >= 0 adoption; econ adoption choropleth; subtitle convention locked; dict-title bug fixed |
| Session 1 | ~14 Jun 2026 | Scenario consolidated to `'2025 Reference Case'`; fuel prices + HDD/CDD rewired to AEO2026 CSVs; cooling re-enabled; baseline fuel costs verified on 331,531 homes |
| Session 2 | 16 Jun 2026 | NPV import fix; three NPV cases; tiered adoption deprecated; econ adopter columns for all three cases |
| Session 1c | 23 Jun 2026 | EIA fetch functions extracted to `eia_api_utils.py`; notebook has zero inline `def` statements |
| Session 1d | 23 Jun 2026 | PEP 8 cleanup: E221/E241 padding, E501 long lines, named API dicts, plain-language comments |

---

## Coding Standards

### Documentation and comments

- **Google-style docstrings** on all new functions -- include Args, Returns, and Raises sections.
- **Comments explain WHY, not what.** A comment that restates the code adds nothing. Explain the reason for a decision, the constraint it satisfies, or the non-obvious consequence.
- **Business logic and domain knowledge.** Any calculation or filter that depends on a research-specific decision (e.g. why a SEER threshold is set where it is, why an income group is excluded, what a specific AEO series ID represents) must have a comment explaining the rationale. Future readers will not have access to the methodology notes.
- **Multi-step processes.** For functions or cells with distinct phases, use labeled step comments so the structure is scannable without reading every line:
  ```python
  # Step 1 -- validate inputs
  # Step 2 -- fetch and tidy data
  # Step 3 -- compute factors and export
  ```
- **Assumptions.** When code assumes something about data shape, value ranges, or upstream processing, make it explicit in a comment:
  ```python
  # Assumes df_baseline has already been filtered to include_heating = True.
  ```
- **Plain language only.** Avoid technical jargon. Do not use terms like "invariant", "round-trips", or internal code-history references. Write as if the reader has never seen the git log.
- **No stale references.** If a comment names a function, file, or module, confirm it still exists before writing it.
- **Type hints** on all new function parameters and returns. For complex types, import from `typing`: `Optional`, `Union`, `Tuple`, `Dict`, `List`. Use `Optional[X]` rather than `X | None` for Python 3.9 compatibility.

### PEP 8 compliance

- **Line length: 88 characters maximum** (Black default). Wrap longer lines using implicit string concatenation, backslash continuation, or by extracting a named variable.
- **No alignment padding in assignments (E221).** Write `x = 1` not `x      = 1`. Extra spaces to align `=` signs across lines violate PEP 8 and create maintenance burden when a key is renamed.
- **No alignment padding in dicts (E241).** Write `"key": value` not `"key":    value`. Same rule as E221.
- **Validate inputs at the top of functions**, before any computation. Check types, ranges, and required columns with informative error messages that name both the expected and actual values.
- **Use specific exception types**: `ValueError` for invalid values, `TypeError` for wrong types, `KeyError` for missing keys. Avoid bare `Exception` or `RuntimeError`.
- **Fail fast, fail loud.** Let errors surface immediately where they originate; do not let a bad value propagate silently through several steps.
- **Graceful fallback where appropriate.** For data fetch operations that may fail for some states or regions, log a warning and continue with a national fallback rather than crashing the whole pipeline.
- **Float64** for all econ and adopter columns (0.0 / 1.0) — avoids pandas FutureWarning.
- **DEBUG = False** as default in `constants.py`; never ship with True.
- **ASCII characters only.** Do not use Unicode symbols in code, comments, or
  markdown cells. Use these ASCII equivalents instead:
  - Arrows: `-->` not `→`; `=>` not `⇒`
  - Em dash: `--` not `—`; en dash: `-` not `–`
  - Division: `/` not `÷`; multiplication: `x` not `×`
  - Check mark: `[OK]` not `✓`
  - Ellipsis: `...` not `…`
  - Box/rule separators: `-` repeated, not `─`

### Print statement conventions

Match the structure of the code to the structure of the output:

- **Independent output lines** → separate `print()` calls.
- **Single long status line** → implicit f-string concatenation (PEP 8 endorsed):
  ```python
  print(
      f"After tidy: {len(df)} rows | "
      f"fuels={sorted(df['fuel_type'].unique())} | "
      f"regions={df['region'].nunique()}"
  )
  ```
- **Formatted summary block** (PASS messages, multi-field summaries) → triple-quoted f-string with a backslash after the opening quotes to suppress the leading blank line:
  ```python
  print(f"""\
  [PASS] Fuel-price factors written
         Shape: {df.shape} | All {ANCHOR_YEAR} factors = 1.0""")
  ```

### API call parameter dicts

Never inline a multi-key parameter dict inside a function call. Define it as a named variable first, then unpack it. This keeps the call site to one readable line and makes the parameters independently inspectable.

```python
# Correct — parameters are named and scannable independently
aeo_params = {
    "facets[scenario][]": SCENARIO_ID,
    "frequency": "annual",
    "start": str(ANCHOR_YEAR),
}
rows = eia_get(f"aeo/{AEO_YEAR}/data", api_key=EIA_API_KEY, **aeo_params)
```

### Simplicity and naming

- Prefer named intermediate variables over complex inline expressions.
- Temporary DataFrames used only to derive the next step should have a descriptive name (`df_tidy`, `df_real`, `df_states`), not a generic name like `df`.

---

## Known Anti-Patterns

Do not suggest any of these:

```
❌ Import from hdd_consumption_utils — use degree_day_consumption_utils instead
❌ Use lessWTP or strict > 0 for adoption decision — always moreWTP >= 0
❌ Let climate/health damages enter the adoption decision
❌ Hardcode 'mp3', 'ref2025_mp3_', 'aeo2026_mp3_', 'iraRef_mp3_', or any scenario prefix
❌ Use old scenario strings: 'AEO2023 Reference Case', 'No Inflation Reduction Act', preIRA, iraRef, aeo2026_mp{mp}_
❌ Rename anything in fetch_aeo_data_and_project_EXPORT_14June2026.py
❌ Add 'Electricity ASHP' or any ASHP variant to EQUIPMENT_SPECS / ALLOWED_TECHNOLOGIES['heating'] — existing-ASHP homes are excluded by design
❌ Read degree-day CSV without int-casting year columns — silent flat 1.0 results
❌ Use full state name as price lookup key ('Pennsylvania') — must be abbreviation ('PA') — fails silently as zero
❌ Apply cooling savings to homes with include_cooling = False
❌ Collapse three NPV cases into one combined value
❌ Derive operating-cost % from ratio formula — always use (new - old) / old * 100 on per-home cols
❌ Route adoption share through pct_change — it is a share (0–100%), not a percent change
❌ Delete the tiered adoption module — prepend deprecation header only
❌ Generate econ adopter columns inside a loop — generate all per MP in a single block
❌ Edit .ipynb JSON directly — backport accepted changes manually
❌ Edit validation_framework.py — never
❌ Silently overwrite a golden value — keep old row with 'superseded by Session N' note
❌ Skip the pre-edit audit — read actual file state before every change
❌ Batch edits across files — one diff at a time
❌ Alignment padding in assignments (E221): `x      = 1` — write `x = 1`
❌ Alignment padding in dicts (E241): `"key":    value` — write `"key": value`
❌ Lines over 88 characters — wrap using implicit concatenation or named variables
❌ Inline multi-key dicts in function calls — define as a named dict before the call
❌ Jargon in comments — use plain language; no internal code-history references
❌ Stale function or module references in comments — confirm they exist before naming them
```
