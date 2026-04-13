# Copilot Debug Prompt — TARE Model: Trace Intermediate Calculations
## Reusable Template

---

## How to Submit This Prompt

1. Open the target module file in VS Code
2. Select its full contents with `Ctrl+A` — Copilot will use this as context
3. Open Copilot Chat (`Ctrl+Shift+I`) and optionally type `@workspace` first
   for broader project awareness
4. Paste the completed prompt below and press Enter
5. Use Copilot's **diff view** to accept or reject each change individually —
   do not bulk-accept without reviewing

**Work on a branch, never on `main`:**
```bash
git checkout main && git pull
git checkout -b debug/trace-[MODULE_NAME]
```

---

## ─── PROMPT START — COPY EVERYTHING BELOW THIS LINE ───

---

````

## Role

You are a Python debugging assistant helping a researcher trace intermediate
calculations in a scientific modeling codebase. Your job is to add temporary,
non-destructive debug print statements. You must not change any logic, data,
or control flow.

---

## Project Context

- **Project:** TARE Model (Tradeoff Analysis for Residential Energy) — Python, pandas, Jupyter
- **File:** [FULL FILE PATH]
  e.g. `cmu_tare_model/private_impact/calculate_lifetime_fuel_costs.py`
- **Target function(s):** [FUNCTION NAME(S)]
  e.g. `calculate_lifetime_fuel_costs`, `calculate_annual_fuel_costs`
- **Python version:** 3.11

---

## Building IDs to Trace

Add `DEBUG_IDS` as a new **optional keyword argument** with a default of `None`
to each target function. This is the only permitted change to a function signature.
All existing callers remain valid because the default is `None`.

```python
# New signature — add DEBUG_IDS=None after all existing parameters:
def function_name(..., existing_params..., DEBUG_IDS=None):
```

At the very top of the function body, immediately after any existing
docstring or empty-DataFrame guard, add this block:

```python
# Create a filtered debug view for the building IDs of interest
# [DEBUG - REMOVE BEFORE MERGING]
_df_debug = None
if DEBUG_IDS is not None and verbose:
    try:
        _df_debug = df_copy[df_copy.index.isin(DEBUG_IDS)].copy()
        print(f"\n[DEBUG] DEBUG_IDS mode active — tracing {len(_df_debug)} of "
              f"{len(df_copy)} homes: {DEBUG_IDS}")
    except Exception as _e:
        print(f"[DEBUG] Could not create _df_debug: {_e}")
```

> **Note on naming:** If the function uses `df` instead of `df_copy` as its
> primary working copy, replace `df_copy` with `df` in the block above.
> Use whichever name is consistent with the rest of the function body.

**When printing intermediate Series or DataFrames inside debug blocks, always
filter using `.isin(DEBUG_IDS)` rather than relying on `_df_debug`, because
many intermediate variables are newly created Series that share the same index
but are not derived from `df_copy` directly:**

```python
_debug_slice = some_series[some_series.index.isin(DEBUG_IDS)]
```

---

## Verbose Flag

The function already has a `verbose` parameter. **All** debug print statements
you add must be wrapped inside an `if verbose:` block. Do not add a new flag.

---

## Task: Add Debug Print Statements at Each Marked Step

The function may use structured comments like `# ===== STEP N: Description =====`
to mark each major stage. For every such comment, add a debug print block
**immediately after it** (not before, not elsewhere in the step) that:

1. Prints a separator header identifying the step
2. Prints the key intermediate variables at that point, filtered to `DEBUG_IDS`

The print block format is defined in the **Output Format** section below.

---

## Specific Variables to Print at Each Step

> **Instructions for filling in this section:**
> For each step or sub-step in the target function, list the variable names
> you want printed. Be explicit — Copilot will print exactly what you specify.
> If a variable is a Series or DataFrame, say so. If it is a scalar, say so.
> If filtering by DEBUG_IDS applies, say so. Remove these instructions when done.

[FILL IN — example structure:]

- **After STEP 0** (parameter validation):
    - `scenario_prefix` (string — print directly, no filtering needed)
    - `policy_scenario` (string)
    - `menu_mp` (int)

- **After STEP 1** (validation tracking, once per loop iteration):
    - Loop variable(s): `[e.g. category, lifetime]` (scalars)
    - `valid_mask` filtered to `DEBUG_IDS` — print value counts
      (`valid_mask[valid_mask.index.isin(DEBUG_IDS)].value_counts()`)

- **After STEP 2** (series initialization):
    - `[intermediate_series_name]` filtered to `DEBUG_IDS`

- **After STEP N** ([description]):
    - `[variable_name]` filtered to `DEBUG_IDS`

- **[Add as many steps as the function has]**

---

## Constraints

1. Do **NOT** modify any calculation logic, masking logic, or return values.
2. Do **NOT** add new imports.
3. You **may** add `DEBUG_IDS=None` as a new optional keyword argument to each
   target function. This is the only permitted signature change. All other
   parameters must remain unchanged and in their original order.
4. All debug print blocks must be inside `if verbose:` checks.
5. Add this comment on the line immediately above every debug block:
   `# [DEBUG - REMOVE BEFORE MERGING]`
6. If a variable might not exist at a given step (e.g., it is only assigned
   conditionally or later in the function), wrap the print in a `try/except`
   and print an explanatory note rather than crashing:
   ```python
   except Exception as _e:
       print(f"  [DEBUG] Could not print [variable_name]: {_e}")
   ```
7. Use the prefix `_debug_` for any temporary local variables you introduce
   inside debug blocks (e.g., `_debug_slice`, `_debug_cols`). This makes
   them easy to identify and remove.

---

## Output Format for Each Debug Block

Use this exact format for every block. Replace the bracketed placeholders:

```python
if verbose:
    # [DEBUG - REMOVE BEFORE MERGING]
    print(f"\n{'='*60}")
    print(f"[DEBUG] STEP [N] — [Description]  |  [loop_var]=[loop_var_value]")
    print(f"{'='*60}")
    try:
        _debug_slice = [variable_name][[variable_name].index.isin(DEBUG_IDS)]
        print(f"  [variable_name] (filtered to DEBUG_IDS):\n{_debug_slice.to_string()}")
    except Exception as _e:
        print(f"  [DEBUG] Could not print [variable_name]: {_e}")
```

**Guidance on the header context field** (`|  [loop_var]=[loop_var_value]`):

- Inside a `for category, lifetime in EQUIPMENT_SPECS.items()` loop:
  use `| category={category}, lifetime={lifetime}`
- Inside a `for year in range(...)` loop:
  use `| year={year_label}`
- Outside any loop (function-level step):
  omit the context field entirely

**For scalar variables** (strings, ints, floats), print them directly
without index filtering:

```python
if verbose:
    # [DEBUG - REMOVE BEFORE MERGING]
    print(f"\n{'='*60}")
    print(f"[DEBUG] STEP [N] — [Description]")
    print(f"{'='*60}")
    print(f"  [variable_name] = {[variable_name]}")
```

---

## Additional Tasks

[LIST ANY EXTRA TASKS HERE — remove this line and replace with specifics, e.g.:]

- Print `df_detailed.shape` after each `pd.concat` call to track how the
  detailed DataFrame grows
- Print the list of column names added to `category_columns_to_mask` at the
  end of each category loop iteration

````

---

## ─── PROMPT END ───

---

## After Running: How to Use the Output

Once Copilot has added the debug statements, run the full model sequentially
with `verbose=True` and your `DEBUG_IDS` list passed in:

```python
# In your scenario notebook, pass DEBUG_IDS when calling the function:
df_main, df_detailed = calculate_lifetime_fuel_costs(
    df=df,
    menu_mp=menu_mp,
    policy_scenario=policy_scenario,
    DEBUG_IDS=[BUILDING_ID_1, BUILDING_ID_2, ...]   # ← new argument
)
```

The debug output will print to the cell output as the function runs, letting
you trace each value from initialization through to the final result.

---

## Cleanup

When you are done tracing, remove all debug statements before merging.
The `# [DEBUG - REMOVE BEFORE MERGING]` marker makes them easy to find:

```bash
# Find every debug block across the project
grep -rn "DEBUG - REMOVE BEFORE MERGING" cmu_tare_model/

# Or discard the entire branch and all changes at once
git checkout -- cmu_tare_model/[PATH_TO_MODULE].py
```

You can also ask Copilot to clean up:
> *"Remove all code blocks marked with `# [DEBUG - REMOVE BEFORE MERGING]`
> and revert the `DEBUG_IDS=None` parameter addition from each function
> signature. Do not change anything else."*
