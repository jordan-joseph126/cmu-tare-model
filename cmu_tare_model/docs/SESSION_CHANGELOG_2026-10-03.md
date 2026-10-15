# Session Changelog -- 2026-10-03

## Session 1: sample rule, ResStock savings check, demand breakdown

> Branch `resstock2025-dual-fuel-codebase-update`. ResStock 2022.1.1 (MP3, MP4).
> Four commits. The study sample is now **221,205 rdu (53,560,591 homes)**, down
> from 221,301 rdu, so **every modeled value moves** and no reference value in
> `REFERENCE_VALUES.md` matches until the next model run is recorded. The model
> has not been re-run yet; that run opens Session 2.
>
> A ResStock row is a representative dwelling unit (rdu), not a home. One rdu
> stands for 242.131013 homes. Every count below is labeled.

---

## 1. Why this session existed

The researcher set one rule on 2 Oct 2026:

- The study sample is every occupied, single-family home with an existing
  heating and cooling system TARE can assign a replacement cost, where ResStock
  applied the retrofit.
- Sample homes keep ResStock's energy values as published.
- A zero is never turned into a blank.
- Checks stop the run. They do not just print a warning.

The code broke that rule in three places, had no check of its energy savings
against ResStock's own numbers, and reported demand only as whole-home totals.

| Problem at the start | Fixed in |
|---|---|
| 96 rdu with a shared cooling system were in the sample, with no replacement cost | Commit 1 |
| A true zero in the baseline heating or cooling total was turned into a blank | Commit 2 |
| Retrofit energy was rounded to 2 decimals; baseline energy was not | Commit 2 |
| Nothing compared TARE's savings with ResStock's own savings columns | Commit 3 |
| County "site energy" columns were a copy of the electricity change | Commit 3 |
| Demand could not be split into heating, cooling, and hot water | Commit 4 |

---

## 2. The four commits at a glance

| # | Commit | What it does | Moves modeled values? |
|---|---|---|---|
| 1 | `6d82f3b` | Shared-cooling homes leave the study sample | Yes, every value |
| 2 | `cf43350` | True zeros kept; retrofit energy no longer rounded | Yes, by at most 0.023 kWh per home |
| 3 | `a70a80a` | Savings check runs in every model run | County site-energy columns only (see section 5.7) |
| 4 | `76eedf1` | Demand read from savings columns; demand breakdown | No |

The plan had five commits. The true-zero change and the rounding change were
committed together in `cf43350`, whose message describes only the rounding.

Two commit subjects do not tell the whole story:

- `cf43350` also contains the true-zero change (section 4.1).
- `a70a80a` says "No modeled value moves", but it also carries an earlier
  change to `demand.py` that makes the county site-energy columns real all-fuel
  numbers. Those two columns moved in that commit (section 5.7).

---

## 3. Commit 1 (`6d82f3b`) -- shared-cooling homes leave the sample

### Plain-language summary

Some homes get their cooling from a system shared with other homes. ResStock
files them as "Central AC", so TARE's list of allowed cooling types let them in.
TARE has no cost data for replacing a shared system, so their cooling
replacement cost was blank and the NPV gave them a $0 credit. Shared heating
was already left out for the same reason. This commit leaves shared cooling out
too.

- 96 rdu (23,245 homes) leave the sample, all of them typed `Central AC`.
- The sample goes from 221,301 rdu to 221,205 rdu (53,560,591 homes).
- The label is `Shared Cooling` in `in.hvac_cooling_efficiency`, spelled the
  same in 2022.1.1 and 2025.1.

Files: `constants.py`, `utils/calculation_utils.py`,
`energy_consumption_and_metadata/process_euss_data.py`,
`energy_consumption_and_metadata/study_sample.py`, and two test files.

### Step 3.1 -- name the label (`constants.py`)

The label gets a constant, placed right after `ALLOWED_TECHNOLOGIES`. It cannot
go inside that dictionary because the dictionary lists cooling types, and
shared cooling is a cooling efficiency label, not a type.

Before (heating comment, last line):

```python
    # 2022.1.1 result: 260,211 rdu (63.0M homes) in the study sample.
```

After:

```python
    # 2022.1.1 result: 260,211 rdu (63.0M homes) pass this heating rule; the
    # cooling rule below brings the study sample to 221,205 rdu (53.56M homes).
```

Before (cooling comment):

```python
    # because the heat pump would add cooling it never had (CLAUDE.md
    # Limitation 11).
    # PLACEHOLDER -- a future session may bring no-AC homes back, modeling the
```

After:

```python
    # because the heat pump would add cooling it never had (CLAUDE.md
    # Limitation 11).
    # Shared cooling is left out too (SHARED_COOLING_EFFICIENCY, below): ResStock
    # files it under 'Central AC', so this list alone does not catch it.
    # PLACEHOLDER -- a future session may bring no-AC homes back, modeling the
```

New, after the closing brace of `ALLOWED_TECHNOLOGIES`:

```python
# in.hvac_cooling_efficiency label for a cooling system shared between homes.
# Left out of the study sample: no cost data for its replacement, as with
# shared heating (96 rdu, 23,245 homes in 2022.1.1; all typed 'Central AC').
SHARED_COOLING_EFFICIENCY = 'Shared Cooling'
```

### Step 3.2 -- apply the rule (`calculation_utils.py`, `identify_valid_homes`)

This is the one line that changes results. It turns `valid_tech_cooling` off
for shared cooling. `include_cooling` and `include_sample` are built from that
flag, so they follow with no formula change.

Before (import):

```python
from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING, ALLOWED_TECHNOLOGIES, VERBOSE
```

After:

```python
from cmu_tare_model.constants import (
    EQUIPMENT_SPECS,
    FUEL_MAPPING,
    ALLOWED_TECHNOLOGIES,
    SHARED_COOLING_EFFICIENCY,
    VERBOSE,
)
```

Before (inside `identify_valid_homes`):

```python
                # Check if the technology type is in the allowed list
                df[tech_flag] = df[tech_col].isin(ALLOWED_TECHNOLOGIES[category])
```

After:

```python
                # Check if the technology type is in the allowed list
                df[tech_flag] = df[tech_col].isin(ALLOWED_TECHNOLOGIES[category])
                # Shared cooling is typed 'Central AC' but has no replacement
                # cost, so it is not a cooling system the study can replace.
                if category == 'cooling':
                    df[tech_flag] &= (
                        df['base_cooling_efficiency'] != SHARED_COOLING_EFFICIENCY)
```

The docstring gained a `Raises` entry: a frame with `cooling_type` but no
`base_cooling_efficiency` raises `KeyError`.

### Step 3.3 -- update the sample comment (`process_euss_data.py`, STEP 4b)

Comment only. The formula below it is unchanged.

Before:

```python
    # The one definition of the study sample: a heating system the study can
    # replace and cost, an existing central or room AC (include_cooling), and
    # applied in every package in the run. Set here once;
    # get_valid_calculation_mask requires it, so every result is limited to it.
    # Homes without AC are left out: the heat pump would add cooling they never
    # had (CLAUDE.md Limitation 11).
```

After:

```python
    # The one definition of the study sample: a heating system the study can
    # replace and cost, a central or room AC of the home's own
    # (include_cooling), and applied in every package in the run. Set here once;
    # get_valid_calculation_mask requires it, so every result is limited to it.
    # Homes without AC are left out: the heat pump would add cooling they never
    # had (CLAUDE.md Limitation 11). Shared cooling is left out: it has no
    # replacement cost (SHARED_COOLING_EFFICIENCY in constants.py).
```

### Step 3.4 -- give shared cooling its own funnel row (`study_sample.py`)

The SI funnel table gains a last row, `no_shared_cooling`. The existing
`central_or_room_ac` row keeps its meaning and its count because it now reads
the cooling type itself.

Before (module docstring):

```
measure package in the run to it, and if it has central or room AC
(include_cooling = True). df_enduse_refactored sets this once as the
```

After:

```
measure package in the run to it, and if it has a central or room AC of its
own, not a shared cooling system (include_cooling = True).
df_enduse_refactored sets this once as the
```

New import: `from cmu_tare_model.constants import ALLOWED_TECHNOLOGIES`.

Before (`build_sample_funnel`, Step 2):

```python
    replaceable_heating = df_in_scope['include_heating'].astype(bool)
    has_ac = df_in_scope['include_cooling'].astype(bool)
    scope_steps = [
        ('heating_fuel', valid_fuel),
        ('no_existing_heat_pump', valid_fuel & no_heat_pump),
        ('replaceable_heating_system', replaceable_heating),
        ('central_or_room_ac', replaceable_heating & has_ac),
    ]
```

After:

```python
    replaceable_heating = df_in_scope['include_heating'].astype(bool)
    # include_cooling also leaves out shared cooling, so the AC step reads the
    # cooling type itself and shared cooling gets its own row.
    has_ac = df_in_scope['cooling_type'].isin(ALLOWED_TECHNOLOGIES['cooling'])
    own_cooling_system = df_in_scope['include_cooling'].astype(bool)
    scope_steps = [
        ('heating_fuel', valid_fuel),
        ('no_existing_heat_pump', valid_fuel & no_heat_pump),
        ('replaceable_heating_system', replaceable_heating),
        ('central_or_room_ac', replaceable_heating & has_ac),
        ('no_shared_cooling', replaceable_heating & own_cooling_system),
    ]
```

The function's inputs and outputs keep their shape. It still raises if the last
row does not equal the `include_sample` count.

### Step 3.5 -- funnel test (`test_study_sample.py`)

The fixture gains a `cooling_type` column and a seventh home with shared
cooling.

Before:

```python
        'bldg_id': [1, 2, 3, 4, 5, 6],
        'weight': [2.0] * 6,
        ...
        'valid_fuel_heating': [True, False, True, True, True, True],
        'include_heating': [True, False, False, False, True, True],
        'include_sample': [True, False, False, False, False, False],
        'include_cooling': [True, False, True, False, False, True],
    }).set_index('bldg_id')
    applicable_bldg_ids = [pd.Index([1, 2, 3, 4, 5])]
    df_package = pd.DataFrame({
        'stage': ['load', 'applicability', 'housing_type'],
        'rdu_count': [9, 8, 5],
        'weighted_count': [18.0, 16.0, 10.0],
    })
```

After:

```python
        'bldg_id': [1, 2, 3, 4, 5, 6, 7],
        'weight': [2.0] * 7,
        ...
        'valid_fuel_heating': [True, False, True, True, True, True, True],
        'include_heating': [True, False, False, False, True, True, True],
        'include_sample': [True, False, False, False, False, False, False],
        # Home 7 is a Central AC by type, but shared, so include_cooling is False.
        'cooling_type': ['Central AC', 'None', 'Room AC', 'None', 'None',
                         'Central AC', 'Central AC'],
        'include_cooling': [True, False, True, False, False, True, False],
    }).set_index('bldg_id')
    applicable_bldg_ids = [pd.Index([1, 2, 3, 4, 5, 7])]
    df_package = pd.DataFrame({
        'stage': ['load', 'applicability', 'housing_type'],
        'rdu_count': [9, 8, 6],
        'weighted_count': [18.0, 16.0, 12.0],
    })
```

Before (assertions):

```python
        'central_or_room_ac']
    assert list(df_funnel['rdu_count']) == [9, 8, 5, 4, 3, 2, 1]
    assert list(df_funnel['removed_rdu'].iloc[1:]) == [1, 3, 1, 1, 1, 1]
```

After:

```python
        'central_or_room_ac', 'no_shared_cooling']
    assert list(df_funnel['rdu_count']) == [9, 8, 6, 5, 4, 3, 2, 1]
    assert list(df_funnel['removed_rdu'].iloc[1:]) == [1, 2, 1, 1, 1, 1, 1]
```

### Step 3.6 -- test of the rule itself (`test_calculation_utils.py`)

New test, no "before":

```python
def test_identify_valid_homes_leaves_out_shared_cooling(monkeypatch):
    """A Central AC that is a shared system is not a valid cooling technology."""
    from cmu_tare_model.utils.calculation_utils import identify_valid_homes

    # The file's shared setup has no cooling category; this test needs only that.
    monkeypatch.setattr(
        'cmu_tare_model.utils.calculation_utils.EQUIPMENT_SPECS', {'cooling': 15})
    df_homes = pd.DataFrame({
        'base_cooling_fuel': ['Electricity'] * 3,
        'cooling_type': ['Central AC', 'Central AC', 'None'],
        'base_cooling_efficiency': ['AC, SEER 13', 'Shared Cooling', 'None'],
    })
    result = identify_valid_homes(df_homes.copy(), verbose=False)
    assert list(result['valid_tech_cooling']) == [True, False, False]
    assert list(result['include_cooling']) == [True, False, False]

    with pytest.raises(KeyError, match='base_cooling_efficiency'):
        identify_valid_homes(
            df_homes.drop(columns=['base_cooling_efficiency']), verbose=False)
```

---

## 4. Commit 2 (`cf43350`) -- true zeros kept, retrofit energy not rounded

### Plain-language summary

Two small changes so that sample homes keep ResStock's values as published.
Both are in `energy_consumption_and_metadata/process_euss_data.py`.

- A home that ResStock reports with zero heating or cooling energy used to show
  a blank in its baseline total. It now shows 0.
- Retrofit energy was rounded to 2 decimals when read. Baseline energy never
  was. Retrofit energy is now read as published.

### Step 4.1 -- keep true zeros (`df_enduse_refactored`, STEP 3)

Before:

```python
        df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)
```

After:

```python
        # A true zero stays a zero; STEP 5 blanks homes outside the sample.
        df_enduse[f'baseline_{category}_consumption'] = total_consumption
```

- Affects `baseline_heating_consumption` (1,020 sample rdu) and
  `baseline_cooling_consumption` (7 sample rdu).
- No calculation reads these two columns. They ship in the Tepper household
  CSV, so two reporting columns change and no modeled value moves.

### Step 4.2 -- stop rounding retrofit energy (`df_enduse_compare`, STEP 3 and 3a)

Eight reads lost their `.round(2)`. They were the only `.round(2)` calls in the
file.

| Read | Runs today (MP3, MP4)? |
|---|---|
| Heating, standard branch | Yes |
| Cooling | Yes |
| Fans, pumps, and heat-pump backup (STEP 3a) | Yes |
| Heating, MP9 branch | No |
| Heating, MP10 branch | No |
| Water heating | No |
| Clothes drying | No |
| Cooking | No |

Before (the three reads that run today):

```python
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp[resstock_col(release, 'heating_electricity')].round(2)
            ...
            df_compare[f'mp{menu_mp}_cooling_consumption'] = df_mp[resstock_col(release, 'cooling_electricity')].round(2)
            ...
            df_compare[col] = df_mp[resstock_column].round(2)
```

After:

```python
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp[resstock_col(release, 'heating_electricity')]
            ...
            df_compare[f'mp{menu_mp}_cooling_consumption'] = df_mp[resstock_col(release, 'cooling_electricity')]
            ...
            df_compare[col] = df_mp[resstock_column]
```

The five inactive reads changed the same way.

- Value-moving. The largest change is 0.005 kWh in one part and 0.023 kWh in a
  home's heating plus cooling total.
- Retrofit energy feeds fuel costs, emissions, and the HOMES savings fraction,
  so those move with it.

Note for the record: while applying this change, the edit tool also removed the
blank line after each of the eight lines. That was noticed from the diff and
the blank lines were put back before the commit. The commit contains only the
eight one-for-one swaps.

---

## 5. Commit 3 (`a70a80a`) -- the savings check

### Plain-language summary

Every model run now compares TARE's heating and cooling energy savings with
ResStock's own published savings, for every study-sample home and every fuel,
and stops if they disagree. There are two tests.

- **Test (a), part by part.** ResStock publishes savings for each heating and
  cooling part (the main system, fans and pumps, heat-pump backup). Added up,
  they must equal TARE's savings within 1 kWh. This proves TARE reads the right
  columns for the right homes with no rounding, blanking, or sign error.
- **Test (b), whole home.** ResStock's whole-home savings for the fuel, less
  the end uses ResStock changes on its own (hot water, solar hot water,
  refrigerator), must match TARE's savings within 1% of the home's baseline use
  of that fuel. This proves TARE's list of parts is complete. It is the test
  that would have caught the backup heat and fans that TARE left out before
  September 2026.

Neither test says anything about later-year projections, prices, costs, or NPV.
They cover base-year (2025) energy only.

Files: `utils/calculation_utils.py`, `docs/resstock_2025_1_column_map.csv`,
`energy_consumption_and_metadata/process_euss_data.py`, two test files, and
(riding along) `adoption_kpis/data_loading.py` and `adoption_kpis/demand.py`.

### Step 5.1 -- name ResStock's savings columns (`calculation_utils.py`)

New function. It was written before this session and committed here.

```python
def get_resstock_savings_column(energy_column: str) -> str:
    """Returns the name of ResStock's own savings column for an energy column.

    Each upgrade file publishes baseline minus upgrade for every energy column
    (positive = use fell). 2022.1.1 adds '.savings' to the energy column name
    ('...energy_consumption.kwh.savings'); 2025.1 renames it instead
    ('...energy_savings..kwh').
    ...
    """
    # 2025.1 first: its '..kwh' ending is the more specific of the two.
    if energy_column.endswith('.energy_consumption..kwh'):
        return energy_column.replace(
            '.energy_consumption..kwh', '.energy_savings..kwh')
    if energy_column.endswith('.energy_consumption.kwh'):
        return f'{energy_column}.savings'
    raise ValueError(
        f"{energy_column!r} is not a ResStock energy consumption column")
```

### Step 5.2 -- three column-map rows (`resstock_2025_1_column_map.csv`)

New rows, so the check can look up each fuel's whole-home total by a
release-neutral name: `natural_gas_total`, `propane_total`, `fuel_oil_total`.
(`electricity_total` already existed.)

```
Whole-home energy,natural_gas_total,out.natural_gas.total.energy_consumption.kwh,out.natural_gas.total.energy_consumption..kwh,kWh,double,renamed,Renamed. Whole-home natural gas use. Read from the upgrade file only for the heating-and-cooling vs whole-home check.
Whole-home energy,propane_total,out.propane.total.energy_consumption.kwh,out.propane.total.energy_consumption..kwh,kWh,double,renamed,Renamed. Whole-home propane use. Read from the upgrade file only for the heating-and-cooling vs whole-home check.
Whole-home energy,fuel_oil_total,out.fuel_oil.total.energy_consumption.kwh,out.fuel_oil.total.energy_consumption..kwh,kWh,double,renamed,Renamed. Whole-home fuel oil use. Read from the upgrade file only for the heating-and-cooling vs whole-home check.
```

### Step 5.3 -- the limits and the check (`process_euss_data.py`)

Two names were added to the import from `calculation_utils`:
`RESSTOCK_FUEL_NAMES` and `get_resstock_savings_column`.

New, placed between `df_enduse_refactored` and `df_enduse_compare`:

```python
# Test (a) limit. Both sides add up the same ResStock numbers, so 1 kWh is far
# above what number storage can explain and far below a part left out.
SAVINGS_MATCH_TOLERANCE_KWH = 1.0

# Share of a home's baseline use of a fuel by which ResStock's whole-home
# savings, less the end uses below, may differ from TARE's heating + cooling
# savings. Every occupied home in both releases is under 0.3%.
WHOLE_HOME_SAVINGS_TOLERANCE = 0.01

# End uses outside heating and cooling that ResStock itself changes when the
# equipment changes (2025.1 refrigerators by up to 1,100 kWh). Test (b) takes
# their savings out of the whole-home savings first.
INTERACTING_END_USES = ('hot_water', 'hot_water_solar_th', 'refrigerator')


def check_savings_against_resstock(
    tare_savings_by_fuel: Dict[str, pd.Series],
    df_mp: pd.DataFrame,
    sample_bldg_ids: pd.Index,
    menu_mp: int,
    release: str = RESSTOCK_RELEASE_THIS_RUN,
) -> pd.DataFrame:
    """Checks TARE's heating + cooling savings against ResStock's savings columns.
    ...
    """
    # Step 1 -- sample rows, and every heating and cooling part ResStock reports
    df_sample_mp = df_mp.loc[sample_bldg_ids]
    part_columns = [
        (fuel, resstock_column)
        for category in ('heating', 'cooling')
        for fuel, _, resstock_column in find_enduse_columns(df_mp.columns, category)]

    # Step 2 -- both tests, one fuel at a time
    rows = []
    for resstock_fuel, fuel in RESSTOCK_FUEL_NAMES.items():
        tare_savings = tare_savings_by_fuel[fuel].loc[sample_bldg_ids]

        # Test (a): ResStock's savings for this fuel's heating and cooling parts
        part_savings_columns = [
            get_resstock_savings_column(resstock_column)
            for part_fuel, resstock_column in part_columns if part_fuel == fuel]
        part_gap = (
            tare_savings - df_sample_mp[part_savings_columns].sum(axis=1, skipna=False)
        ).abs()

        # Test (b): whole home less the end uses ResStock changes on its own.
        # Baseline use is the upgrade total plus its savings, so the upgrade
        # file alone is enough. Not every fuel has every end use (there is no
        # gas refrigerator), so only the columns in the file are read.
        total_column = resstock_col(release, f'{resstock_fuel}_total')
        total_savings_column = get_resstock_savings_column(total_column)
        total_savings = df_sample_mp[total_savings_column]
        interaction_columns = [
            total_savings_column.replace('.total.', f'.{end_use}.')
            for end_use in INTERACTING_END_USES]
        interaction_savings = df_sample_mp[[
            column for column in interaction_columns if column in df_mp.columns
        ]].sum(axis=1, skipna=False)
        baseline_use = df_sample_mp[total_column] + total_savings
        left_over = (total_savings - interaction_savings - tare_savings).abs()

        # Written as "not <=" so a blank counts as a failure.
        fails_parts = ~(part_gap <= SAVINGS_MATCH_TOLERANCE_KWH)
        fails_whole_home = ~(left_over <= WHOLE_HOME_SAVINGS_TOLERANCE * baseline_use)
        n_failed = int((fails_parts | fails_whole_home).sum())
        rows.append({
            'fuel': fuel,
            'rdu': len(sample_bldg_ids),
            'largest_part_gap_kwh': part_gap.max(),
            'largest_left_over_kwh': left_over.max(),
            'largest_left_over_pct': (100 * left_over / baseline_use).max(),
            'rdu_failed': n_failed,
            'status': '[OK]' if n_failed == 0 else '[WARNING]',
        })

    # Step 3 -- print the table, then stop the run if any home failed
    df_check = pd.DataFrame(rows)
    print(f"\nMP{menu_mp} heating + cooling savings vs ResStock's savings columns:")
    print(df_check.to_string(index=False))
    df_failed = df_check[df_check['rdu_failed'] > 0]
    if not df_failed.empty:
        raise ValueError(
            f"MP{menu_mp} heating + cooling savings do not match ResStock's "
            "savings columns for "
            f"{dict(zip(df_failed['fuel'], df_failed['rdu_failed']))} (fuel: rdu)")
    return df_check
```

Three properties worth knowing:

- A home that does not use a fuel must show exactly zero, because 1% of zero
  use is zero. There is no special case in the code for it.
- A blank fails. It is not counted as zero.
- If a future release renames a hot-water or refrigerator column, nothing is
  set aside and the check fails. That is the safe direction.

### Step 5.4 -- how the limits were decided

The check first went in with stricter limits and without the refrigerator. The
researcher asked for looser limits, because other users may change the filter
settings. Testing that on every home the packages apply to, in both releases,
with every filter open, gave this:

| What was tested | Result |
|---|---|
| Test (a), any home, either release | Never off by more than 0.00000000006 kWh |
| Test (b) at 1%, 2022.1.1, all 548,260 applicable rdu | 2 rdu fail, both vacant |
| Test (b) at 1%, 2025.1, hot water set aside only | 179 sample rdu fail for the dual-fuel package; the run would stop |
| Test (b) at 1%, 2025.1, refrigerator also set aside | 0 sample rdu fail; 6 vacant rdu fail with every filter open |
| Any occupied home, either release | Largest share is 0.27% (2022.1.1) and 0.07% (2025.1) |

Decisions:

- **Test (a) limit: 0.000001 kWh became 1 kWh.** The old limit was needlessly
  tight. The largest harmless difference ever seen was 0.023 kWh (when the code
  still rounded). A real mistake is tens to thousands of kWh.
- **Refrigerator and solar hot water are set aside with hot water.** In 2025.1
  ResStock changes refrigerator electricity when the heating and cooling
  equipment changes, by up to about 500 kWh in a sample home. No limit loose
  enough to absorb that would still catch a real mistake.
- **Test (b) limit stays at 1%.** A flat "50 kWh plus 1%" limit was proposed
  and withdrawn. It only helped vacant homes, which TARE leaves out on purpose,
  and it would have dropped the exact-zero rule.
- **A planned note change in the column map was dropped.** The check finds the
  hot-water columns by their name pattern, not through the column map, so the
  map's "Dead code path" notes on the hot-water rows are still true.

Limits of that testing: the 2025.1 numbers come from the raw 2025.1 files, with
the sample rebuilt from the raw columns. The 2025.1 pipeline itself was not run.

### Step 5.5 -- run the check in every model run (`df_enduse_compare`, STEP 7 and 8)

Before:

```python
    def _annual_use(category: str, mp: int) -> pd.Series:
        """All-fuel annual use for one category; NaN for homes left out."""
        by_fuel = get_degree_day_adjusted_consumption_by_fuel(
            df_compare, category, ANCHOR_YEAR, mp)
        return pd.concat(by_fuel.values(), axis=1).sum(axis=1, min_count=1)

    # Heating savings stay NaN for homes without valid heating (never
    # eligible for a rebate). Cooling savings are 0 for a home without
    # cooling, so a heating-only home still gets a fraction.
    ...
    if 'cooling' in VALID_CATEGORIES:
        cooling_savings = (_annual_use('cooling', 0).fillna(0.0)
                           - _annual_use('cooling', menu_mp).fillna(0.0))
    ...
    # Check only, used in no calculation: ResStock's own whole-home change in
    # site energy. It runs a few percent below the HVAC savings above because
    # it includes the cooling the heat pump adds in homes outside cooling scope
    # (no central or room AC), which TARE counts as zero; hot water and
    # refrigerator side effects make up the small rest.
    ...
    return df_compare
```

After:

```python
    # Annual use by fuel, before (mp 0) and after, read once: the fraction
    # below adds the fuels together and the check in STEP 8 keeps them apart.
    use_by_fuel = {
        (category, mp): get_degree_day_adjusted_consumption_by_fuel(
            df_compare, category, ANCHOR_YEAR, mp)
        for category in ('heating', 'cooling') if category in VALID_CATEGORIES
        for mp in (0, menu_mp)}

    def _annual_use(category: str, mp: int) -> pd.Series:
        """All-fuel annual use for one category; NaN for homes left out."""
        return pd.concat(
            use_by_fuel[(category, mp)].values(), axis=1).sum(axis=1, min_count=1)

    # Savings stay NaN for homes outside the study sample.
    ...
    if 'cooling' in VALID_CATEGORIES:
        cooling_savings = (_annual_use('cooling', 0)
                           - _annual_use('cooling', menu_mp))
    ...
    # Check only, used in no calculation: ResStock's own whole-home change in
    # site energy. It differs from the HVAC savings above mostly because
    # ResStock's hot-water use also changes with the heating and cooling
    # equipment. STEP 8 tests what is left, per home and fuel.
    ...
    # ===== STEP 8: Check the savings against ResStock's own savings columns =====
    # The same baseline-minus-retrofit savings as STEP 7, kept apart by fuel.
    tare_savings_by_fuel = {}
    for (category, mp), by_fuel in use_by_fuel.items():
        sign = 1.0 if mp == 0 else -1.0
        for fuel, annual_use in by_fuel.items():
            tare_savings_by_fuel[fuel] = (
                tare_savings_by_fuel.get(fuel, 0.0) + sign * annual_use)
    check_savings_against_resstock(
        tare_savings_by_fuel, df_mp,
        df_compare.index[df_compare['include_sample'].astype(bool)],
        menu_mp, release)

    return df_compare
```

- No modeled value moves. The savings fraction is computed from the same
  numbers in the same order.
- The two `.fillna(0.0)` only ever acted on homes outside the sample, whose
  heating savings are blank anyway. Confirmed on the real data:
  `mp{mp}_hvac_energy_savings_kwh` is filled for exactly the 221,205 sample rdu.
- The old comment was wrong. Homes without AC are no longer in the sample, so
  the whole-home gap is hot-water use, not added cooling.

### Step 5.6 -- tests

`tests/utils/test_enduse_components.py`, three new tests:

```python
def test_savings_column_name_2022():
    """2022.1.1 adds '.savings' to the energy column name."""
    assert get_resstock_savings_column(
        'out.electricity.heating.energy_consumption.kwh'
    ) == 'out.electricity.heating.energy_consumption.kwh.savings'


def test_savings_column_name_2025():
    """2025.1 renames 'energy_consumption' to 'energy_savings'."""
    assert get_resstock_savings_column(
        'out.natural_gas.heating_hp_bkup.energy_consumption..kwh'
    ) == 'out.natural_gas.heating_hp_bkup.energy_savings..kwh'


def test_savings_column_bad_name_raises():
    with pytest.raises(ValueError, match='not a ResStock energy consumption'):
        get_resstock_savings_column('out.load.heating.energy_delivered.kbtu')
```

`tests/energy_consumption_and_metadata/test_process_euss_data.py`, one fixture
and five new tests. The fixture is three retrofitted homes with 2022.1.1 column
names: homes 1 and 2 heated with natural gas, home 3 heated with electricity and
using no natural gas, and no home using propane or fuel oil.

| Test | What it shows |
|---|---|
| `test_savings_check_passes` | Matching numbers pass, with hot-water and refrigerator changes set aside |
| `test_savings_check_left_out_part_raises` | 800 kWh the parts do not explain fails the whole-home test |
| `test_savings_check_part_mismatch_raises` | A 5 kWh disagreement fails the part test alone |
| `test_savings_check_home_without_the_fuel` | Zero use passes at exactly zero; 0.5 kWh there fails |
| `test_savings_check_blank_raises` | A blank fails and is not counted as zero |

The core of the fixture:

```python
        # Retrofit use, then ResStock's savings (baseline minus retrofit).
        f'out.electricity.heating.{energy}': [6000.0, 4000.0, 5000.0],
        f'out.electricity.heating.{energy}.savings': [-6000.0, -4000.0, 7000.0],
        f'out.electricity.cooling.{energy}': [2500.0, 1500.0, 2000.0],
        f'out.electricity.cooling.{energy}.savings': [500.0, 300.0, 400.0],
        f'out.natural_gas.heating.{energy}': [0.0, 0.0, 0.0],
        f'out.natural_gas.heating.{energy}.savings': [20000.0, 15000.0, 0.0],
        # Side effects ResStock reports outside heating and cooling.
        f'out.electricity.hot_water.{energy}.savings': [0.0, 50.0, -30.0],
        f'out.electricity.refrigerator.{energy}.savings': [10.0, 0.0, 0.0],
        ...
        # Whole home: heating + cooling savings plus the side effects.
        f'out.electricity.total.{energy}.savings': [-5490.0, -3650.0, 7370.0],
```

and one of the tests, as the pattern for the rest:

```python
def test_savings_check_left_out_part_raises(savings_check_inputs):
    """Energy the parts do not count shows up against the whole home."""
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        check_savings_against_resstock)
    tare_savings_by_fuel, df_mp, bldg_ids = savings_check_inputs
    # Home 1 used 800 kWh more electricity than its heating and cooling parts
    # explain, as if backup heat had been left out.
    df_mp.loc[1, 'out.electricity.total.energy_consumption.kwh.savings'] -= 800.0
    with pytest.raises(ValueError, match='electricity'):
        check_savings_against_resstock(
            tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1')
```

### Step 5.7 -- an earlier demand change that rode along (value-moving)

`a70a80a` also committed a change to `adoption_kpis/data_loading.py` and
`adoption_kpis/demand.py` that was written before this session and was sitting
in the working tree. It makes the county site-energy columns real.

New constant in `data_loading.py`:

```python
SITE_ENERGY_TOTAL_COL: str = "out.site_energy.total.energy_consumption.kwh"
"""EUSS column for whole-home site energy, ALL fuels (kWh; gas/oil/propane in
kWh-equivalent). Use this for the site-energy change only -- do NOT use it for
electricity demand, which is ELEC_TOTAL_COL."""
```

Before (`compute_scenario_demand`):

```python
    # NAMING NOTE: this "site energy change" is an ALIAS of the electricity
    # change, not an independent all-fuel quantity. ...
    df_demand['site_energy_change_kwh'] = df_demand['elec_demand_change_kwh']
```

After:

```python
    # Site energy counts every fuel (gas/oil/propane in kWh-equivalent), so it
    # is NOT the electricity change: electricity rises when a fossil system is
    # replaced, while site energy falls because the fuel is no longer burned.
    df_demand['site_energy_change_kwh'] = (
        df_demand['retrofit_site_energy_kwh'] - df_demand['baseline_site_energy_kwh']
    )
```

Before (`aggregate_demand`):

```python
    grouped['pct_site_energy_change'] = grouped['pct_elec_demand_change']
```

After:

```python
    # Site energy is its own all-fuel quantity, so its percent is taken against
    # baseline site energy, not baseline electricity. Blank only if a county's
    # baseline site energy is zero; zero-change homes still count.
    grouped['pct_site_energy_change'] = np.where(
        grouped['weighted_baseline_site'] != 0,
        grouped['weighted_site_change'] / grouped['weighted_baseline_site'] * 100,
        np.nan,
    )
```

Before (`aggregate_demand`, accounting check; the two symbols were a warning
sign and a check mark):

```python
    if not np.isclose(total_sum, total_agg, rtol=1e-6):
        print(f"<warning sign> DEMAND ACCOUNTING MISMATCH: sum={total_sum:.0f}, agg={total_agg:.0f}")
    elif verbose:
        print("<check mark> Demand accounting check passed")
```

After:

```python
    if not np.isclose(total_sum, total_agg, rtol=1e-6):
        raise ValueError(
            "Demand accounting mismatch: per-home total "
            f"{total_sum:.0f} kWh vs grouped total {total_agg:.0f} kWh")
    if verbose:
        print("[OK] Demand accounting check passed")
```

**This moves two columns of the county table**, `site_energy_change_gwh` and
`pct_site_energy_change`, in this commit, although the commit subject says no
modeled value moves. Nationally they are now -602,739.9 GWh (-32.07%) for MP3
and -879,495.2 GWh (-46.80%) for MP4. The electricity columns did not move.

---

## 6. Commit 4 (`76eedf1`) -- demand from savings columns, and the breakdown

### Plain-language summary

The demand module now reads each change from ResStock's own savings columns
and can split the electricity change into heating, cooling, hot water, and
everything else. It also refuses to treat a blank as zero.

- Every county value is identical to before this commit, in all 3,079 counties,
  for both packages.
- A new function prints a breakdown table for one package. The notebook puts
  MP3 and MP4 side by side.
- The "alias" wording in two documents is corrected.

Files: `adoption_kpis/data_loading.py`, `adoption_kpis/demand.py`,
`tests/adoption_kpis/test_demand_sample.py`, `utils/export_tepper_csv.py`,
`docs/tare_tepper_exports_data_dictionary.md`.

### Step 6.1 -- name the columns (`data_loading.py`)

New import: `from cmu_tare_model.utils.calculation_utils import get_resstock_savings_column`.

New energy column names:

```python
HEATING_ELEC_COL: str = "out.electricity.heating.energy_consumption.kwh"
COOLING_ELEC_COL: str = "out.electricity.cooling.energy_consumption.kwh"
COOLING_FANS_PUMPS_COL: str = (
    "out.electricity.cooling_fans_pumps.energy_consumption.kwh")
HOT_WATER_ELEC_COL: str = "out.electricity.hot_water.energy_consumption.kwh"
```

New savings column names, each derived from the energy column's name:

```python
# ResStock's own savings columns (baseline minus upgrade, positive = use fell).
# Only the upgrade files carry them. An end use's change is minus its savings.
ELEC_TOTAL_SAVINGS_COL: str = get_resstock_savings_column(ELEC_TOTAL_COL)
SITE_ENERGY_TOTAL_SAVINGS_COL: str = get_resstock_savings_column(SITE_ENERGY_TOTAL_COL)
HEATING_ELEC_SAVINGS_COLS: list[str] = [
    get_resstock_savings_column(energy_col)
    for energy_col in (HEATING_ELEC_COL, HP_BACKUP_ELEC_COL, HP_FANS_PUMPS_COL)]
COOLING_ELEC_SAVINGS_COLS: list[str] = [
    get_resstock_savings_column(energy_col)
    for energy_col in (COOLING_ELEC_COL, COOLING_FANS_PUMPS_COL)]
HOT_WATER_ELEC_SAVINGS_COL: str = get_resstock_savings_column(HOT_WATER_ELEC_COL)
```

The savings names cover electricity only. The breakdown is an electricity
breakdown, and site energy has its own whole-home savings column.

### Step 6.2 -- read changes from savings columns (`demand.py`, `compute_scenario_demand`)

New constant:

```python
DEMAND_CHANGE_TOLERANCE_KWH: float = 1.0
"""Largest gap allowed in one home between ResStock's whole-home savings column
and retrofit minus baseline (kWh). Far above what number storage can explain."""
```

Before:

```python
    baseline_total_elec = df_baseline[ELEC_TOTAL_COL].fillna(0)
    retrofit_total_elec = df_upgrade[ELEC_TOTAL_COL].fillna(0)
    baseline_site_energy = df_baseline[SITE_ENERGY_TOTAL_COL].fillna(0)
    retrofit_site_energy = df_upgrade[SITE_ENERGY_TOTAL_COL].fillna(0)

    df_demand = pd.DataFrame({
        'in.state': df_baseline['in.state'],
        'in.county': df_baseline[COUNTY_COL],
        'in.heating_fuel': df_baseline['in.heating_fuel'],
        'weight': df_baseline['weight'],
        'baseline_electric_kwh': baseline_total_elec,
        'baseline_site_energy_kwh': baseline_site_energy,
    }).join(
        pd.DataFrame({
            'retrofit_electric_kwh': retrofit_total_elec,
            'retrofit_site_energy_kwh': retrofit_site_energy,
        }),
        how='inner',
    )
    ...
    df_demand = df_demand.loc[sample_bldg_ids]

    if fuel_filter is not None:
        ...
            print(f"Filtered to '{fuel_filter}': {len(df_demand):,} / {n_before:,} homes")

    df_demand['elec_demand_change_kwh'] = (
        df_demand['retrofit_electric_kwh'] - df_demand['baseline_electric_kwh']
    )
    df_demand['site_energy_change_kwh'] = (
        df_demand['retrofit_site_energy_kwh'] - df_demand['baseline_site_energy_kwh']
    )

    for col in [
        'baseline_electric_kwh', 'retrofit_electric_kwh',
        'elec_demand_change_kwh',
        'baseline_site_energy_kwh', 'retrofit_site_energy_kwh',
        'site_energy_change_kwh',
    ]:
        df_demand[f'weighted_{col}'] = df_demand[col] * df_demand['weight']

    if verbose:
        fuel_label = fuel_filter if fuel_filter else 'all fuels'
        print(f"\n--- Demand Scenario Summary (100% adoption, {fuel_label}) ---")
        print(f"Total homes: {len(df_demand):,}")
        elec_gwh = df_demand['weighted_elec_demand_change_kwh'].sum() / KWH_TO_GWH
        site_gwh = df_demand['weighted_site_energy_change_kwh'].sum() / KWH_TO_GWH
        print(f"Weighted electricity demand change:  {elec_gwh:+,.1f} GWh (grid impact)")
        print(f"Weighted total site energy change:   {site_gwh:+,.1f} GWh (efficiency)")
```

After:

```python
    # Step 1 -- retrofit totals, and each change read from ResStock's own
    # savings columns. Savings are baseline minus upgrade, so the change is
    # 0 - savings (written that way so a zero does not become -0.0).
    df_changes = pd.DataFrame({
        'retrofit_electric_kwh': df_upgrade[ELEC_TOTAL_COL],
        'retrofit_site_energy_kwh': df_upgrade[SITE_ENERGY_TOTAL_COL],
        'elec_demand_change_kwh': 0.0 - df_upgrade[ELEC_TOTAL_SAVINGS_COL],
        'heating_elec_change_kwh': 0.0 - df_upgrade[
            HEATING_ELEC_SAVINGS_COLS].sum(axis=1, skipna=False),
        'cooling_elec_change_kwh': 0.0 - df_upgrade[
            COOLING_ELEC_SAVINGS_COLS].sum(axis=1, skipna=False),
        'hot_water_elec_change_kwh': 0.0 - df_upgrade[HOT_WATER_ELEC_SAVINGS_COL],
        # Site energy counts every fuel (gas/oil/propane in kWh-equivalent), so
        # it is NOT the electricity change: electricity rises when a fossil
        # system is replaced, while site energy falls because the fuel is no
        # longer burned.
        'site_energy_change_kwh': 0.0 - df_upgrade[SITE_ENERGY_TOTAL_SAVINGS_COL],
    })
    df_demand = pd.DataFrame({
        'in.state': df_baseline['in.state'],
        'in.county': df_baseline[COUNTY_COL],
        'in.heating_fuel': df_baseline['in.heating_fuel'],
        'weight': df_baseline['weight'],
        'baseline_electric_kwh': df_baseline[ELEC_TOTAL_COL],
        'baseline_site_energy_kwh': df_baseline[SITE_ENERGY_TOTAL_COL],
    }).join(df_changes, how='inner')
    ...
    df_demand = df_demand.loc[sample_bldg_ids]

    # Step 3 -- stop on a blank, which would otherwise count as zero use
    level_cols = [
        'baseline_electric_kwh', 'retrofit_electric_kwh',
        'baseline_site_energy_kwh', 'retrofit_site_energy_kwh']
    change_cols = [
        'elec_demand_change_kwh', 'heating_elec_change_kwh',
        'cooling_elec_change_kwh', 'hot_water_elec_change_kwh',
        'site_energy_change_kwh']
    blank_counts = df_demand[level_cols + change_cols].isna().sum()
    if blank_counts.any():
        raise ValueError(
            "Sample homes have blank energy values (column: rdu): "
            f"{blank_counts[blank_counts > 0].to_dict()}")

    # Step 4 -- ResStock's whole-home savings must agree with retrofit minus
    # baseline, the way the change used to be computed
    for change_col, quantity in (('elec_demand_change_kwh', 'electric'),
                                 ('site_energy_change_kwh', 'site_energy')):
        retrofit_minus_baseline = (
            df_demand[f'retrofit_{quantity}_kwh']
            - df_demand[f'baseline_{quantity}_kwh'])
        largest_gap = (df_demand[change_col] - retrofit_minus_baseline).abs().max()
        if largest_gap > DEMAND_CHANGE_TOLERANCE_KWH:
            raise ValueError(
                f"{change_col}: ResStock's savings column disagrees with "
                f"retrofit minus baseline by up to {largest_gap:,.3f} kWh in a home")

    # Step 5 -- everything outside heating, cooling, and hot water
    df_demand['other_elec_change_kwh'] = (
        df_demand['elec_demand_change_kwh']
        - df_demand['heating_elec_change_kwh']
        - df_demand['cooling_elec_change_kwh']
        - df_demand['hot_water_elec_change_kwh'])

    if fuel_filter is not None:
        ...
            print(f"Filtered to '{fuel_filter}': {len(df_demand):,} / {n_before:,} rdu")

    for col in level_cols + change_cols + ['other_elec_change_kwh']:
        df_demand[f'weighted_{col}'] = df_demand[col] * df_demand['weight']

    if verbose:
        fuel_label = fuel_filter if fuel_filter else 'all fuels'
        print(f"\n--- Demand Scenario Summary (100% adoption, {fuel_label}) ---")
        print(f"Study-sample rdu: {len(df_demand):,}")
        print(build_demand_breakdown(df_demand).to_string(
            float_format='{:,.3f}'.format))
```

What changed, in words:

- The four `.fillna(0)` are gone. A sample home with a blank now raises.
- The whole-home change is read from the savings column, and the function
  raises if that disagrees with retrofit minus baseline by more than 1 kWh. On
  the real data the largest gap is 0.00000000006 kWh.
- Four new per-home columns: heating, cooling, hot water, and "other".
- Two printed labels say "rdu" where they said "homes", because they count rows.

The docstring was updated to match, and its minus sign on the
`elec_demand_change_kwh` line was replaced with a plain hyphen.

### Step 6.3 -- the breakdown (`demand.py`, new function)

```python
def build_demand_breakdown(df_demand: pd.DataFrame) -> pd.Series:
    """Summarize one package's demand change at 100% adoption, in GWh.
    ...
    """
    def weighted_gwh(col: str) -> float:
        return df_demand[f'weighted_{col}'].sum() / KWH_TO_GWH

    breakdown = {
        'Baseline electricity': weighted_gwh('baseline_electric_kwh'),
        'Net heating': weighted_gwh('heating_elec_change_kwh'),
        'Net cooling': weighted_gwh('cooling_elec_change_kwh'),
        'Net hot water': weighted_gwh('hot_water_elec_change_kwh'),
        'Other end uses': weighted_gwh('other_elec_change_kwh'),
        'Post-retrofit electricity': weighted_gwh('retrofit_electric_kwh'),
        'Net change': weighted_gwh('elec_demand_change_kwh'),
    }

    # Heating electricity by the fuel the home heated with before the retrofit:
    # fossil homes add a heat pump's load, electric-resistance homes shed load.
    heating_gwh_by_fuel = df_demand.groupby('in.heating_fuel')[
        'weighted_heating_elec_change_kwh'].sum() / KWH_TO_GWH
    for heating_fuel, heating_gwh in heating_gwh_by_fuel.items():
        breakdown[f'Net heating, {heating_fuel} homes'] = heating_gwh

    baseline_site_gwh = weighted_gwh('baseline_site_energy_kwh')
    site_change_gwh = weighted_gwh('site_energy_change_kwh')
    breakdown['Baseline site energy (all fuels)'] = baseline_site_gwh
    breakdown['Site energy change (all fuels)'] = site_change_gwh
    breakdown['Site energy change (%)'] = site_change_gwh / baseline_site_gwh * 100
    return pd.Series(breakdown, name='GWh')
```

### Step 6.4 -- plain characters only (`demand.py`, `aggregate_demand`)

The docstring and one comment used a Greek capital sigma for "sum", a
multiplication sign, and a long dash. They now read:

```
    ``(sum of weighted_retrofit - sum of weighted_baseline)
    / sum of weighted_baseline x 100``.
```

```python
    # Percentage changes derive from already-computed GWh totals -- with
    # uniform weights, sum(w x change) / sum(w x baseline) equals
    # change_gwh / baseline_gwh.
```

### Step 6.5 -- tests (`test_demand_sample.py`)

Three tests failed at the start of the session because the fixture had no
site-energy column. The fixture was fixed first, then extended with the savings
columns once the function read them.

Before:

```python
@pytest.fixture
def euss_pair():
    """Five homes; each retrofit adds 100 x bldg_id kWh of electricity."""
    bldg_ids = pd.Index([1, 2, 3, 4, 5], name='bldg_id')
    df_baseline = pd.DataFrame({
        'in.state': ['PA'] * 5,
        COUNTY_COL: ['G4200030'] * 5,
        'in.heating_fuel': ['Natural Gas'] * 5,
        'weight': [242.131013] * 5,
        ELEC_TOTAL_COL: [1000.0] * 5,
    }, index=bldg_ids)
    df_upgrade = pd.DataFrame({
        ELEC_TOTAL_COL: 1000.0 + 100.0 * bldg_ids.to_numpy(),
    }, index=bldg_ids)
    return df_baseline, df_upgrade
```

After:

```python
@pytest.fixture
def euss_pair():
    """Five homes. Each retrofit adds 100 x bldg_id kWh of electricity (heating
    up by 150 x bldg_id, cooling down by 40 x bldg_id, hot water down by
    10 x bldg_id) and cuts site energy (all fuels) by 5,000 kWh. Homes 1-3
    heated with natural gas, homes 4-5 with electricity."""
    bldg_ids = pd.Index([1, 2, 3, 4, 5], name='bldg_id')
    per_home = bldg_ids.to_numpy().astype(float)
    df_baseline = pd.DataFrame({
        'in.state': ['PA'] * 5,
        COUNTY_COL: ['G4200030'] * 5,
        'in.heating_fuel': ['Natural Gas'] * 3 + ['Electricity'] * 2,
        'weight': [WEIGHT] * 5,
        ELEC_TOTAL_COL: [1000.0] * 5,
        SITE_ENERGY_TOTAL_COL: [20000.0] * 5,
    }, index=bldg_ids)
    heating_col, backup_col, heating_fans_col = HEATING_ELEC_SAVINGS_COLS
    cooling_col, cooling_fans_col = COOLING_ELEC_SAVINGS_COLS
    df_upgrade = pd.DataFrame({
        ELEC_TOTAL_COL: 1000.0 + 100.0 * per_home,
        SITE_ENERGY_TOTAL_COL: [15000.0] * 5,
        # ResStock's savings: baseline minus upgrade, so a rise is negative.
        ELEC_TOTAL_SAVINGS_COL: -100.0 * per_home,
        SITE_ENERGY_TOTAL_SAVINGS_COL: [5000.0] * 5,
        heating_col: -100.0 * per_home,
        backup_col: -30.0 * per_home,
        heating_fans_col: -20.0 * per_home,
        cooling_col: 30.0 * per_home,
        cooling_fans_col: 10.0 * per_home,
        HOT_WATER_ELEC_SAVINGS_COL: 10.0 * per_home,
    }, index=bldg_ids)
    return df_baseline, df_upgrade
```

The four original tests are unchanged. Five are new:

| Test | What it shows |
|---|---|
| `test_changes_read_from_savings_columns` | Each change is minus ResStock's savings; "other" is what is left |
| `test_site_energy_is_its_own_all_fuel_change` | Site energy falls while electricity rises |
| `test_savings_disagreeing_with_totals_raises` | A savings column that disagrees with retrofit minus baseline stops the run |
| `test_blank_in_sample_home_raises` | A blank in a sample home stops the run; outside the sample it does not |
| `test_breakdown_adds_up` | Baseline plus the net changes is post-retrofit; the fuel rows add to net heating |

Two of them, as the pattern for the rest:

```python
def test_savings_disagreeing_with_totals_raises(euss_pair):
    """ResStock's savings column must agree with retrofit minus baseline."""
    df_b, df_u = euss_pair
    df_u.loc[3, ELEC_TOTAL_SAVINGS_COL] += 50.0
    with pytest.raises(ValueError, match='elec_demand_change_kwh'):
        compute_scenario_demand(df_b, df_u, sample_bldg_ids=df_b.index)


def test_blank_in_sample_home_raises(euss_pair):
    """A blank in a sample home stops the run; outside the sample it does not."""
    df_b, df_u = euss_pair
    df_u.loc[2, HOT_WATER_ELEC_SAVINGS_COL] = np.nan
    with pytest.raises(ValueError, match='blank'):
        compute_scenario_demand(df_b, df_u, sample_bldg_ids=df_b.index)
    result = compute_scenario_demand(df_b, df_u, sample_bldg_ids=pd.Index([1, 3]))
    assert sorted(result.index) == [1, 3]
```

### Step 6.6 -- correct the "alias" wording

`utils/export_tepper_csv.py`, before:

```python
# NOTE: 'site_energy_change_gwh' and 'pct_site_energy_change' are ALIASES of the
# electricity metrics (see demand.py) -- with whole-home electrification measured
# on the electricity total, site-energy change equals electricity change. They
# are NOT independent all-fuel numbers; prefer 'elec_change_gwh' /
# 'pct_elec_demand_change' for an electricity read.
```

After:

```python
# NOTE: 'site_energy_change_gwh' and 'pct_site_energy_change' are all-fuel site
# energy (natural gas, fuel oil, and propane counted in kWh), read from
# ResStock's whole-home site energy (see demand.py). They are separate from the
# electricity metrics and usually have the opposite sign: electricity rises
# while site energy falls. For an electricity read use 'elec_change_gwh' /
# 'pct_elec_demand_change'.
```

`docs/tare_tepper_exports_data_dictionary.md`, section 11, before:

```
**`site_energy_change_gwh` and `pct_site_energy_change` are aliases**, not
independent all-fuel numbers. Both sides are read from whole-home electricity,
and because the retrofit fully electrifies heating and cooling the two measures
converge by construction. For an electricity reading use `elec_change_gwh` and
`pct_elec_demand_change`; do not treat the site-energy pair as a separate
result.
```

After:

```
**`site_energy_change_gwh` and `pct_site_energy_change` are all-fuel numbers**,
separate from the electricity pair. They come from ResStock's whole-home site
energy, which counts natural gas, fuel oil, and propane in kWh alongside
electricity. Electricity usually rises when a fossil system is replaced, while
site energy falls because the fuel is no longer burned, so the two pairs often
have opposite signs. `pct_site_energy_change` is taken against baseline site
energy, not baseline electricity. For an electricity reading use
`elec_change_gwh` and `pct_elec_demand_change`.
```

### Step 6.7 -- the notebook (pasted by the researcher; not in any commit yet)

`tare_model_main_v3_0.ipynb`, 21st cell, Step 3. The cell keeps each package's
per-home frame and prints the side-by-side breakdown.

Before:

```python
demand_results = {}
for mp in selected_mps:
    print(f"\n===== {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")

    df_demand = compute_scenario_demand(
        df_baseline=df_baseline, df_upgrade=upgrade_data[mp],
        sample_bldg_ids=TARE_SAMPLE_IDS['all'], fuel_filter=None, verbose=True,
    )

    demand_results[mp] = aggregate_demand(
        df_demand=df_demand, geo_level='county', verbose=True,
    )
```

After:

```python
from cmu_tare_model.adoption_kpis.demand import build_demand_breakdown
...
demand_results = {}
demand_frames = {}  # each package's per-home frame, kept for the breakdown below
for mp in selected_mps:
    print(f"\n===== {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")

    demand_frames[mp] = compute_scenario_demand(
        df_baseline=df_baseline, df_upgrade=upgrade_data[mp],
        sample_bldg_ids=TARE_SAMPLE_IDS['all'], fuel_filter=None, verbose=False,
    )

    demand_results[mp] = aggregate_demand(
        df_demand=demand_frames[mp], geo_level='county', verbose=True,
    )

# Side-by-side breakdown, GWh at 100% adoption (the last row is a percent).
print(f"\n{'='*60}")
print("DEMAND BREAKDOWN -- GWh at 100% adoption, study sample")
print(f"{'='*60}")
df_demand_breakdown = pd.DataFrame({
    f"MP{mp}": build_demand_breakdown(demand_frames[mp]) for mp in selected_mps
})
print(df_demand_breakdown.to_string(float_format='{:,.3f}'.format))
```

---

## 7. What was measured on the real 2022.1.1 files

These numbers come from running the changed code on the real ResStock files.
They are **not** from a full model run, and none of them is a reference value
yet.

### 7.1 Sample funnel

| Step | rdu | Removed |
|---|---|---|
| `load` | 548,916 | |
| `applicability` | 548,260 | 656 |
| `occupancy` | 482,050 | 66,210 |
| `housing_type` | 331,526 | 150,524 |
| `exclude_AK_HI` | 331,526 | 0 |
| `heating_fuel` | 321,352 | 10,174 |
| `no_existing_heat_pump` | 290,005 | 31,347 |
| `replaceable_heating_system` | 260,211 | 29,794 |
| `central_or_room_ac` | 221,301 | 38,910 |
| `no_shared_cooling` | 221,205 | 96 |

Study sample: 221,205 rdu = 53,560,591 homes, in 3,079 counties.

### 7.2 The savings check, run through the pipeline

| Package | Fuel | (a) largest gap, kWh | (b) largest left over, kWh | (b) largest share of the home's use | rdu failed |
|---|---|---|---|---|---|
| MP3 | electricity | 0.00000000006 | 10.844 | 0.089% | 0 |
| MP3 | natural gas | 0 | 0.586 | 0.039% | 0 |
| MP3 | propane | 0 | 0.586 | 0.035% | 0 |
| MP3 | fuel oil | 0 | 0.293 | 0.005% | 0 |
| MP4 | electricity | 0.00000000003 | 5.568 | 0.043% | 0 |
| MP4 | fossil fuels | 0 | same as MP3 | same as MP3 | 0 |

The left-over amounts are whole multiples of 0.293 kWh, which is 1 kBtu,
ResStock's own unit. That points to bookkeeping inside ResStock's totals. The
cause was not traced.

### 7.3 Demand breakdown (GWh at 100% adoption)

| Row | MP3 | MP4 |
|---|---|---|
| Baseline electricity | 795,464.2 | 795,464.2 |
| Net heating | +294,309.5 | +109,115.7 |
| Net cooling | -16,119.5 | -107,833.8 |
| Net hot water | -13.4 | -3.3 |
| Other end uses | -0.029 | +0.009 |
| Post-retrofit electricity | 1,073,640.8 | 796,742.8 |
| Net change | +278,176.6 | +1,278.6 |
| Net heating, Electricity homes | -83,314.9 | -112,582.6 |
| Net heating, Fuel Oil homes | +46,315.9 | +27,433.7 |
| Net heating, Natural Gas homes | +298,637.2 | +174,532.5 |
| Net heating, Propane homes | +32,671.2 | +19,732.0 |
| Baseline site energy (all fuels) | 1,879,204.4 | 1,879,204.4 |
| Site energy change (all fuels) | -602,739.9 | -879,495.2 |
| Site energy change (%) | -32.07 | -46.80 |

### 7.4 County table

- All six county demand columns are identical, in all 3,079 counties for both
  packages, whether the change is read from the savings column or computed as
  retrofit minus baseline.
- The county summary printed by `aggregate_demand` adds up county values that
  are each rounded to 0.01 GWh, so its total (+278,176.8 GWh for MP3) differs
  from the breakdown's (+278,176.6 GWh) by rounding only.

### 7.5 Tests

| Run | Start of session | End of session |
|---|---|---|
| Without geopandas | 313 passed, 3 skipped | 322 passed, 3 skipped |
| With geopandas | 317 passed, 3 failed, 1 skipped | 334 passed, 1 skipped, 0 failed |

---

## 8. Observations to carry forward

### 8.1 Site energy by county under MP3 and MP4

The county table's site-energy percent runs from -61.74% to +11.75% for MP3
and from -72.36% to -6.22% for MP4.

**The positive MP3 values are not a finding.** Site energy rises under MP3 in
only 3 of 3,079 counties: two in Texas and one in North Dakota. Together they
hold 6 rdu (1,453 homes), with 1 to 3 rdu each. The +11.75% county is a single
rdu. In those six rdu the heat pump's added heating electricity (+11.8 GWh) is
slightly more than the fossil fuel it replaces (-11.5 GWh).

What is safe to say: under MP4, site energy falls in every county. Under MP3 it
falls in every county that has more than three sample rdu.

What this raises: `MIN_HOME_COUNT` is 1, so a county with a single rdu gets a
value on the maps. Any county-level extreme should be checked against its rdu
count before it is quoted.

### 8.2 End uses TARE does not count

The check sets hot water, solar hot water, and the refrigerator aside. TARE
does not count those changes in its costs either.

- In 2022.1.1 only hot water changes, and it is small in total: -13.4 GWh for
  MP3 against +294,309.5 GWh for heating.
- In 2025.1 the refrigerator changes too, by up to about 500 kWh in a sample
  home. Averaged over every home the dual-fuel package applies to, it is about
  5 kWh per home. That deserves a line in the limitations when 2025.1 is
  written up.

### 8.3 Results that are not on the study sample

- The capital-cost workbook tables (`_analyze_*` in `validate_capital_costs.py`,
  used by the notebook's capital-cost cells) do not filter on `include_sample`.
  This is Session 2 work.
- Two notebook print lines count rows but say "homes", and they describe the
  loader's population, not the study sample: "Baseline: N occupied SF homes"
  (setup cell) and "MPn: N applicable homes" (demand cell, Step 1).

### 8.4 Small leftovers seen but not changed

- `process_euss_data.py`, `df_enduse_refactored` STEP 3 still fills blanks with
  0 before adding up the baseline total.
- `adoption_kpis/thermal_cop.py` still has non-plain characters and its own
  `.fillna(0)` calls. The notebook does not call it today.
- A user who turns off the occupancy filter would hit the savings check on a
  handful of vacant homes (2 rdu in 2022.1.1, 6 rdu in 2025.1).

---

## 9. What has not been done

- **No model run.** Every modeled value will move from the shared-cooling
  change. Until a new run is loaded, the notebook still reports the old
  221,301-rdu sample.
- **No reference values recorded.** `REFERENCE_VALUES.md` has no rows for the
  new sample.
- **CLAUDE.md not updated.** Its sample counts, its cooling rule, and its
  limitations list predate this session.
- **`SESSION_LOG.md` not updated.** It has no entry pointing to this file yet.
- **The 2025.1 pipeline was not run.**

Next: Session 2 (`HANDOFF_2026-10-02_session2_prompt.md`).
