Here is a clean, complete summary of the full workflow and findings in this conversation — including the original problem, why it mattered, and the exact code you used.

---

# ✅ **Summary of the Conversation**

## **1. The Issue With Using the NREL National CSV**

You discovered that the **NREL national EUSS CSV (downloaded from NREL)** has **fewer columns** than the combined state-level data.

Your documentation explicitly confirms the issue:

> *“The NREL EUSS database National CSV data were missing several of the metadata columns present in the individual state CSV files.”*

This turned out to be true:

### **Column counts**

* **National CSV:** 302 columns
* **Processed (state-merged) CSV:** 348 columns

### **Missing in the national file**

Your diff revealed **46 missing columns**, including:

* peak load variables
* unmet hours
* hot water end-use disaggregation
* HVAC sizing parameters
* envelope surface areas (roof, walls, ducts, windows)
* infiltration and ventilation parameters

These are all required for:

* retrofit sizing
* cost estimation (Tables S1–S3)
* envelope upgrade calculations
* peak load & safety analysis

### **Conclusion**

Using the national CSV directly would **break your TARE model**, because:

* it omits critical metadata and engineering parameters
* it cannot support replacement, installation, or enclosure cost calculations

Your processed file (merged from all state CSVs) is the **correct and complete** dataset.

---

## **2. EXACT Code Used to Compare Columns and Identify Missing Fields**

```python
import pandas as pd

# --- Update these paths ---
national_path = r"C:\path\to\national_raw_euss.csv"
processed_path = r"C:\path\to\your_processed_combined_euss.csv"

# Load only headers (faster)
national_cols = pd.read_csv(national_path, nrows=0).columns
processed_cols = pd.read_csv(processed_path, nrows=0).columns

# Convert to sets
national_set = set(national_cols)
processed_set = set(processed_cols)

# Differences
missing_in_national = processed_set - national_set
missing_in_processed = national_set - processed_set
common = processed_set & national_set

print("\n=== Columns missing in NATIONAL CSV (but present in your processed file) ===")
print(sorted(missing_in_national))

print("\n=== Columns missing in YOUR PROCESSED FILE (but present in national CSV) ===")
print(sorted(missing_in_processed))

print("\n=== Number of common columns ===")
print(len(common))
```

This script produced:

'''
Number of columns in NATIONAL CSV: 302
Number of columns in PROCESSED CSV: 348

=== Columns missing in NATIONAL CSV (but present in your processed file) ===
['out.electricity.peak_when_cooling.kw', 'out.electricity.peak_when_cooling.kw.savings', 'out.electricity.peak_when_heating.kw', 'out.electricity.peak_when_heating.kw.savings', 'out.hot_water.clothes_washer.gal', 'out.hot_water.clothes_washer.gal.savings', 'out.hot_water.dishwasher.gal', 'out.hot_water.dishwasher.gal.savings', 'out.hot_water.distribution_waste.gal', 'out.hot_water.distribution_waste.gal.savings', 'out.hot_water.fixtures.gal', 'out.hot_water.fixtures.gal.savings', 'out.load.cooling.energy_delivered.kbtu', 'out.load.cooling.energy_delivered.kbtu.savings', 'out.load.cooling.peak.kbtu_hr', 'out.load.cooling.peak.kbtu_hr.savings', 'out.load.heating.energy_delivered.kbtu', 'out.load.heating.energy_delivered.kbtu.savings', 'out.load.heating.peak.kbtu_hr', 'out.load.heating.peak.kbtu_hr.savings', 'out.load.hot_water.energy_delivered.kbtu', 'out.load.hot_water.energy_delivered.kbtu.savings', 'out.params.door_area_ft_2', 'out.params.duct_unconditioned_surface_area_ft_2', 'out.params.floor_area_attic_ft_2', 'out.params.floor_area_attic_insulation_increase_ft_2_delta_r_value', 'out.params.floor_area_conditioned_infiltration_reduction_ft_2_delta_ach_50', 'out.params.floor_area_foundation_ft_2', 'out.params.floor_area_lighting_ft_2', 'out.params.flow_rate_mechanical_ventilation_cfm', 'out.params.rim_joist_area_above_grade_exterior_ft_2', 'out.params.roof_area_ft_2', 'out.params.size_cooling_system_primary_k_btu_h', 'out.params.size_heat_pump_backup_primary_k_btu_h', 'out.params.size_heating_system_primary_k_btu_h', 'out.params.size_heating_system_secondary_k_btu_h', 'out.params.size_water_heater_gal', 'out.params.slab_perimeter_exposed_conditioned_ft', 'out.params.wall_area_above_grade_conditioned_ft_2', 'out.params.wall_area_above_grade_exterior_ft_2', 'out.params.wall_area_below_grade_ft_2', 'out.params.window_area_ft_2', 'out.unmet_hours.cooling.hour', 'out.unmet_hours.cooling.hour.savings', 'out.unmet_hours.heating.hour', 'out.unmet_hours.heating.hour.savings']

=== Columns missing in YOUR PROCESSED FILE (but present in national CSV) ===
[]

=== Number of common columns ===
302
'''

---

## **3. EXACT Code Used to Combine the State-Level CSVs**

```python
import os
import pandas as pd

folder_path = r'C:\Users\Jordan\Research\Paper1\euss_data\resstock_amy2018_release_1.1\state\state_baseline'
output_file_path = r'C:\Users\Jordan\Research\Paper1\euss_data\resstock_amy2018_release_1.1\state\baseline_metadata_and_annual_results.csv'

csv_file_names = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

combined_data = pd.concat([pd.read_csv(os.path.join(folder_path, file), dtype='object') for file in csv_file_names], ignore_index=True, sort=False)
combined_data.to_csv(output_file_path, index=False)

```

### **Why `dtype='object'` was included**

Some columns had mixed types across states → Pandas threw `DtypeWarning`.
Reading as `object` ensured:

* no coercion
* no dropped columns
* consistent schema across states

---

## **4. EXACT Code Used to Verify That All Expected CSV Files Were Present**

```python
import os

states_list = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

csv_file_names = [state + "_baseline_metadata_and_annual_results.csv" for state in states_list]
print(csv_file_names)

folder_path = r'C:\Users\Jordan\Research\Paper1\euss_data\resstock_amy2018_release_1.1\state\state_baseline'
csv_file_names = ['AL_baseline_metadata_and_annual_results.csv', 'AR_baseline_metadata_and_annual_results.csv', 'AZ_baseline_metadata_and_annual_results.csv', 'CA_baseline_metadata_and_annual_results.csv', 'CO_baseline_metadata_and_annual_results.csv', 'CT_baseline_metadata_and_annual_results.csv', 'DC_baseline_metadata_and_annual_results.csv', 'DE_baseline_metadata_and_annual_results.csv', 'FL_baseline_metadata_and_annual_results.csv', 'GA_baseline_metadata_and_annual_results.csv', 'IA_baseline_metadata_and_annual_results.csv', 'ID_baseline_metadata_and_annual_results.csv', 'IL_baseline_metadata_and_annual_results.csv', 'IN_baseline_metadata_and_annual_results.csv', 'KS_baseline_metadata_and_annual_results.csv', 'KY_baseline_metadata_and_annual_results.csv', 'LA_baseline_metadata_and_annual_results.csv', 'MA_baseline_metadata_and_annual_results.csv', 'MD_baseline_metadata_and_annual_results.csv', 'ME_baseline_metadata_and_annual_results.csv', 'MI_baseline_metadata_and_annual_results.csv', 'MN_baseline_metadata_and_annual_results.csv', 'MO_baseline_metadata_and_annual_results.csv', 'MS_baseline_metadata_and_annual_results.csv', 'MT_baseline_metadata_and_annual_results.csv', 'NC_baseline_metadata_and_annual_results.csv', 'ND_baseline_metadata_and_annual_results.csv', 'NE_baseline_metadata_and_annual_results.csv', 'NH_baseline_metadata_and_annual_results.csv', 'NJ_baseline_metadata_and_annual_results.csv', 'NM_baseline_metadata_and_annual_results.csv', 'NV_baseline_metadata_and_annual_results.csv', 'NY_baseline_metadata_and_annual_results.csv', 'OH_baseline_metadata_and_annual_results.csv', 'OK_baseline_metadata_and_annual_results.csv', 'OR_baseline_metadata_and_annual_results.csv', 'PA_baseline_metadata_and_annual_results.csv', 'RI_baseline_metadata_and_annual_results.csv', 'SC_baseline_metadata_and_annual_results.csv', 'SD_baseline_metadata_and_annual_results.csv', 'TN_baseline_metadata_and_annual_results.csv', 'TX_baseline_metadata_and_annual_results.csv', 'UT_baseline_metadata_and_annual_results.csv', 'VA_baseline_metadata_and_annual_results.csv', 'VT_baseline_metadata_and_annual_results.csv', 'WA_baseline_metadata_and_annual_results.csv', 'WI_baseline_metadata_and_annual_results.csv', 'WV_baseline_metadata_and_annual_results.csv', 'WY_baseline_metadata_and_annual_results.csv']

missing_files = []

for file_name in csv_file_names:
    file_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(file_path):
        missing_files.append(file_name)

if len(missing_files) == 0:
    print("All CSV files are present in the directory.")
else:
    print("Missing CSV files:")
    print("\n".join(missing_files))
```

---

## **5. Code Used to Verify All States Were Present in the Combined DataFrame**

```python
states_list = ['AL', 'AR', 'AZ', ... , 'WI', 'WV', 'WY']

missing_states = []

for state in states_list:
    if state not in df_euss_am_baseline['in.state'].values:
        missing_states.append(state)

if len(missing_states) == 0:
    print("All states are present in the DataFrame.")
else:
    print("Missing states:")
    print("\n".join(missing_states))
```

---

# ⭐ **Final Summary**

* The national NREL EUSS dataset is **incomplete** and missing **46 critical metadata columns**, including HVAC sizing, loads, envelope areas, and hot water disaggregation.
* Combining the individual state-level CSVs produced a **full 348-column dataset**, which correctly matches your methodological requirements (Tables S1–S3).
* Your validation code confirmed:

  * **All state CSVs were included**,
  * **No columns were missing in the processed file**,
  * **The national file is a strict subset** of your merged dataset.

If you'd like, I can package all of this into a **PDF or docx documentation file** you can store in your repo.
