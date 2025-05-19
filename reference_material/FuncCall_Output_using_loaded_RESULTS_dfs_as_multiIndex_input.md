# =======================================================================================================
# FUNCTION CALL FOR MULTI-INDEX DF CREATION (uses loaded results output dfs as input):
# =======================================================================================================


# =======================================================================================================
# SPACE HEATING ADOPTION POTENTIAL (MP8, MP9, MP10): HEALTH RCM-CRF SENSITIVITY s 
# =======================================================================================================
# Common parameters
scc = 'upper'

# ========================== AP2  ========================== 
rcm_model = 'ap2'

print(f"""
Adoption Potential Summary Dataframes are then used to create Multi-Index Dataframes for the following:
- Retrofit Scenarios: Basic (MP8), Moderate (MP9), Advanced (MP10)
- SCC Climate Sensitivity: {scc}
- Health Sensitivity (RCM): AP2, EASIUR, InMAP
- Health Sensitivity (CR Function): 'acs' or 'h6c'

------------------------------------------------------------------------------------------------
RCM HEALTH SENSITIVITY: AP2
------------------------------------------------------------------------------------------------
SCC Climate Sensitivity: {scc}
Health RCM Model: {rcm_model}
Health CR Function: 'acs' or 'h6c'

Creating Multi-Index Dataframes for Space Heating - Basic (MP8), Moderate (MP9), Advanced (MP10) Retrofit ...

""")
# Basic Retrofit (MP8)
df_mi_basic_heating_adoption_ap2_acs = create_multiIndex_adoption_df(df_outputs_basic_heating_ap2, 8, 'heating', scc, rcm_model, 'acs')
df_mi_basic_heating_adoption_ap2_h6c = create_multiIndex_adoption_df(df_outputs_basic_heating_ap2, 8, 'heating', scc, rcm_model, 'h6c')
# Moderate Retrofit (MP9)
df_mi_moderate_heating_adoption_ap2_acs = create_multiIndex_adoption_df(df_outputs_moderate_heating_ap2, 9, 'heating', scc, rcm_model, 'acs')
df_mi_moderate_heating_adoption_ap2_h6c = create_multiIndex_adoption_df(df_outputs_moderate_heating_ap2, 9, 'heating', scc, rcm_model, 'h6c')
# Advanced Retrofit (MP10)
df_mi_advanced_heating_adoption_ap2_acs = create_multiIndex_adoption_df(df_outputs_advanced_heating_ap2, 10, 'heating', scc, rcm_model, 'acs')
df_mi_advanced_heating_adoption_ap2_h6c = create_multiIndex_adoption_df(df_outputs_advanced_heating_ap2, 10, 'heating', scc, rcm_model, 'h6c')

# ========================== EASIUR  ========================== 
rcm_model = 'easiur'

print(f"""
------------------------------------------------------------------------------------------------
RCM HEALTH SENSITIVITY: EASIUR
------------------------------------------------------------------------------------------------
SCC Climate Sensitivity: {scc}
Health RCM Model: {rcm_model}
Health CR Function: 'acs' or 'h6c'

Creating Multi-Index Dataframes for Space Heating - Basic (MP8), Moderate (MP9), Advanced (MP10) Retrofit ...

""")
# Basic Retrofit (MP8)
df_mi_basic_heating_adoption_easiur_acs = create_multiIndex_adoption_df(df_outputs_basic_heating_easiur, 8, 'heating', scc, rcm_model, 'acs')
df_mi_basic_heating_adoption_easiur_h6c = create_multiIndex_adoption_df(df_outputs_basic_heating_easiur, 8, 'heating', scc, rcm_model, 'h6c')
# Moderate Retrofit (MP9)
df_mi_moderate_heating_adoption_easiur_acs = create_multiIndex_adoption_df(df_outputs_moderate_heating_easiur, 9, 'heating', scc, rcm_model, 'acs')
df_mi_moderate_heating_adoption_easiur_h6c = create_multiIndex_adoption_df(df_outputs_moderate_heating_easiur, 9, 'heating', scc, rcm_model, 'h6c')
# Advanced Retrofit (MP10)
df_mi_advanced_heating_adoption_easiur_acs = create_multiIndex_adoption_df(df_outputs_advanced_heating_easiur, 10, 'heating', scc, rcm_model, 'acs')
df_mi_advanced_heating_adoption_easiur_h6c = create_multiIndex_adoption_df(df_outputs_advanced_heating_easiur, 10, 'heating', scc, rcm_model, 'h6c')

# ========================== InMAP ========================== 
rcm_model = 'inmap'

print(f"""
------------------------------------------------------------------------------------------------
RCM HEALTH SENSITIVITY: InMAP
------------------------------------------------------------------------------------------------
SCC Climate Sensitivity: {scc}
Health RCM Model: {rcm_model}
Health CR Function: 'acs' or 'h6c'

Creating Multi-Index Dataframes for Space Heating - Basic (MP8), Moderate (MP9), Advanced (MP10) Retrofit ...

""")
# Basic Retrofit (MP8)
df_mi_basic_heating_adoption_inmap_acs = create_multiIndex_adoption_df(df_outputs_basic_heating_inmap, 8, 'heating', scc, rcm_model, 'acs')
df_mi_basic_heating_adoption_inmap_h6c = create_multiIndex_adoption_df(df_outputs_basic_heating_inmap, 8, 'heating', scc, rcm_model, 'h6c')
# Moderate Retrofit (MP9)
df_mi_moderate_heating_adoption_inmap_acs = create_multiIndex_adoption_df(df_outputs_moderate_heating_inmap, 9, 'heating', scc, rcm_model, 'acs')
df_mi_moderate_heating_adoption_inmap_h6c = create_multiIndex_adoption_df(df_outputs_moderate_heating_inmap, 9, 'heating', scc, rcm_model, 'h6c')
# Advanced Retrofit (MP10)
df_mi_advanced_heating_adoption_inmap_acs = create_multiIndex_adoption_df(df_outputs_advanced_heating_inmap, 10, 'heating', scc, rcm_model, 'acs')
df_mi_advanced_heating_adoption_inmap_h6c = create_multiIndex_adoption_df(df_outputs_advanced_heating_inmap, 10, 'heating', scc, rcm_model, 'h6c')



# =======================================================================================================
# OUTPUT (uses loaded results output dfs as input):
# =======================================================================================================

Adoption Potential Summary Dataframes are then used to create Multi-Index Dataframes for the following:
- Retrofit Scenarios: Basic (MP8), Moderate (MP9), Advanced (MP10)
- SCC Climate Sensitivity: upper
- Health Sensitivity (RCM): AP2, EASIUR, InMAP
- Health Sensitivity (CR Function): 'acs' or 'h6c'

------------------------------------------------------------------------------------------------
RCM HEALTH SENSITIVITY: AP2
------------------------------------------------------------------------------------------------
SCC Climate Sensitivity: upper
Health RCM Model: ap2
Health CR Function: 'acs' or 'h6c'

Creating Multi-Index Dataframes for Space Heating - Basic (MP8), Moderate (MP9), Advanced (MP10) Retrofit ...



DEBUGGING FOR: heating (mp8, ap2, acs)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_ap2_acs
  - iraRef_mp8_heating_adoption_upper_ap2_acs

Sample of actual columns in DataFrame (725 total):
Found 16 adoption-related columns:
  1. preIRA_mp8_heating_adoption_upper_ap2_acs
  2. preIRA_mp8_waterHeating_adoption_upper_ap2_acs
  3. preIRA_mp8_clothesDrying_adoption_upper_ap2_acs
  4. preIRA_mp8_cooking_adoption_upper_ap2_acs
  5. preIRA_mp8_heating_adoption_upper_ap2_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp8_heating_adoption_upper_ap2_acs', 'iraRef_mp8_heating_adoption_upper_ap2_acs']

DEBUGGING FOR: heating (mp8, ap2, h6c)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_ap2_h6c
  - iraRef_mp8_heating_adoption_upper_ap2_h6c

Sample of actual columns in DataFrame (725 total):
Found 16 adoption-related columns:
  1. preIRA_mp8_heating_adoption_upper_ap2_acs
  2. preIRA_mp8_waterHeating_adoption_upper_ap2_acs
  3. preIRA_mp8_clothesDrying_adoption_upper_ap2_acs
  4. preIRA_mp8_cooking_adoption_upper_ap2_acs
  5. preIRA_mp8_heating_adoption_upper_ap2_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp8_heating_adoption_upper_ap2_h6c', 'iraRef_mp8_heating_adoption_upper_ap2_h6c']

DEBUGGING FOR: heating (mp9, ap2, acs)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_ap2_acs
  - iraRef_mp9_heating_adoption_upper_ap2_acs

Sample of actual columns in DataFrame (741 total):
Found 16 adoption-related columns:
  1. preIRA_mp9_heating_adoption_upper_ap2_acs
  2. preIRA_mp9_waterHeating_adoption_upper_ap2_acs
  3. preIRA_mp9_clothesDrying_adoption_upper_ap2_acs
  4. preIRA_mp9_cooking_adoption_upper_ap2_acs
  5. preIRA_mp9_heating_adoption_upper_ap2_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp9_heating_adoption_upper_ap2_acs', 'iraRef_mp9_heating_adoption_upper_ap2_acs']

DEBUGGING FOR: heating (mp9, ap2, h6c)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_ap2_h6c
  - iraRef_mp9_heating_adoption_upper_ap2_h6c

Sample of actual columns in DataFrame (741 total):
Found 16 adoption-related columns:
  1. preIRA_mp9_heating_adoption_upper_ap2_acs
  2. preIRA_mp9_waterHeating_adoption_upper_ap2_acs
  3. preIRA_mp9_clothesDrying_adoption_upper_ap2_acs
  4. preIRA_mp9_cooking_adoption_upper_ap2_acs
  5. preIRA_mp9_heating_adoption_upper_ap2_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp9_heating_adoption_upper_ap2_h6c', 'iraRef_mp9_heating_adoption_upper_ap2_h6c']

DEBUGGING FOR: heating (mp10, ap2, acs)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_ap2_acs
  - iraRef_mp10_heating_adoption_upper_ap2_acs

Sample of actual columns in DataFrame (756 total):
Found 16 adoption-related columns:
  1. preIRA_mp10_heating_adoption_upper_ap2_acs
  2. preIRA_mp10_waterHeating_adoption_upper_ap2_acs
  3. preIRA_mp10_clothesDrying_adoption_upper_ap2_acs
  4. preIRA_mp10_cooking_adoption_upper_ap2_acs
  5. preIRA_mp10_heating_adoption_upper_ap2_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp10_heating_adoption_upper_ap2_acs', 'iraRef_mp10_heating_adoption_upper_ap2_acs']

DEBUGGING FOR: heating (mp10, ap2, h6c)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_ap2_h6c
  - iraRef_mp10_heating_adoption_upper_ap2_h6c

Sample of actual columns in DataFrame (756 total):
Found 16 adoption-related columns:
  1. preIRA_mp10_heating_adoption_upper_ap2_acs
  2. preIRA_mp10_waterHeating_adoption_upper_ap2_acs
  3. preIRA_mp10_clothesDrying_adoption_upper_ap2_acs
  4. preIRA_mp10_cooking_adoption_upper_ap2_acs
  5. preIRA_mp10_heating_adoption_upper_ap2_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp10_heating_adoption_upper_ap2_h6c', 'iraRef_mp10_heating_adoption_upper_ap2_h6c']

------------------------------------------------------------------------------------------------
RCM HEALTH SENSITIVITY: EASIUR
------------------------------------------------------------------------------------------------
SCC Climate Sensitivity: upper
Health RCM Model: easiur
Health CR Function: 'acs' or 'h6c'

Creating Multi-Index Dataframes for Space Heating - Basic (MP8), Moderate (MP9), Advanced (MP10) Retrofit ...



DEBUGGING FOR: heating (mp8, easiur, acs)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_easiur_acs
  - iraRef_mp8_heating_adoption_upper_easiur_acs

Sample of actual columns in DataFrame (725 total):
Found 16 adoption-related columns:
  1. preIRA_mp8_heating_adoption_upper_easiur_acs
  2. preIRA_mp8_waterHeating_adoption_upper_easiur_acs
  3. preIRA_mp8_clothesDrying_adoption_upper_easiur_acs
  4. preIRA_mp8_cooking_adoption_upper_easiur_acs
  5. preIRA_mp8_heating_adoption_upper_easiur_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp8_heating_adoption_upper_easiur_acs', 'iraRef_mp8_heating_adoption_upper_easiur_acs']

DEBUGGING FOR: heating (mp8, easiur, h6c)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_easiur_h6c
  - iraRef_mp8_heating_adoption_upper_easiur_h6c

Sample of actual columns in DataFrame (725 total):
Found 16 adoption-related columns:
  1. preIRA_mp8_heating_adoption_upper_easiur_acs
  2. preIRA_mp8_waterHeating_adoption_upper_easiur_acs
  3. preIRA_mp8_clothesDrying_adoption_upper_easiur_acs
  4. preIRA_mp8_cooking_adoption_upper_easiur_acs
  5. preIRA_mp8_heating_adoption_upper_easiur_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp8_heating_adoption_upper_easiur_h6c', 'iraRef_mp8_heating_adoption_upper_easiur_h6c']

DEBUGGING FOR: heating (mp9, easiur, acs)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_easiur_acs
  - iraRef_mp9_heating_adoption_upper_easiur_acs

Sample of actual columns in DataFrame (741 total):
Found 16 adoption-related columns:
  1. preIRA_mp9_heating_adoption_upper_easiur_acs
  2. preIRA_mp9_waterHeating_adoption_upper_easiur_acs
  3. preIRA_mp9_clothesDrying_adoption_upper_easiur_acs
  4. preIRA_mp9_cooking_adoption_upper_easiur_acs
  5. preIRA_mp9_heating_adoption_upper_easiur_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp9_heating_adoption_upper_easiur_acs', 'iraRef_mp9_heating_adoption_upper_easiur_acs']

DEBUGGING FOR: heating (mp9, easiur, h6c)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_easiur_h6c
  - iraRef_mp9_heating_adoption_upper_easiur_h6c

Sample of actual columns in DataFrame (741 total):
Found 16 adoption-related columns:
  1. preIRA_mp9_heating_adoption_upper_easiur_acs
  2. preIRA_mp9_waterHeating_adoption_upper_easiur_acs
  3. preIRA_mp9_clothesDrying_adoption_upper_easiur_acs
  4. preIRA_mp9_cooking_adoption_upper_easiur_acs
  5. preIRA_mp9_heating_adoption_upper_easiur_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp9_heating_adoption_upper_easiur_h6c', 'iraRef_mp9_heating_adoption_upper_easiur_h6c']

DEBUGGING FOR: heating (mp10, easiur, acs)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_easiur_acs
  - iraRef_mp10_heating_adoption_upper_easiur_acs

Sample of actual columns in DataFrame (756 total):
Found 16 adoption-related columns:
  1. preIRA_mp10_heating_adoption_upper_easiur_acs
  2. preIRA_mp10_waterHeating_adoption_upper_easiur_acs
  3. preIRA_mp10_clothesDrying_adoption_upper_easiur_acs
  4. preIRA_mp10_cooking_adoption_upper_easiur_acs
  5. preIRA_mp10_heating_adoption_upper_easiur_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp10_heating_adoption_upper_easiur_acs', 'iraRef_mp10_heating_adoption_upper_easiur_acs']

DEBUGGING FOR: heating (mp10, easiur, h6c)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_easiur_h6c
  - iraRef_mp10_heating_adoption_upper_easiur_h6c

Sample of actual columns in DataFrame (756 total):
Found 16 adoption-related columns:
  1. preIRA_mp10_heating_adoption_upper_easiur_acs
  2. preIRA_mp10_waterHeating_adoption_upper_easiur_acs
  3. preIRA_mp10_clothesDrying_adoption_upper_easiur_acs
  4. preIRA_mp10_cooking_adoption_upper_easiur_acs
  5. preIRA_mp10_heating_adoption_upper_easiur_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp10_heating_adoption_upper_easiur_h6c', 'iraRef_mp10_heating_adoption_upper_easiur_h6c']

------------------------------------------------------------------------------------------------
RCM HEALTH SENSITIVITY: InMAP
------------------------------------------------------------------------------------------------
SCC Climate Sensitivity: upper
Health RCM Model: inmap
Health CR Function: 'acs' or 'h6c'

Creating Multi-Index Dataframes for Space Heating - Basic (MP8), Moderate (MP9), Advanced (MP10) Retrofit ...



DEBUGGING FOR: heating (mp8, inmap, acs)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_inmap_acs
  - iraRef_mp8_heating_adoption_upper_inmap_acs

Sample of actual columns in DataFrame (725 total):
Found 16 adoption-related columns:
  1. preIRA_mp8_heating_adoption_upper_inmap_acs
  2. preIRA_mp8_waterHeating_adoption_upper_inmap_acs
  3. preIRA_mp8_clothesDrying_adoption_upper_inmap_acs
  4. preIRA_mp8_cooking_adoption_upper_inmap_acs
  5. preIRA_mp8_heating_adoption_upper_inmap_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp8_heating_adoption_upper_inmap_acs', 'iraRef_mp8_heating_adoption_upper_inmap_acs']

DEBUGGING FOR: heating (mp8, inmap, h6c)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_inmap_h6c
  - iraRef_mp8_heating_adoption_upper_inmap_h6c

Sample of actual columns in DataFrame (725 total):
Found 16 adoption-related columns:
  1. preIRA_mp8_heating_adoption_upper_inmap_acs
  2. preIRA_mp8_waterHeating_adoption_upper_inmap_acs
  3. preIRA_mp8_clothesDrying_adoption_upper_inmap_acs
  4. preIRA_mp8_cooking_adoption_upper_inmap_acs
  5. preIRA_mp8_heating_adoption_upper_inmap_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp8_heating_adoption_upper_inmap_h6c', 'iraRef_mp8_heating_adoption_upper_inmap_h6c']

DEBUGGING FOR: heating (mp9, inmap, acs)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_inmap_acs
  - iraRef_mp9_heating_adoption_upper_inmap_acs

Sample of actual columns in DataFrame (741 total):
Found 16 adoption-related columns:
  1. preIRA_mp9_heating_adoption_upper_inmap_acs
  2. preIRA_mp9_waterHeating_adoption_upper_inmap_acs
  3. preIRA_mp9_clothesDrying_adoption_upper_inmap_acs
  4. preIRA_mp9_cooking_adoption_upper_inmap_acs
  5. preIRA_mp9_heating_adoption_upper_inmap_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp9_heating_adoption_upper_inmap_acs', 'iraRef_mp9_heating_adoption_upper_inmap_acs']

DEBUGGING FOR: heating (mp9, inmap, h6c)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_inmap_h6c
  - iraRef_mp9_heating_adoption_upper_inmap_h6c

Sample of actual columns in DataFrame (741 total):
Found 16 adoption-related columns:
  1. preIRA_mp9_heating_adoption_upper_inmap_acs
  2. preIRA_mp9_waterHeating_adoption_upper_inmap_acs
  3. preIRA_mp9_clothesDrying_adoption_upper_inmap_acs
  4. preIRA_mp9_cooking_adoption_upper_inmap_acs
  5. preIRA_mp9_heating_adoption_upper_inmap_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp9_heating_adoption_upper_inmap_h6c', 'iraRef_mp9_heating_adoption_upper_inmap_h6c']

DEBUGGING FOR: heating (mp10, inmap, acs)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_inmap_acs
  - iraRef_mp10_heating_adoption_upper_inmap_acs

Sample of actual columns in DataFrame (756 total):
Found 16 adoption-related columns:
  1. preIRA_mp10_heating_adoption_upper_inmap_acs
  2. preIRA_mp10_waterHeating_adoption_upper_inmap_acs
  3. preIRA_mp10_clothesDrying_adoption_upper_inmap_acs
  4. preIRA_mp10_cooking_adoption_upper_inmap_acs
  5. preIRA_mp10_heating_adoption_upper_inmap_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp10_heating_adoption_upper_inmap_acs', 'iraRef_mp10_heating_adoption_upper_inmap_acs']

DEBUGGING FOR: heating (mp10, inmap, h6c)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_inmap_h6c
  - iraRef_mp10_heating_adoption_upper_inmap_h6c

Sample of actual columns in DataFrame (756 total):
Found 16 adoption-related columns:
  1. preIRA_mp10_heating_adoption_upper_inmap_acs
  2. preIRA_mp10_waterHeating_adoption_upper_inmap_acs
  3. preIRA_mp10_clothesDrying_adoption_upper_inmap_acs
  4. preIRA_mp10_cooking_adoption_upper_inmap_acs
  5. preIRA_mp10_heating_adoption_upper_inmap_h6c
  ... and 11 more

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp10_heating_adoption_upper_inmap_h6c', 'iraRef_mp10_heating_adoption_upper_inmap_h6c']