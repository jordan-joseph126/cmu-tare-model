# =======================================================================================================
# FUNCTION CALL FOR CREATE_DF_ADOPTION:
# =======================================================================================================

# ==========================================================================================
# SPACE HEATING ADOPTION POTENTIAL (MP8, MP9, MP10)
# ==========================================================================================

print(f"""
=======================================================================================================
SPACE HEATING ADOPTION POTENTIAL (MP8, MP9, MP10): HEALTH RCM-CRF SENSITIVITY 
=======================================================================================================

Creating adoption potential summary dataframes for the following measure packages and sensitivities:
- Basic Retrofit (MP8): AP2, EASIUR, InMAP
- Moderate Retrofit (MP9): AP2, EASIUR, InMAP
- Advanced Retrofit (MP10): AP2, EASIUR, InMAP
      
""")

# ========== BASIC RETROFIT (MP8) ========== 
df_basic_summary_heating_ap2 = create_df_adoption(df_outputs_basic_heating_ap2, 8, 'heating')
df_basic_summary_heating_easiur = create_df_adoption(df_outputs_basic_heating_easiur, 8, 'heating')
df_basic_summary_heating_inmap = create_df_adoption(df_outputs_basic_heating_inmap, 8, 'heating')

# ========== MODERATE RETROFIT (MP9) ========== 
df_moderate_summary_heating_ap2 = create_df_adoption(df_outputs_moderate_heating_ap2, 9, 'heating')
df_moderate_summary_heating_easiur = create_df_adoption(df_outputs_moderate_heating_easiur, 9, 'heating')
df_moderate_summary_heating_inmap = create_df_adoption(df_outputs_moderate_heating_inmap, 9, 'heating')

# ========== ADVANCED RETROFIT (MP10) ========== 
df_advanced_summary_heating_ap2 = create_df_adoption(df_outputs_advanced_heating_ap2, 10, 'heating')
df_advanced_summary_heating_easiur = create_df_adoption(df_outputs_advanced_heating_easiur, 10, 'heating')
df_advanced_summary_heating_inmap = create_df_adoption(df_outputs_advanced_heating_inmap, 10, 'heating')

# =======================================================================================================
# FUNCTION CALL FOR MULTI-INDEX DF CREATION:
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
df_mi_basic_heating_adoption_ap2_acs = create_multiIndex_adoption_df(df_basic_summary_heating_ap2, 8, 'heating', scc, rcm_model, 'acs')
df_mi_basic_heating_adoption_ap2_h6c = create_multiIndex_adoption_df(df_basic_summary_heating_ap2, 8, 'heating', scc, rcm_model, 'h6c')
# Moderate Retrofit (MP9)
df_mi_moderate_heating_adoption_ap2_acs = create_multiIndex_adoption_df(df_moderate_summary_heating_ap2, 9, 'heating', scc, rcm_model, 'acs')
df_mi_moderate_heating_adoption_ap2_h6c = create_multiIndex_adoption_df(df_moderate_summary_heating_ap2, 9, 'heating', scc, rcm_model, 'h6c')
# Advanced Retrofit (MP10)
df_mi_advanced_heating_adoption_ap2_acs = create_multiIndex_adoption_df(df_advanced_summary_heating_ap2, 10, 'heating', scc, rcm_model, 'acs')
df_mi_advanced_heating_adoption_ap2_h6c = create_multiIndex_adoption_df(df_advanced_summary_heating_ap2, 10, 'heating', scc, rcm_model, 'h6c')

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
df_mi_basic_heating_adoption_easiur_acs = create_multiIndex_adoption_df(df_basic_summary_heating_easiur, 8, 'heating', scc, rcm_model, 'acs')
df_mi_basic_heating_adoption_easiur_h6c = create_multiIndex_adoption_df(df_basic_summary_heating_easiur, 8, 'heating', scc, rcm_model, 'h6c')
# Moderate Retrofit (MP9)
df_mi_moderate_heating_adoption_easiur_acs = create_multiIndex_adoption_df(df_moderate_summary_heating_easiur, 9, 'heating', scc, rcm_model, 'acs')
df_mi_moderate_heating_adoption_easiur_h6c = create_multiIndex_adoption_df(df_moderate_summary_heating_easiur, 9, 'heating', scc, rcm_model, 'h6c')
# Advanced Retrofit (MP10)
df_mi_advanced_heating_adoption_easiur_acs = create_multiIndex_adoption_df(df_advanced_summary_heating_easiur, 10, 'heating', scc, rcm_model, 'acs')
df_mi_advanced_heating_adoption_easiur_h6c = create_multiIndex_adoption_df(df_advanced_summary_heating_easiur, 10, 'heating', scc, rcm_model, 'h6c')

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
df_mi_basic_heating_adoption_inmap_acs = create_multiIndex_adoption_df(df_basic_summary_heating_inmap, 8, 'heating', scc, rcm_model, 'acs')
df_mi_basic_heating_adoption_inmap_h6c = create_multiIndex_adoption_df(df_basic_summary_heating_inmap, 8, 'heating', scc, rcm_model, 'h6c')
# Moderate Retrofit (MP9)
df_mi_moderate_heating_adoption_inmap_acs = create_multiIndex_adoption_df(df_moderate_summary_heating_inmap, 9, 'heating', scc, rcm_model, 'acs')
df_mi_moderate_heating_adoption_inmap_h6c = create_multiIndex_adoption_df(df_moderate_summary_heating_inmap, 9, 'heating', scc, rcm_model, 'h6c')
# Advanced Retrofit (MP10)
df_mi_advanced_heating_adoption_inmap_acs = create_multiIndex_adoption_df(df_advanced_summary_heating_inmap, 10, 'heating', scc, rcm_model, 'acs')
df_mi_advanced_heating_adoption_inmap_h6c = create_multiIndex_adoption_df(df_advanced_summary_heating_inmap, 10, 'heating', scc, rcm_model, 'h6c')


# =======================================================================================================
# OUTPUT AFTER BOTH:
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

Sample of actual columns in DataFrame (28 total):
Found 2 adoption-related columns:
  1. preIRA_mp8_heating_adoption_upper_ap2_acs
  2. iraRef_mp8_heating_adoption_upper_ap2_acs

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp8_heating_adoption_upper_ap2_acs', 'iraRef_mp8_heating_adoption_upper_ap2_acs']

DEBUGGING FOR: heating (mp8, ap2, h6c)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_ap2_h6c
  - iraRef_mp8_heating_adoption_upper_ap2_h6c

Sample of actual columns in DataFrame (28 total):
Found 2 adoption-related columns:
  1. preIRA_mp8_heating_adoption_upper_ap2_acs
  2. iraRef_mp8_heating_adoption_upper_ap2_acs

Checking backward compatibility columns:
  - preIRA_mp8_heating_adoption_lrmer
  - iraRef_mp8_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp8_heating_adoption
  - iraRef_mp8_heating_adoption

Found adoption columns for heating with different pattern:
  1. preIRA_mp8_heating_adoption_upper_ap2_acs
  2. iraRef_mp8_heating_adoption_upper_ap2_acs

Unable to find expected column pattern. Please ensure columns follow the pattern:
  'preIRA_mp{menu_mp}_heating_adoption_{scc}_{rcm_model}_{cr_function}'
  or backup pattern: 'preIRA_mp{menu_mp}_heating_adoption_lrmer'
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp9, ap2, acs)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_ap2_acs
  - iraRef_mp9_heating_adoption_upper_ap2_acs

Sample of actual columns in DataFrame (28 total):
Found 2 adoption-related columns:
  1. preIRA_mp9_heating_adoption_upper_ap2_acs
  2. iraRef_mp9_heating_adoption_upper_ap2_acs

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp9_heating_adoption_upper_ap2_acs', 'iraRef_mp9_heating_adoption_upper_ap2_acs']

DEBUGGING FOR: heating (mp9, ap2, h6c)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_ap2_h6c
  - iraRef_mp9_heating_adoption_upper_ap2_h6c

Sample of actual columns in DataFrame (28 total):
Found 2 adoption-related columns:
  1. preIRA_mp9_heating_adoption_upper_ap2_acs
  2. iraRef_mp9_heating_adoption_upper_ap2_acs

Checking backward compatibility columns:
  - preIRA_mp9_heating_adoption_lrmer
  - iraRef_mp9_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp9_heating_adoption
  - iraRef_mp9_heating_adoption

Found adoption columns for heating with different pattern:
  1. preIRA_mp9_heating_adoption_upper_ap2_acs
  2. iraRef_mp9_heating_adoption_upper_ap2_acs

Unable to find expected column pattern. Please ensure columns follow the pattern:
  'preIRA_mp{menu_mp}_heating_adoption_{scc}_{rcm_model}_{cr_function}'
  or backup pattern: 'preIRA_mp{menu_mp}_heating_adoption_lrmer'
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp10, ap2, acs)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_ap2_acs
  - iraRef_mp10_heating_adoption_upper_ap2_acs

Sample of actual columns in DataFrame (28 total):
Found 2 adoption-related columns:
  1. preIRA_mp10_heating_adoption_upper_ap2_acs
  2. iraRef_mp10_heating_adoption_upper_ap2_acs

Grouping by: base_heating_fuel and lowModerateIncome_designation
Using adoption columns: ['preIRA_mp10_heating_adoption_upper_ap2_acs', 'iraRef_mp10_heating_adoption_upper_ap2_acs']

DEBUGGING FOR: heating (mp10, ap2, h6c)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_ap2_h6c
  - iraRef_mp10_heating_adoption_upper_ap2_h6c

Sample of actual columns in DataFrame (28 total):
Found 2 adoption-related columns:
  1. preIRA_mp10_heating_adoption_upper_ap2_acs
  2. iraRef_mp10_heating_adoption_upper_ap2_acs

Checking backward compatibility columns:
  - preIRA_mp10_heating_adoption_lrmer
  - iraRef_mp10_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp10_heating_adoption
  - iraRef_mp10_heating_adoption

Found adoption columns for heating with different pattern:
  1. preIRA_mp10_heating_adoption_upper_ap2_acs
  2. iraRef_mp10_heating_adoption_upper_ap2_acs

Unable to find expected column pattern. Please ensure columns follow the pattern:
  'preIRA_mp{menu_mp}_heating_adoption_{scc}_{rcm_model}_{cr_function}'
  or backup pattern: 'preIRA_mp{menu_mp}_heating_adoption_lrmer'
Error: Required adoption columns not found for heating

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

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp8_heating_adoption_lrmer
  - iraRef_mp8_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp8_heating_adoption
  - iraRef_mp8_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp8, easiur, h6c)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_easiur_h6c
  - iraRef_mp8_heating_adoption_upper_easiur_h6c

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp8_heating_adoption_lrmer
  - iraRef_mp8_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp8_heating_adoption
  - iraRef_mp8_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp9, easiur, acs)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_easiur_acs
  - iraRef_mp9_heating_adoption_upper_easiur_acs

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp9_heating_adoption_lrmer
  - iraRef_mp9_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp9_heating_adoption
  - iraRef_mp9_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp9, easiur, h6c)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_easiur_h6c
  - iraRef_mp9_heating_adoption_upper_easiur_h6c

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp9_heating_adoption_lrmer
  - iraRef_mp9_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp9_heating_adoption
  - iraRef_mp9_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp10, easiur, acs)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_easiur_acs
  - iraRef_mp10_heating_adoption_upper_easiur_acs

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp10_heating_adoption_lrmer
  - iraRef_mp10_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp10_heating_adoption
  - iraRef_mp10_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp10, easiur, h6c)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_easiur_h6c
  - iraRef_mp10_heating_adoption_upper_easiur_h6c

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp10_heating_adoption_lrmer
  - iraRef_mp10_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp10_heating_adoption
  - iraRef_mp10_heating_adoption
Error: Required adoption columns not found for heating

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

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp8_heating_adoption_lrmer
  - iraRef_mp8_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp8_heating_adoption
  - iraRef_mp8_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp8, inmap, h6c)
Looking for columns:
  - preIRA_mp8_heating_adoption_upper_inmap_h6c
  - iraRef_mp8_heating_adoption_upper_inmap_h6c

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp8_heating_adoption_lrmer
  - iraRef_mp8_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp8_heating_adoption
  - iraRef_mp8_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp9, inmap, acs)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_inmap_acs
  - iraRef_mp9_heating_adoption_upper_inmap_acs

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp9_heating_adoption_lrmer
  - iraRef_mp9_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp9_heating_adoption
  - iraRef_mp9_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp9, inmap, h6c)
Looking for columns:
  - preIRA_mp9_heating_adoption_upper_inmap_h6c
  - iraRef_mp9_heating_adoption_upper_inmap_h6c

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp9_heating_adoption_lrmer
  - iraRef_mp9_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp9_heating_adoption
  - iraRef_mp9_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp10, inmap, acs)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_inmap_acs
  - iraRef_mp10_heating_adoption_upper_inmap_acs

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp10_heating_adoption_lrmer
  - iraRef_mp10_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp10_heating_adoption
  - iraRef_mp10_heating_adoption
Error: Required adoption columns not found for heating

DEBUGGING FOR: heating (mp10, inmap, h6c)
Looking for columns:
  - preIRA_mp10_heating_adoption_upper_inmap_h6c
  - iraRef_mp10_heating_adoption_upper_inmap_h6c

Sample of actual columns in DataFrame (19 total):
  No adoption-related columns found!

Sample of available columns:
  1. state
  2. city
  3. county
  4. puma
  5. percent_AMI

Checking backward compatibility columns:
  - preIRA_mp10_heating_adoption_lrmer
  - iraRef_mp10_heating_adoption_lrmer

Checking alternative columns (without sensitivity parameters):
  - preIRA_mp10_heating_adoption
  - iraRef_mp10_heating_adoption
Error: Required adoption columns not found for heating
