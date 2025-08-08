#FUNCTION CALL AND SAMPLE OUTPUT FOR ADOPTION DECISIONS
##NO IRA/PRE IRA
**FUNCTION CALL**
'''
from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import *

# def adoption_decision(df: pd.DataFrame,
#                       menu_mp: int,
#                       policy_scenario: str,
#                       rcm_model: str,
#                       cr_function: str,
#                       climate_sensitivity: bool = False  # Default is false because we use $190USD2020/mt in Joseph et al. (2025)
# ) -> pd.DataFrame:

# ============= AP2 =============
df_euss_am_mp8_home_ap2 = adoption_decision(
    df=df_euss_am_mp8_home_ap2,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='AP2',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp8_home_ap2 = adoption_decision(
    df=df_euss_am_mp8_home_ap2,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='AP2',
    cr_function='h6c',
    climate_sensitivity=False
    )

# ============ EASIUR =============
df_euss_am_mp8_home_easiur = adoption_decision(
    df=df_euss_am_mp8_home_easiur,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='EASIUR',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp8_home_easiur = adoption_decision(
    df=df_euss_am_mp8_home_easiur,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='EASIUR',
    cr_function='h6c',
    climate_sensitivity=False
    )

# ============ InMAP =============
df_euss_am_mp8_home_inmap = adoption_decision(
    df=df_euss_am_mp8_home_inmap,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='InMAP',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp8_home_inmap = adoption_decision(
    df=df_euss_am_mp8_home_inmap,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='InMAP',
    cr_function='h6c',
    climate_sensitivity=False
    )


print(f"""
====================================================================================================================================================================
ADOPTION FEASIBILITY OF VARIOUS RETROFITS: NO INFLATION REDUCTION ACT
====================================================================================================================================================================
determine_adoption_potential.py file contains the definition for the adoption_decision function.

DATAFRAME FOR MP8 AFTER DETERMINING ADOPTION FEASIBILITY: df_euss_am_mp8_home
      
AP2: 
-----------------------------------------------
{df_euss_am_mp8_home_ap2}

EASIUR:
-----------------------------------------------
{df_euss_am_mp8_home_easiur}

InMAP:
-----------------------------------------------
{df_euss_am_mp8_home_inmap}
      
""")
'''


**OUTPUT**
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Calculating Adoption Potential for heating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: acs

Calculating Adoption Potential for waterHeating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: acs

Calculating Adoption Potential for clothesDrying under 'No Inflation Reduction Act' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: acs

Calculating Adoption Potential for cooking under 'No Inflation Reduction Act' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: acs

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    preIRA_mp8_cooking_health_sensitivity: Masked 7784 values
    preIRA_mp8_cooking_adoption_upper_AP2_acs: Masked 7784 values
    preIRA_mp8_cooking_impact_upper_AP2_acs: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Calculating Adoption Potential for heating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: h6c

Calculating Adoption Potential for waterHeating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: h6c

Calculating Adoption Potential for clothesDrying under 'No Inflation Reduction Act' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: h6c

Calculating Adoption Potential for cooking under 'No Inflation Reduction Act' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: h6c

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    preIRA_mp8_cooking_health_sensitivity: Masked 7784 values
    preIRA_mp8_cooking_adoption_upper_AP2_h6c: Masked 7784 values
    preIRA_mp8_cooking_impact_upper_AP2_h6c: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Calculating Adoption Potential for heating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: acs

Calculating Adoption Potential for waterHeating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: acs

Calculating Adoption Potential for clothesDrying under 'No Inflation Reduction Act' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: acs

Calculating Adoption Potential for cooking under 'No Inflation Reduction Act' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: acs

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    preIRA_mp8_cooking_health_sensitivity: Masked 7784 values
    preIRA_mp8_cooking_adoption_upper_EASIUR_acs: Masked 7784 values
    preIRA_mp8_cooking_impact_upper_EASIUR_acs: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Calculating Adoption Potential for heating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: h6c

Calculating Adoption Potential for waterHeating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: h6c

Calculating Adoption Potential for clothesDrying under 'No Inflation Reduction Act' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: h6c

Calculating Adoption Potential for cooking under 'No Inflation Reduction Act' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: h6c

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    preIRA_mp8_cooking_health_sensitivity: Masked 7784 values
    preIRA_mp8_cooking_adoption_upper_EASIUR_h6c: Masked 7784 values
    preIRA_mp8_cooking_impact_upper_EASIUR_h6c: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Calculating Adoption Potential for heating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: acs

Calculating Adoption Potential for waterHeating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: acs

Calculating Adoption Potential for clothesDrying under 'No Inflation Reduction Act' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: acs

Calculating Adoption Potential for cooking under 'No Inflation Reduction Act' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: acs

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    preIRA_mp8_cooking_health_sensitivity: Masked 7784 values
    preIRA_mp8_cooking_adoption_upper_InMAP_acs: Masked 7784 values
    preIRA_mp8_cooking_impact_upper_InMAP_acs: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Calculating Adoption Potential for heating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: h6c

Calculating Adoption Potential for waterHeating under 'No Inflation Reduction Act' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: h6c

Calculating Adoption Potential for clothesDrying under 'No Inflation Reduction Act' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: h6c

Calculating Adoption Potential for cooking under 'No Inflation Reduction Act' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: h6c

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    preIRA_mp8_cooking_health_sensitivity: Masked 7784 values
    preIRA_mp8_cooking_adoption_upper_InMAP_h6c: Masked 7784 values
    preIRA_mp8_cooking_impact_upper_InMAP_h6c: Masked 7784 values
  Total: Masked 23352 values across 12 columns

====================================================================================================================================================================
ADOPTION FEASIBILITY OF VARIOUS RETROFITS: NO INFLATION REDUCTION ACT
====================================================================================================================================================================
determine_adoption_potential.py file contains the definition for the adoption_decision function.

DATAFRAME FOR MP8 AFTER DETERMINING ADOPTION FEASIBILITY: df_euss_am_mp8_home
      
AP2: 
-----------------------------------------------
         square_footage census_region  census_division census_division_recs  \
bldg_id                                                                       
119              2152.0     Northeast  Middle Atlantic      Middle Atlantic   
122              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
150              1690.0     Northeast  Middle Atlantic      Middle Atlantic   
153              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
162              2663.0     Northeast  Middle Atlantic      Middle Atlantic   
...                 ...           ...              ...                  ...   
549882           1202.0     Northeast  Middle Atlantic      Middle Atlantic   
549915           2176.0     Northeast  Middle Atlantic      Middle Atlantic   
549937            885.0     Northeast  Middle Atlantic      Middle Atlantic   
549963           1690.0     Northeast  Middle Atlantic      Middle Atlantic   
549989           1220.0     Northeast  Middle Atlantic      Middle Atlantic   

        building_america_climate_zone  reeds_balancing_area gea_region state  \
bldg_id                                                                        
119                       Mixed-Humid                   122      RFCEc    PA   
122                       Mixed-Humid                   122      RFCEc    PA   
150                              Cold                   122      RFCEc    PA   
153                       Mixed-Humid                   122      RFCEc    PA   
162                       Mixed-Humid                   122      RFCEc    PA   
...                               ...                   ...        ...   ...   
549882                           Cold                   122      RFCEc    PA   
549915                           Cold                   122      RFCEc    PA   
549937                           Cold                   115      RFCWc    PA   
549963                           Cold                   115      RFCWc    PA   
549989                           Cold                   122      RFCEc    PA   

                            city    county  ...  \
bldg_id                                     ...   
119        Not in a census Place  G4200450  ...   
122      In another census Place  G4200450  ...   
150      In another census Place  G4201190  ...   
153        Not in a census Place  G4200170  ...   
162        Not in a census Place  G4200450  ...   
...                          ...       ...  ...   
549882     Not in a census Place  G4200110  ...   
549915                 Lancaster  G4200710  ...   
549937   In another census Place  G4200050  ...   
549963   In another census Place  G4201290  ...   
549989     Not in a census Place  G4201110  ...   

        preIRA_mp8_clothesDrying_total_npv_lessWTP_upper_AP2_h6c  \
bldg_id                                                            
119                                               -2031.16         
122                                               -2065.16         
150                                               -2244.78         
153                                                    NaN         
162                                               -1071.48         
...                                                    ...         
549882                                            -1502.16         
549915                                             -576.05         
549937                                            -2392.79         
549963                                            -2175.75         
549989                                            -2904.96         

        preIRA_mp8_clothesDrying_total_npv_moreWTP_upper_AP2_h6c  \
bldg_id                                                            
119                                                -723.18         
122                                               -1071.10         
150                                               -1250.72         
153                                                    NaN         
162                                                 -77.42         
...                                                    ...         
549882                                             -508.10         
549915                                              418.01         
549937                                            -1084.81         
549963                                            -1181.69         
549989                                            -1910.90         

        preIRA_mp8_clothesDrying_adoption_upper_AP2_h6c  \
bldg_id                                                   
119                                      Tier 4: Averse   
122                                      Tier 4: Averse   
150                                      Tier 4: Averse   
153                     N/A: Invalid Baseline Fuel/Tech   
162                                      Tier 4: Averse   
...                                                 ...   
549882                                   Tier 4: Averse   
549915            Tier 3: Subsidy-Dependent Feasibility   
549937                                   Tier 4: Averse   
549963                                   Tier 4: Averse   
549989                                   Tier 4: Averse   

        preIRA_mp8_clothesDrying_impact_upper_AP2_h6c  \
bldg_id                                                 
119                                  Public Detriment   
122                                    Public Benefit   
150                                    Public Benefit   
153                   N/A: Invalid Baseline Fuel/Tech   
162                                    Public Benefit   
...                                               ...   
549882                                 Public Benefit   
549915                                 Public Benefit   
549937                                 Public Benefit   
549963                                 Public Benefit   
549989                                 Public Benefit   

         preIRA_mp8_cooking_health_sensitivity  \
bldg_id                                          
119                                   AP2, h6c   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                AP2, h6c   
549963                                     NaN   
549989                                AP2, h6c   

         preIRA_mp8_cooking_benefit_upper_AP2_h6c  \
bldg_id                                             
119                                           0.0   
122                                           NaN   
150                                           NaN   
153                                           NaN   
162                                           NaN   
...                                           ...   
549882                                        NaN   
549915                                        NaN   
549937                                        0.0   
549963                                        NaN   
549989                                        0.0   

        preIRA_mp8_cooking_total_npv_lessWTP_upper_AP2_h6c  \
bldg_id                                                      
119                                               -1899.18   
122                                                    NaN   
150                                                    NaN   
153                                                    NaN   
162                                                    NaN   
...                                                    ...   
549882                                                 NaN   
549915                                                 NaN   
549937                                            -2124.47   
549963                                                 NaN   
549989                                            -1356.82   

         preIRA_mp8_cooking_total_npv_moreWTP_upper_AP2_h6c  \
bldg_id                                                       
119                                                -826.64    
122                                                    NaN    
150                                                    NaN    
153                                                    NaN    
162                                                    NaN    
...                                                    ...    
549882                                                 NaN    
549915                                                 NaN    
549937                                            -1051.93    
549963                                                 NaN    
549989                                             -284.28    

        preIRA_mp8_cooking_adoption_upper_AP2_h6c  \
bldg_id                                             
119                                Tier 4: Averse   
122                                           NaN   
150                                           NaN   
153                                           NaN   
162                                           NaN   
...                                           ...   
549882                                        NaN   
549915                                        NaN   
549937                             Tier 4: Averse   
549963                                        NaN   
549989                             Tier 4: Averse   

        preIRA_mp8_cooking_impact_upper_AP2_h6c  
bldg_id                                          
119                            Public Detriment  
122                                         NaN  
150                                         NaN  
153                                         NaN  
162                                         NaN  
...                                         ...  
549882                                      NaN  
549915                                      NaN  
549937                         Public Detriment  
549963                                      NaN  
549989                         Public Detriment  

[15651 rows x 641 columns]

EASIUR:
-----------------------------------------------
         square_footage census_region  census_division census_division_recs  \
bldg_id                                                                       
119              2152.0     Northeast  Middle Atlantic      Middle Atlantic   
122              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
150              1690.0     Northeast  Middle Atlantic      Middle Atlantic   
153              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
162              2663.0     Northeast  Middle Atlantic      Middle Atlantic   
...                 ...           ...              ...                  ...   
549882           1202.0     Northeast  Middle Atlantic      Middle Atlantic   
549915           2176.0     Northeast  Middle Atlantic      Middle Atlantic   
549937            885.0     Northeast  Middle Atlantic      Middle Atlantic   
549963           1690.0     Northeast  Middle Atlantic      Middle Atlantic   
549989           1220.0     Northeast  Middle Atlantic      Middle Atlantic   

        building_america_climate_zone  reeds_balancing_area gea_region state  \
bldg_id                                                                        
119                       Mixed-Humid                   122      RFCEc    PA   
122                       Mixed-Humid                   122      RFCEc    PA   
150                              Cold                   122      RFCEc    PA   
153                       Mixed-Humid                   122      RFCEc    PA   
162                       Mixed-Humid                   122      RFCEc    PA   
...                               ...                   ...        ...   ...   
549882                           Cold                   122      RFCEc    PA   
549915                           Cold                   122      RFCEc    PA   
549937                           Cold                   115      RFCWc    PA   
549963                           Cold                   115      RFCWc    PA   
549989                           Cold                   122      RFCEc    PA   

                            city    county  ...  \
bldg_id                                     ...   
119        Not in a census Place  G4200450  ...   
122      In another census Place  G4200450  ...   
150      In another census Place  G4201190  ...   
153        Not in a census Place  G4200170  ...   
162        Not in a census Place  G4200450  ...   
...                          ...       ...  ...   
549882     Not in a census Place  G4200110  ...   
549915                 Lancaster  G4200710  ...   
549937   In another census Place  G4200050  ...   
549963   In another census Place  G4201290  ...   
549989     Not in a census Place  G4201110  ...   

        preIRA_mp8_clothesDrying_total_npv_lessWTP_upper_EASIUR_h6c  \
bldg_id                                                               
119                                               -1928.96            
122                                               -2048.64            
150                                               -2204.08            
153                                                    NaN            
162                                               -1012.05            
...                                                    ...            
549882                                            -1476.11            
549915                                             -524.31            
549937                                            -2304.81            
549963                                            -2214.89            
549989                                            -2913.42            

        preIRA_mp8_clothesDrying_total_npv_moreWTP_upper_EASIUR_h6c  \
bldg_id                                                               
119                                                -620.98            
122                                               -1054.58            
150                                               -1210.02            
153                                                    NaN            
162                                                 -17.99            
...                                                    ...            
549882                                             -482.05            
549915                                              469.75            
549937                                             -996.83            
549963                                            -1220.83            
549989                                            -1919.36            

        preIRA_mp8_clothesDrying_adoption_upper_EASIUR_h6c  \
bldg_id                                                      
119                                         Tier 4: Averse   
122                                         Tier 4: Averse   
150                                         Tier 4: Averse   
153                        N/A: Invalid Baseline Fuel/Tech   
162                                         Tier 4: Averse   
...                                                    ...   
549882                                      Tier 4: Averse   
549915               Tier 3: Subsidy-Dependent Feasibility   
549937                                      Tier 4: Averse   
549963                                      Tier 4: Averse   
549989                                      Tier 4: Averse   

        preIRA_mp8_clothesDrying_impact_upper_EASIUR_h6c  \
bldg_id                                                    
119                                       Public Benefit   
122                                       Public Benefit   
150                                       Public Benefit   
153                      N/A: Invalid Baseline Fuel/Tech   
162                                       Public Benefit   
...                                                  ...   
549882                                    Public Benefit   
549915                                    Public Benefit   
549937                                    Public Benefit   
549963                                    Public Benefit   
549989                                    Public Benefit   

         preIRA_mp8_cooking_health_sensitivity  \
bldg_id                                          
119                                EASIUR, h6c   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                             EASIUR, h6c   
549963                                     NaN   
549989                             EASIUR, h6c   

         preIRA_mp8_cooking_benefit_upper_EASIUR_h6c  \
bldg_id                                                
119                                              0.0   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                           0.0   
549963                                           NaN   
549989                                           0.0   

        preIRA_mp8_cooking_total_npv_lessWTP_upper_EASIUR_h6c  \
bldg_id                                                         
119                                               -1736.85      
122                                                    NaN      
150                                                    NaN      
153                                                    NaN      
162                                                    NaN      
...                                                    ...      
549882                                                 NaN      
549915                                                 NaN      
549937                                            -2061.50      
549963                                                 NaN      
549989                                            -1250.34      

         preIRA_mp8_cooking_total_npv_moreWTP_upper_EASIUR_h6c  \
bldg_id                                                          
119                                                -664.31       
122                                                    NaN       
150                                                    NaN       
153                                                    NaN       
162                                                    NaN       
...                                                    ...       
549882                                                 NaN       
549915                                                 NaN       
549937                                             -988.96       
549963                                                 NaN       
549989                                             -177.80       

        preIRA_mp8_cooking_adoption_upper_EASIUR_h6c  \
bldg_id                                                
119                                   Tier 4: Averse   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                Tier 4: Averse   
549963                                           NaN   
549989                                Tier 4: Averse   

        preIRA_mp8_cooking_impact_upper_EASIUR_h6c  
bldg_id                                             
119                               Public Detriment  
122                                            NaN  
150                                            NaN  
153                                            NaN  
162                                            NaN  
...                                            ...  
549882                                         NaN  
549915                                         NaN  
549937                            Public Detriment  
549963                                         NaN  
549989                              Public Benefit  

[15651 rows x 641 columns]

InMAP:
-----------------------------------------------
         square_footage census_region  census_division census_division_recs  \
bldg_id                                                                       
119              2152.0     Northeast  Middle Atlantic      Middle Atlantic   
122              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
150              1690.0     Northeast  Middle Atlantic      Middle Atlantic   
153              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
162              2663.0     Northeast  Middle Atlantic      Middle Atlantic   
...                 ...           ...              ...                  ...   
549882           1202.0     Northeast  Middle Atlantic      Middle Atlantic   
549915           2176.0     Northeast  Middle Atlantic      Middle Atlantic   
549937            885.0     Northeast  Middle Atlantic      Middle Atlantic   
549963           1690.0     Northeast  Middle Atlantic      Middle Atlantic   
549989           1220.0     Northeast  Middle Atlantic      Middle Atlantic   

        building_america_climate_zone  reeds_balancing_area gea_region state  \
bldg_id                                                                        
119                       Mixed-Humid                   122      RFCEc    PA   
122                       Mixed-Humid                   122      RFCEc    PA   
150                              Cold                   122      RFCEc    PA   
153                       Mixed-Humid                   122      RFCEc    PA   
162                       Mixed-Humid                   122      RFCEc    PA   
...                               ...                   ...        ...   ...   
549882                           Cold                   122      RFCEc    PA   
549915                           Cold                   122      RFCEc    PA   
549937                           Cold                   115      RFCWc    PA   
549963                           Cold                   115      RFCWc    PA   
549989                           Cold                   122      RFCEc    PA   

                            city    county  ...  \
bldg_id                                     ...   
119        Not in a census Place  G4200450  ...   
122      In another census Place  G4200450  ...   
150      In another census Place  G4201190  ...   
153        Not in a census Place  G4200170  ...   
162        Not in a census Place  G4200450  ...   
...                          ...       ...  ...   
549882     Not in a census Place  G4200110  ...   
549915                 Lancaster  G4200710  ...   
549937   In another census Place  G4200050  ...   
549963   In another census Place  G4201290  ...   
549989     Not in a census Place  G4201110  ...   

        preIRA_mp8_clothesDrying_total_npv_lessWTP_upper_InMAP_h6c  \
bldg_id                                                              
119                                               -1822.35           
122                                               -2091.94           
150                                               -2137.90           
153                                                    NaN           
162                                               -1167.61           
...                                                    ...           
549882                                            -1384.98           
549915                                             -484.38           
549937                                            -2375.18           
549963                                            -2205.40           
549989                                            -2878.10           

        preIRA_mp8_clothesDrying_total_npv_moreWTP_upper_InMAP_h6c  \
bldg_id                                                              
119                                                -514.37           
122                                               -1097.88           
150                                               -1143.84           
153                                                    NaN           
162                                                -173.55           
...                                                    ...           
549882                                             -390.92           
549915                                              509.68           
549937                                            -1067.20           
549963                                            -1211.34           
549989                                            -1884.04           

        preIRA_mp8_clothesDrying_adoption_upper_InMAP_h6c  \
bldg_id                                                     
119                                        Tier 4: Averse   
122                                        Tier 4: Averse   
150                                        Tier 4: Averse   
153                       N/A: Invalid Baseline Fuel/Tech   
162                                        Tier 4: Averse   
...                                                   ...   
549882                                     Tier 4: Averse   
549915              Tier 3: Subsidy-Dependent Feasibility   
549937                                     Tier 4: Averse   
549963                                     Tier 4: Averse   
549989                                     Tier 4: Averse   

        preIRA_mp8_clothesDrying_impact_upper_InMAP_h6c  \
bldg_id                                                   
119                                      Public Benefit   
122                                      Public Benefit   
150                                      Public Benefit   
153                     N/A: Invalid Baseline Fuel/Tech   
162                                      Public Benefit   
...                                                 ...   
549882                                   Public Benefit   
549915                                   Public Benefit   
549937                                   Public Benefit   
549963                                   Public Benefit   
549989                                   Public Benefit   

         preIRA_mp8_cooking_health_sensitivity  \
bldg_id                                          
119                                 InMAP, h6c   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                              InMAP, h6c   
549963                                     NaN   
549989                              InMAP, h6c   

         preIRA_mp8_cooking_benefit_upper_InMAP_h6c  \
bldg_id                                               
119                                             0.0   
122                                             NaN   
150                                             NaN   
153                                             NaN   
162                                             NaN   
...                                             ...   
549882                                          NaN   
549915                                          NaN   
549937                                          0.0   
549963                                          NaN   
549989                                          0.0   

        preIRA_mp8_cooking_total_npv_lessWTP_upper_InMAP_h6c  \
bldg_id                                                        
119                                               -1519.31     
122                                                    NaN     
150                                                    NaN     
153                                                    NaN     
162                                                    NaN     
...                                                    ...     
549882                                                 NaN     
549915                                                 NaN     
549937                                            -2120.56     
549963                                                 NaN     
549989                                            -1335.27     

         preIRA_mp8_cooking_total_npv_moreWTP_upper_InMAP_h6c  \
bldg_id                                                         
119                                                -446.77      
122                                                    NaN      
150                                                    NaN      
153                                                    NaN      
162                                                    NaN      
...                                                    ...      
549882                                                 NaN      
549915                                                 NaN      
549937                                            -1048.02      
549963                                                 NaN      
549989                                             -262.73      

        preIRA_mp8_cooking_adoption_upper_InMAP_h6c  \
bldg_id                                               
119                                  Tier 4: Averse   
122                                             NaN   
150                                             NaN   
153                                             NaN   
162                                             NaN   
...                                             ...   
549882                                          NaN   
549915                                          NaN   
549937                               Tier 4: Averse   
549963                                          NaN   
549989                               Tier 4: Averse   

        preIRA_mp8_cooking_impact_upper_InMAP_h6c  
bldg_id                                            
119                                Public Benefit  
122                                           NaN  
150                                           NaN  
153                                           NaN  
162                                           NaN  
...                                           ...  
549882                                        NaN  
549915                                        NaN  
549937                           Public Detriment  
549963                                        NaN  
549989                           Public Detriment  

[15651 rows x 641 columns]


##IRA REFERENCE
**FUNCTION CALL**
'''

# from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import *

# def adoption_decision(df: pd.DataFrame,
#                       menu_mp: int,
#                       policy_scenario: str,
#                       rcm_model: str,
#                       cr_function: str,
#                       climate_sensitivity: bool = False  # Default is false because we use $190USD2020/mt in Joseph et al. (2025)
# ) -> pd.DataFrame:

# ============= AP2 =============
df_euss_am_mp8_home_ap2 = adoption_decision(
    df=df_euss_am_mp8_home_ap2,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='AP2',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp8_home_ap2 = adoption_decision(
    df=df_euss_am_mp8_home_ap2,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='AP2',
    cr_function='h6c',
    climate_sensitivity=False
    )

# ============ EASIUR =============
df_euss_am_mp8_home_easiur = adoption_decision(
    df=df_euss_am_mp8_home_easiur,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='EASIUR',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp8_home_easiur = adoption_decision(
    df=df_euss_am_mp8_home_easiur,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='EASIUR',
    cr_function='h6c',
    climate_sensitivity=False
    )

# ============ InMAP =============
df_euss_am_mp8_home_inmap = adoption_decision(
    df=df_euss_am_mp8_home_inmap,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='InMAP',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp8_home_inmap = adoption_decision(
    df=df_euss_am_mp8_home_inmap,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='InMAP',
    cr_function='h6c',
    climate_sensitivity=False
    )


print(f"""
====================================================================================================================================================================
ADOPTION FEASIBILITY OF VARIOUS RETROFITS: AEO2023 Reference Case
====================================================================================================================================================================
determine_adoption_potential.py file contains the definition for the adoption_decision function.

DATAFRAME FOR MP8 AFTER DETERMINING ADOPTION FEASIBILITY: df_euss_am_mp8_home
      
AP2: 
-----------------------------------------------
{df_euss_am_mp8_home_ap2}

EASIUR:
-----------------------------------------------
{df_euss_am_mp8_home_easiur}

InMAP:
-----------------------------------------------
{df_euss_am_mp8_home_inmap}
      
""")

'''



**OUTPUT**


-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Calculating Adoption Potential for heating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: acs

Calculating Adoption Potential for waterHeating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: acs

Calculating Adoption Potential for clothesDrying under 'AEO2023 Reference Case' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: acs

Calculating Adoption Potential for cooking under 'AEO2023 Reference Case' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: acs

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    iraRef_mp8_cooking_health_sensitivity: Masked 7784 values
    iraRef_mp8_cooking_adoption_upper_AP2_acs: Masked 7784 values
    iraRef_mp8_cooking_impact_upper_AP2_acs: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Calculating Adoption Potential for heating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: h6c

Calculating Adoption Potential for waterHeating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: h6c

Calculating Adoption Potential for clothesDrying under 'AEO2023 Reference Case' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: h6c

Calculating Adoption Potential for cooking under 'AEO2023 Reference Case' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: AP2 | cr_function Function: h6c

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    iraRef_mp8_cooking_health_sensitivity: Masked 7784 values
    iraRef_mp8_cooking_adoption_upper_AP2_h6c: Masked 7784 values
    iraRef_mp8_cooking_impact_upper_AP2_h6c: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Calculating Adoption Potential for heating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: acs

Calculating Adoption Potential for waterHeating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: acs

Calculating Adoption Potential for clothesDrying under 'AEO2023 Reference Case' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: acs

Calculating Adoption Potential for cooking under 'AEO2023 Reference Case' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: acs

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    iraRef_mp8_cooking_health_sensitivity: Masked 7784 values
    iraRef_mp8_cooking_adoption_upper_EASIUR_acs: Masked 7784 values
    iraRef_mp8_cooking_impact_upper_EASIUR_acs: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Calculating Adoption Potential for heating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: h6c

Calculating Adoption Potential for waterHeating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: h6c

Calculating Adoption Potential for clothesDrying under 'AEO2023 Reference Case' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: h6c

Calculating Adoption Potential for cooking under 'AEO2023 Reference Case' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: EASIUR | cr_function Function: h6c

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    iraRef_mp8_cooking_health_sensitivity: Masked 7784 values
    iraRef_mp8_cooking_adoption_upper_EASIUR_h6c: Masked 7784 values
    iraRef_mp8_cooking_impact_upper_EASIUR_h6c: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Calculating Adoption Potential for heating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: acs

Calculating Adoption Potential for waterHeating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: acs

Calculating Adoption Potential for clothesDrying under 'AEO2023 Reference Case' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: acs

Calculating Adoption Potential for cooking under 'AEO2023 Reference Case' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: acs

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    iraRef_mp8_cooking_health_sensitivity: Masked 7784 values
    iraRef_mp8_cooking_adoption_upper_InMAP_acs: Masked 7784 values
    iraRef_mp8_cooking_impact_upper_InMAP_acs: Masked 7784 values
  Total: Masked 23352 values across 12 columns
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Calculating Adoption Potential for heating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Found 12266 valid homes out of 15651 for heating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: h6c

Calculating Adoption Potential for waterHeating under 'AEO2023 Reference Case' Scenario...
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Found 14999 valid homes out of 15651 for waterHeating adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: h6c

Calculating Adoption Potential for clothesDrying under 'AEO2023 Reference Case' Scenario...
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Found 14743 valid homes out of 15651 for clothesDrying adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: h6c

Calculating Adoption Potential for cooking under 'AEO2023 Reference Case' Scenario...
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Found 7867 valid homes out of 15651 for cooking adoption potential

 --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): upper
                
                              Health Impact Sensitivity:
                                rcm_model Model: InMAP | cr_function Function: h6c

Verifying masking for all calculated columns:
Masking 12 columns for category 'cooking'
    iraRef_mp8_cooking_health_sensitivity: Masked 7784 values
    iraRef_mp8_cooking_adoption_upper_InMAP_h6c: Masked 7784 values
    iraRef_mp8_cooking_impact_upper_InMAP_h6c: Masked 7784 values
  Total: Masked 23352 values across 12 columns

====================================================================================================================================================================
ADOPTION FEASIBILITY OF VARIOUS RETROFITS: AEO2023 Reference Case
====================================================================================================================================================================
determine_adoption_potential.py file contains the definition for the adoption_decision function.

DATAFRAME FOR MP8 AFTER DETERMINING ADOPTION FEASIBILITY: df_euss_am_mp8_home
      
AP2: 
-----------------------------------------------
         square_footage census_region  census_division census_division_recs  \
bldg_id                                                                       
119              2152.0     Northeast  Middle Atlantic      Middle Atlantic   
122              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
150              1690.0     Northeast  Middle Atlantic      Middle Atlantic   
153              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
162              2663.0     Northeast  Middle Atlantic      Middle Atlantic   
...                 ...           ...              ...                  ...   
549882           1202.0     Northeast  Middle Atlantic      Middle Atlantic   
549915           2176.0     Northeast  Middle Atlantic      Middle Atlantic   
549937            885.0     Northeast  Middle Atlantic      Middle Atlantic   
549963           1690.0     Northeast  Middle Atlantic      Middle Atlantic   
549989           1220.0     Northeast  Middle Atlantic      Middle Atlantic   

        building_america_climate_zone  reeds_balancing_area gea_region state  \
bldg_id                                                                        
119                       Mixed-Humid                   122      RFCEc    PA   
122                       Mixed-Humid                   122      RFCEc    PA   
150                              Cold                   122      RFCEc    PA   
153                       Mixed-Humid                   122      RFCEc    PA   
162                       Mixed-Humid                   122      RFCEc    PA   
...                               ...                   ...        ...   ...   
549882                           Cold                   122      RFCEc    PA   
549915                           Cold                   122      RFCEc    PA   
549937                           Cold                   115      RFCWc    PA   
549963                           Cold                   115      RFCWc    PA   
549989                           Cold                   122      RFCEc    PA   

                            city    county  ...  \
bldg_id                                     ...   
119        Not in a census Place  G4200450  ...   
122      In another census Place  G4200450  ...   
150      In another census Place  G4201190  ...   
153        Not in a census Place  G4200170  ...   
162        Not in a census Place  G4200450  ...   
...                          ...       ...  ...   
549882     Not in a census Place  G4200110  ...   
549915                 Lancaster  G4200710  ...   
549937   In another census Place  G4200050  ...   
549963   In another census Place  G4201290  ...   
549989     Not in a census Place  G4201110  ...   

        iraRef_mp8_clothesDrying_total_npv_lessWTP_upper_AP2_h6c  \
bldg_id                                                            
119                                               -1108.55         
122                                               -1182.20         
150                                               -1354.97         
153                                                    NaN         
162                                                -917.44         
...                                                    ...         
549882                                             -593.03         
549915                                              384.33         
549937                                            -2244.76         
549963                                            -1295.85         
549989                                            -2842.65         

        iraRef_mp8_clothesDrying_total_npv_moreWTP_upper_AP2_h6c  \
bldg_id                                                            
119                                                 199.43         
122                                                -188.14         
150                                                -360.91         
153                                                    NaN         
162                                                  76.62         
...                                                    ...         
549882                                              401.03         
549915                                             1378.39         
549937                                             -936.78         
549963                                             -301.79         
549989                                            -1848.59         

        iraRef_mp8_clothesDrying_adoption_upper_AP2_h6c  \
bldg_id                                                   
119                    Tier 2: Feasible vs. Alternative   
122                                      Tier 4: Averse   
150                                      Tier 4: Averse   
153                     N/A: Invalid Baseline Fuel/Tech   
162               Tier 3: Subsidy-Dependent Feasibility   
...                                                 ...   
549882            Tier 3: Subsidy-Dependent Feasibility   
549915                 Tier 2: Feasible vs. Alternative   
549937                                   Tier 4: Averse   
549963                                   Tier 4: Averse   
549989                                   Tier 4: Averse   

        iraRef_mp8_clothesDrying_impact_upper_AP2_h6c  \
bldg_id                                                 
119                                    Public Benefit   
122                                    Public Benefit   
150                                    Public Benefit   
153                   N/A: Invalid Baseline Fuel/Tech   
162                                    Public Benefit   
...                                               ...   
549882                                 Public Benefit   
549915                                 Public Benefit   
549937                                 Public Benefit   
549963                                 Public Benefit   
549989                                 Public Benefit   

         iraRef_mp8_cooking_health_sensitivity  \
bldg_id                                          
119                                   AP2, h6c   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                AP2, h6c   
549963                                     NaN   
549989                                AP2, h6c   

         iraRef_mp8_cooking_benefit_upper_AP2_h6c  \
bldg_id                                             
119                                          0.00   
122                                           NaN   
150                                           NaN   
153                                           NaN   
162                                           NaN   
...                                           ...   
549882                                        NaN   
549915                                        NaN   
549937                                      23.50   
549963                                        NaN   
549989                                      73.66   

        iraRef_mp8_cooking_total_npv_lessWTP_upper_AP2_h6c  \
bldg_id                                                      
119                                               -1140.05   
122                                                    NaN   
150                                                    NaN   
153                                                    NaN   
162                                                    NaN   
...                                                    ...   
549882                                                 NaN   
549915                                                 NaN   
549937                                            -1980.45   
549963                                                 NaN   
549989                                            -1232.06   

         iraRef_mp8_cooking_total_npv_moreWTP_upper_AP2_h6c  \
bldg_id                                                       
119                                                 -67.51    
122                                                    NaN    
150                                                    NaN    
153                                                    NaN    
162                                                    NaN    
...                                                    ...    
549882                                                 NaN    
549915                                                 NaN    
549937                                             -907.91    
549963                                                 NaN    
549989                                             -159.52    

        iraRef_mp8_cooking_adoption_upper_AP2_h6c  \
bldg_id                                             
119              Tier 2: Feasible vs. Alternative   
122                                           NaN   
150                                           NaN   
153                                           NaN   
162                                           NaN   
...                                           ...   
549882                                        NaN   
549915                                        NaN   
549937                             Tier 4: Averse   
549963                                        NaN   
549989                             Tier 4: Averse   

        iraRef_mp8_cooking_impact_upper_AP2_h6c  
bldg_id                                          
119                            Public Detriment  
122                                         NaN  
150                                         NaN  
153                                         NaN  
162                                         NaN  
...                                         ...  
549882                                      NaN  
549915                                      NaN  
549937                           Public Benefit  
549963                                      NaN  
549989                           Public Benefit  

[15651 rows x 757 columns]

EASIUR:
-----------------------------------------------
         square_footage census_region  census_division census_division_recs  \
bldg_id                                                                       
119              2152.0     Northeast  Middle Atlantic      Middle Atlantic   
122              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
150              1690.0     Northeast  Middle Atlantic      Middle Atlantic   
153              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
162              2663.0     Northeast  Middle Atlantic      Middle Atlantic   
...                 ...           ...              ...                  ...   
549882           1202.0     Northeast  Middle Atlantic      Middle Atlantic   
549915           2176.0     Northeast  Middle Atlantic      Middle Atlantic   
549937            885.0     Northeast  Middle Atlantic      Middle Atlantic   
549963           1690.0     Northeast  Middle Atlantic      Middle Atlantic   
549989           1220.0     Northeast  Middle Atlantic      Middle Atlantic   

        building_america_climate_zone  reeds_balancing_area gea_region state  \
bldg_id                                                                        
119                       Mixed-Humid                   122      RFCEc    PA   
122                       Mixed-Humid                   122      RFCEc    PA   
150                              Cold                   122      RFCEc    PA   
153                       Mixed-Humid                   122      RFCEc    PA   
162                       Mixed-Humid                   122      RFCEc    PA   
...                               ...                   ...        ...   ...   
549882                           Cold                   122      RFCEc    PA   
549915                           Cold                   122      RFCEc    PA   
549937                           Cold                   115      RFCWc    PA   
549963                           Cold                   115      RFCWc    PA   
549989                           Cold                   122      RFCEc    PA   

                            city    county  ...  \
bldg_id                                     ...   
119        Not in a census Place  G4200450  ...   
122      In another census Place  G4200450  ...   
150      In another census Place  G4201190  ...   
153        Not in a census Place  G4200170  ...   
162        Not in a census Place  G4200450  ...   
...                          ...       ...  ...   
549882     Not in a census Place  G4200110  ...   
549915                 Lancaster  G4200710  ...   
549937   In another census Place  G4200050  ...   
549963   In another census Place  G4201290  ...   
549989     Not in a census Place  G4201110  ...   

        iraRef_mp8_clothesDrying_total_npv_lessWTP_upper_EASIUR_h6c  \
bldg_id                                                               
119                                               -1006.35            
122                                               -1165.68            
150                                               -1314.27            
153                                                    NaN            
162                                                -858.02            
...                                                    ...            
549882                                             -566.98            
549915                                              436.07            
549937                                            -2156.79            
549963                                            -1334.98            
549989                                            -2851.11            

        iraRef_mp8_clothesDrying_total_npv_moreWTP_upper_EASIUR_h6c  \
bldg_id                                                               
119                                                 301.63            
122                                                -171.62            
150                                                -320.21            
153                                                    NaN            
162                                                 136.04            
...                                                    ...            
549882                                              427.08            
549915                                             1430.13            
549937                                             -848.81            
549963                                             -340.92            
549989                                            -1857.05            

        iraRef_mp8_clothesDrying_adoption_upper_EASIUR_h6c  \
bldg_id                                                      
119                       Tier 2: Feasible vs. Alternative   
122                                         Tier 4: Averse   
150                                         Tier 4: Averse   
153                        N/A: Invalid Baseline Fuel/Tech   
162                  Tier 3: Subsidy-Dependent Feasibility   
...                                                    ...   
549882               Tier 3: Subsidy-Dependent Feasibility   
549915                    Tier 2: Feasible vs. Alternative   
549937                                      Tier 4: Averse   
549963                                      Tier 4: Averse   
549989                                      Tier 4: Averse   

        iraRef_mp8_clothesDrying_impact_upper_EASIUR_h6c  \
bldg_id                                                    
119                                       Public Benefit   
122                                       Public Benefit   
150                                       Public Benefit   
153                      N/A: Invalid Baseline Fuel/Tech   
162                                       Public Benefit   
...                                                  ...   
549882                                    Public Benefit   
549915                                    Public Benefit   
549937                                    Public Benefit   
549963                                    Public Benefit   
549989                                    Public Benefit   

         iraRef_mp8_cooking_health_sensitivity  \
bldg_id                                          
119                                EASIUR, h6c   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                             EASIUR, h6c   
549963                                     NaN   
549989                             EASIUR, h6c   

         iraRef_mp8_cooking_benefit_upper_EASIUR_h6c  \
bldg_id                                                
119                                             0.00   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                         86.47   
549963                                           NaN   
549989                                        180.14   

        iraRef_mp8_cooking_total_npv_lessWTP_upper_EASIUR_h6c  \
bldg_id                                                         
119                                                -977.72      
122                                                    NaN      
150                                                    NaN      
153                                                    NaN      
162                                                    NaN      
...                                                    ...      
549882                                                 NaN      
549915                                                 NaN      
549937                                            -1917.48      
549963                                                 NaN      
549989                                            -1125.58      

         iraRef_mp8_cooking_total_npv_moreWTP_upper_EASIUR_h6c  \
bldg_id                                                          
119                                                  94.82       
122                                                    NaN       
150                                                    NaN       
153                                                    NaN       
162                                                    NaN       
...                                                    ...       
549882                                                 NaN       
549915                                                 NaN       
549937                                             -844.94       
549963                                                 NaN       
549989                                              -53.04       

        iraRef_mp8_cooking_adoption_upper_EASIUR_h6c  \
bldg_id                                                
119                 Tier 2: Feasible vs. Alternative   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                Tier 4: Averse   
549963                                           NaN   
549989                                Tier 4: Averse   

        iraRef_mp8_cooking_impact_upper_EASIUR_h6c  
bldg_id                                             
119                                 Public Benefit  
122                                            NaN  
150                                            NaN  
153                                            NaN  
162                                            NaN  
...                                            ...  
549882                                         NaN  
549915                                         NaN  
549937                              Public Benefit  
549963                                         NaN  
549989                              Public Benefit  

[15651 rows x 757 columns]

InMAP:
-----------------------------------------------
         square_footage census_region  census_division census_division_recs  \
bldg_id                                                                       
119              2152.0     Northeast  Middle Atlantic      Middle Atlantic   
122              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
150              1690.0     Northeast  Middle Atlantic      Middle Atlantic   
153              2176.0     Northeast  Middle Atlantic      Middle Atlantic   
162              2663.0     Northeast  Middle Atlantic      Middle Atlantic   
...                 ...           ...              ...                  ...   
549882           1202.0     Northeast  Middle Atlantic      Middle Atlantic   
549915           2176.0     Northeast  Middle Atlantic      Middle Atlantic   
549937            885.0     Northeast  Middle Atlantic      Middle Atlantic   
549963           1690.0     Northeast  Middle Atlantic      Middle Atlantic   
549989           1220.0     Northeast  Middle Atlantic      Middle Atlantic   

        building_america_climate_zone  reeds_balancing_area gea_region state  \
bldg_id                                                                        
119                       Mixed-Humid                   122      RFCEc    PA   
122                       Mixed-Humid                   122      RFCEc    PA   
150                              Cold                   122      RFCEc    PA   
153                       Mixed-Humid                   122      RFCEc    PA   
162                       Mixed-Humid                   122      RFCEc    PA   
...                               ...                   ...        ...   ...   
549882                           Cold                   122      RFCEc    PA   
549915                           Cold                   122      RFCEc    PA   
549937                           Cold                   115      RFCWc    PA   
549963                           Cold                   115      RFCWc    PA   
549989                           Cold                   122      RFCEc    PA   

                            city    county  ...  \
bldg_id                                     ...   
119        Not in a census Place  G4200450  ...   
122      In another census Place  G4200450  ...   
150      In another census Place  G4201190  ...   
153        Not in a census Place  G4200170  ...   
162        Not in a census Place  G4200450  ...   
...                          ...       ...  ...   
549882     Not in a census Place  G4200110  ...   
549915                 Lancaster  G4200710  ...   
549937   In another census Place  G4200050  ...   
549963   In another census Place  G4201290  ...   
549989     Not in a census Place  G4201110  ...   

        iraRef_mp8_clothesDrying_total_npv_lessWTP_upper_InMAP_h6c  \
bldg_id                                                              
119                                                -899.74           
122                                               -1208.98           
150                                               -1248.09           
153                                                    NaN           
162                                               -1013.57           
...                                                    ...           
549882                                             -475.86           
549915                                              476.01           
549937                                            -2227.15           
549963                                            -1325.49           
549989                                            -2815.79           

        iraRef_mp8_clothesDrying_total_npv_moreWTP_upper_InMAP_h6c  \
bldg_id                                                              
119                                                 408.24           
122                                                -214.92           
150                                                -254.03           
153                                                    NaN           
162                                                 -19.51           
...                                                    ...           
549882                                              518.20           
549915                                             1470.07           
549937                                             -919.17           
549963                                             -331.43           
549989                                            -1821.73           

        iraRef_mp8_clothesDrying_adoption_upper_InMAP_h6c  \
bldg_id                                                     
119                      Tier 2: Feasible vs. Alternative   
122                                        Tier 4: Averse   
150                                        Tier 4: Averse   
153                       N/A: Invalid Baseline Fuel/Tech   
162                                        Tier 4: Averse   
...                                                   ...   
549882              Tier 3: Subsidy-Dependent Feasibility   
549915                   Tier 2: Feasible vs. Alternative   
549937                                     Tier 4: Averse   
549963                                     Tier 4: Averse   
549989                                     Tier 4: Averse   

        iraRef_mp8_clothesDrying_impact_upper_InMAP_h6c  \
bldg_id                                                   
119                                      Public Benefit   
122                                      Public Benefit   
150                                      Public Benefit   
153                     N/A: Invalid Baseline Fuel/Tech   
162                                      Public Benefit   
...                                                 ...   
549882                                   Public Benefit   
549915                                   Public Benefit   
549937                                   Public Benefit   
549963                                   Public Benefit   
549989                                   Public Benefit   

         iraRef_mp8_cooking_health_sensitivity  \
bldg_id                                          
119                                 InMAP, h6c   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                              InMAP, h6c   
549963                                     NaN   
549989                              InMAP, h6c   

         iraRef_mp8_cooking_benefit_upper_InMAP_h6c  \
bldg_id                                               
119                                            0.00   
122                                             NaN   
150                                             NaN   
153                                             NaN   
162                                             NaN   
...                                             ...   
549882                                          NaN   
549915                                          NaN   
549937                                        27.40   
549963                                          NaN   
549989                                        95.21   

        iraRef_mp8_cooking_total_npv_lessWTP_upper_InMAP_h6c  \
bldg_id                                                        
119                                                -760.18     
122                                                    NaN     
150                                                    NaN     
153                                                    NaN     
162                                                    NaN     
...                                                    ...     
549882                                                 NaN     
549915                                                 NaN     
549937                                            -1976.55     
549963                                                 NaN     
549989                                            -1210.51     

         iraRef_mp8_cooking_total_npv_moreWTP_upper_InMAP_h6c  \
bldg_id                                                         
119                                                 312.36      
122                                                    NaN      
150                                                    NaN      
153                                                    NaN      
162                                                    NaN      
...                                                    ...      
549882                                                 NaN      
549915                                                 NaN      
549937                                             -904.01      
549963                                                 NaN      
549989                                             -137.97      

        iraRef_mp8_cooking_adoption_upper_InMAP_h6c  \
bldg_id                                               
119                Tier 2: Feasible vs. Alternative   
122                                             NaN   
150                                             NaN   
153                                             NaN   
162                                             NaN   
...                                             ...   
549882                                          NaN   
549915                                          NaN   
549937                               Tier 4: Averse   
549963                                          NaN   
549989                               Tier 4: Averse   

        iraRef_mp8_cooking_impact_upper_InMAP_h6c  
bldg_id                                            
119                                Public Benefit  
122                                           NaN  
150                                           NaN  
153                                           NaN  
162                                           NaN  
...                                           ...  
549882                                        NaN  
549915                                        NaN  
549937                             Public Benefit  
549963                                        NaN  
549989                             Public Benefit  

[15651 rows x 757 columns]

