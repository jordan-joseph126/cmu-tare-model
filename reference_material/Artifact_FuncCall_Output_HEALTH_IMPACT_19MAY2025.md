# OUTPUT AS OF MAY 19, 2025

# =======================================================================================================
# CHECKING BASELINE COLUMNS
# =======================================================================================================

# --------------------------------------------------------
# FUNCTION CALL:
# --------------------------------------------------------

# Health Impacts: Baseline Scenario
print(f"""
====================================================================================================================================================================
df_euss_am_baseline_home: DataFrame containing the baseline scenario data
{df_euss_am_baseline_home}
      
df_baseline_damages_health: DataFrame containing the baseline scenario data with health damages
{df_baseline_damages_health}

""")

if print_debug:
    # df_euss_am_baseline_home
    print(f"Shape of df_euss_am_baseline_home: {df_euss_am_baseline_home.shape}")

    # Print columns that contain the word "damages"
    damage_columns = [col for col in df_euss_am_baseline_home.columns if "damages" in col.lower()]
    print("\nColumns containing 'damages':")
    print(damage_columns)

    # df_baseline_damages_health
    print(f"Shape of df_baseline_damages_health: {df_baseline_damages_health.shape}")

    # Print columns that contain the word "damages"
    damage_columns = [col for col in df_baseline_damages_health.columns if "damages" in col.lower()]
    print("\nColumns containing 'damages':")
    print(damage_columns)

# --------------------------------------------------------
# OUTPUT:
# --------------------------------------------------------


====================================================================================================================================================================
df_euss_am_baseline_home: DataFrame containing the baseline scenario data
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

        baseline_cooking_lifetime_damages_health_inmap_acs  \
bldg_id                                                      
119                                                 164.50   
122                                                    NaN   
150                                                    NaN   
153                                                    NaN   
162                                                    NaN   
...                                                    ...   
549882                                                 NaN   
549915                                                 NaN   
549937                                               56.12   
549963                                                 NaN   
549989                                               63.62   

        baseline_cooking_lifetime_damages_health_inmap_h6c fuel_type_heating  \
bldg_id                                                                        
119                                                 422.64        naturalGas   
122                                                    NaN           fuelOil   
150                                                    NaN       electricity   
153                                                    NaN           fuelOil   
162                                                    NaN           fuelOil   
...                                                    ...               ...   
549882                                                 NaN        naturalGas   
549915                                                 NaN       electricity   
549937                                              144.17        naturalGas   
549963                                                 NaN        naturalGas   
549989                                              163.46           fuelOil   

        fuel_type_waterHeating  fuel_type_clothesDrying  fuel_type_cooking  \
bldg_id                                                                      
119                 naturalGas               naturalGas         naturalGas   
122                    fuelOil              electricity        electricity   
150                electricity              electricity        electricity   
153                    fuelOil                      NaN        electricity   
162                 naturalGas              electricity        electricity   
...                        ...                      ...                ...   
549882              naturalGas              electricity        electricity   
549915             electricity              electricity        electricity   
549937              naturalGas               naturalGas         naturalGas   
549963              naturalGas              electricity        electricity   
549989                 fuelOil              electricity            propane   

        baseline_heating_lifetime_fuel_cost  \
bldg_id                                       
119                                 9692.04   
122                                56634.48   
150                                48835.84   
153                                52691.10   
162                                69001.23   
...                                     ...   
549882                                  NaN   
549915                             52898.91   
549937                              8456.00   
549963                             17473.34   
549989                             22285.90   

        baseline_waterHeating_lifetime_fuel_cost  \
bldg_id                                            
119                                      2093.14   
122                                      2681.82   
150                                      2514.43   
153                                     11588.33   
162                                      1702.24   
...                                          ...   
549882                                   2216.55   
549915                                   9119.49   
549937                                   5241.61   
549963                                   1745.16   
549989                                   4427.30   

        baseline_clothesDrying_lifetime_fuel_cost  \
bldg_id                                             
119                                        351.02   
122                                        619.28   
150                                        718.64   
153                                           NaN   
162                                       2222.41   
...                                           ...   
549882                                     997.02   
549915                                    1736.07   
549937                                     982.27   
549963                                     898.31   
549989                                     898.31   

        baseline_cooking_lifetime_fuel_cost  
bldg_id                                      
119                                  589.84  
122                                     NaN  
150                                     NaN  
153                                     NaN  
162                                     NaN  
...                                     ...  
549882                                  NaN  
549915                                  NaN  
549937                               802.30  
549963                                  NaN  
549989                              1691.55  

[15651 rows x 180 columns]
      
df_baseline_damages_health: DataFrame containing the baseline scenario data with health damages
         include_all  include_heating  include_waterHeating  \
bldg_id                                                       
119             True             True                  True   
122            False             True                  True   
150            False             True                  True   
153            False             True                  True   
162            False             True                  True   
...              ...              ...                   ...   
549882         False            False                  True   
549915         False             True                  True   
549937          True             True                  True   
549963         False             True                  True   
549989          True             True                  True   

         include_clothesDrying  include_cooking  valid_tech_heating  \
bldg_id                                                               
119                       True             True                True   
122                       True            False                True   
150                       True            False                True   
153                      False            False                True   
162                       True            False                True   
...                        ...              ...                 ...   
549882                    True            False               False   
549915                    True            False                True   
549937                    True             True                True   
549963                    True            False                True   
549989                    True             True                True   

         valid_tech_waterHeating  valid_fuel_heating  valid_fuel_waterHeating  \
bldg_id                                                                         
119                         True                True                     True   
122                         True                True                     True   
150                         True                True                     True   
153                         True                True                     True   
162                         True                True                     True   
...                          ...                 ...                      ...   
549882                      True                True                     True   
549915                      True                True                     True   
549937                      True                True                     True   
549963                      True                True                     True   
549989                      True                True                     True   

         valid_fuel_clothesDrying  ...  \
bldg_id                            ...   
119                          True  ...   
122                          True  ...   
150                          True  ...   
153                         False  ...   
162                          True  ...   
...                           ...  ...   
549882                       True  ...   
549915                       True  ...   
549937                       True  ...   
549963                       True  ...   
549989                       True  ...   

         baseline_2038_cooking_damages_pm25_easiur_h6c  \
bldg_id                                                  
119                                               6.83   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            3.32   
549963                                             NaN   
549989                                            2.01   

         baseline_2038_cooking_damages_health_easiur_h6c  \
bldg_id                                                    
119                                                25.75   
122                                                  NaN   
150                                                  NaN   
153                                                  NaN   
162                                                  NaN   
...                                                  ...   
549882                                               NaN   
549915                                               NaN   
549937                                             14.34   
549963                                               NaN   
549989                                             11.53   

         baseline_2038_cooking_damages_so2_inmap_acs  \
bldg_id                                                
119                                             0.40   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                          0.27   
549963                                           NaN   
549989                                          0.36   

         baseline_2038_cooking_damages_nox_inmap_acs  \
bldg_id                                                
119                                             7.49   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                          3.09   
549963                                           NaN   
549989                                          3.60   

         baseline_2038_cooking_damages_pm25_inmap_acs  \
bldg_id                                                 
119                                              3.59   
122                                               NaN   
150                                               NaN   
153                                               NaN   
162                                               NaN   
...                                               ...   
549882                                            NaN   
549915                                            NaN   
549937                                           0.40   
549963                                            NaN   
549989                                           0.30   

         baseline_2038_cooking_damages_health_inmap_acs  \
bldg_id                                                   
119                                               11.48   
122                                                 NaN   
150                                                 NaN   
153                                                 NaN   
162                                                 NaN   
...                                                 ...   
549882                                              NaN   
549915                                              NaN   
549937                                             3.76   
549963                                              NaN   
549989                                             4.25   

         baseline_2038_cooking_damages_so2_inmap_h6c  \
bldg_id                                                
119                                             1.03   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                          0.70   
549963                                           NaN   
549989                                          0.92   

         baseline_2038_cooking_damages_nox_inmap_h6c  \
bldg_id                                                
119                                            19.26   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                          7.94   
549963                                           NaN   
549989                                          9.24   

         baseline_2038_cooking_damages_pm25_inmap_h6c  \
bldg_id                                                 
119                                              9.21   
122                                               NaN   
150                                               NaN   
153                                               NaN   
162                                               NaN   
...                                               ...   
549882                                            NaN   
549915                                            NaN   
549937                                           1.02   
549963                                            NaN   
549989                                           0.76   

         baseline_2038_cooking_damages_health_inmap_h6c  
bldg_id                                                  
119                                               29.50  
122                                                 NaN  
150                                                 NaN  
153                                                 NaN  
162                                                 NaN  
...                                                 ...  
549882                                              NaN  
549915                                              NaN  
549937                                             9.65  
549963                                              NaN  
549989                                            10.92  

[15651 rows x 1331 columns]


Shape of df_euss_am_baseline_home: (15651, 180)

Columns containing 'damages':
['baseline_heating_lifetime_damages_climate_lrmer_lower', 'baseline_heating_lifetime_damages_climate_lrmer_central', 'baseline_heating_lifetime_damages_climate_lrmer_upper', 'baseline_heating_lifetime_damages_climate_srmer_lower', 'baseline_heating_lifetime_damages_climate_srmer_central', 'baseline_heating_lifetime_damages_climate_srmer_upper', 'baseline_waterHeating_lifetime_damages_climate_lrmer_lower', 'baseline_waterHeating_lifetime_damages_climate_lrmer_central', 'baseline_waterHeating_lifetime_damages_climate_lrmer_upper', 'baseline_waterHeating_lifetime_damages_climate_srmer_lower', 'baseline_waterHeating_lifetime_damages_climate_srmer_central', 'baseline_waterHeating_lifetime_damages_climate_srmer_upper', 'baseline_clothesDrying_lifetime_damages_climate_lrmer_lower', 'baseline_clothesDrying_lifetime_damages_climate_lrmer_central', 'baseline_clothesDrying_lifetime_damages_climate_lrmer_upper', 'baseline_clothesDrying_lifetime_damages_climate_srmer_lower', 'baseline_clothesDrying_lifetime_damages_climate_srmer_central', 'baseline_clothesDrying_lifetime_damages_climate_srmer_upper', 'baseline_cooking_lifetime_damages_climate_lrmer_lower', 'baseline_cooking_lifetime_damages_climate_lrmer_central', 'baseline_cooking_lifetime_damages_climate_lrmer_upper', 'baseline_cooking_lifetime_damages_climate_srmer_lower', 'baseline_cooking_lifetime_damages_climate_srmer_central', 'baseline_cooking_lifetime_damages_climate_srmer_upper', 'baseline_heating_lifetime_damages_health_ap2_acs', 'baseline_heating_lifetime_damages_health_ap2_h6c', 'baseline_heating_lifetime_damages_health_easiur_acs', 'baseline_heating_lifetime_damages_health_easiur_h6c', 'baseline_heating_lifetime_damages_health_inmap_acs', 'baseline_heating_lifetime_damages_health_inmap_h6c', 'baseline_waterHeating_lifetime_damages_health_ap2_acs', 'baseline_waterHeating_lifetime_damages_health_ap2_h6c', 'baseline_waterHeating_lifetime_damages_health_easiur_acs', 'baseline_waterHeating_lifetime_damages_health_easiur_h6c', 'baseline_waterHeating_lifetime_damages_health_inmap_acs', 'baseline_waterHeating_lifetime_damages_health_inmap_h6c', 'baseline_clothesDrying_lifetime_damages_health_ap2_acs', 'baseline_clothesDrying_lifetime_damages_health_ap2_h6c', 'baseline_clothesDrying_lifetime_damages_health_easiur_acs', 'baseline_clothesDrying_lifetime_damages_health_easiur_h6c', 'baseline_clothesDrying_lifetime_damages_health_inmap_acs', 'baseline_clothesDrying_lifetime_damages_health_inmap_h6c', 'baseline_cooking_lifetime_damages_health_ap2_acs', 'baseline_cooking_lifetime_damages_health_ap2_h6c', 'baseline_cooking_lifetime_damages_health_easiur_acs', 'baseline_cooking_lifetime_damages_health_easiur_h6c', 'baseline_cooking_lifetime_damages_health_inmap_acs', 'baseline_cooking_lifetime_damages_health_inmap_h6c']
Shape of df_baseline_damages_health: (15651, 1331)

Columns containing 'damages':
['baseline_2024_heating_damages_so2_ap2_acs', 'baseline_2024_heating_damages_nox_ap2_acs', 'baseline_2024_heating_damages_pm25_ap2_acs', 'baseline_2024_heating_damages_health_ap2_acs', 'baseline_2024_heating_damages_so2_ap2_h6c', 'baseline_2024_heating_damages_nox_ap2_h6c', 'baseline_2024_heating_damages_pm25_ap2_h6c', 'baseline_2024_heating_damages_health_ap2_h6c', 'baseline_2024_heating_damages_so2_easiur_acs', 'baseline_2024_heating_damages_nox_easiur_acs', 'baseline_2024_heating_damages_pm25_easiur_acs', 'baseline_2024_heating_damages_health_easiur_acs', 'baseline_2024_heating_damages_so2_easiur_h6c', 'baseline_2024_heating_damages_nox_easiur_h6c', 'baseline_2024_heating_damages_pm25_easiur_h6c', 'baseline_2024_heating_damages_health_easiur_h6c', 'baseline_2024_heating_damages_so2_inmap_acs', 'baseline_2024_heating_damages_nox_inmap_acs', 'baseline_2024_heating_damages_pm25_inmap_acs', 'baseline_2024_heating_damages_health_inmap_acs', 'baseline_2024_heating_damages_so2_inmap_h6c', 'baseline_2024_heating_damages_nox_inmap_h6c', 'baseline_2024_heating_damages_pm25_inmap_h6c', 'baseline_2024_heating_damages_health_inmap_h6c', 'baseline_2025_heating_damages_so2_ap2_acs', 'baseline_2025_heating_damages_nox_ap2_acs', 
...
'baseline_2038_cooking_damages_so2_inmap_acs', 'baseline_2038_cooking_damages_nox_inmap_acs', 'baseline_2038_cooking_damages_pm25_inmap_acs', 'baseline_2038_cooking_damages_health_inmap_acs', 'baseline_2038_cooking_damages_so2_inmap_h6c', 'baseline_2038_cooking_damages_nox_inmap_h6c', 'baseline_2038_cooking_damages_pm25_inmap_h6c', 'baseline_2038_cooking_damages_health_inmap_h6c']



# =======================================================================================================
# HEALTH IMPACT FUNCTION CALL AND OUTPUT
# =======================================================================================================

# --------------------------------------------------------
# FUNCTION CALL:
# --------------------------------------------------------

print("""
==================== SCENARIO: No Inflation Reduction Act ==========
""")
df_euss_am_mp8_home, df_mp8_noIRA_damages_climate = calculate_lifetime_climate_impacts(
    df=df_euss_am_mp8_home,
    menu_mp=menu_mp, 
    policy_scenario='No Inflation Reduction Act', 
    df_baseline_damages=df_baseline_damages_climate,
    verbose=True  # Add this parameter
    )

df_euss_am_mp8_home, df_mp8_noIRA_damages_health = calculate_lifetime_health_impacts(
    df=df_euss_am_mp8_home,
    menu_mp=menu_mp, 
    policy_scenario='No Inflation Reduction Act', 
    df_baseline_damages=df_baseline_damages_health,
    debug=False,
    verbose=True  # Add this parameter
    )


print("""
==================== SCENARIO: Inflation Reduction Act (AEO2023 Reference Case) ==========
""")
df_euss_am_mp8_home, df_mp8_IRA_damages_climate = calculate_lifetime_climate_impacts(
    df=df_euss_am_mp8_home,
    menu_mp=menu_mp, 
    policy_scenario='AEO2023 Reference Case', 
    df_baseline_damages=df_baseline_damages_climate,
    verbose=True  # Add this parameter
    )


df_euss_am_mp8_home, df_mp8_IRA_damages_health = calculate_lifetime_health_impacts(
    df=df_euss_am_mp8_home,
    menu_mp=menu_mp, 
    policy_scenario='AEO2023 Reference Case', 
    df_baseline_damages=df_baseline_damages_health,
    debug=False,
    verbose=True  # Add this parameter
    )


print(f"""  
====================================================================================================================================================================
Post-Retrofit (MP8) Marginal Damages: WHOLE-HOME
Scenario: No Inflation Reduction Act and AEO2023 Reference Case
====================================================================================================================================================================
calculate_emissions_damages.py file contains the definition for the calculate_marginal_damages function.
Additional information on emissions and damage factor lookups can be found in the calculate_emissions_damages.py file as well. 
      
CLIMATE DAMAGES: No IRA and IRA-Reference
--------------------------------------------------------
Climate Damages (No IRA): df_mp8_noIRA_damages_climate
{df_mp8_noIRA_damages_climate}

Climate Damages (IRA): df_mp8_IRA_damages_climate
{df_mp8_IRA_damages_climate}

HEALTH DAMAGES: No IRA and IRA-Reference
--------------------------------------------------------
Health Damages (No IRA): df_mp8_noIRA_damages_health
{df_mp8_noIRA_damages_health}

Health Damages (IRA): df_mp8_IRA_damages_health
{df_mp8_IRA_damages_health}

SUMMARY DATAFRAME FOR MP8: df_euss_am_mp8_home
{df_euss_am_mp8_home}
====================================================================================================================================================================
""")

# --------------------------------------------------------
# OUTPUT
# --------------------------------------------------------

==================== SCENARIO: No Inflation Reduction Act ==========

-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
Calculating Climate Emissions and Damages from 2024 to 2039 for heating
Measure package calculation for heating:
  - 12266 homes have valid data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Calculating Climate Emissions and Damages from 2024 to 2036 for waterHeating
Measure package calculation for waterHeating:
  - 14999 homes have valid data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Calculating Climate Emissions and Damages from 2024 to 2037 for clothesDrying
Measure package calculation for clothesDrying:
  - 14743 homes have valid data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Calculating Climate Emissions and Damages from 2024 to 2039 for cooking
Measure package calculation for cooking:
  - 7867 homes have valid data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)

Verifying masking for all calculated columns:

Verifying masking for all calculated columns:
Masking 16 columns for category 'cooking'

Verifying masking for all calculated columns:
Masking 136 columns for category 'cooking'
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
Calculating Health Emissions and Damages from 2024 to 2038 for heating
Measure package calculation for heating:
  - 12266 homes have valid data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Warning: Missing baseline for heating (acs, ap2). Avoided health values skipped.
Warning: Missing baseline for heating (h6c, ap2). Avoided health values skipped.
Warning: Missing baseline for heating (acs, easiur). Avoided health values skipped.
Warning: Missing baseline for heating (h6c, easiur). Avoided health values skipped.
Warning: Missing baseline for heating (acs, inmap). Avoided health values skipped.
Warning: Missing baseline for heating (h6c, inmap). Avoided health values skipped.
Calculating Health Emissions and Damages from 2024 to 2035 for waterHeating
Measure package calculation for waterHeating:
  - 14999 homes have valid data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Warning: Missing baseline for waterHeating (acs, ap2). Avoided health values skipped.
Warning: Missing baseline for waterHeating (h6c, ap2). Avoided health values skipped.
Warning: Missing baseline for waterHeating (acs, easiur). Avoided health values skipped.
Warning: Missing baseline for waterHeating (h6c, easiur). Avoided health values skipped.
Warning: Missing baseline for waterHeating (acs, inmap). Avoided health values skipped.
Warning: Missing baseline for waterHeating (h6c, inmap). Avoided health values skipped.
Calculating Health Emissions and Damages from 2024 to 2036 for clothesDrying
Measure package calculation for clothesDrying:
  - 14743 homes have valid data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Warning: Missing baseline for clothesDrying (acs, ap2). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (h6c, ap2). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (acs, easiur). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (h6c, easiur). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (acs, inmap). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (h6c, inmap). Avoided health values skipped.
Calculating Health Emissions and Damages from 2024 to 2038 for cooking
Measure package calculation for cooking:
  - 7867 homes have valid data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Warning: Missing baseline for cooking (acs, ap2). Avoided health values skipped.
Warning: Missing baseline for cooking (h6c, ap2). Avoided health values skipped.
Warning: Missing baseline for cooking (acs, easiur). Avoided health values skipped.
Warning: Missing baseline for cooking (h6c, easiur). Avoided health values skipped.
Warning: Missing baseline for cooking (acs, inmap). Avoided health values skipped.
Warning: Missing baseline for cooking (h6c, inmap). Avoided health values skipped.

Verifying masking for all calculated columns:

Verifying masking for all calculated columns:
Masking 6 columns for category 'cooking'

Verifying masking for all calculated columns:
Masking 360 columns for category 'cooking'

==================== SCENARIO: Inflation Reduction Act (AEO2023 Reference Case) ==========

-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              
Calculating Climate Emissions and Damages from 2024 to 2039 for heating
Measure package calculation for heating:
  - 12266 homes have valid data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Calculating Climate Emissions and Damages from 2024 to 2036 for waterHeating
Measure package calculation for waterHeating:
  - 14999 homes have valid data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Calculating Climate Emissions and Damages from 2024 to 2037 for clothesDrying
Measure package calculation for clothesDrying:
  - 14743 homes have valid data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Calculating Climate Emissions and Damages from 2024 to 2039 for cooking
Measure package calculation for cooking:
  - 7867 homes have valid data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)

Verifying masking for all calculated columns:

Verifying masking for all calculated columns:
Masking 16 columns for category 'cooking'

Verifying masking for all calculated columns:
Masking 136 columns for category 'cooking'
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              
Calculating Health Emissions and Damages from 2024 to 2038 for heating
Measure package calculation for heating:
  - 12266 homes have valid data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
Warning: Missing baseline for heating (acs, ap2). Avoided health values skipped.
Warning: Missing baseline for heating (h6c, ap2). Avoided health values skipped.
Warning: Missing baseline for heating (acs, easiur). Avoided health values skipped.
Warning: Missing baseline for heating (h6c, easiur). Avoided health values skipped.
Warning: Missing baseline for heating (acs, inmap). Avoided health values skipped.
Warning: Missing baseline for heating (h6c, inmap). Avoided health values skipped.
Calculating Health Emissions and Damages from 2024 to 2035 for waterHeating
Measure package calculation for waterHeating:
  - 14999 homes have valid data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
Warning: Missing baseline for waterHeating (acs, ap2). Avoided health values skipped.
Warning: Missing baseline for waterHeating (h6c, ap2). Avoided health values skipped.
Warning: Missing baseline for waterHeating (acs, easiur). Avoided health values skipped.
Warning: Missing baseline for waterHeating (h6c, easiur). Avoided health values skipped.
Warning: Missing baseline for waterHeating (acs, inmap). Avoided health values skipped.
Warning: Missing baseline for waterHeating (h6c, inmap). Avoided health values skipped.
Calculating Health Emissions and Damages from 2024 to 2036 for clothesDrying
Measure package calculation for clothesDrying:
  - 14743 homes have valid data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
Warning: Missing baseline for clothesDrying (acs, ap2). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (h6c, ap2). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (acs, easiur). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (h6c, easiur). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (acs, inmap). Avoided health values skipped.
Warning: Missing baseline for clothesDrying (h6c, inmap). Avoided health values skipped.
Calculating Health Emissions and Damages from 2024 to 2038 for cooking
Measure package calculation for cooking:
  - 7867 homes have valid data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
Warning: Missing baseline for cooking (acs, ap2). Avoided health values skipped.
Warning: Missing baseline for cooking (h6c, ap2). Avoided health values skipped.
Warning: Missing baseline for cooking (acs, easiur). Avoided health values skipped.
Warning: Missing baseline for cooking (h6c, easiur). Avoided health values skipped.
Warning: Missing baseline for cooking (acs, inmap). Avoided health values skipped.
Warning: Missing baseline for cooking (h6c, inmap). Avoided health values skipped.

Verifying masking for all calculated columns:

Verifying masking for all calculated columns:
Masking 6 columns for category 'cooking'

Verifying masking for all calculated columns:
Masking 360 columns for category 'cooking'
  
====================================================================================================================================================================
Post-Retrofit (MP8) Marginal Damages: WHOLE-HOME
Scenario: No Inflation Reduction Act and AEO2023 Reference Case
====================================================================================================================================================================
calculate_emissions_damages.py file contains the definition for the calculate_marginal_damages function.
Additional information on emissions and damage factor lookups can be found in the calculate_emissions_damages.py file as well. 
      
CLIMATE DAMAGES: No IRA and IRA-Reference
--------------------------------------------------------
Climate Damages (No IRA): df_mp8_noIRA_damages_climate
         include_all  include_heating  include_waterHeating  \
bldg_id                                                       
119             True             True                  True   
122            False             True                  True   
150            False             True                  True   
153            False             True                  True   
162            False             True                  True   
...              ...              ...                   ...   
549882         False            False                  True   
549915         False             True                  True   
549937          True             True                  True   
549963         False             True                  True   
549989          True             True                  True   

         include_clothesDrying  include_cooking  valid_tech_heating  \
bldg_id                                                               
119                       True             True                True   
122                       True            False                True   
150                       True            False                True   
153                      False            False                True   
162                       True            False                True   
...                        ...              ...                 ...   
549882                    True            False               False   
549915                    True            False                True   
549937                    True             True                True   
549963                    True            False                True   
549989                    True             True                True   

         valid_tech_waterHeating  valid_fuel_heating  valid_fuel_waterHeating  \
bldg_id                                                                         
119                         True                True                     True   
122                         True                True                     True   
150                         True                True                     True   
153                         True                True                     True   
162                         True                True                     True   
...                          ...                 ...                      ...   
549882                      True                True                     True   
549915                      True                True                     True   
549937                      True                True                     True   
549963                      True                True                     True   
549989                      True                True                     True   

         valid_fuel_clothesDrying  ...  \
bldg_id                            ...   
119                          True  ...   
122                          True  ...   
150                          True  ...   
153                         False  ...   
162                          True  ...   
...                           ...  ...   
549882                       True  ...   
549915                       True  ...   
549937                       True  ...   
549963                       True  ...   
549989                       True  ...   

         preIRA_mp8_cooking_avoided_damages_climate_lrmer_upper  \
bldg_id                                                           
119                                                  98.39        
122                                                    NaN        
150                                                    NaN        
153                                                    NaN        
162                                                    NaN        
...                                                    ...        
549882                                                 NaN        
549915                                                 NaN        
549937                                               64.98        
549963                                                 NaN        
549989                                              234.20        

         preIRA_mp8_cooking_avoided_mt_co2e_lrmer  \
bldg_id                                             
119                                          0.34   
122                                           NaN   
150                                           NaN   
153                                           NaN   
162                                           NaN   
...                                           ...   
549882                                        NaN   
549915                                        NaN   
549937                                       0.24   
549963                                        NaN   
549989                                       0.87   

         preIRA_mp8_cooking_lifetime_mt_co2e_srmer  \
bldg_id                                              
119                                           6.04   
122                                            NaN   
150                                            NaN   
153                                            NaN   
162                                            NaN   
...                                            ...   
549882                                         NaN   
549915                                         NaN   
549937                                        9.03   
549963                                         NaN   
549989                                        5.16   

         preIRA_mp8_cooking_lifetime_damages_climate_srmer_lower  \
bldg_id                                                            
119                                                 140.60         
122                                                    NaN         
150                                                    NaN         
153                                                    NaN         
162                                                    NaN         
...                                                    ...         
549882                                                 NaN         
549915                                                 NaN         
549937                                              209.98         
549963                                                 NaN         
549989                                              120.19         

         preIRA_mp8_cooking_avoided_damages_climate_srmer_lower  \
bldg_id                                                           
119                                                 -54.51        
122                                                    NaN        
150                                                    NaN        
153                                                    NaN        
162                                                    NaN        
...                                                    ...        
549882                                                 NaN        
549915                                                 NaN        
549937                                              -91.58        
549963                                                 NaN        
549989                                              -33.22        

         preIRA_mp8_cooking_lifetime_damages_climate_srmer_central  \
bldg_id                                                              
119                                                 445.20           
122                                                    NaN           
150                                                    NaN           
153                                                    NaN           
162                                                    NaN           
...                                                    ...           
549882                                                 NaN           
549915                                                 NaN           
549937                                              665.14           
549963                                                 NaN           
549989                                              380.58           

         preIRA_mp8_cooking_avoided_damages_climate_srmer_central  \
bldg_id                                                             
119                                                -172.96          
122                                                    NaN          
150                                                    NaN          
153                                                    NaN          
162                                                    NaN          
...                                                    ...          
549882                                                 NaN          
549915                                                 NaN          
549937                                             -290.71          
549963                                                 NaN          
549989                                             -105.56          

         preIRA_mp8_cooking_lifetime_damages_climate_srmer_upper  \
bldg_id                                                            
119                                                1581.70         
122                                                    NaN         
150                                                    NaN         
153                                                    NaN         
162                                                    NaN         
...                                                    ...         
549882                                                 NaN         
549915                                                 NaN         
549937                                             2363.37         
549963                                                 NaN         
549989                                             1352.12         

         preIRA_mp8_cooking_avoided_damages_climate_srmer_upper  \
bldg_id                                                           
119                                                -615.10        
122                                                    NaN        
150                                                    NaN        
153                                                    NaN        
162                                                    NaN        
...                                                    ...        
549882                                                 NaN        
549915                                                 NaN        
549937                                            -1033.92        
549963                                                 NaN        
549989                                             -375.67        

         preIRA_mp8_cooking_avoided_mt_co2e_srmer  
bldg_id                                            
119                                         -2.36  
122                                           NaN  
150                                           NaN  
153                                           NaN  
162                                           NaN  
...                                           ...  
549882                                        NaN  
549915                                        NaN  
549937                                      -3.96  
549963                                        NaN  
549989                                      -1.44  

[15651 rows x 515 columns]

Climate Damages (IRA): df_mp8_IRA_damages_climate
         include_all  include_heating  include_waterHeating  \
bldg_id                                                       
119             True             True                  True   
122            False             True                  True   
150            False             True                  True   
153            False             True                  True   
162            False             True                  True   
...              ...              ...                   ...   
549882         False            False                  True   
549915         False             True                  True   
549937          True             True                  True   
549963         False             True                  True   
549989          True             True                  True   

         include_clothesDrying  include_cooking  valid_tech_heating  \
bldg_id                                                               
119                       True             True                True   
122                       True            False                True   
150                       True            False                True   
153                      False            False                True   
162                       True            False                True   
...                        ...              ...                 ...   
549882                    True            False               False   
549915                    True            False                True   
549937                    True             True                True   
549963                    True            False                True   
549989                    True             True                True   

         valid_tech_waterHeating  valid_fuel_heating  valid_fuel_waterHeating  \
bldg_id                                                                         
119                         True                True                     True   
122                         True                True                     True   
150                         True                True                     True   
153                         True                True                     True   
162                         True                True                     True   
...                          ...                 ...                      ...   
549882                      True                True                     True   
549915                      True                True                     True   
549937                      True                True                     True   
549963                      True                True                     True   
549989                      True                True                     True   

         valid_fuel_clothesDrying  ...  \
bldg_id                            ...   
119                          True  ...   
122                          True  ...   
150                          True  ...   
153                         False  ...   
162                          True  ...   
...                           ...  ...   
549882                       True  ...   
549915                       True  ...   
549937                       True  ...   
549963                       True  ...   
549989                       True  ...   

         iraRef_mp8_cooking_avoided_damages_climate_lrmer_upper  \
bldg_id                                                           
119                                                 239.83        
122                                                    NaN        
150                                                    NaN        
153                                                    NaN        
162                                                    NaN        
...                                                    ...        
549882                                                 NaN        
549915                                                 NaN        
549937                                              205.79        
549963                                                 NaN        
549989                                              355.11        

         iraRef_mp8_cooking_avoided_mt_co2e_lrmer  \
bldg_id                                             
119                                          0.91   
122                                           NaN   
150                                           NaN   
153                                           NaN   
162                                           NaN   
...                                           ...   
549882                                        NaN   
549915                                        NaN   
549937                                       0.78   
549963                                        NaN   
549989                                       1.35   

         iraRef_mp8_cooking_lifetime_mt_co2e_srmer  \
bldg_id                                              
119                                           5.30   
122                                            NaN   
150                                            NaN   
153                                            NaN   
162                                            NaN   
...                                            ...   
549882                                         NaN   
549915                                         NaN   
549937                                        7.37   
549963                                         NaN   
549989                                        4.53   

         iraRef_mp8_cooking_lifetime_damages_climate_srmer_lower  \
bldg_id                                                            
119                                                 122.61         
122                                                    NaN         
150                                                    NaN         
153                                                    NaN         
162                                                    NaN         
...                                                    ...         
549882                                                 NaN         
549915                                                 NaN         
549937                                              169.77         
549963                                                 NaN         
549989                                              104.81         

         iraRef_mp8_cooking_avoided_damages_climate_srmer_lower  \
bldg_id                                                           
119                                                 -36.52        
122                                                    NaN        
150                                                    NaN        
153                                                    NaN        
162                                                    NaN        
...                                                    ...        
549882                                                 NaN        
549915                                                 NaN        
549937                                              -51.37        
549963                                                 NaN        
549989                                              -17.84        

         iraRef_mp8_cooking_lifetime_damages_climate_srmer_central  \
bldg_id                                                              
119                                                 389.10           
122                                                    NaN           
150                                                    NaN           
153                                                    NaN           
162                                                    NaN           
...                                                    ...           
549882                                                 NaN           
549915                                                 NaN           
549937                                              539.42           
549963                                                 NaN           
549989                                              332.62           

         iraRef_mp8_cooking_avoided_damages_climate_srmer_central  \
bldg_id                                                             
119                                                -116.86          
122                                                    NaN          
150                                                    NaN          
153                                                    NaN          
162                                                    NaN          
...                                                    ...          
549882                                                 NaN          
549915                                                 NaN          
549937                                             -164.99          
549963                                                 NaN          
549989                                              -57.60          

         iraRef_mp8_cooking_lifetime_damages_climate_srmer_upper  \
bldg_id                                                            
119                                                1384.04         
122                                                    NaN         
150                                                    NaN         
153                                                    NaN         
162                                                    NaN         
...                                                    ...         
549882                                                 NaN         
549915                                                 NaN         
549937                                             1919.97         
549963                                                 NaN         
549989                                             1183.15         

         iraRef_mp8_cooking_avoided_damages_climate_srmer_upper  \
bldg_id                                                           
119                                                -417.44        
122                                                    NaN        
150                                                    NaN        
153                                                    NaN        
162                                                    NaN        
...                                                    ...        
549882                                                 NaN        
549915                                                 NaN        
549937                                             -590.52        
549963                                                 NaN        
549989                                             -206.70        

         iraRef_mp8_cooking_avoided_mt_co2e_srmer  
bldg_id                                            
119                                         -1.62  
122                                           NaN  
150                                           NaN  
153                                           NaN  
162                                           NaN  
...                                           ...  
549882                                        NaN  
549915                                        NaN  
549937                                      -2.30  
549963                                        NaN  
549989                                      -0.81  

[15651 rows x 515 columns]

HEALTH DAMAGES: No IRA and IRA-Reference
--------------------------------------------------------
Health Damages (No IRA): df_mp8_noIRA_damages_health
         include_all  include_heating  include_waterHeating  \
bldg_id                                                       
119             True             True                  True   
122            False             True                  True   
150            False             True                  True   
153            False             True                  True   
162            False             True                  True   
...              ...              ...                   ...   
549882         False            False                  True   
549915         False             True                  True   
549937          True             True                  True   
549963         False             True                  True   
549989          True             True                  True   

         include_clothesDrying  include_cooking  valid_tech_heating  \
bldg_id                                                               
119                       True             True                True   
122                       True            False                True   
150                       True            False                True   
153                      False            False                True   
162                       True            False                True   
...                        ...              ...                 ...   
549882                    True            False               False   
549915                    True            False                True   
549937                    True             True                True   
549963                    True            False                True   
549989                    True             True                True   

         valid_tech_waterHeating  valid_fuel_heating  valid_fuel_waterHeating  \
bldg_id                                                                         
119                         True                True                     True   
122                         True                True                     True   
150                         True                True                     True   
153                         True                True                     True   
162                         True                True                     True   
...                          ...                 ...                      ...   
549882                      True                True                     True   
549915                      True                True                     True   
549937                      True                True                     True   
549963                      True                True                     True   
549989                      True                True                     True   

         valid_fuel_clothesDrying  ...  \
bldg_id                            ...   
119                          True  ...   
122                          True  ...   
150                          True  ...   
153                         False  ...   
162                          True  ...   
...                           ...  ...   
549882                       True  ...   
549915                       True  ...   
549937                       True  ...   
549963                       True  ...   
549989                       True  ...   

         preIRA_mp8_2038_cooking_damages_pm25_easiur_h6c  \
bldg_id                                                    
119                                                10.35   
122                                                  NaN   
150                                                  NaN   
153                                                  NaN   
162                                                  NaN   
...                                                  ...   
549882                                               NaN   
549915                                               NaN   
549937                                              3.22   
549963                                               NaN   
549989                                              3.80   

         preIRA_mp8_2038_cooking_damages_health_easiur_h6c  \
bldg_id                                                      
119                                                  28.24   
122                                                    NaN   
150                                                    NaN   
153                                                    NaN   
162                                                    NaN   
...                                                    ...   
549882                                                 NaN   
549915                                                 NaN   
549937                                               12.45   
549963                                                 NaN   
549989                                               14.62   

         preIRA_mp8_2038_cooking_damages_so2_inmap_acs  \
bldg_id                                                  
119                                               4.66   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            3.10   
549963                                             NaN   
549989                                            4.99   

         preIRA_mp8_2038_cooking_damages_nox_inmap_acs  \
bldg_id                                                  
119                                               0.48   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            0.96   
549963                                             NaN   
549989                                            0.97   

         preIRA_mp8_2038_cooking_damages_pm25_inmap_acs  \
bldg_id                                                   
119                                                1.28   
122                                                 NaN   
150                                                 NaN   
153                                                 NaN   
162                                                 NaN   
...                                                 ...   
549882                                              NaN   
549915                                              NaN   
549937                                             0.67   
549963                                              NaN   
549989                                             1.03   

         preIRA_mp8_2038_cooking_damages_health_inmap_acs  \
bldg_id                                                     
119                                                  6.42   
122                                                   NaN   
150                                                   NaN   
153                                                   NaN   
162                                                   NaN   
...                                                   ...   
549882                                                NaN   
549915                                                NaN   
549937                                               4.73   
549963                                                NaN   
549989                                               6.99   

         preIRA_mp8_2038_cooking_damages_so2_inmap_h6c  \
bldg_id                                                  
119                                              11.97   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            7.96   
549963                                             NaN   
549989                                           12.83   

         preIRA_mp8_2038_cooking_damages_nox_inmap_h6c  \
bldg_id                                                  
119                                               1.24   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            2.46   
549963                                             NaN   
549989                                            2.49   

         preIRA_mp8_2038_cooking_damages_pm25_inmap_h6c  \
bldg_id                                                   
119                                                3.28   
122                                                 NaN   
150                                                 NaN   
153                                                 NaN   
162                                                 NaN   
...                                                 ...   
549882                                              NaN   
549915                                              NaN   
549937                                             1.73   
549963                                              NaN   
549989                                             2.63   

         preIRA_mp8_2038_cooking_damages_health_inmap_h6c  
bldg_id                                                    
119                                                 16.49  
122                                                   NaN  
150                                                   NaN  
153                                                   NaN  
162                                                   NaN  
...                                                   ...  
549882                                                NaN  
549915                                                NaN  
549937                                              12.15  
549963                                                NaN  
549989                                              17.96  

[15651 rows x 1331 columns]

Health Damages (IRA): df_mp8_IRA_damages_health
         include_all  include_heating  include_waterHeating  \
bldg_id                                                       
119             True             True                  True   
122            False             True                  True   
150            False             True                  True   
153            False             True                  True   
162            False             True                  True   
...              ...              ...                   ...   
549882         False            False                  True   
549915         False             True                  True   
549937          True             True                  True   
549963         False             True                  True   
549989          True             True                  True   

         include_clothesDrying  include_cooking  valid_tech_heating  \
bldg_id                                                               
119                       True             True                True   
122                       True            False                True   
150                       True            False                True   
153                      False            False                True   
162                       True            False                True   
...                        ...              ...                 ...   
549882                    True            False               False   
549915                    True            False                True   
549937                    True             True                True   
549963                    True            False                True   
549989                    True             True                True   

         valid_tech_waterHeating  valid_fuel_heating  valid_fuel_waterHeating  \
bldg_id                                                                         
119                         True                True                     True   
122                         True                True                     True   
150                         True                True                     True   
153                         True                True                     True   
162                         True                True                     True   
...                          ...                 ...                      ...   
549882                      True                True                     True   
549915                      True                True                     True   
549937                      True                True                     True   
549963                      True                True                     True   
549989                      True                True                     True   

         valid_fuel_clothesDrying  ...  \
bldg_id                            ...   
119                          True  ...   
122                          True  ...   
150                          True  ...   
153                         False  ...   
162                          True  ...   
...                           ...  ...   
549882                       True  ...   
549915                       True  ...   
549937                       True  ...   
549963                       True  ...   
549989                       True  ...   

         iraRef_mp8_2038_cooking_damages_pm25_easiur_h6c  \
bldg_id                                                    
119                                                10.35   
122                                                  NaN   
150                                                  NaN   
153                                                  NaN   
162                                                  NaN   
...                                                  ...   
549882                                               NaN   
549915                                               NaN   
549937                                              3.22   
549963                                               NaN   
549989                                              3.80   

         iraRef_mp8_2038_cooking_damages_health_easiur_h6c  \
bldg_id                                                      
119                                                  28.24   
122                                                    NaN   
150                                                    NaN   
153                                                    NaN   
162                                                    NaN   
...                                                    ...   
549882                                                 NaN   
549915                                                 NaN   
549937                                               12.45   
549963                                                 NaN   
549989                                               14.62   

         iraRef_mp8_2038_cooking_damages_so2_inmap_acs  \
bldg_id                                                  
119                                               4.66   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            3.10   
549963                                             NaN   
549989                                            4.99   

         iraRef_mp8_2038_cooking_damages_nox_inmap_acs  \
bldg_id                                                  
119                                               0.48   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            0.96   
549963                                             NaN   
549989                                            0.97   

         iraRef_mp8_2038_cooking_damages_pm25_inmap_acs  \
bldg_id                                                   
119                                                1.28   
122                                                 NaN   
150                                                 NaN   
153                                                 NaN   
162                                                 NaN   
...                                                 ...   
549882                                              NaN   
549915                                              NaN   
549937                                             0.67   
549963                                              NaN   
549989                                             1.03   

         iraRef_mp8_2038_cooking_damages_health_inmap_acs  \
bldg_id                                                     
119                                                  6.42   
122                                                   NaN   
150                                                   NaN   
153                                                   NaN   
162                                                   NaN   
...                                                   ...   
549882                                                NaN   
549915                                                NaN   
549937                                               4.73   
549963                                                NaN   
549989                                               6.99   

         iraRef_mp8_2038_cooking_damages_so2_inmap_h6c  \
bldg_id                                                  
119                                              11.97   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            7.96   
549963                                             NaN   
549989                                           12.83   

         iraRef_mp8_2038_cooking_damages_nox_inmap_h6c  \
bldg_id                                                  
119                                               1.24   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                            2.46   
549963                                             NaN   
549989                                            2.49   

         iraRef_mp8_2038_cooking_damages_pm25_inmap_h6c  \
bldg_id                                                   
119                                                3.28   
122                                                 NaN   
150                                                 NaN   
153                                                 NaN   
162                                                 NaN   
...                                                 ...   
549882                                              NaN   
549915                                              NaN   
549937                                             1.73   
549963                                              NaN   
549989                                             2.63   

         iraRef_mp8_2038_cooking_damages_health_inmap_h6c  
bldg_id                                                    
119                                                 16.49  
122                                                   NaN  
150                                                   NaN  
153                                                   NaN  
162                                                   NaN  
...                                                   ...  
549882                                                NaN  
549915                                                NaN  
549937                                              12.15  
549963                                                NaN  
549989                                              17.96  

[15651 rows x 1331 columns]

SUMMARY DATAFRAME FOR MP8: df_euss_am_mp8_home
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

        iraRef_mp8_clothesDrying_lifetime_damages_health_easiur_acs  \
bldg_id                                                               
119                                                 115.20            
122                                                  59.93            
150                                                  54.35            
153                                                    NaN            
162                                                 214.84            
...                                                    ...            
549882                                               91.50            
549915                                              154.70            
549937                                              128.24            
549963                                               34.57            
549989                                               56.26            

        iraRef_mp8_clothesDrying_lifetime_damages_health_easiur_h6c  \
bldg_id                                                               
119                                                 295.99            
122                                                 153.98            
150                                                 139.64            
153                                                    NaN            
162                                                 551.98            
...                                                    ...            
549882                                              235.10            
549915                                              397.47            
549937                                              329.48            
549963                                               88.82            
549989                                              144.55            

        iraRef_mp8_clothesDrying_lifetime_damages_health_inmap_acs  \
bldg_id                                                              
119                                                  75.63           
122                                                  39.34           
150                                                  85.59           
153                                                    NaN           
162                                                 141.05           
...                                                    ...           
549882                                              134.50           
549915                                              173.42           
549937                                              129.42           
549963                                               38.82           
549989                                               72.95           

        iraRef_mp8_clothesDrying_lifetime_damages_health_inmap_h6c  \
bldg_id                                                              
119                                                 194.32           
122                                                 101.09           
150                                                 219.91           
153                                                    NaN           
162                                                 362.38           
...                                                    ...           
549882                                              345.57           
549915                                              445.56           
549937                                              332.51           
549963                                               99.73           
549989                                              187.42           

         iraRef_mp8_cooking_lifetime_damages_health_ap2_acs  \
bldg_id                                                       
119                                                 190.90    
122                                                    NaN    
150                                                    NaN    
153                                                    NaN    
162                                                    NaN    
...                                                    ...    
549882                                                 NaN    
549915                                                 NaN    
549937                                               87.16    
549963                                                 NaN    
549989                                              129.84    

         iraRef_mp8_cooking_lifetime_damages_health_ap2_h6c  \
bldg_id                                                       
119                                                 490.47    
122                                                    NaN    
150                                                    NaN    
153                                                    NaN    
162                                                    NaN    
...                                                    ...    
549882                                                 NaN    
549915                                                 NaN    
549937                                              223.94    
549963                                                 NaN    
549989                                              333.60    

        iraRef_mp8_cooking_lifetime_damages_health_easiur_acs  \
bldg_id                                                         
119                                                 221.81      
122                                                    NaN      
150                                                    NaN      
153                                                    NaN      
162                                                    NaN      
...                                                    ...      
549882                                                 NaN      
549915                                                 NaN      
549937                                              116.35      
549963                                                 NaN      
549989                                              122.02      

        iraRef_mp8_cooking_lifetime_damages_health_easiur_h6c  \
bldg_id                                                         
119                                                 569.88      
122                                                    NaN      
150                                                    NaN      
153                                                    NaN      
162                                                    NaN      
...                                                    ...      
549882                                                 NaN      
549915                                                 NaN      
549937                                              298.92      
549963                                                 NaN      
549989                                              313.49      

        iraRef_mp8_cooking_lifetime_damages_health_inmap_acs  \
bldg_id                                                        
119                                                 144.08     
122                                                    NaN     
150                                                    NaN     
153                                                    NaN     
162                                                    NaN     
...                                                    ...     
549882                                                 NaN     
549915                                                 NaN     
549937                                              117.11     
549963                                                 NaN     
549989                                              157.45     

        iraRef_mp8_cooking_lifetime_damages_health_inmap_h6c  
bldg_id                                                       
119                                                 370.18    
122                                                    NaN    
150                                                    NaN    
153                                                    NaN    
162                                                    NaN    
...                                                    ...    
549882                                                 NaN    
549915                                                 NaN    
549937                                              300.89    
549963                                                 NaN    
549989                                              404.53    

[15651 rows x 487 columns]
====================================================================================================================================================================