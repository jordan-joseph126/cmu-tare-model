# ====================================================================================================================================================================
# MODERATE SCENARIO
# ====================================================================================================================================================================


====================================================================================================================================================================
SCENARIO ANALYSIS (NO INFLATION REDUCTION ACT): PUBLIC IMPACT 
====================================================================================================================================================================
Completed Steps:
1. Calculate the baseline marginal damages for climate and health-related emissions
2. Calculate the post-retrofit marginal damages for climate and health-related emissions

---------------------------------------------------------------------------------------------
Step 3: Discount climate and health impacts and calculate lifetime public impacts (public NPV)
---------------------------------------------------------------------------------------------

RESULTS OUTPUT:

-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
Baseline Climate DataFrame: ✓ Valid
Baseline Health DataFrame: ✓ Valid
Retrofit Climate DataFrame: ✓ Valid
Retrofit Health DataFrame: ✓ Valid
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Verifying masking for all calculated columns:
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
Baseline Climate DataFrame: ✓ Valid
Baseline Health DataFrame: ✓ Valid
Retrofit Climate DataFrame: ✓ Valid
Retrofit Health DataFrame: ✓ Valid
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Verifying masking for all calculated columns:
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
Baseline Climate DataFrame: ✓ Valid
Baseline Health DataFrame: ✓ Valid
Retrofit Climate DataFrame: ✓ Valid
Retrofit Health DataFrame: ✓ Valid
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Verifying masking for all calculated columns:
  
====================================================================================================================================================================
DATAFRAME FOR mp9 AFTER CALCULATING PUBLIC NPV (NO IRA): df_euss_am_mp9_home

-----------------------------------------------
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

        preIRA_mp9_clothesDrying_public_npv_central_ap2_h6c  \
bldg_id                                                       
119                                                -117.74    
122                                                 159.93    
150                                                 132.37    
153                                                    NaN    
162                                                 574.56    
...                                                    ...    
549882                                              247.75    
549915                                              415.47    
549937                                              -57.70    
549963                                              190.46    
549989                                              199.98    

        preIRA_mp9_clothesDrying_climate_npv_upper  \
bldg_id                                              
119                                         121.30   
122                                         179.26   
150                                         208.20   
153                                            NaN   
162                                         643.99   
...                                            ...   
549882                                      288.69   
549915                                      502.81   
549937                                      307.95   
549963                                      269.67   
549989                                      260.12   

        preIRA_mp9_clothesDrying_public_npv_upper_ap2_h6c  \
bldg_id                                                     
119                                                -30.69   
122                                                289.13   
150                                                282.40   
153                                                   NaN   
162                                               1038.67   
...                                                   ...   
549882                                             455.79   
549915                                             777.83   
549937                                             163.60   
549963                                             384.68   
549989                                             387.44   

        preIRA_mp9_cooking_climate_npv_lower  \
bldg_id                                        
119                                     7.27   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                  4.84   
549963                                   NaN   
549989                                 17.78   

         preIRA_mp9_cooking_health_npv_ap2_h6c  \
bldg_id                                          
119                                    -336.58   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                 -152.22   
549963                                     NaN   
549989                                 -234.03   

         preIRA_mp9_cooking_public_npv_lower_ap2_h6c  \
bldg_id                                                
119                                          -329.31   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                       -147.39   
549963                                           NaN   
549989                                       -216.25   

        preIRA_mp9_cooking_climate_npv_central  \
bldg_id                                          
119                                      21.95   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                   14.85   
549963                                     NaN   
549989                                   55.47   

         preIRA_mp9_cooking_public_npv_central_ap2_h6c  \
bldg_id                                                  
119                                            -314.63   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                         -137.37   
549963                                             NaN   
549989                                         -178.56   

        preIRA_mp9_cooking_climate_npv_upper  \
bldg_id                                        
119                                    75.80   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                 51.41   
549963                                   NaN   
549989                                195.32   

        preIRA_mp9_cooking_public_npv_upper_ap2_h6c  
bldg_id                                              
119                                         -260.77  
122                                             NaN  
150                                             NaN  
153                                             NaN  
162                                             NaN  
...                                             ...  
549882                                          NaN  
549915                                          NaN  
549937                                      -100.82  
549963                                          NaN  
549989                                       -38.71  

[15651 rows x 597 columns]

-----------------------------------------------
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

        preIRA_mp9_clothesDrying_public_npv_central_easiur_h6c  \
bldg_id                                                          
119                                                 -15.54       
122                                                 176.45       
150                                                 173.07       
153                                                    NaN       
162                                                 633.98       
...                                                    ...       
549882                                              273.80       
549915                                              467.21       
549937                                               30.27       
549963                                              151.32       
549989                                              191.51       

        preIRA_mp9_clothesDrying_climate_npv_upper  \
bldg_id                                              
119                                         121.30   
122                                         179.26   
150                                         208.20   
153                                            NaN   
162                                         643.99   
...                                            ...   
549882                                      288.69   
549915                                      502.81   
549937                                      307.95   
549963                                      269.67   
549989                                      260.12   

        preIRA_mp9_clothesDrying_public_npv_upper_easiur_h6c  \
bldg_id                                                        
119                                                  71.51     
122                                                 305.65     
150                                                 323.10     
153                                                    NaN     
162                                                1098.10     
...                                                    ...     
549882                                              481.84     
549915                                              829.57     
549937                                              251.58     
549963                                              345.54     
549989                                              378.98     

        preIRA_mp9_cooking_climate_npv_lower  \
bldg_id                                        
119                                     7.27   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                  4.84   
549963                                   NaN   
549989                                 17.78   

         preIRA_mp9_cooking_health_npv_easiur_h6c  \
bldg_id                                             
119                                       -174.24   
122                                           NaN   
150                                           NaN   
153                                           NaN   
162                                           NaN   
...                                           ...   
549882                                        NaN   
549915                                        NaN   
549937                                     -89.25   
549963                                        NaN   
549989                                    -127.55   

         preIRA_mp9_cooking_public_npv_lower_easiur_h6c  \
bldg_id                                                   
119                                             -166.97   
122                                                 NaN   
150                                                 NaN   
153                                                 NaN   
162                                                 NaN   
...                                                 ...   
549882                                              NaN   
549915                                              NaN   
549937                                           -84.42   
549963                                              NaN   
549989                                          -109.77   

        preIRA_mp9_cooking_climate_npv_central  \
bldg_id                                          
119                                      21.95   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                   14.85   
549963                                     NaN   
549989                                   55.47   

         preIRA_mp9_cooking_public_npv_central_easiur_h6c  \
bldg_id                                                     
119                                               -152.30   
122                                                   NaN   
150                                                   NaN   
153                                                   NaN   
162                                                   NaN   
...                                                   ...   
549882                                                NaN   
549915                                                NaN   
549937                                             -74.40   
549963                                                NaN   
549989                                             -72.08   

        preIRA_mp9_cooking_climate_npv_upper  \
bldg_id                                        
119                                    75.80   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                 51.41   
549963                                   NaN   
549989                                195.32   

        preIRA_mp9_cooking_public_npv_upper_easiur_h6c  
bldg_id                                                 
119                                             -98.44  
122                                                NaN  
150                                                NaN  
153                                                NaN  
162                                                NaN  
...                                                ...  
549882                                             NaN  
549915                                             NaN  
549937                                          -37.85  
549963                                             NaN  
549989                                           67.77  

[15651 rows x 597 columns]

-----------------------------------------------
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

        preIRA_mp9_clothesDrying_public_npv_central_inmap_h6c  \
bldg_id                                                         
119                                                  91.08      
122                                                 133.15      
150                                                 239.25      
153                                                    NaN      
162                                                 478.43      
...                                                    ...      
549882                                              364.93      
549915                                              507.14      
549937                                              -40.09      
549963                                              160.81      
549989                                              226.83      

        preIRA_mp9_clothesDrying_climate_npv_upper  \
bldg_id                                              
119                                         121.30   
122                                         179.26   
150                                         208.20   
153                                            NaN   
162                                         643.99   
...                                            ...   
549882                                      288.69   
549915                                      502.81   
549937                                      307.95   
549963                                      269.67   
549989                                      260.12   

        preIRA_mp9_clothesDrying_public_npv_upper_inmap_h6c  \
bldg_id                                                       
119                                                 178.12    
122                                                 262.35    
150                                                 389.28    
153                                                    NaN    
162                                                 942.54    
...                                                    ...    
549882                                              572.97    
549915                                              869.50    
549937                                              181.21    
549963                                              355.03    
549989                                              414.30    

        preIRA_mp9_cooking_climate_npv_lower  \
bldg_id                                        
119                                     7.27   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                  4.84   
549963                                   NaN   
549989                                 17.78   

         preIRA_mp9_cooking_health_npv_inmap_h6c  \
bldg_id                                            
119                                        43.29   
122                                          NaN   
150                                          NaN   
153                                          NaN   
162                                          NaN   
...                                          ...   
549882                                       NaN   
549915                                       NaN   
549937                                   -148.32   
549963                                       NaN   
549989                                   -212.48   

         preIRA_mp9_cooking_public_npv_lower_inmap_h6c  \
bldg_id                                                  
119                                              50.56   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                         -143.48   
549963                                             NaN   
549989                                         -194.70   

        preIRA_mp9_cooking_climate_npv_central  \
bldg_id                                          
119                                      21.95   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                   14.85   
549963                                     NaN   
549989                                   55.47   

         preIRA_mp9_cooking_public_npv_central_inmap_h6c  \
bldg_id                                                    
119                                                65.24   
122                                                  NaN   
150                                                  NaN   
153                                                  NaN   
162                                                  NaN   
...                                                  ...   
549882                                               NaN   
549915                                               NaN   
549937                                           -133.47   
549963                                               NaN   
549989                                           -157.02   

        preIRA_mp9_cooking_climate_npv_upper  \
bldg_id                                        
119                                    75.80   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                 51.41   
549963                                   NaN   
549989                                195.32   

        preIRA_mp9_cooking_public_npv_upper_inmap_h6c  
bldg_id                                                
119                                            119.10  
122                                               NaN  
150                                               NaN  
153                                               NaN  
162                                               NaN  
...                                               ...  
549882                                            NaN  
549915                                            NaN  
549937                                         -96.91  
549963                                            NaN  
549989                                         -17.16  

[15651 rows x 597 columns]
      


====================================================================================================================================================================
SCENARIO ANALYSIS (AEO2023 REFERENCE CASE): PUBLIC IMPACT 
====================================================================================================================================================================
Completed Steps:
1. Calculate the baseline marginal damages for climate and health-related emissions
2. Calculate the post-retrofit marginal damages for climate and health-related emissions

---------------------------------------------------------------------------------------------
Step 3: Discount climate and health impacts and calculate lifetime public impacts (public NPV)
---------------------------------------------------------------------------------------------

RESULTS OUTPUT (AEO2023 REFERENCE CASE):

-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              
Baseline Climate DataFrame: ✓ Valid
Baseline Health DataFrame: ✓ Valid
Retrofit Climate DataFrame: ✓ Valid
Retrofit Health DataFrame: ✓ Valid
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Verifying masking for all calculated columns:
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              
Baseline Climate DataFrame: ✓ Valid
Baseline Health DataFrame: ✓ Valid
Retrofit Climate DataFrame: ✓ Valid
Retrofit Health DataFrame: ✓ Valid
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Verifying masking for all calculated columns:
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              
Baseline Climate DataFrame: ✓ Valid
Baseline Health DataFrame: ✓ Valid
Retrofit Climate DataFrame: ✓ Valid
Retrofit Health DataFrame: ✓ Valid
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              
-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'
              

Verifying masking for all calculated columns:
  
=====================================================================================================================================================================
DATAFRAME FOR mp9 AFTER CALCULATING PUBLIC NPV (AEO2023 REFERENCE CASE): df_euss_am_mp9_home

-----------------------------------------------
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

        iraRef_mp9_clothesDrying_public_npv_central_ap2_h6c  \
bldg_id                                                       
119                                                 -96.87    
122                                                 170.78    
150                                                 144.97    
153                                                    NaN    
162                                                 613.50    
...                                                    ...    
549882                                              265.25    
549915                                              445.90    
549937                                              -22.12    
549963                                              200.05    
549989                                              215.72    

        iraRef_mp9_clothesDrying_climate_npv_upper  \
bldg_id                                              
119                                         196.64   
122                                         218.44   
150                                         253.64   
153                                            NaN   
162                                         784.48   
...                                            ...   
549882                                      351.76   
549915                                      612.61   
549937                                      435.67   
549963                                      304.09   
549989                                      316.95   

        iraRef_mp9_clothesDrying_public_npv_upper_ap2_h6c  \
bldg_id                                                     
119                                                 44.65   
122                                                328.31   
150                                                327.84   
153                                                   NaN   
162                                               1179.17   
...                                                   ...   
549882                                             518.86   
549915                                             887.63   
549937                                             291.32   
549963                                             419.10   
549989                                             444.27   

        iraRef_mp9_cooking_climate_npv_lower  \
bldg_id                                        
119                                    18.43   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                 15.74   
549963                                   NaN   
549989                                 27.34   

         iraRef_mp9_cooking_health_npv_ap2_h6c  \
bldg_id                                          
119                                    -336.58   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                 -152.22   
549963                                     NaN   
549989                                 -234.03   

         iraRef_mp9_cooking_public_npv_lower_ap2_h6c  \
bldg_id                                                
119                                          -318.15   
122                                              NaN   
150                                              NaN   
153                                              NaN   
162                                              NaN   
...                                              ...   
549882                                           NaN   
549915                                           NaN   
549937                                       -136.48   
549963                                           NaN   
549989                                       -206.70   

        iraRef_mp9_cooking_climate_npv_central  \
bldg_id                                          
119                                      58.40   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                   49.63   
549963                                     NaN   
549989                                   86.64   

         iraRef_mp9_cooking_public_npv_central_ap2_h6c  \
bldg_id                                                  
119                                            -278.18   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                         -102.60   
549963                                             NaN   
549989                                         -147.39   

        iraRef_mp9_cooking_climate_npv_upper  \
bldg_id                                        
119                                   207.26   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                175.72   
549963                                   NaN   
549989                                307.69   

        iraRef_mp9_cooking_public_npv_upper_ap2_h6c  
bldg_id                                              
119                                         -129.32  
122                                             NaN  
150                                             NaN  
153                                             NaN  
162                                             NaN  
...                                             ...  
549882                                          NaN  
549915                                          NaN  
549937                                        23.50  
549963                                          NaN  
549989                                        73.66  

[15651 rows x 697 columns]

-----------------------------------------------
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

        iraRef_mp9_clothesDrying_public_npv_central_easiur_h6c  \
bldg_id                                                          
119                                                   5.33       
122                                                 187.30       
150                                                 185.66       
153                                                    NaN       
162                                                 672.92       
...                                                    ...       
549882                                              291.30       
549915                                              497.64       
549937                                               65.85       
549963                                              160.92       
549989                                              207.26       

        iraRef_mp9_clothesDrying_climate_npv_upper  \
bldg_id                                              
119                                         196.64   
122                                         218.44   
150                                         253.64   
153                                            NaN   
162                                         784.48   
...                                            ...   
549882                                      351.76   
549915                                      612.61   
549937                                      435.67   
549963                                      304.09   
549989                                      316.95   

        iraRef_mp9_clothesDrying_public_npv_upper_easiur_h6c  \
bldg_id                                                        
119                                                 146.85     
122                                                 344.83     
150                                                 368.54     
153                                                    NaN     
162                                                1238.59     
...                                                    ...     
549882                                              544.91     
549915                                              939.37     
549937                                              379.29     
549963                                              379.97     
549989                                              435.81     

        iraRef_mp9_cooking_climate_npv_lower  \
bldg_id                                        
119                                    18.43   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                 15.74   
549963                                   NaN   
549989                                 27.34   

         iraRef_mp9_cooking_health_npv_easiur_h6c  \
bldg_id                                             
119                                       -174.24   
122                                           NaN   
150                                           NaN   
153                                           NaN   
162                                           NaN   
...                                           ...   
549882                                        NaN   
549915                                        NaN   
549937                                     -89.25   
549963                                        NaN   
549989                                    -127.55   

         iraRef_mp9_cooking_public_npv_lower_easiur_h6c  \
bldg_id                                                   
119                                             -155.82   
122                                                 NaN   
150                                                 NaN   
153                                                 NaN   
162                                                 NaN   
...                                                 ...   
549882                                              NaN   
549915                                              NaN   
549937                                           -73.51   
549963                                              NaN   
549989                                          -100.21   

        iraRef_mp9_cooking_climate_npv_central  \
bldg_id                                          
119                                      58.40   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                   49.63   
549963                                     NaN   
549989                                   86.64   

         iraRef_mp9_cooking_public_npv_central_easiur_h6c  \
bldg_id                                                     
119                                               -115.85   
122                                                   NaN   
150                                                   NaN   
153                                                   NaN   
162                                                   NaN   
...                                                   ...   
549882                                                NaN   
549915                                                NaN   
549937                                             -39.62   
549963                                                NaN   
549989                                             -40.91   

        iraRef_mp9_cooking_climate_npv_upper  \
bldg_id                                        
119                                   207.26   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                175.72   
549963                                   NaN   
549989                                307.69   

        iraRef_mp9_cooking_public_npv_upper_easiur_h6c  
bldg_id                                                 
119                                              33.01  
122                                                NaN  
150                                                NaN  
153                                                NaN  
162                                                NaN  
...                                                ...  
549882                                             NaN  
549915                                             NaN  
549937                                           86.47  
549963                                             NaN  
549989                                          180.14  

[15651 rows x 697 columns]

-----------------------------------------------
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

        iraRef_mp9_clothesDrying_public_npv_central_inmap_h6c  \
bldg_id                                                         
119                                                 111.95      
122                                                 144.00      
150                                                 251.84      
153                                                    NaN      
162                                                 517.37      
...                                                    ...      
549882                                              382.42      
549915                                              537.57      
549937                                               -4.51      
549963                                              170.40      
549989                                              242.58      

        iraRef_mp9_clothesDrying_climate_npv_upper  \
bldg_id                                              
119                                         196.64   
122                                         218.44   
150                                         253.64   
153                                            NaN   
162                                         784.48   
...                                            ...   
549882                                      351.76   
549915                                      612.61   
549937                                      435.67   
549963                                      304.09   
549989                                      316.95   

        iraRef_mp9_clothesDrying_public_npv_upper_inmap_h6c  \
bldg_id                                                       
119                                                 253.46    
122                                                 301.53    
150                                                 434.72    
153                                                    NaN    
162                                                1083.04    
...                                                    ...    
549882                                              636.03    
549915                                              979.31    
549937                                              308.93    
549963                                              389.46    
549989                                              471.13    

        iraRef_mp9_cooking_climate_npv_lower  \
bldg_id                                        
119                                    18.43   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                 15.74   
549963                                   NaN   
549989                                 27.34   

         iraRef_mp9_cooking_health_npv_inmap_h6c  \
bldg_id                                            
119                                        43.29   
122                                          NaN   
150                                          NaN   
153                                          NaN   
162                                          NaN   
...                                          ...   
549882                                       NaN   
549915                                       NaN   
549937                                   -148.32   
549963                                       NaN   
549989                                   -212.48   

         iraRef_mp9_cooking_public_npv_lower_inmap_h6c  \
bldg_id                                                  
119                                              61.72   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                         -132.57   
549963                                             NaN   
549989                                         -185.15   

        iraRef_mp9_cooking_climate_npv_central  \
bldg_id                                          
119                                      58.40   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                   49.63   
549963                                     NaN   
549989                                   86.64   

         iraRef_mp9_cooking_public_npv_central_inmap_h6c  \
bldg_id                                                    
119                                               101.69   
122                                                  NaN   
150                                                  NaN   
153                                                  NaN   
162                                                  NaN   
...                                                  ...   
549882                                               NaN   
549915                                               NaN   
549937                                            -98.69   
549963                                               NaN   
549989                                           -125.85   

        iraRef_mp9_cooking_climate_npv_upper  \
bldg_id                                        
119                                   207.26   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                                175.72   
549963                                   NaN   
549989                                307.69   

        iraRef_mp9_cooking_public_npv_upper_inmap_h6c  
bldg_id                                                
119                                            250.55  
122                                               NaN  
150                                               NaN  
153                                               NaN  
162                                               NaN  
...                                               ...  
549882                                            NaN  
549915                                            NaN  
549937                                          27.40  
549963                                            NaN  
549989                                          95.21  

[15651 rows x 697 columns]    