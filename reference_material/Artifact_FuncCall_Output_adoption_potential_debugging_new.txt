Checking for and fixing duplicate columns...

=== FIXING DUPLICATE COLUMNS ===
Found 24 duplicate column names:
  'preIRA_mp8_clothesDrying_climate_npv_central': appears 2 times
  'iraRef_mp8_heating_climate_npv_lower': appears 2 times
  'iraRef_mp8_waterHeating_climate_npv_lower': appears 2 times
  'iraRef_mp8_waterHeating_climate_npv_central': appears 2 times
  'iraRef_mp8_waterHeating_climate_npv_upper': appears 2 times
  'iraRef_mp8_clothesDrying_climate_npv_lower': appears 2 times
  'iraRef_mp8_clothesDrying_climate_npv_central': appears 2 times
  'iraRef_mp8_clothesDrying_climate_npv_upper': appears 2 times
  'iraRef_mp8_cooking_climate_npv_lower': appears 2 times
  'iraRef_mp8_cooking_climate_npv_central': appears 2 times
  'iraRef_mp8_cooking_climate_npv_upper': appears 2 times
  'preIRA_mp8_heating_climate_npv_lower': appears 2 times
  'preIRA_mp8_heating_climate_npv_central': appears 2 times
  'preIRA_mp8_heating_climate_npv_upper': appears 2 times
  'preIRA_mp8_waterHeating_climate_npv_lower': appears 2 times
  'preIRA_mp8_waterHeating_climate_npv_central': appears 2 times
  'preIRA_mp8_waterHeating_climate_npv_upper': appears 2 times
  'preIRA_mp8_clothesDrying_climate_npv_lower': appears 2 times
  'preIRA_mp8_cooking_climate_npv_upper': appears 2 times
  'preIRA_mp8_cooking_climate_npv_central': appears 2 times
  'preIRA_mp8_clothesDrying_climate_npv_upper': appears 2 times
  'iraRef_mp8_heating_climate_npv_central': appears 2 times
  'iraRef_mp8_heating_climate_npv_upper': appears 2 times
  'preIRA_mp8_cooking_climate_npv_lower': appears 2 times
  Renamed duplicate 'preIRA_mp8_heating_climate_npv_lower' → 'preIRA_mp8_heating_climate_npv_lower_duplicate_1'
  Renamed duplicate 'preIRA_mp8_heating_climate_npv_central' → 'preIRA_mp8_heating_climate_npv_central_duplicate_1'
  Renamed duplicate 'preIRA_mp8_heating_climate_npv_upper' → 'preIRA_mp8_heating_climate_npv_upper_duplicate_1'
  Renamed duplicate 'preIRA_mp8_waterHeating_climate_npv_lower' → 'preIRA_mp8_waterHeating_climate_npv_lower_duplicate_1'
  Renamed duplicate 'preIRA_mp8_waterHeating_climate_npv_central' → 'preIRA_mp8_waterHeating_climate_npv_central_duplicate_1'
  Renamed duplicate 'preIRA_mp8_waterHeating_climate_npv_upper' → 'preIRA_mp8_waterHeating_climate_npv_upper_duplicate_1'
  Renamed duplicate 'preIRA_mp8_clothesDrying_climate_npv_lower' → 'preIRA_mp8_clothesDrying_climate_npv_lower_duplicate_1'
  Renamed duplicate 'preIRA_mp8_clothesDrying_climate_npv_central' → 'preIRA_mp8_clothesDrying_climate_npv_central_duplicate_1'
  Renamed duplicate 'preIRA_mp8_clothesDrying_climate_npv_upper' → 'preIRA_mp8_clothesDrying_climate_npv_upper_duplicate_1'
  Renamed duplicate 'preIRA_mp8_cooking_climate_npv_lower' → 'preIRA_mp8_cooking_climate_npv_lower_duplicate_1'
  Renamed duplicate 'preIRA_mp8_cooking_climate_npv_central' → 'preIRA_mp8_cooking_climate_npv_central_duplicate_1'
  Renamed duplicate 'preIRA_mp8_cooking_climate_npv_upper' → 'preIRA_mp8_cooking_climate_npv_upper_duplicate_1'
  Renamed duplicate 'iraRef_mp8_heating_climate_npv_lower' → 'iraRef_mp8_heating_climate_npv_lower_duplicate_1'
  Renamed duplicate 'iraRef_mp8_heating_climate_npv_central' → 'iraRef_mp8_heating_climate_npv_central_duplicate_1'
  Renamed duplicate 'iraRef_mp8_heating_climate_npv_upper' → 'iraRef_mp8_heating_climate_npv_upper_duplicate_1'
  Renamed duplicate 'iraRef_mp8_waterHeating_climate_npv_lower' → 'iraRef_mp8_waterHeating_climate_npv_lower_duplicate_1'
  Renamed duplicate 'iraRef_mp8_waterHeating_climate_npv_central' → 'iraRef_mp8_waterHeating_climate_npv_central_duplicate_1'
  Renamed duplicate 'iraRef_mp8_waterHeating_climate_npv_upper' → 'iraRef_mp8_waterHeating_climate_npv_upper_duplicate_1'
  Renamed duplicate 'iraRef_mp8_clothesDrying_climate_npv_lower' → 'iraRef_mp8_clothesDrying_climate_npv_lower_duplicate_1'
  Renamed duplicate 'iraRef_mp8_clothesDrying_climate_npv_central' → 'iraRef_mp8_clothesDrying_climate_npv_central_duplicate_1'
  Renamed duplicate 'iraRef_mp8_clothesDrying_climate_npv_upper' → 'iraRef_mp8_clothesDrying_climate_npv_upper_duplicate_1'
  Renamed duplicate 'iraRef_mp8_cooking_climate_npv_lower' → 'iraRef_mp8_cooking_climate_npv_lower_duplicate_1'
  Renamed duplicate 'iraRef_mp8_cooking_climate_npv_central' → 'iraRef_mp8_cooking_climate_npv_central_duplicate_1'
  Renamed duplicate 'iraRef_mp8_cooking_climate_npv_upper' → 'iraRef_mp8_cooking_climate_npv_upper_duplicate_1'
✅ Fixed duplicate columns. DataFrame now has 966 unique column names
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
Starting climate-only adoption analysis for No Inflation Reduction Act

Processing climate-only analysis for heating...
Measure package calculation for heating:
  - 12266 homes have valid data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
  Processing SCC assumption: lower
    Checking required columns for heating/lower...
    ✅ Found column: preIRA_mp8_heating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_heating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_heating_climate_npv_lower
    Converting columns to numeric for heating/lower...
    ✅ Successfully converted all columns for heating/lower
    ✅ Completed climate-only analysis for heating/lower
  Processing SCC assumption: central
    Checking required columns for heating/central...
    ✅ Found column: preIRA_mp8_heating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_heating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_heating_climate_npv_central
    Converting columns to numeric for heating/central...
    ✅ Successfully converted all columns for heating/central
    ✅ Completed climate-only analysis for heating/central
  Processing SCC assumption: upper
    Checking required columns for heating/upper...
    ✅ Found column: preIRA_mp8_heating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_heating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_heating_climate_npv_upper
    Converting columns to numeric for heating/upper...
    ✅ Successfully converted all columns for heating/upper
    ✅ Completed climate-only analysis for heating/upper

Processing climate-only analysis for waterHeating...
Measure package calculation for waterHeating:
  - 14999 homes have valid data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
  Processing SCC assumption: lower
    Checking required columns for waterHeating/lower...
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_waterHeating_climate_npv_lower
    Converting columns to numeric for waterHeating/lower...
    ✅ Successfully converted all columns for waterHeating/lower
    ✅ Completed climate-only analysis for waterHeating/lower
  Processing SCC assumption: central
    Checking required columns for waterHeating/central...
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_waterHeating_climate_npv_central
    Converting columns to numeric for waterHeating/central...
    ✅ Successfully converted all columns for waterHeating/central
    ✅ Completed climate-only analysis for waterHeating/central
  Processing SCC assumption: upper
    Checking required columns for waterHeating/upper...
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_waterHeating_climate_npv_upper
    Converting columns to numeric for waterHeating/upper...
    ✅ Successfully converted all columns for waterHeating/upper
    ✅ Completed climate-only analysis for waterHeating/upper

Processing climate-only analysis for clothesDrying...
Measure package calculation for clothesDrying:
  - 14743 homes have valid data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
  Processing SCC assumption: lower
    Checking required columns for clothesDrying/lower...
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_clothesDrying_climate_npv_lower
    Converting columns to numeric for clothesDrying/lower...
    ✅ Successfully converted all columns for clothesDrying/lower
    ✅ Completed climate-only analysis for clothesDrying/lower
  Processing SCC assumption: central
    Checking required columns for clothesDrying/central...
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_clothesDrying_climate_npv_central
    Converting columns to numeric for clothesDrying/central...
    ✅ Successfully converted all columns for clothesDrying/central
    ✅ Completed climate-only analysis for clothesDrying/central
  Processing SCC assumption: upper
    Checking required columns for clothesDrying/upper...
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_clothesDrying_climate_npv_upper
    Converting columns to numeric for clothesDrying/upper...
    ✅ Successfully converted all columns for clothesDrying/upper
    ✅ Completed climate-only analysis for clothesDrying/upper

Processing climate-only analysis for cooking...
Measure package calculation for cooking:
  - 7867 homes have valid data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
  Processing SCC assumption: lower
    Checking required columns for cooking/lower...
    ✅ Found column: preIRA_mp8_cooking_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_cooking_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_cooking_climate_npv_lower
    Converting columns to numeric for cooking/lower...
    ✅ Successfully converted all columns for cooking/lower
    ✅ Completed climate-only analysis for cooking/lower
  Processing SCC assumption: central
    Checking required columns for cooking/central...
    ✅ Found column: preIRA_mp8_cooking_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_cooking_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_cooking_climate_npv_central
    Converting columns to numeric for cooking/central...
    ✅ Successfully converted all columns for cooking/central
    ✅ Completed climate-only analysis for cooking/central
  Processing SCC assumption: upper
    Checking required columns for cooking/upper...
    ✅ Found column: preIRA_mp8_cooking_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_cooking_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_cooking_climate_npv_upper
    Converting columns to numeric for cooking/upper...
    ✅ Successfully converted all columns for cooking/upper
    ✅ Completed climate-only analysis for cooking/upper

Verifying masking for all calculated columns:
Masking 48 columns for category 'cooking'
    preIRA_mp8_cooking_adoption_climateOnly_lower: Masked 7784 values
    preIRA_mp8_cooking_impact_climateOnly_lower: Masked 7784 values
    preIRA_mp8_cooking_adoption_climateOnly_central: Masked 7784 values
    preIRA_mp8_cooking_impact_climateOnly_central: Masked 7784 values
    preIRA_mp8_cooking_adoption_climateOnly_upper: Masked 7784 values
    preIRA_mp8_cooking_impact_climateOnly_upper: Masked 7784 values
  Total: Masked 46704 values across 48 columns
✅ Climate-only adoption analysis completed successfully



AFTER RUNNING IT A SECOND TIME I GOT THE FOLLOWING OUTPUT. I ALSO INCLUDE THE PRINTED DF:

Checking for and fixing duplicate columns...

=== FIXING DUPLICATE COLUMNS ===
✅ No duplicate columns found
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              
Starting climate-only adoption analysis for No Inflation Reduction Act

Processing climate-only analysis for heating...
Measure package calculation for heating:
  - 12266 homes have valid data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)
  Processing SCC assumption: lower
    Checking required columns for heating/lower...
    ✅ Found column: preIRA_mp8_heating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_heating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_heating_climate_npv_lower
    Converting columns to numeric for heating/lower...
    ✅ Successfully converted all columns for heating/lower
    ✅ Completed climate-only analysis for heating/lower
  Processing SCC assumption: central
    Checking required columns for heating/central...
    ✅ Found column: preIRA_mp8_heating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_heating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_heating_climate_npv_central
    Converting columns to numeric for heating/central...
    ✅ Successfully converted all columns for heating/central
    ✅ Completed climate-only analysis for heating/central
  Processing SCC assumption: upper
    Checking required columns for heating/upper...
    ✅ Found column: preIRA_mp8_heating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_heating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_heating_climate_npv_upper
    Converting columns to numeric for heating/upper...
    ✅ Successfully converted all columns for heating/upper
    ✅ Completed climate-only analysis for heating/upper

Processing climate-only analysis for waterHeating...
Measure package calculation for waterHeating:
  - 14999 homes have valid data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)
  Processing SCC assumption: lower
    Checking required columns for waterHeating/lower...
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_waterHeating_climate_npv_lower
    Converting columns to numeric for waterHeating/lower...
    ✅ Successfully converted all columns for waterHeating/lower
    ✅ Completed climate-only analysis for waterHeating/lower
  Processing SCC assumption: central
    Checking required columns for waterHeating/central...
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_waterHeating_climate_npv_central
    Converting columns to numeric for waterHeating/central...
    ✅ Successfully converted all columns for waterHeating/central
    ✅ Completed climate-only analysis for waterHeating/central
  Processing SCC assumption: upper
    Checking required columns for waterHeating/upper...
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_waterHeating_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_waterHeating_climate_npv_upper
    Converting columns to numeric for waterHeating/upper...
    ✅ Successfully converted all columns for waterHeating/upper
    ✅ Completed climate-only analysis for waterHeating/upper

Processing climate-only analysis for clothesDrying...
Measure package calculation for clothesDrying:
  - 14743 homes have valid data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)
  Processing SCC assumption: lower
    Checking required columns for clothesDrying/lower...
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_clothesDrying_climate_npv_lower
    Converting columns to numeric for clothesDrying/lower...
    ✅ Successfully converted all columns for clothesDrying/lower
    ✅ Completed climate-only analysis for clothesDrying/lower
  Processing SCC assumption: central
    Checking required columns for clothesDrying/central...
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_clothesDrying_climate_npv_central
    Converting columns to numeric for clothesDrying/central...
    ✅ Successfully converted all columns for clothesDrying/central
    ✅ Completed climate-only analysis for clothesDrying/central
  Processing SCC assumption: upper
    Checking required columns for clothesDrying/upper...
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_clothesDrying_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_clothesDrying_climate_npv_upper
    Converting columns to numeric for clothesDrying/upper...
    ✅ Successfully converted all columns for clothesDrying/upper
    ✅ Completed climate-only analysis for clothesDrying/upper

Processing climate-only analysis for cooking...
Measure package calculation for cooking:
  - 7867 homes have valid data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)
  Processing SCC assumption: lower
    Checking required columns for cooking/lower...
    ✅ Found column: preIRA_mp8_cooking_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_cooking_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_cooking_climate_npv_lower
    Converting columns to numeric for cooking/lower...
    ✅ Successfully converted all columns for cooking/lower
    ✅ Completed climate-only analysis for cooking/lower
  Processing SCC assumption: central
    Checking required columns for cooking/central...
    ✅ Found column: preIRA_mp8_cooking_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_cooking_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_cooking_climate_npv_central
    Converting columns to numeric for cooking/central...
    ✅ Successfully converted all columns for cooking/central
    ✅ Completed climate-only analysis for cooking/central
  Processing SCC assumption: upper
    Checking required columns for cooking/upper...
    ✅ Found column: preIRA_mp8_cooking_private_npv_lessWTP
    ✅ Found column: preIRA_mp8_cooking_private_npv_moreWTP
    ✅ Found column: preIRA_mp8_cooking_climate_npv_upper
    Converting columns to numeric for cooking/upper...
    ✅ Successfully converted all columns for cooking/upper
    ✅ Completed climate-only analysis for cooking/upper

Verifying masking for all calculated columns:
Masking 48 columns for category 'cooking'
    preIRA_mp8_cooking_adoption_climateOnly_lower: Masked 7784 values
    preIRA_mp8_cooking_impact_climateOnly_lower: Masked 7784 values
    preIRA_mp8_cooking_adoption_climateOnly_central: Masked 7784 values
    preIRA_mp8_cooking_impact_climateOnly_central: Masked 7784 values
    preIRA_mp8_cooking_adoption_climateOnly_upper: Masked 7784 values
    preIRA_mp8_cooking_impact_climateOnly_upper: Masked 7784 values
  Total: Masked 46704 values across 48 columns
✅ Climate-only adoption analysis completed successfully

====================================================================================================================================================================
df_euss_am_mp8_home_inmap:

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

        preIRA_mp8_cooking_adoption_climateOnly_lower  \
bldg_id                                                 
119                                    Tier 4: Averse   
122                                               NaN   
150                                               NaN   
153                                               NaN   
162                                               NaN   
...                                               ...   
549882                                            NaN   
549915                                            NaN   
549937                                 Tier 4: Averse   
549963                                            NaN   
549989               Tier 2: Feasible vs. Alternative   

        preIRA_mp8_cooking_impact_climateOnly_lower  \
bldg_id                                               
119                                  Public Benefit   
122                                             NaN   
150                                             NaN   
153                                             NaN   
162                                             NaN   
...                                             ...   
549882                                          NaN   
549915                                          NaN   
549937                               Public Benefit   
549963                                          NaN   
549989                               Public Benefit   

        preIRA_mp8_cooking_total_npv_lessWTP_climateOnly_central  \
bldg_id                                                            
119                                               -1504.72         
122                                                    NaN         
150                                                    NaN         
153                                                    NaN         
162                                                    NaN         
...                                                    ...         
549882                                                 NaN         
549915                                                 NaN         
549937                                            -2621.26         
549963                                                 NaN         
549989                                             -277.03         

        preIRA_mp8_cooking_total_npv_moreWTP_climateOnly_central  \
bldg_id                                                            
119                                                -432.18         
122                                                    NaN         
150                                                    NaN         
153                                                    NaN         
162                                                    NaN         
...                                                    ...         
549882                                                 NaN         
549915                                                 NaN         
549937                                            -1548.72         
549963                                                 NaN         
549989                                              795.51         

         preIRA_mp8_cooking_adoption_climateOnly_central  \
bldg_id                                                    
119                                       Tier 4: Averse   
122                                                  NaN   
150                                                  NaN   
153                                                  NaN   
162                                                  NaN   
...                                                  ...   
549882                                               NaN   
549915                                               NaN   
549937                                    Tier 4: Averse   
549963                                               NaN   
549989                  Tier 2: Feasible vs. Alternative   

         preIRA_mp8_cooking_impact_climateOnly_central  \
bldg_id                                                  
119                                     Public Benefit   
122                                                NaN   
150                                                NaN   
153                                                NaN   
162                                                NaN   
...                                                ...   
549882                                             NaN   
549915                                             NaN   
549937                                  Public Benefit   
549963                                             NaN   
549989                                  Public Benefit   

        preIRA_mp8_cooking_total_npv_lessWTP_climateOnly_upper  \
bldg_id                                                          
119                                               -1448.87       
122                                                    NaN       
150                                                    NaN       
153                                                    NaN       
162                                                    NaN       
...                                                    ...       
549882                                                 NaN       
549915                                                 NaN       
549937                                            -2584.10       
549963                                                 NaN       
549989                                             -141.13       

         preIRA_mp8_cooking_total_npv_moreWTP_climateOnly_upper  \
bldg_id                                                           
119                                                -376.33        
122                                                    NaN        
150                                                    NaN        
153                                                    NaN        
162                                                    NaN        
...                                                    ...        
549882                                                 NaN        
549915                                                 NaN        
549937                                            -1511.56        
549963                                                 NaN        
549989                                              931.41        

        preIRA_mp8_cooking_adoption_climateOnly_upper  \
bldg_id                                                 
119                                    Tier 4: Averse   
122                                               NaN   
150                                               NaN   
153                                               NaN   
162                                               NaN   
...                                               ...   
549882                                            NaN   
549915                                            NaN   
549937                                 Tier 4: Averse   
549963                                            NaN   
549989               Tier 2: Feasible vs. Alternative   

        preIRA_mp8_cooking_impact_climateOnly_upper  
bldg_id                                              
119                                  Public Benefit  
122                                             NaN  
150                                             NaN  
153                                             NaN  
162                                             NaN  
...                                             ...  
549882                                          NaN  
549915                                          NaN  
549937                               Public Benefit  
549963                                          NaN  
549989                               Public Benefit  

[15651 rows x 1014 columns]