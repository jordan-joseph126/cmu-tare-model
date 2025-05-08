# Lifetime Private Impact Sample Output

## Processing Log

```
-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'
              

Determining lifetime private impacts for category: heating with lifetime: 15
Measure package calculation for heating:
  - 12266 homes have valid baseline data
  - 15649 homes will receive retrofits
  - 12266 homes have both valid data AND will receive retrofits
  - 3385 homes excluded (values will be NaN)

Calculating costs for heating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for heating with lifetime: 15 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp8_
          

Determining lifetime private impacts for category: waterHeating with lifetime: 12
Measure package calculation for waterHeating:
  - 14999 homes have valid baseline data
  - 15551 homes will receive retrofits
  - 14999 homes have both valid data AND will receive retrofits
  - 652 homes excluded (values will be NaN)

Calculating costs for waterHeating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for waterHeating with lifetime: 12 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp8_
          

Determining lifetime private impacts for category: clothesDrying with lifetime: 13
Measure package calculation for clothesDrying:
  - 14743 homes have valid baseline data
  - 14743 homes will receive retrofits
  - 14743 homes have both valid data AND will receive retrofits
  - 908 homes excluded (values will be NaN)

Calculating costs for clothesDrying...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for clothesDrying with lifetime: 13 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp8_
          

Determining lifetime private impacts for category: cooking with lifetime: 15
Measure package calculation for cooking:
  - 7867 homes have valid baseline data
  - 7867 homes will receive retrofits
  - 7867 homes have both valid data AND will receive retrofits
  - 7784 homes excluded (values will be NaN)

Calculating costs for cooking...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for cooking with lifetime: 15 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp8_
          

Verifying masking for all calculated columns:

Verifying masking for all calculated columns:
Masking 4 columns for category 'cooking'

Private NPV calculation completed. Added 27 new columns.
```

## Function Signature

```python
def calculate_private_NPV(
        df: pd.DataFrame,
        df_fuel_costs: pd.DataFrame,
        df_baseline_costs: pd.DataFrame,
        input_mp: str,
        menu_mp: int,
        policy_scenario: str,
        discounting_method: str = 'private_fixed',
        base_year: int = 2024,
        verbose: bool = True
) -> pd.DataFrame:
```

## Results DataFrame (df_euss_am_mp8_home)

### AP2 Model Output Sample
```
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

        preIRA_mp8_waterHeating_private_npv_lessWTP  \
bldg_id                                               
119                                        -1765.16   
122                                        -1383.53   
150                                        -2806.06   
153                                         2542.63   
162                                        -3134.66   
...                                             ...   
549882                                     -3506.17   
549915                                       340.91   
549937                                     -2360.91   
549963                                     -2209.05   
549989                                      -430.73   

        preIRA_mp8_waterHeating_private_npv_moreWTP  \
bldg_id                                               
119                                          126.32   
122                                         1478.91   
150                                        -1732.07   
153                                         5250.19   
162                                        -1041.72   
...                                             ...   
549882                                     -1844.93   
549915                                      1479.02   
549937                                      -469.43   
549963                                      -547.81   
549989                                      2276.83   

        preIRA_mp8_clothesDrying_total_capitalCost  \
bldg_id                                              
119                                        1821.14   
122                                        2557.23   
150                                        2762.90   
153                                            NaN   
162                                        2839.30   
...                                            ...   
549882                                     2284.80   
549915                                     1923.19   
549937                                     2054.78   
549963                                     2854.96   
549989                                     3586.93   

        preIRA_mp8_clothesDrying_net_capitalCost  \
bldg_id                                            
119                                       513.16   
122                                      1563.17   
150                                      1768.84   
153                                          NaN   
162                                      1845.24   
...                                          ...   
549882                                   1290.74   
549915                                    929.13   
549937                                    746.80   
549963                                   1860.90   
549989                                   2592.87   

         preIRA_mp8_clothesDrying_private_npv_lessWTP  \
bldg_id                                                 
119                                          -2000.47   
122                                          -2354.29   
150                                          -2527.18   
153                                               NaN   
162                                          -2110.15   
...                                               ...   
549882                                       -1957.95   
549915                                       -1353.88   
549937                                       -2556.39   
549963                                       -2560.43   
549989                                       -3292.40   

         preIRA_mp8_clothesDrying_private_npv_moreWTP  \
bldg_id                                                 
119                                           -692.49   
122                                          -1360.23   
150                                          -1533.12   
153                                               NaN   
162                                          -1116.09   
...                                               ...   
549882                                        -963.89   
549915                                        -359.82   
549937                                       -1248.41   
549963                                       -1566.37   
549989                                       -2298.34   

        preIRA_mp8_cooking_total_capitalCost  \
bldg_id                                        
119                                  1226.37   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                               1464.24   
549963                                   NaN   
549989                               1731.32   

         preIRA_mp8_cooking_net_capitalCost  \
bldg_id                                       
119                                  153.83   
122                                     NaN   
150                                     NaN   
153                                     NaN   
162                                     NaN   
...                                     ...   
549882                                  NaN   
549915                                  NaN   
549937                               391.70   
549963                                  NaN   
549989                               658.78   

        preIRA_mp8_cooking_private_npv_lessWTP  \
bldg_id                                          
119                                   -1638.41   
122                                        NaN   
150                                        NaN   
153                                        NaN   
162                                        NaN   
...                                        ...   
549882                                     NaN   
549915                                     NaN   
549937                                -2023.65   
549963                                     NaN   
549989                                -1318.11   

        preIRA_mp8_cooking_private_npv_moreWTP  
bldg_id                                         
119                                    -565.87  
122                                        NaN  
150                                        NaN  
153                                        NaN  
162                                        NaN  
...                                        ...  
549882                                     NaN  
549915                                     NaN  
549937                                 -951.11  
549963                                     NaN  
549989                                 -245.57  

[15651 rows x 597 columns]
```

**Key Observations:**
1. The output shows private NPV calculations for different equipment categories (heating, waterHeating, clothesDrying, cooking)
2. Processing correctly handles valid homes vs excluded homes using the validation framework
3. Multiple NPV metrics are calculated (lessWTP, moreWTP variants)
4. For each category, the process calculates:
   - Capital costs (total and net)
   - Private NPV calculations with different willingness-to-pay assumptions
5. Values are properly masked for homes that don't meet validation criteria (shown as NaN)
6. The same calculations are performed across different health impact models (AP2, EASIUR, InMAP)
