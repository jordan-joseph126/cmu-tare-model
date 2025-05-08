# Lifetime Fuel Cost Sample Output

## df_main_sample DataFrame (created using df_euss_am_mp8_home)

```
        base_cooking_fuel  base_electricity_cooking_consumption  \
bldg_id                                                           
119           Natural Gas                                 32.53   
122           Electricity                                   NaN   
150           Electricity                                   NaN   
153           Electricity                                   NaN   
162           Electricity                                   NaN   
...                   ...                                   ...   
549882        Electricity                                   NaN   
549915        Electricity                                   NaN   
549937        Natural Gas                                 44.25   
549963        Electricity                                   NaN   
549989            Propane                                 27.84   

         base_naturalGas_cooking_consumption  \
bldg_id                                        
119                                   953.65   
122                                      NaN   
150                                      NaN   
153                                      NaN   
162                                      NaN   
...                                      ...   
549882                                   NaN   
549915                                   NaN   
549937                               1297.13   
549963                                   NaN   
549989                                  0.00   

         base_propane_cooking_consumption  valid_fuel_cooking  \
bldg_id                                                         
119                                  0.00                True   
122                                   NaN               False   
150                                   NaN               False   
153                                   NaN               False   
162                                   NaN               False   
...                                   ...                 ...   
549882                                NaN               False   
549915                                NaN               False   
549937                               0.00                True   
549963                                NaN               False   
549989                             813.57                True   

         include_cooking upgrade_cooking_range  \
bldg_id                                          
119                 True  Electric, 100% Usage   
122                False                   NaN   
150                False                   NaN   
153                False                   NaN   
162                False                   NaN   
...                  ...                   ...   
549882             False                   NaN   
549915             False                   NaN   
549937              True  Electric, 120% Usage   
549963             False                   NaN   
549989              True  Electric, 100% Usage   

         preIRA_mp8_cooking_lifetime_fuelCost  \
bldg_id                                         
119                                   1240.31   
122                                       NaN   
150                                       NaN   
153                                       NaN   
162                                       NaN   
...                                       ...   
549882                                    NaN   
549915                                    NaN   
549937                                1685.42   
549963                                    NaN   
549989                                1060.28   

         iraRef_mp8_cooking_lifetime_fuelCost  \
bldg_id                                         
119                                   1214.19   
122                                       NaN   
150                                       NaN   
153                                       NaN   
162                                       NaN   
...                                       ...   
549882                                    NaN   
549915                                    NaN   
549937                                1649.94   
549963                                    NaN   
549989                                1037.95   

         baseline_cooking_lifetime_fuelCost  
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

[15651 rows x 10 columns]
```

## df_detailed_test DataFrame (created using df_mp8_IRA_fuelCosts)

```
         include_cooking  valid_fuel_cooking  \
bldg_id                                        
119                 True                True   
122                False               False   
150                False               False   
153                False               False   
162                False               False   
...                  ...                 ...   
549882             False               False   
549915             False               False   
549937              True                True   
549963             False               False   
549989              True                True   

         iraRef_mp8_2024_heating_fuelCost  baseline_2024_heating_fuelCost  \
bldg_id                                                                     
119                                473.70                          731.14   
122                               1067.58                         4201.54   
150                                567.43                         3201.04   
153                               1172.57                         3908.99   
162                               1208.25                         5118.99   
...                                   ...                             ...   
549882                               0.00                            0.00   
549915                             804.44                         3467.36   
549937                             517.25                          637.89   
549963                             848.00                         1318.13   
549989                             338.66                         1653.32   

         iraRef_mp8_2025_heating_fuelCost  baseline_2025_heating_fuelCost  \
bldg_id                                                                     
119                                467.30                          691.86   
122                               1053.14                         3950.78   
150                                559.76                         3199.59   
153                               1156.71                         3675.69   
162                               1191.91                         4813.48   
...                                   ...                             ...   
549882                               0.00                            0.00   
549915                             793.57                         3465.80   
549937                             510.26                          603.63   
549963                             836.53                         1247.32   
549989                             334.08                         1554.65   

         iraRef_mp8_2026_heating_fuelCost  baseline_2026_heating_fuelCost  \
bldg_id                                                                     
119                                459.93                          661.81   
122                               1036.55                         3863.56   
150                                550.94                         3168.72   
153                               1138.49                         3594.55   
162                               1173.13                         4707.22   
...                                   ...                             ...   
549882                               0.00                            0.00   
549915                             781.07                         3432.36   
549937                             502.22                          577.41   
549963                             823.35                         1193.15   
549989                             328.82                         1520.33   

         iraRef_mp8_2027_heating_fuelCost  baseline_2027_heating_fuelCost  \
bldg_id                                                                     
119                                458.97                          646.55   
122                               1034.38                         3801.58   
150                                549.79                         3169.61   
153                               1136.11                         3536.88   
162                               1170.68                         4631.70   
...                                   ...                             ...   
549882                               0.00                            0.00   
549915                             779.43                         3433.32   
549937                             501.17                          564.10   
549963                             821.63                         1165.64   
549989                             328.13                         1495.94   

         ...  baseline_2035_cooking_fuelCost  \
bldg_id  ...                                   
119      ...                           39.02   
122      ...                             NaN   
150      ...                             NaN   
153      ...                             NaN   
162      ...                             NaN   
...      ...                             ...   
549882   ...                             NaN   
549915   ...                             NaN   
549937   ...                           53.07   
549963   ...                             NaN   
549989   ...                          115.40   

         iraRef_mp8_2036_cooking_fuelCost  baseline_2036_cooking_fuelCost  \
bldg_id                                                                     
119                                 84.01                           38.84   
122                                   NaN                             NaN   
150                                   NaN                             NaN   
153                                   NaN                             NaN   
162                                   NaN                             NaN   
...                                   ...                             ...   
549882                                NaN                             NaN   
549915                                NaN                             NaN   
549937                             114.15                           52.84   
549963                                NaN                             NaN   
549989                              71.81                          116.58   

         iraRef_mp8_2037_cooking_fuelCost  baseline_2037_cooking_fuelCost  \
bldg_id                                                                     
119                                 84.44                           39.25   
122                                   NaN                             NaN   
150                                   NaN                             NaN   
153                                   NaN                             NaN   
162                                   NaN                             NaN   
...                                   ...                             ...   
549882                                NaN                             NaN   
549915                                NaN                             NaN   
549937                             114.74                           53.38   
549963                                NaN                             NaN   
549989                              72.18                          117.97   

         iraRef_mp8_2038_cooking_fuelCost  baseline_2038_cooking_fuelCost  \
bldg_id                                                                     
119                                 85.44                           39.49   
122                                   NaN                             NaN   
150                                   NaN                             NaN   
153                                   NaN                             NaN   
162                                   NaN                             NaN   
...                                   ...                             ...   
549882                                NaN                             NaN   
549915                                NaN                             NaN   
549937                             116.10                           53.72   
549963                                NaN                             NaN   
549989                              73.04                          119.31   

         iraRef_mp8_cooking_lifetime_fuelCost  \
bldg_id                                         
119                                   1214.19   
122                                       NaN   
150                                       NaN   
153                                       NaN   
162                                       NaN   
...                                       ...   
549882                                    NaN   
549915                                    NaN   
549937                                1649.94   
549963                                    NaN   
549989                                1037.95   

         iraRef_mp8_cooking_lifetime_savings_fuelCost  \
bldg_id                                                 
119                                           -624.35   
122                                               NaN   
150                                               NaN   
153                                               NaN   
162                                               NaN   
...                                               ...   
549882                                            NaN   
549915                                            NaN   
549937                                        -847.64   
549963                                            NaN   
549989                                         653.60   

         baseline_cooking_lifetime_fuelCost  
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

[15651 rows x 124 columns]
```

**Key Observations:**
1. The output shows fuel cost calculations for different policy scenarios (baseline, preIRA, iraRef)
2. Values are properly masked for homes that don't meet validation criteria (shown as NaN)
3. Lifetime fuel costs are calculated for each equipment category
4. The detailed test dataframe shows yearly fuel costs from 2024-2038
5. Lifetime savings calculations appear to work correctly (comparing retrofit scenarios to baseline)
