FUNCTION CALL FOR SPACE HEATING create_df_adoption:

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

EXAMPLE PRINTED OUTPUT OF ADOPTION POTENTIAL COLUMNS:

print(df_basic_summary_heating_ap2)
state                     city    county       puma  percent_AMI  \
bldg_id                                                                    
239        AL    Not in a census Place  G0100390  G01002300       190.14   
273        AL  In another census Place  G0100150  G01001100       162.63   
307        AL    Not in a census Place  G0100850  G01002100        67.03   
409        AL    Not in a census Place  G0100050  G01002400        43.05   
517        AL  In another census Place  G0101270  G01001400        65.16   
...       ...                      ...       ...        ...          ...   
548226     WY  In another census Place  G5600050  G56000200        68.65   
548228     WY    Not in a census Place  G5600010  G56000300       184.83   
548417     WY                   Casper  G5600250  G56000400        54.92   
549740     WY  In another census Place  G5600390  G56000100       185.72   
549938     WY    Not in a census Place  G5600010  G56000300        83.77   

        lowModerateIncome_designation base_heating_fuel  include_heating  \
bldg_id                                                                    
239            Middle-to-Upper-Income       Electricity             True   
273            Middle-to-Upper-Income       Electricity             True   
307                        Low-Income       Electricity            False   
409                        Low-Income       Natural Gas             True   
517                        Low-Income       Electricity            False   
...                               ...               ...              ...   
548226                     Low-Income       Natural Gas             True   
548228         Middle-to-Upper-Income       Natural Gas             True   
548417                     Low-Income       Natural Gas             True   
549740         Middle-to-Upper-Income       Electricity             True   
549938                Moderate-Income       Natural Gas            False   

         preIRA_mp8_heating_public_npv_upper_AP2_acs  \
bldg_id                                                
239                                          6144.65   
273                                          9634.06   
307                                              NaN   
409                                           610.03   
517                                              NaN   
...                                              ...   
548226                                      37324.77   
548228                                      17462.61   
548417                                       8642.10   
549740                                      23392.72   
549938                                           NaN   

         preIRA_mp8_heating_total_capitalCost  \
bldg_id                                         
239                                  30731.84   
273                                  31000.40   
307                                       NaN   
409                                  33993.80   
517                                       NaN   
...                                       ...   
548226                               49664.00   
548228                               30727.08   
548417                               20467.39   
549740                               27323.76   
549938                                    NaN   

         preIRA_mp8_heating_private_npv_lessWTP  \
bldg_id                                           
239                                   -26542.20   
273                                   -24449.13   
307                                         NaN   
409                                   -33627.49   
517                                         NaN   
...                                         ...   
548226                                -35739.75   
548228                                -25214.55   
548417                                -17564.22   
549740                                 -6150.08   
549938                                      NaN   

         preIRA_mp8_heating_total_npv_lessWTP_upper_AP2_acs  \
bldg_id                                                       
239                                              -20397.55    
273                                              -14815.07    
307                                                    NaN    
409                                              -33017.46    
517                                                    NaN    
...                                                    ...    
548226                                             1585.02    
548228                                            -7751.94    
548417                                            -8922.12    
549740                                            17242.64    
549938                                                 NaN    

         preIRA_mp8_heating_net_capitalCost  \
bldg_id                                       
239                                22573.23   
273                                21520.06   
307                                     NaN   
409                                30264.86   
517                                     NaN   
...                                     ...   
548226                             45475.91   
548228                             26935.26   
548417                             16922.81   
549740                             19284.83   
549938                                  NaN   

         preIRA_mp8_heating_private_npv_moreWTP  \
bldg_id                                           
239                                   -18383.59   
273                                   -14968.79   
307                                         NaN   
409                                   -29898.55   
517                                         NaN   
...                                         ...   
548226                                -31551.66   
548228                                -21422.73   
548417                                -14019.64   
549740                                  1888.85   
549938                                      NaN   

         preIRA_mp8_heating_total_npv_moreWTP_upper_AP2_acs  \
bldg_id                                                       
239                                              -12238.94    
273                                               -5334.73    
307                                                    NaN    
409                                              -29288.52    
517                                                    NaN    
...                                                    ...    
548226                                             5773.11    
548228                                            -3960.12    
548417                                            -5377.54    
549740                                            25281.57    
549938                                                 NaN    

        preIRA_mp8_heating_adoption_upper_AP2_acs  mp8_heating_rebate_amount  \
bldg_id                                                                        
239                                Tier 4: Averse                        0.0   
273                                Tier 4: Averse                        0.0   
307               N/A: Invalid Baseline Fuel/Tech                        NaN   
409                                Tier 4: Averse                     8000.0   
517               N/A: Invalid Baseline Fuel/Tech                        NaN   
...                                           ...                        ...   
548226      Tier 3: Subsidy-Dependent Feasibility                     8000.0   
548228                             Tier 4: Averse                        0.0   
548417                             Tier 4: Averse                     8000.0   
549740           Tier 2: Feasible vs. Alternative                        0.0   
549938            N/A: Invalid Baseline Fuel/Tech                        NaN   

         iraRef_mp8_heating_public_npv_upper_AP2_acs  \
bldg_id                                                
239                                          6676.90   
273                                         10301.08   
307                                              NaN   
409                                           708.97   
517                                              NaN   
...                                              ...   
548226                                      39899.89   
548228                                      19117.07   
548417                                       9383.67   
549740                                      26411.44   
549938                                           NaN   

         iraRef_mp8_heating_total_capitalCost  \
bldg_id                                         
239                                  30731.84   
273                                  31000.40   
307                                       NaN   
409                                  25993.80   
517                                       NaN   
...                                       ...   
548226                               41664.00   
548228                               30727.08   
548417                               12467.39   
549740                               27323.76   
549938                                    NaN   

         iraRef_mp8_heating_private_npv_lessWTP  \
bldg_id                                           
239                                   -26482.53   
273                                   -24374.35   
307                                         NaN   
409                                   -25616.42   
517                                         NaN   
...                                         ...   
548226                                -27455.14   
548228                                -25031.71   
548417                                 -9482.25   
549740                                 -5984.13   
549938                                      NaN   

         iraRef_mp8_heating_total_npv_lessWTP_upper_AP2_acs  \
bldg_id                                                       
239                                              -19805.63    
273                                              -14073.27    
307                                                    NaN    
409                                              -24907.45    
517                                                    NaN    
...                                                    ...    
548226                                            12444.75    
548228                                            -5914.64    
548417                                              -98.58    
549740                                            20427.31    
549938                                                 NaN    

         iraRef_mp8_heating_net_capitalCost  \
bldg_id                                       
239                                22573.23   
273                                21520.06   
307                                     NaN   
409                                22264.86   
517                                     NaN   
...                                     ...   
548226                             37475.91   
548228                             26935.26   
548417                              8922.81   
549740                             19284.83   
549938                                  NaN   

         iraRef_mp8_heating_private_npv_moreWTP  \
bldg_id                                           
239                                   -18323.92   
273                                   -14894.01   
307                                         NaN   
409                                   -21887.48   
517                                         NaN   
...                                         ...   
548226                                -23267.05   
548228                                -21239.89   
548417                                 -5937.67   
549740                                  2054.80   
549938                                      NaN   

         iraRef_mp8_heating_total_npv_moreWTP_upper_AP2_acs  \
bldg_id                                                       
239                                              -11647.02    
273                                               -4592.93    
307                                                    NaN    
409                                              -21178.51    
517                                                    NaN    
...                                                    ...    
548226                                            16632.84    
548228                                            -2122.82    
548417                                             3446.00    
549740                                            28466.24    
549938                                                 NaN    

        iraRef_mp8_heating_adoption_upper_AP2_acs  \
bldg_id                                             
239                                Tier 4: Averse   
273                                Tier 4: Averse   
307               N/A: Invalid Baseline Fuel/Tech   
409                                Tier 4: Averse   
517               N/A: Invalid Baseline Fuel/Tech   
...                                           ...   
548226      Tier 3: Subsidy-Dependent Feasibility   
548228                             Tier 4: Averse   
548417      Tier 3: Subsidy-Dependent Feasibility   
549740           Tier 2: Feasible vs. Alternative   
549938            N/A: Invalid Baseline Fuel/Tech   

         preIRA_mp8_heating_avoided_mt_co2e_lrmer  \
bldg_id                                             
239                                         26.37   
273                                         40.93   
307                                           NaN   
409                                          2.66   
517                                           NaN   
...                                           ...   
548226                                     163.75   
548228                                      76.00   
548417                                      37.74   
549740                                      95.59   
549938                                        NaN   

         iraRef_mp8_heating_avoided_mt_co2e_lrmer  \
bldg_id                                             
239                                         28.66   
273                                         43.79   
307                                           NaN   
409                                          3.09   
517                                           NaN   
...                                           ...   
548226                                     175.08   
548228                                      83.28   
548417                                      41.00   
549740                                     108.85   
549938                                        NaN   

         iraRef_mp8_heating_benefit_upper_AP2_acs  
bldg_id                                            
239                                       6676.90  
273                                      10301.08  
307                                           NaN  
409                                          0.00  
517                                           NaN  
...                                           ...  
548226                                   31899.89  
548228                                   19117.07  
548417                                    1383.67  
549740                                   26411.44  
549938                                        NaN  

[331531 rows x 28 columns]