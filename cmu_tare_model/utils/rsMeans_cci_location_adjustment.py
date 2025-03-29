# import os
# import pandas as pd
# import re

# # import from cmu-tare-model
# from config import PROJECT_ROOT
# print(f"Project root directory: {PROJECT_ROOT}")

# # pd.set_option("display.max_columns", None)
# # pd.reset_option('display.max_columns')
# # pd.set_option('display.max_rows', None)
# # pd.reset_option('display.max_rows')

# """
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RSMEANS CITY COST INDEX
# Adjustment Factors for Construction
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# """

# # Here is the original text block as a multiline string.
# # The original data (with + / - signs) is copy and pasted from the website into the text block below.
# data_from_RSMeans_website = """
# United States	Material	Installation	Average
# +30 City Average	+4.20	+1.20	+3.00
 
# Alabama	Material	Installation	Average
# Birmingham	+5.18	+0.42	+3.38
# Huntsville	+4.97	+0.56	+3.38
# Mobile	+5.41	+0.29	+3.55
# Montgomery	+5.44	+0.98	+3.87
# Tuscaloosa	+4.97	+0.14	+3.26
 
# Alaska	Material	Installation	Average
# Anchorage	+4.26	+0.97	+2.94
# Fairbanks	+4.31	+0.52	+2.74
# Juneau	+4.03	+0.97	+2.69

# Arizona	Material	Installation	Average
# Flagstaff	+5.25	+1.12	+3.85
# Mesa/Tempe	+4.16	+2.22	+3.44
# Phoenix	+4.10	+1.37	+3.17
# Prescott	+5.17	+2.23	+4.13
# Tucson	+4.10	+0.98	+3.01
 
# Arkansas	Material	Installation	Average
# Fort Smith	+4.52	+0.00	+3.06
# Jonesboro	+4.54	+9.09	+6.05
# Little Rock	+4.61	+0.15	+3.02
# Pine Bluff	+4.24	+0.00	+2.90
# Texarkana	+4.08	+6.46	+4.84
 
# California	Material	Installation	Average
# Anaheim	+4.45	+2.13	+3.34
# Bakersﬁeld	+5.00	+1.68	+3.37
# Fresno	+4.81	+2.32	+3.60
# Los Angeles	+4.66	+1.69	+3.21
# Oakland	+5.36	+2.71	+3.97
# Oxnard	+4.63	+1.81	+3.27
# Redding	+3.67	+2.36	+3.03
# Riverside	+4.45	+1.57	+3.07
# Sacramento	+5.19	+2.49	+3.86
# San Diego	+4.32	+2.81	+3.59
# San Francisco	+5.27	+2.72	+3.89
# San Jose	+4.41	+3.49	+3.89
# Santa Barbara	+4.66	+1.58	+3.20
# Stockton	+4.49	+2.60	+3.53
# Vallejo	+5.13	+2.47	+3.74
 
# Colorado	Material	Installation	Average
# Colorado Springs	+4.30	+0.14	+2.83
# Denver	+4.22	+2.02	+3.56
# Fort Collins	+4.15	+2.44	+3.47
# Grand Junction	+5.07	+0.55	+3.42
# Greeley	+4.27	+2.01	+3.40
# Pueblo	+5.17	+0.14	+3.52
 
# Connecticut	Material	Installation	Average
# Bridgeport	+4.41	+1.01	+2.82
# Bristol	+4.34	+0.93	+2.74
# Hartford	+4.53	+1.01	+2.82
# New Britain	+4.36	+0.93	+2.74
# New Haven	+4.49	+1.01	+2.71
# Norwalk	+4.43	+0.93	+2.73
# Stamford	+4.42	+0.88	+2.75
# Waterbury	+4.43	+1.01	+2.82
 
# D.C.	Material	Installation	Average
# Washington	+3.37	+1.94	+2.83
 
# Delaware	Material	Installation	Average
# Wilmington	+3.97	+0.27	+2.30
 
# Florida	Material	Installation	Average
# Daytona Beach	+4.88	+0.14	+3.05
# Fort Lauderdale	+4.41	-2.01	+2.14
# Jacksonville	+4.79	+0.15	+3.24
# Melbourne	+4.84	+0.00	+3.21
# Miami	+4.59	+0.30	+2.99
# Orlando	+4.60	+0.00	+3.07
# Panama City	+4.92	+1.25	+3.72
# Pensacola	+4.69	+0.30	+3.26
# St. Petersburg	+4.14	+0.61	+2.96
# Tallahassee	+5.14	-0.46	+3.23
# Tampa	+4.34	+0.15	+2.84
 
# Georgia	Material	Installation	Average
# Albany	+5.88	+0.28	+3.86
# Atlanta	+4.04	+0.93	+2.81
# Augusta	+4.02	+0.14	+2.66
# Columbus	+5.99	+0.28	+3.86
# Macon	+5.86	+0.14	+3.63
# Savannah	+5.96	+0.82	+4.07
# Valdosta	+5.98	+2.11	+4.71
 
# Hawaii	Material	Installation	Average
# Honolulu	+3.80	+0.08	+2.17
 
# Hawaii States & Poss.	Material	Installation	Average
# Guam	+4.32	+0.19	+3.34
 
# Idaho	Material	Installation	Average
# Boise	+5.32	+3.15	+4.51
# Lewiston	+4.86	+0.00	+3.07
# Pocatello	+5.36	+2.42	+4.28
# Twin Falls	+5.40	+0.91	+3.72
 
# Illinois	Material	Installation	Average
# Chicago	+3.05	+1.57	+2.32
# Decatur	+4.84	+0.09	+2.66
# East St. Louis	+4.77	+1.98	+3.44
# Joliet	+3.11	+0.14	+1.61
# Peoria	+4.60	+0.09	+2.51
# Rockford	+4.65	+0.08	+2.35
# Springﬁeld	+4.90	+2.69	+3.92
 
# Indiana	Material	Installation	Average
# Anderson	+5.21	+0.12	+3.23
# Bloomington	+5.02	+3.61	+4.43
# Evansville	+4.93	+0.12	+2.97
# Fort Wayne	+5.29	+0.00	+3.27
# Gary	+5.35	+0.09	+2.94
# Indianapolis	+4.76	+0.00	+2.83
# Muncie	+4.99	+0.00	+3.14
# South Bend	+6.06	+0.12	+3.72
# Terre Haute	+4.89	+0.12	+2.97
 
# Iowa	Material	Installation	Average
# Cedar Rapids	+5.13	+0.12	+3.14
# Council Bluffs	+5.19	+0.12	+3.31
# Davenport	+5.08	+0.10	+2.90
# Des Moines	+6.15	+0.34	+3.77
# Dubuque	+5.18	+0.12	+3.10
# Sioux City	+5.05	+6.23	+5.49
# Waterloo	+6.03	+0.13	+3.71
 
# Kansas	Material	Installation	Average
# Dodge City	+4.88	-0.27	+2.94
# Kansas City	+5.42	+2.03	+4.02
# Salina	+4.95	+3.57	+4.39
# Topeka	+5.57	+0.26	+3.61
# Wichita	+5.07	+4.66	+4.91
 
# Kentucky	Material	Installation	Average
# Bowling Green	+4.97	-0.51	+2.85
# Lexington	+5.03	+0.13	+3.09
# Louisville	+5.58	+0.13	+3.46
# Owensboro	+4.86	+0.00	+2.90
 
# Louisiana	Material	Installation	Average
# Alexandria	+4.14	+0.15	+2.74
# Baton Rouge	+5.38	+0.14	+3.53
# Lake Charles	+5.22	+0.14	+3.43
# Monroe	+4.04	+0.15	+2.77
# New Orleans	+5.33	+0.14	+3.46
# Shreveport	+4.29	+0.00	+2.85
 
# Maine	Material	Installation	Average
# Augusta	+6.62	+5.05	+5.94
# Bangor	+6.15	+2.99	+4.91
# Lewiston	+6.23	+2.97	+4.99
# Portland	+6.09	+2.97	+4.92
 
# Maryland	Material	Installation	Average
# Baltimore	+3.36	+0.12	+2.12
# Hagerstown	+3.49	+0.34	+2.26
 
# Massachusetts	Material	Installation	Average
# Boston	+4.73	+0.15	+2.46
# Brockton	+5.20	+0.50	+2.92
# Fall River	+5.21	+0.43	+2.95
# Hyannis	+5.14	+0.43	+2.89
# Lawrence	+5.17	+0.48	+2.84
# Lowell	+5.11	+0.48	+2.86
# New Bedford	+5.25	+0.51	+2.96
# Pittsﬁeld	+5.10	+0.19	+2.90
# Springﬁeld	+5.19	+0.36	+2.93
# Worcester	+5.19	+0.42	+2.83
 
# Michigan	Material	Installation	Average
# Ann Arbor	+4.73	+0.69	+3.03
# Dearborn	+4.85	+0.68	+3.02
# Detroit	+5.09	+0.10	+2.86
# Flint	+4.85	+0.11	+2.86
# Grand Rapids	+5.46	+0.12	+3.31
# Kalamazoo	+5.33	+0.00	+3.25
# Lansing	+4.60	+0.11	+2.88
# Muskegon	+5.42	+2.24	+4.08
# Saginaw	+4.87	+0.11	+2.92
 
# Minnesota	Material	Installation	Average
# Duluth	+5.87	+0.97	+3.72
# Minneapolis	+4.21	+0.09	+2.34
# Rochester	+5.25	+0.90	+3.35
# Saint Paul	+5.76	+0.52	+3.33
# St. Cloud	+5.30	+2.19	+3.79
 
# Mississippi	Material	Installation	Average
# Biloxi	+4.11	+0.15	+2.76
# Greenville	+4.56	+1.21	+3.43
# Jackson	+4.39	+0.15	+2.98
# Meridian	+4.18	+0.15	+2.79
 
# Missouri	Material	Installation	Average
# Cape Girardeau	+4.08	+1.96	+3.25
# Columbia	+4.57	+1.97	+3.51
# Joplin	+4.17	+0.00	+2.59
# Kansas City	+4.38	+0.39	+2.59
# Springﬁeld	+4.71	+0.12	+2.98
# St. Joseph	+4.34	+0.11	+2.52
# St. Louis	+4.02	+0.00	+2.25
 
# Montana	Material	Installation	Average
# Billings	+5.46	+7.20	+6.13
# Butte	+5.46	+3.02	+4.55
# Great Falls	+5.48	+6.22	+5.66
# Helena	+5.49	+3.02	+4.57
# Missoula	+5.21	+4.36	+4.86
 
# Nebraska	Material	Installation	Average
# Grand Island	+5.15	+0.13	+3.25
# Lincoln	+5.46	+0.00	+3.44
# North Platte	+4.55	+0.13	+2.94
# Omaha	+5.57	+0.12	+3.44
 
# Nevada	Material	Installation	Average
# Carson City	+5.78	+0.00	+3.53
# Las Vegas	+6.04	+3.46	+4.94
# Reno	+5.81	+0.36	+3.72
 
# New Hampshire	Material	Installation	Average
# Manchester	+4.89	+0.97	+3.27
# Nashua	+4.80	+1.08	+3.17
# Portsmouth	+4.72	+1.19	+3.23
 
# New Jersey	Material	Installation	Average
# Camden	+5.03	+0.31	+2.59
# Elizabeth	+3.60	+0.36	+1.99
# Jersey City	+3.74	+0.43	+2.00
# Newark	+3.82	+0.43	+2.05
# Paterson	+3.68	+0.43	+2.08
# Trenton	+4.94	+0.15	+2.57
 
# New Mexico	Material	Installation	Average
# Albuquerque	+5.97	+0.27	+3.88
# Farmington	+5.73	+0.27	+3.75
# Las Cruces	+5.33	+0.28	+3.51
# Roswell	+5.23	+0.27	+3.48
# Santa Fe	+5.84	+0.27	+3.87
 
# New York	Material	Installation	Average
# Albany	+5.37	+0.18	+2.94
# Binghamton	+4.21	+0.10	+2.41
# Buffalo	+2.62	+1.55	+2.17
# Hicksville	+4.21	+0.44	+2.23
# New York	+3.94	+0.96	+2.27
# Riverhead	+4.28	+0.45	+2.25
# Rochester	+5.18	+0.10	+2.98
# Schenectady	+5.13	+0.18	+2.94
# Syracuse	+4.19	+0.10	+2.42
# Utica	+4.38	+0.20	+2.47
# Watertown	+4.20	+0.10	+2.54
# White Plains	+3.86	+0.20	+1.85
# Yonkers	+4.01	+0.20	+1.83
 
# North Carolina	Material	Installation	Average
# Asheville	+4.78	+1.04	+3.34
# Charlotte	+4.74	+1.78	+3.68
# Durham	+6.14	+0.89	+4.43
# Fayetteville	+5.00	+0.30	+3.37
# Greensboro	+6.04	+0.89	+4.36
# Raleigh	+6.02	+1.04	+4.30
# Wilmington	+4.69	+0.30	+3.25
# Winston-Salem	+6.06	+0.88	+4.25
 
# North Dakota	Material	Installation	Average
# Bismarck	+5.30	+4.25	+4.87
# Fargo	+4.58	+2.12	+3.65
# Grand Forks	+4.98	+0.76	+3.33
# Minot	+4.98	+1.26	+3.54
 
# Ohio	Material	Installation	Average
# Akron	+4.72	+0.00	+2.85
# Canton	+4.75	+0.74	+3.28
# Cincinnati	+3.92	+1.24	+2.89
# Cleveland	+4.76	+1.19	+3.34
# Columbus	+4.52	+0.00	+2.85
# Dayton	+3.96	+0.00	+2.46
# Lorain	+4.78	+0.12	+3.00
# Springﬁeld	+3.86	+0.00	+2.45
# Toledo	+4.65	+0.86	+3.16
# Youngstown	+4.86	+0.00	+3.03
 
# Oklahoma	Material	Installation	Average
# Enid	+5.03	+0.15	+3.37
# Lawton	+5.11	+0.30	+3.48
# Muskogee	+4.73	+0.16	+3.17
# Oklahoma City	+4.83	+0.29	+3.23
# Tulsa	+4.69	+0.15	+3.11
 
# Oregon	Material	Installation	Average
# Eugene	+4.81	+0.10	+2.72
# Medford	+4.84	+0.00	+2.81
# Portland	+4.69	+0.10	+2.69
# Salem	+5.01	+0.10	+2.97
 
# Pennsylvania	Material	Installation	Average
# Allentown	+4.24	+0.09	+2.36
# Altoona	+4.29	+0.11	+2.45
# Erie	+4.30	+0.10	+2.43
# Harrisburg	+3.99	+0.11	+2.38
# Philadelphia	+3.80	+0.52	+2.17
# Pittsburgh	+6.11	+1.27	+4.06
# Reading	+3.83	+0.10	+2.21
# Scranton	+4.31	+0.10	+2.45
# York	+3.90	+0.10	+2.21
 
# Rhode Island	Material	Installation	Average
# Providence	+5.82	+0.18	+3.18
 
# South Carolina	Material	Installation	Average
# Charleston	+5.41	+4.29	+5.06
# Columbia	+5.39	+3.98	+4.99
# Florence	+5.61	+3.98	+5.00
# Greenville	+5.61	+3.24	+4.76
# Spartanburg	+5.60	+3.39	+4.87
 
# South Dakota	Material	Installation	Average
# Aberdeen	+5.09	+0.14	+3.34
# Pierre	+5.17	+0.86	+3.60
# Rapid City	+5.20	+0.82	+3.66
# Sioux Falls	+5.50	+1.14	+3.82
 
# Tennessee	Material	Installation	Average
# Chattanooga	+3.94	+0.29	+2.67
# Jackson	+4.62	+0.17	+3.20
# Johnson City	+3.83	+0.33	+2.67
# Knoxville	+3.97	+0.15	+2.66
# Memphis	+3.80	+0.00	+2.44
# Nashville	+4.17	+2.53	+3.58
 
# Texas	Material	Installation	Average
# Abilene	+5.42	+0.16	+3.67
# Amarillo	+5.85	+0.16	+3.96
# Austin	+4.93	+0.00	+3.27
# Beaumont	+4.35	+0.00	+2.93
# Corpus Christi	+4.31	+0.16	+2.86
# Dallas	+4.24	+0.15	+2.80
# El Paso	+4.86	+0.00	+3.18
# Fort Worth	+5.24	+0.00	+3.49
# Houston	+4.47	+0.00	+3.00
# Laredo	+4.33	+0.16	+2.91
# Lubbock	+5.23	+0.16	+3.60
# Odessa	+5.31	+0.16	+3.54
# San Antonio	+3.93	+0.00	+2.74
# Waco	+5.41	+0.16	+3.57
# Wichita Falls	+5.37	+0.80	+3.82
 
# Utah	Material	Installation	Average
# Logan	+5.33	+1.37	+3.97
# Ogden	+5.33	+1.37	+4.02
# Provo	+5.31	+1.51	+3.96
# Salt Lake City	+5.29	+1.92	+4.13
 
# Vermont	Material	Installation	Average
# Burlington	+5.25	+0.62	+3.51
# Rutland	+5.30	+0.75	+3.58
 
# Virginia	Material	Installation	Average
# Alexandria	+4.83	+3.21	+4.20
# Arlington	+4.80	+2.37	+3.86
# Newport News	+5.31	+0.29	+3.61
# Norfolk	+5.36	+0.29	+3.58
# Portsmouth	+5.34	+9.83	+6.77
# Richmond	+5.19	-2.88	+2.25
# Roanoke	+4.81	+5.03	+4.79
 
# Washington	Material	Installation	Average
# Everett	+4.64	+0.10	+2.62
# Richland	+4.02	+0.11	+2.49
# Seattle	+4.34	+4.47	+4.41
# Spokane	+4.00	+0.00	+2.47
# Tacoma	+4.62	+0.00	+2.62
# Vancouver	+4.58	+0.00	+2.67
# Yakima	+4.62	+0.10	+2.78
 
# West Virginia	Material	Installation	Average
# Charleston	+5.93	-1.30	+2.85
# Huntington	+5.95	+0.21	+3.54
# Parkersburg	+4.89	+0.11	+2.95
# Wheeling	+4.73	+0.33	+2.92
 
# Wisconsin	Material	Installation	Average
# Eau Claire	+5.27	+0.10	+3.00
# Green Bay	+5.17	+0.10	+2.97
# Kenosha	+5.03	+0.10	+2.85
# La Crosse	+5.25	+0.10	+3.04
# Madison	+5.75	+0.10	+3.31
# Milwaukee	+4.02	+4.15	+4.11
# Racine	+4.96	+0.00	+2.76
 
# Wyoming	Material	Installation	Average
# Casper	+6.04	+0.00	+4.05
# Cheyenne	+6.02	+0.14	+3.87
# Rock Springs	+5.79	+0.84	+4.16
# """  

# # Regex to detect lines that end with "Material Installation Average"
# # We'll treat those lines as "State" headers:
# header_pattern = re.compile(r"(.*)\s+Material\s+Installation\s+Average\s*$")

# # Regex to detect "data lines" with:
# #    City  (+/- number)  (+/- number)  (+/- number)
# data_pattern = re.compile(r"^(.*?)\s+([+\-]\d+\.\d+)\s+([+\-]\d+\.\d+)\s+([+\-]\d+\.\d+)$")

# current_state = None
# rows = []

# for line in data_from_RSMeans_website.split("\n"):
#     line = line.strip()
#     if not line:
#         continue  # skip blank lines

#     # Check if this line is a "State" line
#     header_match = header_pattern.match(line)
#     if header_match:
#         # The part before "Material Installation Average" is the State (or region)
#         current_state = header_match.group(1).strip()
#         continue

#     # Otherwise, check if this line matches the data pattern
#     data_match = data_pattern.match(line)
#     if data_match and current_state is not None:
#         city = data_match.group(1).strip()
#         material_str = data_match.group(2)  # e.g. "+4.20"
#         install_str = data_match.group(3)
#         average_str = data_match.group(4)

#         material_int = float(material_str)
#         install_int = float(install_str)
#         average_int = float(average_str)
#         joseph_erl_cci = average_int / 3.00

#         # Convert the percentage difference relative to the national average into location adjustment factors
#         # Location Adjustment Factors: Convert e.g. "+4.20" => 4.20 => factor = 1 + 4.20/100 = 1.0420
#         def convert_to_factor(signed_str):
#             # float("+4.20") => 4.2, float("-2.01") => -2.01
#             offset = float(signed_str)
#             return (offset / 100.0) + 1.0

#         factor_material = convert_to_factor(material_str)
#         factor_install  = convert_to_factor(install_str)
#         factor_average  = convert_to_factor(average_str)

#         rows.append({
#             "State": current_state,
#             "City": city,
#             "cci_material": material_int,
#             "cci_installation": install_int,
#             "cci_average": average_int,
#             "joseph_erl_cci": joseph_erl_cci,
#             # "cci_loc_adjust_factor_material": factor_material,
#             # "cci_loc_adjust_factor_installation": factor_install,
#             # "cci_loc_adjust_factor_avg": factor_average,
#         })

# # Build DataFrame
# df_rsMeans_cityCostIndex = pd.DataFrame(rows)

# # Use CCI to adjust for cost differences when compared to the national average
# # Accounts for the costs of materials, labor and equipment and compares it to a national average of 30 major U.S. cities
# # loc_adjust_factor_map = df_rsMeans_cityCostIndex.set_index('City')['cci_loc_adjust_factor_avg'].to_dict()
# loc_adjust_factor_map = df_rsMeans_cityCostIndex.set_index('City')['cci_average'].to_dict()
# loc_adjust_factor_30cities = (3.00 / 100) + 1.0

# # Use CCI to adjust for cost differences when compared to the national average
# # Function to map city to its average cost
# def map_loc_adjust_factor(city):
#     if city in loc_adjust_factor_map:
#         return loc_adjust_factor_map[city]
#     elif city == 'Not in a census Place' or city == 'In another census Place':
#         return loc_adjust_factor_map.get('+30 City Average')
#     else:
#         return loc_adjust_factor_map.get('+30 City Average')

# print(f"""
# ==============================================================================================================
# RSMEANS CITY COST INDEX
# Adjustment Factors for Construction
# ==============================================================================================================
# 30 City Average CCI: 3.00
# 30 City Average Location Adjustment Factor: {loc_adjust_factor_30cities}

# Dataframe with CCI and Location Adjustment Factors:

# {df_rsMeans_cityCostIndex}

# """)