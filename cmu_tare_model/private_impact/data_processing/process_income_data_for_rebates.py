import os
import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2025_2024

# ============================================================================
# Set print_verbose to True for detailed output, or False for minimal output.
# This module runs on import and is imported in several places, so verbose
# output is off by default to avoid repeating the same messages each time.
print_verbose = False
# ============================================================================

"""
----------------------------------------------------------------------------
AREA MEDIAN INCOME FOR LMI DESIGNATION AND IRA REBATE ELIGIBILITY
----------------------------------------------------------------------------
Area Median Income comes from the U.S. Census Bureau American Community Survey
5-Year table B19013 (median household income), vintage 2024, downloaded from
data.census.gov. One file holds both county rows (GEO_ID prefix '0500000US')
and state rows (prefix '0400000US'). The B19013 estimate is reported in
USD2024 and is inflated to the model reference year (USD2025) so it shares a
common basis with household incomes used elsewhere.
"""

# Map state FIPS codes to two-letter USPS abbreviations. The Census file
# carries only full state names, but the model joins state-level income on the
# two-letter abbreviation, so it is derived from the FIPS code embedded in
# GEO_ID. Covers the 50 states plus DC (11) and Puerto Rico (72).
STATE_FIPS_TO_USPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}

# Step 1 -- locate and read the Census B19013 file.
# Row 0 is the machine-readable header; row 1 is a human-readable label row
# that must be skipped. Selecting the four named columns drops the trailing
# empty column created by the line-ending comma in the source file.
filename = "ACSDT5Y2024.B19013-Data.csv"
relative_path = os.path.join(
    "cmu_tare_model", "data", "ami_calculations_data",
    "ACSDT5Y2024.B19013", filename
)
file_path = os.path.join(PROJECT_ROOT, relative_path)

if print_verbose:
    print(f"Retrieved data for filename: {filename}")
    print(f"Located at filepath: {file_path}")

df_b19013 = pd.read_csv(file_path, skiprows=[1])
df_b19013 = df_b19013.loc[:, ["GEO_ID", "NAME", "B19013_001E", "B19013_001M"]]

# The estimate column carries one suppressed value ('-') that must become NaN
# so it does not block numeric inflation; coerce turns any non-numeric entry
# into NaN.
df_b19013["B19013_001E"] = pd.to_numeric(
    df_b19013["B19013_001E"], errors="coerce"
)

# Step 2 -- split the single file into county and state rows by GEO_ID prefix.
county_mask = df_b19013["GEO_ID"].str.startswith("0500000US")
state_mask = df_b19013["GEO_ID"].str.startswith("0400000US")

df_county_medianIncome = df_b19013[county_mask].copy()
df_state_medianIncome = df_b19013[state_mask].copy()

# Step 3 -- derive the join keys the model expects.
# County: build the NHGIS GISJOIN string from the 5-digit FIPS in GEO_ID.
# Example: '0500000US01001' -> 'G0100010' (G + state(2) + '0' + county(3) + '0').
df_county_medianIncome["gis_joinID_county"] = (
    "G"
    + df_county_medianIncome["GEO_ID"].str[-5:-3]
    + "0"
    + df_county_medianIncome["GEO_ID"].str[-3:]
    + "0"
)

# State: the two-letter abbreviation comes from the state FIPS in GEO_ID.
# Example: '0400000US01' -> '01' -> 'AL'.
df_state_medianIncome["state_abbrev"] = (
    df_state_medianIncome["GEO_ID"].str[-2:].map(STATE_FIPS_TO_USPS)
)

# Step 4 -- inflate the area median income from USD2024 to the reference year.
df_county_medianIncome["median_income_USD2025"] = round(
    df_county_medianIncome["B19013_001E"] * cpi_ratio_2025_2024, 2
)
df_state_medianIncome["median_income_USD2025"] = round(
    df_state_medianIncome["B19013_001E"] * cpi_ratio_2025_2024, 2
)

if print_verbose:
    print(
        f"County rows: {len(df_county_medianIncome)} | "
        f"State rows: {len(df_state_medianIncome)}"
    )
