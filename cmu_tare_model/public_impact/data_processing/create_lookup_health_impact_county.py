# Description: This script processes the RCM data for use in creating a health-related emissions marginal social cost lookup dictionary.
# Marginal Social Costs for Health-Related Emissions
# ======================================================================================================================
import os
import pandas as pd

# import from cmu_tare-model package
from config import PROJECT_ROOT

# =======================================================================================================================
# Set print_verbose to True for detailed output, or False for minimal output
# By default, verbose is set to False because define_scenario_params is imported multiple times in the codebase
# and we don't want to print the same information multiple times.
print_verbose = False
# =======================================================================================================================


# UPDATE TO USE THE 2023USD INPUT CUSTOM VSL VALUES INSTEAD OF THE 2006USD VALUES INFLATED TO 2023USD!!!
def process_rcm_data(filename: str, PROJECT_ROOT: str) -> pd.DataFrame:
    """
    Processes the RCM data for county-level health-related emissions marginal social costs.

    This function reads a CSV file containing RCM data, retains only annual data, 
    converts county_fips codes to a 5-digit string format. 
    
    Args:
        filename (str): 
            Name of the CSV file (e.g. "rcm_msc_county_vsl1271_usd2023_ground_acs.csv").
        PROJECT_ROOT (str): 
            Path to the root directory of the project.

    Returns:
        pd.DataFrame: 
            A DataFrame containing only annual data, with columns:
            - 'county_fips' (5-digit string)
            - 'state' (from the 'state_abbr' column)
            - 'damage_usd2023' (from the 'damage' column)
            plus all other original columns.

    Raises:
        FileNotFoundError: 
            If the CSV file is not found at the specified path.
        KeyError: 
            If expected columns (like 'season' or 'damage') are missing.
    """
    # Construct the absolute path to the CSV file
    relative_path = os.path.join("cmu_tare_model", "data", "marginal_social_costs", filename)
    file_path = os.path.join(PROJECT_ROOT, relative_path)
    df_rcm_msc_data = pd.read_csv(file_path)

    # Filter to retain only 'annual' season rows
    df_rcm_msc_data = df_rcm_msc_data[df_rcm_msc_data['season'] == 'annual']

    # Original RCM data uses 'fips' as the county identifier, EUSS dataframe uses 'county_fips'
    # Convert 'county_fips' to zero-padded string
    df_rcm_msc_data['county_fips'] = df_rcm_msc_data['fips'].astype(str).str.zfill(5)

    # create a new column 'state' from the 'state_abbr' column
    df_rcm_msc_data['state'] = df_rcm_msc_data['state_abbr']

    # Create new column 'damage_usd2023' from the 'damage' column to clarify dollar year of VSL
    df_rcm_msc_data['damage_usd2023'] = df_rcm_msc_data['damage']

    return df_rcm_msc_data


def create_lookup_nested(df_rcm_msc_data: pd.DataFrame) -> dict:
    """
    Creates a nested lookup dictionary keyed by (county_fips, state), 
    then model, then pollutant. Also calculates state-level averages as fallbacks.
    Everything stored in lowercase for pollutant keys.

    Args:
        df_rcm_msc_data (pd.DataFrame):
            DataFrame containing columns:
            - 'county_fips' (zero-padded string)
            - 'state'
            - 'model'
            - 'pollutant'
            - 'damage_usd2023'

    Returns:
        dict: 
            A nested dictionary of the form:
            {
                (county_fips, state): {
                    model: {
                        pollutant: damage_usd2023,
                        ...
                    },
                    ...
                },
                ...
                # Special state-level fallback entries
                ('STATE_AVG', state): {
                    model: {
                        pollutant: state_avg_damage,
                        ...
                    },
                    ...
                }
            }
    """
    # First, create the standard county-level lookup dictionary
    lookup_health_rcm_msc = {}
    for _, row in df_rcm_msc_data.iterrows():
        county_key = (row['county_fips'], row['state'])
        model = row['model']

        # Make sure pollutant is stored as lowercase
        pollutant = row['pollutant'].lower()
        
        damage = row['damage_usd2023']

        # Ensure nested dictionaries exist
        if county_key not in lookup_health_rcm_msc:
            lookup_health_rcm_msc[county_key] = {}
        if model not in lookup_health_rcm_msc[county_key]:
            lookup_health_rcm_msc[county_key][model] = {}

        # Store with lowercase pollutant
        lookup_health_rcm_msc[county_key][model][pollutant] = damage

    # Now calculate state averages and add them as special entries
    print("\nCalculating state-level averages for fallback...")
    
    # Group by state, model, and pollutant to calculate state averages
    state_avgs = df_rcm_msc_data.groupby(['state', 'model', 'pollutant'])['damage_usd2023'].mean()
    
    # Add state averages to the lookup dictionary with special 'STATE_AVG' marker
    for (state, model, pollutant), avg_damage in state_avgs.items():
        state_key = ('STATE_AVG', state)
        
        # Ensure nested dictionaries exist
        if state_key not in lookup_health_rcm_msc:
            lookup_health_rcm_msc[state_key] = {}
        if model not in lookup_health_rcm_msc[state_key]:
            lookup_health_rcm_msc[state_key][model] = {}
            
        # Store state average with lowercase pollutant
        lookup_health_rcm_msc[state_key][model][pollutant.lower()] = avg_damage
    
    # Add a section to print some statistics about the state averages
    states = df_rcm_msc_data['state'].unique()
    models = df_rcm_msc_data['model'].unique()
    pollutants = df_rcm_msc_data['pollutant'].unique()
    
    print(f"Added state-level fallbacks for {len(states)} states, {len(models)} models, and {len(pollutants)} pollutants:")
    print(f"- States: {', '.join(sorted(states))}")
    print(f"- Models: {', '.join(sorted(models))}")
    print(f"- Pollutants: {', '.join(sorted(pollutants))}")
    
    return lookup_health_rcm_msc


def get_health_impact_with_fallback(
    lookup_dict: dict, 
    county_key: tuple, 
    model: str, 
    pollutant: str,
    debug: bool = False
) -> float:
    """
    Get health impact value with automatic fallback to state-level average if county-specific data is missing.
    
    Args:
        lookup_dict: The nested lookup dictionary from create_lookup_nested.
        county_key: Tuple of (county_fips, state).
        model: Model name (e.g., 'AP2', 'EASIUR', 'InMAP').
        pollutant: Pollutant name (will be converted to lowercase).
        debug: If True, print diagnostic information.
        
    Returns:
        float: The health impact value, or None if not found even with fallback.
    """
    pollutant = pollutant.lower()
    
    # First try exact county lookup
    county_value = lookup_dict.get(county_key, {}).get(model, {}).get(pollutant)
    if county_value is not None:
        return county_value
    
    # If county lookup failed and we have a state, try state average
    if county_key and len(county_key) >= 2:
        state = county_key[1]
        state_key = ('STATE_AVG', state)
        state_value = lookup_dict.get(state_key, {}).get(model, {}).get(pollutant)
        
        if state_value is not None:
            if debug:
                print(f"Using state average for {state} instead of county {county_key[0]} for {model}, {pollutant}")
            return state_value
    
    if debug:
        print(f"No data found for {county_key}, {model}, {pollutant} (even with state fallback)")
    
    # If all lookups failed, return None
    return None


def analyze_health_impact_coverage(
    df: pd.DataFrame,
    lookup_health_fossil_fuel_acs: dict,
    lookup_health_fossil_fuel_h6c: dict,
    lookup_health_electricity_acs: dict,
    lookup_health_electricity_h6c: dict,
    rcm_models: list,
    cr_functions: list,
    sample_pollutants: list = ['so2', 'nox', 'pm25'],
    verbose: bool = False
) -> dict:
    """
    Analyze the coverage of health impact data in the lookup tables.
    
    This function checks county and state coverage across different model and
    concentration-response function combinations.
    
    Args:
        df: DataFrame containing county_key or components to build it.
        lookup_health_fossil_fuel_acs: Health impact lookup for fossil fuels with ACS C-R function.
        lookup_health_fossil_fuel_h6c: Health impact lookup for fossil fuels with H6C C-R function.
        lookup_health_electricity_acs: Health impact lookup for electricity with ACS C-R function.
        lookup_health_electricity_h6c: Health impact lookup for electricity with H6C C-R function.
        rcm_models: List of RCM models to check.
        cr_functions: List of concentration-response functions to check.
        sample_pollutants: List of pollutants to check for coverage.
        verbose: Whether to print detailed information.
        
    Returns:
        dict: Summary statistics of data coverage.
    """
    # Ensure county_key exists in the DataFrame
    if 'county_key' not in df.columns:
        if 'county_fips' in df.columns and 'state' in df.columns:
            df['county_key'] = list(zip(df['county_fips'], df['state']))
        else:
            raise ValueError("DataFrame must contain 'county_key' or both 'county_fips' and 'state' columns")
    
    if verbose:
        print("\n" + "="*80)
        print("HEALTH IMPACT DATA COVERAGE ANALYSIS")
        print("="*80)
    
    # Get unique counties and states from the DataFrame
    counties_in_data = df['county_key'].unique()
    states_in_data = set()
    valid_county_keys = []
    
    # Extract valid county keys and states
    for key in counties_in_data:
        if isinstance(key, tuple) and len(key) >= 2 and key[1] is not None:
            valid_county_keys.append(key)
            states_in_data.add(key[1])
    
    total_counties = len(valid_county_keys)
    total_states = len(states_in_data)
    
    # Initialize coverage statistics
    coverage_stats = {
        'total_counties': total_counties,
        'total_states': total_states,
        'states': sorted(states_in_data),
        'fossil_fuel': {
            'county_level': {rcm: {cr: 0 for cr in cr_functions} for rcm in rcm_models},
            'state_fallback': {rcm: {cr: 0 for cr in cr_functions} for rcm in rcm_models},
            'missing': {rcm: {cr: 0 for cr in cr_functions} for rcm in rcm_models}
        },
        'electricity': {
            'county_level': {rcm: {cr: 0 for cr in cr_functions} for rcm in rcm_models},
            'state_fallback': {rcm: {cr: 0 for cr in cr_functions} for rcm in rcm_models},
            'missing': {rcm: {cr: 0 for cr in cr_functions} for rcm in rcm_models}
        }
    }
    
    # List to store counties with missing data for detailed reporting
    missing_data_examples = []
    
    # Check data availability for each county, model, and CR function
    for county_key in valid_county_keys:
        county_fips, state = county_key
        state_key = ('STATE_AVG', state)
        
        for rcm in rcm_models:
            for cr in cr_functions:
                # Select the appropriate lookup table based on CR function
                fossil_lookup = lookup_health_fossil_fuel_acs if cr == 'acs' else lookup_health_fossil_fuel_h6c
                electricity_lookup = lookup_health_electricity_acs if cr == 'acs' else lookup_health_electricity_h6c
                
                # Check fossil fuel data availability
                fossil_county_data = any(
                    fossil_lookup.get(county_key, {}).get(rcm, {}).get(poll) is not None
                    for poll in sample_pollutants
                )
                
                fossil_state_data = any(
                    fossil_lookup.get(state_key, {}).get(rcm, {}).get(poll) is not None
                    for poll in sample_pollutants
                )
                
                # Check electricity data availability
                electricity_county_data = any(
                    electricity_lookup.get(county_key, {}).get(rcm, {}).get(poll) is not None
                    for poll in sample_pollutants
                )
                
                electricity_state_data = any(
                    electricity_lookup.get(state_key, {}).get(rcm, {}).get(poll) is not None
                    for poll in sample_pollutants
                )
                
                # Update statistics
                if fossil_county_data:
                    coverage_stats['fossil_fuel']['county_level'][rcm][cr] += 1
                elif fossil_state_data:
                    coverage_stats['fossil_fuel']['state_fallback'][rcm][cr] += 1
                else:
                    coverage_stats['fossil_fuel']['missing'][rcm][cr] += 1
                    
                    # Store example of missing data if this is the first few we've encountered
                    if len(missing_data_examples) < 5 and (county_key, rcm, cr, 'fossil_fuel') not in missing_data_examples:
                        missing_data_examples.append((county_key, rcm, cr, 'fossil_fuel'))
                
                if electricity_county_data:
                    coverage_stats['electricity']['county_level'][rcm][cr] += 1
                elif electricity_state_data:
                    coverage_stats['electricity']['state_fallback'][rcm][cr] += 1
                else:
                    coverage_stats['electricity']['missing'][rcm][cr] += 1
                    
                    # Store example of missing data
                    if len(missing_data_examples) < 10 and (county_key, rcm, cr, 'electricity') not in missing_data_examples:
                        missing_data_examples.append((county_key, rcm, cr, 'electricity'))
    
    # Print coverage summary if verbose
    if verbose:
        print(f"\nAnalyzed {total_counties} counties across {total_states} states")
        
        print("\nFOSSIL FUEL HEALTH IMPACT COVERAGE:")
        print("-" * 40)
        for rcm in rcm_models:
            for cr in cr_functions:
                county_count = coverage_stats['fossil_fuel']['county_level'][rcm][cr]
                state_count = coverage_stats['fossil_fuel']['state_fallback'][rcm][cr]
                missing_count = coverage_stats['fossil_fuel']['missing'][rcm][cr]
                
                county_pct = county_count / total_counties * 100
                state_pct = state_count / total_counties * 100
                missing_pct = missing_count / total_counties * 100
                
                print(f"Model: {rcm}, CR Function: {cr}")
                print(f"  - County-level data: {county_count}/{total_counties} counties ({county_pct:.1f}%)")
                print(f"  - State-level fallback: {state_count}/{total_counties} counties ({state_pct:.1f}%)")
                print(f"  - Missing data: {missing_count}/{total_counties} counties ({missing_pct:.1f}%)")
                print(f"  - Total coverage: {county_count + state_count}/{total_counties} counties ({county_pct + state_pct:.1f}%)")
        
        print("\nELECTRICITY HEALTH IMPACT COVERAGE:")
        print("-" * 40)
        for rcm in rcm_models:
            for cr in cr_functions:
                county_count = coverage_stats['electricity']['county_level'][rcm][cr]
                state_count = coverage_stats['electricity']['state_fallback'][rcm][cr]
                missing_count = coverage_stats['electricity']['missing'][rcm][cr]
                
                county_pct = county_count / total_counties * 100
                state_pct = state_count / total_counties * 100
                missing_pct = missing_count / total_counties * 100
                
                print(f"Model: {rcm}, CR Function: {cr}")
                print(f"  - County-level data: {county_count}/{total_counties} counties ({county_pct:.1f}%)")
                print(f"  - State-level fallback: {state_count}/{total_counties} counties ({state_pct:.1f}%)")
                print(f"  - Missing data: {missing_count}/{total_counties} counties ({missing_pct:.1f}%)")
                print(f"  - Total coverage: {county_count + state_count}/{total_counties} counties ({county_pct + state_pct:.1f}%)")
        
        # Print examples of counties with missing data
        if missing_data_examples:
            print("\nEXAMPLES OF COUNTIES WITH MISSING DATA:")
            print("-" * 40)
            for county_key, rcm, cr, source_type in missing_data_examples:
                print(f"County: {county_key[0]} (State: {county_key[1]}), Model: {rcm}, CR Function: {cr}, Type: {source_type}")
        
        # Print states covered
        print("\nSTATES COVERED IN DATA:")
        print("-" * 40)
        state_list = sorted(states_in_data)
        state_chunks = [state_list[i:i+10] for i in range(0, len(state_list), 10)]
        for chunk in state_chunks:
            print(", ".join(chunk))
        
        # Check for missing states (if we know what all states should be)
        all_us_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
        }
        missing_states = all_us_states - states_in_data
        if missing_states:
            print("\nMISSING STATES:")
            print("-" * 40)
            missing_state_chunks = [sorted(list(missing_states))[i:i+10] for i in range(0, len(missing_states), 10)]
            for chunk in missing_state_chunks:
                print(", ".join(chunk))
    
    return coverage_stats


# ======================================================================================================================
# MARGINAL SOCIAL COSTS FOR HEALTH-RELATED EMISSIONS: FOSSIL FUEL COMBUSTION
# ======================================================================================================================
# Load data for Fossil Fuel MSC (Ground-level stack)
# ======================================================================================================================

df_health_rcm_ground_acs = process_rcm_data("rcm_msc_county_vsl1271_usd2023_ground_acs.csv", PROJECT_ROOT)
df_health_rcm_ground_h6c = process_rcm_data("rcm_msc_county_vsl1271_usd2023_ground_h6c.csv", PROJECT_ROOT)

# Create lookup dictionaries for fossil fuel (ground-level stack)
lookup_health_fossil_fuel_acs = create_lookup_nested(df_health_rcm_ground_acs)
lookup_health_fossil_fuel_h6c = create_lookup_nested(df_health_rcm_ground_h6c)

if print_verbose:
    print(f"""
    ======================================================================================================================
    HEALTH IMPACTS (MARGINAL SOCIAL COSTS): FOSSIL FUELS
    ======================================================================================================================
    Model: RCM
    Output Area: county_fips
    Geography: Counties
    Pollutant: NOX,SO2,PM25
    VSL: 12.71M (Inflated from 11.3M USD2021 to 12.71M USD2023)
        CPI Ratio for 2021 to 2023: 1.1244861054729305
        VSL2023 = VSL2021 * 1.1244861054729305 = 11.3M * 1.1244861054729305 = 12.71 M
    Stack Height: ground level

    C-R Function: ACS or H6C
    ======================================================================================================================
    - Clean the data so that only annual values are retained. 
    - Convert 'county_fips' to string and pad with leading zeros (to ensure a 5-digit format)

    DATAFRAME: Ground Level, ACS C-R Function

    {df_health_rcm_ground_acs}
        
    DATAFRAME: Ground Level, H6C C-R Function

    {df_health_rcm_ground_h6c}
        
    =======================================================================================================================
    CREATES LOOKUP DICTIONARY: HEALTH IMPACTS (MSC) FROM FOSSIL FUELS
    ======================================================================================================================
    """)


# ======================================================================================================================
# HEALTH IMPACTS (MARGINAL SOCIAL COSTS): ELECTRICITY GENERATION
# ====================================================================================================================
# Load data for Electricity MSC (Elevated/High-stack)
# ====================================================================================================================

df_health_rcm_elevated_acs = process_rcm_data("rcm_msc_county_vsl1271_usd2023_elevated_acs.csv", PROJECT_ROOT)
df_health_rcm_elevated_h6c = process_rcm_data("rcm_msc_county_vsl1271_usd2023_elevated_h6c.csv", PROJECT_ROOT)

# Create lookup dictionaries for electricity generation (elevated/high-stack)
lookup_health_electricity_acs = create_lookup_nested(df_health_rcm_elevated_acs)
lookup_health_electricity_h6c = create_lookup_nested(df_health_rcm_elevated_h6c)

if print_verbose:
    print(f"""
    ======================================================================================================================
    HEALTH IMPACTS (MARGINAL SOCIAL COSTS): ELECTRICITY GENERATION
    ======================================================================================================================
    Model: RCM
    Output Area: county_fips
    Geography: Counties
    Pollutant: NOX,SO2,PM25
    VSL: 12.71M (Inflated from 11.3M USD2021 to 12.71M USD2023)
        CPI Ratio for 2021 to 2023: 1.1244861054729305
        VSL2023 = VSL2021 * 1.1244861054729305 = 11.3M * 1.1244861054729305 = 12.71 M
    Stack Height: high stack

    C-R Function: ACS or H6C
    ======================================================================================================================
    - Clean the data so that only annual values are retained. 
    - Convert 'county_fips' to string and pad with leading zeros (to ensure a 5-digit format)

    DATAFRAME: Elevated (High Stack), ACS C-R Function

    {df_health_rcm_elevated_acs}
        
    DATAFRAME: Elevated (High Stack), H6C C-R Function

    {df_health_rcm_elevated_h6c}
        
    =======================================================================================================================
    CREATES LOOKUP DICTIONARY: HEALTH IMPACTS (MSC) FROM ELECTRICITY GENERATION
    ======================================================================================================================
    """)
