import re
from typing import Dict, List, Optional, Union, Set
import pandas as pd

def generate_column_patterns(
    categories: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    mp_number: int = 8
) -> Dict[str, List[str]]:
    """Generate column name patterns for energy analysis data.
    
    Creates a dictionary of pattern groups that can be used for filtering DataFrame columns
    based on equipment categories, scenarios, metrics, and measure package number.
    
    Args:
        categories: List of equipment categories. Defaults to ['heating', 'waterHeating', 
                   'clothesDrying', 'cooking'].
        scenarios: List of scenario types. Defaults to ['baseline', 'preIRA', 'iraRef'].
        metrics: List of metric types to include. Defaults to ['consumption', 'mt_co2e', 
                'damages_climate', 'damages_health'].
        mp_number: Measure package number. Defaults to 8.
        
    Returns:
        Dict[str, List[str]]: Dictionary of pattern groups that can be used for filtering,
                             where keys are pattern categories and values are lists of 
                             pattern strings.
                             
    Raises:
        ValueError: If mp_number is not a positive integer.
    """
    # Validate inputs
    if mp_number <= 0:
        raise ValueError("mp_number must be a positive integer")
    
    # Default values
    if categories is None:
        categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
    
    if scenarios is None:
        scenarios = ['baseline', 'preIRA', 'iraRef']
    
    if metrics is None:
        metrics = [
            'consumption',              # Energy consumption
            'mt_co2e',                  # Emissions
            'damages_climate',          # Climate damages
            'damages_health'            # Health damages
        ]
    
    # Initialize pattern groups dictionary with empty lists
    patterns = {
        'metadata': [
            'square_footage', 'census_', 'region', 'county', 'city', 'state',
            'income', 'federal_poverty', 'occupancy', 'vintage', 'building_type'
        ],
        'base_equipment': [],
        'consumption': [],
        'emissions': [],
        'climate_damages': [],
        'health_damages': [],
        'costs': [],
        'npv': [],
        'adoption': []
    }
    
    # Generate base equipment patterns
    for category in categories:
        # Add base fuel type pattern
        patterns['base_equipment'].append(f"base_{category}_fuel")
        patterns['base_equipment'].append(f"{category}_type")
        
        # Add patterns for each fuel type
        for fuel in ['electricity', 'naturalGas', 'propane', 'fuelOil']:
            patterns['base_equipment'].append(f"base_{fuel}_{category}_consumption")
    
    # Generate scenario-specific patterns
    for category in categories:
        for scenario in scenarios:
            # Set prefix based on scenario
            if scenario == 'baseline':
                prefix = 'baseline_'
            else:
                prefix = f"{scenario}_mp{mp_number}_"
            
            # Consumption patterns
            if 'consumption' in metrics:
                # Basic consumption pattern
                patterns['consumption'].append(f"{prefix}{category}_consumption")
                
                # Year-specific consumption patterns
                patterns['consumption'].append(f"{prefix}[0-9]{{4}}_{category}_consumption")
                patterns['consumption'].append(f"{prefix}[0-9]{{4}}_{category}_reduction_consumption")

            # Emissions patterns
            if 'mt_co2e' in metrics:
                for mer_type in ['lrmer', 'srmer']:  # Long-run and short-run marginal emission rates
                    # Lifetime emissions
                    patterns['emissions'].append(f"{prefix}{category}_lifetime_mt_co2e_{mer_type}")
                    # Avoided emissions
                    patterns['emissions'].append(f"{prefix}{category}_avoided_mt_co2e_{mer_type}")
            
            # Climate damage patterns
            if 'damages_climate' in metrics:
                # Generate patterns for different MERs and SCC scenarios
                for mer in ['lrmer', 'srmer']:  # Marginal emission rate types
                    for scc in ['lower', 'central', 'upper']:  # Social cost of carbon scenarios
                        # Lifetime climate damages
                        patterns['climate_damages'].append(
                            f"{prefix}{category}_lifetime_damages_climate_{mer}_{scc}")
                        # Avoided climate damages
                        patterns['climate_damages'].append(
                            f"{prefix}{category}_avoided_damages_climate_{mer}_{scc}")
            
            # Health damage patterns
            if 'damages_health' in metrics:
                # Generate patterns for different health models and concentration-response functions
                for model in ['AP2', 'EASIUR', 'InMAP']:  # Air pollution models
                    for cr in ['acs', 'h6c']:  # Concentration-response functions
                        # Lifetime health damages
                        patterns['health_damages'].append(
                            f"{prefix}{category}_lifetime_damages_health_{model}_{cr}")
                        # Avoided health damages
                        patterns['health_damages'].append(
                            f"{prefix}{category}_avoided_damages_health_{model}_{cr}")
            
            # Cost patterns - various capital, installation, and operational costs
            patterns['costs'].append(f"{prefix}{category}_total_capitalCost")
            patterns['costs'].append(f"{prefix}{category}_net_capitalCost")
            patterns['costs'].append(f"{prefix}{category}_lifetime_fuelCost")
            patterns['costs'].append(f"mp{mp_number}_{category}_installationCost")
            patterns['costs'].append(f"mp{mp_number}_{category}_replacementCost")
            patterns['costs'].append(f"mp{mp_number}_{category}_installation_premium")
            patterns['costs'].append(f"mp{mp_number}_{category}_rebate_amount")
            
            # NPV (Net Present Value) patterns for different calculation approaches
            patterns['npv'].append(f"{prefix}{category}_climate_npv_")
            patterns['npv'].append(f"{prefix}{category}_health_npv_")
            patterns['npv'].append(f"{prefix}{category}_public_npv_")
            patterns['npv'].append(f"{prefix}{category}_private_npv_")
            patterns['npv'].append(f"{prefix}{category}_total_npv_")
            
            # Adoption patterns for benefit, adoption rate, and impact analysis
            patterns['adoption'].append(f"{prefix}{category}_benefit_upper_")
            patterns['adoption'].append(f"{prefix}{category}_adoption_upper_")
            patterns['adoption'].append(f"{prefix}{category}_impact_upper_")
            patterns['adoption'].append(f"{prefix}{category}_health_sensitivity")
    
    return patterns


def create_sample_df(
    df: pd.DataFrame,
    include_groups: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    mp_number: int = 8,
    regex_patterns: Optional[Union[str, List[str]]] = None  # New parameter
) -> pd.DataFrame:
    """Create a filtered DataFrame using column patterns generated for each group.
    
    This function filters the input DataFrame to include only columns that match
    the patterns specified by the include_groups, categories, scenarios, and metrics
    parameters.
    
    Args:
        df: The DataFrame to filter.
        include_groups: Groups of columns to include. Options are 'metadata', 
                       'base_equipment', 'consumption', 'emissions', 'climate_damages',
                       'health_damages', 'costs', 'npv', and 'adoption'.
                       Defaults to ['metadata', 'consumption', 'emissions', 
                       'climate_damages', 'health_damages'].
        categories: Equipment categories to filter for. See generate_column_patterns()
                   for defaults.
        scenarios: Scenario types to filter for. See generate_column_patterns()
                  for defaults.
        metrics: Metric types to filter for. See generate_column_patterns() 
                for defaults.
        mp_number: Measure package number. Defaults to 8.
        regex_patterns: A string or list of strings to use as regex patterns for direct 
                       column matching. If provided, columns matching any of these 
                       patterns will be included in addition to those matched by 
                       other parameters.
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only columns matching the specified patterns.
        
    Raises:
        ValueError: If df is None or empty, or if mp_number is not a positive integer.
        KeyError: If an invalid group name is provided in include_groups.
    """
    # ===================================================================================
    # INPUT VALIDATION:
    # ===================================================================================
    # Check if DataFrame is None or empty
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be None or empty")
    # Check if mp_number is a positive integer
    if mp_number <= 0:
        raise ValueError("mp_number must be a positive integer")
    
    # Sets default include_groups if none are specified
    if include_groups is None:
        include_groups = ['metadata', 'consumption', 'emissions', 'climate_damages', 'health_damages']
    
    # ===================================================================================
    # GROUP VALIDATION:
    # ===================================================================================
    # Verifies that all requested groups are valid options
    valid_groups = {'metadata', 'base_equipment', 'consumption', 'emissions', 
                    'climate_damages', 'health_damages', 'costs', 'npv', 'adoption'}
    invalid_groups = set(include_groups) - valid_groups

    # Raises KeyError if any invalid groups are requested
    if invalid_groups:
        raise KeyError(f"Invalid group names: {', '.join(invalid_groups)}. "
                       f"Valid options are: {', '.join(valid_groups)}")
    
    # ===================================================================================
    # GENERATE COLUMN PATTERNS:
    # ===================================================================================
    # Calls generate_column_patterns() to create the pattern dictionary
    patterns = generate_column_patterns(
        categories=categories,
        scenarios=scenarios,
        metrics=metrics,
        mp_number=mp_number
    )
    
    # ===================================================================================
    # PATTERN COLLECTION:
    # ===================================================================================
    # Creates a list of all pattern strings from the requested groups
    selected_patterns = []
    for group in include_groups:
        if group in patterns:
            selected_patterns.extend(patterns[group])
    
    # Add custom regex patterns if provided
    if regex_patterns is not None:
        # Convert single string to list for consistent handling
        if isinstance(regex_patterns, str):
            regex_patterns = [regex_patterns]
        selected_patterns.extend(regex_patterns)
    
    # Create list of columns to keep by matching patterns
    columns_to_keep: List[str] = []
    
    # ====================================================================================
    # PATTERN CATEGORIZATION: Separate regex and simple string patterns
    # ====================================================================================
    # Pre-compile regex patterns for efficiency
    regex_pattern_dict = {}
    simple_patterns = []
    
    # Separate regex patterns from simple string patterns
    for pattern in selected_patterns:
        if any(char in pattern for char in ['[', ']', '(', ')', '{', '}', '.', '*', '+']):
            # This is a regex pattern
            regex_pattern_dict[pattern] = re.compile(pattern)
        else:
            # This is a simple string pattern
            simple_patterns.append(pattern)
    
    # ===================================================================================
    # COLUMN MATCHING:
    # ===================================================================================
    # Match columns against patterns
    for col in df.columns:
        # First check simple string matching (faster)
        matched = False
        for pattern in simple_patterns:
            if pattern in col:
                columns_to_keep.append(col)
                matched = True
                break
        
        # If not matched with simple patterns, try regex
        if not matched:
            for pattern, regex in regex_pattern_dict.items():
                if regex.search(col):
                    columns_to_keep.append(col)
                    break
    
    # Returns a new DataFrame containing only the columns that matched patterns
    return df[columns_to_keep]
