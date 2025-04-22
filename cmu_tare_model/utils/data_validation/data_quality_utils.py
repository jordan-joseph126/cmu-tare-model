import numpy as np
import pandas as pd
from typing import List, Optional, Union

from cmu_tare_model.constants import ALLOWED_TECHNOLOGIES, EQUIPMENT_SPECS, FUEL_MAPPING
# Import the retrofit function at the top of the file
from cmu_tare_model.utils.data_validation.retrofit_status_utils import get_retrofit_homes_mask

# ======================================================================================================
# MASKING AND VALIDITY FLAGS FOR BASELINE AND RETROFIT DATA
# ======================================================================================================


# NEW FUNCTION: For validation purposes - determines which fuels are valid for analysis
def get_valid_fuel_types(category: str) -> List[str]:
    """
    Returns the list of valid fuel types for a category.
    
    Args:
        category: Equipment category name.
        
    Returns:
        List of valid fuel type strings for the specified category.
        
    Raises:
        ValueError: If an invalid category is provided.
    """
    # Tech filters handle excluding heat pump technologies for heating and water heating
    # So we can keep electricity as a valid fuel type.
    if category in ['heating', 'waterHeating']:
        return ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil']
    
    # Heat pump clothes dryers are different from the existing electric resistance dryers in EUSS.
    # So we can keep electricity as a valid fuel type for clothes drying.
    elif category == 'clothesDrying':
        return ['Electricity', 'Natural Gas', 'Propane']
    
    # We exclude electricity for cooking because the electric upgrade in MP7 is the same technology as the baseline.
    elif category == 'cooking':
        return ['Natural Gas', 'Propane']
    
    else:
        raise ValueError(f"Invalid category. Must be one of the following: {EQUIPMENT_SPECS.keys()}")


# NEW FUNCTION: For masking purposes - gets ALL consumption columns regardless of validity
def get_all_possible_fuel_columns(category: str) -> List[str]:
    """
    Returns all possible fuel consumption columns for a category.
    
    Args:
        category: Equipment category name.
        
    Returns:
        List of column names for all possible fuel consumption measurements.
        
    Raises:
        ValueError: If an invalid category is provided.
    """
    if category not in EQUIPMENT_SPECS:
        raise ValueError(f"Invalid category. Must be one of the following: {EQUIPMENT_SPECS.keys()}")
    
    if category in ['heating', 'waterHeating']:
        # All four fuel types are available for heating and water heating
        return [f'base_{fuel}_{category}_consumption' for fuel in FUEL_MAPPING.values()]
    
    else:
        # Fuel oil is not available for clothes drying or cooking
        return [f'base_{fuel}_{category}_consumption' for fuel in FUEL_MAPPING.values() 
                if fuel != 'fuelOil']


def get_post_retrofit_columns(category: str, menu_mp: int) -> List[str]:
    """
    Returns the post-retrofit consumption column name for a category and measure package.
    
    Args:
        category: Equipment category name.
        menu_mp: The measure package number.
        
    Returns:
        List containing the post-retrofit consumption column name.
        
    Raises:
        ValueError: If an invalid category is provided.
    """
    if category not in EQUIPMENT_SPECS:
        raise ValueError(f"Invalid category. Must be one of the following: {EQUIPMENT_SPECS.keys()}")
    
    # Just return the basic consumption column for this measure package and category
    return [f'mp{menu_mp}_{category}_consumption']


def identify_valid_homes(df: pd.DataFrame) -> pd.DataFrame:
    """Creates comprehensive data quality flags for all categories.
    
    This function adds columns to track the quality and validity of data
    across all equipment categories. Technology validation is only applied
    to heating and water heating categories.
    
    Args:
        df: DataFrame containing energy consumption data.
        
    Returns:
        DataFrame with added data quality flags.
    """
    # Initialize the overall inclusion flag
    df['include_all'] = True
    print("\nCreating data quality flags for all categories")
    
    for category in EQUIPMENT_SPECS.keys():
        print(f"\n--- Processing {category} ---")
        
        # Create fuel validity flag
        fuel_flag = f'valid_fuel_{category}'
        fuel_col = f'base_{category}_fuel'
        
        # UPDATED: Uses get_valid_fuel_types() instead of previous validation approach
        if fuel_col in df.columns:
            # Print some diagnostic info about the values
            print(f"Values in {fuel_col} (top 5):")
            print(df[fuel_col].value_counts().head(5))
            
            # Get valid fuel types for this category
            valid_fuel_types = get_valid_fuel_types(category)
            df[fuel_flag] = df[fuel_col].isin(valid_fuel_types)

            # Invalid fuel count and percentage
            invalid_fuel_count = (~df[fuel_flag]).sum()
            invalid_fuel_pct = (invalid_fuel_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {category}: Found {invalid_fuel_count} homes ({invalid_fuel_pct:.1f}%) with invalid fuel types")
            
            # Show what's being filtered
            if invalid_fuel_count > 0:
                invalid_fuels = df.loc[~df[fuel_flag], fuel_col].value_counts()
                print("  Invalid fuel types (top 5):")
                print(invalid_fuels.head(5))
        else:
            print(f"  Warning: Column {fuel_col} not found")
            df[fuel_flag] = True
        
        # Handle technology validation only for heating and water heating
        if category in ['heating', 'waterHeating']:
            # Create technology validity flag
            tech_flag = f'valid_tech_{category}'
            tech_col = f'{category}_type'
            
            if tech_col in df.columns and category in ALLOWED_TECHNOLOGIES:
                # Print some diagnostic info
                print(f"Values in {tech_col} (top 5):")
                print(df[tech_col].value_counts().head(5))
                
                print(f"Allowed values for {category}:")
                print(ALLOWED_TECHNOLOGIES[category])
                
                # Check if the technology type is in the allowed list
                df[tech_flag] = df[tech_col].isin(ALLOWED_TECHNOLOGIES[category])

                # Invalid technology count and percentage
                invalid_tech_count = (~df[tech_flag]).sum()
                invalid_tech_pct = (invalid_tech_count / len(df)) * 100 if len(df) > 0 else 0
                print(f"  {category}: Found {invalid_tech_count} homes ({invalid_tech_pct:.1f}%) with invalid technology types")
                
                # Show what's being filtered
                if invalid_tech_count > 0:
                    invalid_techs = df.loc[~df[tech_flag], tech_col].value_counts()
                    print("  Invalid technology types (top 5):")
                    print(invalid_techs.head(5))
                
                # Create category inclusion flag based on both fuel and tech validity
                include_col = f'include_{category}'
                df[include_col] = df[fuel_flag] & df[tech_flag]
            else:
                if category not in ALLOWED_TECHNOLOGIES:
                    print(f"  {category}: No allowed technologies defined")
                elif tech_col not in df.columns:
                    print(f"  {category}: Warning - Column {tech_col} not found")
                
                # Set inclusion flag based only on fuel validity
                include_col = f'include_{category}'
                df[include_col] = df[fuel_flag]
        else:
            # For clothes drying and cooking, only use fuel validation
            print(f"  {category}: Technology validation not applicable (no technology type column)")
            include_col = f'include_{category}'
            df[include_col] = df[fuel_flag]
        
        # Print exclusion summary
        excluded_count = (~df[include_col]).sum()
        excluded_pct = (excluded_count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {category}: Total {excluded_count} homes ({excluded_pct:.1f}%) excluded from analysis")
        
        # Update the overall inclusion flag
        df['include_all'] &= df[include_col]
    
    overall_excluded = (~df['include_all']).sum()
    overall_pct = (overall_excluded / len(df)) * 100 if len(df) > 0 else 0
    print(f"\nTotal {overall_excluded} homes ({overall_pct:.1f}%) excluded from all categories")
    return df


def mask_invalid_data(df: pd.DataFrame, menu_mp: Optional[int] = None) -> pd.DataFrame:
    """
    Sets consumption values to NaN based on inclusion flags.
    
    Args:
        df: DataFrame with inclusion flags already created.
        menu_mp: Optional measure package number for post-retrofit masking.
        
    Returns:
        DataFrame with consumption values set to NaN for invalid records.
    """
    print("Applying NaN masking based on inclusion flags")
    
    for category in EQUIPMENT_SPECS.keys():
        include_col = f'include_{category}'
        
        if include_col not in df.columns:
            print(f"  {category}: Warning - Inclusion flag '{include_col}' not found. Skipping masking.")
            continue
        
        # Get all baseline consumption columns for this category
        columns_to_mask = get_all_possible_fuel_columns(category)
        
        # Add the total baseline consumption column
        total_col = f'baseline_{category}_consumption'
        if total_col in df.columns:
            columns_to_mask.append(total_col)
            
        # Add post-retrofit column if menu_mp is provided
        if menu_mp != 0:
            post_retrofit_cols = get_post_retrofit_columns(category, menu_mp)
            columns_to_mask.extend(post_retrofit_cols)
        
        # Apply masking to all collected columns
        df = mask_category_specific_data(df, columns_to_mask, category, verbose=True)
    
    return df


# ====================================================================================================================================================================
# UTILITY FUNCTION FOR APPLYING CATEGORY-SPECIFIC MASKING
# ====================================================================================================================================================================

def mask_category_specific_data(df: pd.DataFrame, 
                                 columns: List[str], 
                                 category: str,
                                 verbose: bool = False) -> pd.DataFrame:
    """
    Applies NaN masking to specified columns based on a category's inclusion flag.
    
    This utility function applies NaN masking to all provided columns based 
    on the inclusion flag for the specified category. It can be used anywhere
    in the codebase after calculations to ensure data quality.
    
    Args:
        df: DataFrame with inclusion flags already created.
        columns: List of column names to apply masking to.
        category: The equipment category that determines which inclusion flag to use.
        verbose: Whether to print details about masking operations.
        
    Returns:
        DataFrame with specified columns masked based on the category's inclusion flag.
        
    Raises:
        ValueError: If the category's inclusion flag is not found in the DataFrame.
    """
    include_col = f'include_{category}'
    
    if include_col not in df.columns:
        raise ValueError(f"Inclusion flag '{include_col}' not found in DataFrame")
    
    # Filter out columns that don't exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        if verbose:
            print(f"  {category}: No valid columns found for masking")
        return df
    
    masked_count = 0
    for col in valid_columns:
        # Count values before masking for better reporting
        non_nan_before = df[col].notna().sum()
        df.loc[~df[include_col], col] = np.nan
        non_nan_after = df[col].notna().sum()
        masked_this_col = non_nan_before - non_nan_after
        
        if masked_this_col > 0 and verbose:
            print(f"    {col}: Masked {masked_this_col} values")
            masked_count += masked_this_col
    
    if masked_count > 0 and verbose:
        print(f"  {category}: Masked {masked_count} values across {len(valid_columns)} columns")
    
    return df

# ====================================================================================================================================================================
# COMPREHENSIVE FUNCTION FOR VALIDATING DATA AND APPLYING MASKING
# ====================================================================================================================================================================

def get_valid_calculation_mask(
    df: pd.DataFrame, 
    category: str, 
    menu_mp: Union[int, str] = 0,
    verbose: bool = True
) -> pd.Series:
    """
    Combines data validation and retrofit status for comprehensive masking.
    
    This function addresses a key integration issue between the data validation
    system and the retrofit status tracking system. It ensures:
    - For baseline scenarios: Only homes with valid data are processed
    - For measure packages: Only homes with both valid data AND scheduled for retrofits are processed
    
    Args:
        df: DataFrame containing the validation flags and retrofit information.
        category: Equipment category (e.g., 'heating', 'waterHeating').
        menu_mp: Measure package identifier (0 for baseline, nonzero for measure packages).
        verbose: Whether to print information about valid homes.
        
    Returns:
        Series of boolean values indicating which homes should be included in calculations.
        
    Raises:
        ValueError: If the inclusion flag for the given category doesn't exist in the DataFrame.
    """
    # Standardize menu_mp to facilitate comparisons
    menu_mp_str = str(menu_mp)
    is_baseline = menu_mp_str == "0"
    
    # Check if inclusion flag exists
    include_col = f'include_{category}'
    if include_col not in df.columns:
        raise ValueError(f"Inclusion flag '{include_col}' not found in DataFrame. "
                         f"Ensure identify_valid_homes() has been called.")
    
    # Get data validation mask
    data_valid_mask = df[include_col]
    
    # For baseline scenarios, only use data validation
    if is_baseline:
        if verbose:
            valid_count = data_valid_mask.sum()
            invalid_count = (~data_valid_mask).sum()
            print(f"Baseline calculation for {category}:")
            print(f"  - {valid_count} homes have valid data")
            print(f"  - {invalid_count} homes have invalid data (values will be NaN)")
        
        return data_valid_mask
    # For measure packages, combine with retrofit status
    else:        
        retrofit_mask = get_retrofit_homes_mask(df, category, menu_mp, verbose=False)
        combined_mask = data_valid_mask & retrofit_mask
        
        if verbose:
            valid_data_count = data_valid_mask.sum()
            retrofit_count = retrofit_mask.sum()
            final_count = combined_mask.sum()
            
            print(f"Measure package calculation for {category}:")
            print(f"  - {valid_data_count} homes have valid baseline data")
            print(f"  - {retrofit_count} homes will receive retrofits")
            print(f"  - {final_count} homes have both valid data AND will receive retrofits")
            print(f"  - {len(df) - final_count} homes excluded (values will be NaN)")
        
        # Check if all homes are excluded
        if combined_mask.sum() == 0:
            print(f"WARNING: All homes excluded for {category}. Check data quality and retrofit criteria.")
        
        return combined_mask