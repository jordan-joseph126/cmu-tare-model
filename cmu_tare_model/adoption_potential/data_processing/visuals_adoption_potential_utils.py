import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Dict, Any, Union

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def verify_columns_exist(
        df: pd.DataFrame, 
        required_cols: List[str], 
        optional_cols: Optional[List[str]] = None) -> List[str]:
    """
    Verifies that required columns exist in the DataFrame and returns available columns.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
        optional_cols: List of optional column names (defaults to None)
        
    Returns:
        List of column names that exist in the DataFrame
        
    Raises:
        ValueError: If any required columns are missing
    """
    # Check required columns
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing: {missing_required}")
        
    # Start with required columns that exist
    existing_cols = [col for col in required_cols if col in df.columns]
    
    # Add optional columns that exist
    if optional_cols:
        existing_optional = [col for col in optional_cols if col in df.columns]
        existing_cols.extend(existing_optional)
    
    return existing_cols


def detect_adoption_columns(
    df: pd.DataFrame,
    menu_mp: int,
    category: str,
    scc: str = 'upper',
    rcm_model: str = 'AP2',
    cr_function: str = 'acs'
) -> List[str]:
    """
    Detects the two adoption columns following the simplified naming convention.

    Args:
        df: DataFrame to check.
        menu_mp: Measure package identifier.
        category: Equipment category.
        scc: Social cost of carbon assumption.
        rcm_model: RCM model.
        cr_function: Concentration-response function.

    Returns:
        List containing [preIRA_adoption_col, iraRef_adoption_col].

    Raises:
        ValueError: If either column is missing.
    """
    # Fix: Ensure we're using the parameters passed to the function
    pre_col = f'preIRA_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    ira_col = f'iraRef_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    
    # Print diagnostic info about columns
    print(f"Looking for adoption columns:")
    print(f"  - Pre-IRA: {pre_col}")
    print(f"  - IRA: {ira_col}")
    
    # Use verify_columns_exist for consistent column checking
    return verify_columns_exist(df, [pre_col, ira_col])


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and sorts the DataFrame columns by scenario and tier in the specified order.
    
    Args:
        df: DataFrame with multi-index columns
    
    Returns:
        Filtered DataFrame with columns ordered by scenario then tier
    """
    # Define desired scenario and tier orders
    scenario_order = {
        # Extract the prefix part like 'preIRA' or 'iraRef'
        name.split('_')[0]: i for i, name in enumerate(['preIRA', 'iraRef'])
    }
    
    tier_order = {
        'Tier 1: Feasible': 0,
        'Tier 2: Feasible vs. Alternative': 1,
        'Tier 3: Subsidy-Dependent Feasibility': 2,
        'Total Adoption Potential': 3,
        'Total Adoption Potential (Additional Subsidy)': 4
    }
    
    # Print column structure for debugging
    print("\nColumn structure in filter_columns:")
    print(f"  - DataFrame shape: {df.shape}")
    print(f"  - Column type: {type(df.columns)}")
    if isinstance(df.columns, pd.MultiIndex):
        print(f"  - Column levels: {df.columns.names}")
        print(f"  - First few columns: {[col for col in df.columns[:3]]}")
    else:
        print(f"  - First few columns: {list(df.columns[:3])}")
    
    # Filter to keep only columns with these tiers
    if isinstance(df.columns, pd.MultiIndex):
        keep_columns = [col for col in df.columns if col[1] in tier_order]
        
        if not keep_columns:
            print("  Warning: No columns match tier ordering. Keeping all columns.")
            keep_columns = df.columns
    else:
        # If not MultiIndex, cannot filter by tier
        print("  Warning: Not a MultiIndex. Cannot filter by tier.")
        return df
    
    # Keep a copy of the original DataFrame with filtered columns
    result_df = df.loc[:, keep_columns]
    
    # Create custom sorting key function
    def sort_key(column):
        # Extract scenario prefix
        scenario_prefix = column[0].split('_')[0]
        # Get scenario order (default to high value if not found)
        scenario_idx = scenario_order.get(scenario_prefix, 999)
        # Get tier order (default to high value if not found)
        tier_idx = tier_order.get(column[1], 999)
        # Return tuple for sorting (first by scenario, then by tier)
        return (scenario_idx, tier_idx)
    
    # Sort columns using the custom sorting key
    try:
        sorted_columns = sorted(result_df.columns, key=sort_key)
        # Return sorted DataFrame
        return result_df.loc[:, sorted_columns]
    except Exception as e:
        print(f"  Error sorting columns: {str(e)}")
        # Return unsorted filtered DataFrame if sorting fails
        return result_df


def plot_adoption_rate_bar(
        df: pd.DataFrame, 
        scenarios: List[str], 
        title: str, 
        x_label: str, 
        y_label: str, 
        ax: plt.Axes) -> None:
    """
    Creates a stacked bar chart visualizing adoption rates by tier.
    
    Args:
        df: DataFrame with multi-index columns containing adoption percentages
        scenarios: List of column names for adoption data
        title: Title for the plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        ax: Matplotlib axes to plot on
    """
    # Print detailed diagnostics about the input DataFrame
    print(f"\nPlotting '{title}':")
    print(f"- Shape: {df.shape}")
    print(f"- Index: {type(df.index)}")
    if isinstance(df.columns, pd.MultiIndex):
        print(f"- Column levels: {df.columns.names}")
        print(f"- First few columns: {[col for col in df.columns[:3]]}")
    else:
        print(f"- Columns: {list(df.columns[:5])}")
    
    # Print requested scenarios
    print(f"- Requested scenarios: {scenarios}")
    
    # Define tiers and their display order
    adoption_tiers = [
        'Tier 1: Feasible', 
        'Tier 2: Feasible vs. Alternative',
        'Tier 3: Subsidy-Dependent Feasibility'
    ]
    
    # Define color mapping
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    # Check if DataFrame has expected MultiIndex structure
    if df.empty or not isinstance(df.columns, pd.MultiIndex):
        ax.text(0.5, 0.5, "No data or incorrect column structure", 
              ha='center', va='center', transform=ax.transAxes)
        print("  Error: DataFrame is empty or doesn't have MultiIndex columns")
        return
    
    # Check if tiers exist in data
    available_tiers = set([col[1] for col in df.columns if isinstance(col, tuple) and len(col) > 1])
    print(f"- Available tiers: {available_tiers}")
    missing_tiers = set(adoption_tiers) - available_tiers
    if missing_tiers:
        print(f"  Warning: Missing tiers: {missing_tiers}")
    
    # Print some diagnostic info about tiers
    tier_check = []
    for scenario in scenarios:
        tier_check.append(f"Scenario {scenario} tiers:")
        for tier in adoption_tiers:
            if (scenario, tier) in df.columns:
                values = df[(scenario, tier)]
                non_zero = (values > 0).sum()
                total = len(values)
                tier_check.append(f"  - {tier}: {non_zero}/{total} non-zero values ({non_zero/total*100:.1f}%)")
                # Print min/max/mean for non-zero values
                if non_zero > 0:
                    non_zero_values = values[values > 0]
                    tier_check.append(f"    Min: {non_zero_values.min():.1f}, Max: {non_zero_values.max():.1f}, Mean: {non_zero_values.mean():.1f}")
            else:
                tier_check.append(f"  - {tier}: Missing from DataFrame")
    
    print("\n".join(tier_check))
    
    # Filter columns to ensure proper ordering
    try:
        # Filter columns only if they have tier information
        tier_columns = [col for col in df.columns if col[1] in adoption_tiers]
        if tier_columns:
            df_filtered = filter_columns(df)
        else:
            print("  Warning: No tier columns found. Using original DataFrame.")
            df_filtered = df
    except Exception as e:
        print(f"  Error filtering columns: {str(e)}")
        df_filtered = df
    
    # Get rows and prepare for plotting
    n = len(df_filtered.index)
    bar_width = 0.35
    index = list(range(n))
    
    # Plot each scenario
    for i, scenario in enumerate(scenarios):
        # Verify scenario exists in columns
        scenario_columns = [col for col in df_filtered.columns if col[0] == scenario]
        if not scenario_columns:
            print(f"  Warning: Scenario '{scenario}' not found in columns")
            continue
            
        # Position bars for this scenario
        scenario_index = np.array(index) + i * bar_width
        
        # Initialize bottom values for stacking
        bottom = np.zeros(n)
        
        # Plot each tier as a stacked bar
        for tier in adoption_tiers:
            if (scenario, tier) in df_filtered.columns:
                tier_values = df_filtered[(scenario, tier)].values
                ax.bar(scenario_index, tier_values, bar_width, 
                     bottom=bottom, color=color_mapping[tier], 
                     edgecolor='white', label=tier if i == 0 else "")
                # Update bottom for next tier
                bottom += tier_values
            else:
                print(f"  Warning: Tier '{tier}' not found for scenario '{scenario}'")
    
    # Configure plot appearance
    ax.set_xlabel(x_label, fontweight='bold', fontsize=20)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=20)
    ax.set_title(title, fontweight='bold', fontsize=20)
    
    # Set x-ticks at the middle of each group
    ax.set_xticks([i + bar_width / 2 * (len(scenarios) - 1) for i in range(n)])
    
    # Handle different index structures for x-tick labels
    if isinstance(df_filtered.index, pd.MultiIndex):
        # For MultiIndex, use all levels but format them nicely
        x_labels = []
        for idx in df_filtered.index:
            # Join the index levels, but skip None or empty values
            label_parts = [str(level) for level in idx if level is not None and str(level).strip()]
            x_labels.append('\n'.join(label_parts))
    else:
        # For regular index, use index values directly
        x_labels = [str(idx) for idx in df_filtered.index]
    
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # Configure tick labels
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # Set y-axis limits
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylim(0, 100)


def filter_by_fuel(
        df: pd.DataFrame,
        fuel_types: List[str]) -> pd.DataFrame:
    """
    Filters DataFrame by fuel type across different index structures.
    
    Args:
        df: DataFrame to filter
        fuel_types: List of fuel types to include
        
    Returns:
        Filtered DataFrame
    """
    # Print diagnostic info
    print(f"\nFiltering by fuel types: {fuel_types}")
    print(f"- DataFrame shape before filtering: {df.shape}")
    print(f"- Index type: {type(df.index)}")
    
    if isinstance(df.index, pd.MultiIndex):
        print(f"- Index levels: {df.index.names}")
    elif len(df.columns) > 0:
        print(f"- First few columns: {list(df.columns[:5])}")
    
    try:
        # Handle MultiIndex case
        if isinstance(df.index, pd.MultiIndex):
            # Check for exact fuel level name first
            if 'base_fuel' in df.index.names:
                result = df.loc[df.index.get_level_values('base_fuel').isin(fuel_types)]
                print(f"- Filtered by 'base_fuel' index level. Shape after filtering: {result.shape}")
                return result
            
            # Check for category-specific fuel columns in index
            category_fuel_levels = []
            for i, name in enumerate(df.index.names):
                if name and '_fuel' in str(name).lower():
                    category_fuel_levels.append((i, name))
            
            if category_fuel_levels:
                # Use the first matching level
                level_idx, level_name = category_fuel_levels[0]
                print(f"- Filtering by fuel type using index level: {level_name}")
                result = df.loc[df.index.get_level_values(level_idx).isin(fuel_types)]
                print(f"- Shape after filtering: {result.shape}")
                return result
        
        # Handle DataFrame columns - check for standard naming patterns
        
        # First try category-specific fuel columns (e.g., 'base_heating_fuel')
        category_fuel_cols = [col for col in df.columns if '_fuel' in col.lower()]
        if category_fuel_cols:
            col = category_fuel_cols[0]  # Use the first matching column
            print(f"- Filtering by column: {col}")
            result = df[df[col].isin(fuel_types)]
            print(f"- Shape after filtering: {result.shape}")
            return result
        
        # Then try generic 'fuel' column
        elif 'fuel' in df.columns:
            print("- Filtering by 'fuel' column")
            result = df[df['fuel'].isin(fuel_types)]
            print(f"- Shape after filtering: {result.shape}")
            return result
            
        # If no matching structure, return unfiltered
        print("! Warning: Could not identify fuel column. Structure not recognized.")
        print(f"- Available columns: {list(df.columns[:5])}")
        print(f"- Available index levels: {df.index.names}")
        return df
        
    except Exception as e:
        print(f"! Error filtering by fuel: {str(e)}")
        return df
