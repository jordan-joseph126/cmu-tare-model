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
    pre_col = f'preIRA_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    ira_col = f'iraRef_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    
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
    
    # Filter to keep only columns with these tiers
    keep_columns = [col for col in df.columns if col[1] in tier_order]
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
    sorted_columns = sorted(result_df.columns, key=sort_key)
    
    # Return sorted DataFrame
    return result_df.loc[:, sorted_columns]


# ==============================================================================
# NEEDS TO BE UPDATED FOR SIMPLIFIED APPROACH ABOVE!!!
# ==============================================================================
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
    # Filter for adoption tier columns
    adoption_tiers = ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 
                     'Tier 3: Subsidy-Dependent Feasibility']
    
    adoption_data = df.loc[:, df.columns.get_level_values(1).isin(adoption_tiers)]
    adoption_data.columns = adoption_data.columns.remove_unused_levels()

    # Define color mapping
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    # Set up plot dimensions
    n = len(adoption_data.index)
    bar_width = 0.35
    index = list(range(n))

    # Plot each scenario as a group of stacked bars
    for i, scenario in enumerate(scenarios):
        # Check if all tier columns exist for this scenario
        has_all_tiers = all((scenario, tier) in adoption_data.columns for tier in adoption_tiers)
        
        if has_all_tiers:
            # Get tier values
            tier1 = adoption_data[scenario, 'Tier 1: Feasible'].values
            tier2 = adoption_data[scenario, 'Tier 2: Feasible vs. Alternative'].values
            tier3 = adoption_data[scenario, 'Tier 3: Subsidy-Dependent Feasibility'].values

            # Position bars
            scenario_index = np.array(index) + i * bar_width
            
            # Create stacked bars
            ax.bar(scenario_index, tier1, bar_width, 
                 color=color_mapping['Tier 1: Feasible'], edgecolor='white')
            ax.bar(scenario_index, tier2, bar_width, bottom=tier1, 
                 color=color_mapping['Tier 2: Feasible vs. Alternative'], edgecolor='white')
            ax.bar(scenario_index, tier3, bar_width, bottom=(tier1+tier2), 
                 color=color_mapping['Tier 3: Subsidy-Dependent Feasibility'], edgecolor='white')

    # Configure plot appearance
    ax.set_xlabel(x_label, fontweight='bold', fontsize=20)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=20)
    ax.set_title(title, fontweight='bold', fontsize=20)
    
    # Set x-ticks at the middle of each group
    ax.set_xticks([i + bar_width / 2 for i in range(n)])
    ax.set_xticklabels([f'{name[1]}' for name in adoption_data.index.tolist()], 
                      rotation=90, ha='right')

    # Configure tick labels
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # Set y-axis limits
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylim(0, 100)


def filter_by_fuel(df: pd.DataFrame, fuel_types: List[str]) -> pd.DataFrame:
    """
    Filters DataFrame by fuel type across different index structures.
    
    Args:
        df: DataFrame to filter
        fuel_types: List of fuel types to include
        
    Returns:
        Filtered DataFrame
    """
    # Handle MultiIndex case
    if isinstance(df.index, pd.MultiIndex):
        # Try to find fuel level in index
        fuel_levels = [i for i, name in enumerate(df.index.names) 
                     if name and 'fuel' in str(name).lower()]
        
        if fuel_levels:
            # Filter by the first fuel level found
            return df.loc[df.index.get_level_values(fuel_levels[0]).isin(fuel_types)]
    
    # Handle regular index with fuel column
    elif 'fuel' in df.columns:
        return df[df['fuel'].isin(fuel_types)]
    
    # If no matching structure, return unfiltered
    print("Warning: Could not apply fuel filter. Structure not recognized.")
    return df
