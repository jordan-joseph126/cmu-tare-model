import pandas as pd

# import functions.tare_setup as tare_setup
from cmu_tare_model.energy_consumption_and_metadata.project_future_energy_consumption import lookup_hdd_factor

EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}

def precompute_hdd_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute heating degree day (HDD) factors for each region and year.

    Parameters:
        df (DataFrame): Input data.

    Returns:
        dict: HDD factors mapped by year and region.
    """
    
    max_lifetime = max(EQUIPMENT_SPECS.values())
    years = range(2024, 2024 + max_lifetime + 1)
    hdd_factors_df = pd.DataFrame(index=df.index, columns=years)

    for year_label in years:
        hdd_factors_df[year_label] = df['census_division'].map(
            lambda x: lookup_hdd_factor.get(x, lookup_hdd_factor['National']).get(year_label, 1.0)
        )

    return hdd_factors_df
