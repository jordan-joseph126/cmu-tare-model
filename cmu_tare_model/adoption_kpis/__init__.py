"""Adoption KPI computation modules for the TARE model."""
from .data_loading import load_euss_baseline, load_euss_upgrade, mp_to_upgrade
from .spark_gap import calculate_spark_gap
from .thermal_cop import compute_thermal_cop, compute_breakeven_cop, assign_breakeven_category
from .demand import compute_scenario_demand, aggregate_demand, aggregate_demand_by_state
from .bill_savings import compute_bill_savings_ratio, aggregate_bill_savings, aggregate_bill_savings_by_state
from .compute_adoption_rate import compute_adoption_rate
from .visualize_geospatial_data import (
    prepare_state_geodataframe,
    create_choropleth_map,
    plot_combined_choropleth,
    plot_national_county_choropleth,
    prepare_county_geodataframe,
    plot_categorical_breakeven_map,
)
from .visualize_tabular_data import pct_change, make_symmetric_norm, print_column_summary
