"""Adoption KPI computation modules for the TARE model."""
from .data_loading import load_euss_baseline, load_euss_upgrade, mp_to_upgrade
from .spark_gap import calculate_spark_gap
from .thermal_cop import compute_thermal_cop, compute_breakeven_cop
from .demand import compute_scenario_demand, aggregate_demand, aggregate_demand_by_state
from .bill_savings import compute_bill_savings_ratio, aggregate_bill_savings, aggregate_bill_savings_by_state
