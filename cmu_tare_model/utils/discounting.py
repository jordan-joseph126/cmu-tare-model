# For climate and health impacts module
def calculate_discount_factor(base_year: int, target_year: int, discounting_method: str) -> float:
    """
    Calculate the discount factor to convert future values to present values.
    
    Args:
        base_year (int): The reference year to discount to (e.g., 2024)
        target_year (int): The future year to discount from (e.g., 2030)
        discount_rate (float): Annual social discount rate (default: 0.02 or 2%)
        
    Returns:
        float: Discount factor to multiply with future values
    """
    if discounting_method == 'public':
        discount_rate = 0.02
    elif discounting_method == 'private_fixed':
        discount_rate = 0.07
    else:
        raise ValueError("Invalid discounting method. Use 'public' or 'private_fixed'.")
    
    years_difference = target_year - base_year
    # Cannot have negative years (future must be >= base year)
    years_difference = max(0, years_difference)
    
    # Formula: PV = FV / (1+r)^t
    discount_factor = 1 / ((1 + discount_rate) ** years_difference)
    
    return discount_factor
