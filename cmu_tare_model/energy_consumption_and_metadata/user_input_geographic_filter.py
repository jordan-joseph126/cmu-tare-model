import pandas as pd
from typing import List

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
USER INPUT FOR GEOGRAPHIC FILTERS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

menu_prompt = """
Would you like to filter for a specific state's data? Please enter one of the following:
N. I'd like to analyze all of the United States.
Y. I'd like to filter data for a specific state.
"""

city_prompt = """
To accurately characterize load profile, it is recommended to select subsets of data with >= 1000 models (~240,000 representative dwelling units).

The following cities (number of models also shown) are available for this state:
"""

city_menu_prompt = """
Would you like to filter a subset of city-level data? Please enter one of the following:
N. I'd like to analyze all of my selected state.
Y. I'd like to filter by city in the state.
"""

def get_menu_choice(prompt: str,
                    choices: List[str]) -> str:
    """
    Prompts the user with a menu and returns a validated choice.

    This function loops until the user inputs a valid option
    that exists in the provided choices.

    Args:
        prompt: The message to display to the user.
        choices: A list of valid choices (e.g., ['N', 'Y']).

    Returns:
        The validated user input.
    """
    while True:
        choice = input(prompt).upper()
        if choice in choices:
            return choice
        print("Invalid option. Please try again.")


def get_state_choice(df_copy: pd.DataFrame) -> str:
    """Prompts the user to input a two-letter state abbreviation and validates it.

    This function repeats until the user provides a valid state abbreviation
    that exists within the 'in.state' column of the given DataFrame.

    Args:
        df_copy: A pandas DataFrame that includes an 'in.state' column.

    Returns:
        The two-letter abbreviation of the state.

    Raises:
        ValueError: If the DataFrame does not contain an 'in.state' column.
    """
    if 'in.state' not in df_copy.columns:
        raise ValueError("DataFrame missing required 'in.state' column")
        
    while True:
        input_state = input("Which state would you like to analyze data for? Please enter the two-letter abbreviation: ").upper()
        if df_copy['in.state'].eq(input_state).any():
            return input_state
        print("Invalid state abbreviation. Please try again.")


def get_city_choice(df_copy: pd.DataFrame,
                    input_state: str) -> str:
    """Prompts the user to input a city name within a given state and validates it.

    This function repeats until the user provides a valid city name
    that exists in the 'in.city' column of the DataFrame, matching
    the specified state.

    Args:
        df_copy: A pandas DataFrame that includes an 'in.city' column.
        input_state: The two-letter abbreviation of the selected state.

    Returns:
        The validated city name.

    Raises:
        ValueError: If the DataFrame does not contain an 'in.city' column.
    """
    if 'in.city' not in df_copy.columns:
        raise ValueError("DataFrame missing required 'in.city' column")
        
    while True:
        input_cityFilter = input("Please enter the city name ONLY (e.g., Pittsburgh): ")
        # city_filter checks for an exact match of state and city in the format 'ST, CityName'
        city_filter = df_copy['in.city'].eq(f"{input_state}, {input_cityFilter}")
        if city_filter.any():
            return input_cityFilter
        print("Invalid city name. Please try again.")
