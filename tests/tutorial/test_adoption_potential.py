import unittest
import pandas as pd

# Assuming calculate_public_npv and adoption_decision are defined elsewhere
# from your_module import calculate_public_npv, adoption_decision

class TestAdoptionPotential(unittest.TestCase):
    def setUp(self):
        # Initialize your dataframes here
        self.df_base = pd.DataFrame()
        self.df_baseline_damages = pd.DataFrame()
        self.df_mp_damages_template = pd.DataFrame()
        self.policy_scenario = "Some Policy Scenario"

        # Add additional test data for other equipment types
        self.df_baseline_damages['baseline_2024_waterHeating_damages_climate_lrmer'] = [150, 250]
        self.df_baseline_damages['baseline_2024_waterHeating_damages_health'] = [60, 70]
        self.df_baseline_damages['baseline_2024_clothesDrying_damages_climate_lrmer'] = [80, 120]
        self.df_baseline_damages['baseline_2024_clothesDrying_damages_health'] = [30, 40]
        self.df_baseline_damages['baseline_2024_cooking_damages_climate_lrmer'] = [70, 90]
        self.df_baseline_damages['baseline_2024_cooking_damages_health'] = [20, 25]

        # Add SRMER data
        self.df_baseline_damages['baseline_2024_heating_damages_climate_srmer'] = [95, 190]
        self.df_baseline_damages['baseline_2025_heating_damages_climate_srmer'] = [85, 180]

    def test_interest_rate_impact(self):
        """Test that different interest rates affect NPV calculations correctly"""
        menu_mp = 8
        rates = [0.02, 0.05, 0.10]
        npvs = []
        
        for rate in rates:
            result = calculate_public_npv(
                self.df_base,
                self.df_baseline_damages,
                self.df_mp_damages_template,
                menu_mp,
                self.policy_scenario,
                interest_rate=rate
            )
            npvs.append(result[f'iraRef_mp{menu_mp}_heating_public_npv_lrmer'][0])
        
        self.assertGreater(npvs[0], npvs[1])
        self.assertGreater(npvs[1], npvs[2])

    def test_other_equipment_types(self):
        """Test calculations for water heating, clothes drying, and cooking"""
        menu_mp = 8
        equipment_types = ['waterHeating', 'clothesDrying', 'cooking']
        
        for equip_type in equipment_types:
            with self.subTest(equipment_type=equip_type):
                result = calculate_public_npv(
                    self.df_base,
                    self.df_baseline_damages,
                    self.df_mp_damages_template,
                    menu_mp,
                    self.policy_scenario
                )
                
                self.assertIn(
                    f'iraRef_mp{menu_mp}_{equip_type}_public_npv_lrmer',
                    result.columns
                )
                self.assertIn(
                    f'iraRef_mp{menu_mp}_{equip_type}_climate_npv_lrmer',
                    result.columns
                )
                self.assertIn(
                    f'iraRef_mp{menu_mp}_{equip_type}_health_npv',
                    result.columns
                )

    def test_srmer_calculations(self):
        """Test that SRMER calculations work correctly"""
        menu_mp = 8
        result = calculate_public_npv(
            self.df_base,
            self.df_baseline_damages,
            self.df_mp_damages_template,
            menu_mp,
            self.policy_scenario
        )
        
        self.assertIn(f'iraRef_mp{menu_mp}_heating_public_npv_lrmer', result.columns)
        self.assertIn(f'iraRef_mp{menu_mp}_heating_public_npv_srmer', result.columns)
        
        lrmer_value = result[f'iraRef_mp{menu_mp}_heating_public_npv_lrmer'][0]
        srmer_value = result[f'iraRef_mp{menu_mp}_heating_public_npv_srmer'][0]
        self.assertNotEqual(lrmer_value, srmer_value)
        self.assertGreater(lrmer_value * 0.5, srmer_value * 0.1)

    def test_no_ira_scenario(self):
        """Test calculations under No IRA scenario"""
        menu_mp = 8
        result = calculate_public_npv(
            self.df_base,
            self.df_baseline_damages,
            self.df_mp_damages_template,
            menu_mp,
            "No Inflation Reduction Act"
        )
        
        self.assertIn(f'preIRA_mp{menu_mp}_heating_public_npv_lrmer', result.columns)
        
        adoption_result = adoption_decision(result, "No Inflation Reduction Act")
        self.assertEqual(
            adoption_result[f'preIRA_mp{menu_mp}_heating_additional_public_benefit_lrmer'][0],
            0.0
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        menu_mp = 8
        
        zero_damages = self.df_baseline_damages.copy()
        zero_damages.loc[:, :] = 0
        
        result = calculate_public_npv(
            self.df_base,
            zero_damages,
            self.df_mp_damages_template,
            menu_mp,
            self.policy_scenario
        )
        
        self.assertLessEqual(
            result[f'iraRef_mp{menu_mp}_heating_public_npv_lrmer'].max(),
            0
        )
        
        large_damages = self.df_baseline_damages.copy()
        large_damages.loc[:, :] = 1e6
        
        result = calculate_public_npv(
            self.df_base,
            large_damages,
            self.df_mp_damages_template,
            menu_mp,
            self.policy_scenario
        )
        
        self.assertGreater(
            result[f'iraRef_mp{menu_mp}_heating_public_npv_lrmer'].min(),
            0
        )

# Run the tests
unittest.main(argv=[''], verbosity=2, exit=False)

