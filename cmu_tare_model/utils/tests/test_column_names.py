"""Comprehensive unit tests for column_names.py

This test suite ensures all column naming patterns work correctly across
REMDB v3 and v4 scenarios, preventing naming mismatch bugs.
"""

import pytest
from cmu_tare_model.utils.column_names import (
    create_cost_col,
    create_rebate_col,
    create_capital_col,
    create_npv_col,
    create_enclosure_cost_col,
    create_weatherization_rebate_col,
    create_installation_premium_col,
    create_combined_heating_cooling_col,
    create_climate_npv_col,
    create_health_npv_col,
    create_public_npv_col,
    create_lifetime_damages_col,
    create_avoided_damages_col,
    create_adoption_col,
    create_total_npv_col,
)


# =============================================================================
# TESTS: COST COLUMNS
# =============================================================================

class TestCostCol:
    """Test create_cost_col() function."""
    
    def test_remdb_v3_upgrade(self):
        """Test REMDB v3 upgrade cost column."""
        result = create_cost_col(3, 'heating', 'upgrade')
        assert result == 'mp3_heating_upgrade_installed_cost'
    
    def test_remdb_v3_replacement(self):
        """Test REMDB v3 replacement cost column."""
        result = create_cost_col(8, 'waterHeating', 'replacement')
        assert result == 'mp8_waterHeating_replacement_installed_cost'
    
    def test_remdb_v4_low(self):
        """Test REMDB v4 low percentile."""
        result = create_cost_col(3, 'heating', 'upgrade', cost_scenario='remdb_v4_low')
        assert result == 'mp3_heating_upgrade_installed_cost_low'
    
    def test_remdb_v4_mid(self):
        """Test REMDB v4 mid percentile."""
        result = create_cost_col(4, 'cooling', 'replacement', cost_scenario='remdb_v4_mid')
        assert result == 'mp4_cooling_replacement_installed_cost_mid'
    
    def test_remdb_v4_high(self):
        """Test REMDB v4 high percentile."""
        result = create_cost_col(7, 'clothesDrying', 'upgrade', cost_scenario='remdb_v4_high')
        assert result == 'mp7_clothesDrying_upgrade_installed_cost_high'
    
    def test_all_categories(self):
        """Test all equipment categories."""
        categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking', 'cooling']
        for cat in categories:
            result = create_cost_col(3, cat, 'upgrade')
            assert cat in result


class TestRebateCol:
    """Test create_rebate_col() function."""
    
    def test_remdb_v3(self):
        """Test REMDB v3 rebate column."""
        result = create_rebate_col(3, 'heating')
        assert result == 'mp3_heating_rebate_amount'
    
    def test_remdb_v4_low(self):
        """Test REMDB v4 low percentile rebate."""
        result = create_rebate_col(8, 'waterHeating', cost_scenario='remdb_v4_low')
        assert result == 'mp8_waterHeating_rebate_amount_low'
    
    def test_remdb_v4_mid(self):
        """Test REMDB v4 mid percentile rebate."""
        result = create_rebate_col(9, 'heating', cost_scenario='remdb_v4_mid')
        assert result == 'mp9_heating_rebate_amount_mid'
    
    def test_remdb_v4_high(self):
        """Test REMDB v4 high percentile rebate."""
        result = create_rebate_col(10, 'cooking', cost_scenario='remdb_v4_high')
        assert result == 'mp10_cooking_rebate_amount_high'


class TestCapitalCol:
    """Test create_capital_col() function."""
    
    def test_remdb_v3_total(self):
        """Test REMDB v3 total capital cost."""
        result = create_capital_col('iraRef_mp3_', 'heating', net=False)
        assert result == 'iraRef_mp3_heating_total_capital_cost'
    
    def test_remdb_v3_net(self):
        """Test REMDB v3 net capital cost."""
        result = create_capital_col('preIRA_mp8_', 'waterHeating', net=True)
        assert result == 'preIRA_mp8_waterHeating_net_capital_cost'
    
    def test_remdb_v4_total_mid(self):
        """Test REMDB v4 total capital cost with mid percentile."""
        result = create_capital_col('iraRef_mp3_', 'heating', net=False, cost_scenario='remdb_v4_mid')
        assert result == 'iraRef_mp3_heating_total_capital_cost_mid'
    
    def test_remdb_v4_net_high(self):
        """Test REMDB v4 net capital cost with high percentile."""
        result = create_capital_col('baseline_', 'cooling', net=True, cost_scenario='remdb_v4_high')
        assert result == 'baseline_cooling_net_capital_cost_high'
    
    def test_remdb_v4_low(self):
        """Test REMDB v4 low percentile."""
        result = create_capital_col('iraRef_mp9_', 'heating', net=False, cost_scenario='remdb_v4_low')
        assert result == 'iraRef_mp9_heating_total_capital_cost_low'


class TestNpvCol:
    """Test create_npv_col() function."""
    
    def test_remdb_v3_less_wtp_no_suffix(self):
        """Test REMDB v3 lessWTP without method suffix."""
        result = create_npv_col('iraRef_mp3_', 'heating', 'lessWTP')
        assert result == 'iraRef_mp3_heating_private_npv_lessWTP'
    
    def test_remdb_v3_more_wtp_fixed_low(self):
        """Test REMDB v3 moreWTP with fixed_low suffix."""
        result = create_npv_col('preIRA_mp8_', 'waterHeating', 'moreWTP', method_suffix='_fixed_low')
        assert result == 'preIRA_mp8_waterHeating_private_npv_moreWTP_fixed_low'
    
    def test_remdb_v3_fixed_high(self):
        """Test REMDB v3 with fixed_high suffix."""
        result = create_npv_col('iraRef_mp9_', 'heating', 'lessWTP', method_suffix='_fixed_high')
        assert result == 'iraRef_mp9_heating_private_npv_lessWTP_fixed_high'
    
    def test_remdb_v4_mid_percentile(self):
        """Test REMDB v4 with mid percentile."""
        result = create_npv_col('iraRef_mp3_', 'heating', 'lessWTP', percentile='mid')
        assert result == 'iraRef_mp3_heating_private_npv_lessWTP_mid'
    
    def test_remdb_v4_low_percentile(self):
        """Test REMDB v4 with low percentile."""
        result = create_npv_col('iraRef_mp4_', 'cooling', 'moreWTP', percentile='low')
        assert result == 'iraRef_mp4_cooling_private_npv_moreWTP_low'
    
    def test_remdb_v4_high_percentile(self):
        """Test REMDB v4 with high percentile."""
        result = create_npv_col('preIRA_mp10_', 'heating', 'lessWTP', percentile='high')
        assert result == 'preIRA_mp10_heating_private_npv_lessWTP_high'


# =============================================================================
# TESTS: ENCLOSURE & WEATHERIZATION COLUMNS
# =============================================================================

class TestEnclosureCostCol:
    """Test create_enclosure_cost_col() function."""
    
    def test_remdb_v3_mp9(self):
        """Test REMDB v3 MP9 enclosure cost."""
        result = create_enclosure_cost_col(9)
        assert result == 'mp9_enclosure_upgradeCost'
    
    def test_remdb_v3_mp10(self):
        """Test REMDB v3 MP10 enclosure cost."""
        result = create_enclosure_cost_col(10)
        assert result == 'mp10_enclosure_upgradeCost'
    
    def test_remdb_v4_mid_mp9(self):
        """Test REMDB v4 mid percentile MP9."""
        result = create_enclosure_cost_col(9, cost_scenario='remdb_v4_mid')
        assert result == 'mp9_enclosure_upgrade_installed_cost_mid'
    
    def test_remdb_v4_low_mp10(self):
        """Test REMDB v4 low percentile MP10."""
        result = create_enclosure_cost_col(10, cost_scenario='remdb_v4_low')
        assert result == 'mp10_enclosure_upgrade_installed_cost_low'
    
    def test_remdb_v4_high_mp9(self):
        """Test REMDB v4 high percentile MP9."""
        result = create_enclosure_cost_col(9, cost_scenario='remdb_v4_high')
        assert result == 'mp9_enclosure_upgrade_installed_cost_high'


class TestWeatherizationRebateCol:
    """Test create_weatherization_rebate_col() function."""
    
    def test_remdb_v3(self):
        """Test REMDB v3 weatherization rebate."""
        result = create_weatherization_rebate_col()
        assert result == 'weatherization_rebate_amount'
    
    def test_remdb_v4_low(self):
        """Test REMDB v4 low percentile."""
        result = create_weatherization_rebate_col(cost_scenario='remdb_v4_low')
        assert result == 'weatherization_rebate_amount_low'
    
    def test_remdb_v4_mid(self):
        """Test REMDB v4 mid percentile."""
        result = create_weatherization_rebate_col(cost_scenario='remdb_v4_mid')
        assert result == 'weatherization_rebate_amount_mid'
    
    def test_remdb_v4_high(self):
        """Test REMDB v4 high percentile."""
        result = create_weatherization_rebate_col(cost_scenario='remdb_v4_high')
        assert result == 'weatherization_rebate_amount_high'


class TestInstallationPremiumCol:
    """Test create_installation_premium_col() function."""
    
    def test_heating(self):
        """Test heating installation premium."""
        result = create_installation_premium_col(3, 'heating')
        assert result == 'mp3_heating_installation_premium'
    
    def test_different_mps(self):
        """Test different measure packages."""
        for mp in [3, 4, 7, 8, 9, 10]:
            result = create_installation_premium_col(mp, 'heating')
            assert result == f'mp{mp}_heating_installation_premium'
    
    def test_other_categories(self):
        """Test other equipment categories."""
        result = create_installation_premium_col(5, 'waterHeating')
        assert result == 'mp5_waterHeating_installation_premium'


class TestCombinedHeatingCoolingCol:
    """Test create_combined_heating_cooling_col() function."""
    
    def test_remdb_v3_replacement(self):
        """Test REMDB v3 combined replacement cost."""
        result = create_combined_heating_cooling_col(3, 'replacement_installed_cost')
        assert result == 'mp3_heating_and_cooling_replacement_installed_cost'
    
    def test_remdb_v3_net_capital(self):
        """Test REMDB v3 combined net capital cost."""
        result = create_combined_heating_cooling_col(4, 'net_capital_cost')
        assert result == 'mp4_heating_and_cooling_net_capital_cost'
    
    def test_remdb_v4_mid(self):
        """Test REMDB v4 mid percentile."""
        result = create_combined_heating_cooling_col(7, 'replacement_installed_cost', 
                                              cost_scenario='remdb_v4_mid')
        assert result == 'mp7_heating_and_cooling_replacement_installed_cost_mid'
    
    def test_remdb_v4_high(self):
        """Test REMDB v4 high percentile."""
        result = create_combined_heating_cooling_col(3, 'net_capital_cost',
                                              cost_scenario='remdb_v4_high')
        assert result == 'mp3_heating_and_cooling_net_capital_cost_high'


# =============================================================================
# TESTS: PUBLIC IMPACT NPV COLUMNS
# =============================================================================

class TestClimateNpvCol:
    """Test create_climate_npv_col() function."""
    
    def test_lower_scc(self):
        """Test lower SCC assumption."""
        result = create_climate_npv_col('iraRef_mp3_', 'heating', 'lower')
        assert result == 'iraRef_mp3_heating_climate_npv_lower'
    
    def test_central_scc(self):
        """Test central SCC assumption."""
        result = create_climate_npv_col('preIRA_mp8_', 'waterHeating', 'central')
        assert result == 'preIRA_mp8_waterHeating_climate_npv_central'
    
    def test_upper_scc(self):
        """Test upper SCC assumption."""
        result = create_climate_npv_col('baseline_', 'heating', 'upper')
        assert result == 'baseline_heating_climate_npv_upper'


class TestHealthNpvCol:
    """Test create_health_npv_col() function."""
    
    def test_inmap_acs(self):
        """Test InMAP model with ACS CR function."""
        result = create_health_npv_col('iraRef_mp3_', 'heating', 'inmap', 'acs')
        assert result == 'iraRef_mp3_heating_health_npv_inmap_acs'
    
    def test_ap2_h6c(self):
        """Test AP2 model with H6C CR function."""
        result = create_health_npv_col('preIRA_mp9_', 'waterHeating', 'ap2', 'h6c')
        assert result == 'preIRA_mp9_waterHeating_health_npv_ap2_h6c'
    
    def test_easiur_acs(self):
        """Test EASIUR model with ACS CR function."""
        result = create_health_npv_col('baseline_', 'heating', 'easiur', 'acs')
        assert result == 'baseline_heating_health_npv_easiur_acs'


class TestPublicNpvCol:
    """Test create_public_npv_col() function."""
    
    def test_central_inmap_acs(self):
        """Test central SCC with InMAP ACS."""
        result = create_public_npv_col('iraRef_mp3_', 'heating', 'central', 'inmap', 'acs')
        assert result == 'iraRef_mp3_heating_public_npv_central_inmap_acs'
    
    def test_lower_ap2_h6c(self):
        """Test lower SCC with AP2 H6C."""
        result = create_public_npv_col('preIRA_mp8_', 'waterHeating', 'lower', 'ap2', 'h6c')
        assert result == 'preIRA_mp8_waterHeating_public_npv_lower_ap2_h6c'
    
    def test_upper_easiur_acs(self):
        """Test upper SCC with EASIUR ACS."""
        result = create_public_npv_col('baseline_', 'heating', 'upper', 'easiur', 'acs')
        assert result == 'baseline_heating_public_npv_upper_easiur_acs'


class TestLifetimeDamagesCol:
    """Test create_lifetime_damages_col() function."""
    
    def test_climate_lrmer_central(self):
        """Test climate damages with LRMER central SCC."""
        result = create_lifetime_damages_col('baseline_', 'heating', 'climate', 'lrmer', 'central')
        assert result == 'baseline_heating_lifetime_damages_climate_lrmer_central'
    
    def test_climate_srmer_upper(self):
        """Test climate damages with SRMER upper SCC."""
        result = create_lifetime_damages_col('iraRef_mp3_', 'waterHeating', 'climate', 'srmer', 'upper')
        assert result == 'iraRef_mp3_waterHeating_lifetime_damages_climate_srmer_upper'
    
    def test_health_inmap_acs(self):
        """Test health damages with InMAP ACS."""
        result = create_lifetime_damages_col('preIRA_mp8_', 'heating', 'health', 'inmap', 'acs')
        assert result == 'preIRA_mp8_heating_lifetime_damages_health_inmap_acs'
    
    def test_health_ap2_h6c(self):
        """Test health damages with AP2 H6C."""
        result = create_lifetime_damages_col('baseline_', 'waterHeating', 'health', 'ap2', 'h6c')
        assert result == 'baseline_waterHeating_lifetime_damages_health_ap2_h6c'


class TestAvoidedDamagesCol:
    """Test create_avoided_damages_col() function."""
    
    def test_climate_lrmer_central(self):
        """Test avoided climate damages with LRMER central SCC."""
        result = create_avoided_damages_col('iraRef_mp3_', 'heating', 'climate', 'lrmer', 'central')
        assert result == 'iraRef_mp3_heating_avoided_damages_climate_lrmer_central'
    
    def test_climate_srmer_lower(self):
        """Test avoided climate damages with SRMER lower SCC."""
        result = create_avoided_damages_col('preIRA_mp9_', 'waterHeating', 'climate', 'srmer', 'lower')
        assert result == 'preIRA_mp9_waterHeating_avoided_damages_climate_srmer_lower'
    
    def test_health_inmap_acs(self):
        """Test avoided health damages with InMAP ACS."""
        result = create_avoided_damages_col('iraRef_mp4_', 'heating', 'health', 'inmap', 'acs')
        assert result == 'iraRef_mp4_heating_avoided_damages_health_inmap_acs'
    
    def test_health_easiur_h6c(self):
        """Test avoided health damages with EASIUR H6C."""
        result = create_avoided_damages_col('iraRef_mp10_', 'waterHeating', 'health', 'easiur', 'h6c')
        assert result == 'iraRef_mp10_waterHeating_avoided_damages_health_easiur_h6c'


# =============================================================================
# TESTS: ADOPTION POTENTIAL COLUMNS
# =============================================================================

class TestAdoptionCol:
    """Test create_adoption_col() function."""
    
    def test_health_sensitivity(self):
        """Test health sensitivity column."""
        result = create_adoption_col('iraRef_mp3_', 'heating', 'health_sensitivity')
        assert result == 'iraRef_mp3_heating_health_sensitivity'
    
    def test_benefit(self):
        """Test benefit column."""
        result = create_adoption_col('iraRef_mp3_', 'heating', 'benefit', 
                            'central', 'inmap', 'acs')
        assert result == 'iraRef_mp3_heating_benefit_central_inmap_acs'
    
    def test_adoption_no_suffix(self):
        """Test adoption column without method suffix."""
        result = create_adoption_col('preIRA_mp8_', 'waterHeating', 'adoption',
                            'upper', 'ap2', 'h6c')
        assert result == 'preIRA_mp8_waterHeating_adoption_upper_ap2_h6c'
    
    def test_adoption_with_suffix(self):
        """Test adoption column with method suffix."""
        result = create_adoption_col('iraRef_mp3_', 'heating', 'adoption',
                            'central', 'inmap', 'acs', method_suffix='_fixed_low')
        assert result == 'iraRef_mp3_heating_adoption_central_inmap_acs_fixed_low'
    
    def test_impact(self):
        """Test impact column."""
        result = create_adoption_col('iraRef_mp9_', 'heating', 'impact',
                            'lower', 'easiur', 'acs')
        assert result == 'iraRef_mp9_heating_impact_lower_easiur_acs'
    
    def test_invalid_column_type(self):
        """Test invalid column type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_adoption_col('iraRef_mp3_', 'heating', 'invalid_type')
        assert 'Invalid column_type' in str(exc_info.value)


class TestTotalNpvCol:
    """Test create_total_npv_col() function."""
    
    def test_combined_no_suffix(self):
        """Test combined total NPV without method suffix."""
        result = create_total_npv_col('iraRef_mp3_', 'heating', 'central', 'inmap', 'acs')
        assert result == 'iraRef_mp3_heating_total_npv_central_inmap_acs'
    
    def test_combined_with_suffix(self):
        """Test combined total NPV with method suffix."""
        result = create_total_npv_col('preIRA_mp8_', 'waterHeating', 'upper', 'ap2', 'h6c',
                              method_suffix='_fixed_low')
        assert result == 'preIRA_mp8_waterHeating_total_npv_upper_ap2_h6c_fixed_low'
    
    def test_climate_only_central(self):
        """Test climate-only total NPV with central assumption."""
        result = create_total_npv_col('iraRef_mp3_', 'heating', 'central',
                              climate_only=True, discount_rate='fixed_3pct')
        assert result == 'iraRef_mp3_heating_total_npv_climateOnly_central_fixed_3pct'
    
    def test_climate_only_lower(self):
        """Test climate-only total NPV with lower assumption."""
        result = create_total_npv_col('baseline_', 'waterHeating', 'lower',
                              climate_only=True, discount_rate='fixed_2pct')
        assert result == 'baseline_waterHeating_total_npv_climateOnly_lower_fixed_2pct'


# =============================================================================
# PARAMETRIC TESTS FOR COMPREHENSIVE COVERAGE
# =============================================================================

class TestParametricCombinations:
    """Parametric tests covering many valid parameter combinations."""
    
    @pytest.mark.parametrize("menu_mp", [0, 3, 4, 7, 8, 9, 10])
    @pytest.mark.parametrize("category", ['heating', 'waterHeating', 'clothesDrying', 'cooking', 'cooling'])
    @pytest.mark.parametrize("cost_scenario", ['remdb_v3', 'remdb_v4_low', 'remdb_v4_mid', 'remdb_v4_high'])
    def test_cost_col_all_combinations(self, menu_mp, category, cost_scenario):
        """Test cost_col with all valid combinations."""
        result = create_cost_col(menu_mp, category, 'upgrade', cost_scenario)
        assert f'mp{menu_mp}' in result
        assert category in result
        assert 'upgrade_installed_cost' in result
    
    @pytest.mark.parametrize("scc", ['lower', 'central', 'upper'])
    @pytest.mark.parametrize("rcm", ['ap2', 'easiur', 'inmap'])
    @pytest.mark.parametrize("cr", ['acs', 'h6c'])
    def test_public_npv_all_combinations(self, scc, rcm, cr):
        """Test public_npv_col with all valid sensitivity combinations."""
        result = create_public_npv_col('iraRef_mp3_', 'heating', scc, rcm, cr)
        assert scc in result
        assert rcm in result
        assert cr in result
        assert 'public_npv' in result
    
    @pytest.mark.parametrize("wtp", ['lessWTP', 'moreWTP'])
    @pytest.mark.parametrize("percentile", [None, 'low', 'mid', 'high'])
    def test_npv_col_wtp_percentile_combinations(self, wtp, percentile):
        """Test npv_col with all WTP and percentile combinations."""
        result = create_npv_col('iraRef_mp3_', 'heating', wtp, percentile=percentile)
        assert wtp in result
        assert 'private_npv' in result
        if percentile:
            assert percentile in result


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests to verify real-world usage patterns."""
    
    def test_v3_complete_workflow(self):
        """Test complete v3 workflow for a single MP."""
        mp = 3
        category = 'heating'
        scenario = 'iraRef_mp3_'
        
        # Cost columns
        upgrade_cost = create_cost_col(mp, category, 'upgrade')
        replacement_cost = create_cost_col(mp, category, 'replacement')
        rebate = create_rebate_col(mp, category)
        
        # Capital columns
        total_capital = create_capital_col(scenario, category, net=False)
        net_capital = create_capital_col(scenario, category, net=True)
        
        # NPV columns
        private_npv_less = create_npv_col(scenario, category, 'lessWTP')
        private_npv_more = create_npv_col(scenario, category, 'moreWTP')
        
        # Public columns
        climate_npv = create_climate_npv_col(scenario, category, 'central')
        health_npv = create_health_npv_col(scenario, category, 'inmap', 'acs')
        public_npv = create_public_npv_col(scenario, category, 'central', 'inmap', 'acs')
        
        # All should be valid strings without percentile suffixes
        for col in [upgrade_cost, replacement_cost, rebate, total_capital, 
                   net_capital, private_npv_less, private_npv_more,
                   climate_npv, health_npv, public_npv]:
            assert isinstance(col, str)
            assert len(col) > 0
            assert '_low' not in col
            assert '_mid' not in col
            assert '_high' not in col
    
    def test_v4_complete_workflow(self):
        """Test complete v4 workflow for a single MP."""
        mp = 3
        category = 'heating'
        scenario = 'iraRef_mp3_'
        cost_scenario = 'remdb_v4_mid'
        
        # Cost columns
        upgrade_cost = create_cost_col(mp, category, 'upgrade', cost_scenario)
        replacement_cost = create_cost_col(mp, category, 'replacement', cost_scenario)
        rebate = create_rebate_col(mp, category, cost_scenario)
        
        # Capital columns
        total_capital = create_capital_col(scenario, category, net=False, cost_scenario=cost_scenario)
        net_capital = create_capital_col(scenario, category, net=True, cost_scenario=cost_scenario)
        
        # NPV columns (v4 uses percentile instead of method_suffix)
        private_npv_less = create_npv_col(scenario, category, 'lessWTP', percentile='mid')
        private_npv_more = create_npv_col(scenario, category, 'moreWTP', percentile='mid')
        
        # All cost-related columns should have '_mid' suffix
        for col in [upgrade_cost, replacement_cost, rebate, total_capital, 
                   net_capital, private_npv_less, private_npv_more]:
            assert isinstance(col, str)
            assert '_mid' in col
    
    def test_mp9_mp10_special_cases(self):
        """Test special cases for MP9 and MP10 with enclosure costs."""
        for mp in [9, 10]:
            # Enclosure costs
            enclosure_v3 = create_enclosure_cost_col(mp)
            enclosure_v4 = create_enclosure_cost_col(mp, 'remdb_v4_mid')
            
            assert 'enclosure' in enclosure_v3
            assert 'enclosure' in enclosure_v4
            assert f'mp{mp}' in enclosure_v3
            assert f'mp{mp}' in enclosure_v4
            assert '_mid' not in enclosure_v3
            assert '_mid' in enclosure_v4
            
            # Weatherization rebates
            weather_rebate_v3 = create_weatherization_rebate_col()
            weather_rebate_v4 = create_weatherization_rebate_col('remdb_v4_mid')
            
            assert 'weatherization' in weather_rebate_v3
            assert 'weatherization' in weather_rebate_v4
            assert '_mid' not in weather_rebate_v3
            assert '_mid' in weather_rebate_v4
    
    def test_combined_heating_cooling_mps(self):
        """Test MPs with combined heating and cooling upgrades (3, 4, 7)."""
        for mp in [3, 4, 7]:
            combined_v3 = create_combined_heating_cooling_col(mp, 'replacement_installed_cost')
            combined_v4 = create_combined_heating_cooling_col(mp, 'replacement_installed_cost', 
                                                       'remdb_v4_mid')
            
            assert 'heating_and_cooling' in combined_v3
            assert 'heating_and_cooling' in combined_v4
            assert f'mp{mp}' in combined_v3
            assert '_mid' not in combined_v3
            assert '_mid' in combined_v4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
