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
    
    def test_v3_upgrade(self):
        """Test v3 upgrade cost column."""
        result = create_cost_col(menu_mp=3, category='heating', cost_type='upgrade', cost_scenario='v3')
        assert result == 'mp3_heating_upgrade_installed_cost_v3'
    
    def test_v3_replacement(self):
        """Test v3 replacement cost column."""
        result = create_cost_col(menu_mp=8, category='waterHeating', cost_type='replacement', cost_scenario='v3')
        assert result == 'mp8_waterHeating_replacement_installed_cost_v3'
    
    def test_v4LOW(self):
        """Test v4LOW cost scenario."""
        result = create_cost_col(menu_mp=3, category='heating', cost_type='upgrade', cost_scenario='v4LOW')
        assert result == 'mp3_heating_upgrade_installed_cost_v4LOW'
    
    def test_v4MID(self):
        """Test v4MID cost scenario."""
        result = create_cost_col(menu_mp=4, category='cooling', cost_type='replacement', cost_scenario='v4MID')
        assert result == 'mp4_cooling_replacement_installed_cost_v4MID'
    
    def test_v4HIGH(self):
        """Test v4HIGH cost scenario."""
        result = create_cost_col(menu_mp=7, category='clothesDrying', cost_type='upgrade', cost_scenario='v4HIGH')
        assert result == 'mp7_clothesDrying_upgrade_installed_cost_v4HIGH'
    
    def test_all_categories(self):
        """Test all equipment categories."""
        categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking', 'cooling']
        for cat in categories:
            result = create_cost_col(menu_mp=3, category=cat, cost_type='upgrade', cost_scenario='v3')
            assert cat in result


class TestRebateCol:
    """Test create_rebate_col() function."""
    
    def test_v3(self):
        """Test v3 rebate column."""
        result = create_rebate_col(menu_mp=3, category='heating', cost_scenario='v3')
        assert result == 'mp3_heating_rebate_amount_v3'
    
    def test_v4LOW(self):
        """Test v4LOW rebate."""
        result = create_rebate_col(menu_mp=8, category='waterHeating', cost_scenario='v4LOW')
        assert result == 'mp8_waterHeating_rebate_amount_v4LOW'
    
    def test_v4MID(self):
        """Test v4MID rebate."""
        result = create_rebate_col(menu_mp=9, category='heating', cost_scenario='v4MID')
        assert result == 'mp9_heating_rebate_amount_v4MID'
    
    def test_v4HIGH(self):
        """Test v4HIGH rebate."""
        result = create_rebate_col(menu_mp=10, category='cooking', cost_scenario='v4HIGH')
        assert result == 'mp10_cooking_rebate_amount_v4HIGH'


class TestCapitalCol:
    """Test create_capital_col() function."""
    
    def test_v3_total(self):
        """Test v3 total capital cost."""
        result = create_capital_col(scenario_prefix='iraRef_mp3_', category='heating', net=False, cost_scenario='v3')
        assert result == 'iraRef_mp3_heating_total_capital_cost_v3'
    
    def test_v3_net(self):
        """Test v3 net capital cost."""
        result = create_capital_col(scenario_prefix='preIRA_mp8_', category='waterHeating', net=True, cost_scenario='v3')
        assert result == 'preIRA_mp8_waterHeating_net_capital_cost_v3'
    
    def test_v4MID_total(self):
        """Test v4MID total capital cost."""
        result = create_capital_col(scenario_prefix='iraRef_mp3_', category='heating', net=False, cost_scenario='v4MID')
        assert result == 'iraRef_mp3_heating_total_capital_cost_v4MID'
    
    def test_v4HIGH_net(self):
        """Test v4HIGH net capital cost."""
        result = create_capital_col(scenario_prefix='baseline_', category='cooling', net=True, cost_scenario='v4HIGH')
        assert result == 'baseline_cooling_net_capital_cost_v4HIGH'
    
    def test_v4LOW(self):
        """Test v4LOW."""
        result = create_capital_col(scenario_prefix='iraRef_mp9_', category='heating', net=False, cost_scenario='v4LOW')
        assert result == 'iraRef_mp9_heating_total_capital_cost_v4LOW'


class TestNpvCol:
    """Test create_npv_col() function."""
    
    def test_v3_less_wtp_no_suffix(self):
        """Test v3 lessWTP without method suffix."""
        result = create_npv_col(scenario_prefix='iraRef_mp3_', category='heating', wtp='lessWTP', cost_scenario='v3', method_suffix='')
        assert result == 'iraRef_mp3_heating_private_npv_lessWTP_v3'
    
    def test_v3_more_wtp_fixed_low(self):
        """Test v3 moreWTP with fixed_low suffix."""
        result = create_npv_col(scenario_prefix='preIRA_mp8_', category='waterHeating', wtp='moreWTP', cost_scenario='v3', method_suffix='_fixed_low')
        assert result == 'preIRA_mp8_waterHeating_private_npv_moreWTP_v3_fixed_low'
    
    def test_v3_fixed_high(self):
        """Test v3 with fixed_high suffix."""
        result = create_npv_col(scenario_prefix='iraRef_mp9_', category='heating', wtp='lessWTP', cost_scenario='v3', method_suffix='_fixed_high')
        assert result == 'iraRef_mp9_heating_private_npv_lessWTP_v3_fixed_high'
    
    def test_v4MID(self):
        """Test v4MID cost scenario."""
        result = create_npv_col(scenario_prefix='iraRef_mp3_', category='heating', wtp='lessWTP', cost_scenario='v4MID', method_suffix='')
        assert result == 'iraRef_mp3_heating_private_npv_lessWTP_v4MID'
    
    def test_v4LOW(self):
        """Test v4LOW cost scenario."""
        result = create_npv_col(scenario_prefix='iraRef_mp4_', category='cooling', wtp='moreWTP', cost_scenario='v4LOW', method_suffix='')
        assert result == 'iraRef_mp4_cooling_private_npv_moreWTP_v4LOW'
    
    def test_v4HIGH(self):
        """Test v4HIGH cost scenario."""
        result = create_npv_col(scenario_prefix='preIRA_mp10_', category='heating', wtp='lessWTP', cost_scenario='v4HIGH', method_suffix='')
        assert result == 'preIRA_mp10_heating_private_npv_lessWTP_v4HIGH'


# =============================================================================
# TESTS: ENCLOSURE & WEATHERIZATION COLUMNS
# =============================================================================

class TestEnclosureCostCol:
    """Test create_enclosure_cost_col() function."""
    
    def test_v3_mp9(self):
        """Test v3 MP9 enclosure cost."""
        result = create_enclosure_cost_col(menu_mp=9, cost_scenario='v3')
        assert result == 'mp9_enclosure_upgrade_installed_cost_v3'
    
    def test_v3_mp10(self):
        """Test v3 MP10 enclosure cost."""
        result = create_enclosure_cost_col(menu_mp=10, cost_scenario='v3')
        assert result == 'mp10_enclosure_upgrade_installed_cost_v3'
    
    def test_v4MID_mp9(self):
        """Test v4MID MP9."""
        result = create_enclosure_cost_col(menu_mp=9, cost_scenario='v4MID')
        assert result == 'mp9_enclosure_upgrade_installed_cost_v4MID'
    
    def test_v4LOW_mp10(self):
        """Test v4LOW MP10."""
        result = create_enclosure_cost_col(menu_mp=10, cost_scenario='v4LOW')
        assert result == 'mp10_enclosure_upgrade_installed_cost_v4LOW'
    
    def test_v4HIGH_mp9(self):
        """Test v4HIGH MP9."""
        result = create_enclosure_cost_col(menu_mp=9, cost_scenario='v4HIGH')
        assert result == 'mp9_enclosure_upgrade_installed_cost_v4HIGH'


class TestWeatherizationRebateCol:
    """Test create_weatherization_rebate_col() function."""
    
    def test_v3(self):
        """Test v3 weatherization rebate."""
        result = create_weatherization_rebate_col(cost_scenario='v3')
        assert result == 'weatherization_rebate_amount_v3'
    
    def test_v4LOW(self):
        """Test v4LOW."""
        result = create_weatherization_rebate_col(cost_scenario='v4LOW')
        assert result == 'weatherization_rebate_amount_v4LOW'
    
    def test_v4MID(self):
        """Test v4MID."""
        result = create_weatherization_rebate_col(cost_scenario='v4MID')
        assert result == 'weatherization_rebate_amount_v4MID'
    
    def test_v4HIGH(self):
        """Test v4HIGH."""
        result = create_weatherization_rebate_col(cost_scenario='v4HIGH')
        assert result == 'weatherization_rebate_amount_v4HIGH'


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
    
    def test_v3_replacement(self):
        """Test v3 combined replacement cost."""
        result = create_combined_heating_cooling_col(menu_mp=3, cost_type='replacement_installed_cost', cost_scenario='v3')
        assert result == 'mp3_heating_and_cooling_replacement_installed_cost_v3'
    
    def test_v3_net_capital(self):
        """Test v3 combined net capital cost."""
        result = create_combined_heating_cooling_col(menu_mp=4, cost_type='net_capital_cost', cost_scenario='v3')
        assert result == 'mp4_heating_and_cooling_net_capital_cost_v3'
    
    def test_v4MID(self):
        """Test v4MID."""
        result = create_combined_heating_cooling_col(menu_mp=7, cost_type='replacement_installed_cost', 
                                              cost_scenario='v4MID')
        assert result == 'mp7_heating_and_cooling_replacement_installed_cost_v4MID'
    
    def test_v4HIGH(self):
        """Test v4HIGH."""
        result = create_combined_heating_cooling_col(menu_mp=3, cost_type='net_capital_cost',
                                              cost_scenario='v4HIGH')
        assert result == 'mp3_heating_and_cooling_net_capital_cost_v4HIGH'


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
        result = create_adoption_col(scenario_prefix='iraRef_mp3_', category='heating', column_type='health_sensitivity', cost_scenario='v3', method_suffix='_fixed_low')
        assert result == 'iraRef_mp3_heating_health_sensitivity'
    
    def test_benefit(self):
        """Test benefit column."""
        result = create_adoption_col(scenario_prefix='iraRef_mp3_', category='heating', column_type='benefit', cost_scenario='v3', method_suffix='_fixed_low', scc_assumption='central', rcm_model='inmap', cr_function='acs')
        assert result == 'iraRef_mp3_heating_benefit_central_inmap_acs_v3'
    
    def test_adoption_no_suffix(self):
        """Test adoption column without method suffix."""
        result = create_adoption_col(scenario_prefix='preIRA_mp8_', category='waterHeating', column_type='adoption', cost_scenario='v3', method_suffix='', scc_assumption='upper', rcm_model='ap2', cr_function='h6c')
        assert result == 'preIRA_mp8_waterHeating_adoption_upper_ap2_h6c_v3'
    
    def test_adoption_with_suffix(self):
        """Test adoption column with method suffix."""
        result = create_adoption_col(scenario_prefix='iraRef_mp3_', category='heating', column_type='adoption', cost_scenario='v3', method_suffix='_fixed_low', scc_assumption='central', rcm_model='inmap', cr_function='acs')
        assert result == 'iraRef_mp3_heating_adoption_central_inmap_acs_v3_fixed_low'
    
    def test_impact(self):
        """Test impact column."""
        result = create_adoption_col(scenario_prefix='iraRef_mp9_', category='heating', column_type='impact', cost_scenario='v3', method_suffix='_fixed_low', scc_assumption='lower', rcm_model='easiur', cr_function='acs')
        assert result == 'iraRef_mp9_heating_impact_lower_easiur_acs_v3'
    
    def test_adoption_v4MID(self):
        """Test adoption column with v4MID."""
        result = create_adoption_col(scenario_prefix='iraRef_mp3_', category='heating', column_type='adoption', cost_scenario='v4MID', method_suffix='_fixed_low', scc_assumption='central', rcm_model='inmap', cr_function='acs')
        assert result == 'iraRef_mp3_heating_adoption_central_inmap_acs_v4MID_fixed_low'
    
    def test_invalid_column_type(self):
        """Test invalid column type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_adoption_col(scenario_prefix='iraRef_mp3_', category='heating', column_type='invalid_type', cost_scenario='v3', method_suffix='')
        assert 'Invalid column_type' in str(exc_info.value)


class TestTotalNpvCol:
    """Test create_total_npv_col() function."""
    
    def test_combined_no_suffix(self):
        """Test combined total NPV without method suffix."""
        result = create_total_npv_col(scenario_prefix='iraRef_mp3_', category='heating', cost_scenario='v3', method_suffix='', scc_assumption='central', rcm_model='inmap', cr_function='acs')
        assert result == 'iraRef_mp3_heating_total_npv_central_inmap_acs_v3'
    
    def test_combined_with_suffix(self):
        """Test combined total NPV with method suffix."""
        result = create_total_npv_col(scenario_prefix='preIRA_mp8_', category='waterHeating', cost_scenario='v3', method_suffix='_fixed_low', scc_assumption='upper', rcm_model='ap2', cr_function='h6c')
        assert result == 'preIRA_mp8_waterHeating_total_npv_upper_ap2_h6c_v3_fixed_low'
    
    def test_combined_with_v4MID(self):
        """Test combined total NPV with v4MID cost scenario."""
        result = create_total_npv_col(scenario_prefix='iraRef_mp3_', category='heating', cost_scenario='v4MID', method_suffix='_fixed_low', scc_assumption='central', rcm_model='inmap', cr_function='acs')
        assert result == 'iraRef_mp3_heating_total_npv_central_inmap_acs_v4MID_fixed_low'
    
    def test_climate_only_central(self):
        """Test climate-only total NPV with central assumption."""
        result = create_total_npv_col(scenario_prefix='iraRef_mp3_', category='heating', cost_scenario='v3', method_suffix='_fixed_base', scc_assumption='central', climate_only=True)
        assert result == 'iraRef_mp3_heating_total_npv_climateOnly_central_v3_fixed_base'
    
    def test_climate_only_lower(self):
        """Test climate-only total NPV with lower assumption."""
        result = create_total_npv_col(scenario_prefix='baseline_', category='waterHeating', cost_scenario='v3', method_suffix='_fixed_2pct', scc_assumption='lower', climate_only=True)
        assert result == 'baseline_waterHeating_total_npv_climateOnly_lower_v3_fixed_2pct'
    
    def test_health_only(self):
        """Test health-only total NPV."""
        result = create_total_npv_col(scenario_prefix='iraRef_mp3_', category='heating', cost_scenario='v3', method_suffix='_fixed_low', rcm_model='inmap', cr_function='acs', health_only=True)
        assert result == 'iraRef_mp3_heating_total_npv_healthOnly_inmap_acs_v3_fixed_low'
    
    def test_health_only_v4HIGH(self):
        """Test health-only total NPV with v4HIGH."""
        result = create_total_npv_col(scenario_prefix='preIRA_mp8_', category='cooking', cost_scenario='v4HIGH', method_suffix='_fixed_high', rcm_model='ap2', cr_function='h6c', health_only=True)
        assert result == 'preIRA_mp8_cooking_total_npv_healthOnly_ap2_h6c_v4HIGH_fixed_high'


# =============================================================================
# PARAMETRIC TESTS FOR COMPREHENSIVE COVERAGE
# =============================================================================

class TestParametricCombinations:
    """Parametric tests covering many valid parameter combinations."""
    
    @pytest.mark.parametrize("menu_mp", [0, 3, 4, 7, 8, 9, 10])
    @pytest.mark.parametrize("category", ['heating', 'waterHeating', 'clothesDrying', 'cooking', 'cooling'])
    @pytest.mark.parametrize("cost_scenario", ['v3', 'v4LOW', 'v4MID', 'v4HIGH'])
    def test_cost_col_all_combinations(self, menu_mp, category, cost_scenario):
        """Test cost_col with all valid combinations."""
        result = create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)
        assert f'mp{menu_mp}' in result
        assert category in result
        assert 'upgrade_installed_cost' in result
        assert result.endswith(f'_{cost_scenario}')
    
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
    @pytest.mark.parametrize("cost_scenario", ['v3', 'v4LOW', 'v4MID', 'v4HIGH'])
    def test_npv_col_wtp_cost_scenario_combinations(self, wtp, cost_scenario):
        """Test npv_col with all WTP and cost_scenario combinations."""
        result = create_npv_col(scenario_prefix='iraRef_mp3_', category='heating', wtp=wtp, cost_scenario=cost_scenario, method_suffix='')
        assert wtp in result
        assert 'private_npv' in result
        assert result.endswith(f'_{cost_scenario}')


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
        cs = 'v3'
        
        # Cost columns
        upgrade_cost = create_cost_col(menu_mp=mp, category=category, cost_type='upgrade', cost_scenario=cs)
        replacement_cost = create_cost_col(menu_mp=mp, category=category, cost_type='replacement', cost_scenario=cs)
        rebate = create_rebate_col(menu_mp=mp, category=category, cost_scenario=cs)
        
        # Capital columns
        total_capital = create_capital_col(scenario_prefix=scenario, category=category, net=False, cost_scenario=cs)
        net_capital = create_capital_col(scenario_prefix=scenario, category=category, net=True, cost_scenario=cs)
        
        # NPV columns
        private_npv_less = create_npv_col(scenario_prefix=scenario, category=category, wtp='lessWTP', cost_scenario=cs, method_suffix='')
        private_npv_more = create_npv_col(scenario_prefix=scenario, category=category, wtp='moreWTP', cost_scenario=cs, method_suffix='')
        
        # Public columns (no cost_scenario dependency)
        climate_npv = create_climate_npv_col(scenario, category, 'central')
        health_npv = create_health_npv_col(scenario, category, 'inmap', 'acs')
        public_npv = create_public_npv_col(scenario, category, 'central', 'inmap', 'acs')
        
        # All cost-scenario-dependent columns should contain _v3
        for col in [upgrade_cost, replacement_cost, rebate, total_capital, net_capital, private_npv_less, private_npv_more]:
            assert '_v3' in col
    
    def test_v4_complete_workflow(self):
        """Test complete v4 workflow for a single MP."""
        mp = 3
        category = 'heating'
        scenario = 'iraRef_mp3_'
        cs = 'v4MID'
        
        # Cost columns
        upgrade_cost = create_cost_col(menu_mp=mp, category=category, cost_type='upgrade', cost_scenario=cs)
        replacement_cost = create_cost_col(menu_mp=mp, category=category, cost_type='replacement', cost_scenario=cs)
        rebate = create_rebate_col(menu_mp=mp, category=category, cost_scenario=cs)
        
        # Capital columns
        total_capital = create_capital_col(scenario_prefix=scenario, category=category, net=False, cost_scenario=cs)
        net_capital = create_capital_col(scenario_prefix=scenario, category=category, net=True, cost_scenario=cs)
        
        # NPV columns
        private_npv_less = create_npv_col(scenario_prefix=scenario, category=category, wtp='lessWTP', cost_scenario=cs, method_suffix='')
        private_npv_more = create_npv_col(scenario_prefix=scenario, category=category, wtp='moreWTP', cost_scenario=cs, method_suffix='')
        
        # All cost-scenario-dependent columns should contain _v4MID
        for col in [upgrade_cost, replacement_cost, rebate, total_capital, net_capital, private_npv_less, private_npv_more]:
            assert '_v4MID' in col
    
    def test_mp9_mp10_special_cases(self):
        """Test special cases for MP9 and MP10 with enclosure costs."""
        for mp in [9, 10]:
            # Enclosure costs
            enclosure_v3 = create_enclosure_cost_col(menu_mp=mp, cost_scenario='v3')
            enclosure_v4 = create_enclosure_cost_col(menu_mp=mp, cost_scenario='v4MID')
            
            assert 'enclosure' in enclosure_v3
            assert 'enclosure' in enclosure_v4
            assert f'mp{mp}' in enclosure_v3
            assert f'mp{mp}' in enclosure_v4
            assert '_v3' in enclosure_v3
            assert '_v4MID' in enclosure_v4
            
            # Weatherization rebates
            weather_rebate_v3 = create_weatherization_rebate_col(cost_scenario='v3')
            weather_rebate_v4 = create_weatherization_rebate_col(cost_scenario='v4MID')
            
            assert 'weatherization' in weather_rebate_v3
            assert 'weatherization' in weather_rebate_v4
            assert '_v3' in weather_rebate_v3
            assert '_v4MID' in weather_rebate_v4
    
    def test_combined_heating_cooling_mps(self):
        """Test MPs with combined heating and cooling upgrades (3, 4, 7)."""
        for mp in [3, 4, 7]:
            combined_v3 = create_combined_heating_cooling_col(menu_mp=mp, cost_type='replacement_installed_cost', cost_scenario='v3')
            combined_v4 = create_combined_heating_cooling_col(menu_mp=mp, cost_type='replacement_installed_cost', 
                                                       cost_scenario='v4MID')
            
            assert 'heating_and_cooling' in combined_v3
            assert 'heating_and_cooling' in combined_v4
            assert f'mp{mp}' in combined_v3
            assert '_v3' in combined_v3
            assert '_v4MID' in combined_v4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
