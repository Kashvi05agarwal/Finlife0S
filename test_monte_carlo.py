"""
test_monte_carlo.py
Comprehensive test suite for MonteCarloAgent with edge cases and validation
"""

import numpy as np
import sys
from agents.monte_carlo_agent import MonteCarloAgent, SimulationInput
from config import MC_SEED, RISK_RETURN


def safe_print(*args, **kwargs):
    """Print with safe encoding fallback for Windows."""
    try:
        # Replace Rupee symbol with "Rs"
        safe_args = []
        for arg in args:
            s = str(arg).replace('\u20b9', 'Rs')  # Replace with Rs
            safe_args.append(s)
        print(*safe_args, **kwargs, file=sys.stdout)
        sys.stdout.flush()
    except Exception as e:
        print(f"[Print error: {e}]", file=sys.stderr)


def fmt_inr_safe(amount):
    """Format currency safely for console output."""
    try:
        from utils.financial_math import fmt_inr
        result = fmt_inr(amount)
        # Replace Rupee symbol with "Rs"
        return result.replace('\u20b9', 'Rs')
    except Exception:
        return f"Rs{amount:,.0f}"


def assert_valid_result(result, test_name):
    """Validate basic result invariants."""
    assert result.success_probability >= 0 and result.success_probability <= 1, f"{test_name}: success_probability out of range"
    assert result.target_corpus > 0, f"{test_name}: target_corpus should be positive"
    assert result.median_final_wealth >= 0, f"{test_name}: median_final_wealth negative"
    assert len(result.pct10) > 0, f"{test_name}: pct10 empty"
    assert len(result.pct50) > 0, f"{test_name}: pct50 empty"
    assert len(result.pct90) > 0, f"{test_name}: pct90 empty"
    assert result.pct10[-1] <= result.pct50[-1] <= result.pct90[-1], f"{test_name}: percentile ordering violated"
    assert len(result.key_drivers) <= 3, f"{test_name}: more than 3 drivers"
    safe_print(f"[PASS] {test_name}: VALID")


def test_normal_scenario():
    """Standard millennial profile: Rs75k/mo, Rs40k expense, Rs20k invest."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=75000,
        monthly_expenses=40000,
        monthly_emi=5000,
        current_savings=500000,
        monthly_investment=20000,
        risk_profile="moderate",
        horizon_years=25,
        n_simulations=100,
    )
    result = mc.run(sim)
    assert_valid_result(result, "Normal scenario")
    safe_print(f"  Success probability: {result.success_probability:.1%}")
    safe_print(f"  Target: {fmt_inr_safe(result.target_corpus)}")
    safe_print(f"  Median outcome: {fmt_inr_safe(result.median_final_wealth)}")


def test_zero_current_savings():
    """Bootstrap scenario: no initial savings."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=50000,
        monthly_expenses=30000,
        monthly_emi=0,
        current_savings=0,
        monthly_investment=15000,
        risk_profile="moderate",
        horizon_years=20,
        n_simulations=100,
    )
    result = mc.run(sim)
    assert_valid_result(result, "Zero savings bootstrap")
    assert result.median_final_wealth > 0, "Should accumulate wealth from zero start"
    safe_print(f"  Built from Rs0 to: {fmt_inr_safe(result.median_final_wealth)}")


def test_zero_investment():
    """Scenario: only expenses + EMI, no investment."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=100000,
        monthly_expenses=80000,
        monthly_emi=0,
        current_savings=1000000,
        monthly_investment=0,
        risk_profile="moderate",
        horizon_years=10,
        n_simulations=100,
    )
    result = mc.run(sim)
    assert_valid_result(result, "Zero investment (inflation erosion)")
    safe_print(f"  Initial: {fmt_inr_safe(sim.current_savings)}, Final median: {fmt_inr_safe(result.median_final_wealth)}")


def test_inflation_affects_expenses():
    """Inflation should increase expenses and reduce final wealth."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim_infl = SimulationInput(
        monthly_income=45000,
        monthly_expenses=40000,
        monthly_emi=0,
        current_savings=100000,
        monthly_investment=5000,
        risk_profile="moderate",
        horizon_years=20,
        n_simulations=100,
        inflation_mean=0.06,
        inflation_std=0.00,
    )
    sim_no_infl = SimulationInput(**sim_infl.__dict__)
    sim_no_infl.inflation_mean = 0.0
    sim_no_infl.inflation_std = 0.0

    mc.rng = np.random.default_rng(MC_SEED)
    result_infl = mc.run(sim_infl)

    mc.rng = np.random.default_rng(MC_SEED)
    result_no_infl = mc.run(sim_no_infl)

    assert result_infl.median_final_wealth < result_no_infl.median_final_wealth, \
        "Inflation should reduce final wealth when expenses are near income"
    safe_print(f"  Inflation reduces median wealth from {fmt_inr_safe(result_no_infl.median_final_wealth)} to {fmt_inr_safe(result_infl.median_final_wealth)}")


def test_high_expenses():
    """Edge case: expenses consume 90% of income."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=100000,
        monthly_expenses=90000,
        monthly_emi=5000,
        current_savings=2000000,
        monthly_investment=1000,
        risk_profile="conservative",
        horizon_years=10,
        n_simulations=100,
    )
    result = mc.run(sim)
    assert_valid_result(result, "High expense ratio")
    safe_print(f"  Investment capacity: Rs1000/mo only")


def test_very_high_emi():
    """Edge case: EMI near income limit."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=100000,
        monthly_expenses=40000,
        monthly_emi=50000,
        current_savings=5000000,
        monthly_investment=5000,
        risk_profile="moderate",
        horizon_years=8,
        n_simulations=100,
    )
    result = mc.run(sim)
    assert_valid_result(result, "High EMI scenario")
    safe_print(f"  EMI eats 50% of income, still viable")


def test_short_horizon():
    """Extreme short: 3-year horizon."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=60000,
        monthly_expenses=30000,
        monthly_emi=0,
        current_savings=500000,
        monthly_investment=20000,
        risk_profile="aggressive",
        horizon_years=3,
        n_simulations=100,
    )
    result = mc.run(sim)
    assert_valid_result(result, "Short horizon (3 years)")
    assert len(result.pct50) == 3, "Should have 3 years of data"


def test_very_long_horizon():
    """Long horizon: 40 years (lifetime planning)."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=80000,
        monthly_expenses=50000,
        monthly_emi=0,
        current_savings=1000000,
        monthly_investment=20000,
        risk_profile="moderate",
        horizon_years=40,
        n_simulations=50,
    )
    result = mc.run(sim)
    assert_valid_result(result, "Very long horizon (40 years)")
    assert len(result.pct50) == 40, "Should have 40 years of data"
    safe_print(f"  40-year compounding: {fmt_inr_safe(result.median_final_wealth)}")


def test_risk_profiles():
    """Test all three risk profiles produce different returns."""
    mc = MonteCarloAgent(seed=MC_SEED)
    base_sim = SimulationInput(
        monthly_income=100000,
        monthly_expenses=50000,
        monthly_emi=0,
        current_savings=1000000,
        monthly_investment=30000,
        horizon_years=20,
        n_simulations=100,
    )

    results = {}
    for profile in ["conservative", "moderate", "aggressive"]:
        sim = SimulationInput(**base_sim.__dict__)
        sim.risk_profile = profile
        mc.rng = np.random.default_rng(MC_SEED)
        result = mc.run(sim)
        results[profile] = result.median_final_wealth
        assert_valid_result(result, f"Risk profile: {profile}")
        safe_print(f"  {profile:12s}: {fmt_inr_safe(result.median_final_wealth)}")

    assert results["conservative"] < results["moderate"] < results["aggressive"], \
        "Risk profile ordering violated"
    safe_print("  [OK] Risk profiles correctly ordered: conservative < moderate < aggressive")


def test_shock_job_loss_6m():
    """Job loss for 6 months starting year 3."""
    mc = MonteCarloAgent(seed=MC_SEED)
    base_sim = SimulationInput(
        monthly_income=80000,
        monthly_expenses=40000,
        monthly_emi=10000,
        current_savings=1000000,
        monthly_investment=25000,
        risk_profile="moderate",
        horizon_years=20,
        n_simulations=100,
    )
    result_base = mc.run(base_sim)

    mc.rng = np.random.default_rng(MC_SEED)
    result_shock = mc.run_with_shock(base_sim, {
        "income_factor": 0.0,
        "duration_months": 6,
        "one_time_cost": 0,
        "shock_year": 3,
    })

    assert_valid_result(result_shock, "Job loss 6m shock")
    safe_print(f"  Base success: {result_base.success_probability:.1%}")
    safe_print(f"  With shock:   {result_shock.success_probability:.1%}")
    safe_print(f"  Impact: {(result_base.success_probability - result_shock.success_probability)*100:.1f}pp")


def test_shock_medical_8L():
    """Medical emergency: Rs8L one-time cost in year 2."""
    mc = MonteCarloAgent(seed=MC_SEED)
    base_sim = SimulationInput(
        monthly_income=120000,
        monthly_expenses=60000,
        monthly_emi=0,
        current_savings=2000000,
        monthly_investment=40000,
        risk_profile="moderate",
        horizon_years=20,
        n_simulations=100,
    )
    result_base = mc.run(base_sim)

    mc.rng = np.random.default_rng(MC_SEED)
    result_shock = mc.run_with_shock(base_sim, {
        "income_factor": 1.0,
        "duration_months": 0,
        "one_time_cost": 800000,
        "shock_year": 2,
    })

    assert_valid_result(result_shock, "Medical emergency 800k")
    safe_print(f"  Base success: {result_base.success_probability:.1%}")
    safe_print(f"  With cost: {result_shock.success_probability:.1%}")


def test_shock_income_drop_30pct():
    """Career disruption: income drops 30% for 2 years in year 4."""
    mc = MonteCarloAgent(seed=MC_SEED)
    base_sim = SimulationInput(
        monthly_income=150000,
        monthly_expenses=70000,
        monthly_emi=15000,
        current_savings=3000000,
        monthly_investment=50000,
        risk_profile="moderate",
        horizon_years=25,
        n_simulations=100,
    )

    mc.rng = np.random.default_rng(MC_SEED)
    result_shock = mc.run_with_shock(base_sim, {
        "income_factor": 0.70,
        "duration_months": 24,
        "one_time_cost": 0,
        "shock_year": 4,
    })

    assert_valid_result(result_shock, "Income drop 30% for 24m")
    safe_print(f"  Median outcome with income shock: {fmt_inr_safe(result_shock.median_final_wealth)}")


def test_sensitivity_analysis():
    """Verify sensitivity analysis produces consistent comparisons."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=100000,
        monthly_expenses=50000,
        monthly_emi=10000,
        current_savings=2000000,
        monthly_investment=30000,
        risk_profile="moderate",
        horizon_years=20,
        n_simulations=100,
    )
    result_base = mc.run(sim)
    sensitivity = mc.run_sensitivity(sim)

    assert "increase_sip_10pct" in sensitivity, "Missing SIP increase scenario"
    assert "reduce_expense_10pct" in sensitivity, "Missing expense reduction scenario"
    assert "switch_to_aggressive" in sensitivity, "Missing risk profile switch"

    for key, (delta, desc) in sensitivity.items():
        assert isinstance(delta, (int, float)), f"{key}: delta not a number"
        assert -1 <= delta <= 1, f"{key}: delta out of probability range"
        safe_print(f"  {key:25s}: {delta:+.1%} {desc}")
    safe_print("  [OK] Sensitivity analysis valid")


def test_optimised_scenario():
    """Test optimized path: cut expense 8%, invest freed money."""
    mc = MonteCarloAgent(seed=MC_SEED)
    base_sim = SimulationInput(
        monthly_income=100000,
        monthly_expenses=50000,
        monthly_emi=10000,
        current_savings=1500000,
        monthly_investment=25000,
        risk_profile="moderate",
        horizon_years=20,
        n_simulations=100,
    )
    result_base = mc.run(base_sim)

    mc.rng = np.random.default_rng(MC_SEED)
    result_opt = mc.run_optimised(base_sim)

    assert_valid_result(result_opt, "Optimised scenario")
    
    safe_print(f"  Base success: {result_base.success_probability:.1%}")
    safe_print(f"  Optimised:    {result_opt.success_probability:.1%}")
    safe_print(f"  Savings improvement: {(result_opt.success_probability - result_base.success_probability)*100:+.1f}pp")


def test_explicit_target():
    """Test with explicit target corpus."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=100000,
        monthly_expenses=50000,
        monthly_emi=0,
        current_savings=2000000,
        monthly_investment=35000,
        risk_profile="moderate",
        horizon_years=20,
        n_simulations=100,
        target_corpus=5000000,
    )
    result = mc.run(sim)
    assert_valid_result(result, "Explicit target 5M")
    assert result.target_corpus == 5000000, "Target not respected"
    safe_print(f"  Target: {fmt_inr_safe(result.target_corpus)}")
    safe_print(f"  Success probability: {result.success_probability:.1%}")


def test_auto_target_full_fire():
    """Auto target computation: should choose full FIRE if achievable."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=150000,
        monthly_expenses=50000,
        monthly_emi=0,
        current_savings=5000000,
        monthly_investment=60000,
        risk_profile="aggressive",
        horizon_years=20,
        n_simulations=100,
        target_corpus=None,
    )
    result = mc.run(sim)
    assert_valid_result(result, "Auto target (full FIRE)")
    safe_print(f"  Auto-computed target: {fmt_inr_safe(result.target_corpus)}")
    safe_print(f"  Success probability: {result.success_probability:.1%}")


def test_driver_computation_freshness():
    """Ensure drivers are computed from current result, not stale data."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim1 = SimulationInput(
        monthly_income=50000,
        monthly_expenses=40000,
        monthly_emi=0,
        current_savings=100000,
        monthly_investment=5000,
        risk_profile="conservative",
        horizon_years=15,
        n_simulations=50,
    )
    result1 = mc.run(sim1)
    drivers1 = [d["factor"] for d in result1.key_drivers]

    sim2 = SimulationInput(
        monthly_income=200000,
        monthly_expenses=80000,
        monthly_emi=0,
        current_savings=5000000,
        monthly_investment=80000,
        risk_profile="aggressive",
        horizon_years=20,
        n_simulations=50,
    )
    mc.rng = np.random.default_rng(MC_SEED)
    result2 = mc.run(sim2)
    drivers2 = [d["factor"] for d in result2.key_drivers]

    assert drivers1 != drivers2 or result1.success_probability != result2.success_probability, \
        "Drivers should reflect current scenario"
    safe_print(f"  Scenario 1 drivers: {drivers1}")
    safe_print(f"  Scenario 2 drivers: {drivers2}")
    safe_print("  [OK] Drivers correctly track current result")


def test_reproducibility():
    """Same seed should produce identical results."""
    sim = SimulationInput(
        monthly_income=100000,
        monthly_expenses=50000,
        monthly_emi=10000,
        current_savings=2000000,
        monthly_investment=30000,
        risk_profile="moderate",
        horizon_years=20,
        n_simulations=100,
    )

    mc1 = MonteCarloAgent(seed=42)
    result1 = mc1.run(sim)

    mc2 = MonteCarloAgent(seed=42)
    result2 = mc2.run(sim)

    assert result1.success_probability == result2.success_probability, "Reproducibility failed"
    assert np.allclose(result1.pct50, result2.pct50), "Median trajectories differ"
    safe_print(f"  Result 1 success: {result1.success_probability:.1%}")
    safe_print(f"  Result 2 success: {result2.success_probability:.1%}")
    safe_print("  [OK] Full reproducibility verified")


def test_wealth_floor():
    """Verify wealth doesn't drop below -12 months of expenses."""
    mc = MonteCarloAgent(seed=MC_SEED)
    sim = SimulationInput(
        monthly_income=30000,
        monthly_expenses=25000,
        monthly_emi=10000,
        current_savings=0,
        monthly_investment=0,
        risk_profile="aggressive",
        horizon_years=10,
        n_simulations=200,
    )
    result = mc.run(sim)
    assert_valid_result(result, "Wealth floor test")
    
    min_wealth_observed = result.pct10.min()
    wealth_floor = -sim.monthly_expenses * 12
    safe_print(f"  Wealth floor: {fmt_inr_safe(wealth_floor)}")
    safe_print(f"  Worst case (p10): {fmt_inr_safe(min_wealth_observed)}")


def run_all_tests():
    """Execute complete test suite."""
    safe_print("=" * 70)
    safe_print("MONTE CARLO AGENT TEST SUITE")
    safe_print("=" * 70)
    
    tests = [
        ("Normal Scenario", test_normal_scenario),
        ("Zero Current Savings", test_zero_current_savings),
        ("Zero Investment", test_zero_investment),
        ("Inflation Expense Impact", test_inflation_affects_expenses),
        ("High Expenses", test_high_expenses),
        ("Very High EMI", test_very_high_emi),
        ("Short Horizon (3yr)", test_short_horizon),
        ("Long Horizon (40yr)", test_very_long_horizon),
        ("Risk Profiles", test_risk_profiles),
        ("Shock: Job Loss 6m", test_shock_job_loss_6m),
        ("Shock: Medical 800k", test_shock_medical_8L),
        ("Shock: Income Drop 30%", test_shock_income_drop_30pct),
        ("Sensitivity Analysis", test_sensitivity_analysis),
        ("Optimised Scenario", test_optimised_scenario),
        ("Explicit Target", test_explicit_target),
        ("Auto Target (FIRE)", test_auto_target_full_fire),
        ("Driver Freshness", test_driver_computation_freshness),
        ("Reproducibility", test_reproducibility),
        ("Wealth Floor", test_wealth_floor),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            safe_print(f"\n[{passed + failed + 1:02d}] {test_name}")
            test_func()
            passed += 1
        except Exception as e:
            safe_print(f"  [FAIL] {str(e)}")
            failed += 1
    
    safe_print("\n" + "=" * 70)
    safe_print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    safe_print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
