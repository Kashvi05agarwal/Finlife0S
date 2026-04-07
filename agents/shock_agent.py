"""
agents/shock_agent.py
Life Shock Simulator — re-runs Monte Carlo with each shock applied.
Uses a calibrated 10-year intermediate target (~50% base prob) so
probability deltas are always visible and meaningful in the demo.
"""
from dataclasses import dataclass
from config import SHOCK_PRESETS
from utils.financial_math import fmt_inr


@dataclass
class ShockResult:
    shock_name:           str
    base_probability:     float
    shocked_probability:  float
    prob_delta:           float
    base_corpus_p50:      float
    shocked_corpus_p50:   float
    corpus_delta:         float
    years_delayed:        float
    description:          str
    recommendation:       str


SHOCK_DESCRIPTIONS = {
    "job_loss_6m":    ("💼 Job Loss (6 months)",
                       "You lose all income for 6 months — SIPs paused, emergency fund depleted."),
    "income_drop_30": ("📉 Income Drop 30% (2 years)",
                       "Career switch or pay cut: 30% less income for 2 years."),
    "medical_8L":     ("🏥 Medical Emergency (₹8L)",
                       "Unplanned medical expense of ₹8 lakh hits immediately."),
    "marriage_15L":   ("💍 Marriage Expense (₹15L)",
                       "One-time marriage cost of ₹15 lakh drawn from savings/investments."),
    "career_switch":  ("🔄 Career Switch (30% drop, 2yr)",
                       "Switch to a new field: 30% income drop for 2 years before recovery."),
}

SHOCK_RECS = {
    "job_loss_6m":    "Build 9-month emergency fund. Consider income-protection insurance.",
    "income_drop_30": "Keep 12 months emergency buffer. Reduce fixed commitments before switching.",
    "medical_8L":     "A ₹10L family floater health insurance is non-negotiable. Cost: ~₹15,000/yr.",
    "marriage_15L":   "Start a dedicated 3-year marriage SIP now. Don't use retirement savings.",
    "career_switch":  "Prepare 12-month cash cover. Vest ESOPs and clear high-interest debt first.",
}


class ShockAgent:

    def run_all(self, mc_agent, sim_input, base_result) -> list:
        """
        Run all preset shocks against a calibrated 10yr intermediate scenario
        so base probability sits near 50% — making deltas clearly visible.
        """
        import numpy as np
        from config import MC_SEED
        from agents.monte_carlo_agent import SimulationInput

        mid_horizon = min(10, sim_input.horizon_years)

        # Calibration pass: get 10yr median wealth (no fixed target)
        mc_agent.rng = np.random.default_rng(MC_SEED)
        cal = SimulationInput(
            monthly_income=sim_input.monthly_income,
            monthly_expenses=sim_input.monthly_expenses,
            monthly_emi=sim_input.monthly_emi,
            current_savings=sim_input.current_savings,
            monthly_investment=sim_input.monthly_investment,
            risk_profile=sim_input.risk_profile,
            horizon_years=mid_horizon,
            n_simulations=150,
            income_growth_mean=sim_input.income_growth_mean,
            income_growth_std=sim_input.income_growth_std,
        )
        cal_r = mc_agent.run(cal)
        # Target = 10yr median → base probability ≈ 50%
        mid_target = cal_r.median_final_wealth

        mid_si = SimulationInput(
            monthly_income=sim_input.monthly_income,
            monthly_expenses=sim_input.monthly_expenses,
            monthly_emi=sim_input.monthly_emi,
            current_savings=sim_input.current_savings,
            monthly_investment=sim_input.monthly_investment,
            risk_profile=sim_input.risk_profile,
            horizon_years=mid_horizon,
            target_corpus=mid_target,
            n_simulations=250,
            income_growth_mean=sim_input.income_growth_mean,
            income_growth_std=sim_input.income_growth_std,
        )
        mc_agent.rng = np.random.default_rng(MC_SEED)
        mid_base = mc_agent.run(mid_si)

        results = []
        for key, params in SHOCK_PRESETS.items():
            mc_agent.rng = np.random.default_rng(MC_SEED)
            sr = self._run_one(key, params, mc_agent, mid_si, mid_base)
            results.append(sr)
        return results

    def run_one(self, shock_key: str, mc_agent, sim_input, base_result) -> "ShockResult":
        params = SHOCK_PRESETS.get(shock_key)
        if not params:
            raise ValueError(f"Unknown shock key: {shock_key}")
        return self._run_one(shock_key, params, mc_agent, sim_input, base_result)

    def run_custom(self, income_factor: float, duration_months: int,
                   one_time_cost: float, mc_agent, sim_input, base_result) -> "ShockResult":
        params = {
            "income_factor": income_factor,
            "duration_months": duration_months,
            "one_time_cost": one_time_cost,
        }
        return self._run_one("custom", params, mc_agent, sim_input, base_result)

    def _run_one(self, key: str, params: dict,
                 mc_agent, sim_input, base_result) -> "ShockResult":
        shocked_result = mc_agent.run_with_shock(sim_input, {**params, "shock_year": 2})

        prob_delta   = shocked_result.success_probability - base_result.success_probability
        corpus_delta = shocked_result.median_final_wealth  - base_result.median_final_wealth
        years_delay  = self._estimate_delay(base_result.median_final_wealth,
                                             shocked_result.median_final_wealth)

        name, desc = SHOCK_DESCRIPTIONS.get(key, ("Custom", "Custom shock scenario"))
        rec        = SHOCK_RECS.get(key, "Build a larger emergency buffer as protection.")

        return ShockResult(
            shock_name=name,
            base_probability=base_result.success_probability,
            shocked_probability=shocked_result.success_probability,
            prob_delta=prob_delta,
            base_corpus_p50=base_result.median_final_wealth,
            shocked_corpus_p50=shocked_result.median_final_wealth,
            corpus_delta=corpus_delta,
            years_delayed=years_delay,
            description=desc,
            recommendation=rec,
        )

    def _estimate_delay(self, base: float, shocked: float) -> float:
        if shocked >= base or base <= 0:
            return 0.0
        import math
        try:
            return round(min(math.log(base / max(shocked, 1)) / math.log(1.11), 10.0), 1)
        except (ValueError, ZeroDivisionError):
            return 0.0
