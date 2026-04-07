"""
agents/monte_carlo_agent.py
Monte Carlo Financial Simulation Engine
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config import (
    MC_SIMULATIONS, MC_HORIZON_YEARS, MC_SEED,
    RETURN_MEAN, RETURN_STD,
    INFLATION_MEAN, INFLATION_STD,
    INCOME_GROWTH_MEAN, INCOME_GROWTH_STD,
    RISK_RETURN,
)
from utils.financial_math import fire_corpus, fmt_inr


@dataclass
class SimulationInput:
    monthly_income:        float
    monthly_expenses:      float
    monthly_emi:           float
    current_savings:       float
    monthly_investment:    float
    risk_profile:          str   = "moderate"
    horizon_years:         int   = MC_HORIZON_YEARS
    n_simulations:         int   = MC_SIMULATIONS
    target_corpus:         Optional[float] = None
    income_growth_mean:    float = INCOME_GROWTH_MEAN
    income_growth_std:     float = INCOME_GROWTH_STD
    inflation_mean:        float = INFLATION_MEAN
    inflation_std:         float = INFLATION_STD
    shock_income_factor:   float = 1.0
    shock_duration_months: int   = 0
    shock_one_time_cost:   float = 0.0
    shock_year:            int   = 0


@dataclass
class SimulationResult:
    success_probability:        float
    target_corpus:              float
    pct10:  np.ndarray = field(default_factory=lambda: np.array([]))
    pct50:  np.ndarray = field(default_factory=lambda: np.array([]))
    pct90:  np.ndarray = field(default_factory=lambda: np.array([]))
    wealth_matrix:              np.ndarray = field(default_factory=lambda: np.array([]))
    final_wealth_distribution:  np.ndarray = field(default_factory=lambda: np.array([]))
    median_final_wealth:        float = 0.0
    mean_final_wealth:          float = 0.0
    worst_case_p10:             float = 0.0
    best_case_p90:              float = 0.0
    key_drivers:                list  = field(default_factory=list)
    years_range:                list  = field(default_factory=list)
    # Audit trail: every agent decision logged here
    audit_log:                  list  = field(default_factory=list)


class MonteCarloAgent:
    def __init__(self, seed: int = MC_SEED):
        self.rng = np.random.default_rng(seed)
        self.last_input:  Optional[SimulationInput]  = None
        self.last_result: Optional[SimulationResult] = None

    def run(self, sim_input: SimulationInput) -> SimulationResult:
        self.last_input = sim_input
        inp = sim_input
        audit = []

        rp          = RISK_RETURN.get(inp.risk_profile, RISK_RETURN["moderate"])
        return_mean = rp["mean"]
        return_std  = rp["std"]
        audit.append(f"Risk profile '{inp.risk_profile}': expected return {return_mean*100:.1f}% ±{return_std*100:.1f}%")

        H = inp.horizon_years
        N = inp.n_simulations

        log_mean = np.log(1 + return_mean) - 0.5 * (return_std ** 2)
        returns  = self.rng.lognormal(log_mean, return_std, size=(N, H)) - 1

        inflation     = self.rng.normal(inp.inflation_mean, inp.inflation_std, size=(N, H))
        inflation     = np.clip(inflation, 0.02, 0.15)
        income_growth = self.rng.normal(inp.income_growth_mean, inp.income_growth_std, size=(N, H))
        income_growth = np.clip(income_growth, -0.10, 0.30)

        wealth         = np.zeros((N, H + 1))
        wealth[:, 0]   = inp.current_savings
        monthly_invest = np.full(N, inp.monthly_investment)
        monthly_income = np.full(N, inp.monthly_income)
        monthly_expenses = np.full(N, inp.monthly_expenses)

        for yr in range(H):
            income_factor = 1.0
            extra_cost    = 0.0
            if inp.shock_year > 0:
                shock_start_yr     = inp.shock_year - 1
                shock_duration_yrs = inp.shock_duration_months / 12
                shock_end_yr       = shock_start_yr + shock_duration_yrs
                if shock_start_yr <= yr < shock_end_yr:
                    overlap       = min(yr + 1, shock_end_yr) - max(yr, shock_start_yr)
                    frac          = min(1.0, overlap)
                    income_factor = 1.0 - (1.0 - inp.shock_income_factor) * frac
                if yr == inp.shock_year - 1:
                    extra_cost = inp.shock_one_time_cost

            effective_income = monthly_income * income_factor
            monthly_surplus  = np.maximum(effective_income - monthly_expenses - inp.monthly_emi, 0)
            invest_this_yr   = np.minimum(monthly_invest, monthly_surplus)
            annual_invest    = invest_this_yr * 12

            r          = returns[:, yr]
            new_wealth = (
                wealth[:, yr] * (1 + r)
                + annual_invest * (1 + r / 2)
                - extra_cost
            )
            min_wealth        = -monthly_expenses * 12
            wealth[:, yr + 1] = np.maximum(new_wealth, min_wealth)

            monthly_income  = monthly_income * (1 + income_growth[:, yr])
            monthly_invest  = monthly_invest * (1 + income_growth[:, yr] * 0.5)
            monthly_expenses = monthly_expenses * (1 + inflation[:, yr])

        final_wealth = wealth[:, H]

        # ── Resolve target corpus ──────────────────────────────────────────
        target = inp.target_corpus
        if target is None:
            full_fire = fire_corpus(
                annual_expenses_today=inp.monthly_expenses * 12,
                inflation_rate=inp.inflation_mean,
                years_to_retire=H,
            )
            p70_wealth = float(np.percentile(final_wealth, 70))
            if p70_wealth < full_fire * 0.60:
                target = full_fire * 0.5
                audit.append(f"Target set to 50% FIRE (₹{target/1e5:.1f}L) — full FIRE (₹{full_fire/1e5:.1f}L) unreachable at current trajectory")
            else:
                target = full_fire
                audit.append(f"Target set to full FIRE corpus: ₹{target/1e5:.1f}L")

        success_prob = float(np.mean(final_wealth >= target))
        audit.append(f"Success probability: {success_prob*100:.1f}% ({int(success_prob*N)}/{N} paths reached ₹{target/1e5:.1f}L)")

        pct10 = np.percentile(wealth[:, 1:], 10, axis=0)
        pct50 = np.percentile(wealth[:, 1:], 50, axis=0)
        pct90 = np.percentile(wealth[:, 1:], 90, axis=0)

        median_wealth = float(np.median(final_wealth))
        result = SimulationResult(
            success_probability=success_prob,
            target_corpus=target,
            pct10=pct10, pct50=pct50, pct90=pct90,
            wealth_matrix=wealth[:, 1:],
            final_wealth_distribution=final_wealth,
            median_final_wealth=median_wealth,
            mean_final_wealth=float(np.mean(final_wealth)),
            worst_case_p10=float(np.percentile(final_wealth, 10)),
            best_case_p90=float(np.percentile(final_wealth, 90)),
            key_drivers=self._compute_drivers(sim_input, success_prob, target, median_wealth),
            years_range=list(range(1, H + 1)),
            audit_log=audit,
        )
        self.last_result = result
        return result

    def run_sensitivity(self, sim_input: SimulationInput) -> dict:
        original_result = self.last_result
        base   = self.run(sim_input).success_probability
        target = self.last_result.target_corpus
        results = {}

        def _delta(modified_input):
            modified_input.target_corpus = target
            self.rng = np.random.default_rng(MC_SEED)
            r = self.run(modified_input)
            return r.success_probability - base

        try:
            inp_sip = SimulationInput(**sim_input.__dict__)
            inp_sip.monthly_investment = sim_input.monthly_investment * 1.10
            results["increase_sip_10pct"] = (_delta(inp_sip),
                f"Increase SIP by 10% ({fmt_inr(sim_input.monthly_investment * 0.1)}/mo)")

            inp_exp = SimulationInput(**sim_input.__dict__)
            inp_exp.monthly_expenses = sim_input.monthly_expenses * 0.90
            results["reduce_expense_10pct"] = (_delta(inp_exp),
                f"Reduce expenses by 10% ({fmt_inr(sim_input.monthly_expenses * 0.1)}/mo)")

            inp_ret = SimulationInput(**sim_input.__dict__)
            inp_ret.risk_profile = "aggressive"
            results["switch_to_aggressive"] = (_delta(inp_ret),
                "Switch to equity-heavy (aggressive) portfolio")

            if sim_input.horizon_years < 30:
                inp_early = SimulationInput(**sim_input.__dict__)
                inp_early.horizon_years = sim_input.horizon_years + 5
                results["5_years_earlier"] = (_delta(inp_early),
                    "Had started investing 5 years earlier")
        finally:
            self.last_result = original_result

        return results

    def run_with_shock(self, sim_input: SimulationInput, shock_params: dict) -> SimulationResult:
        shocked = SimulationInput(**sim_input.__dict__)
        shocked.shock_income_factor    = shock_params.get("income_factor", 1.0)
        shocked.shock_duration_months  = shock_params.get("duration_months", 0)
        shocked.shock_one_time_cost    = shock_params.get("one_time_cost", 0.0)
        shocked.shock_year             = shock_params.get("shock_year", 3)
        if self.last_result:
            shocked.target_corpus = self.last_result.target_corpus
        self.rng = np.random.default_rng(MC_SEED)
        return self.run(shocked)

    def run_optimised(self, sim_input: SimulationInput) -> SimulationResult:
        """
        Optimised scenario: redirect expense savings into investment.
        ALWAYS produces investment >= base investment.
        Uses reduced expenses to compute new surplus, then invests the freed money.
        """
        opt = SimulationInput(**sim_input.__dict__)

        # Step 1: cut expenses 8%
        opt.monthly_expenses = sim_input.monthly_expenses * 0.92
        freed_monthly = sim_input.monthly_expenses - opt.monthly_expenses

        # Step 2: invest freed money + keep base investment
        # Use updated surplus (with lower expenses) for safety cap
        new_surplus = sim_input.monthly_income - opt.monthly_expenses - sim_input.monthly_emi
        # Investment = base + freed, but never exceed 80% of new surplus
        opt.monthly_investment = min(
            sim_input.monthly_investment + freed_monthly,
            new_surplus * 0.80
        )
        # Hard floor: optimised investment MUST be >= base
        opt.monthly_investment = max(opt.monthly_investment, sim_input.monthly_investment)

        # Step 3: model slightly better career (1% extra income growth from focus on career)
        opt.income_growth_mean = min(sim_input.income_growth_mean + 0.01, 0.15)

        # Step 4: use same target corpus as base for fair comparison
        if self.last_result is not None and self.last_result.target_corpus:
            opt.target_corpus = self.last_result.target_corpus

        self.rng = np.random.default_rng(MC_SEED)
        return self.run(opt)

    def _compute_drivers(self, inp: SimulationInput, prob: float, target: float, median_final_wealth: float) -> list:
        drivers       = []
        invest_ratio  = inp.monthly_investment / inp.monthly_income if inp.monthly_income > 0 else 0
        expense_ratio = inp.monthly_expenses / inp.monthly_income   if inp.monthly_income > 0 else 0
        H = inp.horizon_years

        # Driver 1: Investment rate
        if invest_ratio < 0.10:
            shortfall = inp.monthly_income * 0.20 - inp.monthly_investment
            drivers.append({"factor": "Low Investment Rate",
                "value": f"{invest_ratio*100:.1f}% of income (target: 20%)",
                "impact": "high", "direction": "negative",
                "description": f"Only {invest_ratio*100:.1f}% of income invested. "
                    f"Adding {fmt_inr(shortfall)}/mo could lift probability by 15–25pp."})
        elif invest_ratio < 0.20:
            gap = inp.monthly_income * 0.20 - inp.monthly_investment
            drivers.append({"factor": "Below 20% Investment Rate",
                "value": f"{invest_ratio*100:.1f}% of income (need 20%)",
                "impact": "high", "direction": "negative",
                "description": f"Investing {invest_ratio*100:.1f}% vs 20% target. "
                    f"Gap: {fmt_inr(gap)}/mo. This single change has the highest leverage."})
        else:
            drivers.append({"factor": "Strong Investment Discipline",
                "value": f"{invest_ratio*100:.1f}% of income",
                "impact": "high", "direction": "positive",
                "description": f"Investing {invest_ratio*100:.1f}% — top quartile. "
                    f"This discipline is the primary driver of your {prob*100:.0f}% probability."})

        # Driver 2: Time horizon
        if H < 10:
            drivers.append({"factor": "Short Time Horizon",
                "value": f"{H} years to goal",
                "impact": "high", "direction": "negative",
                "description": f"Only {H} years limits compounding power. "
                    f"Each extra year adds significant corpus — even 2 more years makes a large difference."})
        elif H >= 15:
            drivers.append({"factor": "Long Compounding Runway",
                "value": f"{H} years",
                "impact": "high", "direction": "positive",
                "description": f"{H} years is powerful — the last 10 years generate more wealth "
                    f"than the first {H-10} combined. Protect this runway."})
        else:
            direction = "positive" if H >= 12 else "negative"
            drivers.append({"factor": "Moderate Time Horizon",
                "value": f"{H} years",
                "impact": "medium", "direction": direction,
                "description": f"{H} years gives reasonable room. Consistency matters more than timing."})

        # Driver 3: Gap vs target OR expense pressure
        corpus_ratio = (median_final_wealth / target) if (target > 0) else 0
        if expense_ratio > 0.60:
            drivers.append({"factor": "High Expense Ratio",
                "value": f"{expense_ratio*100:.0f}% of income on expenses",
                "impact": "medium", "direction": "negative",
                "description": f"Expenses eat {expense_ratio*100:.0f}% of income. "
                    f"A 5% cut ({fmt_inr(inp.monthly_income * 0.05)}/mo) invested for {H}yr "
                    f"adds ~{fmt_inr(inp.monthly_income * 0.05 * 12 * (1.11**H - 1) / 0.11)}."})
        elif corpus_ratio < 0.70:
            median_w  = median_final_wealth
            gap_amt   = max(0, target - median_w)
            drivers.append({"factor": "Corpus Gap vs Goal",
                "value": f"On track for {corpus_ratio*100:.0f}% of goal",
                "impact": "high", "direction": "negative",
                "description": f"Median trajectory reaches {fmt_inr(median_w)}, "
                    f"short of {fmt_inr(target)} goal by {fmt_inr(gap_amt)}. "
                    f"Adding {fmt_inr(gap_amt / max(1, H * 12 * 2))}/mo closes half this gap."})
        else:
            drivers.append({"factor": "On-Track Trajectory",
                "value": f"Median at {corpus_ratio*100:.0f}% of goal",
                "impact": "medium", "direction": "positive",
                "description": f"Median corpus covers {corpus_ratio*100:.0f}% of goal. "
                    f"Add 10% annual SIP step-up to close remaining gap."})

        return drivers[:3]