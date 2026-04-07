"""
agents/goal_agent.py
Converts raw goals into:
  - Required corpus (inflation-adjusted)
  - Monthly SIP needed
  - Shortfall from current trajectory
  - Probability of achievement (from MC engine)
"""
from dataclasses import dataclass, field
from config import GOAL_INFLATION, GOAL_TARGETS_TODAY, RISK_RETURN
from utils.financial_math import (
    sip_required, sip_future_value, lumpsum_future_value,
    inflate, fire_corpus, fmt_inr,
)


GOAL_LABELS = {
    "house":      "🏠 Home Down Payment",
    "retirement": "🌴 FIRE / Retirement",
    "child":      "👶 Child's Education",
    "marriage":   "💍 Marriage",
    "vehicle":    "🚗 Vehicle",
    "emergency":  "🛡️ Emergency Fund",
    "travel":     "✈️ Travel Fund",
    "startup":    "🚀 Startup Capital",
}


@dataclass
class GoalPlan:
    goal_key:       str
    label:          str
    horizon_years:  int
    target_today:   float       # in today's ₹
    target_future:  float       # inflation-adjusted ₹ at deadline
    sip_required:   float       # monthly SIP needed
    current_trajectory: float   # what current SIP grows to
    shortfall:      float       # gap at deadline
    probability:    float       # 0–1 from MC (set by MC agent externally)
    priority:       str         # high / medium / low
    note:           str = ""


class GoalAgent:
    """
    Decomposes user goals into actionable financial targets.
    """

    def run(self, user_data: dict) -> list:
        """
        Returns list[GoalPlan] for each goal in user_data["goals"].
        """
        goals        = user_data.get("goals", [])
        income       = user_data["monthly_income"]
        expenses     = user_data["monthly_expenses"]
        emi          = user_data.get("monthly_emi", 0)
        savings      = user_data.get("current_savings", 0)
        invest       = user_data.get("monthly_investment", 0)
        age          = user_data.get("age", 30)
        retire_age   = user_data.get("retirement_age", 50)
        risk         = user_data.get("risk_profile", "moderate")

        rp     = RISK_RETURN.get(risk, RISK_RETURN["moderate"])
        r_mean = rp["mean"]

        plans = []
        for g in goals:
            plan = self._plan_goal(
                goal_key=g, income=income, expenses=expenses,
                emi=emi, savings=savings, invest=invest,
                age=age, retire_age=retire_age, r=r_mean,
            )
            if plan:
                plans.append(plan)

        # Sort by priority then horizon
        priority_order = {"high": 0, "medium": 1, "low": 2}
        plans.sort(key=lambda p: (priority_order.get(p.priority, 1), p.horizon_years))

        return plans

    def attach_probabilities(self, plans: list, mc_agent, user_data: dict) -> list:
        """
        Run per-goal MC simulation and attach probability to each GoalPlan.
        Uses full current resources per goal — shows "if you focused everything
        on THIS goal, what is your probability?" — the most useful planning metric.
        """
        from agents.monte_carlo_agent import SimulationInput
        from config import MC_SEED
        import numpy as np

        income_growth = user_data.get("income_growth_pct", 10) / 100

        for plan in plans:
            mc_agent.rng = np.random.default_rng(MC_SEED)
            si = SimulationInput(
                monthly_income=user_data["monthly_income"],
                monthly_expenses=user_data["monthly_expenses"],
                monthly_emi=user_data.get("monthly_emi", 0),
                current_savings=user_data.get("current_savings", 0),
                monthly_investment=user_data.get("monthly_investment", 0),
                risk_profile=user_data.get("risk_profile", "moderate"),
                horizon_years=max(1, plan.horizon_years),
                target_corpus=plan.target_future,
                n_simulations=200,
                income_growth_mean=income_growth,
            )
            result = mc_agent.run(si)
            plan.probability = result.success_probability
        return plans

    # ── Private ───────────────────────────────────────────────────────────────

    def _plan_goal(self, goal_key: str, income: float, expenses: float,
                   emi: float, savings: float, invest: float,
                   age: int, retire_age: int, r: float) -> GoalPlan:

        inf_rate = GOAL_INFLATION.get(goal_key, 0.06)

        # Horizon
        if goal_key == "retirement":
            horizon = max(1, retire_age - age)
        elif goal_key == "emergency":
            horizon = 1
        elif goal_key == "travel":
            horizon = 1
        elif goal_key == "house":
            horizon = 5
        elif goal_key == "marriage":
            horizon = 3
        elif goal_key == "vehicle":
            horizon = 2
        elif goal_key == "child":
            horizon = 18
        else:
            horizon = 5

        # Target corpus today
        if goal_key == "emergency":
            target_today = (expenses + emi) * 6
        elif goal_key == "retirement":
            target_today = None   # use FIRE formula
        else:
            target_today = GOAL_TARGETS_TODAY.get(goal_key, 1_000_000)

        # Target in future ₹
        if goal_key == "retirement":
            target_future = fire_corpus(
                annual_expenses_today=expenses * 12,
                inflation_rate=inf_rate,
                years_to_retire=horizon,
            )
            target_today = target_future / ((1 + inf_rate) ** horizon)
        else:
            target_future = inflate(target_today, inf_rate, horizon)

        # What current savings + SIP reaches by deadline
        fv_savings = lumpsum_future_value(savings * 0.3, r, horizon)   # 30% of savings allocated
        fv_sip     = sip_future_value(invest * 0.3, r, horizon)         # 30% of SIP allocated
        current_trajectory = fv_savings + fv_sip

        # SIP needed for just this goal (allocate none of existing savings)
        sip_needed = sip_required(max(0, target_future - lumpsum_future_value(0, r, horizon)), r, horizon)

        shortfall = max(0, target_future - current_trajectory)

        # Priority
        if goal_key in ("emergency", "retirement") or horizon <= 2:
            priority = "high"
        elif horizon <= 7:
            priority = "medium"
        else:
            priority = "low"

        # Note
        note = self._goal_note(goal_key, target_future, sip_needed, horizon, shortfall)

        return GoalPlan(
            goal_key=goal_key,
            label=GOAL_LABELS.get(goal_key, goal_key.title()),
            horizon_years=horizon,
            target_today=target_today,
            target_future=target_future,
            sip_required=sip_needed,
            current_trajectory=current_trajectory,
            shortfall=shortfall,
            probability=0.0,   # set externally by MC agent
            priority=priority,
            note=note,
        )

    def _goal_note(self, key: str, target: float, sip: float,
                   horizon: int, shortfall: float) -> str:
        if key == "emergency":
            return "Keep in liquid fund / high-yield savings. Not for investment."
        if key == "retirement":
            return f"Use index funds + NPS Tier 1. ELSS for tax efficiency. Start today."
        if key == "house":
            return f"Target {fmt_inr(target)} down payment in {horizon}yr. Use balanced hybrid fund."
        if key == "child":
            return f"18-year horizon — pure equity. Don't touch until goal year."
        if shortfall > sip * 12:
            return f"Shortfall of {fmt_inr(shortfall)} — need {fmt_inr(sip)}/mo dedicated SIP."
        return f"Achievable with {fmt_inr(sip)}/mo dedicated SIP."
