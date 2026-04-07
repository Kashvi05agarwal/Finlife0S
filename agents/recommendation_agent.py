"""
agents/recommendation_agent.py
Generates top-5 ranked recommendations with ₹ impact + confidence score.
"""
from dataclasses import dataclass
from utils.financial_math import (
    sip_future_value, lumpsum_future_value,
    idle_cash_loss, opportunity_cost,
    fmt_inr,
)
from config import REC_IMPACT_HORIZON


@dataclass
class Recommendation:
    rank:         int
    title:        str
    description:  str
    action:       str          # specific, 1-line action
    rupee_impact: float        # ₹ impact over 20 years
    impact_label: str          # formatted ₹ impact string
    confidence:   float        # 0–1
    urgency:      str          # immediate / 3-months / 6-months
    category:     str          # invest / protect / save / tax / debt


class RecommendationAgent:

    def run(self, user_data: dict, health_report, mc_base_prob: float) -> list:
        """
        Generate and rank up to 5 recommendations.
        Returns list[Recommendation] sorted by rupee_impact desc.
        """
        recs = []
        income  = user_data["monthly_income"]
        expense = user_data["monthly_expenses"]
        savings = user_data.get("current_savings", 0)
        invest  = user_data.get("monthly_investment", 0)
        emi     = user_data.get("monthly_emi", 0)
        age     = user_data.get("age", 30)
        surplus = income - expense - emi
        H       = REC_IMPACT_HORIZON

        # ── R1: Increase SIP ─────────────────────────────────────────────────
        target_invest = income * 0.20
        if invest < target_invest:
            gap   = target_invest - invest
            impact = sip_future_value(gap, 0.11, H)
            recs.append(Recommendation(
                rank=0, category="invest",
                title="Increase Monthly SIP",
                description=(
                    f"You're investing {invest/income*100:.0f}% of income. "
                    f"The 20% benchmark needs {fmt_inr(target_invest)}/mo."
                ),
                action=f"Set up additional SIP of {fmt_inr(gap)}/mo in a Nifty 50 index fund.",
                rupee_impact=impact,
                impact_label=fmt_inr(impact),
                confidence=0.92,
                urgency="immediate",
            ))

        # ── R2: Emergency Fund ───────────────────────────────────────────────
        target_ef = (expense + emi) * 6
        if savings < target_ef:
            shortfall = target_ef - savings
            # Opportunity: having this reduces shock risk, protecting future corpus
            impact = shortfall * 1.5   # proxy: 1.5x the shortfall as risk-adjusted savings
            expense_base = expense + emi
            months_covered = savings / expense_base if expense_base > 0 else 0.0
            transfer_amount = max(0.0, min(shortfall, max(0.0, surplus * 3)))
            auto_save = max(0.0, min(shortfall / 6, max(0.0, surplus * 0.3)))
            recs.append(Recommendation(
                rank=0, category="protect",
                title="Build Emergency Fund",
                description=(
                    f"Only {months_covered:.1f} months covered. "
                    f"Target: {fmt_inr(target_ef)} in liquid fund."
                ),
                action=f"Transfer {fmt_inr(transfer_amount)} to a liquid fund now. Auto-save {fmt_inr(auto_save)}/mo.",
                rupee_impact=impact,
                impact_label=fmt_inr(impact),
                confidence=0.95,
                urgency="immediate",
            ))

        # ── R3: Idle Cash Opportunity ────────────────────────────────────────
        idle_cash = max(0, savings - target_ef)
        if idle_cash > 50_000:
            impact = idle_cash_loss(idle_cash / 12, 0.11, H)
            recs.append(Recommendation(
                rank=0, category="invest",
                title="Deploy Idle Savings",
                description=(
                    f"{fmt_inr(idle_cash)} sitting in savings account earns 3.5% — "
                    f"losing {fmt_inr(impact)} to opportunity cost over {H} years."
                ),
                action=f"Move {fmt_inr(idle_cash)} to Flexi-cap or index fund as lump sum. Do it in 2-3 tranches.",
                rupee_impact=impact,
                impact_label=fmt_inr(impact),
                confidence=0.88,
                urgency="3-months",
            ))

        # ── R4: Tax Savings ──────────────────────────────────────────────────
        annual_income = income * 12
        if annual_income > 700_000:
            # Max 80C + 80D + NPS = ~2.5L savings on ₹46,800 tax
            tax_saving_investment = 150_000   # 80C
            nps_additional = 50_000            # 80CCD(1B)
            total_deduction = tax_saving_investment + nps_additional
            tax_saved_pa = total_deduction * 0.20   # ~20% marginal rate
            impact_20yr = tax_saved_pa * H * (1 + 0.11) ** (H // 2)
            recs.append(Recommendation(
                rank=0, category="tax",
                title="Max Tax-Saving Investments",
                description=(
                    f"Potential annual tax saving: {fmt_inr(tax_saved_pa)} "
                    f"via 80C + NPS 80CCD(1B) — likely not fully utilised."
                ),
                action=f"Invest {fmt_inr(tax_saving_investment)} in ELSS + {fmt_inr(nps_additional)} in NPS Tier 1 this FY.",
                rupee_impact=impact_20yr,
                impact_label=fmt_inr(impact_20yr),
                confidence=0.82,
                urgency="6-months",
            ))

        # ── R5: Debt Prepayment ──────────────────────────────────────────────
        if emi > income * 0.25:
            # Interest cost saved by prepaying
            # Assume 10% interest rate, 10yr remaining
            approx_loan = emi * 100   # rough proxy
            interest_saved = approx_loan * 0.10 * 5   # 5 yrs of interest
            recs.append(Recommendation(
                rank=0, category="debt",
                title="Accelerate Loan Repayment",
                description=(
                    f"EMI of {fmt_inr(emi)}/mo is {emi/income*100:.0f}% of income. "
                    f"Early prepayment unlocks future surplus."
                ),
                action=f"Direct next bonus/windfall to principal prepayment. Target: reduce EMI by {fmt_inr(emi*0.2)}/mo.",
                rupee_impact=interest_saved,
                impact_label=fmt_inr(interest_saved),
                confidence=0.78,
                urgency="6-months",
            ))

        # ── R6: SIP Step-Up (correct formula) ───────────────────────────────
        # Impact = FV of stepped-up SIP - FV of flat SIP, over H years
        # Stepped-up: each year invest grows 10%. Compute compound series.
        import math
        r_annual = 0.11
        r_monthly = r_annual / 12
        n_months = H * 12
        step_rate = 0.10  # 10% annual increase

        def fv_stepped_sip(monthly_sip, r_m, n_months, step_annual):
            """FV of SIP with annual step-up using closed-form approximation."""
            # Approximate: use midpoint growth rate
            avg_monthly = monthly_sip * ((1 + step_annual) ** (n_months/12/2))
            return avg_monthly * ((1 + r_m) ** n_months - 1) / r_m * (1 + r_m)

        fv_flat    = sip_future_value(invest, r_annual, H)
        fv_stepped = fv_stepped_sip(invest, r_monthly, n_months, step_rate)
        impact_stepup = max(0, fv_stepped - fv_flat)

        recs.append(Recommendation(
            rank=0, category="invest",
            title="Set Up 10% Annual SIP Step-Up",
            description=(
                f"A flat {fmt_inr(invest)}/mo SIP builds {fmt_inr(fv_flat)} in {H} years. "
                f"With 10% annual step-up: {fmt_inr(fv_stepped)}. "
                f"Difference: {fmt_inr(impact_stepup)} from one habit change."
            ),
            action=f"Enable 10% annual step-up in your SIP portal. Zerodha, Groww, MFCentral all support this.",
            rupee_impact=impact_stepup,
            impact_label=fmt_inr(impact_stepup),
            confidence=0.90,
            urgency="3-months",
        ))

        # ── R7: Term Insurance (if income > 4L/yr and age < 50) ─────────────
        if income * 12 > 400_000 and age < 50:
            cover_needed = income * 12 * 15   # 15x annual income rule
            annual_premium = int(cover_needed * 0.0006 * (1 + (age-25)*0.04))  # age-adjusted
            annual_premium = max(8_000, annual_premium)
            # Impact: protection of entire corpus in case of death
            impact_ins = cover_needed * 0.15   # rough: 15% of cover as risk-adjusted value
            recs.append(Recommendation(
                rank=0, category="protect",
                title="Get " + fmt_inr(cover_needed) + " Term Insurance",
                description=(
                    f"15× annual income rule requires {fmt_inr(cover_needed)} cover. "
                    f"Annual premium at age {age}: ~{fmt_inr(annual_premium)}/yr ({fmt_inr(annual_premium//12)}/mo). "
                    f"Without it, one event wipes out everything you've built."
                ),
                action=f"Compare term plans on Policybazaar/Ditto. Get {fmt_inr(cover_needed)} cover, 30-yr term, online-only.",
                rupee_impact=impact_ins,
                impact_label=fmt_inr(annual_premium) + "/yr premium",
                confidence=0.97,
                urgency="immediate",
            ))

        # ── Sort by rupee impact, take top 5, assign ranks ───────────────────
        recs.sort(key=lambda r: r.rupee_impact, reverse=True)
        for i, rec in enumerate(recs[:5]):
            rec.rank = i + 1
        return recs[:5]
