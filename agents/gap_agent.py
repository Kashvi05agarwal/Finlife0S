"""
agents/gap_agent.py  — Financial Health Score + Gap Detection
"""
from dataclasses import dataclass, field
from config import (
    EMERGENCY_FUND_MONTHS, DEBT_TO_INCOME_MAX,
    SAVINGS_RATE_MIN, INSURANCE_COVER_MULTIPLE,
)
from utils.financial_math import recommended_term_cover, term_premium_estimate, fmt_inr, fmt_pct


@dataclass
class HealthDimension:
    name:      str
    score:     int
    severity:  str
    value:     str
    target:    str
    issue:     str
    fix:       str
    fix_rupee: str = ""


@dataclass
class HealthReport:
    overall_score:  int
    grade:          str
    dimensions:     list = field(default_factory=list)
    critical_count: int  = 0
    moderate_count: int  = 0
    summary:        str  = ""


class GapAgent:
    def run(self, user_data: dict) -> HealthReport:
        income   = user_data["monthly_income"]
        expenses = user_data["monthly_expenses"]
        emi      = user_data.get("monthly_emi", 0)
        savings  = user_data.get("current_savings", 0)
        age      = user_data.get("age", 30)
        invest   = user_data.get("monthly_investment", 0)

        dims = [
            self._emergency_fund(savings, expenses, emi),
            self._debt_health(emi, income),
            self._savings_rate(income, expenses, emi, invest),
            self._retirement_readiness(savings, income, age, invest),
            self._insurance(income, age, savings),
        ]

        critical = sum(1 for d in dims if d.severity == "critical")
        moderate = sum(1 for d in dims if d.severity == "moderate")
        raw      = sum(d.score for d in dims) / len(dims)
        overall  = max(0, min(100, int(raw - critical * 8 - moderate * 3)))

        return HealthReport(
            overall_score=overall,
            grade=self._grade(overall),
            dimensions=dims,
            critical_count=critical,
            moderate_count=moderate,
            summary=self._summary(overall, dims, critical, moderate),
        )

    def projected_score(self, user_data: dict, months: int = 12) -> HealthReport:
        """Simulate health score after N months of following all recommendations."""
        income   = user_data["monthly_income"]
        expenses = user_data["monthly_expenses"]
        emi      = user_data.get("monthly_emi", 0)
        savings  = user_data.get("current_savings", 0)
        invest   = user_data.get("monthly_investment", 0)

        # Model 12 months of improvement:
        # 1. Grow savings by SIP contributions
        # 2. Redirect 6K/mo toward emergency fund
        monthly_ef_build = min(6000, max(0, income - expenses - emi - invest) * 0.5)
        projected_savings = savings + (invest + monthly_ef_build) * months

        # 3. Insurance: assume recommended if annual income > 5L
        ud_proj = {**user_data, "current_savings": projected_savings}
        return self.run(ud_proj)

    # ── Dimension scorers ─────────────────────────────────────────────────

    def _emergency_fund(self, savings, expenses, emi):
        need   = (expenses + emi) * EMERGENCY_FUND_MONTHS
        months = savings / (expenses + emi) if (expenses + emi) > 0 else 0
        if months >= 6:
            return HealthDimension("Emergency Fund", 100, "ok",
                f"{months:.1f} months covered", "6 months",
                "Emergency fund fully covered.", "Review annually.", "")
        elif months >= 3:
            shortfall = need - savings
            return HealthDimension("Emergency Fund", 65, "moderate",
                f"{months:.1f} months ({fmt_inr(savings)})", f"6 months ({fmt_inr(need)})",
                f"Only {months:.1f} months — target is 6.",
                f"Top up by {fmt_inr(shortfall)}.", fmt_inr(shortfall))
        else:
            shortfall = need - savings
            return HealthDimension("Emergency Fund", 25, "critical",
                f"{months:.1f} months ({fmt_inr(savings)})", f"6 months ({fmt_inr(need)})",
                f"Only {months:.1f} months — dangerously low.",
                f"Build to {fmt_inr(need)} urgently (liquid fund).", fmt_inr(shortfall))

    def _debt_health(self, emi, income):
        ratio = emi / income if income > 0 else 0
        if ratio == 0:
            return HealthDimension("Debt Health", 100, "ok",
                "No EMI obligations", f"<35% of income ({fmt_inr(income*0.35)}/mo)",
                "No debt — excellent.", "Stay debt-free.", "")
        elif ratio <= 0.20:
            return HealthDimension("Debt Health", 85, "ok",
                f"EMI {fmt_inr(emi)}/mo ({fmt_pct(ratio)})", f"<35% ({fmt_inr(income*0.35)}/mo)",
                f"EMI is {fmt_pct(ratio)} — healthy.", "Prepay if rate > 8%.", "")
        elif ratio <= 0.35:
            excess = emi - income * 0.20
            return HealthDimension("Debt Health", 55, "moderate",
                f"EMI {fmt_inr(emi)}/mo ({fmt_pct(ratio)})", f"<35% ({fmt_inr(income*0.35)}/mo)",
                f"EMI is {fmt_pct(ratio)} — approaching limit.",
                f"Avoid new loans. Reduce EMI by {fmt_inr(excess)}/mo.", fmt_inr(excess*12))
        else:
            excess = emi - income * 0.20
            return HealthDimension("Debt Health", 15, "critical",
                f"EMI {fmt_inr(emi)}/mo ({fmt_pct(ratio)})", f"<35% ({fmt_inr(income*0.35)}/mo)",
                f"EMI is {fmt_pct(ratio)} — DANGER ZONE.",
                f"Prepay high-interest loan immediately. Target: {fmt_inr(excess)}/mo reduction.",
                fmt_inr(excess*12))

    def _savings_rate(self, income, expenses, emi, invest):
        irate = invest / income if income > 0 else 0
        if irate >= 0.25:
            return HealthDimension("Investment Rate", 100, "ok",
                f"{fmt_inr(invest)}/mo ({fmt_pct(irate)})", f"20%+ ({fmt_inr(income*0.20)}/mo)",
                f"Investing {fmt_pct(irate)} — excellent.", "Keep it up, add 10% step-up.", "")
        elif irate >= 0.15:
            gap = income * 0.25 - invest
            return HealthDimension("Investment Rate", 75, "ok",
                f"{fmt_inr(invest)}/mo ({fmt_pct(irate)})", f"20%+ ({fmt_inr(income*0.20)}/mo)",
                f"Investing {fmt_pct(irate)} — good.",
                f"Stretch to 25% — add {fmt_inr(gap)}/mo.", fmt_inr(gap))
        elif irate >= 0.08:
            gap = income * 0.20 - invest
            return HealthDimension("Investment Rate", 50, "moderate",
                f"{fmt_inr(invest)}/mo ({fmt_pct(irate)})", f"20%+ ({fmt_inr(income*0.20)}/mo)",
                f"Only {fmt_pct(irate)} invested — below 20% benchmark.",
                f"Increase SIP by {fmt_inr(gap)}/mo.", fmt_inr(gap))
        else:
            target = income * 0.20
            return HealthDimension("Investment Rate", 20, "critical",
                f"{fmt_inr(invest)}/mo ({fmt_pct(irate)})", f"20%+ ({fmt_inr(income*0.20)}/mo)",
                f"Only {fmt_pct(irate)} — wealth creation stalled.",
                f"Start SIP of {fmt_inr(target)}/mo immediately.", fmt_inr(target-invest))

    def _retirement_readiness(self, savings, income, age, invest):
        benchmarks = {25:0, 30:1, 35:2, 40:3, 45:4, 50:6, 55:8, 60:10}
        ages = sorted(benchmarks.keys())
        target_x = 0
        for i, a in enumerate(ages):
            if age <= a:
                if i == 0:
                    target_x = benchmarks[a]
                else:
                    prev_a = ages[i-1]
                    frac = (age - prev_a) / (a - prev_a)
                    target_x = benchmarks[prev_a] + frac*(benchmarks[a]-benchmarks[prev_a])
                break
        else:
            target_x = 12

        annual = income * 12
        target = annual * target_x
        ratio  = min(savings / target, 1.5) if target > 0 else 1.0

        if ratio >= 1.0:
            return HealthDimension("Retirement Readiness", 100, "ok",
                f"At {ratio*100:.0f}% of age benchmark", f"{target_x:.0f}x income = {fmt_inr(target)}",
                "On track for retirement.", "Review every 3 years.", "")
        elif ratio >= 0.7:
            shortfall = target - savings
            return HealthDimension("Retirement Readiness", 65, "moderate",
                f"{fmt_inr(savings)} ({ratio*100:.0f}% of benchmark)", f"{target_x:.0f}x = {fmt_inr(target)}",
                f"At {ratio*100:.0f}% of benchmark — some gap.",
                f"Boost SIP by {fmt_inr(shortfall/120)}/mo.", fmt_inr(shortfall))
        elif ratio >= 0.4:
            shortfall = target - savings
            return HealthDimension("Retirement Readiness", 35, "moderate",
                f"{fmt_inr(savings)} ({ratio*100:.0f}% of benchmark)", f"{target_x:.0f}x = {fmt_inr(target)}",
                f"Only {ratio*100:.0f}% of benchmark — retirement at risk.",
                f"Urgently increase equity SIPs. Gap: {fmt_inr(shortfall)}.", fmt_inr(shortfall))
        else:
            shortfall = target - savings
            return HealthDimension("Retirement Readiness", 10, "critical",
                f"{fmt_inr(savings)} ({ratio*100:.0f}% of benchmark)", f"{target_x:.0f}x = {fmt_inr(target)}",
                f"Only {ratio*100:.0f}% of benchmark — critical gap.",
                f"Maximise 80C, NPS + equity SIPs now. Shortfall: {fmt_inr(shortfall)}.", fmt_inr(shortfall))

    def _insurance(self, income, age, savings):
        annual      = income * 12
        recommended = recommended_term_cover(annual, max(1, 60-age))
        annual_prem = term_premium_estimate(recommended, age)
        estimated   = savings * 5

        if savings >= annual * 2:
            return HealthDimension("Insurance Coverage", 80, "minor",
                "Partial cover estimated", f"15x income = {fmt_inr(recommended)}",
                "Insurance status unknown — verify your cover.",
                f"Confirm you hold {fmt_inr(recommended)} term plan. Premium ~{fmt_inr(annual_prem)}/yr.",
                fmt_inr(annual_prem))
        elif estimated < recommended * 0.3:
            return HealthDimension("Insurance Coverage", 20, "critical",
                f"~{fmt_inr(estimated)} estimated cover", f"15x income = {fmt_inr(recommended)}",
                f"Likely under-insured. Need {fmt_inr(recommended)} cover.",
                f"Buy term plan today. Annual premium: ~{fmt_inr(annual_prem)}.",
                fmt_inr(annual_prem))
        else:
            return HealthDimension("Insurance Coverage", 55, "moderate",
                f"Partial cover (~{fmt_inr(estimated)})", f"15x income = {fmt_inr(recommended)}",
                "Partial cover — verify and top up.",
                f"Top up to {fmt_inr(recommended)}. Premium ~{fmt_inr(annual_prem)}/yr.",
                fmt_inr(annual_prem))

    def _grade(self, score):
        if score >= 85: return "A"
        if score >= 70: return "B"
        if score >= 55: return "C"
        if score >= 40: return "D"
        return "F"

    def _summary(self, score, dims, critical, moderate):
        if critical >= 2:
            return f"⚠️ {critical} critical issues need fixing before anything else."
        if critical == 1:
            c = next(d for d in dims if d.severity == "critical")
            return f"🔴 Fix this first: {c.name}. Everything else can wait."
        if moderate >= 2:
            return f"🟡 Good base — {moderate} gaps are silently costing you wealth."
        if score >= 80:
            return "✅ Strong foundation. Focus on growing and optimising."
        return "🟢 On track — keep building consistency."