"""
utils/financial_math.py
Core deterministic financial formulas used across all agents.
"""
import math
import numpy as np


# ── SIP / Future Value ──────────────────────────────────────────────────────

def sip_future_value(monthly_sip: float, annual_rate: float, years: int) -> float:
    """Future value of a monthly SIP at a given annual rate."""
    if years <= 0 or monthly_sip <= 0:
        return 0.0
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return monthly_sip * n
    return monthly_sip * ((math.pow(1 + r, n) - 1) / r) * (1 + r)


def sip_required(target_corpus: float, annual_rate: float, years: int) -> float:
    """Monthly SIP needed to reach a target corpus."""
    if years <= 0 or target_corpus <= 0:
        return 0.0
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return target_corpus / n if n > 0 else 0
    return target_corpus * r / (((1 + r) ** n - 1) * (1 + r))


def lumpsum_future_value(principal: float, annual_rate: float, years: int) -> float:
    """Future value of a lump sum investment."""
    if years <= 0:
        return principal
    return principal * math.pow(1 + annual_rate, years)


# ── Inflation ───────────────────────────────────────────────────────────────

def inflate(amount: float, inflation_rate: float, years: int) -> float:
    """Inflate an amount to future value."""
    return amount * math.pow(1 + inflation_rate, years)


def deflate(future_amount: float, inflation_rate: float, years: int) -> float:
    """Bring a future amount to present value."""
    if years <= 0:
        return future_amount
    return future_amount / math.pow(1 + inflation_rate, years)


# ── FIRE / Retirement ───────────────────────────────────────────────────────

def fire_corpus(annual_expenses_today: float, inflation_rate: float,
                years_to_retire: int, withdrawal_rate: float = 0.04) -> float:
    """
    FIRE corpus needed (4% rule, inflation-adjusted).
    Returns corpus needed AT retirement date (in future rupees).
    """
    future_expenses = inflate(annual_expenses_today, inflation_rate, years_to_retire)
    return future_expenses / withdrawal_rate


def years_to_corpus(
    current_corpus: float,
    monthly_sip: float,
    target_corpus: float,
    annual_rate: float,
    max_years: int = 40,
) -> float:
    """Binary search: how many years to reach a target corpus at given SIP + rate."""
    if current_corpus >= target_corpus:
        return 0.0
    for y in range(1, max_years + 1):
        fv = (lumpsum_future_value(current_corpus, annual_rate, y)
              + sip_future_value(monthly_sip, annual_rate, y))
        if fv >= target_corpus:
            return float(y)
    return float(max_years + 1)   # won't reach in time


# ── Debt ────────────────────────────────────────────────────────────────────

def emi_calculator(principal: float, annual_rate: float, tenure_months: int) -> float:
    """Standard EMI formula."""
    if tenure_months <= 0:
        return 0.0
    if annual_rate == 0:
        return principal / tenure_months
    r = annual_rate / 12
    return principal * r * math.pow(1 + r, tenure_months) / (math.pow(1 + r, tenure_months) - 1)


def loan_outstanding(principal: float, annual_rate: float,
                     tenure_months: int, paid_months: int) -> float:
    """Remaining principal after `paid_months` payments."""
    if tenure_months <= 0:
        return principal
    if annual_rate == 0:
        payment = principal / tenure_months
        remaining = principal - payment * min(paid_months, tenure_months)
        return max(0.0, remaining)
    emi = emi_calculator(principal, annual_rate, tenure_months)
    r = annual_rate / 12
    return (principal * math.pow(1 + r, paid_months)
            - emi * (math.pow(1 + r, paid_months) - 1) / r)


# ── Tax ─────────────────────────────────────────────────────────────────────

def tax_liability_new_regime(annual_income: float) -> float:
    """
    Updated New Tax Regime (FY 2025-26 / AY 2026-27)
    Includes Section 87A rebate (₹60,000 max, applies if taxable income ≤ ₹12L).
    """

    # Standard deduction for salaried taxpayers
    STANDARD_DEDUCTION = 75000
    taxable_income = max(0, annual_income - STANDARD_DEDUCTION)

    slabs = [
        (300_000, 0.00),
        (300_000, 0.05),
        (300_000, 0.10),
        (300_000, 0.15),
        (300_000, 0.20),
        (float('inf'), 0.30),
    ]

    tax = 0.0
    remaining = taxable_income
    for limit, rate in slabs:
        chunk = min(remaining, limit)
        tax += chunk * rate
        remaining -= chunk
        if remaining <= 0:
            break

    if taxable_income <= 1_200_000:
        tax = 0.0
    else:
        tax *= 1.04

    return max(0.0, tax)


def get_new_regime_breakdown(annual_income: float) -> dict:
    """Return a slab-by-slab breakdown for the new regime calculation."""
    STANDARD_DEDUCTION = 75000
    taxable_income = max(0, annual_income - STANDARD_DEDUCTION)

    slabs = [
        (300_000, 0.00, "₹0–3L"),
        (300_000, 0.05, "₹3–6L"),
        (300_000, 0.10, "₹6–9L"),
        (300_000, 0.15, "₹9–12L"),
        (300_000, 0.20, "₹12–15L"),
        (float('inf'), 0.30, "₹15L+"),
    ]

    remaining = taxable_income
    breakdown = []
    base_tax = 0.0

    for limit, rate, label in slabs:
        if remaining <= 0:
            break
        chunk = min(remaining, limit)
        slab_tax = chunk * rate
        breakdown.append({
            "slab": label,
            "amount": chunk,
            "rate": rate,
            "tax": slab_tax,
        })
        base_tax += slab_tax
        remaining -= chunk

    rebate = base_tax if taxable_income <= 1_200_000 else 0.0
    post_rebate_tax = max(0.0, base_tax - rebate)
    cess = post_rebate_tax * 0.04
    total_tax = post_rebate_tax + cess

    return {
        "annual_income": annual_income,
        "taxable_income": taxable_income,
        "breakdown": breakdown,
        "base_tax": base_tax,
        "rebate": rebate,
        "cess": cess,
        "total_tax": total_tax,
    }


def tax_liability_old_regime(annual_income: float, deductions_80c: float = 150_000,
                              deductions_other: float = 50_000) -> float:
    """Old Indian tax regime with standard deductions."""
    taxable = max(0, annual_income - deductions_80c - deductions_other - 50_000)
    slabs = [
        (250_000, 0.00),
        (250_000, 0.05),
        (500_000, 0.20),
        (float('inf'), 0.30),
    ]
    tax = 0.0
    remaining = max(0, taxable - 250_000)
    for limit, rate in slabs:
        chunk = min(remaining, limit)
        tax += chunk * rate
        remaining -= chunk
        if remaining <= 0:
            break
    return tax * 1.04


# ── Opportunity Cost ────────────────────────────────────────────────────────

def opportunity_cost(amount: float, annual_rate: float, years: int) -> float:
    """What amount would grow to if invested instead of spent."""
    return lumpsum_future_value(amount, annual_rate, years) - amount


def idle_cash_loss(monthly_idle: float, annual_rate: float, years: int) -> float:
    """Loss from keeping money in savings account vs investing."""
    sa_rate = 0.035   # typical savings account rate
    invested_fv = sip_future_value(monthly_idle, annual_rate, years)
    idle_fv     = sip_future_value(monthly_idle, sa_rate,     years)
    return invested_fv - idle_fv


# ── Insurance ───────────────────────────────────────────────────────────────

def recommended_term_cover(annual_income: float, years_to_retire: int,
                            outstanding_loans: float = 0) -> float:
    """Recommended life cover = 15x income + outstanding loans."""
    return annual_income * 15 + outstanding_loans


def term_premium_estimate(cover_amount: float, age: int) -> float:
    """Rough annual premium estimate for term insurance."""
    base_rate = 0.0006  # 0.06% of cover (conservative)
    age_factor = 1 + max(0, (age - 25) * 0.04)
    return cover_amount * base_rate * age_factor


# ── Formatting helpers ──────────────────────────────────────────────────────

def fmt_inr(amount: float) -> str:
    """Format number as Indian currency string."""
    amount = abs(amount)
    if amount >= 1e7:
        return f"₹{amount/1e7:.2f}Cr"
    if amount >= 1e5:
        return f"₹{amount/1e5:.1f}L"
    if amount >= 1e3:
        return f"₹{amount/1e3:.0f}K"
    return f"₹{amount:.0f}"


def fmt_pct(value: float) -> str:
    return f"{value*100:.1f}%"
