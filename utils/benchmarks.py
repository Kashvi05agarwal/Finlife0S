"""
utils/benchmarks.py
Peer comparison data by income bracket.
"""
from config import PEER_BENCHMARKS
from utils.financial_math import fmt_inr, fmt_pct


def get_peer_bracket(monthly_income: float) -> dict:
    """Return benchmark stats for a given income bracket."""
    for (lo, hi), stats in PEER_BENCHMARKS.items():
        if lo <= monthly_income < hi:
            return {"bracket": (lo, hi), **stats}
    # fallback: highest bracket
    k = list(PEER_BENCHMARKS.keys())[-1]
    return {"bracket": k, **PEER_BENCHMARKS[k]}


def peer_comparison(user_data: dict) -> dict:
    """
    Compare user metrics to peers in same income bracket.
    Returns dict with gap analysis for each dimension.
    """
    income   = user_data["monthly_income"]
    expense  = user_data["monthly_expenses"]
    emi      = user_data.get("monthly_emi", 0)
    savings  = user_data.get("current_savings", 0)
    monthly_invest = user_data.get("monthly_investment", 0)

    peer = get_peer_bracket(income)

    surplus       = income - expense - emi
    savings_rate  = surplus / income if income > 0 else 0
    sip_pct       = monthly_invest / income if income > 0 else 0
    monthly_need  = expense + emi
    emergency_m   = savings / monthly_need if monthly_need > 0 else 0
    # Cap display at 12 months — anything above is "excellent" not "444%"
    emergency_m   = min(emergency_m, 12.0)

    peer_savings_rate = peer["savings_rate"]
    peer_sip_pct      = peer["sip_pct"]
    peer_emergency    = peer["emergency_months"]

    lo, hi = peer["bracket"]
    bracket_label = (
        f"{fmt_inr(lo*12)}–{fmt_inr(hi*12)}/yr"
        if hi < 1e8 else f">{fmt_inr(lo*12)}/yr"
    )

    return {
        "bracket_label": bracket_label,
        "metrics": {
            "savings_rate": {
                "user":  savings_rate,
                "peer":  peer_savings_rate,
                "gap":   savings_rate - peer_savings_rate,
                "label": "Savings Rate",
            },
            "sip_rate": {
                "user":  sip_pct,
                "peer":  peer_sip_pct,
                "gap":   sip_pct - peer_sip_pct,
                "label": "Investment Rate (% income)",
            },
            "emergency_months": {
                "user":  emergency_m,
                "peer":  peer_emergency,
                "gap":   emergency_m - peer_emergency,
                "label": "Emergency Fund (months)",
            },
        },
    }