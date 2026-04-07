# ============================================================
#  FinLife OS — Central Configuration
# ============================================================

# Monte Carlo Parameters
MC_SIMULATIONS = 300
MC_HORIZON_YEARS = 30
MC_SEED = 42

# Return Assumptions (annual)
RETURN_MEAN = 0.12        # 12% equity CAGR
RETURN_STD  = 0.08        # ±8% std dev
INFLATION_MEAN = 0.06     # 6% inflation
INFLATION_STD  = 0.015    # ±1.5%
INCOME_GROWTH_MEAN = 0.08 # 8% salary growth
INCOME_GROWTH_STD  = 0.02 # ±2%

# Risk-Profile return adjustments
RISK_RETURN = {
    "conservative": {"mean": 0.085, "std": 0.05},
    "moderate":     {"mean": 0.11,  "std": 0.07},
    "aggressive":   {"mean": 0.13,  "std": 0.10},
}

# Financial Health Thresholds
EMERGENCY_FUND_MONTHS   = 6     # Target months of expenses
DEBT_TO_INCOME_MAX      = 0.35  # Max EMI/income ratio
SAVINGS_RATE_MIN        = 0.20  # Min savings rate
INSURANCE_COVER_MULTIPLE = 15   # 15x annual income term cover

# Goal inflation rates
GOAL_INFLATION = {
    "house":      0.08,
    "retirement": 0.06,
    "child":      0.10,
    "marriage":   0.07,
    "vehicle":    0.05,
    "emergency":  0.06,
    "travel":     0.07,
    "startup":    0.06,
}

# Goal default targets (today's value ₹)
GOAL_TARGETS_TODAY = {
    "house":      1_500_000,
    "retirement": None,     # computed from corpus formula
    "child":      5_000_000,
    "marriage":   1_200_000,
    "vehicle":    800_000,
    "emergency":  None,     # computed from expenses
    "travel":     200_000,
    "startup":    3_000_000,
}

# Peer benchmarks by income bracket (monthly ₹)
PEER_BENCHMARKS = {
    (0, 30_000):       {"savings_rate": 0.08, "emergency_months": 2, "sip_pct": 0.05},
    (30_000, 60_000):  {"savings_rate": 0.15, "emergency_months": 3, "sip_pct": 0.10},
    (60_000, 100_000): {"savings_rate": 0.22, "emergency_months": 4, "sip_pct": 0.15},
    (100_000, 200_000):{"savings_rate": 0.28, "emergency_months": 5, "sip_pct": 0.20},
    (200_000, 10**9):  {"savings_rate": 0.35, "emergency_months": 6, "sip_pct": 0.25},
}

# Life Shock Presets
SHOCK_PRESETS = {
    "job_loss_6m":    {"income_factor": 0.0,  "duration_months": 6,  "one_time_cost": 0},
    "income_drop_30": {"income_factor": 0.70, "duration_months": 24, "one_time_cost": 0},
    "medical_8L":     {"income_factor": 1.0,  "duration_months": 0,  "one_time_cost": 800_000},
    "marriage_15L":   {"income_factor": 1.0,  "duration_months": 0,  "one_time_cost": 1_500_000},
    "career_switch":  {"income_factor": 0.70, "duration_months": 24, "one_time_cost": 0},
}

# Recommendation impact multipliers
REC_IMPACT_HORIZON = 20   # years for opportunity cost calc
