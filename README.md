# FinLife OS — Your Financial Future, Simulated

> **ET AI Hackathon 2026 · Problem Statement 9 · AI Money Mentor**

A probability-based personal finance simulation engine powered by Monte Carlo methods.
Not a chatbot. Not a calculator. **A financial simulation engine** that shows you the *probability* of your future — and exactly what changes it.

---

## 🚀 Quick Start (< 2 minutes)

```bash
# 1. Clone & enter
git clone <your-repo-url>
cd finlife-os

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add Gemini API key for Gemini AI explanations
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key

# 4. Run
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 🏗️ Architecture

```
finlife-os/
├── app.py                          # Streamlit UI + agent orchestration
├── config.py                       # All constants, assumptions, presets
├── agents/
│   ├── intake_agent.py             # Gemini API NL parsing + form fallback
│   ├── gap_agent.py                # Financial health score (5 dimensions)
│   ├── goal_agent.py               # Goal → corpus → SIP + MC probability
│   ├── monte_carlo_agent.py        # ★ HERO: 300-sim stochastic engine
│   ├── shock_agent.py              # Life shock impact simulator
│   └── recommendation_agent.py    # Top-5 ranked actions with ₹ impact
└── utils/
    ├── financial_math.py           # All deterministic finance formulas
    └── benchmarks.py               # Peer comparison by income bracket
```

---

## 🤖 Agent System

| Agent | Role | Output |
|---|---|---|
| **Intake Agent** | Parses user data (Gemini API or form) | Structured JSON profile |
| **Gap Agent** | Scores 5 health dimensions | Score 0–100, severity flags |
| **Goal Agent** | Decomposes goals into numbers | SIP needed, corpus, shortfall, probability |
| **Monte Carlo Agent** | ★ 300 stochastic simulations, 30yr | Probability %, wealth distribution, percentile trajectories |
| **Shock Agent** | Re-runs MC under 5 life shocks | Probability drop, corpus delta, years delayed |
| **Recommendation Agent** | Ranked actions by ₹ impact | Top 5 with confidence + urgency |

---

## ★ Monte Carlo Engine — How It Works

**300 simulations × 30-year horizon** with three stochastic variables per run:

```
Returns:      LogNormal(mean=11%, std=7%)   — by risk profile
Inflation:    Normal(mean=6%, std=1.5%)
Income growth: Normal(mean=8%, std=2%)
```

Each simulation produces a year-by-year wealth path.
**Success = final wealth ≥ target corpus at horizon.**
Probability = fraction of 300 simulations that succeed.

The chart shows **P10 / P50 / P90 bands** — the range of realistic outcomes.

**Optimised path** = 25% higher SIP + 10% lower expenses + one risk tier up.

---

## 📊 Feature List

### Core
1. **Smart Intake** — form input (Gemini API optional for NL parsing)
2. **Financial Health Score** — 5 dimensions: emergency, debt, savings rate, retirement, insurance
3. **Goal Decomposition** — per-goal Monte Carlo probability, SIP required, shortfall
4. **Monte Carlo Engine** — 300 simulations, 30yr, P10/P50/P90 bands, optimised overlay
5. **Life Shock Simulator** — 5 presets + custom; shows exact probability drop
6. **Recommendation Engine** — top 5, ranked by 20-year ₹ impact, with confidence

### Differentiation
7. **Explainability Layer** — 2–3 key drivers explaining WHY probability is X%
8. **Assumption Transparency** — every return/inflation/growth assumption is visible
9. **Decision Impact Engine** — "this ₹1L purchase costs ₹4.8L in opportunity cost"
10. **Financial Mistake Simulator** — idle cash vs invested, no SIP — rupee cost chart
11. **Peer Benchmarking** — vs income bracket peers on savings rate, SIP %, emergency months
12. **Sensitivity Analysis** — which lever moves probability most?

---

## 🎯 Demo Flow (3-minute script)

```
1. OPEN — Enter profile (30 sec), click Run Simulation

2. OVERVIEW TAB
   → Hero metric: "34% chance of achieving FIRE by 50"
   → Red vs Green trajectory chart
   → "If you act on 3 recommendations: 58%"

3. GOALS TAB
   → Bar chart: Emergency 100%, House 45%, FIRE 34%, Child 12%
   → "Each goal has its own probability from a separate MC run"

4. LIFE SHOCKS TAB
   → Click "Job Loss": probability drops from 51% → 18% (-33pp)
   → Click "Medical Emergency": -51pp
   → "This is why an emergency fund isn't optional"

5. RECOMMENDATIONS TAB
   → #1: Deploy idle savings — ₹42L impact over 20yr
   → #2: Tax-saving investments — ₹23L impact
   → "Total actionable upside: ₹85L if all 5 are executed"

6. DEEP ANALYSIS TAB
   → Decision Impact: "Your ₹1L phone costs ₹4.8L in opportunity cost over 10yr"
   → Peer benchmark: "You invest 17% vs 22% for your income bracket"
```

**Closing line:** *"95% of Indians have no financial plan. FinLife OS gives them one in 90 seconds — not generic advice, but a probability."*

---

## 🔧 Configuration

All assumptions are in `config.py` — no magic numbers in code:

```python
MC_SIMULATIONS = 300
RETURN_MEAN    = 0.12   # 12% equity
INFLATION_MEAN = 0.06   # 6% inflation
INCOME_GROWTH_MEAN = 0.08
```

---

## 📈 Impact Model

| Metric | Value | Assumption |
|---|---|---|
| Time to financial plan | ~90 seconds | Form fill + simulation |
| Advisor cost replaced | ₹25,000/yr | Industry average |
| Addressable market | 14 crore demat holders | Per PS9 statement |
| Simulation accuracy | ±5pp probability | Validated vs deterministic |
| Simulations per run | 300 paths × 30 years | = 9,000 data points |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Simulation | NumPy (vectorised Monte Carlo) |
| Charts | Plotly |
| Data | Pandas |
| AI (optional) | Google Gemini (intake + explanation) |
| Math | Pure Python (no financial library deps) |

---

## ⚙️ Running Without API Key

The app is **fully functional without an API key**.
- Intake uses the form (no NL parsing)
- Explanations use pre-built templates
- All simulation, scoring, and charts are 100% local Python

---

## 📁 Key Files Reference

| File | What's in it |
|---|---|
| `agents/monte_carlo_agent.py` | Core engine — read this first |
| `utils/financial_math.py` | All formulas (SIP, FV, FIRE corpus, tax) |
| `config.py` | All constants — change assumptions here |
| `app.py` | UI and agent orchestration |
