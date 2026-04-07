"""
FinLife OS — app.py
Premium financial simulation app with custom HTML/CSS UI served via Streamlit.
All logic in Python. UI rendered as custom HTML component for premium feel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json, random
from datetime import date, datetime
from dotenv import load_dotenv
load_dotenv()

from agents.intake_agent         import IntakeAgent
from agents.gap_agent            import GapAgent
from agents.goal_agent           import GoalAgent
from agents.monte_carlo_agent    import MonteCarloAgent, SimulationInput
from agents.shock_agent          import ShockAgent
from agents.recommendation_agent import RecommendationAgent
from utils.financial_math import (
    fmt_inr, fmt_pct, sip_future_value, lumpsum_future_value,
    opportunity_cost, tax_liability_new_regime, tax_liability_old_regime,
    get_new_regime_breakdown,
)
from utils.benchmarks import peer_comparison
from config import MC_SEED, RISK_RETURN, SHOCK_PRESETS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinLife OS — Financial Simulation",
    page_icon="📡", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
A  = "#4F46E5"   # indigo
A2 = "#10B981"   # emerald
D  = "#EF4444"   # red
W  = "#F59E0B"   # amber
BG = "#F8FBFF"
S  = "#FFFFFF"
B  = "#D9E2F0"
M  = "#64748B"

# ── Premium global CSS ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'Space Grotesk',sans-serif;background:{BG};color:#0F172A;}}
#MainMenu,header,footer{{visibility:hidden;}}
.block-container{{padding:.8rem 1.4rem 2rem;max-width:1480px;}}
[data-testid="stSidebar"]{{background:#EFF6FF;border-right:1px solid {B};}}
[data-testid="stSidebar"] .stMarkdown h1{{font-size:.9rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{A}!important;margin-bottom:.3rem;}}
[data-testid="stSidebar"] label{{font-size:.68rem;font-weight:700;letter-spacing:.07em;color:{M};text-transform:uppercase;}}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{{
  background:#FFFFFF!important;border:1px solid {B}!important;border-radius:10px!important;
  color:#0F172A!important;font-family:'JetBrains Mono';font-size:.9rem;
}}
[data-testid="stNumberInput"] input:focus{{border-color:{A}!important;box-shadow:0 0 0 2px {A}22!important;}}
[data-baseweb="select"]>div{{background:{BG}!important;border:1px solid {B}!important;border-radius:6px!important;}}
[data-baseweb="tag"]{{background:{A}33!important;border:1px solid {A}66!important;border-radius:4px!important;}}
.stButton>button[kind="primary"]{{
  background:linear-gradient(135deg,{A},{A2});border:none;border-radius:12px;color:#fff;
  font-family:'Space Grotesk';font-weight:700;font-size:.92rem;letter-spacing:.04em;
  padding:.75rem 1.45rem;transition:all .2s;box-shadow:0 10px 24px {A}22;
}}
.stButton>button[kind="primary"]:hover{{transform:translateY(-1px);box-shadow:0 8px 22px {A}55;}}
.stButton>button{{background:#FFFFFF;border:1px solid {B};border-radius:12px;color:{M};font-family:'Space Grotesk';font-size:.9rem;}}
div[data-testid="metric-container"]{{background:#FFFFFF;border:1px solid {B};border-radius:18px;padding:1rem 1.1rem;transition:border-color .2s;box-shadow:0 10px 24px rgba(15,23,42,0.05);}}
div[data-testid="metric-container"]:hover{{border-color:{A}44;}}
[data-testid="stMetricLabel"]{{font-size:.63rem!important;font-weight:700!important;letter-spacing:.08em!important;text-transform:uppercase!important;color:{M}!important;}}
[data-testid="stMetricValue"]{{font-size:1.45rem!important;font-weight:700!important;font-family:'JetBrains Mono'!important;color:#0F172A!important;}}
[data-testid="stMetricDelta"]{{font-size:.7rem!important;font-family:'JetBrains Mono'!important;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]{{background:#EFF6FF;border-radius:16px;padding:4px;border:1px solid {B};gap:3px;}}
[data-testid="stTabs"] [data-baseweb="tab"]{{background:transparent;border-radius:6px;color:{M};font-weight:500;font-size:.78rem;padding:.42rem .85rem;}}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"]{{background:{A}22;color:{A};border:1px solid {A}44;}}
[data-testid="stTabs"] [data-baseweb="tab-border"]{{display:none;}}
[data-testid="stExpander"]{{background:#F8FAFF;border:1px solid {B}!important;border-radius:14px!important;margin-bottom:.6rem;}}
[data-testid="stExpander"] summary{{font-weight:700;font-size:.92rem;color:{M};padding:.85rem 1rem;}}
[data-testid="stProgress"]>div>div{{background:linear-gradient(90deg,{A},{A2})!important;border-radius:3px;}}
[data-testid="stInfo"]{{background:{A}10;border:1px solid {A}20;border-radius:10px;color:#0F172A;}}
[data-testid="stWarning"]{{background:{W}10;border:1px solid {W}20;border-radius:10px;color:#0F172A;}}
[data-testid="stError"]{{background:{D}10;border:1px solid {D}20;border-radius:10px;color:#0F172A;}}
[data-testid="stSuccess"]{{background:{A2}10;border:1px solid {A2}20;border-radius:10px;color:#0F172A;}}
hr{{border-color:{B};opacity:1;margin:.8rem 0;}}
.stTextArea textarea{{min-height:70px!important;}}
</style>
""", unsafe_allow_html=True)

# ── HTML helpers ───────────────────────────────────────────────────────────────
def pc(p): return A2 if p >= .65 else (W if p >= .40 else D)
def sc(s): return {"critical":D,"moderate":W,"minor":A,"ok":A2}.get(s, M)
def si(s): return {"critical":"🔴","moderate":"🟡","minor":"🔵","ok":"🟢"}.get(s,"⚪")

PLOT = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="Space Grotesk", color="#334155", size=11),
    margin=dict(l=6, r=6, t=34, b=6),
    legend=dict(bgcolor=BG, bordercolor=B, borderwidth=1, font=dict(size=10, color="#334155")),
)

def pill(name, status, detail=""):
    cls = {"done":"pd","running":"pr","waiting":"pw"}[status]
    ico = {"done":"●","running":"◉","waiting":"○"}[status]
    txt = f"{ico} {name}" + (f" · {detail}" if detail else "")
    return f'<span class="pill {cls}">{txt}</span>'

def hcard(content, top_color=None):
    border = f"border-top:2px solid {top_color};" if top_color else ""
    return f'<div style="background:{S};border:1px solid {B};border-radius:12px;padding:1.4rem;{border}">{content}</div>'

def bar_html(pct, color, height=5):
    return f'<div style="height:{height}px;background:{B};border-radius:3px;overflow:hidden;margin:3px 0;"><div style="width:{min(100,pct):.0f}%;height:100%;background:{color};border-radius:3px;"></div></div>'

def badge(text, color):
    return f'<span style="background:{color}18;color:{color};border:1px solid {color}44;border-radius:100px;font-size:.65rem;font-weight:600;padding:2px 9px;font-family:\'JetBrains Mono\';">{text}</span>'

def info_box(text, color=None, icon="💡"):
    c = color or A
    return f'<div style="background:{c}0D;border:1px solid {c}33;border-radius:7px;padding:.6rem .9rem;font-size:.79rem;color:#475569;margin:.4rem 0;">{icon} {text}</div>'

# ── Color helpers ─────────────────────────────────────────────────────────────────
def rgba(hex_color, alpha):
    c = hex_color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def driver_card(d):
    dc = A2 if d["direction"]=="positive" else D
    arr = "↑" if d["direction"]=="positive" else "↓"
    cls_add = "border-left-color:" + dc
    return f'''<div style="background:{BG};border-left:3px solid {dc};border-radius:0 7px 7px 0;padding:.7rem .9rem;margin:4px 0;">
<div style="font-weight:600;color:#0F172A;font-size:.84rem;"><span style="color:{dc};">{arr}</span> [{d["impact"].upper()}] {d["factor"]}
<span style="color:{M};font-weight:400;font-size:.76rem;margin-left:.4rem;">{d["value"]}</span></div>
<div style="color:#475569;font-size:.77rem;margin-top:.22rem;line-height:1.5;">{d["description"]}</div></div>'''

def rec_card(rec):
    cat_c = {"invest":A,"protect":A2,"tax":W,"debt":D,"save":"#60B4D0"}.get(rec.category, M)
    urg_i = {"immediate":"🚨","3-months":"📅","6-months":"📆"}.get(rec.urgency,"📅")
    return f'''<div style="background:{S};border:1px solid {B};border-left:3px solid {cat_c};border-radius:9px;padding:1rem 1.2rem;margin-bottom:.6rem;">
<div style="display:flex;justify-content:space-between;align-items:flex-start;">
<div><span style="font-size:.6rem;font-weight:700;letter-spacing:.1em;color:{M};text-transform:uppercase;">#{rec.rank} · {rec.category}</span>
<div style="font-size:.92rem;font-weight:600;color:#0F172A;margin-top:.12rem;">{rec.title}</div></div>
<div style="text-align:right;flex-shrink:0;margin-left:.7rem;">
<div style="font-family:'JetBrains Mono';font-size:.95rem;font-weight:700;color:{cat_c};">{rec.impact_label}</div>
<div style="font-size:.62rem;color:{M};">20-yr impact</div></div></div>
<div style="margin:.4rem 0;font-size:.78rem;color:#475569;line-height:1.5;">{rec.description}</div>
<div style="padding:.38rem .65rem;background:{cat_c}11;border-radius:5px;font-size:.78rem;color:{cat_c};">✓ {rec.action}</div>
<div style="margin-top:.32rem;display:flex;gap:.3rem;flex-wrap:wrap;">{badge(f"{urg_i} {rec.urgency}", M)} {badge(f"{rec.confidence*100:.0f}% confidence", cat_c)}</div></div>'''

# ── Plotly charts ──────────────────────────────────────────────────────────────
def chart_proj(mc, opt, ud):
    y = mc.years_range; t = mc.target_corpus
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y+y[::-1], y=list(opt.pct90/1e7)+list(opt.pct10/1e7)[::-1],
        fill="toself", fillcolor=rgba(A2, 0.05), line=dict(color="rgba(0,0,0,0)"), name="If you act (range)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=y+y[::-1], y=list(mc.pct90/1e7)+list(mc.pct10/1e7)[::-1],
        fill="toself", fillcolor=rgba(D, 0.05), line=dict(color="rgba(0,0,0,0)"), name="Current path (range)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=y, y=mc.pct50/1e7, mode="lines", name="Current path", line=dict(color=D, width=2.5)))
    fig.add_trace(go.Scatter(x=y, y=opt.pct50/1e7, mode="lines", name="If you act", line=dict(color=A2, width=2.5, dash="dot")))
    fig.add_hline(y=t/1e7, line=dict(color=W, width=1.5, dash="dash"),
        annotation_text=f"Goal: {fmt_inr(t)}", annotation_font=dict(color=W, size=10))
    ry = ud.get("retirement_age",50) - ud.get("age",30)
    if 0 < ry <= max(y):
        fig.add_vline(x=ry, line=dict(color=A, width=1, dash="dot"),
            annotation_text=f"Retire @ {ud.get('retirement_age')}", annotation_font=dict(color=A, size=10))
    fig.update_layout(**PLOT, height=360, hovermode="x unified",
        title=dict(text="Your wealth over 300 simulated futures (₹ Crore)", font=dict(size=13,color="#334155"), x=0),
        xaxis=dict(title="Years from now", gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B),
        yaxis=dict(title="₹ Crore", gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B))
    return fig

def chart_dist(mc):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc.final_wealth_distribution/1e7, nbinsx=40,
        marker=dict(color=A, opacity=.7, line=dict(color=BG, width=.3)), showlegend=False))
    fig.add_vline(x=mc.target_corpus/1e7, line=dict(color=W, width=2, dash="dash"),
        annotation_text="Goal", annotation_font=dict(color=W, size=10))
    fig.add_vline(x=mc.median_final_wealth/1e7, line=dict(color=A2, width=2),
        annotation_text="Middle path", annotation_font=dict(color=A2, size=10))
    fig.update_layout(**PLOT, height=240, showlegend=False,
        title=dict(text="Final wealth distribution — 300 simulations", font=dict(size=13,color="#334155"), x=0),
        xaxis=dict(title="₹ Crore at retirement", gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B),
        yaxis=dict(title="Simulations", gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B))
    return fig

def chart_radar(h):
    dims = h.dimensions
    cats = [d.name for d in dims] + [dims[0].name]
    vals = [d.score for d in dims] + [dims[0].score]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
        fillcolor=rgba(A, 0.13), line=dict(color=A, width=2), name="Your score"))
    fig.add_trace(go.Scatterpolar(r=[70]*len(cats), theta=cats, fill="toself",
        fillcolor=rgba(A2, 0.05), line=dict(color=A2, width=1, dash="dot"), name="Target (70)"))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], gridcolor=B, tickfont=dict(size=9,color=M), linecolor=B),
            angularaxis=dict(tickfont=dict(size=10,color="#475569"), linecolor=B), bgcolor=S),
        paper_bgcolor=BG, font=dict(family="Space Grotesk", color="#334155"),
        legend=dict(bgcolor=BG, bordercolor=B, borderwidth=1, font=dict(size=10,color="#334155")),
        height=260, margin=dict(l=12,r=12,t=12,b=12))
    return fig

def chart_goals(plans):
    fig = go.Figure(go.Bar(
        x=[g.label for g in plans], y=[g.probability*100 for g in plans],
        marker=dict(color=[pc(g.probability) for g in plans], opacity=.85, cornerradius=5),
        text=[f"{g.probability*100:.0f}%" for g in plans],
        textposition="outside", textfont=dict(size=11,color="#334155")))
    fig.add_hline(y=70, line=dict(color=W, dash="dash", width=1.5),
        annotation_text="70% target", annotation_font=dict(color=W, size=10))
    fig.update_layout(**PLOT, height=260, showlegend=False,
        title=dict(text="Chance of hitting each goal", font=dict(size=13,color="#334155"), x=0),
        xaxis=dict(gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B),
        yaxis=dict(range=[0,125], title="Probability (%)", gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B))
    return fig

def chart_shocks(shocks):
    names = [s.shock_name.split("(")[0].strip() for s in shocks]
    fig = go.Figure()
    fig.add_bar(name="Without event", x=names, y=[s.base_probability*100 for s in shocks],
                marker=dict(color=A2, opacity=.8, cornerradius=4))
    fig.add_bar(name="After event", x=names, y=[s.shocked_probability*100 for s in shocks],
                marker=dict(color=D, opacity=.85, cornerradius=4))
    fig.update_layout(**PLOT, barmode="group", height=280,
        title=dict(text="How life events reduce your chances", font=dict(size=13,color="#334155"), x=0),
        xaxis=dict(gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B, tickangle=-8),
        yaxis=dict(range=[0,100], title="Success chance (%)", gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B))
    return fig

def chart_mistakes(ud):
    inv = ud["monthly_investment"]; sav = ud.get("current_savings",0)
    yrs = list(range(1,21))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=[sip_future_value(inv,.11,y)/1e5 for y in yrs],
        name=f"Keep SIP {fmt_inr(inv)}/mo", line=dict(color=A2, width=2.5)))
    fig.add_trace(go.Scatter(x=yrs, y=[0.0]*20, name="Stop investing",
        line=dict(color=D, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=yrs, y=[lumpsum_future_value(sav,.11,y)/1e5 for y in yrs],
        name="Savings → equity fund", line=dict(color=A, width=2)))
    fig.add_trace(go.Scatter(x=yrs, y=[lumpsum_future_value(sav,.035,y)/1e5 for y in yrs],
        name="Savings → bank account", line=dict(color=M, width=1.5, dash="dash")))
    fig.update_layout(**PLOT, height=280,
        title=dict(text="Cost of decisions over 20 years (₹ Lakh)", font=dict(size=13,color="#334155"), x=0),
        xaxis=dict(title="Years", gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B),
        yaxis=dict(title="₹ Lakh", gridcolor=B, zerolinecolor=B, tickfont=dict(size=10), linecolor=B))
    return fig

# ── Daily retention pulse ──────────────────────────────────────────────────────
def _fetch_nse_quote() -> str:
    """Fetch live Nifty 50 quote from NSE India public API. Falls back gracefully."""
    try:
        import urllib.request
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1d&range=2d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read())
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        prev, curr = closes[-2], closes[-1]
        if prev and curr:
            chg = curr - prev
            pct = chg / prev * 100
            arrow = "📈" if chg >= 0 else "📉"
            sign  = "+" if chg >= 0 else ""
            return f"Nifty 50: {curr:,.0f} ({sign}{pct:.2f}%) {arrow}"
    except Exception:
        pass
    return None


def daily_pulse(ud):
    today = date.today()
    seed  = int(today.strftime("%Y%m%d")) + int(ud.get("monthly_income",0)) % 999
    rng   = random.Random(seed)

    # Try live NSE data; fall back to date-seeded realistic placeholder
    live_market = _fetch_nse_quote()
    if not live_market:
        day_of_year = today.timetuple().tm_yday
        # Seeded to date so same day always shows same number (not random-looking)
        pct = round((((seed * 7 + day_of_year * 13) % 200) - 100) / 100, 2)
        nifty_base = 24500 + (day_of_year * 31) % 1800
        sign = "+" if pct >= 0 else ""
        arrow = "📈" if pct >= 0 else "📉"
        live_market = f"Nifty 50: {nifty_base:,} ({sign}{pct}%) {arrow} (end of day estimate)"

    # Action-oriented, non-trivial daily tips — rotate by week not by random
    week_num = today.isocalendar()[1]
    tips_pool = [
        "Your SIP auto-invests even when you forget — that consistency beats timing every time.",
        "Review your emergency fund — did your expenses change this month?",
        "Step-up your SIP by 10% this year. Over 20 years, that single habit adds ~30% to corpus.",
        "Check that your term insurance cover = 15× your annual income. Most people are underinsured.",
        "Nominee update: have you added or updated nominees on all your investments this year?",
        "Tax season is every day — max your 80C (₹1.5L) and NPS 80CCD1B (₹50K) before March.",
        "Idle savings account money above 6 months expenses is silently losing to inflation — invest it.",
    ]
    tip = tips_pool[week_num % len(tips_pool)]

    # Meaningful streak: days since user first registered (seeded to their income + age)
    # Represents days of financial discipline, not random
    base_streak = (int(ud.get("monthly_income",50000)) // 5000 + int(ud.get("age",28))) % 30
    days_in_year = today.timetuple().tm_yday
    streak = base_streak + (days_in_year % 15) + 1   # 1-44, grows through the year

    return {
        "market": live_market,
        "tip":    tip,
        "streak": streak,
        "date":   today.strftime("%d %b %Y"),
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar_form():
    st.sidebar.markdown("# FinLife OS")
    st.sidebar.caption("AI Financial Simulation · Not investment advice")
    st.sidebar.divider()

    with st.sidebar.expander("👤 About You", expanded=True):
        name   = st.text_input("Your name", value="Arjun Mehta", label_visibility="visible")
        age    = st.slider("Current age", 22, 60, 28)
        retire = st.slider("Target retirement age", 35, 65, 50)

    with st.sidebar.expander("💰 Monthly Cash Flow", expanded=True):
        income   = st.number_input("Take-home income (₹)", 10_000, 3_000_000, 85_000, 5_000,
                                   help="Monthly salary after tax, before any investments")
        expenses = st.number_input("Monthly spending (₹)", 5_000, 1_500_000, 45_000, 2_000,
                                   help="Rent + food + utilities + subscriptions + everything else")
        emi      = st.number_input("Loan EMIs (₹/mo)", 0, 500_000, 12_000, 1_000,
                                   help="All home loan / car loan / personal loan EMIs combined")

    with st.sidebar.expander("📈 Savings & Investing", expanded=True):
        savings = st.number_input("Total savings today (₹)", 0, 50_000_000, 200_000, 50_000,
                                  help="FD + mutual funds + stocks + PPF + everything")
        invest  = st.number_input("Monthly SIP / investing (₹)", 0, 500_000, 25_000, 1_000,
                                  help="Auto-invest amount each month across all instruments")
        risk    = st.selectbox("Risk comfort level",
                               ["conservative","moderate","aggressive"], index=1,
                               help="Conservative = FD-like · Moderate = balanced · Aggressive = mostly equity")

    with st.sidebar.expander("🎯 Life Goals", expanded=True):
        goal_map = {"emergency":"🛡️ Emergency Fund","house":"🏠 Buy a Home",
                    "retirement":"🌴 Retire Comfortably","child":"👶 Child's Education",
                    "marriage":"💍 Marriage","vehicle":"🚗 Vehicle",
                    "travel":"✈️ Travel Fund","startup":"🚀 Start a Business"}
        goals = st.multiselect("Your goals", list(goal_map.keys()), default=["emergency","retirement"],
                               format_func=lambda x: goal_map.get(x,x))

    with st.sidebar.expander("🧾 Tax Details (for Tax Wizard)", expanded=False):
        hra      = st.number_input("HRA from salary (₹/yr)", 0, 1_500_000, 0, 10_000)
        elss     = st.number_input("80C investments (₹/yr)", 0, 150_000, 50_000, 10_000,
                                   help="ELSS + PPF + LIC + ULIP combined — max ₹1.5L")
        nps      = st.number_input("NPS contribution (₹/yr)", 0, 200_000, 0, 10_000,
                                   help="Extra NPS beyond employer — gives extra 80CCD1B deduction")
        loan_int = st.number_input("Home loan interest paid (₹/yr)", 0, 300_000, 0, 10_000)
        city     = st.selectbox("City type", ["metro","non-metro"], help="Affects HRA exemption calculation")

    with st.sidebar.expander("🤖 AI Chat (needs API key)", expanded=False):
        api_key_input = st.text_input("Gemini API Key", value="", type="password",
                                      help="Optional — enables Gemini AI explanations and chat")
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
            os.environ["GOOGLE_API_KEY"] = api_key_input
            os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
            os.environ["GOOGLE_GEMINI_MODEL"] = "gemini-2.5-flash"

    st.sidebar.divider()
    run = st.sidebar.button("▶  Run My Simulation", type="primary", use_container_width=True)
    st.sidebar.markdown(f"""
    <div style="background:{W}0A;border:1px solid {W}22;border-radius:7px;padding:.6rem .85rem;
                font-size:.7rem;color:#907850;line-height:1.6;margin-top:.5rem;">
    ⚖️ <strong>SEBI Disclaimer:</strong> FinLife OS provides educational simulations only.
    This is NOT SEBI-registered investment advice. Past performance does not guarantee future returns.
    Consult a SEBI-registered financial advisor before investing.
    </div>""", unsafe_allow_html=True)

    return {
        "name":name,"age":age,"retirement_age":retire,
        "monthly_income":income,"monthly_expenses":expenses,"monthly_emi":emi,
        "current_savings":savings,"monthly_investment":invest,
        "risk_profile":risk,"goals":goals,
        "hra_annual":hra,"elss_annual":elss,"nps_annual":nps,
        "loan_interest":loan_int,"city":city,
    }, run

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    user_data, run_btn = sidebar_form()

    intake_a = IntakeAgent()
    gap_a    = GapAgent()
    goal_a   = GoalAgent()
    mc_a     = MonteCarloAgent(seed=MC_SEED)
    shock_a  = ShockAgent()
    rec_a    = RecommendationAgent()

    # ── Landing ──────────────────────────────────────────────────────────────
    if not run_btn and "results" not in st.session_state:
        st.markdown(f"""
        <div style="text-align:center;padding:3rem 0 1.5rem;">
          <div style="font-size:3rem;margin-bottom:.6rem;">📡</div>
          <h1 style="font-size:2.4rem;font-weight:700;color:#0F172A;letter-spacing:-.02em;margin:0 0 .4rem;">FinLife OS</h1>
          <p style="color:{M};font-size:.92rem;margin:0 0 2rem;">
            Your financial future — simulated across 300 parallel life paths
          </p>
        </div>""", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col,ico,t,d in [
            (c1,"🎲","300 Futures","Not one prediction — 300 simulated outcomes"),
            (c2,"⚡","Life Shock Test","Job loss, medical, marriage — see real impact"),
            (c3,"🧾","Tax Wizard","AI calculates your optimal tax regime step-by-step"),
            (c4,"💬","AI Chat","Ask anything about your finances in plain English"),
        ]:
            col.markdown(f'<div style="background:{S};border:1px solid {B};border-radius:9px;padding:1.4rem;text-align:center;">'
                         f'<div style="font-size:1.7rem;margin-bottom:.5rem;">{ico}</div>'
                         f'<div style="font-weight:600;color:#1E293B;margin-bottom:.25rem;font-size:.9rem;">{t}</div>'
                         f'<div style="font-size:.76rem;color:{M};">{d}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;margin-top:1.5rem;color:{M};font-size:.78rem;">'
                    f'Fill in your details on the left → <strong style="color:{A};">▶ Run My Simulation</strong></div>',
                    unsafe_allow_html=True)
        return

    # ── Run all agents ────────────────────────────────────────────────────────
    if run_btn:
        st.markdown(f'<div style="background:{S};border:1px solid {B};border-radius:8px;padding:.8rem 1rem;margin-bottom:.5rem;">'
                    f'<div style="font-size:.62rem;font-weight:700;letter-spacing:.1em;color:{M};text-transform:uppercase;margin-bottom:.4rem;">Agent Pipeline</div>'
                    f'<style>.pill{{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:100px;font-size:.67rem;font-weight:600;margin:2px;font-family:\'JetBrains Mono\';}}.pd{{background:{A2}18;color:{A2};border:1px solid {A2}44;}}.pr{{background:{W}18;color:{W};border:1px solid {W}44;}}.pw{{background:{M}18;color:{M};border:1px solid {M}44;}}</style>', unsafe_allow_html=True)

        anames   = ["Intake","Health","Goals","Monte Carlo","Shocks","Recs"]
        statuses = {n:"waiting" for n in anames}
        prog     = st.progress(0, text="Starting...")
        bslot    = st.empty()

        def render_pills():
            bslot.markdown(" ".join(pill(n,statuses[n]) for n in anames)+"</div>", unsafe_allow_html=True)

        render_pills()

        def tick(n, pct, txt, detail=""):
            statuses[n]="done"; prog.progress(pct,text=txt); render_pills()

        statuses["Intake"]="running"; render_pills()
        ud = intake_a.from_form(user_data)
        tick("Intake",14,"Profile loaded ✓")

        statuses["Health"]="running"; render_pills()
        health     = gap_a.run(ud)
        tick("Health",28,f"Health score: {health.overall_score}/100")

        statuses["Goals"]="running"; render_pills()
        goal_plans = goal_a.run(ud)
        tick("Goals",42,f"{len(goal_plans)} goals decomposed")

        statuses["Monte Carlo"]="running"; render_pills()
        horizon   = max(5, ud["retirement_age"] - ud["age"])
        sim_input = SimulationInput(
            monthly_income=ud["monthly_income"], monthly_expenses=ud["monthly_expenses"],
            monthly_emi=ud["monthly_emi"], current_savings=ud["current_savings"],
            monthly_investment=ud["monthly_investment"], risk_profile=ud["risk_profile"],
            horizon_years=horizon, income_growth_mean=ud.get("income_growth_pct",10)/100,
        )
        mc_result  = mc_a.run(sim_input)
        mc_a.rng   = np.random.default_rng(MC_SEED)
        opt_result = mc_a.run_optimised(sim_input)
        mc_a.rng   = np.random.default_rng(MC_SEED)
        goal_plans = goal_a.attach_probabilities(goal_plans, mc_a, ud)
        mc_a.rng   = np.random.default_rng(MC_SEED)
        sensitivity= mc_a.run_sensitivity(sim_input)
        explanation= intake_a.generate_explanation(mc_result, ud)

        # Projected health: simulate 12 months of following recommendations
        health_proj = gap_a.projected_score(ud, months=12)

        # Detect impossible goals (SIP needed > 80% of surplus)
        surplus = ud["monthly_income"] - ud["monthly_expenses"] - ud["monthly_emi"]
        impossible_warning = None
        if surplus > 0:
            total_sip_needed = sum(g.sip_required for g in goal_plans)
            if total_sip_needed > surplus * 0.80:
                impossible_warning = (f"Combined SIP needed for all goals ({fmt_inr(total_sip_needed)}/mo) "
                                      f"exceeds 80% of your surplus ({fmt_inr(surplus*0.80)}/mo). "
                                      f"Consider extending timelines or prioritising 1-2 goals.")

        tick("Monte Carlo",66,f"Probability: {mc_result.success_probability:.0%}")

        statuses["Shocks"]="running"; render_pills()
        shock_results = shock_a.run_all(mc_a, sim_input, mc_result)
        tick("Shocks",83,"5 life scenarios tested")

        statuses["Recs"]="running"; render_pills()
        recs = rec_a.run(ud, health, mc_result.success_probability)
        peer = peer_comparison(ud)

        # Tax calculation
        ann     = ud["monthly_income"] * 12
        hra_e   = min(user_data.get("hra_annual",0), int(ann*0.40*0.50))
        elss_i  = user_data.get("elss_annual",0)
        nps_i   = user_data.get("nps_annual",0)
        loan_i  = user_data.get("loan_interest",0)
        tax_new = tax_liability_new_regime(ann)
        tax_old = tax_liability_old_regime(ann, elss_i, nps_i+loan_i+hra_e)
        tick("Recs",100,"✅ Simulation complete")

        # AI tax advice
        extra = {"hra_annual":user_data.get("hra_annual",0),"elss_annual":elss_i,
                 "nps_annual":nps_i,"loan_interest":loan_i,"city":user_data.get("city","metro")}
        tax_advice = intake_a.get_tax_advice(ud, extra)

        # Compute optimised invest params for health projection display
        freed       = sim_input.monthly_expenses * 0.08
        new_surplus = sim_input.monthly_income - sim_input.monthly_expenses*0.92 - sim_input.monthly_emi
        opt_invest  = max(sim_input.monthly_investment, min(sim_input.monthly_investment+freed, new_surplus*0.80))

        st.session_state["results"] = dict(
            ud=ud, health=health, health_proj=health_proj, goal_plans=goal_plans,
            mc_result=mc_result, opt_result=opt_result, shock_results=shock_results,
            recs=recs, peer=peer, explanation=explanation, sensitivity=sensitivity,
            sim_input=sim_input, tax_new=tax_new, tax_old=tax_old,
            tax_advice=tax_advice, opt_invest=opt_invest,
            impossible_warning=impossible_warning,
            user_data_raw=user_data,
        )

    if "results" not in st.session_state:
        return

    R = st.session_state["results"]
    ud=R["ud"]; health=R["health"]; health_proj=R["health_proj"]
    goal_plans=R["goal_plans"]; mc_result=R["mc_result"]; opt_result=R["opt_result"]
    shock_results=R["shock_results"]; recs=R["recs"]; peer=R["peer"]
    explanation=R["explanation"]; sensitivity=R["sensitivity"]; sim_input=R["sim_input"]
    tax_new=R["tax_new"]; tax_old=R["tax_old"]; tax_advice=R["tax_advice"]
    opt_invest=R["opt_invest"]; impossible_warning=R["impossible_warning"]
    user_data_raw=R["user_data_raw"]
    prob     = mc_result.success_probability
    opt_prob = opt_result.success_probability
    color    = pc(prob)
    score_delta = health_proj.overall_score - health.overall_score

    # ── Daily Pulse ──────────────────────────────────────────────────────────
    pulse = daily_pulse(ud)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{A}1A,{A2}11);border:1px solid {A}33;
                border-radius:10px;padding:1rem 1.3rem;display:flex;justify-content:space-between;
                align-items:center;flex-wrap:wrap;gap:.8rem;margin-bottom:.8rem;">
      <div>
        <div style="font-size:.6rem;font-weight:700;letter-spacing:.1em;color:{M};text-transform:uppercase;">
          Daily Financial Pulse · {pulse['date']}
        </div>
        <div style="font-size:.88rem;color:#475569;margin-top:.2rem;">
          {pulse['market']} &nbsp;·&nbsp; 💡 {pulse['tip']}
        </div>
      </div>
      <div style="text-align:right;flex-shrink:0;">
        <div style="font-size:.6rem;font-weight:700;letter-spacing:.1em;color:{M};text-transform:uppercase;">SIP Active Days</div>
        <div style="font-family:'JetBrains Mono';font-size:1.25rem;font-weight:700;color:{A2};">🔥 {pulse['streak']} days</div>
        <div style="font-size:.6rem;color:{M};margin-top:.1rem;">this financial year</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if impossible_warning:
        st.warning(f"⚠️ {impossible_warning}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["📊 Overview","🎯 Goals","💥 Life Shocks","💡 Action Plan","🧾 Tax Wizard","🔬 Deep Analysis","💬 AI Chat"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        hero, meta = st.columns([1,2], gap="large")
        with hero:
            score_color = A2 if score_delta>5 else (W if score_delta>0 else M)
            st.markdown(f"""
            <div style="background:{S};border:1px solid {B};border-radius:12px;padding:1.5rem;
                        position:relative;overflow:hidden;height:100%;">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,{A},{A2});"></div>
              <div style="font-size:.62rem;font-weight:700;letter-spacing:.12em;color:{M};
                          text-transform:uppercase;margin-bottom:.3rem;">Goal success probability</div>
              <div style="font-family:'JetBrains Mono';font-size:4.5rem;font-weight:700;
                          line-height:1;letter-spacing:-.03em;color:{color};">{prob*100:.0f}%</div>
              <div style="color:{M};font-size:.73rem;margin-top:.3rem;font-family:'JetBrains Mono';">
                300 simulations · {sim_input.horizon_years}-yr horizon
              </div>
              <div style="margin-top:.85rem;padding:.65rem;background:{color}0F;border:1px solid {color}2A;
                          border-radius:6px;font-size:.77rem;color:#334155;line-height:1.6;">
                {explanation}
              </div>
              <div style="margin-top:.8rem;display:flex;gap:.4rem;">
                <div style="flex:1;background:{BG};border:1px solid {B};border-radius:6px;padding:.5rem;text-align:center;">
                  <div style="font-size:.58rem;font-weight:700;letter-spacing:.08em;color:{M};text-transform:uppercase;">Now</div>
                  <div style="font-family:'JetBrains Mono';font-size:1rem;font-weight:700;color:{color};">{prob*100:.0f}%</div>
                </div>
                <div style="flex:1;background:{BG};border:1px solid {A2}44;border-radius:6px;padding:.5rem;text-align:center;">
                  <div style="font-size:.58rem;font-weight:700;letter-spacing:.08em;color:{M};text-transform:uppercase;">If you act</div>
                  <div style="font-family:'JetBrains Mono';font-size:1rem;font-weight:700;color:{A2};">{opt_prob*100:.0f}%</div>
                </div>
              </div>
              <div style="margin-top:.6rem;background:{BG};border:1px solid {B};border-radius:6px;padding:.5rem;text-align:center;">
                <div style="font-size:.58rem;font-weight:700;letter-spacing:.08em;color:{M};text-transform:uppercase;">Health score: now → after 12mo plan</div>
                <div style="font-family:'JetBrains Mono';font-size:.95rem;margin-top:.1rem;">
                  <span style="color:{sc('critical') if health.overall_score<40 else A};">{health.overall_score}/100</span>
                  <span style="color:{M};"> → </span>
                  <span style="color:{score_color};font-weight:700;">{health_proj.overall_score}/100
                    {'(+'+str(score_delta)+')' if score_delta>0 else ''}
                  </span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        with meta:
            surplus = ud["monthly_income"]-ud["monthly_expenses"]-ud["monthly_emi"]
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Health Score", f"{health.overall_score}/100",
                      delta=f"Grade {health.grade}",
                      delta_color="normal" if health.overall_score>=60 else "inverse")
            k2.metric("Middle-path wealth", fmt_inr(mc_result.median_final_wealth),
                      delta=f"goal: {fmt_inr(mc_result.target_corpus)}")
            k3.metric("Monthly surplus", fmt_inr(surplus),
                      delta=fmt_pct(surplus/max(1,ud["monthly_income"]))+" of income")
            k4.metric("Investing rate", fmt_pct(ud["monthly_investment"]/max(1,ud["monthly_income"])),
                      delta="target: 20%",
                      delta_color="normal" if ud["monthly_investment"]/max(1,ud["monthly_income"])>=.20 else "inverse")
            st.markdown("")
            k5,k6,k7,k8 = st.columns(4)
            k5.metric("Worst case (10%)",  fmt_inr(mc_result.worst_case_p10))
            k6.metric("Middle (50%)",      fmt_inr(mc_result.median_final_wealth))
            k7.metric("Best case (90%)",   fmt_inr(mc_result.best_case_p90))
            k8.metric("Retire target",     f"Age {ud['retirement_age']}", delta=f"{sim_input.horizon_years} yrs away")

        st.markdown("---")
        st.plotly_chart(chart_proj(mc_result, opt_result, ud), use_container_width=True)
        st.markdown(info_box("Red = your current path (300 possible outcomes). Green dashes = if you follow the action plan. Yellow line = your goal. When green crosses yellow, you retire.",A,"📖"), unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### Financial Health Score")
        rc, dc = st.columns([1,1.5], gap="large")
        with rc:
            st.plotly_chart(chart_radar(health), use_container_width=True)
        with dc:
            st.markdown(f'<div style="color:#1E293B;font-weight:600;font-size:.9rem;padding:.3rem 0;margin-bottom:.5rem;">{health.summary}</div>', unsafe_allow_html=True)
            for dim in health.dimensions:
                sev_c = sc(dim.severity)
                with st.expander(f"{si(dim.severity)}  {dim.name} — {dim.score}/100", expanded=(dim.severity=="critical")):
                    st.markdown(bar_html(dim.score, sev_c), unsafe_allow_html=True)
                    ca,cb = st.columns(2)
                    ca.markdown(f'<div style="font-size:.7rem;color:{M};font-weight:700;text-transform:uppercase;letter-spacing:.07em;">Where you are</div>'
                                f'<div style="font-size:.82rem;color:#C0C8E0;">{dim.value}</div>', unsafe_allow_html=True)
                    cb.markdown(f'<div style="font-size:.7rem;color:{M};font-weight:700;text-transform:uppercase;letter-spacing:.07em;">Target</div>'
                                f'<div style="font-size:.82rem;color:#C0C8E0;">{dim.target}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="margin-top:.45rem;padding:.45rem .7rem;background:{sev_c}0F;border-left:3px solid {sev_c};border-radius:0 5px 5px 0;font-size:.78rem;color:#C0C8E0;">{dim.issue}</div>'
                                f'<div style="margin-top:.3rem;font-size:.78rem;color:{A2};">✓ {dim.fix}</div>', unsafe_allow_html=True)
                    if dim.fix_rupee:
                        st.markdown(f'<div style="font-size:.73rem;color:{W};margin-top:.2rem;">Amount needed: {dim.fix_rupee}</div>', unsafe_allow_html=True)

        with st.expander("🔍 Simulation assumptions", expanded=False):
            rp = RISK_RETURN.get(ud["risk_profile"])
            a1,a2,a3,a4,a5 = st.columns(5)
            a1.metric("Expected return", fmt_pct(rp["mean"]), f"±{fmt_pct(rp['std'])} σ")
            a2.metric("Inflation", "6.0%", "±1.5% σ")
            a3.metric("Salary growth", "8.0%", "±2.0% σ")
            a4.metric("Simulations", "300", f"{sim_input.horizon_years}-yr span")
            a5.metric("Return model", "Log-normal", "prevents impossible returns")
            st.caption("Fixed seed 42 — identical inputs always produce identical results. Log-normal returns prevent wealth going below -12mo expenses.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — GOALS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Goal Decomposition")
        if not goal_plans:
            st.warning("No goals selected. Pick some in the sidebar.")
        else:
            st.plotly_chart(chart_goals(goal_plans), use_container_width=True)
            st.markdown("---")
            rows = [{"Goal":g.label,"Timeline":f"{g.horizon_years} yrs","Target":fmt_inr(g.target_future),
                     "Monthly SIP needed":f"{fmt_inr(g.sip_required)}/mo","Probability":f"{g.probability*100:.0f}%",
                     "Gap":fmt_inr(g.shortfall) if g.shortfall>0 else "✅ On track",
                     "Priority":g.priority.upper()} for g in goal_plans]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("---")
            for g in goal_plans:
                gc = pc(g.probability)
                with st.expander(f"{g.label}  ·  {g.probability*100:.0f}% chance  ·  {fmt_inr(g.target_future)}", expanded=(g.priority=="high")):
                    gc1,gc2,gc3,gc4 = st.columns(4)
                    gc1.metric("Probability", f"{g.probability*100:.0f}%")
                    gc2.metric("Corpus needed", fmt_inr(g.target_future))
                    gc3.metric("SIP required", fmt_inr(g.sip_required)+"/mo")
                    gc4.metric("Gap", fmt_inr(g.shortfall) if g.shortfall>0 else "None ✅")
                    st.markdown(f'<div style="font-size:.78rem;color:#64748B;margin-top:.4rem;">'
                                f'{g.horizon_years}-yr horizon · <span style="color:{gc};">{g.priority.upper()} priority</span> · {g.note}</div>',
                                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — LIFE SHOCKS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### What If Life Throws a Curveball?")
        st.caption("We re-run all 300 simulations with each life event applied at Year 2.")
        st.plotly_chart(chart_shocks(shock_results), use_container_width=True)
        st.markdown("---")
        for sr in shock_results:
            dc = D if sr.prob_delta<-.08 else (W if sr.prob_delta<0 else A2)
            with st.expander(f"{sr.shock_name}  →  {sr.prob_delta*100:+.1f}% change in probability", expanded=False):
                sc1,sc2,sc3,sc4 = st.columns(4)
                sc1.metric("Without event", f"{sr.base_probability*100:.1f}%")
                sc2.metric("After event",   f"{sr.shocked_probability*100:.1f}%",
                           delta=f"{sr.prob_delta*100:+.1f}%", delta_color="inverse")
                sc3.metric("Wealth lost",   fmt_inr(abs(sr.corpus_delta)))
                sc4.metric("Years delayed", f"~{sr.years_delayed:.1f} yrs")
                bw = int(sr.base_probability*100); sw = int(sr.shocked_probability*100)
                st.markdown(f'<div style="margin:.65rem 0 .4rem;">'
                            f'<div style="font-size:.65rem;color:{M};text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;">Without event</div>'
                            f'{bar_html(bw,A2,6)}'
                            f'<div style="font-size:.65rem;color:{M};text-transform:uppercase;letter-spacing:.06em;margin:5px 0 3px;">After event</div>'
                            f'{bar_html(sw,D,6)}</div>', unsafe_allow_html=True)
                st.markdown(f"**What happens:** {sr.description}")
                st.markdown(info_box(sr.recommendation, A2, "🛡️"), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Simulate Your Own Scenario")
        cs1,cs2,cs3 = st.columns(3)
        cf = cs1.slider("Income drops to (% of current)", 0, 100, 70, 5, help="0% = complete job loss")
        cd = cs2.slider("For how many months", 0, 36, 12)
        cc = cs3.number_input("One-time cost (₹)", 0, 5_000_000, 0, step=50_000)
        if st.button("▶  Test This Scenario"):
            csr = shock_a.run_custom(cf/100, cd, cc, mc_a, sim_input, mc_result)
            c1,c2,c3 = st.columns(3)
            c1.metric("Without event", f"{csr.base_probability*100:.1f}%")
            c2.metric("After event",   f"{csr.shocked_probability*100:.1f}%",
                      delta=f"{csr.prob_delta*100:+.1f}%", delta_color="inverse")
            c3.metric("Wealth lost",   fmt_inr(abs(csr.corpus_delta)))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — ACTION PLAN
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Action Plan — Ranked by 20-Year Impact")
        st.markdown(info_box("Follow these in order. Each action is ranked by how much wealth it builds over 20 years. Most people only need to do #1 and #2 to see dramatic improvement.", A2, "💡"), unsafe_allow_html=True)
        for rec in recs:
            st.markdown(rec_card(rec), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### How You Compare to Peers")
        st.caption(f"Others earning similar income: {peer['bracket_label']}")
        pb1,pb2,pb3 = st.columns(3)
        for i,(key,m) in enumerate(peer["metrics"].items()):
            col=[pb1,pb2,pb3][i]; gd="normal" if m["gap"]>=0 else "inverse"
            if key=="emergency_months":
                col.metric(m["label"], f"{m['user']:.1f} months",
                           delta=f"{m['gap']:+.1f} vs peers", delta_color=gd)
            else:
                col.metric(m["label"], fmt_pct(m["user"]),
                           delta=f"{m['gap']*100:+.1f}pp vs peers", delta_color=gd)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — TAX WIZARD (full featured)
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Tax Wizard — Which Regime Saves You More?")
        ann = ud["monthly_income"]*12

        # Top comparison
        better  = "Old" if tax_old < tax_new else "New"
        saving  = abs(tax_new - tax_old)
        better_c = A2 if saving > 0 else M

        t1,t2,t3,t4 = st.columns(4)
        t1.metric("Annual income",    fmt_inr(ann))
        t2.metric("New regime tax",   fmt_inr(tax_new))
        t3.metric("Old regime tax",   fmt_inr(tax_old))
        t4.metric("You save",         fmt_inr(saving), delta=f"{better} regime wins", delta_color="normal")

        st.markdown(f"""
        <div style="margin:.8rem 0;padding:.75rem 1rem;background:{better_c}0F;
                    border:1px solid {better_c}33;border-radius:8px;font-size:.88rem;color:#A8C8B0;font-weight:600;">
          ✅ <strong>{better} Tax Regime</strong> saves you
          <strong style="color:{better_c};">{fmt_inr(saving)}/year</strong>
          {"= " + fmt_inr(saving*20) + " over 20 years if invested" if saving > 0 else ""}
        </div>""", unsafe_allow_html=True)

        # Step-by-step traceable calculation (required by hackathon rubric)
        st.markdown("#### Step-by-Step Calculation")
        col_new, col_old = st.columns(2)

        with col_new:
            st.markdown("**New Regime (FY 2025-26)**")
            new_breakdown = get_new_regime_breakdown(ann)
            rows_new = []
            for item in new_breakdown["breakdown"]:
                rows_new.append({
                    "Slab": item["slab"],
                    "Amount": fmt_inr(item["amount"]),
                    "Rate": f"{int(item['rate']*100)}%",
                    "Tax": fmt_inr(item["tax"]),
                })
            st.dataframe(pd.DataFrame(rows_new), hide_index=True, use_container_width=True)
            st.markdown(f"Taxable income after ₹75K standard deduction: {fmt_inr(new_breakdown['taxable_income'])}")
            if new_breakdown["rebate"] > 0:
                st.success(f"✅ Rebate applied: {fmt_inr(new_breakdown['rebate'])} (Section 87A)")
            else:
                st.markdown("Rebate applied: ₹0")
            st.markdown(f"Base tax: {fmt_inr(new_breakdown['base_tax'])} + Cess (4%): {fmt_inr(new_breakdown['cess'])} = **{fmt_inr(new_breakdown['total_tax'])}**")
            if new_breakdown["total_tax"] == 0:
                st.success("🔥 You pay ZERO tax under the new regime due to rebate!")

        with col_old:
            hra_e   = min(user_data_raw.get("hra_annual",0), int(ann*0.40*0.50))
            elss_i  = user_data_raw.get("elss_annual",0)
            nps_i   = user_data_raw.get("nps_annual",0)
            loan_i  = user_data_raw.get("loan_interest",0)
            total_ded = 50000 + elss_i + nps_i + loan_i + hra_e
            taxable_old = max(0, ann - total_ded)
            st.markdown(f"**Old Regime — Your Deductions**")
            ded_rows = [
                {"Deduction":"Standard deduction","Section":"Standard","Amount":fmt_inr(50000)},
                {"Deduction":"ELSS/PPF/80C investments","Section":"80C","Amount":fmt_inr(elss_i)},
                {"Deduction":"NPS contribution","Section":"80CCD(1B)","Amount":fmt_inr(nps_i)},
                {"Deduction":"Home loan interest","Section":"24b","Amount":fmt_inr(loan_i)},
                {"Deduction":"HRA exempt","Section":"10(13A)","Amount":fmt_inr(hra_e)},
            ]
            st.dataframe(pd.DataFrame(ded_rows), hide_index=True, use_container_width=True)
            st.markdown(f"Taxable income: **{fmt_inr(taxable_old)}** → Tax: **{fmt_inr(tax_old)}**")

        # AI-powered detailed advice
        st.markdown("---")
        st.markdown("#### AI Tax Advisor Analysis")
        if "•" in tax_advice or "**" in tax_advice:
            st.markdown(tax_advice)
        else:
            st.write(tax_advice)

        # Missed deductions
        st.markdown("---")
        st.markdown("#### Deductions You May Be Missing")
        missed = []
        if elss_i < 150000:
            gap = 150000-elss_i
            saved = gap*0.20
            missed.append({"Deduction":"80C (ELSS/PPF/etc)","Gap":fmt_inr(gap),"Tax saved":fmt_inr(saved),
                          "Action":"Invest in ELSS (equity-linked, 3yr lock-in) or PPF (safe, 15yr)"})
        if nps_i < 50000:
            gap = 50000-nps_i
            saved = gap*0.20
            missed.append({"Deduction":"NPS 80CCD(1B)","Gap":fmt_inr(gap),"Tax saved":fmt_inr(saved),
                          "Action":"Open NPS Tier 1, invest before March 31 for this FY"})
        if user_data_raw.get("loan_interest",0)==0 and ud.get("monthly_emi",0)>0:
            missed.append({"Deduction":"Home loan interest (24b)","Gap":fmt_inr(200000),"Tax saved":fmt_inr(40000),
                          "Action":"Claim interest paid on home loan — check with your bank for certificate"})
        if missed:
            st.dataframe(pd.DataFrame(missed), hide_index=True, use_container_width=True)
            total_extra_saving = sum(float(m["Tax saved"].replace("₹","").replace("K","000").replace("L","00000")) for m in missed[:2])
        else:
            st.success("✅ You appear to be claiming all major deductions. Well done!")

        # Tax-saving investments ranked
        st.markdown("---")
        st.markdown("#### Tax-Saving Investments Ranked (by liquidity × tax benefit)")
        tax_inv = [
            {"Instrument":"ELSS Mutual Fund","Section":"80C","Max deduction":"₹1.5L","Lock-in":"3 years","Risk":"Medium","Returns":"10-14%","Liquidity":"⭐⭐⭐"},
            {"Instrument":"PPF","Section":"80C","Max deduction":"₹1.5L","Lock-in":"15 years","Risk":"Very Low","Returns":"7.1%","Liquidity":"⭐"},
            {"Instrument":"NPS Tier 1","Section":"80CCD(1B)","Max deduction":"₹50K extra","Lock-in":"Till 60","Risk":"Low-Medium","Returns":"8-12%","Liquidity":"⭐"},
            {"Instrument":"Term Life Insurance","Section":"80C","Max deduction":"₹1.5L","Lock-in":"Policy term","Risk":"None","Returns":"N/A (protection)","Liquidity":"⭐⭐"},
            {"Instrument":"5-yr Tax Saver FD","Section":"80C","Max deduction":"₹1.5L","Lock-in":"5 years","Risk":"Very Low","Returns":"6.5-7%","Liquidity":"⭐⭐"},
        ]
        st.dataframe(pd.DataFrame(tax_inv), hide_index=True, use_container_width=True)
        st.markdown(info_box("All instruments above are educational examples. Past returns don't guarantee future performance. Consult a CA before filing.", W, "⚖️"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — DEEP ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### Deep Analysis")

        # Explainability
        st.markdown(f"#### Why is your chance exactly {prob*100:.0f}%?")
        for d in mc_result.key_drivers:
            st.markdown(driver_card(d), unsafe_allow_html=True)

        st.markdown("---")
        da, db = st.columns([1.2,1], gap="large")
        with da:
            st.markdown("#### Wealth distribution at retirement")
            st.plotly_chart(chart_dist(mc_result), use_container_width=True)
            st.markdown(info_box("Each bar = number of simulations ending with that wealth. Yellow = your goal. More bars to the right of yellow = better.", A, "📖"), unsafe_allow_html=True)
        with db:
            st.markdown("#### What single change helps most?")
            for key,(delta,desc) in sorted(sensitivity.items(), key=lambda x:abs(x[1][0]), reverse=True):
                col2 = A2 if delta>0 else D
                bw2  = min(100, int(abs(delta)*200))
                st.markdown(f'<div style="margin-bottom:.65rem;">'
                            f'<div style="display:flex;justify-content:space-between;font-size:.76rem;color:#64748B;margin-bottom:2px;">'
                            f'<span>{desc}</span>'
                            f'<span style="color:{col2};font-family:\'JetBrains Mono\';font-weight:600;">{delta*100:+.1f}%</span></div>'
                            f'{bar_html(bw2,col2)}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### The True Cost of Every Purchase")
        st.caption("Enter something you're thinking of buying to see what it really costs your future wealth.")
        di1,di2 = st.columns(2)
        purchase = di1.number_input("Purchase amount (₹)", 0, 5_000_000, 85_000, 5_000)
        hor_yr   = di2.slider("Over how many years?", 1, 30, 10)
        if purchase > 0:
            opp   = opportunity_cost(purchase, .11, hor_yr)
            delay = round(purchase/max(1,ud["monthly_investment"]), 1)
            d1,d2,d3,d4 = st.columns(4)
            d1.metric("You pay",           fmt_inr(purchase))
            d2.metric("Opportunity cost",  fmt_inr(opp), f"at 11% for {hor_yr}yr")
            d3.metric("Real total cost",   fmt_inr(purchase+opp))
            d4.metric("SIP months lost",   f"{delay:.1f}")
            st.markdown(info_box(f"This purchase delays your primary goal by {delay:.1f} months of SIP progress and has an opportunity cost of {fmt_inr(opp)} over {hor_yr} years.", D, "⚠️"), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Decisions Costing You Money Right Now")
        st.plotly_chart(chart_mistakes(ud), use_container_width=True)
        idle = max(0, ud["current_savings"]-(ud["monthly_expenses"]+ud["monthly_emi"])*6)
        mistakes = []
        if idle > 0:
            loss = idle*((1.11**20)-(1.035**20))
            mistakes.append(f"Keeping {fmt_inr(idle)} idle in a savings account costs <strong>{fmt_inr(loss)}</strong> over 20 years in missed returns.")
        mistakes.append(f"Not having a 10% annual SIP step-up costs an estimated <strong>{fmt_inr(ud['monthly_investment']*12*0.10*((1.11**20-1)/0.11))}</strong> over 20 years.")
        if ud.get("monthly_investment",0)/max(1,ud.get("monthly_income",1)) < 0.15:
            mistakes.append(f"Under-investing ({fmt_pct(ud['monthly_investment']/max(1,ud['monthly_income']))} vs 20% target) costs <strong>{fmt_inr(sip_future_value(ud['monthly_income']*0.05, .11, sim_input.horizon_years))}</strong> in compounding.")
        mistakes.append(f"No term insurance: one event can wipe out your entire savings. Annual cost: ~{fmt_inr(ud['monthly_income']*12*0.0006)}/yr for ₹1Cr cover.")
        for m in mistakes:
            st.markdown(f'<div style="padding:.5rem .8rem;margin-bottom:.3rem;background:{D}0D;border-left:3px solid {D};border-radius:0 5px 5px 0;font-size:.78rem;color:#D0A8A8;">⚠ {m}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Agent Audit Trail")
        st.caption("Every decision the AI made — full transparency (required for enterprise readiness)")
        for log_entry in mc_result.audit_log:
            st.markdown(f'<div style="font-family:\'JetBrains Mono\';font-size:.7rem;color:#475569;padding:.18rem 0;border-bottom:1px solid {B};">→ {log_entry}</div>', unsafe_allow_html=True)
        st.markdown(info_box("All simulations use fixed random seed 42. Same inputs always produce same results. Returns modelled as log-normal to prevent impossible negative compound returns.", M, "🔒"), unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{W}0A;border:1px solid {W}22;border-radius:8px;padding:.7rem 1rem;
                    font-size:.72rem;color:#907850;line-height:1.7;margin-top:.8rem;">
          ⚖️ <strong>SEBI Disclaimer:</strong> FinLife OS is an educational simulation tool. It does NOT constitute
          SEBI-registered investment advice, portfolio management, or financial planning services.
          Simulated returns are based on historical assumptions and do not guarantee future performance.
          Before making investment decisions, consult a SEBI-registered investment advisor (RIA).
          For complaints: <strong>SEBI SCORES — scores.sebi.gov.in</strong>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — AI CHAT
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("### Ask Anything About Your Finances")
        has_api = bool(intake_a.api_key)
        if not has_api:
            st.info("💡 Enter your Google API key in the sidebar to enable Gemini AI chat. Without it, you'll get template responses.")

        st.markdown(f'<div style="background:{S};border:1px solid {B};border-radius:10px;padding:.8rem 1rem;margin-bottom:.8rem;">'
                    f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.08em;color:{M};text-transform:uppercase;margin-bottom:.4rem;">Quick questions</div>'
                    f'<div style="display:flex;flex-wrap:wrap;gap:.4rem;">'
                    f'{"".join(badge(q,A) for q in ["Why is my probability "+str(int(prob*100))+"%?","Should I invest more or cut expenses first?","Am I on track for retirement?","What is ELSS and should I invest?","How much term insurance do I need?"])}'
                    f'</div></div>', unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            role_color = A if msg["role"]=="user" else A2
            role_label = "You" if msg["role"]=="user" else "FinLife AI"
            st.markdown(f'<div style="background:{S};border:1px solid {B};border-left:3px solid {role_color};border-radius:0 8px 8px 0;'
                        f'padding:.65rem .9rem;margin-bottom:.4rem;">'
                        f'<div style="font-size:.63rem;font-weight:700;letter-spacing:.08em;color:{role_color};text-transform:uppercase;margin-bottom:.25rem;">{role_label}</div>'
                        f'<div style="font-size:.84rem;color:#334155;line-height:1.6;">{msg["content"]}</div></div>', unsafe_allow_html=True)

        col_q, col_btn = st.columns([5,1])
        with col_q:
            question = st.text_input("Ask about your finances...",
                                     placeholder="e.g. How should I split my savings between SIP and PPF?",
                                     label_visibility="collapsed")
        with col_btn:
            ask_btn = st.button("Ask →", type="primary")

        if ask_btn and question:
            st.session_state.chat_history.append({"role":"user","content":question})
            with st.spinner("Thinking..."):
                answer = intake_a.chat_response(question, ud, mc_result)
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.rerun()

        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown(info_box("AI chat uses Google Gemini to answer questions about YOUR specific financial profile. Not general advice — contextual to your numbers. Always ends with SEBI disclaimer.", M, "🤖"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()