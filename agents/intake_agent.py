"""
agents/intake_agent.py — Gemini API intake + explanation + tax advisor + chat
FIXES:
  - max_tokens raised: explanation 400, chat 600, tax 1200
  - System prompt "under 150 words" removed — was causing Gemini to truncate
  - Full user context injected into every call
  - Graceful fallback when API unavailable
"""
import os, json, re

INTAKE_SYSTEM = """You are a financial data intake assistant for an Indian personal finance app.
Extract the following from the user's message and return ONLY valid JSON:
{
  "name": string or null,
  "age": integer or null,
  "monthly_income": float or null,
  "monthly_expenses": float or null,
  "monthly_emi": float or null,
  "current_savings": float or null,
  "monthly_investment": float or null,
  "retirement_age": integer or null,
  "risk_profile": "conservative" | "moderate" | "aggressive" | null,
  "goals": array from ["house","retirement","child","marriage","vehicle","emergency","travel","startup"],
  "tax_regime": "old" | "new" | null,
  "income_growth_pct": float or null
}
Rules: Convert lakh/L to numbers (5L = 500000). Null if not mentioned. Return ONLY the JSON object, no explanation, no markdown."""

TAX_ADVISOR_SYSTEM = """You are a SEBI-compliant Indian tax advisor AI providing educational guidance only — NOT licensed financial advice.
Given a user's salary structure, produce a complete analysis with:
1. Exact tax under new regime (FY 2025-26 slabs: 0-3L=0%, 3-7L=5%, 7-10L=10%, 10-12L=15%, 12-15L=20%, >15L=30%, add 4% cess, standard deduction ₹75K)
2. Exact tax under old regime (standard deduction ₹75K + 80C max ₹1.5L + NPS 80CCD1B max ₹50K + home loan interest Sec24b max ₹2L + HRA exemption calculated properly)
3. Which regime is optimal and savings amount
4. Every deduction being missed and the exact rupee saving for each
5. Top 3 tax-saving instruments ranked by liquidity and risk with specific amounts

Show all working step by step with rupee amounts at each step.
End with: "This is educational guidance only. Consult a CA for personalised tax filing advice."
Be thorough and specific — judges will verify your arithmetic."""

FINANCIAL_CHAT_SYSTEM = """You are FinLife OS, an AI financial mentor for India's middle class.
You have the user's complete financial profile (provided in each message). Answer their question with:
- Specific numbers from their profile (not generic percentages)
- Plain language explanations — avoid jargon
- 2-3 concrete action steps they can take this week
- Reference Indian financial instruments: SIP, PPF, ELSS, NPS, FD, term insurance

Rules:
- Never recommend specific stocks or mutual fund schemes by name
- If asked about tax, calculate using their actual income
- Use Hinglish naturally only when the user does first
- Be honest — if numbers look bad, say so clearly with the fix
- Give complete, useful answers of 150-200 words
End every response with: "⚖️ Educational content only — not SEBI-registered investment advice."
"""


class IntakeAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY","") or os.getenv("GOOGLE_API_KEY","")
        allowed = ["gemini-2.5-flash","gemini-2.5-pro","gemini-1.5-flash","gemini-1.5-pro"]
        env_model = os.getenv("GEMINI_MODEL","") or os.getenv("GOOGLE_GEMINI_MODEL","")
        self.gemini_model = env_model if env_model in allowed else "gemini-2.5-flash"

    def _client(self):
        import google.genai as genai
        return genai.Client(api_key=self.api_key) if self.api_key else genai.Client()

    def _extract_text(self, resp) -> str:
        """Robustly extract text from any Gemini response shape."""
        # Primary: .text attribute (most common)
        if hasattr(resp, "text") and resp.text:
            return str(resp.text).strip()
        # Secondary: candidates[0].content.parts[0].text
        try:
            cand = resp.candidates[0]
            if hasattr(cand, "content"):
                content = cand.content
                if hasattr(content, "parts") and content.parts:
                    return str(content.parts[0].text).strip()
                if hasattr(content, "text"):
                    return str(content.text).strip()
        except Exception:
            pass
        # Tertiary: dict-style
        try:
            if isinstance(resp, dict):
                return resp.get("text","") or str(resp)
        except Exception:
            pass
        return str(resp).strip()

    def _gemini_call(self, system: str, user_prompt: str, max_tokens: int = 600, temp: float = 0.3) -> str:
        """Single unified Gemini call with proper config."""
        import google.genai as genai
        client = self._client()
        full_prompt = f"{system}\n\n---\n\n{user_prompt}"
        resp = client.models.generate_content(
            model=self.gemini_model,
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=max_tokens,
            ),
        )
        return self._extract_text(resp)

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_natural_language(self, text: str) -> dict:
        if not self.api_key:
            return {"error": "No API key"}
        try:
            raw = self._gemini_call(INTAKE_SYSTEM, text, max_tokens=512, temp=0.0)
            raw = re.sub(r"```json|```", "", raw).strip()
            return self._fill_defaults(json.loads(raw))
        except Exception as e:
            return {"error": str(e)}

    def from_form(self, form_data: dict) -> dict:
        return self._fill_defaults(form_data)

    def _fill_defaults(self, data: dict) -> dict:
        defaults = {
            "name":"User","age":30,"monthly_income":80_000,"monthly_expenses":45_000,
            "monthly_emi":0,"current_savings":200_000,"monthly_investment":10_000,
            "retirement_age":50,"risk_profile":"moderate","goals":["emergency","retirement"],
            "tax_regime":"new","income_growth_pct":10,
        }
        result = {}
        for k, v in defaults.items():
            val = data.get(k)
            result[k] = val if (val is not None and val != "" and val != []) else v
        surplus = result["monthly_income"] - result["monthly_expenses"] - result["monthly_emi"]
        result["monthly_investment"] = min(result["monthly_investment"], max(0, surplus * 0.9))
        return result

    def generate_explanation(self, mc_result, user_data: dict) -> str:
        """Generate 3-4 sentence personalised explanation. Raises token limit to get full response."""
        if not self.api_key:
            return self._template_explanation(mc_result, user_data)
        try:
            from utils.financial_math import fmt_inr
            prob = mc_result.success_probability * 100
            inv_rate = user_data.get("monthly_investment",0) / max(user_data.get("monthly_income",1),1) * 100
            prompt = f"""User financial profile:
- Name: {user_data.get('name','User')}, Age: {user_data.get('age')}
- Monthly income: {fmt_inr(user_data.get('monthly_income',0))}
- Monthly SIP: {fmt_inr(user_data.get('monthly_investment',0))} ({inv_rate:.1f}% of income)
- Current savings: {fmt_inr(user_data.get('current_savings',0))}
- Monthly surplus: {fmt_inr(user_data.get('monthly_income',0)-user_data.get('monthly_expenses',0)-user_data.get('monthly_emi',0))}
- Target retirement age: {user_data.get('retirement_age',50)}
- Monte Carlo probability: {prob:.1f}%
- Target corpus needed: {fmt_inr(mc_result.target_corpus)}
- Median projected wealth: {fmt_inr(mc_result.median_final_wealth)}
- Key drivers: {[d['factor'] for d in mc_result.key_drivers]}

Task: Write exactly 3 sentences explaining WHY the probability is {prob:.0f}%.
Sentence 1: State the probability and the #1 reason (use specific rupee numbers from the profile).
Sentence 2: Identify the biggest gap or strength (use specific numbers).
Sentence 3: Give ONE concrete action with a specific rupee amount and timeline.
Tone: direct financial advisor, not cheerful bot. No preamble. Start directly with the explanation."""
            result = self._gemini_call(FINANCIAL_CHAT_SYSTEM, prompt, max_tokens=400, temp=0.2)
            # Strip SEBI disclaimer if Gemini added it at the end of explanation
            result = result.replace("⚖️ Educational content only — not SEBI-registered investment advice.", "").strip()
            return result
        except Exception:
            return self._template_explanation(mc_result, user_data)

    def get_tax_advice(self, user_data: dict, extra_inputs: dict) -> str:
        """Full step-by-step tax regime comparison."""
        if not self.api_key:
            return self._template_tax_advice(user_data, extra_inputs)
        try:
            from utils.financial_math import fmt_inr
            annual = user_data["monthly_income"] * 12
            prompt = f"""Calculate tax for this user's salary structure:
- Annual gross income: ₹{annual:,.0f}
- HRA received: ₹{extra_inputs.get('hra_annual',0):,.0f}/yr
- City type: {extra_inputs.get('city','metro')} (affects HRA: metro=50% basic, non-metro=40% basic)
- Basic salary: approximately 40% of gross = ₹{annual*0.4:,.0f}
- 80C investments (ELSS/PPF/EPF): ₹{extra_inputs.get('elss_annual',0):,.0f}/yr (max allowed ₹1,50,000)
- NPS 80CCD(1B) additional: ₹{extra_inputs.get('nps_annual',0):,.0f}/yr (max ₹50,000)
- Home loan interest paid (Sec 24b): ₹{extra_inputs.get('loan_interest',0):,.0f}/yr (max ₹2,00,000)
- Assumed rent paid for HRA: ₹{extra_inputs.get('hra_annual',0)*1.3:,.0f}/yr

Please calculate step by step:
1. Old regime: Show each deduction line item with amount, then taxable income, then slab-wise tax, then cess
2. New regime: Show standard deduction, taxable income, slab-wise tax, cess
3. Compare both: state which saves more and by exactly how much per year and per month
4. List any deductions not fully utilised with exact potential saving
5. Recommend 3 instruments to optimise further (with liquidity and risk rating)"""
            return self._gemini_call(TAX_ADVISOR_SYSTEM, prompt, max_tokens=1200, temp=0.1)
        except Exception:
            return self._template_tax_advice(user_data, extra_inputs)

    def chat_response(self, question: str, user_data: dict, mc_result=None) -> str:
        """Full contextual AI chat with complete user profile injected."""
        if not self.api_key:
            return "Please set your Gemini API key in the sidebar to enable AI chat."
        try:
            from utils.financial_math import fmt_inr
            surplus = user_data.get('monthly_income',0) - user_data.get('monthly_expenses',0) - user_data.get('monthly_emi',0)
            context = f"""Complete user financial profile (use these numbers in your answer):
- Name: {user_data.get('name','User')}, Age: {user_data.get('age')}
- Monthly income: {fmt_inr(user_data.get('monthly_income',0))}
- Monthly expenses: {fmt_inr(user_data.get('monthly_expenses',0))}
- Monthly EMI: {fmt_inr(user_data.get('monthly_emi',0))}
- Monthly surplus (investable): {fmt_inr(surplus)}
- Current SIP/investment: {fmt_inr(user_data.get('monthly_investment',0))}/month ({user_data.get('monthly_investment',0)/max(user_data.get('monthly_income',1),1)*100:.1f}% of income)
- Total savings: {fmt_inr(user_data.get('current_savings',0))}
- Risk profile: {user_data.get('risk_profile','moderate')}
- Goals: {', '.join(user_data.get('goals',[]))}
- Target retirement age: {user_data.get('retirement_age',50)}"""
            if mc_result:
                context += f"\n- Goal success probability: {mc_result.success_probability:.0%}"
                context += f"\n- Target corpus: {fmt_inr(mc_result.target_corpus)}"
                context += f"\n- Median projected wealth: {fmt_inr(mc_result.median_final_wealth)}"
                if hasattr(mc_result, 'sip_needed_for_goal'):
                    context += f"\n- SIP needed for primary goal: {fmt_inr(mc_result.sip_needed_for_goal)}/month"

            prompt = f"{context}\n\nUser question: {question}\n\nAnswer this question using the specific numbers from the profile above. Be concrete, honest, and give actionable steps."
            return self._gemini_call(FINANCIAL_CHAT_SYSTEM, prompt, max_tokens=600, temp=0.3)
        except Exception as e:
            return f"AI response error: {str(e)}. Please check your API key."

    # ── Fallback templates ─────────────────────────────────────────────────────

    def _template_explanation(self, mc_result, user_data: dict) -> str:
        from utils.financial_math import fmt_inr
        prob  = mc_result.success_probability * 100
        inv   = user_data.get("monthly_investment", 0)
        inc   = max(user_data.get("monthly_income", 1), 1)
        irate = inv / inc * 100
        med   = mc_result.median_final_wealth
        tgt   = mc_result.target_corpus
        gap   = max(0, tgt - med)
        if prob >= 70:
            return (f"Your {prob:.0f}% probability is driven by a strong {irate:.0f}% investment rate "
                    f"({fmt_inr(inv)}/mo) and your existing savings of {fmt_inr(user_data.get('current_savings',0))}. "
                    f"Median projection: {fmt_inr(med)} — above target. "
                    f"Add a 10% annual SIP step-up to push this above 85%.")
        elif prob >= 40:
            return (f"At {prob:.0f}%, you're on a moderate path — investing {irate:.0f}% of income vs the 20% target. "
                    f"Your {fmt_inr(inv)}/mo SIP builds {fmt_inr(med)}, leaving a {fmt_inr(gap)} gap to the goal. "
                    f"Increase SIP by {fmt_inr(inc*0.20-inv)}/mo to close this gap.")
        else:
            return (f"Only {prob:.0f}% — critical gap. Your {fmt_inr(inv)}/mo SIP builds only {fmt_inr(med)} "
                    f"vs {fmt_inr(tgt)} needed — a {fmt_inr(gap)} shortfall. "
                    f"Increase SIP to at least {fmt_inr(inc*0.20)}/mo ({fmt_inr(inc*0.20-inv)} more) immediately.")

    def _template_tax_advice(self, user_data: dict, extra: dict) -> str:
        from utils.financial_math import (tax_liability_new_regime, tax_liability_old_regime, fmt_inr)
        annual   = user_data["monthly_income"] * 12
        elss     = extra.get("elss_annual", 0)
        nps      = extra.get("nps_annual", 0)
        home_int = extra.get("loan_interest", 0)
        hra      = extra.get("hra_annual", 0)
        hra_exempt = min(hra, int(annual * 0.40 * 0.50))
        new_t    = tax_liability_new_regime(annual)
        old_t    = tax_liability_old_regime(annual, elss, nps + home_int + hra_exempt)
        winner   = "Old" if old_t < new_t else "New"
        saving   = abs(new_t - old_t)
        lines    = [
            f"**New Regime Tax:** {fmt_inr(new_t)}/yr (₹75K standard deduction only)",
            f"**Old Regime Tax:** {fmt_inr(old_t)}/yr (80C={fmt_inr(elss)}, NPS={fmt_inr(nps)}, "
            f"home interest={fmt_inr(home_int)}, HRA exempt={fmt_inr(hra_exempt)})",
            f"\n✅ **{winner} Regime saves you {fmt_inr(saving)}/yr ({fmt_inr(saving/12)}/mo)**",
        ]
        missed = []
        if elss < 150_000: missed.append(f"80C: ₹{fmt_inr(150_000-elss)} more available → saves ~{fmt_inr((150_000-elss)*0.20)}/yr")
        if nps < 50_000:   missed.append(f"NPS 80CCD(1B): ₹{fmt_inr(50_000-nps)} more → saves ~{fmt_inr((50_000-nps)*0.20)}/yr")
        if elss + nps + home_int == 0:
            missed.append("80D health insurance: up to ₹25K deduction → saves ~₹5K/yr")
        if missed:
            lines += ["\n**Missed deductions:**"] + [f"• {m}" for m in missed]
        lines.append("\n*Educational guidance only. Consult a CA for personalised tax filing.*")
        return "\n".join(lines)
