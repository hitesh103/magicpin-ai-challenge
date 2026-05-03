"""
composer.py — LLM-powered message composition engine.

Takes the 4-context bundle (category, merchant, trigger, customer)
and produces a ComposedMessage using an LLM.

Key design decisions:
- Trigger-kind dispatch: different prompt angles per trigger type
- Context extraction helpers: pull the most relevant facts before LLM call
- Post-LLM validation: check CTA shape, language, no URLs
- Temperature=0 for determinism
- Supports: OpenAI, Anthropic, Gemini, Mistral
"""

import os
import json
import re
import time
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM Provider setup
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

VALID_CTA_TYPES = {
    "open_ended", "binary_yes_no", "binary_confirm_cancel",
    "multi_choice_slot", "none"
}

TRIGGER_STRATEGY_MAP = {
    # External triggers
    "research_digest":        "research_citation",
    "regulation_change":      "compliance_deadline",
    "competitor_opened":      "competitor_curiosity",
    "festival_upcoming":      "festival_prep",
    "ipl_match_today":        "event_realtime",
    "category_seasonal":      "seasonal_action",
    "cde_opportunity":        "professional_dev",
    "supply_alert":           "urgent_compliance",

    # Internal merchant triggers
    "perf_dip":               "empathetic_reframe",
    "perf_spike":             "celebrate_capitalize",
    "milestone_reached":      "celebrate_next",
    "dormant_with_vera":      "curiosity_reactivate",
    "curious_ask_due":        "asking_merchant",
    "active_planning_intent": "action_continuation",
    "renewal_due":            "soft_urgency_renewal",
    "winback_eligible":       "winback_reframe",
    "gbp_unverified":         "quick_win_gbp",
    "review_theme_emerged":   "review_insight",
    "seasonal_perf_dip":      "seasonal_reassurance",

    # Customer triggers
    "recall_due":             "slot_offering_recall",
    "customer_lapsed_hard":   "winback_no_shame",
    "chronic_refill_due":     "prefilled_refill",
    "trial_followup":         "trial_conversion",
    "wedding_package_followup": "bridal_followup",
}

STRATEGY_HINTS = {
    "research_citation": (
        "Frame this as a colleague sharing a relevant research finding. "
        "Lead with the specific journal/source and numbers. Anchor this to the merchant's customer aggregate (e.g., 'relevant to your X high-risk patients'). "
        "Offer to pull the abstract + draft a patient-ed post. Use open_ended CTA."
    ),
    "compliance_deadline": (
        "Frame as a time-sensitive compliance alert. Lead with the deadline date and what changes. "
        "Use loss aversion: explain the risk of non-compliance. Tease the top 1-2 items of the SOP/checklist to spike curiosity. "
        "Offer a concrete action (audit checklist, updated SOP). Use binary_yes_no CTA."
    ),
    "competitor_curiosity": (
        "Use voyeur curiosity. Reference the competitor's offer vs merchant's. "
        "Anchor to the merchant's Locality (e.g., 'A new competitor in Sector 14'). "
        "Ask if they want to see full details. Use open_ended CTA."
    ),
    "festival_prep": (
        "Connect the festival to the specific category's opportunity. "
        "Use loss aversion: quantify missed views if they don't post. "
        "Offer a specific action (GBP post). Use binary_yes_no CTA."
    ),
    "event_realtime": (
        "This is time-sensitive. Give data-backed advice about the event. "
        "Anchor to the merchant's performance spikes/dips. "
        "Offer a concrete deliverable. Use binary_yes_no CTA."
    ),
    "seasonal_action": (
        "Translate the seasonal shift into concrete action. "
        "Lead with the % demand change. Anchor to the merchant's seasonal beats. "
        "Use binary_yes_no CTA."
    ),
    "professional_dev": (
        "Position as a peer sharing a professional opportunity. "
        "Lead with credits and cost. Anchor to the merchant's Identity/City. "
        "Use binary_yes_no CTA."
    ),
    "urgent_compliance": (
        "This is urgent. Lead with specific batch numbers. "
        "Anchor to the merchant's customer count (e.g., 'X of your Y customers'). "
        "Use binary_yes_no CTA."
    ),
    "empathetic_reframe": (
        "Acknowledge the dip without alarm. Reference peer benchmarks (e.g., 'Others in Locality also saw X%'). "
        "Use loss aversion: quantify what can be recovered. Propose a counter-measure. "
        "Use binary_yes_no CTA."
    ),
    "celebrate_capitalize": (
        "Celebrate briefly, then pivot to capitalization. "
        "Identify driver. Anchor to the merchant's views/calls spike. "
        "Propose a follow-on. Use binary_yes_no CTA."
    ),
    "celebrate_next": (
        "Celebrate the milestone with a specific number. "
        "Anchor to the merchant's review themes. Propose next milestone. "
        "Use open_ended or binary_yes_no CTA."
    ),
    "curiosity_reactivate": (
        "Light re-engagement. Ask a single curious question. "
        "Anchor to a specific merchant Signal (e.g., 'Since your last post 22 days ago...'). "
        "Use open_ended CTA."
    ),
    "asking_merchant": (
        "Ask one question about their business. "
        "Anchor to the merchant's local context or category trends. "
        "Offer to turn the answer into a GBP post. Use open_ended CTA."
    ),
    "action_continuation": (
        "Merchant has shown intent. Do NOT ask qualifying questions. "
        "Immediately present a drafted artifact. Anchor to the merchant's previous message. "
        "Use binary_confirm_cancel CTA."
    ),
    "soft_urgency_renewal": (
        "Frame as a reminder with specific days remaining. "
        "Use loss aversion: highlight visibility/offers lost. Use binary_yes_no CTA."
    ),
    "winback_reframe": (
        "Subscription lapsed. Lead with what they're missing (quantify lapsed customers/perf dip). "
        "Anchor to their previous high performance. Use binary_yes_no CTA."
    ),
    "quick_win_gbp": (
        "Frame as a quick win with specific uplift estimate. "
        "Use social proof: compare to verified peers in Locality. Use binary_yes_no CTA."
    ),
    "review_insight": (
        "Surface the specific review theme with count. "
        "Anchor to the merchant's reputation (avg_rating). Propose fix/amplification. "
        "Use binary_yes_no CTA."
    ),
    "seasonal_reassurance": (
        "Reassure dip is normal (peer range in Locality). "
        "Shift focus to retention. Anchor to their active customer base. "
        "Use binary_yes_no CTA."
    ),
    "slot_offering_recall": (
        "Goes to CUSTOMER. Mention exact months since last visit. "
        "Anchor to their specific service history. List slot options EXACTLY as provided in the 'WHY MESSAGING NOW' section. "
        "Include the doctor's/owner's name (e.g., 'Dr. {FirstName}') for warmth. Use multi_choice_slot CTA."
    ),
    "winback_no_shame": (
        "Goes to CUSTOMER. Warmly acknowledge gap. "
        "Anchor to their previous goal/service. Offer new thing. "
        "Use binary_yes_no CTA."
    ),
    "prefilled_refill": (
        "Goes to CUSTOMER. List molecules, date, price, savings. "
        "Anchor to their chronic condition. Use CONFIRM CTA."
    ),
    "trial_conversion": (
        "Goes to CUSTOMER. Reference trial date. "
        "Anchor to their trial feedback. List next options. "
        "Use binary_yes_no CTA."
    ),
    "bridal_followup": (
        "Goes to CUSTOMER. Count down days to wedding. "
        "Anchor to their bridal package. Offer to block slot. "
        "Use binary_yes_no CTA."
    ),
}

# ---------------------------------------------------------------------------
# Context extraction helpers
# ---------------------------------------------------------------------------

def get_relevant_digest(category: dict, trigger: dict) -> str:
    """Find the digest item referenced by the trigger or the most relevant one."""
    top_id = trigger.get("payload", {}).get("top_item_id") or trigger.get("payload", {}).get("digest_item_id")
    for item in category.get("digest", []):
        if item.get("id") == top_id:
            parts = [item["title"]]
            if item.get("source"):
                parts.append(f"[{item['source']}]")
            if item.get("trial_n"):
                parts.append(f"({item['trial_n']:,} participants)")
            if item.get("summary"):
                parts.append(f"Summary: {item['summary']}")
            if item.get("actionable"):
                parts.append(f"Action: {item['actionable']}")
            return " ".join(parts)
    # No direct match — return first digest item if any
    if category.get("digest"):
        item = category["digest"][0]
        return f"{item['title']} [{item.get('source', '')}]"
    return "No digest item available"


def get_active_offers(merchant: dict) -> str:
    active = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    if active:
        return ", ".join(active)
    return "None currently"


def get_peer_comparison(category: dict, merchant: dict) -> str:
    peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0)
    merchant_ctr = merchant.get("performance", {}).get("ctr", 0)
    peer_calls = category.get("peer_stats", {}).get("avg_calls_30d", 0)
    merchant_calls = merchant.get("performance", {}).get("calls", 0)
    lines = []
    if peer_ctr and merchant_ctr:
        diff = merchant_ctr - peer_ctr
        sign = "+" if diff >= 0 else ""
        lines.append(f"CTR: merchant={merchant_ctr:.3f} vs peer={peer_ctr:.3f} ({sign}{diff:.3f})")
    if peer_calls and merchant_calls:
        diff = merchant_calls - peer_calls
        sign = "+" if diff >= 0 else ""
        lines.append(f"Calls: merchant={merchant_calls} vs peer={peer_calls} ({sign}{diff})")
    return " | ".join(lines) if lines else "No peer comparison available"


def get_seasonal_beat(category: dict) -> str:
    month = date.today().month
    month_ranges = {
        "jan": [1], "feb": [2], "mar": [3], "apr": [4],
        "may": [5], "jun": [6], "jul": [7], "aug": [8],
        "sep": [9], "oct": [10], "nov": [11], "dec": [12],
    }
    for beat in category.get("seasonal_beats", []):
        rng = beat.get("month_range", "").lower()
        for abbr, months in month_ranges.items():
            if abbr in rng and month in months:
                return beat["note"]
    return ""


def get_last_conversation_summary(merchant: dict) -> str:
    history = merchant.get("conversation_history", [])
    if not history:
        return "No prior Vera conversation"
    last = history[-1]
    return f'{last["from"].capitalize()} said: "{last["body"][:80]}..." (engagement: {last.get("engagement", "?")})'


def get_customer_block(customer: dict | None) -> str:
    if not customer:
        return ""
    identity = customer.get("identity", {})
    rel = customer.get("relationship", {})
    state = customer.get("state", "unknown")
    prefs = customer.get("preferences", {})

    # Calculate months since last visit
    months_since = ""
    last_visit = rel.get("last_visit", "")
    if last_visit:
        try:
            lv = datetime.strptime(last_visit[:10], "%Y-%m-%d")
            months = (date.today() - lv.date()).days // 30
            months_since = f"{months} months"
        except Exception:
            months_since = ""

    return f"""
CUSTOMER (send message on behalf of merchant):
  Name: {identity.get("name", "Customer")}
  Language preference: {identity.get("language_pref", "en")}
  State: {state}
  Last visit: {last_visit} ({months_since} ago)
  Visits total: {rel.get("visits_total", "?")}
  Services received: {", ".join(rel.get("services_received", []))}
  Preferred slots: {prefs.get("preferred_slots", "any")}
  Consent scope: {", ".join(customer.get("consent", {}).get("scope", []))}
""".strip()


def get_trigger_why_now(trigger: dict) -> str:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    urgency = trigger.get("urgency", 1)

    def format_slots(slots):
        if not slots: return "None"
        return ", ".join([s.get("label", s.get("iso", "?")) for s in slots])

    mapping = {
        "research_digest":        f"New research digest published for {payload.get('category', 'this category')}",
        "regulation_change":      f"Compliance deadline: {payload.get('deadline_iso', 'soon')}. Regulation: {payload.get('top_item_id', '')}",
        "recall_due":             f"Patient/customer recall window just opened. Service due: {payload.get('service_due', '?')}. Last service: {payload.get('last_service_date', '?')}. Recommended slots: {format_slots(payload.get('available_slots', []))}",
        "perf_dip":               f"Performance dropped {abs(payload.get('delta_pct', 0)*100):.0f}% for {payload.get('metric', 'key metric')} over {payload.get('window', '7d')}",
        "perf_spike":             f"Performance spiked +{payload.get('delta_pct', 0)*100:.0f}% for {payload.get('metric', 'key metric')} over {payload.get('window', '7d')}. Likely driver: {payload.get('likely_driver', 'unknown')}",
        "milestone_reached":      f"Merchant approaching {payload.get('milestone_value', '?')} {payload.get('metric', 'milestone')} (currently at {payload.get('value_now', '?')})",
        "dormant_with_vera":      f"No merchant message in {payload.get('days_since_last_merchant_message', '?')} days. Last topic: {payload.get('last_topic', 'unknown')}",
        "curious_ask_due":        f"Weekly curious-ask cadence. Template: {payload.get('ask_template', 'what_is_in_demand')}",
        "active_planning_intent": f"Merchant explicitly showed intent: \"{payload.get('merchant_last_message', '')}\" — about topic: {payload.get('intent_topic', '?')}",
        "renewal_due":            f"Subscription expires in {payload.get('days_remaining', '?')} days. Plan: {payload.get('plan', '?')}. Renewal amount: ₹{payload.get('renewal_amount', '?')}",
        "winback_eligible":       f"Subscription expired {payload.get('days_since_expiry', '?')} days ago. Perf dip: {payload.get('perf_dip_pct', 0)*100:.0f}%. Lapsed customers since expiry: {payload.get('lapsed_customers_added_since_expiry', '?')}",
        "gbp_unverified":         f"GBP is not verified. Estimated uplift from verification: {payload.get('estimated_uplift_pct', 0)*100:.0f}% more calls",
        "review_theme_emerged":   f"Review theme '{payload.get('theme', '?')}' has {payload.get('occurrences_30d', '?')} occurrences in 30d (trend: {payload.get('trend', '?')}). Quote: \"{payload.get('common_quote', '')}\"",
        "ipl_match_today":        f"IPL match: {payload.get('match', '?')} at {payload.get('venue', '?')}, {payload.get('match_time_iso', '?')}. Weeknight: {payload.get('is_weeknight', '?')}",
        "festival_upcoming":      f"Festival: {payload.get('festival', '?')} on {payload.get('date', '?')} ({payload.get('days_until', '?')} days away)",
        "supply_alert":           f"Supply alert: {payload.get('molecule', '?')} batches {payload.get('affected_batches', [])} recalled by {payload.get('manufacturer', '?')}",
        "chronic_refill_due":     f"Chronic refill due. Molecules: {', '.join(payload.get('molecule_list', []))}. Stock runs out: {payload.get('stock_runs_out_iso', '?')}. Delivery address saved: {payload.get('delivery_address_saved', False)}",
        "customer_lapsed_hard":   f"Customer lapsed {payload.get('days_since_last_visit', '?')} days ago. Previous focus: {payload.get('previous_focus', '?')}. Previous membership: {payload.get('previous_membership_months', '?')} months",
        "trial_followup":         f"Customer completed trial on {payload.get('trial_date', '?')}. Next session options: {payload.get('next_session_options', [])}. Recommended slots: {format_slots(payload.get('available_slots', []))}",
        "wedding_package_followup": f"Wedding date: {payload.get('wedding_date', '?')} ({payload.get('days_to_wedding', '?')} days away). Trial completed: {payload.get('trial_completed', '?')}. Next window: {payload.get('next_step_window_open', '?')}",
        "seasonal_perf_dip":      f"Seasonal dip: {payload.get('metric', '?')} down {abs(payload.get('delta_pct', 0)*100):.0f}% over {payload.get('window', '?')}. Expected seasonal: {payload.get('is_expected_seasonal', True)}. Note: {payload.get('season_note', '')}",
        "category_seasonal":      f"Seasonal trends: {payload.get('trends', [])}. Season: {payload.get('season', '?')}",
        "cde_opportunity":        f"CDE webinar: {payload.get('credits', '?')} credits. Fee: {payload.get('fee', '?')}",
        "competitor_opened":      f"New competitor '{payload.get('competitor_name', '?')}' opened {payload.get('distance_km', '?')}km away on {payload.get('opened_date', '?')}. Their offer: {payload.get('their_offer', '?')}",
    }

    desc = mapping.get(kind, f"Trigger kind: {kind} | Payload: {json.dumps(payload)[:200]}")
    return f"[Urgency {urgency}/5] {desc}"


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, retries: int = 3) -> str:
    """Call the configured LLM provider. Returns raw text. Retries on rate limit."""

    for attempt in range(retries):
        try:
            return _call_llm_once(system_prompt, user_prompt)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "Rate limit" in err_str
            if is_rate_limit and attempt < retries - 1:
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                print(f"[RATE LIMIT] Retrying in {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("LLM call failed after all retries")


def _call_llm_once(system_prompt: str, user_prompt: str) -> str:
    """Single LLM call — do not call directly; use call_llm() for retry logic."""

    if LLM_PROVIDER in ("openai", "mistral"):
        from openai import OpenAI
        # Mistral is OpenAI-compatible — just swap the base_url
        if LLM_PROVIDER == "mistral":
            client = OpenAI(
                api_key=LLM_API_KEY,
                base_url="https://api.mistral.ai/v1",
            )
            model = LLM_MODEL or "mistral-large-latest"
            # Mistral supports JSON mode via response_format
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
        else:
            client = OpenAI(api_key=LLM_API_KEY)
            model = LLM_MODEL or "gpt-4o-mini"
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
        return resp.choices[0].message.content

    elif LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=LLM_API_KEY)
        model = LLM_MODEL or "claude-3-5-sonnet-20241022"
        msg = client.messages.create(
            model=model,
            max_tokens=700,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        return msg.content[0].text

    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=LLM_API_KEY)
        model_name = LLM_MODEL or "gemini-1.5-flash"
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                max_output_tokens=700,
                response_mime_type="application/json",
            ),
        )
        resp = model.generate_content(user_prompt)
        return resp.text

    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}. Supported: openai, mistral, anthropic, gemini")


# ---------------------------------------------------------------------------
# SYSTEM PROMPT (core Vera persona)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Vera, magicpin's AI marketing assistant for merchants. You compose short, highly personalized WhatsApp messages.

ABSOLUTE RULES (violating any = score of 0):
1. NO URLs in the message body.
2. Every message must contain at least 1 verifiable fact from the provided context (number, date, source citation, peer stat).
3. Do NOT fabricate data. If a number isn't in the context, don't cite it.
4. ONE primary CTA at the very end of the message. NOT in the middle.
5. Never re-introduce yourself after turn 1.
6. ANCHORING: Always cross-reference the primary trigger with at least one other Merchant Signal (e.g., stale posts, low CTR, high lapse rate) to make the message feel "why me" and "why now".

VOICE RULES BY CATEGORY:
- dentists: peer_clinical — technical vocabulary OK (fluoride varnish, caries, OPG). Taboo: "guaranteed", "100% safe". Address as "Dr. {FirstName}".
- salons: warm_practical — fellow operator tone. Use owner first name. Emojis OK.
- restaurants: operator_to_operator — use "covers", "AOV", "delivery radius". No hype.
- gyms: coaching_motivational — encourage, no guilt, use "members", "conversion".
- pharmacies: trustworthy_precise — precise molecule names, batch numbers. No alarming language.

LANGUAGE RULES:
- If merchant/customer language includes "hi": use Hindi-English code-mix naturally (not forced).
- Match the merchant's communication style from conversation history.

COMPULSION LEVERS (use 1-2 per message):
- Specificity: concrete number, date, citation (JIDA p.14, 38%, 2,100 patients)
- Loss aversion: Quantify the cost of inaction (e.g., "you're missing ~50 views/week", "78 patients haven't seen an update in 22 days"). Be EXPLICIT.
- Social proof: Use Locality and Peer Stats (e.g., "Other clinics in Lajpat Nagar are seeing 3% CTR").
- Effort externalization: "I've drafted this — just say go"
- Curiosity: "want to see who?" / "want the full list?"
- Reciprocity: "I noticed X about your account, thought you'd want to know"
- Asking the merchant: "what's your most-asked service this week?"
- Single binary commitment: Reply YES / STOP

LENGTH: 50-130 words ideal. Concise. No long preambles.

OUTPUT FORMAT (JSON only, no other text):
{
  "body": "<the WhatsApp message>",
  "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "<from trigger>",
  "rationale": "<1 sentence: what hook/lever was used and why>"
}"""


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------

def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """
    Core composition function.
    Inputs are plain dicts loaded from the dataset JSON.
    Returns a dict with keys: body, cta, send_as, suppression_key, rationale.
    """
    trigger_kind = trigger.get("kind", "unknown")
    strategy = TRIGGER_STRATEGY_MAP.get(trigger_kind, "asking_merchant")
    strategy_hint = STRATEGY_HINTS.get(strategy, "Compose a relevant, specific message.")

    # Determine send_as
    send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"

    user_prompt = f"""You are about to compose a WhatsApp message. Follow these steps:
1. IDENTIFY relevant facts from the Category/Merchant context.
2. ANCHOR the primary Trigger to at least one other Signal or performance metric.
3. Quantify Loss Aversion if applicable.
4. COMPOSE the message following the Category Voice and Absolute Rules.

COMPOSE A MESSAGE using this strategy: {strategy}
Strategy guidance: {strategy_hint}

WHY MESSAGING NOW:
{get_trigger_why_now(trigger)}

CATEGORY ({category.get("slug", "unknown")}):
  Voice/tone: {category.get("voice", {}).get("tone", "professional")}
  Taboo words: {", ".join(category.get("voice", {}).get("vocab_taboo", [])[:5])}
  Key terminology: {", ".join(category.get("voice", {}).get("vocab_allowed", [])[:8])}
  Active offer catalog: {json.dumps([o["title"] for o in category.get("offer_catalog", [])[:4]])}
  Peer stats: avg_CTR={category.get("peer_stats", {}).get("avg_ctr", "?")}, avg_calls_30d={category.get("peer_stats", {}).get("avg_calls_30d", "?")}, avg_rating={category.get("peer_stats", {}).get("avg_rating", "?")}
  Peer comparison: {get_peer_comparison(category, merchant)}
  Most relevant digest: {get_relevant_digest(category, trigger)}
  Seasonal beat (current month): {get_seasonal_beat(category) or "None"}

MERCHANT:
  Business name: {merchant.get("identity", {}).get("name", "?")}
  Owner first name: {merchant.get("identity", {}).get("owner_first_name", "")}
  City, Locality: {merchant.get("identity", {}).get("locality", "?")}, {merchant.get("identity", {}).get("city", "?")}
  Languages: {", ".join(merchant.get("identity", {}).get("languages", ["en"]))}
  Verified GBP: {merchant.get("identity", {}).get("verified", False)}
  Subscription: {merchant.get("subscription", {}).get("status", "?")} — {merchant.get("subscription", {}).get("days_remaining", "?")} days remaining (plan: {merchant.get("subscription", {}).get("plan", "?")})
  Performance (30d): views={merchant.get("performance", {}).get("views", "?")}, calls={merchant.get("performance", {}).get("calls", "?")}, directions={merchant.get("performance", {}).get("directions", "?")}, CTR={merchant.get("performance", {}).get("ctr", "?")}
  7-day delta: {json.dumps(merchant.get("performance", {}).get("delta_7d", {}))}
  Active offers: {get_active_offers(merchant)}
  Customer aggregate: {json.dumps(merchant.get("customer_aggregate", {}))}
  Signals: {", ".join(merchant.get("signals", []))}
  Review themes: {json.dumps(merchant.get("review_themes", [])[:3])}
  Last Vera interaction: {get_last_conversation_summary(merchant)}

{get_customer_block(customer) if customer else "CUSTOMER: None (merchant-facing message)"}

TRIGGER SUPPRESSION KEY: {trigger.get("suppression_key", "")}

IMPORTANT: send_as should be "{send_as}" based on trigger scope.

Return ONLY the JSON object. Do not include your reasoning in the final JSON, only use it to inform the 'body' and 'rationale'."""

    raw = call_llm(SYSTEM_PROMPT, user_prompt)

    # Parse and validate
    result = _parse_and_validate(raw, trigger, send_as)
    return result


def _parse_and_validate(raw: str, trigger: dict, send_as: str) -> dict:
    """Parse LLM output and enforce constraints."""
    # Extract JSON
    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        return _fallback_message(trigger, send_as)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return _fallback_message(trigger, send_as)

    body = data.get("body", "").strip()
    cta = data.get("cta", "open_ended")
    rationale = data.get("rationale", "")

    # Enforce: no URLs
    if re.search(r'https?://', body):
        body = re.sub(r'https?://\S+', '', body).strip()

    # Enforce: valid CTA type
    if cta not in VALID_CTA_TYPES:
        cta = "open_ended"

    # Enforce: send_as
    result_send_as = data.get("send_as", send_as)
    if result_send_as not in ("vera", "merchant_on_behalf"):
        result_send_as = send_as

    return {
        "body": body,
        "cta": cta,
        "send_as": result_send_as,
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": rationale or "Composed from 4-context framework",
    }


def _fallback_message(trigger: dict, send_as: str) -> dict:
    """Emergency fallback if LLM call or parsing fails."""
    return {
        "body": "Quick check-in — kya main kuch helpful bhej sakti hoon aapke liye abhi? Reply YES to continue.",
        "cta": "binary_yes_no",
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", "fallback"),
        "rationale": "Fallback message: LLM composition failed.",
    }
