"""
bot.py — Vera Bot: FastAPI HTTP server for the magicpin AI Challenge.

Exposes 5 endpoints:
  GET  /v1/healthz    — liveness probe
  GET  /v1/metadata   — bot identity
  POST /v1/context    — receive context push
  POST /v1/tick       — periodic wake-up; bot initiates proactive messages
  POST /v1/reply      — receive merchant/customer reply
  POST /v1/teardown   — optional: wipe all state at end of test

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from context_store import ContextStore
from conversation_manager import ConversationManager
from composer import compose
from reply_handler import handle_reply

load_dotenv()

# ---------------------------------------------------------------------------
# App + global state
# ---------------------------------------------------------------------------
app = FastAPI(title="Vera Bot", version="1.0.0")
START_TIME = time.time()

ctx_store = ContextStore()
conv_manager = ConversationManager()

# Tracks which (merchant_id, trigger_id) combos we've already acted on this session
# to prevent duplicate sends across ticks
_acted_triggers: set[str] = set()

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str = "merchant"
    message: str
    received_at: str = ""
    turn_number: int = 1


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def calculate_trigger_priority(trigger: dict) -> int:
    """Score triggers so we act on the most important ones first."""
    urgency = trigger.get("urgency", 1)
    kind_weights = {
        "supply_alert": 10, "active_planning_intent": 9,
        "regulation_change": 8, "renewal_due": 7,
        "perf_dip": 6, "recall_due": 6, "chronic_refill_due": 6,
        "competitor_opened": 5, "perf_spike": 5, "review_theme_emerged": 5,
        "winback_eligible": 4, "milestone_reached": 4,
        "festival_upcoming": 3, "ipl_match_today": 3, "category_seasonal": 3,
        "research_digest": 3, "cde_opportunity": 2,
        "gbp_unverified": 2, "customer_lapsed_hard": 4,
        "trial_followup": 3, "wedding_package_followup": 3,
        "curious_ask_due": 2, "dormant_with_vera": 2, "seasonal_perf_dip": 1,
    }
    return urgency * 2 + kind_weights.get(trigger.get("kind", ""), 1)


def is_trigger_expired(trigger: dict) -> bool:
    expires_str = trigger.get("expires_at", "")
    if not expires_str:
        return False
    try:
        expires = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > expires
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = ctx_store.count_by_scope()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.getenv("TEAM_NAME", "Team Vera"),
        "team_members": [os.getenv("TEAM_MEMBERS", "Hitesh")],
        "model": os.getenv("LLM_MODEL") or f"{os.getenv('LLM_PROVIDER', 'openai')}-default",
        "approach": (
            "4-context LLM composer with trigger-kind dispatch. "
            "Deterministic auto-reply detection + intent classification before LLM call. "
            "Trigger prioritization by urgency × kind weight."
        ),
        "contact_email": os.getenv("CONTACT_EMAIL", "team@example.com"),
        "version": "1.0.0",
        "submitted_at": now_iso(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    accepted, current_version = ctx_store.upsert(
        body.scope, body.context_id, body.version, body.payload
    )

    if not accepted:
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            },
        )

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": now_iso(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Main proactive engine. Called by judge every 5 simulated minutes.
    Returns list of actions — one per (merchant, trigger) pair we decide to act on.
    """
    actions = []

    # --- 1. Collect and rank active triggers ---
    candidates = []
    for trg_id in body.available_triggers:
        bundle = ctx_store.get_trigger_bundle(trg_id)
        if not bundle:
            continue

        trigger = bundle["trigger"]
        merchant = bundle["merchant"]

        # Skip if expired
        if is_trigger_expired(trigger):
            continue

        # Skip if we already acted on this trigger
        act_key = f"{trigger.get('merchant_id', '')}:{trg_id}"
        if act_key in _acted_triggers:
            continue

        # Skip if suppression key was already used
        suppression_key = trigger.get("suppression_key", "")
        if suppression_key and conv_manager.is_suppressed(suppression_key):
            continue

        # Skip if merchant context is missing (can't compose without it)
        if not merchant:
            continue

        priority = calculate_trigger_priority(trigger)
        candidates.append((priority, trg_id, bundle))

    # Sort by priority (highest first), cap at 20
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:20]

    # --- 2. Define task for parallel execution ---
    async def process_trigger(bundle):
        trigger = bundle["trigger"]
        merchant = bundle["merchant"]
        category = bundle["category"]
        customer = bundle["customer"]

        merchant_id = trigger.get("merchant_id", "")
        customer_id = trigger.get("customer_id")
        conv_id = f"conv_{merchant_id}_{trigger['id']}"

        if conv_manager.is_ended(conv_id):
            return None

        try:
            composed = await compose(category or {}, merchant, trigger, customer)
            body_text = composed.get("body", "").strip()
            if not body_text:
                return None
            
            if conv_manager.is_duplicate_body(conv_id, body_text):
                return None

            # Record turn
            conv_manager.get_or_create(conv_id, merchant_id, customer_id)
            conv_manager.add_bot_turn(conv_id, body_text, composed.get("cta", ""))

            # Suppress key
            suppression_key = trigger.get("suppression_key", "")
            if suppression_key:
                conv_manager.suppress(suppression_key)
            
            # Mark acted
            _acted_triggers.add(f"{merchant_id}:{trigger['id']}")

            # Build template params
            owner_name = merchant.get("identity", {}).get("owner_first_name", "") or merchant.get("identity", {}).get("name", "")
            body_excerpt = body_text[:100]
            cta_hint = "Reply YES to continue" if composed.get("cta") == "binary_yes_no" else "Reply to continue"

            return {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": composed.get("send_as", "vera"),
                "trigger_id": trigger["id"],
                "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
                "template_params": [owner_name, body_excerpt, cta_hint],
                "body": body_text,
                "cta": composed.get("cta", "open_ended"),
                "suppression_key": suppression_key,
                "rationale": composed.get("rationale", ""),
            }
        except Exception as e:
            print(f"[COMPOSE ERROR] {trigger['id']}: {e}")
            return None

    # --- 3. Execute in parallel ---
    import asyncio
    tasks = [process_trigger(bundle) for _, _, bundle in candidates]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    actions = [r for r in results if r is not None]

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Handle a merchant/customer reply in an ongoing conversation.
    Returns: {action: send|wait|end, body?, cta?, wait_seconds?, rationale}
    """
    conv_id = body.conversation_id
    merchant_id = body.merchant_id or ""
    customer_id = body.customer_id

    # If conversation is marked ended, ignore
    if conv_manager.is_ended(conv_id):
        return {"action": "end", "rationale": "Conversation already ended."}

    # Ensure conversation exists
    conv_manager.get_or_create(conv_id, merchant_id, customer_id)

    # Record incoming turn
    conv_manager.add_merchant_turn(conv_id, body.message)

    # Get current history (before this turn was added, so -1)
    all_turns = conv_manager.get_turns(conv_id)
    history_for_handler = all_turns[:-1]  # exclude the turn we just added

    # Get merchant + category context
    merchant = ctx_store.get("merchant", merchant_id) if merchant_id else None
    category_slug = (merchant or {}).get("category_slug")
    category = ctx_store.get("category", category_slug) if category_slug else None

    # Handle the reply
    result = handle_reply(
        conversation_id=conv_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
        conversation_history=history_for_handler,
        merchant=merchant,
        category=category,
    )

    # If bot is sending, record the bot turn
    if result.get("action") == "send":
        bot_body = result.get("body", "")
        if bot_body:
            conv_manager.add_bot_turn(conv_id, bot_body, result.get("cta", ""))

    # If bot is ending, mark conversation ended
    if result.get("action") == "end":
        conv_manager.mark_ended(conv_id)

    return result


@app.post("/v1/teardown")
async def teardown():
    """Optional: wipe all state at end of test."""
    ctx_store.teardown()
    _acted_triggers.clear()
    return {"status": "torn_down", "ts": now_iso()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
