"""
reply_handler.py — Handle multi-turn merchant/customer replies.

Called when the judge posts a reply to /v1/reply.
Returns action: send | wait | end

Decision flow:
  1. Classify message (auto_reply, hostile, intent_commit, off_topic, engaged)
  2. If auto_reply → deterministic response (no LLM cost)
  3. If hostile → graceful end
  4. If intent_commit → LLM with action_mode instruction
  5. If off_topic → LLM with redirect instruction
  6. If engaged → LLM with continue instruction
"""

import json
import re
from auto_reply_detector import (
    classify_message,
    build_auto_reply_response,
    build_hostile_response,
)
from composer import call_llm, SYSTEM_PROMPT


REPLY_SYSTEM_PROMPT = """You are Vera, continuing an ongoing WhatsApp conversation with a merchant.

RULES:
1. NO URLs in any message.
2. Do NOT re-introduce yourself.
3. Do NOT repeat information already stated in the conversation.
4. Keep replies concise: 40-100 words.
5. ONE CTA at the end only.
6. Match the merchant's language (Hindi-English mix if they use it).

OUTPUT FORMAT (JSON only):
{
  "action": "send" | "wait" | "end",
  "body": "<reply message — required if action=send>",
  "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "none",
  "wait_seconds": <int — required if action=wait>,
  "rationale": "<1 sentence explaining this reply choice>"
}"""


def handle_reply(
    conversation_id: str,
    merchant_id: str,
    customer_id: str | None,
    from_role: str,
    message: str,
    turn_number: int,
    conversation_history: list[dict],
    merchant: dict | None,
    category: dict | None,
) -> dict:
    """
    Process an incoming reply and return the bot's next action.

    Args:
        conversation_id: The conversation ID
        merchant_id: The merchant ID
        customer_id: Customer ID (if customer-facing conversation)
        from_role: "merchant" or "customer"
        message: The incoming message text
        turn_number: Current turn number (1-indexed)
        conversation_history: List of prior turns [{role, body}]
        merchant: MerchantContext payload (may be None)
        category: CategoryContext payload (may be None)

    Returns:
        dict with keys: action, body (if send), cta, wait_seconds (if wait), rationale
    """

    # Count how many times this exact message appeared before
    repeat_count = _count_repeats(message, conversation_history)
    print(f"[DEBUG] handle_reply: msg='{message[:30]}...', repeat_count={repeat_count}, history_len={len(conversation_history)}")

    # --- Step 1: Classify the incoming message ---
    classification = classify_message(message, repeat_count)
    msg_type = classification["type"]
    strategy = classification["strategy"]
    print(f"[DEBUG] handle_reply: classification={classification}")

    # --- Step 2: Route based on classification ---

    # Auto-reply: deterministic response
    if msg_type == "auto_reply":
        return build_auto_reply_response(strategy, conversation_id)

    # Hostile: graceful exit
    if msg_type == "hostile":
        resp = build_hostile_response()
        # After sending the apology, mark for end on next call
        # (the judge will see action=send, then we'll end on the next reply)
        return resp

    # Intent commit, off-topic, engaged: LLM-powered response
    mode_instruction = _get_mode_instruction(msg_type, message, merchant, category)

    history_text = _format_history(conversation_history)
    merchant_name = (merchant or {}).get("identity", {}).get("name", "this merchant")
    owner_name = (merchant or {}).get("identity", {}).get("owner_first_name", "")
    languages = (merchant or {}).get("identity", {}).get("languages", ["en"])
    lang_note = "Use Hindi-English code-mix" if "hi" in languages else "Use English"

    user_prompt = f"""Continue this conversation with {merchant_name}{" (owner: " + owner_name + ")" if owner_name else ""}.

CONVERSATION SO FAR:
{history_text}

LATEST MESSAGE from {from_role}: "{message}"

INSTRUCTION: {mode_instruction}
LANGUAGE NOTE: {lang_note}
TURN NUMBER: {turn_number}

Return ONLY the JSON object."""

    raw = call_llm(REPLY_SYSTEM_PROMPT, user_prompt)
    result = _parse_reply_response(raw)
    return result


def _count_repeats(message: str, history: list[dict]) -> int:
    """Count how many times this message appeared before from merchant side."""
    msg_norm = message.strip().lower()
    merchant_msgs = [
        t.get("body", "").strip().lower()
        for t in history
        if t.get("role") in ("merchant", "customer")
    ]
    return merchant_msgs.count(msg_norm)


def _get_mode_instruction(msg_type: str, message: str, merchant: dict | None, category: dict | None) -> str:
    """Get the LLM instruction based on message classification."""
    if msg_type == "intent_commit":
        return (
            "The merchant has EXPLICITLY committed to moving forward. "
            "DO NOT ask another qualifying question. "
            "Switch to ACTION MODE immediately. "
            "Present the concrete next step, a drafted artifact, or a confirmation request. "
            "Use binary_confirm_cancel CTA."
        )
    elif msg_type == "off_topic":
        return (
            "The merchant asked about something outside Vera's scope. "
            "Politely decline in 1 sentence, then redirect back to the original conversation topic. "
            "Use open_ended CTA."
        )
    else:  # engaged
        return (
            "The merchant is engaged. Advance the conversation by exactly 1 step. "
            "Do NOT repeat anything already said. "
            "Either fulfill their ask or propose the logical next step. "
            "Keep it concise. Match their energy/tone."
        )


def _format_history(history: list[dict]) -> str:
    """Format conversation history for the prompt."""
    lines = []
    for turn in history[-8:]:  # last 8 turns max
        role = turn.get("role", "?").capitalize()
        if role in ("Merchant", "Customer"):
            role = f"Merchant/Customer"
        body = turn.get("body", "")[:150]
        lines.append(f"{role}: {body}")
    return "\n".join(lines) if lines else "(conversation just started)"


def _parse_reply_response(raw: str) -> dict:
    """Parse LLM JSON output for reply actions."""
    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        return _fallback_reply()

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return _fallback_reply()

    action = data.get("action", "send")
    if action not in ("send", "wait", "end"):
        action = "send"

    body = data.get("body", "").strip()

    # Remove URLs if any slipped through
    body = re.sub(r'https?://\S+', '', body).strip()

    cta = data.get("cta", "open_ended")
    if cta not in ("open_ended", "binary_yes_no", "binary_confirm_cancel", "none"):
        cta = "open_ended"

    result = {
        "action": action,
        "rationale": data.get("rationale", "Continued conversation"),
    }

    if action == "send":
        result["body"] = body
        result["cta"] = cta
    elif action == "wait":
        result["wait_seconds"] = int(data.get("wait_seconds", 1800))

    return result


def _fallback_reply() -> dict:
    """Emergency fallback if LLM reply parsing fails."""
    return {
        "action": "send",
        "body": "Got it! Ek minute — main aage ki details prepare kar rahi hoon. Kya main proceed karun?",
        "cta": "binary_yes_no",
        "rationale": "Fallback reply: LLM response parsing failed.",
    }
