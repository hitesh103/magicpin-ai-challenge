"""
auto_reply_detector.py — Detect WhatsApp Business canned auto-replies.

Production Vera wastes 2-3 turns on auto-replies.
This module detects them in 1-2 turns — a key scoring differentiator.

Strategy:
  Turn 1: Pattern match → one more attempt with explicit "please show this to the owner"
  Turn 2: Same message again → wait (back off 4h)
  Turn 3+: Still auto-reply → end gracefully
"""

AUTO_REPLY_PATTERNS = [
    # English patterns
    "thank you for contacting",
    "will respond shortly",
    "will get back to you",
    "out of office",
    "automated message",
    "automated response",
    "this is an automated",
    "auto-reply",
    "autoreply",
    "i am currently unavailable",
    "we are currently unavailable",
    "we will reply as soon as possible",
    "we will get back to you",
    "please note this is an automated",

    # Hindi-English mix patterns (common in WA Business)
    "main ek automated assistant hoon",
    "aapki jaankari ke liye shukriya",
    "bahut-bahut shukriya",
    "hamari team se contact karenge",
    "jaldi se reply karenge",
    "team tak pahuncha deti hoon",
    "team tak pahuncha dunga",
    "hamari team aapse baat karegi",
    "aapka message receive ho gaya",
    "ham jaldi aapse sampark karenge",
    "aapki baat team tak pahunch jayegi",
]

HOSTILE_PATTERNS = [
    "stop messaging", "stop contacting", "not interested",
    "remove me", "unsubscribe", "do not contact", "dont contact",
    "spam", "annoying", "irritating", "harassment",
    "band karo", "mat bhejo", "mujhe nahi chahiye",
    "mujhe contact mat karo", "chhodo mujhe",
    "please stop", "leave me alone",
]

INTENT_COMMIT_PATTERNS = [
    "let's do it", "lets do it", "let us do it",
    "go ahead", "go for it", "proceed",
    "yes please", "yes do it", "yes, do it",
    "please do it", "please go ahead",
    "ok do it", "okay do it", "ok let's go",
    "start it", "send it", "draft it", "create it",
    "chaliye karte hain", "haan karo", "karo isko",
    "theek hai karo", "bilkul karo", "yes confirm",
    "confirm", "confirmed", "approved", "i approve",
    "sounds good, do it", "sounds great, proceed",
    "what's next", "whats next", "next step",
    "do it now", "go now",
]

OFF_TOPIC_PATTERNS = [
    "gst", "income tax", "legal", "court", "lawyer",
    "police", "complaint", "refund", "lawsuit",
    "stock market", "share price", "invest",
]


def classify_message(message: str, repeat_count: int) -> dict:
    """
    Classify an incoming merchant/customer message.

    Returns:
        {
          "type": "auto_reply" | "hostile" | "intent_commit" | "off_topic" | "engaged",
          "strategy": "one_more_try" | "wait" | "end" | "action_mode" | "redirect" | "normal",
          "confidence": "high" | "medium",
          "repeat_count": int
        }
    """
    msg_lower = message.strip().lower()

    # --- 1. Auto-reply detection ---
    is_pattern_auto = any(p in msg_lower for p in AUTO_REPLY_PATTERNS)
    is_auto = is_pattern_auto or repeat_count >= 2

    if is_auto:
        if repeat_count >= 3:
            strategy = "end"
        elif repeat_count == 2:
            strategy = "wait"
        else:
            strategy = "one_more_try"
        return {
            "type": "auto_reply",
            "strategy": strategy,
            "confidence": "high" if repeat_count >= 2 else "medium",
            "repeat_count": repeat_count,
        }

    # --- 2. Hostile / opt-out ---
    is_hostile = any(p in msg_lower for p in HOSTILE_PATTERNS)
    if is_hostile:
        return {
            "type": "hostile",
            "strategy": "end",
            "confidence": "high",
            "repeat_count": 0,
        }

    # --- 3. Intent commit ---
    is_commit = any(p in msg_lower for p in INTENT_COMMIT_PATTERNS)
    if is_commit:
        return {
            "type": "intent_commit",
            "strategy": "action_mode",
            "confidence": "high",
            "repeat_count": 0,
        }

    # --- 4. Off-topic ---
    is_off_topic = any(p in msg_lower for p in OFF_TOPIC_PATTERNS)
    if is_off_topic:
        return {
            "type": "off_topic",
            "strategy": "redirect",
            "confidence": "medium",
            "repeat_count": 0,
        }

    # --- 5. Normal engaged reply ---
    return {
        "type": "engaged",
        "strategy": "normal",
        "confidence": "high",
        "repeat_count": 0,
    }


def build_auto_reply_response(strategy: str, conv_id: str) -> dict:
    """
    Build a pre-canned response for auto-reply situations
    (not LLM-powered — deterministic and fast).
    """
    if strategy == "one_more_try":
        return {
            "action": "send",
            "body": "Lagta hai yeh ek auto-reply hai 😊 Jab owner isko dekhe, sirf 'YES' type karein aur main agla step setup kar lungi.",
            "cta": "binary_yes_no",
            "rationale": "Detected likely auto-reply (pattern match). One friendly prompt to redirect to owner.",
        }
    elif strategy == "wait":
        return {
            "action": "wait",
            "wait_seconds": 14400,  # 4 hours
            "rationale": "Auto-reply detected twice in a row. Backing off 4 hours to wait for the owner to be available.",
        }
    else:  # "end"
        return {
            "action": "end",
            "rationale": "Auto-reply detected 3+ times consecutively. No real engagement signal. Closing conversation gracefully.",
        }


def build_hostile_response() -> dict:
    return {
        "action": "send",
        "body": "Apologies, bilkul samajh gaye — I won't message again. Agar kabhi zaroorat ho toh 'Hi Vera' likh dena. Shukriya! 🙏",
        "cta": "none",
        "rationale": "Merchant signaled opt-out. Sending a polite goodbye and ending the conversation.",
    }
