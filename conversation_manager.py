"""
conversation_manager.py — Multi-turn conversation state tracker.

Tracks every conversation's history, detects repetition,
marks ended conversations, and maintains suppression keys.
"""

from threading import Lock
from datetime import datetime


class ConversationManager:
    def __init__(self):
        self._convs: dict[str, dict] = {}
        self._suppressed_keys: set[str] = set()  # global suppression (cross-conv)
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Conversation lifecycle
    # ------------------------------------------------------------------
    def get_or_create(self, conv_id: str, merchant_id: str, customer_id: str | None = None) -> dict:
        with self._lock:
            if conv_id not in self._convs:
                self._convs[conv_id] = {
                    "conv_id": conv_id,
                    "merchant_id": merchant_id,
                    "customer_id": customer_id,
                    "turns": [],           # list of {role, body, ts}
                    "sent_bodies": [],     # for anti-repetition
                    "ended": False,
                    "created_at": datetime.utcnow().isoformat(),
                }
            return self._convs[conv_id]

    def mark_ended(self, conv_id: str):
        with self._lock:
            if conv_id in self._convs:
                self._convs[conv_id]["ended"] = True

    def is_ended(self, conv_id: str) -> bool:
        return self._convs.get(conv_id, {}).get("ended", False)

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------
    def add_bot_turn(self, conv_id: str, body: str, cta: str = ""):
        with self._lock:
            conv = self._convs.get(conv_id)
            if conv:
                conv["turns"].append({
                    "role": "bot", "body": body, "cta": cta,
                    "ts": datetime.utcnow().isoformat()
                })
                conv["sent_bodies"].append(body.strip().lower())

    def add_merchant_turn(self, conv_id: str, body: str):
        with self._lock:
            conv = self._convs.get(conv_id)
            if conv:
                conv["turns"].append({
                    "role": "merchant", "body": body,
                    "ts": datetime.utcnow().isoformat()
                })

    def get_turns(self, conv_id: str) -> list[dict]:
        return self._convs.get(conv_id, {}).get("turns", [])

    def get_turn_count(self, conv_id: str) -> int:
        return len(self.get_turns(conv_id))

    def get_last_bot_body(self, conv_id: str) -> str | None:
        for turn in reversed(self.get_turns(conv_id)):
            if turn["role"] == "bot":
                return turn["body"]
        return None

    # ------------------------------------------------------------------
    # Anti-repetition check
    # ------------------------------------------------------------------
    def is_duplicate_body(self, conv_id: str, body: str) -> bool:
        """Check if this exact body was already sent in this conversation."""
        sent = self._convs.get(conv_id, {}).get("sent_bodies", [])
        return body.strip().lower() in sent

    # ------------------------------------------------------------------
    # Suppression keys (cross-conversation dedup)
    # ------------------------------------------------------------------
    def suppress(self, key: str):
        with self._lock:
            self._suppressed_keys.add(key)

    def is_suppressed(self, key: str) -> bool:
        return key in self._suppressed_keys

    def unsuppress(self, key: str):
        with self._lock:
            self._suppressed_keys.discard(key)

    # ------------------------------------------------------------------
    # Format for LLM prompt
    # ------------------------------------------------------------------
    def format_history_for_prompt(self, conv_id: str, max_turns: int = 6) -> str:
        turns = self.get_turns(conv_id)[-max_turns:]
        lines = []
        for t in turns:
            role = "Vera" if t["role"] == "bot" else "Merchant"
            lines.append(f"{role}: {t['body']}")
        return "\n".join(lines) if lines else "(no prior turns)"

    # ------------------------------------------------------------------
    # Count merchant auto-reply repetitions
    # ------------------------------------------------------------------
    def count_merchant_repeats(self, conv_id: str, message: str) -> int:
        """How many times has the merchant sent this exact message?"""
        msg_norm = message.strip().lower()
        merchant_turns = [
            t["body"].strip().lower()
            for t in self.get_turns(conv_id)
            if t["role"] == "merchant"
        ]
        return merchant_turns.count(msg_norm)
