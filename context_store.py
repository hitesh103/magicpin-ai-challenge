"""
context_store.py — Versioned in-memory context store.

Stores all 4 context types (category, merchant, customer, trigger)
received from the judge via POST /v1/context.

Rules:
- Keyed by (scope, context_id)
- Idempotent by (context_id, version): re-posting same version is no-op
- Higher version replaces lower version atomically
"""

from threading import Lock


class ContextStore:
    def __init__(self):
        self._store: dict[tuple[str, str], dict] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def upsert(self, scope: str, context_id: str, version: int, payload: dict) -> tuple[bool, int]:
        """
        Try to store context.
        Returns (accepted, current_version).
        accepted=False means the caller already has a >= version.
        """
        key = (scope, context_id)
        with self._lock:
            current = self._store.get(key)
            if current and current["version"] >= version:
                return False, current["version"]
            self._store[key] = {"version": version, "payload": payload}
            return True, version

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get(self, scope: str, context_id: str) -> dict | None:
        """Return the payload for a given (scope, context_id), or None."""
        entry = self._store.get((scope, context_id))
        return entry["payload"] if entry else None

    def get_version(self, scope: str, context_id: str) -> int:
        """Return the stored version, or 0 if not found."""
        entry = self._store.get((scope, context_id))
        return entry["version"] if entry else 0

    # ------------------------------------------------------------------
    # Bulk reads
    # ------------------------------------------------------------------
    def all_by_scope(self, scope: str) -> list[dict]:
        """Return all payloads for a given scope."""
        with self._lock:
            return [v["payload"] for (s, _), v in self._store.items() if s == scope]

    def all_triggers(self) -> list[dict]:
        return self.all_by_scope("trigger")

    def all_merchants(self) -> list[dict]:
        return self.all_by_scope("merchant")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def count_by_scope(self) -> dict:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _) in self._store:
                if scope in counts:
                    counts[scope] += 1
        return counts

    # ------------------------------------------------------------------
    # Helpers for assembling the 4-context bundle
    # ------------------------------------------------------------------
    def get_merchant_bundle(self, merchant_id: str) -> dict:
        """Return merchant + its linked category in one dict."""
        merchant = self.get("merchant", merchant_id)
        if not merchant:
            return {}
        category_slug = merchant.get("category_slug")
        category = self.get("category", category_slug) if category_slug else None
        return {"merchant": merchant, "category": category}

    def get_trigger_bundle(self, trigger_id: str) -> dict | None:
        """Return trigger + merchant + category + optional customer."""
        trigger = self.get("trigger", trigger_id)
        if not trigger:
            return None

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")

        merchant = self.get("merchant", merchant_id) if merchant_id else None
        category_slug = merchant.get("category_slug") if merchant else None
        category = self.get("category", category_slug) if category_slug else None
        customer = self.get("customer", customer_id) if customer_id else None

        return {
            "trigger": trigger,
            "merchant": merchant,
            "category": category,
            "customer": customer,
        }

    def teardown(self):
        """Wipe all stored context (called on POST /v1/teardown)."""
        with self._lock:
            self._store.clear()
