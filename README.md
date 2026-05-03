# Vera — magicpin AI Challenge Submission

## Approach

**Core architecture**: A 4-context LLM composition engine with trigger-kind dispatch, deterministic pre-classification, and multi-turn conversation management.

### Key Design Decisions

#### 1. Trigger-Kind Dispatch
Instead of a single monolithic prompt, we map every trigger `kind` to a named strategy (e.g., `research_citation`, `slot_offering_recall`, `action_continuation`). Each strategy has a specific hint injected into the prompt, guiding the LLM to use the right compulsion lever and CTA type for that trigger family.

This means:
- A `research_digest` trigger always gets source-citation framing + open-ended CTA
- An `active_planning_intent` trigger always immediately drafts an artifact + binary CTA (never asks qualifying questions)
- A `supply_alert` trigger always leads with batch numbers + derived count from merchant data

#### 2. Pre-LLM Classification (No LLM Cost for Edge Cases)
Before calling the LLM for replies, we classify the incoming merchant message:
- **Auto-reply detected**: Deterministic response (no LLM call) — saves latency + cost
- **Hostile**: Graceful goodbye (no LLM call)
- **Intent commit**: LLM called with "switch to action mode" instruction
- **Engaged**: LLM called with "advance 1 step" instruction

Detection escalation:
- Turn 1: Pattern match → one friendly prompt to redirect to owner
- Turn 2: Same message again → wait 4 hours
- Turn 3+: Still auto-reply → end gracefully

#### 3. Context Extraction Before LLM
We extract the most relevant facts *before* the LLM call:
- `get_relevant_digest()` — finds the exact digest item referenced by the trigger
- `get_peer_comparison()` — computes merchant CTR vs peer CTR delta
- `get_seasonal_beat()` — looks up the current month's seasonal note
- `get_trigger_why_now()` — translates trigger payload into human-readable "why now"

This reduces hallucination risk: the LLM gets pre-processed, specific facts, not raw JSON blobs.

#### 4. Tick Prioritization
Triggers are scored: `priority = urgency × 2 + kind_weight`. This ensures `supply_alert` (urgency 5, kind_weight 10) always fires before `curious_ask_due` (urgency 1, kind_weight 2).

#### 5. Anti-Repetition + Suppression
- Every sent message body is stored per conversation
- Duplicate bodies are skipped before sending
- Global suppression keys prevent re-sending cross-tick

### What Additional Context Would Have Helped

1. **Real merchant slot schedules**: For `recall_due` and `appointment` triggers, we had to use placeholder slot times. Real available slots from a scheduling API would make customer-facing messages much more actionable.

2. **Actual competitor data**: For `competitor_opened` triggers, the competitor's offer is in the trigger payload, but knowing the competitor's rating/review count/photos would enable better comparison framing.

3. **Historical engagement rates by trigger kind**: Knowing which trigger families actually drive merchant replies would let us tune the `kind_weight` scoring more precisely.

4. **Merchant's preferred send time**: WhatsApp messages sent at 7am vs 11am have very different read rates. Per-merchant preferred-contact-time data would improve send timing.

### Tradeoffs

- **LLM provider**: Using a single provider (configurable). In production, would A/B test Claude vs GPT-4o for different trigger families.
- **Context size**: We pass pre-extracted facts rather than raw JSON to stay within token limits and improve specificity. Tradeoff: if a rare field isn't extracted, it won't be used.
- **Temperature=0**: Fully deterministic. Tradeoff: less creative variation, but consistent and auditable.

## Running the Bot

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your LLM API key

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn bot:app --host 0.0.0.0 --port 8080

# 4. Run local tests
python judge_simulator.py

# 5. Generate submission.jsonl
python generate_submission.py
```

## File Structure

```
.
├── bot.py                    # FastAPI server (5 endpoints)
├── composer.py               # LLM composition engine
├── context_store.py          # Versioned in-memory context store
├── conversation_manager.py   # Multi-turn state tracking
├── auto_reply_detector.py    # Auto-reply / intent / hostile detection
├── reply_handler.py          # Multi-turn reply composition
├── generate_submission.py    # Generate submission.jsonl
├── submission.jsonl          # 30 test-pair outputs
├── requirements.txt
├── .env.example
└── README.md
```
