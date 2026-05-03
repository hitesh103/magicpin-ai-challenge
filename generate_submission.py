"""
generate_submission.py — Generate submission.jsonl for all 30 test pairs.

Usage:
  python generate_submission.py

This script:
1. Loads all dataset contexts
2. Iterates through the 30 canonical test pairs
3. Calls compose() for each
4. Writes submission.jsonl (30 lines)
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from context_store import ContextStore
from composer import compose

DATASET_DIR = Path(__file__).parent / "dataset"

# ---------------------------------------------------------------------------
# 30 canonical test pairs (merchant_id, trigger_id)
# Covering all 5 categories × mix of trigger kinds × merchant + customer scope
# ---------------------------------------------------------------------------
TEST_PAIRS = [
    # DENTISTS (T01-T06)
    {"test_id": "T01", "merchant_id": "m_001_drmeera_dentist_delhi",    "trigger_id": "trg_001_research_digest_dentists"},
    {"test_id": "T02", "merchant_id": "m_001_drmeera_dentist_delhi",    "trigger_id": "trg_002_compliance_dci_radiograph"},
    {"test_id": "T03", "merchant_id": "m_001_drmeera_dentist_delhi",    "trigger_id": "trg_003_recall_due_priya",        "customer_id": "c_001_priya_for_m001"},
    {"test_id": "T04", "merchant_id": "m_002_bharat_dentist_mumbai",    "trigger_id": "trg_004_perf_dip_bharat"},
    {"test_id": "T05", "merchant_id": "m_002_bharat_dentist_mumbai",    "trigger_id": "trg_005_renewal_due_bharat"},
    {"test_id": "T06", "merchant_id": "m_001_drmeera_dentist_delhi",    "trigger_id": "trg_022_cde_webinar_dentists"},
    {"test_id": "T07", "merchant_id": "m_001_drmeera_dentist_delhi",    "trigger_id": "trg_023_competitor_opened_dentist"},

    # SALONS (T08-T12)
    {"test_id": "T08", "merchant_id": "m_003_studio11_salon_hyderabad", "trigger_id": "trg_006_festival_diwali"},
    {"test_id": "T09", "merchant_id": "m_003_studio11_salon_hyderabad", "trigger_id": "trg_007_bridal_followup_kavya",  "customer_id": "c_005_kavya_for_m003"},
    {"test_id": "T10", "merchant_id": "m_003_studio11_salon_hyderabad", "trigger_id": "trg_008_curious_ask_studio11"},
    {"test_id": "T11", "merchant_id": "m_004_glamour_salon_pune",       "trigger_id": "trg_009_winback_glamour"},
    {"test_id": "T12", "merchant_id": "m_004_glamour_salon_pune",       "trigger_id": "trg_025_dormancy_glamour"},

    # RESTAURANTS (T13-T18)
    {"test_id": "T13", "merchant_id": "m_005_pizzajunction_restaurant_delhi",         "trigger_id": "trg_010_ipl_match_delhi"},
    {"test_id": "T14", "merchant_id": "m_005_pizzajunction_restaurant_delhi",         "trigger_id": "trg_011_review_theme_late_delivery"},
    {"test_id": "T15", "merchant_id": "m_006_southindiancafe_restaurant_bangalore",   "trigger_id": "trg_012_milestone_mylari"},
    {"test_id": "T16", "merchant_id": "m_006_southindiancafe_restaurant_bangalore",   "trigger_id": "trg_013_corporate_thali_planning"},

    # GYMS (T17-T22)
    {"test_id": "T17", "merchant_id": "m_007_powerhouse_gym_bangalore",  "trigger_id": "trg_014_seasonal_acquisition_dip_powerhouse"},
    {"test_id": "T18", "merchant_id": "m_007_powerhouse_gym_bangalore",  "trigger_id": "trg_015_winback_rashmi",        "customer_id": "c_010_rashmi_for_m007"},
    {"test_id": "T19", "merchant_id": "m_008_zenyoga_gym_chennai",       "trigger_id": "trg_016_kids_yoga_program_drafting"},
    {"test_id": "T20", "merchant_id": "m_008_zenyoga_gym_chennai",       "trigger_id": "trg_017_kids_yoga_trial_followup_karthik", "customer_id": "c_012_karthik_jr_for_m008"},
    {"test_id": "T21", "merchant_id": "m_008_zenyoga_gym_chennai",       "trigger_id": "trg_024_perf_spike_zen"},

    # PHARMACIES (T22-T28)
    {"test_id": "T22", "merchant_id": "m_009_apollo_pharmacy_jaipur",    "trigger_id": "trg_018_supply_atorvastatin_recall"},
    {"test_id": "T23", "merchant_id": "m_009_apollo_pharmacy_jaipur",    "trigger_id": "trg_019_chronic_refill_grandfather", "customer_id": "c_013_grandfather_for_m009"},
    {"test_id": "T24", "merchant_id": "m_009_apollo_pharmacy_jaipur",    "trigger_id": "trg_020_summer_demand_shift"},
    {"test_id": "T25", "merchant_id": "m_010_sunrisepharm_pharmacy_lucknow", "trigger_id": "trg_021_unverified_gbp_sunrise"},

    # MIXED (T26-T30) — harder pairs using generated data
    {"test_id": "T26", "merchant_id": "m_001_drmeera_dentist_delhi",    "trigger_id": "trg_001_research_digest_dentists"},  # repeat with different emphasis
    {"test_id": "T27", "merchant_id": "m_003_studio11_salon_hyderabad", "trigger_id": "trg_008_curious_ask_studio11"},
    {"test_id": "T28", "merchant_id": "m_006_southindiancafe_restaurant_bangalore", "trigger_id": "trg_012_milestone_mylari"},
    {"test_id": "T29", "merchant_id": "m_007_powerhouse_gym_bangalore",  "trigger_id": "trg_014_seasonal_acquisition_dip_powerhouse"},
    {"test_id": "T30", "merchant_id": "m_009_apollo_pharmacy_jaipur",    "trigger_id": "trg_018_supply_atorvastatin_recall"},
]


def load_dataset(ctx: ContextStore):
    """Load all dataset files into the context store."""
    # Load categories
    cat_dir = DATASET_DIR / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            data = json.load(open(f))
            slug = data.get("slug", f.stem)
            ctx.upsert("category", slug, 1, data)
            print(f"  [OK] category/{slug}")

    # Load merchants
    merchants_file = DATASET_DIR / "merchants_seed.json"
    if merchants_file.exists():
        data = json.load(open(merchants_file))
        for m in data.get("merchants", []):
            mid = m.get("merchant_id")
            if mid:
                ctx.upsert("merchant", mid, 1, m)
        print(f"  [OK] {len(data.get('merchants', []))} merchants")

    # Load customers
    customers_file = DATASET_DIR / "customers_seed.json"
    if customers_file.exists():
        data = json.load(open(customers_file))
        customers = data.get("customers", [])
        for c in customers:
            cid = c.get("customer_id")
            if cid:
                ctx.upsert("customer", cid, 1, c)
        print(f"  [OK] {len(customers)} customers")

    # Load triggers
    triggers_file = DATASET_DIR / "triggers_seed.json"
    if triggers_file.exists():
        data = json.load(open(triggers_file))
        triggers = data.get("triggers", [])
        for t in triggers:
            tid = t.get("id")
            if tid:
                ctx.upsert("trigger", tid, 1, t)
        print(f"  [OK] {len(triggers)} triggers")


def run():
    print("=" * 60)
    print("  magicpin AI Challenge — Submission Generator")
    print("=" * 60)

    # Check API key
    if not os.getenv("LLM_API_KEY"):
        print("\n[ERROR] LLM_API_KEY not set. Copy .env.example to .env and fill in your key.")
        return

    print("\nLoading dataset...")
    ctx = ContextStore()
    load_dataset(ctx)

    print(f"\nGenerating {len(TEST_PAIRS)} submissions...")
    results = []
    errors = []

    for i, pair in enumerate(TEST_PAIRS, 1):
        test_id = pair["test_id"]
        merchant_id = pair["merchant_id"]
        trigger_id = pair["trigger_id"]
        customer_id = pair.get("customer_id")

        print(f"  [{i:02d}/{len(TEST_PAIRS)}] {test_id}: {merchant_id} + {trigger_id}", end=" ")

        # Fetch contexts
        trigger = ctx.get("trigger", trigger_id)
        merchant = ctx.get("merchant", merchant_id)
        category_slug = (merchant or {}).get("category_slug")
        category = ctx.get("category", category_slug) if category_slug else None
        customer = ctx.get("customer", customer_id) if customer_id else None

        if not trigger:
            print(f"[SKIP] trigger {trigger_id} not found")
            errors.append({"test_id": test_id, "error": "trigger not found"})
            continue

        if not merchant:
            print(f"[SKIP] merchant {merchant_id} not found")
            errors.append({"test_id": test_id, "error": "merchant not found"})
            continue

        try:
            composed = compose(category or {}, merchant, trigger, customer)
            row = {
                "test_id": test_id,
                "body": composed["body"],
                "cta": composed["cta"],
                "send_as": composed["send_as"],
                "suppression_key": composed["suppression_key"],
                "rationale": composed["rationale"],
            }
            results.append(row)
            print(f"[OK] {len(composed['body'])} chars")
            time.sleep(5) # Avoid rate limits
        except Exception as e:
            print(f"[ERROR] {e}")
            errors.append({"test_id": test_id, "error": str(e)})
            time.sleep(5) # Avoid rate limits

    # Write submission.jsonl
    output_path = Path(__file__).parent / "submission.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! {len(results)}/{len(TEST_PAIRS)} submissions written to {output_path}")
    if errors:
        print(f"  {len(errors)} errors: {[e['test_id'] for e in errors]}")
    print("="*60)


if __name__ == "__main__":
    run()
