"""Test contextual enrichment with real OpenAI embeddings.
Scores are RRF (Reciprocal Rank Fusion) — always small. Focus on RANKING ORDER.
"""
import mentedb, tempfile, os

API_KEY = os.environ["MENTEDB_OPENAI_API_KEY"]

def make_db():
    return mentedb.MenteDB(data_dir=tempfile.mkdtemp(), embedding_provider="openai", embedding_api_key=API_KEY)

def test(label, db, queries):
    print(f"\n--- {label} ---")
    for q in queries:
        hits = db.search_text(q, k=5)
        print(f"\n  '{q}'")
        for rank, h in enumerate(hits, 1):
            mem = db.get_memory(h.id)
            c = mem['content'][:65]
            markers = []
            if "hearing" in c.lower(): markers.append("🎧")
            if "knife" in c.lower() and "health" in q.lower(): markers.append("❌")
            m = f" {' '.join(markers)}" if markers else ""
            print(f"    #{rank} [{h.score:.4f}] {c}{m}")

print("Scores are RRF — focus on RANK ORDER, not magnitude.\n")

# ===== TEST 1: KNIFE =====
print("=" * 70)
print("TEST 1: KNIFE DISAMBIGUATION")
print("  Expected: 'cooking' → grandmother #1, 'survival' → camping #1")
print("=" * 70)

q1 = ["cooking equipment", "outdoor survival gear", "kitchen tools from family"]

db1 = make_db()
db1.store_extracted([
    {"content": "User has a knife from grandmother (kitchen use)", "memory_type": "fact", "tags": ["knife"], "confidence": 0.9,
     "embedding_key": "User has a knife from grandmother"},
    {"content": "User bought a survival knife for camping trips", "memory_type": "fact", "tags": ["knife"], "confidence": 0.9,
     "embedding_key": "User bought a survival knife for camping"},
])
test("FLAT", db1, q1)

db2 = make_db()
db2.store_extracted([
    {"content": "User has a knife from grandmother (kitchen use)", "memory_type": "fact", "tags": ["knife", "cooking", "family_heirloom"], "confidence": 0.9,
     "embedding_key": "User has a knife from grandmother [context: cooking_tool, kitchen_equipment, family_heirloom]"},
    {"content": "User bought a survival knife for camping trips", "memory_type": "fact", "tags": ["knife", "survival", "outdoor"], "confidence": 0.9,
     "embedding_key": "User bought a survival knife for camping [context: outdoor_gear, camping_equipment, safety_tool]"},
])
test("CONTEXTUAL", db2, q1)

# ===== TEST 2: HEALTH DEVICES =====
print("\n" + "=" * 70)
print("TEST 2: HEALTH DEVICES")
print("  Expected: hearing aids should rank HIGHER with context")
print("=" * 70)

q2 = ["health-related devices user uses daily", "health devices", "medical equipment"]

db3 = make_db()
db3.store_extracted([
    {"content": "User tracks steps with Fitbit Versa 3", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User tracks steps with Fitbit Versa 3 smartwatch"},
    {"content": "User tests blood sugar with Accu-Chek Aviva Nano", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User tests blood sugar with Accu-Chek Aviva Nano"},
    {"content": "User uses nebulizer for inhalation treatments", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User uses nebulizer machine for inhalation treatments"},
    {"content": "User ordered replacement batteries for Phonak hearing aids", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User ordered replacement batteries for Phonak BTE hearing aids"},
])
test("FLAT", db3, q2)

db4 = make_db()
db4.store_extracted([
    {"content": "User tracks steps with Fitbit Versa 3", "memory_type": "fact", "tags": ["health_device"], "confidence": 0.9,
     "embedding_key": "User tracks steps with Fitbit Versa 3 smartwatch [context: health_device, fitness_tracker, daily_use_device]"},
    {"content": "User tests blood sugar with Accu-Chek Aviva Nano", "memory_type": "fact", "tags": ["health_device"], "confidence": 0.9,
     "embedding_key": "User tests blood sugar with Accu-Chek Aviva Nano [context: health_device, medical_device, daily_use_device]"},
    {"content": "User uses nebulizer for inhalation treatments", "memory_type": "fact", "tags": ["health_device"], "confidence": 0.9,
     "embedding_key": "User uses nebulizer machine for inhalation treatments [context: health_device, medical_device, daily_use_device]"},
    {"content": "User ordered replacement batteries for Phonak hearing aids", "memory_type": "fact", "tags": ["health_device"], "confidence": 0.9,
     "embedding_key": "User ordered replacement batteries for Phonak BTE hearing aids [context: health_device, medical_device, daily_use_device]"},
])
test("CONTEXTUAL", db4, q2)

# ===== TEST 3: FAILURE MODES =====
print("\n" + "=" * 70)
print("TEST 3: FAILURE MODES")
print("=" * 70)

print("\n3A: WRONG CONTEXT (knife as health_device)")
db5 = make_db()
db5.store_extracted([
    {"content": "User has a kitchen knife", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User has a kitchen knife [context: health_device, medical_equipment]"},
    {"content": "User has a Fitbit Versa 3", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User has a Fitbit Versa 3 [context: health_device, fitness_tracker]"},
])
test("WRONG CONTEXT", db5, ["health devices", "cooking equipment"])

print("\n3B: OVER-TAGGING (14 vs 2)")
db6 = make_db()
db6.store_extracted([
    {"content": "User has a Fitbit Versa 3", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "Fitbit Versa 3 [context: health_device, fitness_tracker, wearable, smartwatch, daily_use_device, exercise_tool, step_counter, heart_rate_monitor, sleep_tracker, activity_tracker, wellness_device, personal_electronics, gift_from_spouse, birthday_present]"},
    {"content": "User tests blood sugar with Accu-Chek", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User tests blood sugar with Accu-Chek [context: health_device, medical_device]"},
])
test("OVER-TAGGED", db6, ["health devices", "fitness tracker", "birthday present"])

print("\n3C: SPLIT CONTEXT")
db7 = make_db()
db7.store_extracted([
    {"content": "User uses hearing aids daily for health", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User uses hearing aids daily [context: health_device, daily_use_device]"},
    {"content": "User ordered hearing aid batteries on Amazon", "memory_type": "fact", "tags": [], "confidence": 0.9,
     "embedding_key": "User ordered hearing aid batteries on Amazon [context: shopping, amazon_purchase, electronics]"},
])
test("SPLIT CONTEXT", db7, ["health devices", "amazon purchases", "hearing aids"])

print("\nDone! Compare rank positions between FLAT and CONTEXTUAL.")
