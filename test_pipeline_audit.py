"""
Full pipeline audit: Are entities + community summaries stored and searched?
"""
import mentedb, tempfile, os, json

API_KEY = os.environ.get("MENTEDB_OPENAI_API_KEY") or os.environ.get("MENTEDB_ANTHROPIC_API_KEY")
assert API_KEY, "Set MENTEDB_OPENAI_API_KEY or MENTEDB_ANTHROPIC_API_KEY"

# Detect provider and set the env vars that build_extraction_config_from_env reads
if os.environ.get("MENTEDB_ANTHROPIC_API_KEY"):
    LLM_PROVIDER = "anthropic"
    os.environ["MENTEDB_LLM_API_KEY"] = os.environ["MENTEDB_ANTHROPIC_API_KEY"]
    os.environ["MENTEDB_LLM_PROVIDER"] = "anthropic"
elif os.environ.get("MENTEDB_OPENAI_API_KEY"):
    LLM_PROVIDER = "openai"
    os.environ["MENTEDB_LLM_API_KEY"] = os.environ["MENTEDB_OPENAI_API_KEY"]
    os.environ["MENTEDB_LLM_PROVIDER"] = "openai"

# Use OpenAI key for embeddings (even if LLM is Anthropic)
EMBED_KEY = os.environ.get("MENTEDB_OPENAI_API_KEY", API_KEY)

db = mentedb.MenteDB(data_dir=tempfile.mkdtemp(), embedding_provider="openai", embedding_api_key=EMBED_KEY)

print("=" * 70)
print("STEP 1: Extract from a hearing aids battery conversation")
print("=" * 70)

conversation = """User: I need help ordering some replacement batteries for my hearing aids. I've been using the same set for months now.
Assistant: I'd be happy to help! What type of hearing aids do you have?
User: They're Phonak BTE hearing aids, size 13 batteries. I use them every day for about 12-16 hours.
Assistant: For Phonak BTE hearing aids using size 13 batteries, I recommend ordering from Amazon. With your daily usage of 12-16 hours, each battery should last 7-10 days.
User: Great, I'll order from Amazon since I'm already a Prime member. By the way, I also use these when doing my guided breathing sessions with my Fitbit Versa 3."""

extracted = db.extract(conversation, provider=LLM_PROVIDER)
print(f"\nExtracted {len(extracted)} items:")
entities_found = 0
for i, item in enumerate(extracted):
    is_entity = item.get("entity_name") is not None
    if is_entity: entities_found += 1
    etype = f" [ENTITY: {item.get('entity_name')} / {item.get('entity_type')}]" if is_entity else ""
    attrs = item.get("entity_attributes", {})
    cat = attrs.get("category", "NO CATEGORY") if attrs else "NO ATTRS"
    print(f"  {i+1}. {'🔷 ENTITY' if is_entity else '📝 FACT'} | {item['content'][:80]}")
    if is_entity:
        print(f"       category: {cat}")
        print(f"       embed_key: {item.get('embedding_key', 'N/A')[:100]}")
print(f"\nEntities extracted: {entities_found}")

print("\n" + "=" * 70)
print("STEP 2: Store extracted memories")
print("=" * 70)

result = db.store_extracted(extracted)
stored_ids = result.get("stored_ids", [])
print(f"Stored {len(stored_ids)} memories")

# Check what's actually in the DB
print("\nAll stored memories:")
for mid in stored_ids:
    mem = db.get_memory(mid)
    tags = mem.get('tags', [])
    is_entity = any(t.startswith('entity_name:') for t in tags)
    marker = "🔷" if is_entity else "📝"
    print(f"  {marker} [{mem['memory_type']}] {mem['content'][:80]}")
    if is_entity:
        print(f"     tags: {tags}")

print("\n" + "=" * 70)
print("STEP 3: Build community summaries")
print("=" * 70)

try:
    community_ids = db.build_communities()
    print(f"Built {len(community_ids)} community summaries:")
    for cid in community_ids:
        mem = db.get_memory(cid)
        print(f"  🏘️ {mem['content'][:120]}")
        print(f"     tags: {mem.get('tags', [])}")
except Exception as e:
    print(f"❌ build_communities failed: {e}")

print("\n" + "=" * 70)
print("STEP 4: Search for 'health devices' — what do we find?")
print("=" * 70)

hits = db.search_text("health-related devices user uses daily", k=10)
print(f"\nTop {len(hits)} results for 'health-related devices user uses daily':")
for rank, h in enumerate(hits, 1):
    mem = db.get_memory(h.id)
    tags = mem.get('tags', [])
    is_entity = any(t.startswith('entity_name:') for t in tags)
    is_community = 'community_summary' in tags
    marker = "🏘️" if is_community else ("🔷" if is_entity else "📝")
    hearing = " 🎧" if "hearing" in mem['content'].lower() else ""
    print(f"  #{rank} {marker} [{h.score:.4f}] {mem['content'][:70]}{hearing}")

# Also try the expanded search (what the benchmark uses)
print(f"\nExpanded search (what benchmark uses):")
try:
    hits2 = db.search_expanded("health-related devices user uses daily", k=10, provider=LLM_PROVIDER)
    for rank, h in enumerate(hits2, 1):
        mem = db.get_memory(h.id)
        tags = mem.get('tags', [])
        is_entity = any(t.startswith('entity_name:') for t in tags)
        is_community = 'community_summary' in tags
        marker = "🏘️" if is_community else ("🔷" if is_entity else "📝")
        hearing = " 🎧" if "hearing" in mem['content'].lower() else ""
        print(f"  #{rank} {marker} [{h.score:.4f}] {mem['content'][:70]}{hearing}")
except Exception as e:
    print(f"  search_expanded failed: {e}")

print("\nDone!")
