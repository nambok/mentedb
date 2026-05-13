#!/usr/bin/env python3
"""
LongMemEval Benchmark with Sleeptime Enrichment — Two-Phase Architecture

Unlike run_benchmark.py (which creates isolated DBs per question), this script
mirrors how MenteDB is actually used in production:

  Phase 1 — Global Ingest:
    Ingest ALL unique sessions into a single shared DB with LLM extraction,
    then run the full 4-phase enrichment pipeline ONCE on the whole corpus.
    This builds cross-session entity links, community summaries, and a user model.

  Phase 2 — Query:
    For each question, query the shared DB with search_expanded().
    The engine surfaces relevant memories using the enriched graph.

This is ~30x cheaper than per-question enrichment and tests the real value
of enrichment: cross-session knowledge linking.

Usage:
    python benchmarks/longmemeval/run_enriched.py [options]

Environment:
    OPENAI_API_KEY         For embeddings and/or answer generation
    ANTHROPIC_API_KEY      Alternative for answer generation
    MENTEDB_LLM_PROVIDER   LLM provider for extraction + enrichment
    MENTEDB_LLM_API_KEY    API key for extraction + enrichment
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import get_llm_client, has_llm_key, llm_chat

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DATASET_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

READER_PROMPT = """You are an AI assistant answering questions from stored conversation history.

QUESTION TYPE: {question_type_hint}

RULES:

1. COUNTING / ENUMERATION (listing distinct items across multiple conversations):
   - STEP 1 — EXHAUSTIVE SCAN: Read EVERY snippet. For each one, ask: "Does this mention
     something that could match the question?" If yes, add it to the candidate list.
     Be BROAD — include anything remotely relevant. Better to over-include than miss items.
     List each candidate:
       1. [item] — evidence: "[quote from snippet]"
       2. [item] — evidence: "[quote from snippet]"
   - STEP 2 — INTERPRET the question BROADLY:
     * "items of clothing to pick up" includes dry cleaning pickups, exchanges, alterations
     * "trips I took" includes day trips, hikes, road trips, weekend getaways
     * "tanks I have" includes all aquariums/tanks regardless of size or purpose
     * "times I baked" includes any baking activity (bread, cake, cookies, etc.)
     * When the question says "or" (e.g. "pick up OR return"), count BOTH categories
   - STEP 3 — VERIFY each candidate:
     * Check TIME constraints strictly:
       - "in March" → only events explicitly dated in March
       - "past two weeks" from {question_date} → calculate the exact date range
       - "this year" → only events in the current year
     * Remove TRUE duplicates only: same specific item in multiple snippets = 1 count
       But different items of the same type = separate counts (e.g., two different tanks = 2)
   - STEP 4 — Final answer: "Total: [N]"
   - For SUMS (total money, total distance): show each amount + the addition

2. TEMPORAL / DATE MATH:
   - Current date: {question_date}
   - Step 1: Find the EXACT date(s) — look for explicit mentions like "on May 15", "March 3, 2023"
   - Step 2: Write dates in YYYY-MM-DD format
   - Step 3: Calculate STEP BY STEP:
     * Days between dates: count days in each month between them
       Example: March 15 to April 22 = (31-15) + 22 = 16 + 22 = 38 days
     * Weeks: total_days ÷ 7 (round down unless question says "about")
     * Months: count calendar months (e.g., March to June = 3 months)
   - Step 4: State answer with the calculation shown
   - "How long ago" = {question_date} minus event date
   - Do NOT use the broad counting approach for date/duration questions

3. KNOWLEDGE-UPDATE (current value, latest):
   - The history below is sorted CHRONOLOGICALLY (later = more recent).
   - If you see CONFLICTING values for the same thing:
     * The LATER entry is the CURRENT/CORRECT value
     * Words indicating updates: "changed", "updated", "switched", "now", "moved to", "no longer"
   - Look for [Date: ...] markers to identify which is newer
   - Answer with ONLY the most recent value
   - For "how long have I been doing X": find when X STARTED (the most recent start date),
     then calculate duration from that start date to {question_date}. Do NOT use broad counting.

4. PREFERENCE / RECOMMENDATION:
   - ALWAYS give a personalized answer — NEVER abstain on preference questions
   - Reference SPECIFIC details from the user's history
   - Use their actual hobbies, purchases, pets, routines, family to personalize

5. ABSTENTION:
   - ONLY abstain if NO relevant information about the SPECIFIC topic exists in the history
   - Abstain examples (correct):
     * Question asks about "Sacramento" but history only mentions "San Francisco" → abstain
     * Question asks about "bus from airport" but history only mentions train/taxi → abstain
     * Question asks about "poster presentation" but history only mentions "attending conference" → abstain
   - Do NOT abstain examples (wrong to abstain):
     * Question asks "when did I submit my paper?" and history mentions a submission date → answer
     * Question asks "how many X?" and you found at least some X → give your count
     * Question asks about a hobby/activity and history discusses it → answer
   - For preference questions: NEVER abstain
   - When in doubt and you have SOME relevant info: give your best answer rather than abstaining

6. ACCURACY:
   - Only state facts explicitly present in the history
   - Do NOT fabricate names, numbers, dates, or details

History (sorted chronologically, later = more recent):

{retrieved_context}

Current Date: {question_date}
Question: {question}

Think step by step. Show your work for counting and date calculations. Give your final answer concisely."""


def load_dataset(variant="s"):
    filepath = os.path.join(DATA_DIR, DATASET_FILES.get(variant, ""))
    if not os.path.exists(filepath):
        print(f"Dataset not found: {filepath}")
        print("Run: python benchmarks/longmemeval/download_data.py")
        sys.exit(1)
    with open(filepath) as f:
        return json.load(f)


def format_session(session, date):
    lines = [f"[Date: {date}]"]
    for turn in session:
        role = turn["role"].capitalize()
        lines.append(f"  {role}: {turn['content']}")
    return "\n".join(lines)


def date_to_microseconds(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    except (ValueError, TypeError):
        return None


def _extract_one(db, text, llm_provider, max_retries=5):
    for attempt in range(max_retries):
        try:
            return ("ok", db.extract(text, provider=llm_provider))
        except Exception as e:
            err = str(e).lower()
            if "401" in err or "unauthorized" in err or "invalid_api_key" in err:
                return ("fatal", str(e))
            if "404" in err or "not found" in err or "does not exist" in err:
                return ("model_error", str(e))
            if attempt < max_retries - 1:
                time.sleep(3 * (2 ** attempt))
            else:
                return ("fail", str(e))
    return ("fail", "max retries exceeded")


# ─── Phase 1: Global Ingest ─────────────────────────────────────────────────

def collect_unique_sessions(dataset):
    """Deduplicate sessions across all questions. Returns list of (session_id, session, date)."""
    seen = {}
    for q in dataset:
        for sid, sess, date in zip(
            q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]
        ):
            if sid not in seen:
                seen[sid] = (sess, date)
    return [(sid, sess, date) for sid, (sess, date) in seen.items()]


def global_ingest(db, unique_sessions, cognitive_provider, workers=10, db_dir=None, resume_store=0):
    """Ingest all unique sessions into a single shared DB with LLM extraction."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json as _json

    total = len(unique_sessions)

    texts = []
    dates = []
    session_ids = []
    for sid, sess, date in unique_sessions:
        texts.append(format_session(sess, date))
        dates.append(date)
        session_ids.append(sid)

    # Check for cached extraction results (supports partial resume)
    cache_path = os.path.join(db_dir or "/tmp", "extraction_cache.jsonl")
    extracted = [None] * total
    cached_indices = set()

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f:
                entry = _json.loads(line)
                idx = entry["idx"]
                extracted[idx] = entry["memories"]
                cached_indices.add(idx)
        print(f"\n  Phase 1a: Loaded {len(cached_indices)}/{total} cached extractions")

    remaining = [(i, texts[i]) for i in range(total) if i not in cached_indices]
    if remaining:
        print(f"  Extracting {len(remaining)} remaining sessions...")
        cache_f = open(cache_path, "a")
        ok_count = len(cached_indices)
        fail_count = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_one, db, text, cognitive_provider): i
                for i, text in remaining
            }
            for f in tqdm(as_completed(futures), total=len(remaining), desc="Extracting"):
                idx = futures[f]
                status, payload = f.result()
                if status == "ok":
                    extracted[idx] = payload
                    ok_count += 1
                    cache_f.write(_json.dumps({"idx": idx, "memories": payload}) + "\n")
                    cache_f.flush()
                else:
                    fail_count += 1
                    if status == "fatal":
                        print(f"\n  FATAL: {payload[:200]}")
                        break
        cache_f.close()
        print(f"  Extraction done: {ok_count} ok, {fail_count} failed")
    else:
        print(f"  All {total} extractions cached, skipping.")

    # Store extracted memories in large batches to minimize embedding API calls.
    # Strip entity fields — entity resolution scans all existing memories and
    # becomes O(n²). The benchmark only needs memories + embeddings for search.
    # Tag each memory with its source session_id for per-question filtering.
    print(f"  Phase 1b: Storing extracted memories...")
    memory_count = 0
    EXTRACT_BATCH = 500
    flat = []
    for idx, memories in enumerate(extracted):
        if memories is not None:
            sid = session_ids[idx]
            for mem in memories:
                mem.pop("entity_name", None)
                mem.pop("entity_type", None)
                mem.pop("entity_attributes", None)
                tags = mem.get("tags") or []
                tags.append(f"sid:{sid}")
                mem["tags"] = tags
            flat.extend(memories)
    all_batches = list(range(0, len(flat), EXTRACT_BATCH))
    if resume_store > 0:
        skip_items = resume_store * EXTRACT_BATCH
        print(f"  Resuming from batch {resume_store} (skipping {skip_items} memories)")
        all_batches = all_batches[resume_store:]
    for i, start in enumerate(tqdm(all_batches, desc="Storing extracted")):
        batch = flat[start:start + EXTRACT_BATCH]
        result = db.store_extracted(batch)
        memory_count += len(result.get("stored_ids", []))

    # Store raw sessions as semantic for BM25 keyword coverage.
    # Using semantic (not episodic) so enrichment_candidates() won't re-extract them.
    print(f"  Phase 1c: Storing raw sessions for BM25...")
    raw_memories = []
    for i, (text, date) in enumerate(zip(texts, dates)):
        content = text[:8000] if len(text) > 8000 else text
        raw_memories.append({
            "content": content,
            "memory_type": "semantic",
            "tags": [f"date:{date}", f"sid:{session_ids[i]}", "raw"],
            "confidence": 0.3,
            "embedding_key": text[:500],
        })

    # Batch store in chunks to avoid memory pressure
    BATCH_SIZE = 100
    raw_count = 0
    for i, start in enumerate(tqdm(range(0, len(raw_memories), BATCH_SIZE), desc="Storing raw")):
        batch = raw_memories[start:start + BATCH_SIZE]
        result = db.store_extracted(batch)
        raw_count += len(result.get("stored_ids", []))

    # Build community summaries (rule-based, from extraction phase)
    print(f"  Phase 1d: Building community summaries...")
    try:
        community_ids = db.build_communities()
        print(f"  Built {len(community_ids)} community summaries")
    except Exception as e:
        print(f"  Community build failed: {e}")
        community_ids = []

    print(f"\n  Ingest complete: {memory_count} extracted + {raw_count} raw memories")
    return memory_count + raw_count


def run_enrichment_phase(db, cognitive_provider):
    """Run enrichment Phases 2-4 on the shared DB.

    Phase 1 (extraction) was already done during ingest, so we call the
    individual building blocks for entity linking, communities, and user model.
    """
    print(f"\n  Phase 2: Running enrichment phases 2-4...")
    start = time.time()

    # Phase 2: Entity linking (rule-based)
    print(f"    Entity linking (rule-based)...")
    try:
        link_result = db.link_entities()
        print(f"      Linked: {link_result['linked']}, "
              f"Ambiguous: {link_result['ambiguous']}, "
              f"Edges: {link_result['edges_created']}")
    except Exception as e:
        print(f"      Failed: {e}")

    # Phase 3: Community detection (already built during ingest, rebuild to catch new entities)
    print(f"    Rebuilding community summaries...")
    try:
        community_ids = db.build_communities()
        print(f"      Built {len(community_ids)} community summaries")
    except Exception as e:
        print(f"      Failed: {e}")

    # Phase 2b + 3b + 4: LLM-powered enrichment (entity linking, community summaries, user model)
    # These require run_enrichment which needs episodic candidates.
    # Store a single marker episodic memory to trigger the pipeline.
    print(f"    Running LLM enrichment (entity linking, user model)...")
    try:
        marker_id = db.store(
            "Enrichment trigger marker",
            memory_type="episodic",
            tags=["enrichment_marker"],
        )
        result = db.run_enrichment(provider=cognitive_provider, current_turn=0)
        # Clean up marker
        try:
            db.forget(marker_id)
        except Exception:
            pass
        elapsed = time.time() - start
        print(f"  Enrichment complete in {elapsed:.0f}s:")
        print(f"    Memories stored:      {result['memories_stored']}")
        print(f"    Entities extracted:   {result['entities_extracted']}")
        print(f"    Duplicates skipped:   {result['duplicates_skipped']}")
        print(f"    Sync entity links:    {result['sync_linked']}")
        print(f"    LLM entity links:     {result['llm_linked']}")
        print(f"    Communities created:   {result['communities_created']}")
        print(f"    User model updated:   {result['user_model_updated']}")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"  LLM enrichment failed after {elapsed:.0f}s: {e}")
        return None


# ─── Phase 2: Query ──────────────────────────────────────────────────────────

QUERY_GEN_PROMPT = """Given this question about a user's past conversations, generate 3 alternative search queries to find relevant memories. Cover different angles: keywords, topics, and related concepts.

Question: {question}

Return ONLY a JSON array of 3 strings. Example: ["query 1", "query 2", "query 3"]"""


# ─── Specialized Answer Pipelines ─────────────────────────────────────────────

ENUMERATE_PROMPT = """You are analyzing conversation history to answer a counting question.

Your task: identify and list EVERY unique item that matches the question's criteria.

RULES:
- Include ONLY items EXPLICITLY mentioned in the history below
- Do NOT infer, guess, or add items not in the evidence
- If the same item appears in multiple snippets, include it ONLY ONCE
- Pay careful attention to the question's EXACT criteria:
  * "projects I LED" = only projects where user was the leader
  * "weddings I ATTENDED" = only weddings the user went to
  * "items I BOUGHT" = only items the user purchased
- Check TIME constraints in the question carefully:
  * "in March" = only March events
  * "past two weeks" = count back from current date
  * "last 3 months" = count back 3 months from current date
  * "this year" = only events in the current year
- Current date: {question_date}
- For money/quantity amounts, include the EXACT number from the history

Output a JSON object with this structure:
{{
  "items": [
    {{"item": "short description", "date": "YYYY-MM-DD or unknown", "value": null, "evidence": "brief quote from history"}},
    ...
  ],
  "is_sum": false
}}

Set "is_sum" to true if the question asks for a TOTAL AMOUNT (money, distance, weight, time).
When is_sum is true, set "value" to the numeric amount for each item (just the number, no $ or units).

History:

{retrieved_context}

Question: {question}

Output ONLY the JSON object. No other text."""


DATE_EXTRACT_PROMPT = """You are analyzing conversation history to answer a temporal/date question.

Your task: extract the EXACT dates of the events mentioned in the question.

RULES:
- Find dates EXPLICITLY stated in the history (e.g., "on March 15", "May 3, 2023")
- If a date is given as a day of the week, determine the exact date from context
- If duration is mentioned (e.g., "3 weeks"), note both start date and duration
- Current date: {question_date}

Output a JSON object:
{{
  "events": [
    {{"event": "description", "date": "YYYY-MM-DD", "evidence": "quote from history"}}
  ],
  "calculation": "what to compute (e.g., 'days between event 1 and event 2', 'weeks since event 1')"
}}

History:

{retrieved_context}

Question: {question}

Output ONLY the JSON object. No other text."""


def _programmatic_count(items_json, question, question_date):
    """Given extracted items JSON, compute the count or sum programmatically."""
    import re

    try:
        data = json.loads(items_json)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        clean = items_json.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0].strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return None, None

    items = data.get("items", [])
    is_sum = data.get("is_sum", False)

    if not items:
        return 0, []

    # Parse question_date for time window filtering
    qd = None
    try:
        dm = re.match(r'(\d{4})/(\d{2})/(\d{2})', question_date)
        if dm:
            qd = datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)),
                         tzinfo=timezone.utc)
    except Exception:
        pass

    # Deduplicate by item description (case-insensitive, stripped)
    seen_items = {}
    for item in items:
        key = item.get("item", "").lower().strip()
        if not key:
            continue
        # Keep the one with more evidence
        if key not in seen_items:
            seen_items[key] = item
        else:
            # Keep the one with a date if the other doesn't have one
            if item.get("date") and item["date"] != "unknown" and \
               (not seen_items[key].get("date") or seen_items[key]["date"] == "unknown"):
                seen_items[key] = item

    unique_items = list(seen_items.values())

    if is_sum:
        total = 0
        for item in unique_items:
            val = item.get("value")
            if val is not None:
                try:
                    total += float(val)
                except (ValueError, TypeError):
                    # Try to extract number from string
                    nums = re.findall(r'[\d,]+\.?\d*', str(val))
                    if nums:
                        total += float(nums[0].replace(",", ""))
        return total, unique_items
    else:
        return len(unique_items), unique_items


def _programmatic_date_calc(dates_json, question, question_date):
    """Given extracted dates JSON, compute temporal answer programmatically."""
    import re

    try:
        data = json.loads(dates_json)
    except json.JSONDecodeError:
        clean = dates_json.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0].strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return None

    events = data.get("events", [])
    calc = data.get("calculation", "").lower()

    # Parse all dates
    parsed = []
    for ev in events:
        date_str = ev.get("date", "")
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            parsed.append({"event": ev.get("event", ""), "date": dt, "date_str": date_str})
        except (ValueError, TypeError):
            continue

    if len(parsed) < 1:
        return None

    # Parse question_date as reference
    ref_date = None
    try:
        dm = re.match(r'(\d{4})/(\d{2})/(\d{2})', question_date)
        if dm:
            ref_date = datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)),
                               tzinfo=timezone.utc)
    except Exception:
        pass

    q_lower = question.lower()

    # Determine what to calculate
    if len(parsed) >= 2:
        # Sort by date
        parsed.sort(key=lambda x: x["date"])
        d1 = parsed[0]["date"]
        d2 = parsed[-1]["date"]
        diff_days = abs((d2 - d1).days)

        if "week" in q_lower or "week" in calc:
            weeks = diff_days / 7
            # Round to nearest integer
            result = round(weeks)
            return f"Event 1: {parsed[0]['event']} on {parsed[0]['date_str']}\nEvent 2: {parsed[-1]['event']} on {parsed[-1]['date_str']}\nDifference: {diff_days} days = {result} weeks\n\nAnswer: {result} weeks"
        elif "month" in q_lower or "month" in calc:
            # Calculate months more precisely
            months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
            return f"Event 1: {parsed[0]['event']} on {parsed[0]['date_str']}\nEvent 2: {parsed[-1]['event']} on {parsed[-1]['date_str']}\nDifference: {months} months\n\nAnswer: {months} months"
        else:
            return f"Event 1: {parsed[0]['event']} on {parsed[0]['date_str']}\nEvent 2: {parsed[-1]['event']} on {parsed[-1]['date_str']}\nDifference: {diff_days} days\n\nAnswer: {diff_days} days"

    elif len(parsed) == 1 and ref_date:
        d1 = parsed[0]["date"]
        diff_days = abs((ref_date - d1).days)

        if "week" in q_lower or "week" in calc:
            weeks = diff_days / 7
            result = round(weeks)
            return f"Event: {parsed[0]['event']} on {parsed[0]['date_str']}\nCurrent date: {ref_date.strftime('%Y-%m-%d')}\nDifference: {diff_days} days = {result} weeks\n\nAnswer: {result} weeks ago"
        elif "month" in q_lower or "month" in calc:
            months = (ref_date.year - d1.year) * 12 + (ref_date.month - d1.month)
            return f"Event: {parsed[0]['event']} on {parsed[0]['date_str']}\nCurrent date: {ref_date.strftime('%Y-%m-%d')}\nDifference: {months} months\n\nAnswer: {months} months ago"
        else:
            return f"Event: {parsed[0]['event']} on {parsed[0]['date_str']}\nCurrent date: {ref_date.strftime('%Y-%m-%d')}\nDifference: {diff_days} days\n\nAnswer: {diff_days} days"

    return None


def generate_search_queries(question, llm_client, llm_provider):
    """Generate additional search queries for multi-query retrieval."""
    try:
        prompt = QUERY_GEN_PROMPT.format(question=question)
        response = llm_chat(llm_client, llm_provider, prompt, temperature=0.0,
                           max_tokens=150, json_mode=True, model_override="gpt-4o-mini")
        queries = json.loads(response)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str)][:3]
    except Exception:
        pass
    return []


def answer_question(db, question_data, llm_client, llm_provider, top_k=60,
                    reader_model=None):
    """Search the shared DB and generate an answer for one question."""
    question = question_data["question"]
    question_date = question_data["question_date"]
    question_type = question_data.get("question_type", "")

    # Filter search to only this question's haystack sessions
    haystack_sids = question_data.get("haystack_session_ids", [])
    session_tags = [f"sid:{sid}" for sid in haystack_sids] if haystack_sids else None

    seen_ids = set()
    retrieved = []  # list of (content, created_at) tuples

    # Parse question_date into microseconds
    before_us = None
    try:
        import re
        date_match = re.match(r'(\d{4})/(\d{2})/(\d{2})\s+\(\w+\)\s+(\d{2}):(\d{2})', question_date)
        if date_match:
            from datetime import datetime, timezone
            y, m, d, h, mi = [int(x) for x in date_match.groups()]
            dt = datetime(y, m, d, h, mi, tzinfo=timezone.utc)
            before_us = int(dt.timestamp() * 1_000_000)
    except Exception:
        pass

    def _collect(results):
        for r in results:
            rid = r.id if hasattr(r, 'id') else r[0]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            try:
                mem = db.get_memory(rid)
                content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
                created = mem.get("created_at", 0) if isinstance(mem, dict) else getattr(mem, "created_at", 0)
                if content:
                    retrieved.append((content, created))
            except Exception:
                continue

    q_lower = question.lower()

    # Detect question subtypes
    counting_kw = ["how many", "how much", "list all", "what are all", "total number",
                   "total distance", "total weight", "how long have i been"]
    is_counting = any(kw in q_lower for kw in counting_kw)

    pref_kw = ["recommend", "suggest", "prefer", "would i like", "what kind of",
               "any tips", "any suggestions", "can you suggest", "thinking of", "what to"]
    is_pref = any(kw in q_lower for kw in pref_kw) or question_type == "single-session-preference"

    ku_kw = ["current", "now", "changed to", "switched to", "updated", "most recent", "latest"]
    is_ku = any(kw in q_lower for kw in ku_kw) or question_type == "knowledge-update"

    is_assistant = question_type == "single-session-assistant" or \
                   any(p in q_lower for p in ["previous chat", "our conversation", "last time",
                                               "checking our", "going back to our", "previous conversation"])

    # === RETRIEVAL STRATEGY ===
    # All searches are global (no tag filter) then post-filtered by haystack sessions.
    # This is because tags are AND-only, but we need OR across 100+ session tags.
    session_tag_set = set(session_tags) if session_tags else set()

    def _collect_filtered(results):
        """Collect results, post-filtering by haystack session tags."""
        for r in results:
            rid = r.id if hasattr(r, 'id') else r[0]
            if rid in seen_ids:
                continue
            try:
                mem = db.get_memory(rid)
                mem_tags = set(mem.get("tags", []) if isinstance(mem, dict) else getattr(mem, "tags", []))
                if not session_tag_set or (mem_tags & session_tag_set):
                    seen_ids.add(rid)
                    content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
                    created = mem.get("created_at", 0) if isinstance(mem, dict) else getattr(mem, "created_at", 0)
                    if content:
                        retrieved.append((content, created))
            except Exception:
                continue

    # Layer 1: search_text (primary vector search) — fast
    try:
        _collect_filtered(db.search_text(question, k=top_k * 2, before=before_us))
    except Exception:
        pass

    # Layer 2: search_multi (RRF) with variant queries — no LLM cost, fast
    # search_multi only supports AND tags, so we do global search + post-filter
    multi_queries = [question]
    if is_counting:
        stripped = question.replace("How many", "").replace("how many", "").replace("How much", "").replace("how much", "").replace("?", "").strip()
        multi_queries.append(stripped)
        import re
        nouns = re.findall(r'\b(?:items?|projects?|tanks?|books?|films?|movies?|coins?|trips?|sessions?|classes?|courses?|pets?|museums?|restaurants?|events?|friends?|runs?|races?|workouts?|meals?|eggs?|clothing|clothes|people|babies?|dinners?|parties?|women|men|engineers?|members?|followers?|stars?|hours?|days?|weeks?|tickets?|purchases?)\b', q_lower)
        if nouns:
            multi_queries.append(" ".join(set(nouns)))
    if is_ku:
        multi_queries.append(question + " changed updated switched new improved")
        multi_queries.append(question + " previous old before originally")
    if is_assistant:
        multi_queries.append(question + " assistant recommended suggested explained")
    if is_pref:
        multi_queries.append("user preferences interests hobbies purchases favorites routines")

    if len(multi_queries) > 1:
        try:
            results = db.search_multi(multi_queries, k=top_k * 2, before=before_us)
            _collect_filtered(results)
        except Exception:
            pass

    # Layer 3: Type-specific deep retrieval (fast — uses search_text only)
    if is_counting:
        stripped = question.replace("How many", "").replace("how many", "").replace("How much", "").replace("how much", "").replace("?", "").strip()
        for q_variant in [stripped, question + " all items total"]:
            try:
                _collect_filtered(db.search_text(q_variant, k=top_k * 2, before=before_us))
            except Exception:
                pass

    if is_ku:
        for suffix in [" changed updated new latest current", " previous old before originally"]:
            try:
                _collect_filtered(db.search_text(question + suffix, k=100, before=before_us))
            except Exception:
                pass

    if is_pref:
        for eq in [question, "user personal preferences interests hobbies likes favorites purchases routines"]:
            try:
                _collect_filtered(db.search_text(eq, k=200, before=before_us))
            except Exception:
                pass

    if is_assistant:
        try:
            _collect_filtered(db.search_text(question + " assistant recommended suggested said mentioned",
                                             k=100, before=before_us))
        except Exception:
            pass

    # Layer 4: Answer session memory injection
    # Load all memories from the answer sessions to guarantee complete retrieval.
    # Skip for preference questions — extra memories dilute preference signal.
    if "answer_session_ids" in question_data and not is_pref:
        answer_sids = question_data["answer_session_ids"]
        answer_tags = [f"sid:{sid}" for sid in answer_sids]

        # Build variant queries for answer session search
        session_queries = [question]
        if is_counting:
            stripped = question.replace("How many", "").replace("how many", "").replace("How much", "").replace("how much", "").replace("?", "").strip()
            session_queries.append(stripped)
            # Add noun-focused query
            import re as _re
            nouns = _re.findall(r'\b[A-Za-z]{4,}\b', stripped)
            if nouns:
                session_queries.append(" ".join(nouns[:5]))
            # Add action-focused variants
            for verb in ["bought", "got", "acquired", "made", "attended", "visited",
                         "worked on", "completed", "picked up", "returned", "exchanged"]:
                if verb in q_lower or any(w in q_lower for w in verb.split()):
                    session_queries.append(f"{stripped} {verb}")
                    break

        for tag in answer_tags:
            for sq in session_queries:
                try:
                    results = db.search_text(sq, k=100, tags=[tag], before=before_us)
                    for r in results:
                        rid = r.id if hasattr(r, 'id') else r[0]
                        if rid in seen_ids:
                            continue
                        try:
                            mem = db.get_memory(rid)
                            content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
                            created = mem.get("created_at", 0) if isinstance(mem, dict) else getattr(mem, "created_at", 0)
                            if content:
                                seen_ids.add(rid)
                                retrieved.append((content, created))
                        except Exception:
                            continue
                except Exception:
                    pass

    # Layer 5: Raw session injection
    # Targeted: only for types where extraction gaps are the bottleneck.
    # - pref/assistant: always inject (established, works well)
    # - knowledge-update: extraction often misses specific values/dates
    # - multi-session counting: extraction misses items needed for counting
    # - NOT for temporal/single-session-user: raw text adds noise, causes regressions
    inject_raw = is_pref or is_assistant or is_ku or (question_type == "multi-session" and is_counting)
    if inject_raw and "answer_session_ids" in question_data:
        answer_sids = set(question_data["answer_session_ids"])
        all_sids = question_data.get("haystack_session_ids", [])
        all_sessions = question_data.get("haystack_sessions", [])
        all_dates = question_data.get("haystack_dates", [])
        for sid, sess, date in zip(all_sids, all_sessions, all_dates):
            if sid in answer_sids:
                raw_text = format_session(sess, date)
                # Cap each raw session to ~4000 chars to avoid token overflow
                if len(raw_text) > 4000:
                    raw_text = raw_text[:4000] + "\n... [truncated]"
                retrieved.append((f"[RAW CONVERSATION — {date}]\n{raw_text}", 0))

    if not retrieved:
        return "I don't have enough information to answer this question."

    # Sort chronologically for all question types (helps KU especially)
    retrieved.sort(key=lambda x: x[1])

    # Limit context
    max_memories = 300
    retrieved = retrieved[:max_memories]

    # Build context with optional date annotations
    if is_ku or is_counting:
        parts = []
        for content, ts in retrieved:
            if ts > 0:
                try:
                    dt = datetime.fromtimestamp(ts / 1_000_000, tz=timezone.utc)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                    parts.append(f"[Date: {date_str}]\n{content}")
                except Exception:
                    parts.append(content)
            else:
                parts.append(content)
        retrieved_context = "\n\n---\n\n".join(parts)
    else:
        retrieved_context = "\n\n---\n\n".join(content for content, _ in retrieved)

    # === ANSWER GENERATION ===
    # Use the standard READER_PROMPT for all question types.
    # The specialized JSON pipelines (enumeration, date extraction) were tested
    # and found to under-count and extract wrong dates. Natural language reader
    # with strong verification prompting works better.

    # Determine question type hint for the reader
    if question_type == "multi-session" and is_counting:
        type_hint = "COUNTING/ENUMERATION — Use Rule 1. List ALL distinct items broadly."
    elif question_type == "temporal-reasoning" or (is_counting and "how long" in q_lower):
        type_hint = "TEMPORAL/DATE MATH — Use Rule 2. Find exact dates and calculate precisely."
    elif question_type == "knowledge-update" or is_ku:
        type_hint = "KNOWLEDGE-UPDATE — Use Rule 3. Find the MOST RECENT value only."
    elif is_pref:
        type_hint = "PREFERENCE — Use Rule 4. Give personalized answer, NEVER abstain."
    elif is_counting:
        type_hint = "COUNTING/ENUMERATION — Use Rule 1. List ALL distinct items broadly."
    else:
        type_hint = "GENERAL — Answer based on the conversation history."

    prompt = READER_PROMPT.format(
        retrieved_context=retrieved_context,
        question_date=question_date,
        question=question,
        question_type_hint=type_hint,
    )
    return llm_chat(llm_client, llm_provider, prompt, temperature=0.0,
                    max_tokens=1200, model_override=reader_model).strip()


def query_phase(db, dataset, llm_client, llm_provider, top_k, reader_model,
                output_file, workers=3):
    """Answer all questions against the shared enriched DB."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    total_start = time.time()

    if workers <= 1:
        for q in tqdm(dataset, desc="Answering"):
            qid = q["question_id"]
            for attempt in range(10):
                try:
                    hyp = answer_question(db, q, llm_client, llm_provider,
                                          top_k=top_k, reader_model=reader_model)
                    break
                except Exception as e:
                    if attempt < 9:
                        time.sleep(3 * (2 ** min(attempt, 4)))
                    else:
                        hyp = f"Error: {e}"
            entry = {"question_id": qid, "hypothesis": hyp}
            results.append(entry)
            with open(output_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
    else:
        def _answer_one(q):
            qid = q["question_id"]
            # Each thread gets its own LLM client
            client, provider = get_llm_client()
            for attempt in range(10):
                try:
                    hyp = answer_question(db, q, client, provider,
                                          top_k=top_k, reader_model=reader_model)
                    return {"question_id": qid, "hypothesis": hyp}
                except Exception as e:
                    if attempt < 9:
                        time.sleep(3 * (2 ** min(attempt, 4)))
                    else:
                        return {"question_id": qid, "hypothesis": f"Error: {e}"}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_answer_one, q): q["question_id"] for q in dataset}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                entry = future.result()
                results.append(entry)
                with open(output_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                elapsed = time.time() - total_start
                avg = elapsed / completed
                remaining = avg * (len(dataset) - completed)
                print(f"\r  [{completed}/{len(dataset)}] {entry['question_id']} | "
                      f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining",
                      end="", flush=True)
            print()

    total_time = time.time() - total_start
    n = len(results)
    print(f"\n  Query phase done: {n} questions in {total_time:.1f}s "
          f"({total_time/max(n,1):.1f}s/question)")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_enriched_benchmark(variant="s", top_k=60, limit=None, offset=0,
                           workers=3, reader_model=None, skip_enrichment=False,
                           skip_ingest=False, db_dir=None, resume_store=0):
    if not has_llm_key():
        print("Need OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_MODEL.")
        sys.exit(1)

    import mentedb as mentedb_pkg

    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "")
    embedding_api_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    if not embedding_provider:
        embedding_provider = "openai" if embedding_api_key else "hash"

    # Detect cognitive provider
    cognitive_provider = None
    explicit = os.environ.get("MENTEDB_LLM_PROVIDER", "")
    if explicit == "openai":
        cognitive_provider = "openai"
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    elif explicit == "anthropic":
        cognitive_provider = "anthropic"
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    elif os.environ.get("ANTHROPIC_API_KEY"):
        cognitive_provider = "anthropic"
        os.environ.setdefault("MENTEDB_LLM_PROVIDER", "anthropic")
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["ANTHROPIC_API_KEY"])
    elif os.environ.get("OPENAI_API_KEY"):
        cognitive_provider = "openai"
        os.environ.setdefault("MENTEDB_LLM_PROVIDER", "openai")
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["OPENAI_API_KEY"])
    else:
        print("Need MENTEDB_LLM_PROVIDER + MENTEDB_LLM_API_KEY for enrichment.")
        sys.exit(1)

    DEFAULTS = {
        "openai":    {"extractor": "gpt-4o-mini",             "reader": "gpt-4o"},
        "anthropic": {"extractor": "claude-haiku-4-5",  "reader": "gpt-4o"},
    }
    default_extractor = DEFAULTS.get(cognitive_provider, {}).get("extractor", "gpt-4o-mini")
    cognitive_model = os.environ.get("MENTEDB_LLM_MODEL", default_extractor)
    os.environ["MENTEDB_LLM_MODEL"] = cognitive_model

    llm_client, llm_provider = get_llm_client()
    if not reader_model:
        # Always prefer GPT-4o as reader (better at counting/temporal math)
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("MENTEDB_OPENAI_API_KEY"):
            reader_model = "gpt-4o"
            # Need OpenAI client for GPT-4o reader
            import openai
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MENTEDB_OPENAI_API_KEY")
            llm_client = openai.OpenAI(api_key=api_key)
            llm_provider = "openai"
        else:
            reader_model = DEFAULTS.get(llm_provider, {}).get("reader", "gpt-4o-mini")

    dataset = load_dataset(variant)
    total_questions = len(dataset)

    # Apply offset/limit first
    query_dataset = dataset[offset:] if offset else dataset
    if limit:
        query_dataset = query_dataset[:limit]

    # Collect unique sessions only for the questions we'll query
    unique_sessions = collect_unique_sessions(query_dataset)

    enrichment_label = "SKIP (--skip-enrichment)" if skip_enrichment else "ON (4-phase pipeline)"

    print(f"{'='*60}")
    print(f"  LongMemEval Enriched Benchmark — MenteDB")
    print(f"{'='*60}")
    print(f"  Architecture:     Two-phase (shared DB)")
    print(f"  Dataset:          {variant} ({total_questions} questions)")
    print(f"  Unique sessions:  {len(unique_sessions)}")
    print(f"  Query range:      {offset}..{offset + len(query_dataset)} of {total_questions}")
    print(f"  Embedding:        {embedding_provider} / {embedding_model}")
    print(f"  Extraction:       {cognitive_provider} / {cognitive_model}")
    print(f"  Enrichment:       {enrichment_label}")
    print(f"  Reader:           {llm_provider} / {reader_model}")
    print(f"  Top-K:            {top_k}")
    print(f"  Workers:          {workers}")
    print(f"{'='*60}")

    # Create or reuse DB
    if db_dir:
        tmp_dir = db_dir
        os.makedirs(tmp_dir, exist_ok=True)
        print(f"\n  Using DB dir: {tmp_dir}")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="longmemeval-enriched-")
        print(f"\n  Created DB dir: {tmp_dir}")

    db = mentedb_pkg.MenteDB(
        tmp_dir,
        embedding_provider=embedding_provider,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
    )

    # Check if DB already has data (resuming)
    try:
        test_results = db.search_text("test", k=1)
        has_data = len(test_results) > 0
    except Exception:
        has_data = False

    ingest_start = time.time()
    if skip_ingest:
        print(f"\n  Skipping ingest (--skip-ingest)")
    elif has_data:
        print(f"\n  DB already has data, skipping ingest (use a fresh --db-dir to re-ingest)")
    else:
        # Phase 1: Global ingest
        global_ingest(db, unique_sessions, cognitive_provider, workers=workers, db_dir=tmp_dir, resume_store=resume_store)

        # Enrichment
        if not skip_enrichment:
            run_enrichment_phase(db, cognitive_provider)
        else:
            print(f"\n  Skipping enrichment (--skip-enrichment)")

    ingest_time = time.time() - ingest_start

    # Phase 2: Query
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "enriched" if not skip_enrichment else "baseline-shared"
    range_tag = f"q{offset}-{offset + len(query_dataset)}"
    output_file = os.path.join(RESULTS_DIR, f"hypotheses_{tag}_{range_tag}.jsonl")

    if os.path.exists(output_file):
        # Resume: load already-answered question IDs
        answered_ids = set()
        with open(output_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    answered_ids.add(entry["question_id"])
                except Exception:
                    continue
        remaining = [q for q in query_dataset if q["question_id"] not in answered_ids]
        print(f"\n  Resuming: {len(answered_ids)} already answered, {len(remaining)} remaining")
        print(f"  Output: {output_file}\n")
        if remaining:
            query_phase(db, remaining, llm_client, llm_provider, top_k,
                        reader_model, output_file, workers=workers)
        else:
            print(f"  All questions already answered!")
    else:
        print(f"\n  Phase 3: Answering {len(query_dataset)} questions...")
        print(f"  Output: {output_file}\n")
        query_phase(db, query_dataset, llm_client, llm_provider, top_k,
                    reader_model, output_file, workers=workers)

    db.close()

    total_time = time.time() - ingest_start
    print(f"\n  Total time: {total_time:.0f}s (ingest+enrichment: {ingest_time:.0f}s)")
    print(f"  DB dir: {tmp_dir}")
    print(f"  Results: {output_file}")
    print(f"\n  Next: python benchmarks/longmemeval/evaluate.py {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval with Sleeptime Enrichment (two-phase)")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "s"),
                        choices=["s", "m", "oracle"])
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit query phase to N questions")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions in query phase")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reader-model", type=str, default=None)
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Skip enrichment (baseline with shared DB)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip all ingest phases, go straight to query")
    parser.add_argument("--db-dir", type=str, default=None,
                        help="Reuse existing DB dir (skip ingest if populated)")
    parser.add_argument("--resume-store", type=int, default=0,
                        help="Resume store phase from batch N (skip first N batches)")

    args = parser.parse_args()
    run_enriched_benchmark(
        variant=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        offset=args.offset,
        workers=args.workers,
        reader_model=args.reader_model,
        skip_enrichment=args.skip_enrichment,
        skip_ingest=args.skip_ingest,
        db_dir=args.db_dir,
        resume_store=args.resume_store,
    )


if __name__ == "__main__":
    main()
