"""LLM-parsed agent file ingestion for the harness, mirroring production.

The hosted platform parses agent files with the engine's LLM path (the
AGENT_FILE_PROMPT contract) and only falls back to the deterministic
segmenter on unusable completions. The local pyo3 binding exposes only the
deterministic parser, so this module reproduces the production parse:
the same prompt contract, a real model on Bedrock, atoms stored through the
SDK write path (embedded with the local candle provider).

Prompt text matches crates/mentedb/src/agent_file.rs AGENT_FILE_PROMPT.
"""

import json
import os
import time

PROMPT = (
    "You parse agent instruction files into individual memories for a memory "
    "database. The file may be in ANY format (markdown, plain text, YAML, JSON "
    "persona, numbered lists), ANY language, for ANY kind of agent (coding "
    "assistant, customer support, sales, trading, scheduling, personal).\n\n"
    "Return ONLY a JSON array. Each element:\n"
    '{"content": string, "type": "semantic" | "procedural" | "anti_pattern", '
    '"section": string, "trigger": optional string, "exemplars": optional array of strings}\n\n'
    "Rules:\n"
    "- One atomic instruction or fact per element. Split compound rules.\n"
    "- content must be self contained: include the context needed to apply it "
    "alone (which project, product, tool, or situation it belongs to). Preserve "
    "the meaning exactly; keep concrete values, names, numbers, and commands; "
    "never invent or generalize away specifics.\n"
    "- type: anti_pattern for things the agent must never do; procedural for how "
    "to do something, workflows, commands, and rules of conduct; semantic for "
    "facts, preferences, and background.\n"
    "- trigger: set ONLY when the rule governs one specific recurring action or "
    "activity of the agent, and name it as a short lowercase kebab case slug. "
    "Examples across domains: git-commit, pr-create, order-refund, "
    "ticket-escalation, trade-entry, email-send, meeting-schedule, code-change, "
    "reply-style. When in doubt, omit trigger.\n"
    "- exemplars: when you set a trigger for a standing directive that governs a "
    "whole activity (a way of working or replying rather than one discrete "
    "command), also give 3 to 5 short example user requests that would enter "
    "that activity, phrased the way real users ask. Omit for narrow tool "
    "actions.\n"
    "- IMPORTANT: rules that apply to EVERY act of a broad activity, such as "
    "how to write user visible text, how to style components, or conventions "
    "for any code in a language, are standing directives even when phrased as "
    "small tips. They MUST get a trigger naming the activity (ui-text, "
    "styling, go-code) plus exemplars. These are the rules retrieval cannot "
    "find from task wording alone, so the trigger is their only road.\n"
    "- section: the top level heading or theme of the file this element came "
    "from, a short stable name repeated exactly for every element of that part "
    "(for example App-server API, Testing, Refund policy). Elements of one part "
    "of the file must share the same section string.\n"
    "- Skip pure formatting, tables of contents, and import references.\n"
    "- Output the JSON array only. No markdown fences, no commentary."
)

CHUNK_CHARS = int(os.environ.get("MDB_CHUNK_CHARS", "12000"))


def chunks_of(content: str):
    if len(content) <= CHUNK_CHARS:
        return [content]
    out, cur = [], ""
    for line in content.splitlines(keepends=True):
        if len(cur) + len(line) > CHUNK_CHARS and cur and line.startswith("#"):
            out.append(cur)
            cur = ""
        cur += line
        if len(cur) > CHUNK_CHARS * 2:
            out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur)
    return out


def parse_atoms(raw: str):
    start, end = raw.find("["), raw.rfind("]")
    if start < 0:
        return []
    items = None
    if end > start:
        try:
            items = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            items = None
    if items is None:
        # A truncated completion (output token cap) cuts the array mid
        # element. Recover every complete object up to the cut.
        tail = raw[start:]
        cut = tail.rfind("}")
        if cut < 0:
            return []
        try:
            items = json.loads(tail[: cut + 1] + "]")
        except json.JSONDecodeError:
            return []
    atoms = []
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("content"), str):
            continue
        content = it["content"].strip()
        if len(content) < 12:
            continue
        mtype = it.get("type") or "semantic"
        if mtype not in ("semantic", "procedural", "anti_pattern"):
            mtype = "semantic"
        tags = []
        sec = it.get("section")
        if isinstance(sec, str):
            slug = "-".join(
                p for p in "".join(c if c.isalnum() else "-" for c in sec.lower()).split("-") if p
            )[:48].strip("-")
            if slug:
                tags.append(f"section:{slug}")
        trig = it.get("trigger")
        if isinstance(trig, str):
            slug = "-".join(
                p for p in "".join(c if c.isalnum() else "-" for c in trig.lower()).split("-") if p
            )
            if slug and len(slug) <= 48 and slug != "always":
                tags.append(f"trigger:{slug}")
        atoms.append({"content": content[:1800], "memory_type": mtype, "tags": tags})
        # Exemplar turns are mode activation anchors, mirroring the engine
        # parser: stored as separate atoms tagged mode-exemplar:<trigger>.
        trig_slug = next((t.split(":", 1)[1] for t in tags if t.startswith("trigger:")), None)
        exemplars = it.get("exemplars")
        if trig_slug and isinstance(exemplars, list):
            for ex in exemplars[:6]:
                if isinstance(ex, str) and 4 <= len(ex.strip()) <= 400:
                    atoms.append({"content": ex.strip(), "memory_type": "semantic",
                                  "tags": [f"mode-exemplar:{trig_slug}"]})
    return atoms


def parse_file(content: str, model: str):
    import boto3

    client = boto3.Session(
        profile_name=os.environ.get("AWS_PROFILE"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    ).client("bedrock-runtime")
    atoms = []
    calls = 0
    for chunk in chunks_of(content):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": chunk}],
            "system": PROMPT,
        }
        for attempt in range(4):
            try:
                r = client.invoke_model(
                    modelId=model,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                data = json.loads(r["body"].read())
                text = "".join(b.get("text", "") for b in data.get("content", []))
                atoms.extend(parse_atoms(text))
                calls += 1
                break
            except Exception:
                if attempt < 3:
                    time.sleep(2**attempt * 2)
                    continue
                raise
    return atoms, calls


def llm_ingest(db, content: str, model: str, cache_path: str | None = None) -> dict:
    """Parse (or load a cached parse) and store. Caching the parse pins the
    atom set so retrieval and compliance reruns measure retrieval and model
    variance, never parse variance."""
    calls = 0
    if cache_path and os.path.exists(cache_path):
        atoms = json.load(open(cache_path))
        parsed_by = "llm-cached"
    else:
        atoms, calls = parse_file(content, model)
        parsed_by = "llm"
        if cache_path:
            json.dump(atoms, open(cache_path, "w"), indent=1)
    stored = 0
    for a in atoms:
        db.store(a["content"], memory_type=a["memory_type"], tags=a["tags"] or None)
        stored += 1
    return {"parsed_by": parsed_by, "llm_chunks": calls, "stored": stored}
