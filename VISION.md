# Vision

MenteDB is building toward AI memory that works like a mind, not a database.

Every memory system today does the same thing: store text, embed it, search by similarity, return matches. That's a library. Human memory is nothing like a library. It reconstructs, forgets strategically, has gut feelings, dreams up connections, and builds identity over time.

This document describes what MenteDB has already built, what's in progress, and where its heading.

---

## What's Built

These features are implemented and shipping today.

### Write Time Intelligence
Every memory goes through a quality pipeline before storage. Cosine threshold based contradiction detection, deduplication, and obsolescence checking. Graph edges are created automatically when conflicts or relationships are detected. No LLM required for this path, so its fast, deterministic, and cheap.

### Belief Propagation
When a fact changes, confidence cascades through the knowledge graph. Four propagation formulas based on edge type (Caused, Supports, Contradicts, Supersedes). Stale beliefs are suppressed rather than returned. This is why MenteDB returns correct results even when beliefs change over time.

### Pain Signals
The system tracks negative experiences. When a similar situation arises, it surfaces warnings: "last time you did X it broke Y." Exponential decay so old pain fades, but recent pain is vivid. This is the embryonic version of gut feelings.

### Phantom Detection
The system detects knowledge gaps. When you mention something it has no information about, it flags it. Dual strategy: registered entity lookup and heuristic detection (capitalized words, quoted terms, technical terms). This is the embryonic version of "feeling of knowing."

### Attention Aware Context Assembly
Context is ordered using a U curve that respects how transformers actually attend to token sequences. Critical information goes at the beginning and end of context (high attention zones). Supporting information fills the middle. Five attention zones with different placement strategies.

### Delta Serving
The system tracks what the agent already knows. On subsequent turns it only sends what changed. 90% reduction in memory retrieval tokens over multi turn conversations.

### Cognitive Memory Tiers
Six memory types (Episodic, Semantic, Procedural, AntiPattern, Reasoning, Correction) and five cognitive tiers (Working, Episodic, Semantic, Procedural, Archival) that mirror human memory architecture.

### Knowledge Graph
CSR/CSC graph with eight edge types, BFS/DFS traversal, subgraph extraction, shortest path, and transitive contradiction and cycle detection.

### Custom Storage Engine
Built from scratch in Rust. Write ahead log with LZ4 compression and CRC32 integrity. CLOCK eviction buffer pool. 32KB aligned pages. Crash recovery via WAL replay. No external database dependencies.

### HNSW Vector Index
From scratch multi layer approximate nearest neighbor search with hybrid scoring that fuses vector similarity, tag filters, temporal recency, and salience.

### LLM Extraction Pipeline
Automatic memory extraction from raw conversations. Multi provider support (OpenAI, Anthropic, Ollama). Quality filtering, deduplication, and contradiction checking on extracted memories.

### Consolidation and Decay
Exponential half life decay with logarithmic access boost. Union Find clustering for memory merging. Archival pipeline for old low salience memories. GDPR compliant deletion planning.

### Trajectory Tracking
Records decision states across turns. Tracks open questions, completed decisions, and interrupted threads. Predicts what topics might come up next.

---

## What's In Progress

### Wiring the Hot Path
The most impactful near term work. The context assembler, HNSW index, and belief propagation all exist as standalone features but need to be fully connected in the primary `process_turn` path so every interaction benefits from the full cognitive pipeline.

### Persisting Cognitive State
Pain signals, phantom memories, and trajectory tracking currently live in memory. They need to survive restarts by persisting through the storage engine.

### Temporal Validity
Adding `valid_at` and `invalid_at` timestamps to memory edges. When a fact is contradicted, the old edge gets invalidated with a timestamp rather than just creating a Supersedes edge. This preserves full temporal history and enables "what did the agent believe at time X?" queries.

---

## Where Its Heading

These are the ideas that would make AI memory genuinely feel like a mind. Some are extensions of what already exists. Some are new territory.

### Reconstructive Memory
Instead of retrieving stored records verbatim, reconstruct context from fragments based on the current situation. The same underlying memories should produce different context when debugging vs planning vs reflecting. Belief propagation and context assembly are the foundation.

### Intelligent Forgetting
Retrieval induced suppression: successfully recalling one memory weakens competing memories, making future retrieval cleaner. Gist extraction: over time, detailed episodic memories compress into concise semantic lessons. "Env vars caused the failure, always check staging first" instead of the full 200 line conversation. Consolidation and decay are the foundation.

### Feeling of Knowing
When search returns low confidence partial matches, surface a metacognitive signal: "I have fragmented knowledge about this topic" rather than returning nothing. Track retrieval failures to detect growing knowledge gaps. Phantom detection is the foundation.

### Cognitive Mode Awareness
The mode of work biases what gets remembered. Debugging mode surfaces errors and pain signals more aggressively. Creative mode surfaces analogies and broad associations. Review mode surfaces contradictions and uncertainties. Same query, different mode, different context.

### Dream Engine
A background process that randomly activates memories from different domains, checks for structural similarities that were never explicitly connected, and creates new insight memories. "Your deployment pattern is structurally identical to your database migration pattern." This is where analogical reasoning comes from. Not search. Random creative recombination.

### Emotional Valence
Every memory carries an affect score beyond just salience. Emotionally charged experiences (breakthroughs, frustrations) get stronger retention. Fast pre signals fire before full context assembly: "negative affect associated with this topic area." Pain signals are the prototype, expanded to include positive affect.

### Emergent Identity
Periodically analyze the full memory corpus and extract emergent identity statements. "This agent consistently prefers X over Y." "Strongest expertise in A, weakest in B." Not programmed. Grown from accumulated experience. Memory tiers and cognitive state are the foundation.

### Memory Evolution Through Use
When a memory is used in context and the conversation adds nuance or correction, update the memory in place. Memories that are frequently discussed become more refined and accurate. Memories never accessed stay raw. The system gets smarter just from being used.

### Prospective Memory
Context triggered future intentions. "When entity X comes up, remind about Y." "After N turns without mentioning topic Z, proactively surface it." Not calendar reminders. Situational activations that fire when the right conditions arise naturally. Trajectory tracking is the foundation.

### Context Dependent Recall
Memories organized by situational frames, not just content similarity. "The debugging session on March 12 where we fixed the auth bug" is a frame. When you enter a similar situation later, the whole frame activates together. Walking back into the room reminds you why you went there.

### Automatic Tier Promotion
Episodic memories that get accessed repeatedly promote to semantic. Unused semantic memories demote to archival. Working memory has a hard cap with aggressive eviction. The cognitive tier hierarchy becomes a living system, not just labels.

### Spreading Activation
Accessing one memory temporarily boosts related memories in the knowledge graph. "Python" activates "Django" activates "web framework" activates "deployment." Activation spreads through association networks, surfacing contextually relevant knowledge that wouldnt match a direct similarity search.

---

## The Goal

The goal isnt a better database. Its a system where retrieval IS creation, where forgetting sharpens intelligence, where the system develops intuition from experience, and where memory evolves through use.

A mind.
