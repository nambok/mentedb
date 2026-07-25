//! Agent file ingestion: turn a standing instruction file (CLAUDE.md,
//! AGENTS.md, .cursorrules, a system prompt) into individual memories that
//! are retrieved on demand instead of injected whole into every prompt.
//!
//! Two parsers, one contract:
//! - The primary parser is an LLM (`ingest_agent_file_llm`, behind the
//!   `enrichment` feature): agent files come in any format, any language,
//!   for any kind of agent, and only a model parses that reliably. This is
//!   a one time cost per file, nothing like a model call per write; the
//!   write path stays deterministic. The model also names action triggers
//!   with an open vocabulary (git-commit, order-refund, trade-entry), so
//!   nothing here is specific to developer files.
//! - The fallback parser (`ingest_agent_file`) is deterministic markdown
//!   segmentation for installs with no LLM configured: heading paths for
//!   self contained memories, sentence level atomization, type
//!   classification, and action triggers by embedding similarity to anchor
//!   sentences, never string matching.
//! - Nothing is marked `scope:always` by either parser. The point of
//!   ingesting an agent file is that no rule rides every turn; rules arrive
//!   by topic or by action.
//!
//! Memories go through the normal write pipeline one by one, so write time
//! deduplication and supersession apply: re ingesting an edited file updates
//! the store instead of duplicating it. Embeddings are computed in batches
//! ahead of the stores, so a large file does not pay one provider round trip
//! per line.

use mentedb_core::memory::{MemoryNode, MemoryType};
use mentedb_core::types::{AgentId, UserId};

use crate::{MenteDb, MenteResult};

/// Options for [`MenteDb::ingest_agent_file`]. All thresholds live here, not
/// as magic numbers in the implementation.
#[derive(Debug, Clone)]
pub struct AgentFileIngestOptions {
    /// Owner of the created memories; nil means shared knowledge.
    pub agent_id: AgentId,
    /// Orthogonal user owner; nil means shared.
    pub user_id: UserId,
    /// A provenance tag added to every memory, so the whole ingest can be
    /// listed, re done, or removed as a unit. Default `source:agent-file`.
    pub source_tag: String,
    /// Long prose above this many characters is split into atomic
    /// statements.
    pub max_atom_chars: usize,
    /// Segments shorter than this are noise (stray words, separators) and
    /// are skipped.
    pub min_atom_chars: usize,
    /// Hard cap on stored content length per memory.
    pub max_content_chars: usize,
    /// Batch size for embedding calls.
    pub embed_batch_size: usize,
    /// Minimum cosine similarity between an atom and a trigger anchor for
    /// the atom to receive that action trigger tag. Embedding thresholds are
    /// embedder coupled; the default is backed by measured separations on
    /// the bundled candle embedder (2026-07: true commit rules 0.53 to 0.72
    /// including rephrasings, nearest non commit rules 0.35, unrelated prose
    /// under 0.22), and prefers erring high, a rule firing at the wrong
    /// moment erodes trust faster than a missing rule. Non semantic hash
    /// embeddings show no separation at all, so installs on the hash
    /// fallback assign no anchor triggers.
    pub trigger_min_similarity: f32,
    /// Chunk size in characters for LLM parsing of large files; chunks split
    /// at section boundaries.
    pub llm_chunk_chars: usize,
}

impl Default for AgentFileIngestOptions {
    fn default() -> Self {
        Self {
            agent_id: AgentId::nil(),
            user_id: UserId::nil(),
            source_tag: "source:agent-file".to_string(),
            max_atom_chars: 350,
            min_atom_chars: 12,
            max_content_chars: 1800,
            embed_batch_size: 256,
            trigger_min_similarity: 0.45,
            llm_chunk_chars: 24_000,
        }
    }
}

/// What an ingest did, in numbers the caller can show a user.
#[derive(Debug, Clone, Default)]
pub struct AgentFileIngestReport {
    /// Atomic candidates produced by segmentation.
    pub candidates: usize,
    /// Memories actually stored (candidates minus dedup and failures).
    pub stored: usize,
    /// Candidates the write pipeline folded into an existing memory.
    pub deduplicated: usize,
    /// Stored memories by type.
    pub semantic: usize,
    pub procedural: usize,
    pub anti_pattern: usize,
    /// Stored memories carrying a `trigger:` action tag.
    pub trigger_tagged: usize,
    /// Distinct section paths seen.
    pub sections: usize,
    /// Rough size of the whole file in tokens (length divided by four, an
    /// estimate for comparison displays, not a tokenizer).
    pub file_token_estimate: usize,
    /// Rough average tokens per stored memory (same estimate).
    pub avg_memory_token_estimate: usize,
    /// Which parser produced the atoms: "llm" or "deterministic".
    pub parsed_by: &'static str,
    /// The distinct action trigger names assigned, so callers can discover
    /// the open vocabulary the parser chose ("order-refund",
    /// "ticket-escalation") instead of guessing slugs.
    pub triggers: Vec<String>,
    /// Number of chunks sent to the LLM (zero for the deterministic parser).
    pub llm_chunks: usize,
}

/// One segmented, classified atom ready to store. Exposed so harnesses and
/// the demo can inspect classification without storing.
#[derive(Debug, Clone)]
pub struct AgentFileAtom {
    /// Section path, for example "Conventions > Commits".
    pub section: String,
    /// The atomic statement, without the section prefix.
    pub text: String,
    /// Full stored content: section prefix plus text.
    pub content: String,
    pub memory_type: MemoryType,
    pub tags: Vec<String>,
}

/// Action trigger anchors: each trigger is defined by exemplar sentences,
/// and an atom gets the trigger when its embedding lands close enough to an
/// anchor. Phrasing independent by construction: "no co-author trailers",
/// "when committing, keep the subject to one line", and "commit style is
/// conventional" all land near the same anchors without any string
/// matching. The gate is embedding based, so its threshold is embedder
/// coupled and lives in [`AgentFileIngestOptions::trigger_min_similarity`],
/// never hardcoded.
const TRIGGER_ANCHORS: &[(&str, &[&str])] = &[
    (
        "git-commit",
        &[
            "rules for writing git commit messages, their style, format, and subject lines",
            "what a commit message must look like when committing code changes",
            "conventions for commit trailers, signing, and commit authorship",
        ],
    ),
    (
        "pr-create",
        &[
            "rules for opening pull requests and writing PR descriptions, titles, and bodies",
            "what to include when creating a pull request for review",
        ],
    ),
    (
        "git-push",
        &[
            "rules about pushing code to remote branches, force pushing, and protected branches",
            "when it is allowed to push commits to the main branch",
        ],
    ),
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Best matching trigger for an atom embedding, if any anchor clears the
/// gate. Anchors are grouped per trigger; the best anchor wins.
fn trigger_for_embedding(
    atom: &[f32],
    anchor_embeddings: &[(usize, Vec<f32>)],
    gate: f32,
) -> Option<&'static str> {
    let mut best: Option<(usize, f32)> = None;
    for (group, emb) in anchor_embeddings {
        let sim = cosine(atom, emb);
        if best.is_none_or(|(_, b)| sim > b) {
            best = Some((*group, sim));
        }
    }
    match best {
        Some((group, sim)) if sim >= gate => Some(TRIGGER_ANCHORS[group].0),
        _ => None,
    }
}

const RULE_STARTS: &[&str] = &[
    "never", "no ", "do not", "don't", "always", "must", "avoid", "use ", "prefer", "keep ",
    "remove ", "treat ",
];
const RULE_CONTAINS: &[&str] = &["never", "must not", "do not", "don't"];
const PROC_SECTION: &[&str] = &[
    "command", "build", "test", "lint", "deploy", "release", "workflow", "setup", "install",
];
const PROC_STARTS: &[&str] = &["run ", "to ", "call ", "execute ", "verify "];
const PROC_CODE: &[&str] = &[
    "`cargo ", "`npm ", "`npx ", "`pip ", "`docker ", "`git ", "`sh ", "`bash ",
];

fn lower_contains(haystack_lower: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack_lower.contains(n))
}

fn lower_starts(haystack_lower: &str, starts: &[&str]) -> bool {
    starts.iter().any(|s| haystack_lower.starts_with(s))
}

/// Segment a markdown agent file into (section path, text) candidates.
fn segment(md: &str) -> Vec<(String, String)> {
    let mut path: Vec<(usize, String)> = Vec::new();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut buf_is_bullet = false;
    let mut in_code = false;
    let mut code_buf: Vec<String> = Vec::new();
    // True when the line directly above the opening fence held text (no
    // blank line between): only that text is a caption for the block.
    let mut caption_adjacent = false;

    fn section_of(path: &[(usize, String)]) -> String {
        path.iter()
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(" > ")
    }

    fn flush(
        buf: &mut Vec<String>,
        is_bullet: &mut bool,
        out: &mut Vec<(String, String)>,
        sec: &str,
    ) {
        let text = buf
            .iter()
            .map(|l| l.trim())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        buf.clear();
        *is_bullet = false;
        if text.len() >= 4 {
            out.push((sec.to_string(), text));
        }
    }

    for line in md.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_code {
                let code = code_buf.join("\n").trim().to_string();
                if !code.is_empty() {
                    // Fuse the block with the short caption right above it
                    // ("Run these before committing:"), so the commands are
                    // retrievable from natural language questions instead of
                    // embedding as bare code.
                    let caption = match out.last() {
                        Some((sec, text))
                            if caption_adjacent
                                && *sec == section_of(&path)
                                && text.len() <= 120
                                && !text.starts_with("code: ") =>
                        {
                            Some(out.pop().expect("last exists").1)
                        }
                        _ => None,
                    };
                    let content = match caption {
                        Some(c) => format!("{c} code: {code}"),
                        None => format!("code: {code}"),
                    };
                    out.push((section_of(&path), content));
                }
                code_buf.clear();
                in_code = false;
            } else {
                caption_adjacent = !buf.is_empty();
                flush(&mut buf, &mut buf_is_bullet, &mut out, &section_of(&path));
                in_code = true;
            }
            continue;
        }
        if in_code {
            code_buf.push(line.to_string());
            continue;
        }
        // Headings adjust the section path.
        let hashes = line.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&hashes) && line.chars().nth(hashes) == Some(' ') {
            flush(&mut buf, &mut buf_is_bullet, &mut out, &section_of(&path));
            let title = line[hashes + 1..].trim().to_string();
            path.retain(|(l, _)| *l < hashes);
            path.push((hashes, title));
            continue;
        }
        // Import style lines (@file) are references, not memories.
        if trimmed.starts_with('@') && !trimmed.contains(' ') {
            continue;
        }
        let is_bullet_line = {
            let t = trimmed;
            t.starts_with("- ")
                || t.starts_with("* ")
                || t.starts_with("+ ")
                || t.chars().take_while(|c| c.is_ascii_digit()).count() > 0
                    && t.trim_start_matches(|c: char| c.is_ascii_digit())
                        .starts_with(". ")
        };
        if is_bullet_line {
            flush(&mut buf, &mut buf_is_bullet, &mut out, &section_of(&path));
            let stripped = trimmed
                .trim_start_matches(|c: char| {
                    c == '-' || c == '*' || c == '+' || c.is_ascii_digit()
                })
                .trim_start_matches(". ")
                .trim_start()
                .to_string();
            buf.push(stripped);
            buf_is_bullet = true;
            continue;
        }
        if trimmed.is_empty() {
            flush(&mut buf, &mut buf_is_bullet, &mut out, &section_of(&path));
            continue;
        }
        if buf_is_bullet && (line.starts_with("  ") || line.starts_with('\t')) {
            buf.push(trimmed.to_string());
            continue;
        }
        buf.push(trimmed.to_string());
    }
    flush(&mut buf, &mut buf_is_bullet, &mut out, &section_of(&path));
    out
}

/// Split long prose into atomic statements at sentence and separator
/// boundaries, re grouped to at most `max_chars` per atom.
fn atomize(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars || text.starts_with("code: ") {
        return vec![text.to_string()];
    }
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        current.push(c);
        let boundary = match c {
            '.' | '!' | '?' => chars.peek().is_none_or(|n| n.is_whitespace()),
            '\u{00b7}' | '|' => true,
            _ => false,
        };
        if boundary && current.trim().len() >= 40 {
            parts.push(
                current
                    .trim()
                    .trim_end_matches(['\u{00b7}', '|'])
                    .trim()
                    .to_string(),
            );
            current = String::new();
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    // Re group small parts up to the cap.
    let mut out: Vec<String> = Vec::new();
    let mut chunk = String::new();
    for part in parts {
        if !chunk.is_empty() && chunk.len() + part.len() + 1 > max_chars {
            out.push(chunk.clone());
            chunk.clear();
        }
        if chunk.is_empty() {
            chunk = part;
        } else {
            chunk.push(' ');
            chunk.push_str(&part);
        }
    }
    if !chunk.is_empty() {
        out.push(chunk);
    }
    out
}

fn section_slug(section: &str) -> String {
    let mut slug = String::new();
    for c in section.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
        if slug.len() >= 48 {
            break;
        }
    }
    slug.trim_matches('-').to_string()
}

/// Classify one atom. Deliberately deterministic; ambiguous content lands on
/// `Semantic`, which only affects type quotas, never whether it is stored.
fn classify(section: &str, text: &str) -> (MemoryType, Vec<String>) {
    let lower = text.to_lowercase();
    let section_lower = section.to_lowercase();
    let mut tags = Vec::new();
    let slug = section_slug(section);
    if !slug.is_empty() {
        tags.push(format!("section:{slug}"));
    }
    let is_rule = lower_starts(&lower, RULE_STARTS) || lower_contains(&lower, RULE_CONTAINS);
    let memory_type = if text.starts_with("code: ")
        || lower_starts(&lower, PROC_STARTS)
        || lower_contains(&lower, PROC_CODE)
        || lower_contains(&section_lower, PROC_SECTION)
    {
        MemoryType::Procedural
    } else if is_rule && lower_contains(&lower, RULE_CONTAINS) {
        MemoryType::AntiPattern
    } else if is_rule {
        MemoryType::Procedural
    } else {
        MemoryType::Semantic
    };
    (memory_type, tags)
}

/// Segment and classify a file without storing anything. The demo and the
/// coverage harness use this to show classification before ingest.
pub fn plan_agent_file(content: &str, opts: &AgentFileIngestOptions) -> Vec<AgentFileAtom> {
    let mut atoms = Vec::new();
    for (section, text) in segment(content) {
        for atom_text in atomize(&text, opts.max_atom_chars) {
            if atom_text.len() < opts.min_atom_chars {
                continue;
            }
            let (memory_type, tags) = classify(&section, &atom_text);
            let mut stored = if section.is_empty() {
                atom_text.clone()
            } else {
                format!("{section}: {atom_text}")
            };
            stored.truncate(opts.max_content_chars);
            atoms.push(AgentFileAtom {
                section: section.clone(),
                text: atom_text,
                content: stored,
                memory_type,
                tags,
            });
        }
    }
    atoms
}

impl MenteDb {
    /// Ingest an agent instruction file as individual memories. See the
    /// module docs for the design; the short version: atomic memories,
    /// section provenance, action triggers where they apply, nothing pinned
    /// to every turn, normal write pipeline so re ingesting an edited file
    /// deduplicates instead of duplicating.
    pub fn ingest_agent_file(
        &self,
        content: &str,
        opts: &AgentFileIngestOptions,
    ) -> MenteResult<AgentFileIngestReport> {
        let atoms = plan_agent_file(content, opts);
        let mut report = AgentFileIngestReport {
            candidates: atoms.len(),
            file_token_estimate: content.len() / 4,
            parsed_by: "deterministic",
            ..Default::default()
        };
        self.store_atoms(atoms, opts, &mut report, true)?;
        Ok(report)
    }

    /// Shared tail for both parsers: batch embed, assign anchor based
    /// triggers when asked (the fallback parser; the LLM names its own),
    /// store through the full write pipeline, and fill the report.
    fn store_atoms(
        &self,
        mut atoms: Vec<AgentFileAtom>,
        opts: &AgentFileIngestOptions,
        report: &mut AgentFileIngestReport,
        assign_anchor_triggers: bool,
    ) -> MenteResult<()> {
        // Batch embed ahead of the stores so a large file pays one provider
        // round trip per batch, not per line.
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(atoms.len());
        for chunk in atoms.chunks(opts.embed_batch_size.max(1)) {
            let texts: Vec<&str> = chunk.iter().map(|a| a.content.as_str()).collect();
            match self.embed_batch(&texts)? {
                Some(mut batch) => embeddings.append(&mut batch),
                None => {
                    embeddings.extend(std::iter::repeat_n(Vec::new(), texts.len()));
                }
            }
        }

        // Fallback trigger assignment: embedding similarity to anchor
        // sentences, phrasing independent by construction. Skipped when the
        // LLM already named triggers.
        if assign_anchor_triggers {
            let anchor_texts: Vec<&str> = TRIGGER_ANCHORS
                .iter()
                .flat_map(|(_, anchors)| anchors.iter().copied())
                .collect();
            if let Some(anchor_embs) = self.embed_batch(&anchor_texts)? {
                let mut grouped: Vec<(usize, Vec<f32>)> = Vec::new();
                let mut i = 0;
                for (group, (_, anchors)) in TRIGGER_ANCHORS.iter().enumerate() {
                    for _ in anchors.iter() {
                        grouped.push((group, anchor_embs[i].clone()));
                        i += 1;
                    }
                }
                for (atom, emb) in atoms.iter_mut().zip(&embeddings) {
                    if let Some(trigger) =
                        trigger_for_embedding(emb, &grouped, opts.trigger_min_similarity)
                    {
                        atom.tags.push(format!("trigger:{trigger}"));
                    }
                }
            }
        }

        let mut sections = std::collections::HashSet::new();
        let before = self.count_by_tag(&opts.source_tag);
        let mut stored_tokens = 0usize;
        for (atom, embedding) in atoms.iter().zip(embeddings) {
            let mut node = MemoryNode::new(
                opts.agent_id,
                atom.memory_type,
                atom.content.clone(),
                embedding,
            );
            node.user_id = opts.user_id;
            node.tags = atom.tags.clone();
            node.tags.push(opts.source_tag.clone());
            self.store(node)?;
            sections.insert(atom.section.clone());
            stored_tokens += atom.content.len() / 4;
            match atom.memory_type {
                MemoryType::Procedural => report.procedural += 1,
                MemoryType::AntiPattern => report.anti_pattern += 1,
                _ => report.semantic += 1,
            }
            if let Some(trigger) = atom.tags.iter().find_map(|t| t.strip_prefix("trigger:")) {
                report.trigger_tagged += 1;
                if !report.triggers.iter().any(|t| t == trigger) {
                    report.triggers.push(trigger.to_string());
                }
            }
        }
        let after = self.count_by_tag(&opts.source_tag);
        report.stored = after.saturating_sub(before);
        report.deduplicated = report.candidates.saturating_sub(report.stored);
        report.sections = sections.len();
        if report.stored > 0 {
            report.avg_memory_token_estimate = stored_tokens / report.candidates.max(1);
        }
        Ok(())
    }

    /// How many live memories carry a tag. Bitmap candidates are verified
    /// against the node, mirroring count_standing_rules.
    fn count_by_tag(&self, tag: &str) -> usize {
        self.index
            .bitmap
            .query_tag(tag)
            .into_iter()
            .filter(|id| {
                self.get_memory(*id)
                    .map(|n| n.tags.iter().any(|t| t == tag))
                    .unwrap_or(false)
            })
            .count()
    }
}

/// System prompt for the LLM parser. The contract: any file, any format,
/// any language, any kind of agent, atomic self contained memories, open
/// trigger vocabulary, JSON only.
#[cfg(feature = "enrichment")]
const AGENT_FILE_PROMPT: &str = "You parse agent instruction files into individual memories for a memory database. The file may be in ANY format (markdown, plain text, YAML, JSON persona, numbered lists), ANY language, for ANY kind of agent (coding assistant, customer support, sales, trading, scheduling, personal).\n\nReturn ONLY a JSON array. Each element:\n{\"content\": string, \"type\": \"semantic\" | \"procedural\" | \"anti_pattern\", \"trigger\": optional string, \"exemplars\": optional array of strings}\n\nRules:\n- One atomic instruction or fact per element. Split compound rules.\n- content must be self contained: include the context needed to apply it alone (which project, product, tool, or situation it belongs to). Preserve the meaning exactly; keep concrete values, names, numbers, and commands; never invent or generalize away specifics.\n- type: anti_pattern for things the agent must never do; procedural for how to do something, workflows, commands, and rules of conduct; semantic for facts, preferences, and background.\n- trigger: set ONLY when the rule governs one specific recurring action or activity of the agent, and name it as a short lowercase kebab case slug. Examples across domains: git-commit, pr-create, order-refund, ticket-escalation, trade-entry, email-send, meeting-schedule, code-change, reply-style. When in doubt, omit trigger.\n- exemplars: when you set a trigger for a standing directive that governs a whole activity (a way of working or replying rather than one discrete command), also give 3 to 5 short example user requests that would enter that activity, phrased the way real users ask (for code-change: \"fix this bug in the auth flow\", \"refactor the retry logic\"). Omit for narrow tool actions.\n- Skip pure formatting, tables of contents, and import references.\n- Output the JSON array only. No markdown fences, no commentary.";

/// Split a large file into LLM sized chunks at section boundaries so no
/// rule is cut in half.
#[cfg(feature = "enrichment")]
fn chunk_for_llm(content: &str, chunk_chars: usize) -> Vec<String> {
    if content.len() <= chunk_chars {
        return vec![content.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in content.lines() {
        if current.len() + line.len() > chunk_chars && !current.is_empty() && line.starts_with('#')
        {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
        // Hard stop for files without headings.
        if current.len() > chunk_chars * 2 {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Validate an LLM proposed trigger into the tag vocabulary: lowercase
/// kebab case, bounded length, never a pin.
#[cfg(feature = "enrichment")]
fn valid_trigger(raw: &str) -> Option<String> {
    let slug: String = raw
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if slug.is_empty() || slug.len() > 48 || slug == "always" {
        return None;
    }
    Some(slug)
}

/// Parse the LLM response into atoms. Tolerant of fences and stray prose
/// around the array; strict about each element.
#[cfg(feature = "enrichment")]
fn parse_llm_atoms(raw: &str, opts: &AgentFileIngestOptions) -> Vec<AgentFileAtom> {
    let start = match raw.find('[') {
        Some(i) => i,
        None => return Vec::new(),
    };
    let end = match raw.rfind(']') {
        Some(i) if i > start => i,
        _ => return Vec::new(),
    };
    let parsed: Vec<serde_json::Value> = match serde_json::from_str(&raw[start..=end]) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut atoms = Vec::new();
    for item in parsed {
        let Some(content) = item.get("content").and_then(|c| c.as_str()) else {
            continue;
        };
        let mut content = content.trim().to_string();
        if content.len() < opts.min_atom_chars {
            continue;
        }
        content.truncate(opts.max_content_chars);
        let memory_type = match item.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "procedural" => MemoryType::Procedural,
            "anti_pattern" => MemoryType::AntiPattern,
            _ => MemoryType::Semantic,
        };
        let mut tags = Vec::new();
        let mut trigger_slug = None;
        if let Some(trigger) = item
            .get("trigger")
            .and_then(|t| t.as_str())
            .and_then(valid_trigger)
        {
            tags.push(format!("trigger:{trigger}"));
            trigger_slug = Some(trigger);
        }
        atoms.push(AgentFileAtom {
            section: String::new(),
            text: content.clone(),
            content,
            memory_type,
            tags,
        });
        // Standing directives arrive with exemplar turns: short example
        // requests that enter the mode. They are stored as activation
        // anchors for the injection mode channel (mode-exemplar tags keep
        // them out of ordinary context), never as instructions.
        if let (Some(trigger), Some(exemplars)) = (
            trigger_slug,
            item.get("exemplars").and_then(|e| e.as_array()),
        ) {
            for ex in exemplars.iter().take(6) {
                let Some(text) = ex.as_str() else { continue };
                let text = text.trim();
                if text.len() < 4 || text.len() > 400 {
                    continue;
                }
                atoms.push(AgentFileAtom {
                    section: String::new(),
                    text: text.to_string(),
                    content: text.to_string(),
                    memory_type: MemoryType::Semantic,
                    tags: vec![format!("mode-exemplar:{trigger}")],
                });
            }
        }
    }
    atoms
}

#[cfg(feature = "enrichment")]
impl MenteDb {
    /// Ingest an agent file with an LLM parser: any format, any language,
    /// any kind of agent, open trigger vocabulary. One model pass per file
    /// (chunked at section boundaries for large files), a one time cost;
    /// the write path stays deterministic. Falls back to the deterministic
    /// parser when the model returns nothing usable, so ingest never comes
    /// back empty because of a bad completion.
    pub async fn ingest_agent_file_llm<P: mentedb_extraction::provider::ExtractionProvider>(
        &self,
        content: &str,
        opts: &AgentFileIngestOptions,
        provider: &P,
    ) -> MenteResult<AgentFileIngestReport> {
        let chunks = chunk_for_llm(content, opts.llm_chunk_chars);
        let mut atoms: Vec<AgentFileAtom> = Vec::new();
        for chunk in &chunks {
            let raw = provider
                .extract(chunk, AGENT_FILE_PROMPT)
                .await
                .map_err(|e| crate::MenteError::Storage(format!("agent file parse: {e}")))?;
            atoms.extend(parse_llm_atoms(&raw, opts));
        }
        if atoms.is_empty() {
            let mut report = self.ingest_agent_file(content, opts)?;
            report.llm_chunks = chunks.len();
            return Ok(report);
        }
        let mut report = AgentFileIngestReport {
            candidates: atoms.len(),
            file_token_estimate: content.len() / 4,
            parsed_by: "llm",
            llm_chunks: chunks.len(),
            ..Default::default()
        };
        self.store_atoms(atoms, opts, &mut report, false)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_headings_bullets_paragraphs_code() {
        let md = "# Project\n\nIntro paragraph about the project.\n\n## Conventions\n\n- Commit messages: conventional style, single line, no emojis\n- Use thiserror for error types\n\n```bash\ncargo test\n```\n";
        let segs = segment(md);
        let texts: Vec<&str> = segs.iter().map(|(_, t)| t.as_str()).collect();
        assert!(texts.iter().any(|t| t.contains("Intro paragraph")));
        assert!(texts.iter().any(|t| t.contains("conventional style")));
        assert!(texts.iter().any(|t| t.starts_with("code: cargo test")));
        let conv = segs
            .iter()
            .find(|(_, t)| t.contains("thiserror"))
            .expect("bullet");
        assert_eq!(conv.0, "Project > Conventions", "section path must nest");
    }

    #[test]
    fn atomizes_long_prose_without_losing_content() {
        let text = "First rule of the system. Second rule is longer and matters a great deal to everyone involved. Third rule closes it out for good measure and then some.".repeat(4);
        let atoms = atomize(&text, 120);
        assert!(atoms.len() > 3, "long prose must split: {}", atoms.len());
        assert!(atoms.iter().all(|a| a.len() <= 200));
        let rejoined: usize = atoms.iter().map(|a| a.len()).sum();
        assert!(
            rejoined as f64 > text.len() as f64 * 0.95,
            "splitting must not lose content"
        );
    }

    #[test]
    fn code_blocks_fuse_with_adjacent_captions_only() {
        // Adjacent caption, no blank line: fused into one retrievable atom.
        let md = "# Ops\n\nRun these before committing:\n```bash\ncargo test\n```\n";
        let segs = segment(md);
        assert!(
            segs.iter()
                .any(|(_, t)| t.contains("before committing") && t.contains("code: cargo test")),
            "adjacent caption must fuse with its code block: {segs:?}"
        );
        // Blank line between: the bullet is not a caption, the block stands
        // alone.
        let md = "# Ops\n\n- Use thiserror for error types\n\n```bash\ncargo test\n```\n";
        let segs = segment(md);
        assert!(
            segs.iter().any(|(_, t)| t == "code: cargo test"),
            "non adjacent block stays bare: {segs:?}"
        );
        assert!(
            segs.iter().any(|(_, t)| t.contains("thiserror")),
            "the bullet survives on its own: {segs:?}"
        );
    }

    #[test]
    fn classify_never_assigns_triggers() {
        // Triggers are assigned by embedding similarity to anchors or named
        // by the LLM parser, never by string matching in classification:
        // phrase matching false fires on prose about commitment and misses
        // every rephrase.
        for (section, text) in [
            (
                "Conventions",
                "Commit messages: conventional style (feat:, fix:), single line, no emojis",
            ),
            (
                "Discipline",
                "Never waver on the commitment to the plan even when pushed to the limit",
            ),
        ] {
            let (_, tags) = classify(section, text);
            assert!(
                !tags.iter().any(|t| t.starts_with("trigger:")),
                "classification must not string match triggers: {tags:?}"
            );
        }
    }

    #[test]
    fn never_marks_anything_always() {
        let md = "# Rules\n\n- NEVER force push to main under any circumstances\n- Always run the full test suite before committing changes\n";
        let atoms = plan_agent_file(md, &AgentFileIngestOptions::default());
        assert!(!atoms.is_empty());
        for atom in &atoms {
            assert!(
                !atom.tags.iter().any(|t| t == "scope:always"),
                "agent file ingest must never pin: {:?}",
                atom.tags
            );
        }
    }

    #[test]
    fn classifies_types_sanely() {
        let (t, _) = classify("Commands", "code: cargo build --release");
        assert_eq!(t, MemoryType::Procedural);
        let (t, _) = classify("Rules", "Never store secrets in the repository");
        assert_eq!(t, MemoryType::AntiPattern);
        let (t, _) = classify("Overview", "The engine uses a CSR graph for relationships");
        assert_eq!(t, MemoryType::Semantic);
    }

    #[test]
    fn section_slugs_are_clean() {
        assert_eq!(section_slug("Project > Conventions"), "project-conventions");
        assert_eq!(section_slug(""), "");
    }
}
