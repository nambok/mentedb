//! Agent file ingestion: turn a standing instruction file (CLAUDE.md,
//! AGENTS.md, .cursorrules, a system prompt) into individual memories that
//! are retrieved on demand instead of injected whole into every prompt.
//!
//! The design is deterministic end to end, no LLM anywhere:
//! - A markdown segmenter walks headings, bullets, paragraphs, and fenced
//!   code blocks, carrying the section path so every memory stays self
//!   contained ("Conventions: no emojis in commits", never a bare fragment).
//! - Long prose is split into atomic statements at sentence and separator
//!   boundaries, because retrieval granularity dies when a whole section is
//!   one memory.
//! - A classifier assigns memory types (procedural, anti pattern, semantic)
//!   and tags: a `section:` tag for provenance and, for rules that govern a
//!   class of agent action, a `trigger:` tag so the action cued channel
//!   (`recall_for_action`) surfaces them at the moment the action runs.
//! - Nothing is marked `scope:always`. The point of ingesting an agent file
//!   is that no rule rides every turn; rules arrive by topic or by action.
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

/// Action trigger vocabulary: context specific phrases, checked as plain
/// lowercase substrings. Bare verbs like "commit" or "push" are deliberately
/// absent: they false fire on prose about commitment or pushing back, and a
/// rule firing at the wrong moment erodes trust faster than a missing rule.
const TRIGGER_PHRASES: &[(&str, &[&str])] = &[
    (
        "git-commit",
        &[
            "git commit",
            "commit message",
            "commit subject",
            "commit style",
            "commit convention",
            "conventional commit",
            "co-authored-by",
            "co-author",
            "gpgsign",
        ],
    ),
    (
        "pr-create",
        &[
            "pull request",
            "pr description",
            "pr body",
            "pr title",
            "pr descriptions",
        ],
    ),
    (
        "git-push",
        &[
            "git push",
            "force push",
            "force-push",
            "push to main",
            "push to master",
        ],
    ),
];

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
                    out.push((section_of(&path), format!("code: {code}")));
                }
                code_buf.clear();
                in_code = false;
            } else {
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
    for (trigger, phrases) in TRIGGER_PHRASES {
        if lower_contains(&lower, phrases) {
            tags.push(format!("trigger:{trigger}"));
            break;
        }
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
            ..Default::default()
        };
        let mut sections = std::collections::HashSet::new();

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
            if atom.tags.iter().any(|t| t.starts_with("trigger:")) {
                report.trigger_tagged += 1;
            }
        }
        let after = self.count_by_tag(&opts.source_tag);
        report.stored = after.saturating_sub(before);
        report.deduplicated = report.candidates.saturating_sub(report.stored);
        report.sections = sections.len();
        if report.stored > 0 {
            report.avg_memory_token_estimate = stored_tokens / report.candidates.max(1);
        }
        Ok(report)
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
    fn triggers_require_context_phrases() {
        // A commit style rule is tagged.
        let (_, tags) = classify(
            "Conventions",
            "Commit messages: conventional style (feat:, fix:), single line, no emojis",
        );
        assert!(tags.iter().any(|t| t == "trigger:git-commit"), "{tags:?}");
        // Prose about commitment or pushing through is not.
        let (_, tags) = classify(
            "Discipline",
            "Never waver on the commitment to the plan even when pushed to the limit",
        );
        assert!(
            !tags.iter().any(|t| t.starts_with("trigger:")),
            "bare verbs must not trigger: {tags:?}"
        );
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
