//! Backend for the client side memory tool (`memory_20250818`).
//!
//! The model issues file style commands (`view`, `create`, `str_replace`,
//! `insert`, `delete`, `rename`) against a `/memories` directory. The
//! developer's harness forwards each command to a backend and copies the
//! returned text into the `tool_result` block. This module is that backend,
//! on top of a [`MenteDb`]: every memory tool file is one memory whose
//! content is the file text, addressed by a `memtool:path:<path>` tag and
//! marked `source:memory_tool`, so the files live beside (and are searchable
//! with) the rest of the owner's memories.
//!
//! Files belong to a [`MemoryToolScope`] (an agent and a user). A command
//! only ever sees and changes the files of its own scope, so one agent can
//! never list, read, overwrite, rename onto or delete another agent's files,
//! even at the same path.
//!
//! Command level failures (missing file, bad range, duplicate `old_str`)
//! come back as a [`MemoryToolResponse`] with `is_error` set and the error
//! text the memory tool spec defines, so the model can read the message and
//! correct itself. Only engine faults are returned as `Err`.
//!
//! The parsing and formatting helpers are pure functions, so a server can
//! validate a request before it opens a database, and other hosts can reuse
//! them unchanged.

use std::collections::{BTreeMap, HashSet};

use mentedb_core::MemoryNode;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId, UserId};
use serde::{Deserialize, Serialize};

use crate::MenteDb;
use crate::hosting::MemoryPatch;

/// The root every memory tool path must live under.
pub const MEMORY_ROOT: &str = "/memories";
/// Line count ceiling for `view`, matching the reference implementation.
pub const MAX_VIEW_LINES: usize = 999_999;
/// Tag prefix that binds a memory to its memory tool path.
pub const PATH_TAG_PREFIX: &str = "memtool:path:";
/// Marker tag on every memory created through the memory tool.
pub const SOURCE_TAG: &str = "source:memory_tool";

/// A memory tool call: the fields the model sends in the `tool_use` input,
/// discriminated by `command`. Unknown fields are ignored, so a harness can
/// forward the tool input as is.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryToolRequest {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_range: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_str: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_str: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub insert_line: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub insert_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_path: Option<String>,
}

/// The result of one command: `content` is the text for the `tool_result`
/// block and `is_error` maps onto the block's `is_error` flag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryToolResponse {
    pub content: String,
    pub is_error: bool,
}

impl MemoryToolResponse {
    /// A successful result.
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// A command level error the model can read and act on.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// A parsed and path validated memory tool command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryCommand {
    View {
        path: String,
        view_range: Option<(i64, i64)>,
    },
    Create {
        path: String,
        file_text: String,
    },
    StrReplace {
        path: String,
        old_str: String,
        new_str: String,
    },
    Insert {
        path: String,
        insert_line: i64,
        insert_text: String,
    },
    Delete {
        path: String,
    },
    Rename {
        old_path: String,
        new_path: String,
    },
}

impl MemoryCommand {
    /// True for commands that change stored state. `view` is the only read.
    pub fn is_mutating(&self) -> bool {
        !matches!(self, MemoryCommand::View { .. })
    }
}

/// The owner of a set of memory tool files. Two scopes that differ in agent
/// or user never see each other's files. The nil agent and nil user form a
/// scope of their own, used when a caller names neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryToolScope {
    pub agent_id: AgentId,
    pub user_id: UserId,
}

impl MemoryToolScope {
    /// The scope of one agent and one user.
    pub fn new(agent_id: AgentId, user_id: UserId) -> Self {
        Self { agent_id, user_id }
    }

    /// The scope owned by no agent and no user.
    pub fn unowned() -> Self {
        Self::new(AgentId::nil(), UserId::nil())
    }

    /// Whether `node` belongs to this scope.
    pub fn owns(&self, node: &MemoryNode) -> bool {
        node.agent_id == self.agent_id && node.user_id == self.user_id
    }
}

/// What [`execute`] did: the response for the model, and how the number of
/// stored memories changed (positive for files created, negative for files
/// deleted), for hosts that meter memory counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryToolOutcome {
    pub response: MemoryToolResponse,
    pub memory_delta: i64,
}

impl MemoryToolOutcome {
    fn unchanged(response: MemoryToolResponse) -> Self {
        Self {
            response,
            memory_delta: 0,
        }
    }
}

/// Validate a memory tool path and return it in normalized form (no
/// trailing slash). Every path must be absolute, live under `/memories`, and
/// be free of traversal: no `.` or `..` segments, no empty segments, no
/// backslashes, no percent encoded dots, slashes, backslashes or percent
/// signs (so double encoding is refused too), and no control characters.
/// Paths map onto tags, never onto a filesystem, so nothing can escape the
/// database either way; this is spec conformance plus defense in depth.
pub fn validate_path(path: &str) -> Result<String, String> {
    let trimmed = path.trim();
    let normalized = if trimmed.len() > 1 {
        trimmed.trim_end_matches('/')
    } else {
        trimmed
    };
    let invalid = || format!("Error: Invalid path `{path}`. Paths must live under {MEMORY_ROOT}.");
    if normalized != MEMORY_ROOT && !normalized.starts_with("/memories/") {
        return Err(invalid());
    }
    let lower = normalized.to_ascii_lowercase();
    if normalized.contains('\\')
        || normalized.chars().any(char::is_control)
        || ["%2e", "%2f", "%5c", "%25"]
            .iter()
            .any(|encoded| lower.contains(encoded))
    {
        return Err(invalid());
    }
    for segment in normalized.trim_start_matches('/').split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(invalid());
        }
    }
    Ok(normalized.to_string())
}

/// Parse a raw request into a typed command, validating paths and required
/// fields. Errors are memory tool error texts, not transport errors.
pub fn parse_command(req: &MemoryToolRequest) -> Result<MemoryCommand, String> {
    let path = |field: &str, value: &Option<String>| -> Result<String, String> {
        let raw = value.as_deref().ok_or_else(|| {
            format!(
                "Error: `{field}` is required for the {} command",
                req.command
            )
        })?;
        validate_path(raw)
    };
    match req.command.as_str() {
        "view" => {
            let view_range = match &req.view_range {
                None => None,
                Some(r) if r.len() == 2 => Some((r[0], r[1])),
                Some(r) => {
                    return Err(format!(
                        "Error: Invalid `view_range` parameter: {r:?}. It should be [start_line, end_line]"
                    ));
                }
            };
            Ok(MemoryCommand::View {
                path: path("path", &req.path)?,
                view_range,
            })
        }
        "create" => Ok(MemoryCommand::Create {
            path: path("path", &req.path)?,
            file_text: req.file_text.clone().ok_or_else(|| {
                "Error: `file_text` is required for the create command".to_string()
            })?,
        }),
        "str_replace" => Ok(MemoryCommand::StrReplace {
            path: path("path", &req.path)?,
            old_str: req.old_str.clone().ok_or_else(|| {
                "Error: `old_str` is required for the str_replace command".to_string()
            })?,
            // new_str is optional: leaving it out deletes old_str.
            new_str: req.new_str.clone().unwrap_or_default(),
        }),
        "insert" => Ok(MemoryCommand::Insert {
            path: path("path", &req.path)?,
            insert_line: req.insert_line.ok_or_else(|| {
                "Error: `insert_line` is required for the insert command".to_string()
            })?,
            insert_text: req.insert_text.clone().ok_or_else(|| {
                "Error: `insert_text` is required for the insert command".to_string()
            })?,
        }),
        "delete" => Ok(MemoryCommand::Delete {
            path: path("path", &req.path)?,
        }),
        "rename" => Ok(MemoryCommand::Rename {
            old_path: path("old_path", &req.old_path)?,
            new_path: path("new_path", &req.new_path)?,
        }),
        other => Err(format!("Error: unknown command {other}")),
    }
}

/// The tag that binds a memory to its memory tool path.
pub fn path_tag(path: &str) -> String {
    format!("{PATH_TAG_PREFIX}{path}")
}

/// The memory tool path a memory is filed under, if any.
pub fn path_from_tags(tags: &[String]) -> Option<&str> {
    tags.iter()
        .find_map(|t| t.strip_prefix(PATH_TAG_PREFIX))
        .filter(|p| !p.is_empty())
}

/// True when `file_path` lives under directory `dir` (any depth).
pub fn is_under_dir(dir: &str, file_path: &str) -> bool {
    file_path
        .strip_prefix(dir)
        .is_some_and(|rest| rest.starts_with('/'))
}

/// `du` style human readable size.
pub fn human_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}K", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}M", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Directory listing per the memory tool spec: the directory itself, then
/// files and implied subdirectories up to 2 levels deep, sizes first, sorted
/// by path. Hidden segments (starting with `.`) are left out.
pub fn format_dir_listing(dir: &str, files: &[(String, usize)]) -> String {
    let mut entries: BTreeMap<String, usize> = BTreeMap::new();
    let mut total = 0usize;
    for (file_path, size) in files {
        let Some(rest) = file_path.strip_prefix(dir) else {
            continue;
        };
        let Some(rel) = rest.strip_prefix('/') else {
            continue;
        };
        total += size;
        let segments: Vec<&str> = rel.split('/').collect();
        if segments.iter().any(|s| s.starts_with('.')) {
            continue;
        }
        // The file itself, when within 2 levels of the directory.
        if segments.len() <= 2 {
            *entries.entry(file_path.clone()).or_insert(0) += size;
        }
        // Implied parent directories within 2 levels accumulate the size.
        for depth in 1..segments.len().min(3) {
            let sub = format!("{dir}/{}", segments[..depth].join("/"));
            if sub != *file_path {
                *entries.entry(sub).or_insert(0) += size;
            }
        }
    }
    let mut out = format!(
        "Here're the files and directories up to 2 levels deep in {dir}, excluding hidden items and node_modules:\n{}\t{dir}",
        human_size(total)
    );
    for (entry_path, size) in entries {
        out.push_str(&format!("\n{}\t{entry_path}", human_size(size)));
    }
    out
}

/// File view per the memory tool spec: a header plus 1 indexed lines,
/// numbers right aligned to width 6, tab separated. `view_range` is
/// `[start, end]` with `end == -1` meaning through the end of the file.
pub fn format_file_view(
    path: &str,
    content: &str,
    view_range: Option<(i64, i64)>,
) -> Result<String, String> {
    let body = content.strip_suffix('\n').unwrap_or(content);
    let lines: Vec<&str> = body.split('\n').collect();
    if lines.len() > MAX_VIEW_LINES {
        return Err(format!(
            "File {path} exceeds maximum line limit of 999,999 lines."
        ));
    }
    let (start, end) = match view_range {
        None => (1usize, lines.len()),
        Some((s, e)) => {
            let n = lines.len() as i64;
            if s < 1 || s > n || (e != -1 && e < s) {
                return Err(format!(
                    "Error: Invalid `view_range` parameter: [{s}, {e}]. It should be within the range of lines of the file: [1, {n}]"
                ));
            }
            let end = if e == -1 {
                lines.len()
            } else {
                (e as usize).min(lines.len())
            };
            (s as usize, end)
        }
    };
    let mut out = format!("Here's the content of {path} with line numbers:");
    for (idx, line) in lines
        .iter()
        .enumerate()
        .skip(start - 1)
        .take(end.saturating_sub(start - 1))
    {
        out.push_str(&format!("\n{:6}\t{line}", idx + 1));
    }
    Ok(out)
}

/// `str_replace` on file content. Exactly one occurrence must match; zero or
/// several return the spec's error texts.
pub fn apply_str_replace(
    path: &str,
    content: &str,
    old_str: &str,
    new_str: &str,
) -> Result<String, String> {
    if old_str.is_empty() {
        return Err(format!(
            "No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
        ));
    }
    let occurrences: Vec<usize> = content.match_indices(old_str).map(|(i, _)| i).collect();
    match occurrences.len() {
        0 => Err(format!(
            "No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
        )),
        1 => Ok(content.replacen(old_str, new_str, 1)),
        _ => {
            let line_numbers: Vec<String> = occurrences
                .iter()
                .map(|&offset| (content[..offset].matches('\n').count() + 1).to_string())
                .collect();
            Err(format!(
                "No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines: {}. Please ensure it is unique",
                line_numbers.join(", ")
            ))
        }
    }
}

/// `insert` on file content: `insert_text` goes after line `insert_line`,
/// with 0 meaning the start of the file.
pub fn apply_insert(content: &str, insert_line: i64, insert_text: &str) -> Result<String, String> {
    let mut lines: Vec<&str> = content.split('\n').collect();
    let n = lines.len() as i64;
    if insert_line < 0 || insert_line > n {
        return Err(format!(
            "Error: Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: [0, {n}]"
        ));
    }
    let inserted = insert_text.strip_suffix('\n').unwrap_or(insert_text);
    lines.insert(insert_line as usize, inserted);
    Ok(lines.join("\n"))
}

/// One memory tool file: its path, backing memory id, and the memory.
#[derive(Debug, Clone)]
pub struct MemoryFile {
    pub path: String,
    pub id: MemoryId,
    pub node: MemoryNode,
}

/// Load `ids` and keep the memory tool files `scope` owns, oldest first (ties
/// broken by id), so every lookup resolves a path the same way.
fn owned_files(db: &MenteDb, scope: &MemoryToolScope, ids: Vec<MemoryId>) -> Vec<MemoryFile> {
    let mut out: Vec<MemoryFile> = ids
        .into_iter()
        .filter_map(|id| db.get_memory(id).ok().map(|node| (id, node)))
        .filter(|(_, node)| scope.owns(node) && node.tags.iter().any(|t| t == SOURCE_TAG))
        .filter_map(|(id, node)| {
            let path = path_from_tags(&node.tags)?.to_string();
            Some(MemoryFile { path, id, node })
        })
        .collect();
    out.sort_by(|a, b| {
        (a.node.created_at, a.id.to_string()).cmp(&(b.node.created_at, b.id.to_string()))
    });
    out
}

/// Every memory tool file `scope` owns. Found through the tag index, so the
/// cost follows the number of memory tool files, not the size of the store.
pub fn list_files(db: &MenteDb, scope: &MemoryToolScope) -> Vec<MemoryFile> {
    owned_files(db, scope, db.memory_ids_with_tag(SOURCE_TAG))
}

/// The file `scope` owns at exactly `path`, if any.
pub fn find_file(db: &MenteDb, scope: &MemoryToolScope, path: &str) -> Option<MemoryFile> {
    owned_files(db, scope, db.memory_ids_with_tag(&path_tag(path)))
        .into_iter()
        .find(|f| f.path == path)
}

/// Run one command against `db` for `scope`. Mutating commands assume the
/// caller serializes writes for the scope (two concurrent creates of a new
/// path would otherwise both store a file).
pub fn execute(
    db: &MenteDb,
    scope: &MemoryToolScope,
    command: MemoryCommand,
) -> MenteResult<MemoryToolOutcome> {
    match command {
        MemoryCommand::View { path, view_range } => Ok(MemoryToolOutcome::unchanged(view(
            db, scope, &path, view_range,
        ))),
        MemoryCommand::Create { path, file_text } => create(db, scope, &path, &file_text),
        MemoryCommand::StrReplace {
            path,
            old_str,
            new_str,
        } => str_replace(db, scope, &path, &old_str, &new_str).map(MemoryToolOutcome::unchanged),
        MemoryCommand::Insert {
            path,
            insert_line,
            insert_text,
        } => insert(db, scope, &path, insert_line, &insert_text).map(MemoryToolOutcome::unchanged),
        MemoryCommand::Delete { path } => delete(db, scope, &path),
        MemoryCommand::Rename { old_path, new_path } => {
            rename(db, scope, &old_path, &new_path).map(MemoryToolOutcome::unchanged)
        }
    }
}

/// [`parse_command`] then [`execute`]: a parse failure becomes an error
/// response, not an `Err`.
pub fn handle(
    db: &MenteDb,
    scope: &MemoryToolScope,
    request: &MemoryToolRequest,
) -> MenteResult<MemoryToolOutcome> {
    match parse_command(request) {
        Ok(command) => execute(db, scope, command),
        Err(msg) => Ok(MemoryToolOutcome::unchanged(MemoryToolResponse::error(msg))),
    }
}

fn view(
    db: &MenteDb,
    scope: &MemoryToolScope,
    path: &str,
    view_range: Option<(i64, i64)>,
) -> MemoryToolResponse {
    if let Some(file) = find_file(db, scope, path) {
        return match format_file_view(path, &file.node.content, view_range) {
            Ok(text) => MemoryToolResponse::ok(text),
            Err(msg) => MemoryToolResponse::error(msg),
        };
    }
    let files: Vec<(String, usize)> = list_files(db, scope)
        .into_iter()
        .map(|f| (f.path, f.node.content.len()))
        .collect();
    let has_children = files.iter().any(|(p, _)| is_under_dir(path, p));
    if path == MEMORY_ROOT || has_children {
        MemoryToolResponse::ok(format_dir_listing(path, &files))
    } else {
        MemoryToolResponse::error(format!(
            "The path {path} does not exist. Please provide a valid path."
        ))
    }
}

/// Store new or changed file content. An update keeps the memory's id, tags
/// and owner; the embedding is cleared so the engine embeds the new text.
/// Goes through the batch store, which skips paraphrase dedup: two files with
/// similar text at different paths are two files, and an edit must never be
/// dropped as a duplicate of another file.
fn persist(
    db: &MenteDb,
    scope: &MemoryToolScope,
    path: &str,
    existing: Option<MemoryFile>,
    content: &str,
) -> MenteResult<bool> {
    let created = existing.is_none();
    let node = match existing {
        Some(file) => {
            let mut node = file.node;
            node.content = content.to_string();
            node.embedding = Vec::new();
            node
        }
        None => {
            let mut node = MemoryNode::new(
                scope.agent_id,
                MemoryType::Semantic,
                content.to_string(),
                Vec::new(),
            );
            node.user_id = scope.user_id;
            node.tags = vec![path_tag(path), SOURCE_TAG.to_string()];
            // Where the file was first created; a later rename moves the
            // path tag, while provenance keeps the original path.
            node.provenance = Some(
                crate::Provenance::new(crate::SourceKind::File, scope.agent_id)
                    .with_source_ref(path),
            );
            node
        }
    };
    db.store_batch(vec![node])?;
    Ok(created)
}

fn create(
    db: &MenteDb,
    scope: &MemoryToolScope,
    path: &str,
    file_text: &str,
) -> MenteResult<MemoryToolOutcome> {
    if path == MEMORY_ROOT {
        return Ok(MemoryToolOutcome::unchanged(MemoryToolResponse::error(
            format!("Error: Cannot create a file at the {MEMORY_ROOT} directory itself"),
        )));
    }
    let existing = find_file(db, scope, path);
    let created = persist(db, scope, path, existing, file_text)?;
    Ok(MemoryToolOutcome {
        response: MemoryToolResponse::ok(format!("File created successfully at: {path}")),
        memory_delta: i64::from(created),
    })
}

fn str_replace(
    db: &MenteDb,
    scope: &MemoryToolScope,
    path: &str,
    old_str: &str,
    new_str: &str,
) -> MenteResult<MemoryToolResponse> {
    let Some(file) = find_file(db, scope, path) else {
        return Ok(MemoryToolResponse::error(format!(
            "Error: The path {path} does not exist. Please provide a valid path."
        )));
    };
    match apply_str_replace(path, &file.node.content, old_str, new_str) {
        Ok(updated) => {
            persist(db, scope, path, Some(file), &updated)?;
            Ok(MemoryToolResponse::ok("The memory file has been edited."))
        }
        Err(msg) => Ok(MemoryToolResponse::error(msg)),
    }
}

fn insert(
    db: &MenteDb,
    scope: &MemoryToolScope,
    path: &str,
    insert_line: i64,
    insert_text: &str,
) -> MenteResult<MemoryToolResponse> {
    let Some(file) = find_file(db, scope, path) else {
        return Ok(MemoryToolResponse::error(format!(
            "Error: The path {path} does not exist"
        )));
    };
    match apply_insert(&file.node.content, insert_line, insert_text) {
        Ok(updated) => {
            persist(db, scope, path, Some(file), &updated)?;
            Ok(MemoryToolResponse::ok(format!(
                "The file {path} has been edited."
            )))
        }
        Err(msg) => Ok(MemoryToolResponse::error(msg)),
    }
}

fn delete(db: &MenteDb, scope: &MemoryToolScope, path: &str) -> MenteResult<MemoryToolOutcome> {
    if path == MEMORY_ROOT {
        return Ok(MemoryToolOutcome::unchanged(MemoryToolResponse::error(
            format!("Error: Cannot delete the {MEMORY_ROOT} directory itself"),
        )));
    }
    let targets: Vec<MemoryFile> = list_files(db, scope)
        .into_iter()
        .filter(|f| f.path == path || is_under_dir(path, &f.path))
        .collect();
    if targets.is_empty() {
        return Ok(MemoryToolOutcome::unchanged(MemoryToolResponse::error(
            format!("Error: The path {path} does not exist"),
        )));
    }
    let mut deleted = 0i64;
    for file in targets {
        db.forget(file.id)?;
        deleted += 1;
    }
    Ok(MemoryToolOutcome {
        response: MemoryToolResponse::ok(format!("Successfully deleted {path}")),
        memory_delta: -deleted,
    })
}

fn rename(
    db: &MenteDb,
    scope: &MemoryToolScope,
    old_path: &str,
    new_path: &str,
) -> MenteResult<MemoryToolResponse> {
    if old_path == MEMORY_ROOT {
        return Ok(MemoryToolResponse::error(format!(
            "Error: Cannot rename the {MEMORY_ROOT} directory itself"
        )));
    }
    let files = list_files(db, scope);
    // (source file, destination path) pairs: an exact file rename, or every
    // file under a directory rename.
    let moves: Vec<(&MemoryFile, String)> = files
        .iter()
        .filter_map(|f| {
            if f.path == old_path {
                Some((f, new_path.to_string()))
            } else {
                f.path
                    .strip_prefix(old_path)
                    .filter(|rest| rest.starts_with('/'))
                    .map(|rest| (f, format!("{new_path}{rest}")))
            }
        })
        .collect();
    if moves.is_empty() {
        return Ok(MemoryToolResponse::error(format!(
            "Error: The path {old_path} does not exist"
        )));
    }
    let sources: HashSet<&str> = moves.iter().map(|(f, _)| f.path.as_str()).collect();
    // A destination is taken when a file that is not moving sits at it, or
    // when it is a directory holding such files.
    let collision = moves.iter().any(|(_, dest)| {
        files.iter().any(|f| {
            !sources.contains(f.path.as_str()) && (f.path == *dest || is_under_dir(dest, &f.path))
        })
    });
    if collision {
        return Ok(MemoryToolResponse::error(format!(
            "Error: The destination {new_path} already exists"
        )));
    }
    for (file, dest) in moves {
        let tags: Vec<String> = file
            .node
            .tags
            .iter()
            .map(|tag| {
                if tag.starts_with(PATH_TAG_PREFIX) {
                    path_tag(&dest)
                } else {
                    tag.clone()
                }
            })
            .collect();
        // A tag only change: the content and vector stay, so no re-embed.
        db.update_memory_fields(
            file.id,
            &MemoryPatch {
                tags: Some(tags),
                ..MemoryPatch::default()
            },
        )?;
    }
    Ok(MemoryToolResponse::ok(format!(
        "Successfully renamed {old_path} to {new_path}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(command: &str) -> MemoryToolRequest {
        MemoryToolRequest {
            command: command.to_string(),
            ..MemoryToolRequest::default()
        }
    }

    #[test]
    fn validate_path_accepts_root_and_children() {
        assert_eq!(validate_path("/memories").unwrap(), "/memories");
        assert_eq!(validate_path("/memories/").unwrap(), "/memories");
        assert_eq!(
            validate_path("/memories/notes.txt").unwrap(),
            "/memories/notes.txt"
        );
        assert_eq!(
            validate_path("/memories/a/b/c.md").unwrap(),
            "/memories/a/b/c.md"
        );
        assert_eq!(
            validate_path("/memories/100 percent.md").unwrap(),
            "/memories/100 percent.md"
        );
    }

    #[test]
    fn validate_path_rejects_escape_attempts() {
        for bad in [
            "/etc/passwd",
            "/memoriesx/notes.txt",
            "/memories/../secrets.env",
            "/memories/a/../../x",
            "/memories/..",
            "/memories/./x",
            "/memories//x",
            "//memories/x",
            "/memories/%2e%2e%2fx",
            "/memories/%2E%2E/x",
            "/memories/a%2fb",
            "/memories/%252e%252e/x",
            "/memories/..%5cx",
            "/memories/..\\x",
            "/memories/a\0b",
            "/memories/a\nb",
            "memories/x",
            "../memories/x",
            "",
            "/",
        ] {
            assert!(validate_path(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn parse_command_view() {
        let mut r = req("view");
        r.path = Some("/memories".to_string());
        assert_eq!(
            parse_command(&r).unwrap(),
            MemoryCommand::View {
                path: "/memories".to_string(),
                view_range: None
            }
        );
        r.path = Some("/memories/notes.txt".to_string());
        r.view_range = Some(vec![1, -1]);
        assert_eq!(
            parse_command(&r).unwrap(),
            MemoryCommand::View {
                path: "/memories/notes.txt".to_string(),
                view_range: Some((1, -1))
            }
        );
        r.view_range = Some(vec![1, 2, 3]);
        assert!(parse_command(&r).is_err());
    }

    #[test]
    fn parse_command_requires_fields() {
        let mut r = req("create");
        r.path = Some("/memories/notes.txt".to_string());
        assert!(parse_command(&r).unwrap_err().contains("file_text"));
        r.file_text = Some("hello".to_string());
        assert_eq!(
            parse_command(&r).unwrap(),
            MemoryCommand::Create {
                path: "/memories/notes.txt".to_string(),
                file_text: "hello".to_string()
            }
        );
        assert!(parse_command(&req("view")).unwrap_err().contains("`path`"));
        let mut r = req("rename");
        r.old_path = Some("/memories/a".to_string());
        assert!(parse_command(&r).unwrap_err().contains("`new_path`"));
        r.new_path = Some("/memories/../b".to_string());
        assert!(parse_command(&r).unwrap_err().contains("Invalid path"));
    }

    #[test]
    fn parse_command_str_replace_new_str_optional() {
        let mut r = req("str_replace");
        r.path = Some("/memories/p.txt".to_string());
        r.old_str = Some("blue".to_string());
        assert_eq!(
            parse_command(&r).unwrap(),
            MemoryCommand::StrReplace {
                path: "/memories/p.txt".to_string(),
                old_str: "blue".to_string(),
                new_str: String::new()
            }
        );
    }

    #[test]
    fn parse_command_insert_delete_rename() {
        let mut r = req("insert");
        r.path = Some("/memories/todo.txt".to_string());
        r.insert_line = Some(2);
        r.insert_text = Some("- item\n".to_string());
        assert_eq!(
            parse_command(&r).unwrap(),
            MemoryCommand::Insert {
                path: "/memories/todo.txt".to_string(),
                insert_line: 2,
                insert_text: "- item\n".to_string()
            }
        );

        let mut r = req("delete");
        r.path = Some("/memories/old.txt".to_string());
        assert!(parse_command(&r).unwrap().is_mutating());

        let mut r = req("rename");
        r.old_path = Some("/memories/draft.txt".to_string());
        r.new_path = Some("/memories/final.txt".to_string());
        assert_eq!(
            parse_command(&r).unwrap(),
            MemoryCommand::Rename {
                old_path: "/memories/draft.txt".to_string(),
                new_path: "/memories/final.txt".to_string()
            }
        );

        assert!(
            parse_command(&req("bogus"))
                .unwrap_err()
                .contains("unknown command")
        );
    }

    #[test]
    fn view_is_the_only_read_command() {
        let mut r = req("view");
        r.path = Some("/memories".to_string());
        assert!(!parse_command(&r).unwrap().is_mutating());
    }

    #[test]
    fn request_ignores_unknown_fields() {
        let r: MemoryToolRequest = serde_json::from_value(serde_json::json!({
            "command": "view",
            "path": "/memories",
            "agent_id": "anything",
        }))
        .unwrap();
        assert_eq!(r.command, "view");
    }

    #[test]
    fn path_tag_roundtrip() {
        let tag = path_tag("/memories/notes.txt");
        assert_eq!(tag, "memtool:path:/memories/notes.txt");
        let tags = vec!["source:memory_tool".to_string(), tag];
        assert_eq!(path_from_tags(&tags), Some("/memories/notes.txt"));
        assert_eq!(path_from_tags(&["scope:always".to_string()]), None);
    }

    #[test]
    fn is_under_dir_matches_children_only() {
        assert!(is_under_dir("/memories", "/memories/a.txt"));
        assert!(is_under_dir("/memories/a", "/memories/a/b.txt"));
        assert!(!is_under_dir("/memories/a", "/memories/ab.txt"));
        assert!(!is_under_dir("/memories/a", "/memories/a"));
    }

    #[test]
    fn format_file_view_numbers_lines() {
        let out = format_file_view("/memories/n.txt", "Hello World\nline two\n", None).unwrap();
        assert_eq!(
            out,
            "Here's the content of /memories/n.txt with line numbers:\n     1\tHello World\n     2\tline two"
        );
    }

    #[test]
    fn format_file_view_ranges() {
        let content = "a\nb\nc\nd\n";
        let out = format_file_view("/memories/n.txt", content, Some((2, 3))).unwrap();
        assert_eq!(
            out,
            "Here's the content of /memories/n.txt with line numbers:\n     2\tb\n     3\tc"
        );
        let out = format_file_view("/memories/n.txt", content, Some((3, -1))).unwrap();
        assert_eq!(
            out,
            "Here's the content of /memories/n.txt with line numbers:\n     3\tc\n     4\td"
        );
        assert!(format_file_view("/memories/n.txt", content, Some((0, 2))).is_err());
        assert!(format_file_view("/memories/n.txt", content, Some((5, 6))).is_err());
        assert!(format_file_view("/memories/n.txt", content, Some((3, 2))).is_err());
    }

    #[test]
    fn format_dir_listing_two_levels() {
        let files = vec![
            ("/memories/a.txt".to_string(), 1024usize),
            ("/memories/dir/b.txt".to_string(), 2048usize),
            ("/memories/dir/deep/c.txt".to_string(), 512usize),
            ("/memories/dir/deep/deeper/d.txt".to_string(), 100usize),
        ];
        let out = format_dir_listing("/memories", &files);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[0],
            "Here're the files and directories up to 2 levels deep in /memories, excluding hidden items and node_modules:"
        );
        assert!(lines[1].ends_with("\t/memories"));
        assert!(out.contains("\t/memories/a.txt"));
        assert!(out.contains("\t/memories/dir"));
        assert!(out.contains("\t/memories/dir/b.txt"));
        assert!(out.contains("\t/memories/dir/deep"));
        // Files 3 or more levels deep roll into their directories.
        assert!(!out.contains("/memories/dir/deep/c.txt"));
        assert!(!out.contains("/memories/dir/deep/deeper"));
    }

    #[test]
    fn format_dir_listing_empty_root() {
        let out = format_dir_listing("/memories", &[]);
        assert_eq!(
            out,
            "Here're the files and directories up to 2 levels deep in /memories, excluding hidden items and node_modules:\n0B\t/memories"
        );
    }

    #[test]
    fn str_replace_single_occurrence() {
        let out = apply_str_replace("/memories/p.txt", "color: blue\n", "blue", "green").unwrap();
        assert_eq!(out, "color: green\n");
    }

    #[test]
    fn str_replace_missing_and_duplicate() {
        let missing = apply_str_replace("/memories/p.txt", "abc", "zzz", "y").unwrap_err();
        assert_eq!(
            missing,
            "No replacement was performed, old_str `zzz` did not appear verbatim in /memories/p.txt."
        );
        let dup = apply_str_replace("/memories/p.txt", "x\nx\n", "x", "y").unwrap_err();
        assert_eq!(
            dup,
            "No replacement was performed. Multiple occurrences of old_str `x` in lines: 1, 2. Please ensure it is unique"
        );
    }

    #[test]
    fn insert_bounds_and_positions() {
        assert_eq!(apply_insert("a\nb", 0, "top").unwrap(), "top\na\nb");
        assert_eq!(apply_insert("a\nb", 1, "mid\n").unwrap(), "a\nmid\nb");
        assert_eq!(apply_insert("a\nb", 2, "end").unwrap(), "a\nb\nend");
        let err = apply_insert("a\nb", 3, "x").unwrap_err();
        assert_eq!(
            err,
            "Error: Invalid `insert_line` parameter: 3. It should be within the range of lines of the file: [0, 2]"
        );
        assert!(apply_insert("a\nb", -1, "x").is_err());
    }

    #[test]
    fn human_size_units() {
        assert_eq!(human_size(0), "0B");
        assert_eq!(human_size(512), "512B");
        assert_eq!(human_size(1536), "1.5K");
        assert_eq!(human_size(2 * 1024 * 1024), "2.0M");
    }
}
