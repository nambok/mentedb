//! Write time provenance: where a memory came from, recorded once when it is
//! first written and never changed afterwards.
//!
//! A [`Provenance`] names the session the memory was written in, the kind of
//! source it came from (a chat turn, a file, an API call, an MCP tool, an
//! import, or another memory), an optional reference into that source (a file
//! path, a URL, a message id), the agent that wrote it, when it was written,
//! and, for a derived memory, the memories it was derived from.
//!
//! The engine stamps it on the first write of an id and keeps the stored one
//! on every later write of the same id, so no update path can change it.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::types::{AgentId, MemoryId, Timestamp};

/// The kind of source a memory came from.
///
/// Stored as a lowercase string. A kind this build does not know (written by
/// a newer release) reads back as [`SourceKind::Other`] with its name kept,
/// so the record still decodes and the name round trips unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum SourceKind {
    /// A conversation turn (`process_turn`, an ingested chat).
    Chat,
    /// A document or file (chunked document ingestion).
    File,
    /// A direct API write (REST or gRPC store).
    Api,
    /// A tool call made through the Model Context Protocol.
    Mcp,
    /// An import of previously exported memories.
    Import,
    /// Made by the engine from other memories (extraction, consolidation,
    /// entity and community nodes, the user profile).
    Derived,
    /// The writer did not say.
    Unspecified,
    /// Any other kind, by name.
    Other(String),
}

impl SourceKind {
    /// The stored name of this kind.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Chat => "chat",
            Self::File => "file",
            Self::Api => "api",
            Self::Mcp => "mcp",
            Self::Import => "import",
            Self::Derived => "derived",
            Self::Unspecified => "unspecified",
            Self::Other(s) => s,
        }
    }

    /// Parse a stored name. Case and surrounding space are ignored; an
    /// unknown name becomes [`SourceKind::Other`]. An empty name is
    /// [`SourceKind::Unspecified`].
    pub fn parse(s: &str) -> Self {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "chat" => Self::Chat,
            "file" => Self::File,
            "api" => Self::Api,
            "mcp" => Self::Mcp,
            "import" => Self::Import,
            "derived" => Self::Derived,
            "" | "unspecified" => Self::Unspecified,
            _ => Self::Other(lower),
        }
    }
}

impl From<String> for SourceKind {
    fn from(s: String) -> Self {
        Self::parse(&s)
    }
}

impl From<SourceKind> for String {
    fn from(k: SourceKind) -> Self {
        k.as_str().to_string()
    }
}

impl fmt::Display for SourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for SourceKind {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::parse(s))
    }
}

/// Where a memory came from, recorded when it is first written.
///
/// Optional fields are left out of the stored record when unset, so a record
/// carries only what its writer knew.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// The session the memory was written in, if the writer named one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The kind of source it came from.
    pub source: SourceKind,
    /// A reference into the source: a file path, a URL, a message id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_ref: Option<String>,
    /// The agent that wrote it. Usually the owning agent; differs when one
    /// agent writes on behalf of another.
    pub written_by: AgentId,
    /// When it was first written, microseconds since the Unix epoch. Set by
    /// the writer's clock, independent of `created_at`, which a caller may
    /// backdate (an import keeps the original creation time).
    pub written_at: Timestamp,
    /// For a derived memory, the memories it was made from.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derived_from: Vec<MemoryId>,
}

fn now_micros() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

impl Provenance {
    /// A provenance for a memory written now by `written_by` from `source`.
    pub fn new(source: SourceKind, written_by: AgentId) -> Self {
        Self {
            session_id: None,
            source,
            source_ref: None,
            written_by,
            written_at: now_micros(),
            derived_from: Vec::new(),
        }
    }

    /// Set the session (builder style). An empty id leaves it unset.
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        let s = session_id.into();
        self.session_id = if s.is_empty() { None } else { Some(s) };
        self
    }

    /// Set the source reference (builder style). An empty reference leaves
    /// it unset.
    pub fn with_source_ref(mut self, source_ref: impl Into<String>) -> Self {
        let s = source_ref.into();
        self.source_ref = if s.is_empty() { None } else { Some(s) };
        self
    }

    /// A provenance for a memory derived now by `written_by` from `origins`:
    /// source [`SourceKind::Derived`], the origin ids, and the session the
    /// origins share. `origin_sessions` holds each origin's session (None for
    /// an origin with no session or no provenance); the session is inherited
    /// only when every origin has the same one.
    pub fn derived<'a>(
        written_by: AgentId,
        origins: &[MemoryId],
        origin_sessions: impl IntoIterator<Item = Option<&'a str>>,
    ) -> Self {
        let mut p = Self::new(SourceKind::Derived, written_by);
        let mut ids: Vec<MemoryId> = Vec::with_capacity(origins.len());
        for id in origins {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        p.derived_from = ids;
        p.session_id = shared_session(origin_sessions);
        p
    }

    /// True for a memory the engine derived from other memories.
    pub fn is_derived(&self) -> bool {
        self.source == SourceKind::Derived
    }
}

/// The session every item names, or None when any item has none, two items
/// differ, or there are no items.
fn shared_session<'a>(sessions: impl IntoIterator<Item = Option<&'a str>>) -> Option<String> {
    let mut shared: Option<&str> = None;
    let mut any = false;
    for s in sessions {
        any = true;
        let s = s?;
        match shared {
            None => shared = Some(s),
            Some(prev) if prev == s => {}
            Some(_) => return None,
        }
    }
    if any {
        shared.map(str::to_string)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_kind_round_trips_known_and_unknown_names() {
        for k in [
            SourceKind::Chat,
            SourceKind::File,
            SourceKind::Api,
            SourceKind::Mcp,
            SourceKind::Import,
            SourceKind::Derived,
            SourceKind::Unspecified,
        ] {
            let json = serde_json::to_string(&k).unwrap();
            assert_eq!(json, format!("\"{}\"", k.as_str()));
            assert_eq!(serde_json::from_str::<SourceKind>(&json).unwrap(), k);
        }
        // A kind from a newer release still decodes, and keeps its name.
        let k: SourceKind = serde_json::from_str("\"webhook\"").unwrap();
        assert_eq!(k, SourceKind::Other("webhook".into()));
        assert_eq!(serde_json::to_string(&k).unwrap(), "\"webhook\"");
        assert_eq!(SourceKind::parse(" Chat "), SourceKind::Chat);
        assert_eq!(SourceKind::parse(""), SourceKind::Unspecified);
    }

    #[test]
    fn unset_fields_are_left_out_of_the_record() {
        let p = Provenance::new(SourceKind::Api, AgentId::nil());
        let v = serde_json::to_value(&p).unwrap();
        let obj = v.as_object().unwrap();
        assert!(!obj.contains_key("session_id"));
        assert!(!obj.contains_key("source_ref"));
        assert!(!obj.contains_key("derived_from"));
        let back: Provenance = serde_json::from_value(v).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn derived_inherits_only_a_session_every_origin_shares() {
        let a = MemoryId::new();
        let b = MemoryId::new();
        let p = Provenance::derived(AgentId::nil(), &[a, b, a], [Some("s1"), Some("s1")]);
        assert_eq!(p.source, SourceKind::Derived);
        assert_eq!(p.derived_from, vec![a, b], "origins are deduplicated");
        assert_eq!(p.session_id.as_deref(), Some("s1"));

        let mixed = Provenance::derived(AgentId::nil(), &[a, b], [Some("s1"), Some("s2")]);
        assert_eq!(mixed.session_id, None);
        let partial = Provenance::derived(AgentId::nil(), &[a, b], [Some("s1"), None]);
        assert_eq!(partial.session_id, None);
        let none = Provenance::derived(AgentId::nil(), &[], std::iter::empty());
        assert_eq!(none.session_id, None);
    }
}
