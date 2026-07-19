//! Structured export: fill a caller-provided JSON schema from a set of memories
//! using an LLM the embedder supplies.
//!
//! This mirrors the consolidation `LlmJudge` pattern: the cognitive logic and the
//! prompt live in the engine, but the model call is a trait the embedder plugs in
//! (for example AWS Bedrock in the hosted platform), so the engine core takes on no
//! LLM dependency. Instead of dumping raw memories, the model reads them and
//! populates a structure the caller defines, so an app gets, say, a typed
//! `UserProfile` rather than a list of sentences.

use mentedb_core::MemoryNode;
use mentedb_core::error::{MenteError, MenteResult};

/// The model the embedder plugs in for export. Given a system and user prompt,
/// return the model's raw text (expected to be JSON).
pub trait ExportModel: Send + Sync {
    fn complete(&self, system: &str, user: &str) -> MenteResult<String>;
}

const EXPORT_SYSTEM: &str = "You are a data extraction engine. Read the user's memories and output a single JSON value that conforms exactly to the provided JSON schema. Use only information supported by the memories; for any field with no support, use null. Do not invent values. Output only the JSON, with no prose or code fences.";

/// Fill `schema` from `memories` via `model`. `schema` is the JSON Schema (or an
/// example shape) the caller wants populated; `instructions` optionally steers
/// conflict resolution or formatting. Returns the parsed JSON value, or an error
/// if the model does not return valid JSON.
pub fn export_structured(
    memories: &[MemoryNode],
    schema: &str,
    instructions: Option<&str>,
    model: &dyn ExportModel,
) -> MenteResult<serde_json::Value> {
    let mut user = String::with_capacity(256 + memories.len() * 64);
    user.push_str("JSON schema to fill:\n");
    user.push_str(schema.trim());
    if let Some(i) = instructions {
        user.push_str("\n\nAdditional instructions:\n");
        user.push_str(i.trim());
    }
    user.push_str("\n\nMemories:\n");
    for m in memories {
        user.push_str("- ");
        user.push_str(m.content.trim());
        user.push('\n');
    }

    let raw = model.complete(EXPORT_SYSTEM, &user)?;
    let cleaned = strip_code_fences(&raw);
    serde_json::from_str(cleaned).map_err(|e| {
        MenteError::Serialization(format!(
            "structured export: model did not return valid JSON: {e}"
        ))
    })
}

/// Strip an optional ```json ... ``` fence some models wrap JSON in.
fn strip_code_fences(s: &str) -> &str {
    let t = s.trim();
    let t = t
        .strip_prefix("```json")
        .or_else(|| t.strip_prefix("```"))
        .unwrap_or(t);
    let t = t.strip_suffix("```").unwrap_or(t);
    t.trim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;

    struct MockModel {
        reply: String,
        seen_user: std::sync::Mutex<String>,
    }
    impl ExportModel for MockModel {
        fn complete(&self, _system: &str, user: &str) -> MenteResult<String> {
            *self.seen_user.lock().unwrap() = user.to_string();
            Ok(self.reply.clone())
        }
    }

    fn mem(content: &str) -> MemoryNode {
        MemoryNode::new(
            mentedb_core::types::AgentId::nil(),
            MemoryType::Semantic,
            content.to_string(),
            vec![],
        )
    }

    #[test]
    fn fills_schema_and_includes_memories_in_prompt() {
        let model = MockModel {
            reply: r#"{"name":"Nam","city":"NYC"}"#.to_string(),
            seen_user: std::sync::Mutex::new(String::new()),
        };
        let memories = [mem("User's name is Nam"), mem("User lives in NYC")];
        let out = export_structured(
            &memories,
            r#"{"name": "string", "city": "string"}"#,
            None,
            &model,
        )
        .unwrap();
        assert_eq!(out["name"], "Nam");
        assert_eq!(out["city"], "NYC");
        let prompt = model.seen_user.lock().unwrap().clone();
        assert!(prompt.contains("User lives in NYC"));
        assert!(prompt.contains("JSON schema to fill"));
    }

    #[test]
    fn tolerates_code_fenced_json() {
        let model = MockModel {
            reply: "```json\n{\"ok\": true}\n```".to_string(),
            seen_user: std::sync::Mutex::new(String::new()),
        };
        let out = export_structured(&[mem("x")], "{}", None, &model).unwrap();
        assert_eq!(out["ok"], true);
    }

    #[test]
    fn invalid_json_is_an_error() {
        let model = MockModel {
            reply: "not json at all".to_string(),
            seen_user: std::sync::Mutex::new(String::new()),
        };
        assert!(export_structured(&[mem("x")], "{}", None, &model).is_err());
    }
}
