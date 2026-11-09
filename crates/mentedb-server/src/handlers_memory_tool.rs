//! `POST /v1/memory_tool`: the backend for the client side memory tool
//! (`memory_20250818`), with the same request and response contract as the
//! managed service, so a harness works against either unchanged.
//!
//! The body is the tool input the model sent (`command`, `path`,
//! `file_text`, ...), optionally with `agent_id` and `user_id` to pick whose
//! files the command acts on. A token bound to an agent always acts as that
//! agent. The reply is `{"content": "...", "is_error": bool}`, ready to copy
//! into the `tool_result` block. Command failures the model can fix (missing
//! file, bad range, duplicate `old_str`, invalid path) are HTTP 200 with
//! `is_error: true`; auth failures and engine faults are HTTP errors.
//!
//! The command logic lives in [`mentedb::memory_tool`].

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use axum::{Extension, Json};
use mentedb::memory_tool::{self, MemoryToolRequest, MemoryToolResponse, MemoryToolScope};
use mentedb_core::types::UserId;
use serde_json::Value;
use tracing::error;

use crate::auth::AuthenticatedAgent;
use crate::databases::Db;
use crate::error::ApiError;
use crate::handlers::parse_optional_uuid;
use crate::handlers_memory::scope_agent;

type Caller = Option<Extension<AuthenticatedAgent>>;

/// One write lock per database: a mutating command reads the scope's files,
/// then writes, and two of them interleaving (two creates of one new path)
/// would store the file twice. Reads take no lock.
static WRITE_LOCKS: LazyLock<std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(Default::default);

fn write_lock(db: &str) -> Arc<tokio::sync::Mutex<()>> {
    let mut locks = WRITE_LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    locks.entry(db.to_string()).or_default().clone()
}

/// The files a request acts on: the requested agent (which must match a
/// token bound to an agent), else the token's agent, else no agent; and the
/// requested user, else no user.
fn resolve_scope(agent: &Caller, body: &Value) -> Result<MemoryToolScope, ApiError> {
    let agent_id = scope_agent(agent, parse_optional_uuid(body, "agent_id")?)?
        .unwrap_or_else(mentedb_core::types::AgentId::nil);
    let user_id = parse_optional_uuid(body, "user_id")?
        .map(UserId)
        .unwrap_or_else(UserId::nil);
    Ok(MemoryToolScope::new(agent_id, user_id))
}

/// Execute one memory tool command.
pub async fn memory_tool(
    agent: Caller,
    db: Db,
    Json(body): Json<Value>,
) -> Result<Json<MemoryToolResponse>, ApiError> {
    let scope = resolve_scope(&agent, &body)?;
    let request: MemoryToolRequest = serde_json::from_value(body)
        .map_err(|e| ApiError::BadRequest(format!("invalid memory tool request: {e}")))?;
    let command = match memory_tool::parse_command(&request) {
        Ok(c) => c,
        Err(msg) => return Ok(Json(MemoryToolResponse::error(msg))),
    };

    // The route is admitted for read only tokens so they can `view`; every
    // other command needs write access to the database.
    let _guard = if command.is_mutating() {
        if let Some(Extension(authed)) = &agent {
            authed.authorize_db(&db.name, true)?;
        }
        Some(write_lock(&db.name).lock_owned().await)
    } else {
        None
    };

    let handle = db.handle.clone();
    let outcome =
        tokio::task::spawn_blocking(move || memory_tool::execute(&handle, &scope, command))
            .await
            .map_err(|e| ApiError::Internal(format!("memory tool task failed: {e}")))?
            .map_err(|e| {
                error!("memory tool failed: {e}");
                ApiError::Internal(format!("memory tool failed: {e}"))
            })?;
    Ok(Json(outcome.response))
}
