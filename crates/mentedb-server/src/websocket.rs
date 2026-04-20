//! WebSocket handler for live cognition streaming.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn};

use crate::state::AppState;

/// Inbound message from a WebSocket client.
#[derive(Debug, Deserialize)]
struct WsInbound {
    #[serde(rename = "type")]
    msg_type: String,
    data: serde_json::Value,
}

/// Outbound message pushed to a WebSocket client.
#[derive(Debug, Serialize)]
struct WsOutbound {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    alert_type: Option<String>,
    data: serde_json::Value,
}

/// Axum handler that upgrades an HTTP request to a WebSocket connection.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    info!("websocket client connected");

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!("websocket receive error: {e}");
                break;
            }
        };

        match msg {
            Message::Text(text) => {
                let inbound: WsInbound = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        let err = json!({ "type": "error", "data": format!("invalid JSON: {e}") });
                        if socket
                            .send(Message::Text(err.to_string().into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                        continue;
                    }
                };

                let response = process_message(&inbound, &state).await;
                let payload = serde_json::to_string(&response).unwrap_or_default();
                if socket.send(Message::Text(payload.into())).await.is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    info!("websocket client disconnected");
}

async fn process_message(inbound: &WsInbound, state: &AppState) -> WsOutbound {
    match inbound.msg_type.as_str() {
        "query" => {
            let query = inbound.data.as_str().unwrap_or("");
            let db = &*state.db;
            match db.recall(query) {
                Ok(window) => {
                    let memory_count: usize = window.blocks.iter().map(|b| b.memories.len()).sum();
                    WsOutbound {
                        msg_type: "result".into(),
                        alert_type: None,
                        data: json!({
                            "context": window.format,
                            "total_tokens": window.total_tokens,
                            "memory_count": memory_count,
                        }),
                    }
                }
                Err(e) => WsOutbound {
                    msg_type: "error".into(),
                    alert_type: None,
                    data: json!(format!("recall failed: {e}")),
                },
            }
        }
        "token" => {
            // Echo acknowledgment for token-type messages.
            WsOutbound {
                msg_type: "ack".into(),
                alert_type: None,
                data: inbound.data.clone(),
            }
        }
        _ => WsOutbound {
            msg_type: "error".into(),
            alert_type: None,
            data: json!(format!("unknown message type: {}", inbound.msg_type)),
        },
    }
}
