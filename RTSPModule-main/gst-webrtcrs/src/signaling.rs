/// HTTP Signaling Server for WebRTC negotiation.
///
/// Runs on a single configurable port and provides REST endpoints for
/// browsers to establish WebRTC connections to **any** registered stream.
///
/// Endpoints:
/// - `GET  /streams`                        — list all active stream IDs (in registration order)
/// - `POST /webrtc/offer?stream_id=X`       — send SDP offer, receive answer
/// - `POST /webrtc/disconnect`               — disconnect a specific peer by peer_id
/// - `GET  /webrtc/status?stream_id=X`      — get peer count for a stream
/// - `GET  /health`                         — liveness probe

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use crate::hub::SharedManagers;
use crate::webrtc_manager::WebRtcManager;

/// Query parameter for stream identification.
#[derive(Debug, Deserialize)]
struct StreamQuery {
    stream_id: Option<String>,
}

/// SDP offer from the browser.
#[derive(Debug, Deserialize)]
struct OfferRequest {
    sdp: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    sdp_type: Option<String>,
}

/// SDP answer returned to the browser.
#[derive(Debug, Serialize)]
struct AnswerResponse {
    peer_id: String,
    sdp: String,
    #[serde(rename = "type")]
    sdp_type: String,
}

/// Stream info for the /streams endpoint.
#[derive(Debug, Serialize)]
struct StreamInfo {
    stream_id: String,
    peer_count: usize,
}

/// Error response.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

/// Disconnect request from the browser.
#[derive(Debug, Deserialize)]
struct DisconnectRequest {
    peer_id: String,
    stream_id: String,
}

/// Success response for disconnect.
#[derive(Debug, Serialize)]
struct DisconnectResponse {
    disconnected: bool,
    peer_id: String,
}

/// Start the signaling server on the given port.
pub async fn run_server(
    managers: SharedManagers,
    port: u16,
    cancel: tokio_util::sync::CancellationToken,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/streams", get(list_streams))
        .route("/webrtc/offer", post(handle_offer))
        .route("/webrtc/disconnect", post(handle_disconnect))
        .route("/webrtc/status", get(stream_status))
        .route("/health", get(health_check))
        .layer(CorsLayer::permissive())
        .with_state(managers);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    eprintln!("[webrtcrs] Signaling server listening on http://0.0.0.0:{port}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(cancel.cancelled_owned())
        .await?;

    eprintln!("[webrtcrs] Signaling server shut down.");
    Ok(())
}

/// GET /health — simple health check.
async fn health_check() -> &'static str {
    "ok"
}

/// GET /streams — list all active streams in registration order.
/// The order matches the order of start_streaming() calls (cam_index 0 first).
async fn list_streams(State(managers): State<SharedManagers>) -> Json<Vec<StreamInfo>> {
    let list = managers.read().await;
    let mut infos: Vec<StreamInfo> = Vec::with_capacity(list.len());
    for (id, mgr) in list.iter() {
        infos.push(StreamInfo {
            stream_id: id.clone(),
            peer_count: mgr.peer_count().await,
        });
    }
    Json(infos)
}

/// Resolve a stream_id string to a manager, searching the ordered Vec.
fn find_manager(
    managers: &tokio::sync::RwLockReadGuard<'_, Vec<(String, Arc<WebRtcManager>)>>,
    stream_id: &str,
) -> Option<Arc<WebRtcManager>> {
    managers
        .iter()
        .find(|(id, _)| id == stream_id)
        .map(|(_, mgr)| mgr.clone())
}

/// GET /webrtc/status?stream_id=X — peer count for a specific stream.
async fn stream_status(
    State(managers): State<SharedManagers>,
    Query(query): Query<StreamQuery>,
) -> Result<Json<StreamInfo>, (StatusCode, Json<ErrorResponse>)> {
    let stream_id = query.stream_id.unwrap_or_default();
    let list = managers.read().await;

    let mgr = find_manager(&list, &stream_id).ok_or_else(|| {
        let active: Vec<&str> = list.iter().map(|(id, _)| id.as_str()).collect();
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Stream '{}' not found. Active: {:?}", stream_id, active),
            }),
        )
    })?;

    Ok(Json(StreamInfo {
        stream_id,
        peer_count: mgr.peer_count().await,
    }))
}

/// POST /webrtc/offer?stream_id=X — accept a browser SDP offer, return an answer.
async fn handle_offer(
    State(managers): State<SharedManagers>,
    Query(query): Query<StreamQuery>,
    Json(offer_req): Json<OfferRequest>,
) -> Result<Json<AnswerResponse>, (StatusCode, Json<ErrorResponse>)> {
    let stream_id = query.stream_id.unwrap_or_default();

    // Resolve to the correct manager
    let mgr = {
        let list = managers.read().await;
        find_manager(&list, &stream_id).ok_or_else(|| {
            let active: Vec<&str> = list.iter().map(|(id, _)| id.as_str()).collect();
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!(
                        "Stream '{}' not found. Active streams: {:?}",
                        stream_id, active
                    ),
                }),
            )
        })?
    };

    // Parse the SDP offer
    let offer = webrtc::peer_connection::sdp::session_description::RTCSessionDescription::offer(
        offer_req.sdp,
    )
    .map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Invalid SDP offer: {e}"),
            }),
        )
    })?;

    // Create peer connection and get answer
    let (peer_id, answer) = mgr.add_peer(offer).await.map_err(|e| {
        // Return 429 if the error is a max-peers limit, else generic 500
        let msg = format!("{e}");
        let status = if msg.contains("Max peers") {
            StatusCode::TOO_MANY_REQUESTS
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (
            status,
            Json(ErrorResponse {
                error: format!("Failed to create peer: {e}"),
            }),
        )
    })?;

    Ok(Json(AnswerResponse {
        peer_id,
        sdp: answer.sdp,
        sdp_type: "answer".to_owned(),
    }))
}

/// POST /webrtc/disconnect — cleanly remove a single peer connection.
async fn handle_disconnect(
    State(managers): State<SharedManagers>,
    Json(req): Json<DisconnectRequest>,
) -> Result<Json<DisconnectResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mgr = {
        let list = managers.read().await;
        find_manager(&list, &req.stream_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("Stream '{}' not found", req.stream_id),
                }),
            )
        })?
    };

    mgr.remove_peer(&req.peer_id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to disconnect peer: {e}"),
            }),
        )
    })?;

    Ok(Json(DisconnectResponse {
        disconnected: true,
        peer_id: req.peer_id,
    }))
}
