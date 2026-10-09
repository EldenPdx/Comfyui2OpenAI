//! HTTP proxy handler and routing logic
//!
//! This module implements the core proxy functionality, routing incoming OpenAI API
//! requests to ComfyUI backend and handling error responses.

use axum::{
    body::{Body},
    extract::{Path, Query, State},
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response as AxumResponse},
};
use log::{error};
use reqwest::Client;
use serde::Serialize;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::Semaphore;
use crate::backend::BackendPool;
use crate::ws::WebSocketManager;
use crate::comfyui::{generations_response, image_file_response, BatchSender};
use serde_json::Value;

#[derive(Clone)]
pub struct BatchQueue {
    pub sender: BatchSender,
    pub capacity: Arc<Semaphore>,
}

/// Shared state passed to all request handlers
///
/// In web servers, each incoming request is handled by a separate async task.
/// The ProxyState struct allows you to share the HTTP Client and other resources
/// across all these concurrent request handlers without creating duplicates.
/// This is essential for efficient resource usage and connection pooling.
#[derive(Clone)]
pub struct ProxyState {
    /// Reusable HTTP client for ComfyUI backend communication
    pub client: Client,
    /// ComfyUI backend host address
    pub backend_url: String,
    /// ComfyUI backend port number
    pub backend_port: String,
    /// Unique client ID for ComfyUI WebSocket connections
    pub backend_client_id: String,
    /// Maximum allowed request body size in megabytes
    pub max_payload_size_mb: usize,
    /// Request timeout in seconds
    pub timeout: u64,
    /// WebSocket manager for job completion tracking (optional if use_ws is false)
    pub ws_manager: Option<Arc<WebSocketManager>>,
    /// Whether to use or not WebSockets
    pub use_ws: bool,
    /// Pre-loaded workflows (workflows) keyed by model name
    pub workflows: Arc<HashMap<String, Value>>,
    /// ComfyUI output directory for generated image cleanup
    pub output_directory: String,
    /// Whether public one-time image URLs may return image bytes.
    pub public_image_access: bool,
    /// Runtime registry for IMG2.0-S batch backends.
    pub backend_pool: Arc<BackendPool>,
}

/// Error types for proxy operations
///
/// Categorizes different error scenarios to enable appropriate HTTP status codes
/// and error messages in responses
#[derive(Debug, Clone)]
pub enum ProxyError {
    /// Internal server errors (500)
    Internal(String),
    /// Upstream/backend errors (502)
    Upstream(String),
    /// JSON parsing/serialization errors (400)
    Json(String),
    /// Request validation errors (400)
    #[allow(dead_code)]
    Validation(String),
    /// Unimplemented features (400)
    Implementation(String),
    /// Dynamic batch queue is full (429)
    Busy(String),
    /// Dynamic batch worker is unavailable (503)
    Unavailable(String),
    /// Image download access is denied (401)
    Unauthorized(String),
    /// Requested image URL has expired or was already consumed (404)
    #[allow(dead_code)]
    NotFound(String),
}

/// HTTP response body for error messages
#[derive(Serialize)]
struct ErrorResponse {
    /// HTTP status code
    code: u32,
    /// Error message
    message: String,
}

/// Converts ProxyError into an HTTP response with appropriate status code and JSON body
impl IntoResponse for ProxyError {
    fn into_response(self) -> AxumResponse {
        let (status, _code, message) = match self {
            ProxyError::Internal(msg) => {
                error!("❌ Internal error: {}", msg);
                (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", msg)
            }
            ProxyError::Upstream(msg) => {
                error!("❌ Upstream error: {}", msg);
                (StatusCode::BAD_GATEWAY, "UPSTREAM_ERROR", msg)
            }
            ProxyError::Json(msg) => {
                error!("❌ JSON error: {}", msg);
                (StatusCode::BAD_REQUEST, "JSON_ERROR", msg)
            }
            ProxyError::Validation(msg) => {
                error!("❌ Validation error: {}", msg);
                (StatusCode::BAD_REQUEST, "VALIDATION_ERROR", msg)
            }
            ProxyError::Implementation(msg) => {
                error!("❌ Implementation error: {}", msg);
                (StatusCode::BAD_REQUEST, "IMPLEMENTATION_ERROR", msg)
            }
            ProxyError::Busy(msg) => {
                error!("❌ Batch queue full: {}", msg);
                (StatusCode::TOO_MANY_REQUESTS, "TOO_MANY_REQUESTS", msg)
            }
            ProxyError::Unavailable(msg) => {
                error!("❌ Batch worker unavailable: {}", msg);
                (StatusCode::SERVICE_UNAVAILABLE, "SERVICE_UNAVAILABLE", msg)
            }
            ProxyError::Unauthorized(msg) => {
                (StatusCode::UNAUTHORIZED, "UNAUTHORIZED", msg)
            }
            ProxyError::NotFound(msg) => {
                (StatusCode::NOT_FOUND, "NOT_FOUND", msg)
            }

        };

        let error_response = ErrorResponse {
            code: status.as_u16() as u32,
            message,
        };

        let mut response = (status, axum::Json(error_response)).into_response();
        if status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(
                axum::http::header::WWW_AUTHENTICATE,
                axum::http::HeaderValue::from_static("Bearer realm=\"images\""),
            );
        }
        response
    }
}

/// Convert reqwest error to ProxyError
impl From<reqwest::Error> for ProxyError {
    fn from(err: reqwest::Error) -> Self {
        ProxyError::Upstream(format!("Request error: {}", err))
    }
}

/// Convert JSON serialization error to ProxyError
impl From<serde_json::Error> for ProxyError {
    fn from(err: serde_json::Error) -> Self {
        ProxyError::Json(format!("JSON error: {}", err))
    }
}

/// Handles request errors from the HTTP client
///
/// Categorizes the error type (timeout, connection, etc.) and returns an appropriate
/// ProxyError with a user-friendly message.
pub fn handle_request_error(e: reqwest::Error, full_url: &str) -> ProxyError {
    error!("❌ Request failed after send(): {:?}", e);
    error!("❌ Is timeout: {}", e.is_timeout());
    error!("❌ Is connect: {}", e.is_connect());
    error!("❌ Is request: {}", e.is_request());
    error!("❌ Is decode: {}", e.is_decode());
    if e.is_timeout() {
        ProxyError::Upstream(format!(
            "Request timeout to {}: {}",
            full_url, e
        ))
    } else if e.is_connect() {
        ProxyError::Upstream(format!(
            "Connection failed to {}: {} - Check if backend server is running",
            full_url, e
        ))
    } else if e.is_request() {
        ProxyError::Upstream(format!(
            "Request error to {}: {}",
            full_url, e
        ))
    } else {
        ProxyError::Upstream(format!(
            "Network error to {}: {}",
            full_url, e
        ))
    }
}

/// Handles timeout errors for requests that exceed the configured timeout
pub fn handle_timeout_error(full_url: &str, timeout_duration: Duration) -> ProxyError {
    error!(
        "❌ Request timed out after {} seconds",
        timeout_duration.as_secs()
    );
    ProxyError::Upstream(format!(
        "Request hung/timed out after {} seconds to {}",
        timeout_duration.as_secs(),
        full_url
    ))
}

/// Main request handler for all /v1/images/* paths
///
/// Routes incoming OpenAI API requests to the appropriate backend handler
/// based on the path segment (e.g., "generations" for /v1/images/generations).
///
/// # Arguments
/// * `state` - Shared proxy state (client, config, workflows)
/// * `path` - The path segment after /v1/images/ (e.g., "generations")
/// * `params` - URL query parameters
/// * `_method` - HTTP method (POST, GET, etc.)
/// * `headers` - HTTP request headers
/// * `body` - Request body
///
/// # Returns
/// - HTTP response for the requested operation
/// - ProxyError if the path is not implemented or other error occurs
pub async fn proxy_handler(
    State(state): State<Arc<ProxyState>>,
    Path(path): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    method: Method,
    headers: HeaderMap,
    body: Body,
) -> Result<AxumResponse, ProxyError> {

    // Use a full UUID so it is also safe as the per-request output filename suffix.
    let req_id = uuid::Uuid::new_v4().to_string();
    
    // Log request start with visual separator
    log::info!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    log::info!("┃ 🚀 REQUEST START [{}]", req_id);
    log::info!("┃ 📍 Path: /v1/images/{}", path);
    log::info!("┃ 🔗 Method: {}", method);
    log::info!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");

    // Route based on the requested endpoint
    let result = match path.as_str()  {
        // Handle OpenAI image generation API equivalent
        "generations" => {
            generations_response(State(state), Query(params), headers, body, &req_id).await
        }
        "gpu-count" if method == Method::GET => Ok(axum::Json(serde_json::json!({
            "gpu_count": state.backend_pool.ready_count()
        })).into_response()),
        "gpu-count" => Err(ProxyError::Implementation("GPU count only supports GET requests".to_string())),
        "files" if !state.public_image_access => Err(ProxyError::Unauthorized("Unauthorized. You are not on the authorized user whitelist.".to_string())),
        "files" if method == Method::GET => image_file_response(&state.output_directory, params).await,
        "files" => Err(ProxyError::Implementation("Image files only support GET requests".to_string())),
        // Reject unsupported endpoints
        _ => {
            error!("❌ [{}] Path not supported: {}", req_id, path);
            Err(ProxyError::Implementation(format!(
                "Path not supported: {}",
                path
            )))
        }
    };

    // Log request end
    match &result {
        Ok(res) => {
             log::info!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
             log::info!("┃ ✅ REQUEST FINISHED [{}]", req_id);
             log::info!("┃ 📡 Status: {}", res.status());
             log::info!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
        },
        Err(e) => {
             log::error!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
             log::error!("┃ ❌ REQUEST FAILED [{}]", req_id);
             log::error!("┃ 💥 Error: {:?}", e);
             log::error!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
        }
    }

    result
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn image_download_route_is_disabled_without_consuming_files() {
        let root = std::env::temp_dir().join(format!("comfyui-disabled-download-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let filename = format!("image-{}.png", uuid::Uuid::new_v4());
        let image = root.join(&filename);
        std::fs::write(&image, b"image must remain for periodic cleanup").unwrap();
        let (batch_sender, _receiver, batch_capacity) = crate::comfyui::dynamic_batch_channel();
        let backend_pool = Arc::new(crate::backend::BackendPool::new());
        let state = Arc::new(ProxyState {
            client: Client::new(),
            backend_url: "127.0.0.1".to_string(),
            backend_port: "8188".to_string(),
            backend_client_id: "test".to_string(),
            max_payload_size_mb: 10,
            timeout: 30,
            ws_manager: None,
            use_ws: false,
            workflows: Arc::new(HashMap::new()),
            output_directory: root.to_string_lossy().into_owned(),
            public_image_access: false,
            backend_pool: backend_pool.clone(),
        });
        backend_pool.add_ready("test".to_string(), state.clone(), BatchQueue { sender: batch_sender, capacity: batch_capacity });
        let params = HashMap::from([("filename".to_string(), filename)]);
        for method in [Method::GET, Method::HEAD, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS] {
            let result = proxy_handler(
                State(state.clone()), Path("files".to_string()), Query(params.clone()),
                method.clone(), HeaderMap::new(), Body::empty(),
            ).await;
            let response = match result {
                Ok(response) => response,
                Err(error) => error.into_response(),
            };
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{} must not serve images", method);
            assert_eq!(response.headers()["www-authenticate"], "Bearer realm=\"images\"");
            let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body, serde_json::json!({ "code": 401, "message": "Unauthorized. You are not on the authorized user whitelist." }));
            assert_eq!(std::fs::read(&image).unwrap(), b"image must remain for periodic cleanup");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn enabled_image_download_returns_and_consumes_the_file() {
        let root = std::env::temp_dir().join(format!("comfyui-enabled-download-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let filename = format!("image-{}.png", uuid::Uuid::new_v4());
        let image = root.join(&filename);
        std::fs::write(&image, b"png bytes").unwrap();
        let (batch_sender, _receiver, batch_capacity) = crate::comfyui::dynamic_batch_channel();
        let backend_pool = Arc::new(crate::backend::BackendPool::new());
        let state = Arc::new(ProxyState {
            client: Client::new(),
            backend_url: "127.0.0.1".to_string(),
            backend_port: "8188".to_string(),
            backend_client_id: "test".to_string(),
            max_payload_size_mb: 10,
            timeout: 30,
            ws_manager: None,
            use_ws: false,
            workflows: Arc::new(HashMap::new()),
            output_directory: root.to_string_lossy().into_owned(),
            public_image_access: true,
            backend_pool: backend_pool.clone(),
        });
        backend_pool.add_ready("test".to_string(), state.clone(), BatchQueue { sender: batch_sender, capacity: batch_capacity });
        let params = HashMap::from([
            ("filename".to_string(), filename),
            ("subfolder".to_string(), String::new()),
            ("type".to_string(), "output".to_string()),
        ]);

        let response = proxy_handler(
            State(state.clone()), Path("files".to_string()), Query(params.clone()),
            Method::GET, HeaderMap::new(), Body::empty(),
        ).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "image/png");
        assert_eq!(axum::body::to_bytes(response.into_body(), 1024).await.unwrap(), "png bytes");
        assert!(!image.exists());

        let second = proxy_handler(
            State(state), Path("files".to_string()), Query(params),
            Method::GET, HeaderMap::new(), Body::empty(),
        ).await.unwrap_err();
        assert!(matches!(second, ProxyError::NotFound(_)));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn gpu_count_route_reports_ready_backends_and_requires_get() {
        let (batch_sender, _receiver, batch_capacity) = crate::comfyui::dynamic_batch_channel();
        let backend_pool = Arc::new(crate::backend::BackendPool::new());
        let state = Arc::new(ProxyState {
            client: Client::new(),
            backend_url: "127.0.0.1".to_string(),
            backend_port: "8188".to_string(),
            backend_client_id: "test".to_string(),
            max_payload_size_mb: 10,
            timeout: 30,
            ws_manager: None,
            use_ws: false,
            workflows: Arc::new(HashMap::new()),
            output_directory: std::env::temp_dir().to_string_lossy().into_owned(),
            public_image_access: false,
            backend_pool: backend_pool.clone(),
        });
        backend_pool.add_ready(
            "GPU-test".to_string(),
            state.clone(),
            BatchQueue { sender: batch_sender, capacity: batch_capacity },
        );

        let response = proxy_handler(
            State(state.clone()), Path("gpu-count".to_string()), Query(HashMap::new()),
            Method::GET, HeaderMap::new(), Body::empty(),
        ).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), serde_json::json!({"gpu_count": 1}));

        let error = proxy_handler(
            State(state), Path("gpu-count".to_string()), Query(HashMap::new()),
            Method::POST, HeaderMap::new(), Body::empty(),
        ).await.unwrap_err();
        assert!(matches!(error, ProxyError::Implementation(_)));
    }
}
