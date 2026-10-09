use axum::{
    Json, Router,
    extract::{Path, State},
    routing::{get, put},
};
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        Arc, RwLock,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::{
    comfyui::{dynamic_batch_channel, run_dynamic_batch_worker, warm_dynamic_batch_backend},
    proxy::{BatchQueue, ProxyError, ProxyState},
    ws::WebSocketManager,
};

pub struct BackendPool {
    entries: RwLock<Vec<Arc<BackendEntry>>>,
    next: AtomicUsize,
}

struct BackendEntry {
    id: RwLock<String>,
    port: String,
    queue: BatchQueue,
    state: Arc<ProxyState>,
    lifecycle: AtomicU8,
    transition: tokio::sync::Mutex<()>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Lifecycle {
    Warming,
    Ready,
    Draining,
    Unavailable,
}

impl Lifecycle {
    fn name(self) -> &'static str {
        match self {
            Self::Warming => "warming",
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Serialize)]
pub(crate) struct BackendView {
    id: String,
    port: String,
    status: &'static str,
    in_flight_images: usize,
}

#[derive(Deserialize)]
struct RegisterRequest {
    port: u16,
}

impl BackendPool {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            next: AtomicUsize::new(0),
        }
    }

    #[cfg(test)]
    pub fn add_ready(&self, id: String, state: Arc<ProxyState>, queue: BatchQueue) {
        self.entries
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .push(Arc::new(BackendEntry {
                id: RwLock::new(id),
                port: state.backend_port.clone(),
                queue,
                state,
                lifecycle: AtomicU8::new(Lifecycle::Ready as u8),
                transition: tokio::sync::Mutex::new(()),
            }));
    }

    pub fn ready_queues(&self) -> Vec<BatchQueue> {
        self.entries
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .filter(|entry| {
                entry.lifecycle() == Lifecycle::Ready && !entry.queue.sender.is_closed()
            })
            .map(|entry| entry.queue.clone())
            .collect()
    }

    pub fn has_ready(&self) -> bool {
        self.ready_count() > 0
    }

    pub fn ready_count(&self) -> usize {
        self.entries
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .filter(|entry| {
                entry.lifecycle() == Lifecycle::Ready && !entry.queue.sender.is_closed()
            })
            .count()
    }

    pub fn next_index(&self, len: usize) -> usize {
        self.next.fetch_add(1, Ordering::Relaxed) % len
    }

    fn list(&self) -> Vec<BackendView> {
        self.entries
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .map(|entry| entry.view())
            .collect()
    }

    fn insert(&self, entry: Arc<BackendEntry>) {
        self.entries
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .push(entry);
    }

    fn find(&self, id: &str, port: &str) -> Option<Arc<BackendEntry>> {
        self.entries
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .find(|entry| {
                entry.port == port
                    || *entry.id.read().unwrap_or_else(|error| error.into_inner()) == id
            })
            .cloned()
    }

    fn find_by_id(&self, id: &str) -> Option<Arc<BackendEntry>> {
        self.entries
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .find(|entry| *entry.id.read().unwrap_or_else(|error| error.into_inner()) == id)
            .cloned()
    }

    fn remove(&self, target: &Arc<BackendEntry>) {
        self.entries
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .retain(|entry| !Arc::ptr_eq(entry, target));
    }

    fn snapshot(&self) -> Vec<Arc<BackendEntry>> {
        self.entries
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

impl BackendEntry {
    fn lifecycle(&self) -> Lifecycle {
        match self.lifecycle.load(Ordering::Acquire) {
            0 => Lifecycle::Warming,
            1 => Lifecycle::Ready,
            2 => Lifecycle::Draining,
            _ => Lifecycle::Unavailable,
        }
    }

    fn set_lifecycle(&self, lifecycle: Lifecycle) {
        self.lifecycle.store(lifecycle as u8, Ordering::Release);
    }

    fn view(&self) -> BackendView {
        BackendView {
            id: self
                .id
                .read()
                .unwrap_or_else(|error| error.into_inner())
                .clone(),
            port: self.port.clone(),
            status: self.lifecycle().name(),
            in_flight_images: crate::comfyui::BATCH_QUEUE_CAPACITY
                - self.queue.capacity.available_permits(),
        }
    }
}

async fn list_backends(State(state): State<Arc<ProxyState>>) -> Json<Vec<BackendView>> {
    Json(state.backend_pool.list())
}

async fn register_backend(
    State(root): State<Arc<ProxyState>>,
    Path(id): Path<String>,
    Json(request): Json<RegisterRequest>,
) -> Result<Json<BackendView>, ProxyError> {
    register_backend_runtime(root, id, request.port)
        .await
        .map(Json)
}

pub(crate) async fn register_backend_runtime(
    root: Arc<ProxyState>,
    id: String,
    backend_port: u16,
) -> Result<BackendView, ProxyError> {
    if !id.starts_with("GPU-") || id.len() <= 4 {
        return Err(ProxyError::Validation(
            "Backend id must be an NVIDIA GPU UUID".to_string(),
        ));
    }
    let port = backend_port.to_string();
    let health_url = format!("http://{}:{}/system_stats", root.backend_url, port);
    let healthy = tokio::time::timeout(Duration::from_secs(2), root.client.get(&health_url).send())
        .await
        .is_ok_and(|result| result.is_ok_and(|response| response.status().is_success()));
    if !healthy {
        return Err(ProxyError::Unavailable(format!(
            "ComfyUI backend {} is not healthy",
            port
        )));
    }
    if let Some(existing) = root.backend_pool.find(&id, &port) {
        let _transition = existing.transition.lock().await;
        if existing.port != port {
            return Err(ProxyError::Validation(format!(
                "Backend {} is already registered on another port",
                id
            )));
        }
        *existing
            .id
            .write()
            .unwrap_or_else(|error| error.into_inner()) = id;
        if existing.lifecycle() == Lifecycle::Ready {
            return Ok(existing.view());
        }
        existing.set_lifecycle(Lifecycle::Warming);
        match warm_dynamic_batch_backend(&existing.state, &existing.queue).await {
            Ok(()) => existing.set_lifecycle(Lifecycle::Ready),
            Err(error) => {
                existing.set_lifecycle(Lifecycle::Unavailable);
                return Err(error);
            }
        }
        return Ok(existing.view());
    }

    let ws_manager = if root.use_ws {
        Some(
            WebSocketManager::new(
                root.backend_url.clone(),
                port.clone(),
                format!(
                    "{}-{}",
                    root.backend_client_id,
                    id[4..].chars().take(8).collect::<String>()
                ),
            )
            .await
            .map_err(|error| {
                ProxyError::Unavailable(format!("WebSocket initialization failed: {}", error))
            })?,
        )
    } else {
        None
    };
    let (sender, receiver, capacity) = dynamic_batch_channel();
    let queue = BatchQueue { sender, capacity };
    let state = Arc::new(ProxyState {
        client: root.client.clone(),
        backend_url: root.backend_url.clone(),
        backend_port: port.clone(),
        backend_client_id: format!(
            "{}-{}",
            root.backend_client_id,
            id[4..].chars().take(8).collect::<String>()
        ),
        max_payload_size_mb: root.max_payload_size_mb,
        timeout: root.timeout,
        ws_manager,
        use_ws: root.use_ws,
        workflows: root.workflows.clone(),
        output_directory: root.output_directory.clone(),
        public_image_access: root.public_image_access,
        backend_pool: root.backend_pool.clone(),
    });
    let entry = Arc::new(BackendEntry {
        id: RwLock::new(id),
        port,
        queue: queue.clone(),
        state: state.clone(),
        lifecycle: AtomicU8::new(Lifecycle::Warming as u8),
        transition: tokio::sync::Mutex::new(()),
    });
    let _transition = entry.transition.lock().await;
    root.backend_pool.insert(entry.clone());
    tokio::spawn(run_dynamic_batch_worker(Arc::downgrade(&state), receiver));
    match warm_dynamic_batch_backend(&state, &queue).await {
        Ok(()) => entry.set_lifecycle(Lifecycle::Ready),
        Err(error) => {
            entry.set_lifecycle(Lifecycle::Unavailable);
            return Err(error);
        }
    }
    Ok(entry.view())
}

async fn drain_backend(
    State(root): State<Arc<ProxyState>>,
    Path(id): Path<String>,
) -> Result<Json<BackendView>, ProxyError> {
    let entry = root
        .backend_pool
        .find_by_id(&id)
        .ok_or_else(|| ProxyError::NotFound(format!("Backend {} is not registered", id)))?;
    let _transition = entry.transition.lock().await;
    entry.set_lifecycle(Lifecycle::Draining);
    Ok(Json(entry.view()))
}

async fn remove_backend(
    State(root): State<Arc<ProxyState>>,
    Path(id): Path<String>,
) -> Result<Json<BackendView>, ProxyError> {
    let entry = root
        .backend_pool
        .find_by_id(&id)
        .ok_or_else(|| ProxyError::NotFound(format!("Backend {} is not registered", id)))?;
    {
        let _transition = entry.transition.lock().await;
        entry.set_lifecycle(Lifecycle::Draining);
    }
    tokio::time::timeout(Duration::from_secs(root.timeout), async {
        while entry.queue.capacity.available_permits() != crate::comfyui::BATCH_QUEUE_CAPACITY {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .map_err(|_| {
        ProxyError::Upstream(format!(
            "Backend {} did not drain within {} seconds",
            id, root.timeout
        ))
    })?;
    let view = entry.view();
    root.backend_pool.remove(&entry);
    Ok(Json(view))
}

pub async fn check_backends_once(root: Arc<ProxyState>) {
    for entry in root.backend_pool.snapshot() {
        if matches!(entry.lifecycle(), Lifecycle::Warming | Lifecycle::Draining) {
            continue;
        }
        let url = format!("http://{}:{}/system_stats", root.backend_url, entry.port);
        let healthy = tokio::time::timeout(Duration::from_secs(2), root.client.get(url).send())
            .await
            .is_ok_and(|result| result.is_ok_and(|response| response.status().is_success()));
        let _transition = entry.transition.lock().await;
        match (entry.lifecycle(), healthy) {
            (Lifecycle::Ready, false) => {
                log::warn!("ComfyUI backend {} became unavailable", entry.port);
                entry.set_lifecycle(Lifecycle::Unavailable);
            }
            (Lifecycle::Unavailable, true)
                if entry.queue.capacity.available_permits()
                    == crate::comfyui::BATCH_QUEUE_CAPACITY =>
            {
                entry.set_lifecycle(Lifecycle::Warming);
                match warm_dynamic_batch_backend(&entry.state, &entry.queue).await {
                    Ok(()) => {
                        entry.set_lifecycle(Lifecycle::Ready);
                        log::info!("ComfyUI backend {} recovered after warmup", entry.port);
                    }
                    Err(error) => {
                        entry.set_lifecycle(Lifecycle::Unavailable);
                        log::warn!(
                            "ComfyUI backend {} recovery warmup failed: {:?}",
                            entry.port,
                            error
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

pub async fn run_backend_health_loop(root: Arc<ProxyState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        interval.tick().await;
        check_backends_once(root.clone()).await;
    }
}

pub fn control_router(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/backends", get(list_backends))
        .route(
            "/backends/:id",
            put(register_backend).delete(remove_backend),
        )
        .route("/backends/:id/drain", axum::routing::post(drain_backend))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{comfyui, proxy::ProxyState};
    use axum::{
        Router,
        body::{Body, to_bytes},
        extract::State,
        http::{Request, StatusCode},
        response::IntoResponse,
        routing::any,
    };
    use reqwest::Client;
    use serde_json::json;
    use std::{
        collections::HashMap,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };
    use tower::Service;

    fn test_state(pool: Arc<BackendPool>) -> Arc<ProxyState> {
        Arc::new(ProxyState {
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
            backend_pool: pool,
        })
    }

    #[tokio::test]
    async fn empty_control_registry_is_visible_and_public_health_is_unavailable() {
        let state = test_state(Arc::new(BackendPool::new()));
        let mut app = control_router(state.clone());
        let response = app
            .call(
                Request::builder()
                    .uri("/backends")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(to_bytes(response.into_body(), 1024).await.unwrap(), "[]");
        assert_eq!(
            comfyui::health(State(state)).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn registration_returns_ready_only_after_img2_warmup_completes() {
        let root =
            std::env::temp_dir().join(format!("comfy-backend-warmup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let warmed = Arc::new(AtomicBool::new(false));
        let history_root = root.clone();
        let history_warmed = warmed.clone();
        let fake_comfy = Router::new()
            .route("/system_stats", any(|| async { axum::Json(json!({"devices": []})) }))
            .route("/prompt", any(|| async { axum::Json(json!({"prompt_id": "warmup"})) }))
            .route("/queue", any(|| async { axum::Json(json!({"queue_running": [], "queue_pending": []})) }))
            .route("/history/warmup", any(move || {
                let root = history_root.clone();
                let warmed = history_warmed.clone();
                async move {
                    std::fs::write(root.join("warmup.png"), b"warm").unwrap();
                    warmed.store(true, Ordering::SeqCst);
                    axum::Json(json!({"warmup": {
                        "status": {"completed": true, "status_str": "success"},
                        "outputs": {"5": {"images": [{"filename": "warmup.png", "subfolder": "", "type": "output"}]}}
                    }}))
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move { axum::serve(listener, fake_comfy).await.unwrap() });

        let workflow = json!({
            "1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
            "2": {"class_type": "OpenAIBatchCLIPTextEncode", "inputs": {}, "_meta": {"title": "Positive Prompt"}},
            "3": {"class_type": "OpenAIBatchCLIPTextEncode", "inputs": {}, "_meta": {"title": "Negative Prompt"}},
            "4": {"class_type": "OpenAIBatchNoise", "inputs": {}},
            "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test"}}
        });
        let state = Arc::new(ProxyState {
            client: Client::builder().no_proxy().build().unwrap(),
            backend_url: "127.0.0.1".to_string(),
            backend_port: "8188".to_string(),
            backend_client_id: "test".to_string(),
            max_payload_size_mb: 10,
            timeout: 30,
            ws_manager: None,
            use_ws: false,
            workflows: Arc::new(HashMap::from([("IMG2.0-S".to_string(), workflow)])),
            output_directory: root.to_string_lossy().into_owned(),
            public_image_access: false,
            backend_pool: Arc::new(BackendPool::new()),
        });
        let mut app = control_router(state.clone());
        let response = app
            .call(
                Request::builder()
                    .method("PUT")
                    .uri("/backends/GPU-test")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"port": port}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(warmed.load(Ordering::SeqCst));
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body["id"], "GPU-test");
        assert_eq!(body["status"], "ready");
        assert_eq!(comfyui::health(State(state)).await, StatusCode::OK);

        server.abort();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn drain_stops_new_routing_and_delete_waits_for_in_flight_capacity() {
        let pool = Arc::new(BackendPool::new());
        let state = test_state(pool.clone());
        let (sender, receiver, capacity) = comfyui::dynamic_batch_channel();
        let queue = crate::proxy::BatchQueue {
            sender,
            capacity: capacity.clone(),
        };
        pool.add_ready("GPU-test".to_string(), state.clone(), queue);
        let in_flight = capacity.clone().try_acquire_owned().unwrap();
        let mut app = control_router(state.clone());

        let response = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/backends/GPU-test/drain")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            comfyui::health(State(state.clone())).await,
            StatusCode::SERVICE_UNAVAILABLE
        );

        let delete = app.call(
            Request::builder()
                .method("DELETE")
                .uri("/backends/GPU-test")
                .body(Body::empty())
                .unwrap(),
        );
        tokio::pin!(delete);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut delete)
                .await
                .is_err()
        );
        drop(in_flight);
        assert_eq!(delete.await.unwrap().status(), StatusCode::OK);

        let response = app
            .call(
                Request::builder()
                    .uri("/backends")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(to_bytes(response.into_body(), 1024).await.unwrap(), "[]");
        drop(receiver);
    }

    #[tokio::test]
    async fn unhealthy_backend_is_ejected_and_rewarmed_before_recovery() {
        let root =
            std::env::temp_dir().join(format!("comfy-backend-recovery-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let healthy = Arc::new(AtomicBool::new(true));
        let warmups = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let recovery_started = Arc::new(tokio::sync::Semaphore::new(0));
        let release_recovery = Arc::new(tokio::sync::Semaphore::new(0));
        let health_flag = healthy.clone();
        let history_root = root.clone();
        let history_warmups = warmups.clone();
        let history_started = recovery_started.clone();
        let history_release = release_recovery.clone();
        let fake_comfy = Router::new()
            .route("/system_stats", any(move || {
                let healthy = health_flag.clone();
                async move {
                    if healthy.load(Ordering::SeqCst) {
                        (StatusCode::OK, axum::Json(json!({"devices": []}))).into_response()
                    } else {
                        (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({}))).into_response()
                    }
                }
            }))
            .route("/prompt", any(|| async { axum::Json(json!({"prompt_id": "warmup"})) }))
            .route("/queue", any(|| async { axum::Json(json!({"queue_running": [], "queue_pending": []})) }))
            .route("/history/warmup", any(move || {
                let root = history_root.clone();
                let warmups = history_warmups.clone();
                let started = history_started.clone();
                let release = history_release.clone();
                async move {
                    let warmup = warmups.fetch_add(1, Ordering::SeqCst);
                    if warmup == 1 {
                        started.add_permits(1);
                        release.acquire().await.unwrap().forget();
                    }
                    std::fs::write(root.join("warmup.png"), b"warm").unwrap();
                    axum::Json(json!({"warmup": {
                        "status": {"completed": true, "status_str": "success"},
                        "outputs": {"5": {"images": [{"filename": "warmup.png", "subfolder": "", "type": "output"}]}}
                    }}))
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move { axum::serve(listener, fake_comfy).await.unwrap() });
        let workflow = json!({
            "1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
            "2": {"class_type": "OpenAIBatchCLIPTextEncode", "inputs": {}, "_meta": {"title": "Positive Prompt"}},
            "3": {"class_type": "OpenAIBatchCLIPTextEncode", "inputs": {}, "_meta": {"title": "Negative Prompt"}},
            "4": {"class_type": "OpenAIBatchNoise", "inputs": {}},
            "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test"}}
        });
        let state = Arc::new(ProxyState {
            client: Client::builder().no_proxy().build().unwrap(),
            backend_url: "127.0.0.1".to_string(),
            backend_port: "8188".to_string(),
            backend_client_id: "test".to_string(),
            max_payload_size_mb: 10,
            timeout: 30,
            ws_manager: None,
            use_ws: false,
            workflows: Arc::new(HashMap::from([("IMG2.0-S".to_string(), workflow)])),
            output_directory: root.to_string_lossy().into_owned(),
            public_image_access: false,
            backend_pool: Arc::new(BackendPool::new()),
        });
        let mut app = control_router(state.clone());
        let register = || {
            Request::builder()
                .method("PUT")
                .uri("/backends/GPU-test")
                .header("content-type", "application/json")
                .body(Body::from(json!({"port": port}).to_string()))
                .unwrap()
        };
        assert_eq!(app.call(register()).await.unwrap().status(), StatusCode::OK);
        assert_eq!(warmups.load(Ordering::SeqCst), 1);

        healthy.store(false, Ordering::SeqCst);
        check_backends_once(state.clone()).await;
        assert_eq!(
            comfyui::health(State(state.clone())).await,
            StatusCode::SERVICE_UNAVAILABLE
        );

        healthy.store(true, Ordering::SeqCst);
        let recovery_state = state.clone();
        let recovery = tokio::spawn(async move { check_backends_once(recovery_state).await });
        recovery_started.acquire().await.unwrap().forget();
        let mut drain_app = control_router(state.clone());
        let drain = tokio::spawn(async move {
            drain_app
                .call(
                    Request::builder()
                        .method("POST")
                        .uri("/backends/GPU-test/drain")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
        });
        release_recovery.add_permits(1);
        recovery.await.unwrap();
        assert_eq!(drain.await.unwrap().status(), StatusCode::OK);
        assert_eq!(
            comfyui::health(State(state)).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(warmups.load(Ordering::SeqCst), 2);

        server.abort();
        std::fs::remove_dir_all(root).unwrap();
    }
}
