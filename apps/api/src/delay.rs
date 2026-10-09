//! Optional post-generation response delay. Disabled for existing configurations.

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use futures::FutureExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{panic::AssertUnwindSafe, time::Duration};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct DelayConfig {
    pub enabled: bool,
    pub min_seconds: f64,
    pub max_seconds: f64,
}

impl Default for DelayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_seconds: 3.0,
            max_seconds: 8.0,
        }
    }
}

/// Wait only after the downstream handler has produced its complete image response.
pub async fn response_delay(
    State(config): State<DelayConfig>,
    request: Request,
    next: Next,
) -> Response {
    let is_generation = request.uri().path() == "/v1/images/generations";
    let response = next.run(request).await;
    if is_generation && response.status().is_success() {
        config.wait().await;
    }
    response
}

impl DelayConfig {
    pub async fn wait(&self) {
        if !self.enabled {
            return;
        }
        if !self.min_seconds.is_finite()
            || self.min_seconds < 0.0
            || self.min_seconds > self.max_seconds
            || Duration::try_from_secs_f64(self.max_seconds)
                .ok()
                .and_then(|duration| tokio::time::Instant::now().checked_add(duration))
                .is_none()
        {
            log::warn!("Invalid delay_layer range; returning response without delay");
            return;
        }
        // Only the optional wait is isolated; generation errors retain their normal handling.
        let wait = async {
            let seconds = rand::rng().random_range(self.min_seconds..=self.max_seconds);
            log::debug!("Delaying completed image response by {:.3}s", seconds);
            tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
        };
        if AssertUnwindSafe(wait).catch_unwind().await.is_err() {
            log::warn!("Delay layer failed; returning the original response without further delay");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request, StatusCode},
        routing::any,
    };
    use tokio::time::Instant;
    use tower::Service;

    #[tokio::test(start_paused = true)]
    async fn disabled_delay_returns_immediately() {
        let start = Instant::now();
        DelayConfig::default().wait().await;
        assert_eq!(start.elapsed(), std::time::Duration::ZERO);
    }

    #[tokio::test(start_paused = true)]
    async fn each_enabled_request_waits_a_random_fractional_3_to_8_seconds() {
        let config = DelayConfig {
            enabled: true,
            ..Default::default()
        };
        let mut waits = Vec::new();
        for _ in 0..32 {
            let start = Instant::now();
            config.wait().await;
            let seconds = start.elapsed().as_secs_f64();
            // Tokio timers round to millisecond resolution.
            assert!((3.0..=8.001).contains(&seconds), "waited {seconds}s");
            waits.push(seconds);
        }
        assert!(waits.iter().any(|seconds| seconds.fract() != 0.0));
        assert!(waits.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[tokio::test(start_paused = true)]
    async fn invalid_delay_ranges_fail_open_without_waiting() {
        for (min_seconds, max_seconds) in [
            (-1.0, 30.0),
            (30.0, 15.0),
            (f64::NAN, 30.0),
            (15.0, f64::NAN),
            (15.0, f64::INFINITY),
            (f64::NEG_INFINITY, 30.0),
            (15.0, f64::MAX),
            (1e19, 1e19),
        ] {
            let start = Instant::now();
            DelayConfig {
                enabled: true,
                min_seconds,
                max_seconds,
            }
            .wait()
            .await;
            assert_eq!(start.elapsed(), Duration::ZERO);
        }
    }

    #[test]
    fn timer_failure_does_not_escape_the_delay_layer() {
        // A runtime without its timer driver simulates a real timer failure.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(
            DelayConfig {
                enabled: true,
                ..Default::default()
            }
            .wait(),
        );
    }

    #[tokio::test(start_paused = true)]
    async fn http_response_waits_after_generation_without_changing_context_or_data() {
        let start = Instant::now();
        let config: DelayConfig =
            serde_yaml::from_str("enabled: true\nmin_seconds: 15.375\nmax_seconds: 15.375")
                .unwrap();
        let mut app = Router::new()
            .route(
                "/v1/images/*path",
                any(move |request: Request<Body>| async move {
                    assert_eq!(start.elapsed(), Duration::ZERO, "must not delay generation");
                    assert_eq!(request.method(), "POST");
                    assert_eq!(request.headers()["x-request-id"], "request-123");
                    assert_eq!(request.extensions().get::<u64>(), Some(&42));
                    assert_eq!(
                        to_bytes(request.into_body(), 1024).await.unwrap(),
                        "original request"
                    );
                    tokio::time::sleep(Duration::from_millis(2500)).await;
                    axum::response::Response::builder()
                        .status(StatusCode::CREATED)
                        .header("x-image-id", "image-123")
                        .extension(Instant::now())
                        .body(Body::from(r#"{"data":[{"url":"original-image.png"}]}"#))
                        .unwrap()
                }),
            )
            .layer(axum::middleware::from_fn_with_state(config, response_delay));
        let response = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations?client=test")
                    .header("x-request-id", "request-123")
                    .extension(42u64)
                    .body(Body::from("original request"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let completed = response.extensions().get::<Instant>().unwrap();
        assert!((15.375..=15.377).contains(&completed.elapsed().as_secs_f64()));
        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(response.headers()["x-image-id"], "image-123");
        assert_eq!(
            to_bytes(response.into_body(), 1024).await.unwrap(),
            r#"{"data":[{"url":"original-image.png"}]}"#
        );
    }

    #[tokio::test(start_paused = true)]
    async fn errors_and_non_generation_routes_are_not_delayed() {
        for (path, status) in [
            ("/v1/images/generations", StatusCode::BAD_REQUEST),
            ("/v1/images/generations", StatusCode::BAD_GATEWAY),
            ("/v1/images/generations", StatusCode::TOO_MANY_REQUESTS),
            ("/v1/images/files", StatusCode::UNAUTHORIZED),
            ("/health", StatusCode::OK),
            ("/v1/images/other", StatusCode::OK),
        ] {
            let config = DelayConfig {
                enabled: true,
                ..Default::default()
            };
            let mut app = Router::new()
                .route(path, any(move || async move { status }))
                .layer(axum::middleware::from_fn_with_state(config, response_delay));
            let start = Instant::now();
            let response = app
                .call(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(
                start.elapsed(),
                Duration::ZERO,
                "must not delay {path} {status}"
            );
        }
    }

    #[tokio::test]
    async fn ordinary_and_batched_generation_preserve_images_until_response_delivery() {
        use crate::{comfyui, proxy};
        use serde_json::json;
        use std::{
            collections::HashMap,
            sync::{Arc, Mutex},
        };

        let root = std::env::temp_dir().join(format!("comfy-delay-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        // Mock only the external ComfyUI HTTP service, not the generation handlers.
        let generated_at = Arc::new(Mutex::new(None));
        let marker = generated_at.clone();
        let history_root = root.clone();
        let backend = Router::new()
            .route(
                "/prompt",
                any(|| async { axum::Json(json!({"prompt_id": "test"})) }),
            )
            .route(
                "/queue",
                any(|| async { axum::Json(json!({"queue_running": [], "queue_pending": []})) }),
            )
            .route(
                "/history/test",
                any(move || {
                    let marker = marker.clone();
                    let history_root = history_root.clone();
                    async move {
                        *marker.lock().unwrap() = Some(Instant::now());
                        let images = (0..5).map(|index| {
                            let filename = format!("generated-{index}.png");
                            std::fs::write(
                                history_root.join(&filename),
                                b"unchanged generated image",
                            ).unwrap();
                            json!({"filename": filename, "subfolder": "", "type": "output"})
                        }).collect::<Vec<_>>();
                        axum::Json(json!({"test": {
                            "status": {"completed": true, "status_str": "success"},
                            "outputs": {"5": {"images": images}}
                        }}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move { axum::serve(listener, backend).await.unwrap() });
        let workflow = json!({
            "1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
            "2": {"class_type": "OpenAIBatchCLIPTextEncode", "inputs": {}, "_meta": {"title": "Positive Prompt"}},
            "3": {"class_type": "OpenAIBatchCLIPTextEncode", "inputs": {}, "_meta": {"title": "Negative Prompt"}},
            "4": {"class_type": "OpenAIBatchNoise", "inputs": {}},
            "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test"}}
        });
        let (batch_sender, batch_receiver, batch_capacity) = comfyui::dynamic_batch_channel();
        let backend_pool = Arc::new(crate::backend::BackendPool::new());
        let state = Arc::new(proxy::ProxyState {
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            backend_url: "127.0.0.1".to_string(),
            backend_port: port.to_string(),
            backend_client_id: "test".to_string(),
            max_payload_size_mb: 10,
            timeout: 30,
            ws_manager: None,
            use_ws: false,
            workflows: Arc::new(HashMap::from([
                ("ordinary".to_string(), workflow.clone()),
                ("IMG2.0-S".to_string(), workflow),
            ])),
            output_directory: root.to_string_lossy().into_owned(),
            public_image_access: false,
            backend_pool: backend_pool.clone(),
        });
        backend_pool.add_ready(
            "test".to_string(),
            state.clone(),
            proxy::BatchQueue { sender: batch_sender, capacity: batch_capacity },
        );
        let worker = tokio::spawn(comfyui::run_dynamic_batch_worker(
            Arc::downgrade(&state),
            batch_receiver,
        ));
        let config = DelayConfig {
            enabled: true,
            min_seconds: 0.05,
            max_seconds: 0.05,
        };
        let sweep_root = root.clone();
        let sweeper = tokio::spawn(async move {
            loop {
                // Make completed files older than the 30-second TTL without sleeping.
                for entry in std::fs::read_dir(&sweep_root).unwrap().flatten() {
                    if entry.file_name().to_string_lossy().starts_with("image-")
                        && let Ok(file) = std::fs::File::open(entry.path())
                    {
                        let _ = file
                            .set_modified(std::time::SystemTime::now() - Duration::from_secs(60));
                    }
                }
                comfyui::cleanup_stale_images_before(
                    &sweep_root,
                    std::time::SystemTime::now() - Duration::from_secs(30),
                );
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
        let mut app = Router::new()
            .route("/v1/images/*path", any(proxy::proxy_handler))
            .layer(axum::middleware::from_fn_with_state(config, response_delay))
            .with_state(state.clone());
        for model in ["ordinary", "IMG2.0-S"] {
            let response = app
                .call(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/images/generations")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            json!({"model": model, "prompt": "cat", "n": 5, "seed": 42})
                                .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{model}");
            assert!(
                generated_at.lock().unwrap().unwrap().elapsed() >= Duration::from_millis(50),
                "{model}"
            );
            // A response object exists, but the transport has not consumed its body yet.
            let files = std::fs::read_dir(&root).unwrap().count();
            assert_eq!(
                files, 5,
                "{model}: images must survive the delay and await body delivery"
            );
            comfyui::cleanup_stale_images_before(&root, std::time::SystemTime::now());
            assert_eq!(
                std::fs::read_dir(&root).unwrap().count(),
                5,
                "{model}: unconsumed body must protect images"
            );
            let delivered_at = std::time::SystemTime::now();
            let body = to_bytes(response.into_body(), 65536).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["data"].as_array().unwrap().len(), 5);
            assert!(json["created"].as_i64().unwrap() > 0);
            assert!(json["usage"]["total_tokens"].as_u64().unwrap() > 0);
            for image in json["data"].as_array().unwrap() {
                assert!(image["url"].as_str().unwrap().contains("/v1/images/files?"));
            }
            comfyui::cleanup_stale_images_before(&root, delivered_at - Duration::from_secs(1));
            assert_eq!(
                std::fs::read_dir(&root).unwrap().count(),
                5,
                "{model}: TTL must restart at delivery"
            );
            for file in std::fs::read_dir(&root).unwrap() {
                assert_eq!(
                    std::fs::read(file.unwrap().path()).unwrap(),
                    b"unchanged generated image"
                );
            }
            // A future sweep simulates expiry after the full post-delivery retention interval.
            assert_eq!(
                comfyui::cleanup_stale_images_before(
                    &root,
                    std::time::SystemTime::now() + Duration::from_secs(1)
                ),
                5,
                "{model}: delivered images must eventually expire"
            );
        }
        sweeper.abort();
        let _ = sweeper.await;

        // Two concurrent dynamic requests must have independent response lifetimes.
        let request = |model: &str, size: &str| {
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": model, "prompt": "cat", "n": 5, "seed": 42, "size": size}).to_string(),
                ))
                .unwrap()
        };
        let mut sibling = app.clone();
        let (first, second) = tokio::join!(
            app.call(request("IMG2.0-S", "512x512")),
            sibling.call(request("IMG2.0-S", "768x768"))
        );
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(second.status(), StatusCode::OK);
        to_bytes(first.into_body(), 65536).await.unwrap();
        assert_eq!(
            comfyui::cleanup_stale_images_before(
                &root,
                std::time::SystemTime::now() + Duration::from_secs(1)
            ),
            5,
            "a delivered sibling must expire without deleting the unconsumed response's image"
        );
        let second: serde_json::Value =
            serde_json::from_slice(&to_bytes(second.into_body(), 65536).await.unwrap()).unwrap();
        for image in second["data"].as_array().unwrap() {
            let query = image["url"].as_str().unwrap().split_once('?').unwrap().1;
            let params = serde_urlencoded::from_str(query).unwrap();
            let downloaded = comfyui::image_file_response(root.to_str().unwrap(), params)
                .await
                .unwrap();
            assert_eq!(
                to_bytes(downloaded.into_body(), 1024).await.unwrap(),
                "unchanged generated image"
            );
        }
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);

        for model in ["ordinary", "IMG2.0-S"] {
            let response = app.call(request(model, "512x512")).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            drop(response); // Disconnect before consuming the body.
            assert_eq!(
                std::fs::read_dir(&root).unwrap().count(),
                0,
                "{model}: cancelled body must clean up"
            );

            let mut cancelled_app = app.clone();
            let request = request(model, "512x512");
            let pending = tokio::spawn(async move { cancelled_app.call(request).await });
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    let published = std::fs::read_dir(&root)
                        .unwrap()
                        .flatten()
                        .filter(|entry| entry.file_name().to_string_lossy().starts_with("image-"))
                        .count();
                    if published == 5 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .unwrap();
            pending.abort(); // Disconnect during the delay, before a response exists.
            let _ = pending.await;
            assert_eq!(
                std::fs::read_dir(&root).unwrap().count(),
                0,
                "{model}: cancelled delay must clean up"
            );
        }

        let mut immediate = Router::new()
            .route("/v1/images/*path", any(proxy::proxy_handler))
            .with_state(state);
        for model in ["ordinary", "IMG2.0-S"] {
            // No delay middleware at all: response ownership must still protect images.
            let response = immediate.call(request(model, "512x512")).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                comfyui::cleanup_stale_images_before(
                    &root,
                    std::time::SystemTime::now() + Duration::from_secs(60)
                ),
                0
            );
            use futures::StreamExt;
            let mut stream = response.into_body().into_data_stream();
            let bytes = stream.next().await.unwrap().unwrap();
            // Content-Length can make the transport stop without polling an EOF.
            assert_eq!(
                comfyui::cleanup_stale_images_before(
                    &root,
                    std::time::SystemTime::now() + Duration::from_secs(60)
                ),
                0,
                "even the final chunk must stay protected until body handoff finishes"
            );
            let delivered_at = std::time::SystemTime::now();
            drop(stream);
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(comfyui::cleanup_stale_images_before(&root, delivered_at), 0);
            for image in json["data"].as_array().unwrap() {
                let query = image["url"].as_str().unwrap().split_once('?').unwrap().1;
                let params = serde_urlencoded::from_str(query).unwrap();
                let response = comfyui::image_file_response(root.to_str().unwrap(), params)
                    .await
                    .unwrap();
                assert_eq!(
                    to_bytes(response.into_body(), 1024).await.unwrap(),
                    "unchanged generated image"
                );
            }
            assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
        }
        worker.abort();
        server.abort();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn disabled_or_invalid_http_delay_preserves_immediate_success() {
        for config in [
            DelayConfig::default(),
            DelayConfig {
                enabled: true,
                min_seconds: 30.0,
                max_seconds: 15.0,
            },
        ] {
            let mut app = Router::new()
                .route(
                    "/v1/images/generations",
                    any(|| async { "original response" }),
                )
                .layer(axum::middleware::from_fn_with_state(config, response_delay));
            let start = Instant::now();
            let response = app
                .call(
                    Request::builder()
                        .uri("/v1/images/generations")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(start.elapsed(), Duration::ZERO);
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                to_bytes(response.into_body(), 1024).await.unwrap(),
                "original response"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_responses_wait_independently_without_blocking_health() {
        let config = DelayConfig {
            enabled: true,
            min_seconds: 15.375,
            max_seconds: 15.375,
        };
        let app = Router::new()
            .route("/v1/images/generations", any(|| async { "image" }))
            .route("/health", any(|| async { "healthy" }))
            .layer(axum::middleware::from_fn_with_state(config, response_delay));
        let request = || {
            Request::builder()
                .uri("/v1/images/generations")
                .body(Body::empty())
                .unwrap()
        };
        let start = Instant::now();
        let mut first = app.clone();
        let mut second = app.clone();
        let mut health = app;
        let (first, second, ()) =
            tokio::join!(first.call(request()), second.call(request()), async {
                let response = health
                    .call(
                        Request::builder()
                            .uri("/health")
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                assert_eq!(start.elapsed(), Duration::ZERO);
            });
        assert_eq!(first.unwrap().status(), StatusCode::OK);
        assert_eq!(second.unwrap().status(), StatusCode::OK);
        assert!((15.375..=15.377).contains(&start.elapsed().as_secs_f64()));
    }
}
