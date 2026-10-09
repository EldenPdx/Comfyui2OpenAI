//! Optional request outcome monitoring.

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct MonitorConfig {
    pub enabled: bool,
    pub failure_rate_threshold_percent: f64,
    pub sample_interval_seconds: u64,
    pub rising_reminder_seconds: u64,
    pub unresolved_reminder_seconds: u64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            failure_rate_threshold_percent: 2.5,
            sample_interval_seconds: 60,
            rising_reminder_seconds: 300,
            unresolved_reminder_seconds: 1800,
        }
    }
}

#[derive(Clone, Default)]
pub struct Monitor {
    counters: Arc<Counters>,
}

#[derive(Default)]
struct Counters {
    successful: AtomicU64,
    failed: AtomicU64,
    window_successful: AtomicU64,
    window_failed: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    pub successful: u64,
    pub failed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertKind {
    Initial,
    Rising,
    Unresolved,
}

#[derive(Debug, Clone, Copy)]
pub struct Alert {
    pub kind: AlertKind,
    pub snapshot: Snapshot,
    pub failure_rate_percent: f64,
}

pub struct AlertPolicy {
    config: MonitorConfig,
    last_alert_at: Option<Instant>,
    last_alert_rate: f64,
}

impl AlertPolicy {
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            last_alert_at: None,
            last_alert_rate: 0.0,
        }
    }

    pub fn evaluate(&mut self, snapshot: Snapshot, now: Instant) -> Option<Alert> {
        let total = snapshot.successful.saturating_add(snapshot.failed);
        if total == 0 {
            return None;
        }
        let failure_rate_percent = snapshot.failed as f64 * 100.0 / total as f64;
        if failure_rate_percent <= self.config.failure_rate_threshold_percent {
            self.last_alert_at = None;
            self.last_alert_rate = 0.0;
            return None;
        }

        let kind = match self.last_alert_at {
            None => AlertKind::Initial,
            Some(last_alert_at) => {
                let elapsed = now
                    .checked_duration_since(last_alert_at)
                    .unwrap_or(Duration::ZERO);
                if failure_rate_percent > self.last_alert_rate
                    && elapsed >= Duration::from_secs(self.config.rising_reminder_seconds)
                {
                    AlertKind::Rising
                } else if elapsed >= Duration::from_secs(self.config.unresolved_reminder_seconds) {
                    AlertKind::Unresolved
                } else {
                    return None;
                }
            }
        };

        self.last_alert_at = Some(now);
        self.last_alert_rate = failure_rate_percent;
        Some(Alert {
            kind,
            snapshot,
            failure_rate_percent,
        })
    }
}

impl Monitor {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            successful: self.counters.successful.load(Ordering::Relaxed),
            failed: self.counters.failed.load(Ordering::Relaxed),
        }
    }

    pub fn record_status(&self, status: axum::http::StatusCode) {
        let (total, window) = if status.is_success() {
            (&self.counters.successful, &self.counters.window_successful)
        } else {
            (&self.counters.failed, &self.counters.window_failed)
        };
        total.fetch_add(1, Ordering::Relaxed);
        window.fetch_add(1, Ordering::Relaxed);
    }

    fn take_window(&self) -> Snapshot {
        Snapshot {
            successful: self.counters.window_successful.swap(0, Ordering::Relaxed),
            failed: self.counters.window_failed.swap(0, Ordering::Relaxed),
        }
    }
}

pub async fn monitor_request(
    State(monitor): State<Monitor>,
    request: Request,
    next: Next,
) -> Response {
    let response = next.run(request).await;
    monitor.record_status(response.status());
    response
}

pub async fn run_monitor<N, F>(monitor: Monitor, config: MonitorConfig, notify: N)
where
    N: Fn(Alert) -> F + Send + Sync + 'static,
    F: Future<Output = Result<(), String>> + Send,
{
    let mut ticker =
        tokio::time::interval(Duration::from_secs(config.sample_interval_seconds.max(1)));
    ticker.tick().await;
    let mut policy = AlertPolicy::new(config);

    loop {
        ticker.tick().await;
        let window = monitor.take_window();
        let total = window.successful.saturating_add(window.failed);
        if total == 0 {
            continue;
        }
        let failure_rate = window.failed as f64 * 100.0 / total as f64;
        let totals = monitor.snapshot();
        log::info!(
            "Request monitor: window success={}, failed={}, failure_rate={:.2}%; total success={}, failed={}",
            window.successful,
            window.failed,
            failure_rate,
            totals.successful,
            totals.failed
        );

        if let Some(alert) = policy.evaluate(window, Instant::now())
            && let Err(error) = notify(alert).await
        {
            log::error!("Failed to send request failure notification: {}", error);
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
        middleware::from_fn_with_state,
        response::Response,
        routing::get,
    };
    use tower::Service;

    #[tokio::test(start_paused = true)]
    async fn monitor_evaluates_windows_through_the_notification_boundary() {
        let monitor = Monitor::default();
        for _ in 0..97 {
            monitor.record_status(StatusCode::OK);
        }
        for _ in 0..3 {
            monitor.record_status(StatusCode::BAD_GATEWAY);
        }

        let config = MonitorConfig {
            enabled: true,
            sample_interval_seconds: 60,
            ..Default::default()
        };
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(run_monitor(monitor.clone(), config, move |alert| {
            let sender = sender.clone();
            async move {
                sender.send(alert).unwrap();
                Ok(())
            }
        }));
        tokio::task::yield_now().await;
        tokio::time::advance(std::time::Duration::from_secs(60)).await;
        tokio::task::yield_now().await;

        let alert = receiver.try_recv().unwrap();
        assert_eq!(alert.kind, AlertKind::Initial);
        assert_eq!(
            alert.snapshot,
            Snapshot {
                successful: 97,
                failed: 3,
            }
        );
        assert_eq!(
            monitor.snapshot(),
            Snapshot {
                successful: 97,
                failed: 3,
            }
        );
        task.abort();
    }

    #[test]
    fn alert_policy_only_repeats_for_rising_or_unresolved_failure_rate() {
        let config = MonitorConfig {
            enabled: true,
            failure_rate_threshold_percent: 2.5,
            sample_interval_seconds: 60,
            rising_reminder_seconds: 300,
            unresolved_reminder_seconds: 1800,
        };
        let mut policy = AlertPolicy::new(config);
        let started = std::time::Instant::now();

        let initial = policy
            .evaluate(
                Snapshot {
                    successful: 97,
                    failed: 3,
                },
                started,
            )
            .unwrap();
        assert_eq!(initial.kind, AlertKind::Initial);
        assert_eq!(initial.failure_rate_percent, 3.0);

        assert!(
            policy
                .evaluate(
                    Snapshot {
                        successful: 96,
                        failed: 4,
                    },
                    started + std::time::Duration::from_secs(299),
                )
                .is_none()
        );
        assert_eq!(
            policy
                .evaluate(
                    Snapshot {
                        successful: 96,
                        failed: 4,
                    },
                    started + std::time::Duration::from_secs(300),
                )
                .unwrap()
                .kind,
            AlertKind::Rising
        );

        assert!(
            policy
                .evaluate(
                    Snapshot {
                        successful: 96,
                        failed: 4,
                    },
                    started + std::time::Duration::from_secs(2099),
                )
                .is_none()
        );
        assert_eq!(
            policy
                .evaluate(
                    Snapshot {
                        successful: 96,
                        failed: 4,
                    },
                    started + std::time::Duration::from_secs(2100),
                )
                .unwrap()
                .kind,
            AlertKind::Unresolved
        );

        assert!(
            policy
                .evaluate(
                    Snapshot {
                        successful: 195,
                        failed: 5,
                    },
                    started + std::time::Duration::from_secs(2101),
                )
                .is_none()
        );
        assert_eq!(
            policy
                .evaluate(
                    Snapshot {
                        successful: 97,
                        failed: 3,
                    },
                    started + std::time::Duration::from_secs(2102),
                )
                .unwrap()
                .kind,
            AlertKind::Initial
        );
    }

    #[tokio::test]
    async fn middleware_records_outcomes_without_changing_responses() {
        let monitor = Monitor::default();
        let mut app = Router::new()
            .route(
                "/ok",
                get(|| async {
                    Response::builder()
                        .status(StatusCode::CREATED)
                        .header("x-result", "kept")
                        .body(Body::from("created"))
                        .unwrap()
                }),
            )
            .route("/failed", get(|| async { StatusCode::BAD_GATEWAY }))
            .layer(from_fn_with_state(monitor.clone(), monitor_request));

        let response = app
            .call(Request::builder().uri("/ok").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(response.headers()["x-result"], "kept");
        assert_eq!(
            to_bytes(response.into_body(), 1024).await.unwrap(),
            "created"
        );

        let response = app
            .call(
                Request::builder()
                    .uri("/failed")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            monitor.snapshot(),
            Snapshot {
                successful: 1,
                failed: 1,
            }
        );
    }
}
