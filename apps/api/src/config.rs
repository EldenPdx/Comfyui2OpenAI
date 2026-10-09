//! Configuration management for the ComfyUI OpenAI API proxy
//!
//! This module handles loading and validating configuration from YAML files.
//! Configuration can be customized via the CONFIG_PATH environment variable.

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;

/// Proxy server configuration
///
/// Defines where the proxy server listens for incoming OpenAI API requests
#[derive(Debug, Deserialize, Serialize)]
pub struct ServerConfig {
    /// Server bind address (e.g., "0.0.0.0", "127.0.0.1")
    pub host: String,
    /// Server bind port (e.g., 8080)
    pub port: u16,
}

/// ComfyUI backend connection configuration
///
/// Defines how to connect to the ComfyUI server and which workflows to use
#[derive(Debug, Deserialize, Serialize)]
pub struct ComfyUiProxyConfig {
    /// ComfyUI backend host address
    pub host: String,
    /// ComfyUI backend port
    pub port: u16,
    /// Additional ComfyUI backend ports on the same host
    #[serde(default)]
    pub additional_ports: Vec<u16>,
    /// Unique client identifier for WebSocket connections to ComfyUI
    pub client_id: String,
    /// Path to directory containing workflow JSON files
    pub workflows_folder: String,
    /// Whether to use or not WebSockets
    pub use_ws: bool,
}

/// Generated image cleanup configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageCleanupConfig {
    /// ComfyUI output directory containing generated files.
    pub output_directory: String,
    /// Periodic stale-file sweep interval.
    pub interval_seconds: u64,
    /// Retention after response-body handoff; orphan files use their modification time.
    pub max_age_seconds: u64,
}

/// Request routing and timeout configuration
///
/// Controls request handling behavior and limits
#[derive(Debug, Deserialize, Serialize)]
pub struct RoutingConfig {
    /// Timeout in seconds for requests to ComfyUI backend
    pub timeout_seconds: u16,
    /// Maximum request body size in megabytes
    pub max_payload_size_mb: u16,
}

/// Root configuration struct loaded from YAML
///
/// Example config.yaml:
/// ```yaml
/// log_level: debug
/// server:
///   host: "0.0.0.0"
///   port: 8080
/// comfyui_backend:
///   host: "localhost"
///   port: 8188
///   additional_ports: [8189]
///   client_id: "openai-proxy-client"
///   workflows_folder: "./workflows"
/// routing:
///   timeout_seconds: 120
///   max_payload_size_mb: 10
/// ```
#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    /// Logging level: debug, info, warn, error
    pub log_level: String,
    /// Proxy server configuration
    pub server: ServerConfig,
    /// ComfyUI backend configuration
    pub comfyui_backend: ComfyUiProxyConfig,
    /// Request routing configuration
    pub routing: RoutingConfig,
    /// Generated image cleanup configuration
    pub image_cleanup: ImageCleanupConfig,
    /// Whether public one-time image URLs may return image bytes.
    #[serde(default)]
    pub public_image_access: bool,
    /// Optional delay after image generation and before final response delivery.
    #[serde(default)]
    pub delay_layer: crate::delay::DelayConfig,
    /// Optional request outcome monitoring.
    #[serde(default)]
    pub monitoring: crate::monitor::MonitorConfig,
    /// Optional SMTP alert delivery.
    #[serde(default)]
    pub notification: crate::notification::NotificationConfig,
}

impl Config {
    /// Load configuration from a YAML file
    ///
    /// Reads from CONFIG_PATH environment variable or defaults to "./config/config.yaml"
    ///
    /// # Examples
    /// ```no_run
    /// let config = Config::load()?;
    /// println!("Server running on {}:{}", config.server.host, config.server.port);
    /// ```
    ///
    /// # Errors
    /// Returns an error if:
    /// - The configuration file cannot be read
    /// - The YAML syntax is invalid
    /// - Required fields are missing
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path =
            env::var("CONFIG_PATH").unwrap_or_else(|_| "./config/config.yaml".to_string());

        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config file '{}': {}", config_path, e))?;

        let config: Config = serde_yaml::from_str(&config_str)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_configuration_keeps_delay_disabled() {
        let mut yaml: serde_yaml::Value =
            serde_yaml::from_str(include_str!("../config/config.sample.yaml")).unwrap();
        yaml.as_mapping_mut()
            .unwrap()
            .remove(serde_yaml::Value::from("delay_layer"));
        let config: Config = serde_yaml::from_value(yaml).unwrap();
        assert!(!config.delay_layer.enabled);
        assert!(!config.public_image_access);
        assert_eq!(config.delay_layer.min_seconds, 3.0);
        assert_eq!(config.delay_layer.max_seconds, 8.0);
    }

    #[test]
    fn delay_configuration_accepts_an_enable_switch_and_fractional_bounds() {
        let yaml = include_str!("../config/config.sample.yaml");
        let mut value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        value["delay_layer"] =
            serde_yaml::from_str("enabled: true\nmin_seconds: 1.25\nmax_seconds: 2.75").unwrap();
        let config: Config = serde_yaml::from_value(value.clone()).unwrap();
        assert!(config.delay_layer.enabled);
        assert_eq!(config.delay_layer.min_seconds, 1.25);
        assert_eq!(config.delay_layer.max_seconds, 2.75);
        value["delay_layer"] = serde_yaml::from_str("enabled: true").unwrap();
        let config: Config = serde_yaml::from_value(value).unwrap();
        assert!(config.delay_layer.enabled);
        assert_eq!(config.delay_layer.min_seconds, 3.0);
        assert_eq!(config.delay_layer.max_seconds, 8.0);
    }

    #[test]
    fn monitoring_and_notification_switches_are_independent_and_legacy_safe() {
        let mut yaml: serde_yaml::Value =
            serde_yaml::from_str(include_str!("../config/config.sample.yaml")).unwrap();
        yaml.as_mapping_mut()
            .unwrap()
            .remove(serde_yaml::Value::from("monitoring"));
        yaml.as_mapping_mut()
            .unwrap()
            .remove(serde_yaml::Value::from("notification"));
        let config: Config = serde_yaml::from_value(yaml.clone()).unwrap();
        assert!(!config.monitoring.enabled);
        assert!(!config.notification.enabled);

        yaml["monitoring"] = serde_yaml::from_str(
            "enabled: true\nfailure_rate_threshold_percent: 2.5\nsample_interval_seconds: 60\nrising_reminder_seconds: 300\nunresolved_reminder_seconds: 1800",
        )
        .unwrap();
        yaml["notification"] = serde_yaml::from_str(
            "enabled: false\nhost: smtp.qq.com\nport: 587\nusername: test@example.com\npassword_env: SMTP_PASSWORD\nfrom: test@example.com\nto: test@example.com",
        )
        .unwrap();
        let config: Config = serde_yaml::from_value(yaml).unwrap();
        assert!(config.monitoring.enabled);
        assert_eq!(config.monitoring.failure_rate_threshold_percent, 2.5);
        assert!(!config.notification.enabled);
        assert_eq!(config.notification.host, "smtp.qq.com");
        assert_eq!(config.notification.password_env, "SMTP_PASSWORD");
    }

    #[test]
    fn additional_backend_ports_are_optional() {
        let mut yaml: serde_yaml::Value =
            serde_yaml::from_str(include_str!("../config/config.sample.yaml")).unwrap();
        yaml["comfyui_backend"]
            .as_mapping_mut()
            .unwrap()
            .remove(serde_yaml::Value::from("additional_ports"));
        let config: Config = serde_yaml::from_value(yaml).unwrap();
        assert!(config.comfyui_backend.additional_ports.is_empty());
    }
}
