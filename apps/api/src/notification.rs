//! SMTP request failure notifications.

use crate::monitor::{Alert, AlertKind};
use lettre::{
    AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor,
    message::{Mailbox, header::ContentType},
    transport::smtp::authentication::Credentials,
};
use serde::{Deserialize, Serialize};
use std::{env, fmt::Display};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct NotificationConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password_env: String,
    pub from: String,
    pub to: String,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: String::new(),
            port: 587,
            username: String::new(),
            password_env: "SMTP_PASSWORD".to_string(),
            from: String::new(),
            to: String::new(),
        }
    }
}

#[derive(Clone)]
pub struct EmailNotifier<T> {
    transport: T,
    from: Mailbox,
    to: Mailbox,
}

pub type SmtpNotifier = EmailNotifier<AsyncSmtpTransport<Tokio1Executor>>;

impl<T> EmailNotifier<T> {
    fn new(transport: T, from: Mailbox, to: Mailbox) -> Self {
        Self {
            transport,
            from,
            to,
        }
    }
}

impl EmailNotifier<AsyncSmtpTransport<Tokio1Executor>> {
    pub fn from_config(config: &NotificationConfig) -> Result<Option<SmtpNotifier>, String> {
        if !config.enabled {
            return Ok(None);
        }
        let password = env::var(&config.password_env)
            .map_err(|_| format!("{} is not set", config.password_env))?;
        let from = config
            .from
            .parse()
            .map_err(|error| format!("Invalid SMTP from address: {}", error))?;
        let to = config
            .to
            .parse()
            .map_err(|error| format!("Invalid SMTP recipient address: {}", error))?;
        let transport = AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&config.host)
            .map_err(|error| format!("Invalid SMTP host: {}", error))?
            .port(config.port)
            .credentials(Credentials::new(config.username.clone(), password))
            .build();
        Ok(Some(Self::new(transport, from, to)))
    }
}

impl<T> EmailNotifier<T>
where
    T: AsyncTransport + Sync,
    T::Error: Display,
{
    pub async fn send(&self, alert: &Alert) -> Result<(), String> {
        let kind = match alert.kind {
            AlertKind::Initial => "initial threshold breach",
            AlertKind::Rising => "failure rate is still rising",
            AlertKind::Unresolved => "failure rate has not recovered",
        };
        let message = Message::builder()
            .from(self.from.clone())
            .to(self.to.clone())
            .subject("[ComfyUI API] Request failure alert")
            .header(ContentType::TEXT_PLAIN)
            .body(format!(
                "Alert: {}\nSuccessful requests: {}\nFailed requests: {}\nFailure rate: {:.2}%\n\nCheck the API and ComfyUI logs, then verify backend health and recent request errors.",
                kind,
                alert.snapshot.successful,
                alert.snapshot.failed,
                alert.failure_rate_percent,
            ))
            .map_err(|error| format!("Failed to build alert email: {}", error))?;
        self.transport
            .send(message)
            .await
            .map(|_| ())
            .map_err(|error| format!("SMTP send failed: {}", error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{Alert, AlertKind, Snapshot};
    use lettre::transport::stub::AsyncStubTransport;

    #[tokio::test]
    async fn notification_sends_actionable_alert_through_the_mail_transport() {
        let transport = AsyncStubTransport::new_ok();
        let notifier = EmailNotifier::new(
            transport.clone(),
            "test@example.com".parse().unwrap(),
            "test@example.com".parse().unwrap(),
        );
        notifier
            .send(&Alert {
                kind: AlertKind::Initial,
                snapshot: Snapshot {
                    successful: 97,
                    failed: 3,
                },
                failure_rate_percent: 3.0,
            })
            .await
            .unwrap();

        let messages = transport.messages().await;
        assert_eq!(messages.len(), 1);
        assert!(
            messages[0]
                .1
                .contains("Subject: [ComfyUI API] Request failure alert")
        );
        assert!(messages[0].1.contains("Successful requests: 97"));
        assert!(messages[0].1.contains("Failed requests: 3"));
        assert!(messages[0].1.contains("Failure rate: 3.00%"));
        assert!(messages[0].1.contains("Check the API and ComfyUI logs"));
    }
}
