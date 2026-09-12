//! Conversation affinity across independent backends in one `BackendPool`.
//!
//! Every backend URL in `VLLM_BACKEND_URLS` (or discovered via
//! `VLLM_BACKEND_DISCOVERY_URL`) is a separate engine with its own prefix
//! cache. Least-connections selection is free to send consecutive turns of one
//! chat conversation to different backends, so a long, append-only
//! conversation (tens or hundreds of thousands of prompt tokens) is re-prefilled
//! from scratch on every switch. When enabled, the proxy hashes the stable
//! beginning of a conversation — the same digest `vllm_dp_affinity` uses for
//! data-parallel ranks — and keeps later turns on the backend that already
//! holds the prefix in its cache.
//!
//! Affinity never overrides health, and it is bounded: if the pinned backend
//! carries more than `max_imbalance` extra in-flight requests compared to the
//! least-loaded healthy backend, the turn is rebalanced to the least-loaded
//! backend and the conversation is re-pinned there. Pins are keyed by backend
//! URL, so they survive membership changes: a conversation whose backend was
//! removed by discovery is simply re-homed on its next turn.

use std::sync::Arc;
use std::time::Duration;

use moka::sync::Cache;
use serde_json::Value;

use crate::backend_pool::{BackendGuard, BackendPool};
use crate::error::AppError;
use crate::vllm_dp_affinity::conversation_key;

const MAX_AFFINITY_ASSIGNMENTS: u64 = 100_000;

/// Opaque conversation digest used as the affinity key. Contains no prompt
/// content (salted SHA-256, see `vllm_dp_affinity::conversation_key`).
pub type ConversationKey = [u8; 32];

/// Bounded, process-salted, digest-only mapping from a conversation prefix to
/// a backend base URL in the pool.
pub struct BackendConversationAffinity {
    enabled: bool,
    max_imbalance: u32,
    assignments: Cache<ConversationKey, Arc<str>>,
    affinity_salt: [u8; 32],
}

impl BackendConversationAffinity {
    /// `enabled` is the operator flag. With a single-backend pool there is
    /// nothing to pin to and selection degrades to that backend; the flag is
    /// kept so a pool that later grows through discovery starts pinning.
    pub fn new(enabled: bool, max_imbalance: u32, ttl_secs: u64) -> Self {
        let assignments = Cache::builder()
            .max_capacity(MAX_AFFINITY_ASSIGNMENTS)
            .time_to_idle(Duration::from_secs(ttl_secs.max(1)))
            .build();
        Self {
            enabled,
            max_imbalance,
            assignments,
            affinity_salt: rand::random(),
        }
    }

    /// Whether the operator enabled affinity.
    pub fn is_active(&self) -> bool {
        self.enabled
    }

    /// Derive the affinity key for a chat-completions request body, or `None`
    /// when affinity is inactive or the body is not a chat conversation.
    pub fn key_for_chat_request(
        &self,
        request: &Value,
        deployed_model_name: &str,
    ) -> Option<ConversationKey> {
        if !self.enabled {
            return None;
        }
        conversation_key(request, deployed_model_name, &self.affinity_salt)
    }

    /// Pick a backend for `key` and return `(full_url, guard)`, exactly like
    /// `BackendPool::select_url`. Without a key this is plain least-connections.
    pub fn select_url(
        &self,
        pool: &BackendPool,
        key: Option<ConversationKey>,
        path: &str,
    ) -> Result<(String, BackendGuard), AppError> {
        let Some(key) = key else {
            return pool.select_url(path);
        };

        let existing = self.assignments.get(&key);
        let selection = pool
            .select_with_preference(existing.as_deref(), self.max_imbalance)
            .ok_or(AppError::NoBackendsAvailable)?;
        if existing.as_deref() != Some(selection.backend.base_url.as_str()) {
            // New conversation, or the pinned backend was unhealthy/overloaded/
            // removed: remember where this turn actually went so the next turn
            // follows the prefix cache that is being built there.
            self.assignments
                .insert(key, Arc::from(selection.backend.base_url.as_str()));
        }

        metrics::counter!(
            "backend_affinity_lookups_total",
            "outcome" => if existing.is_some() { "hit" } else { "miss" }
        )
        .increment(1);
        metrics::counter!(
            "backend_affinity_selections_total",
            "outcome" => selection.outcome.as_str(),
            "backend" => selection.index.to_string()
        )
        .increment(1);
        metrics::gauge!("backend_affinity_assignments").set(self.assignments.entry_count() as f64);

        let url = selection.backend.url(path);
        let guard = BackendGuard::new(selection.backend);
        Ok((url, guard))
    }

    /// Current pinned backend URL for a key (tests and diagnostics).
    #[cfg(test)]
    fn assignment(&self, key: &ConversationKey) -> Option<String> {
        self.assignments.get(key).map(|s| s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn two_backend_pool() -> BackendPool {
        BackendPool::new(vec![
            "http://b1:8000".to_string(),
            "http://b2:8000".to_string(),
        ])
    }

    fn turn(n: usize) -> Value {
        let mut messages = vec![
            serde_json::json!({"role": "system", "content": "Be concise"}),
            serde_json::json!({"role": "user", "content": "Find the weather"}),
        ];
        for i in 0..n {
            messages
                .push(serde_json::json!({"role": "assistant", "content": format!("Sunny {i}")}));
            messages.push(serde_json::json!({"role": "user", "content": format!("And day {i}?")}));
        }
        serde_json::json!({"model": "client-alias", "messages": messages})
    }

    #[test]
    fn disabled_yields_no_key() {
        let chat = turn(0);
        assert!(BackendConversationAffinity::new(false, 8, 1_200)
            .key_for_chat_request(&chat, "model")
            .is_none());
        assert!(!BackendConversationAffinity::new(false, 8, 1_200).is_active());
        assert!(BackendConversationAffinity::new(true, 8, 1_200).is_active());
        assert!(BackendConversationAffinity::new(true, 8, 1_200)
            .key_for_chat_request(&chat, "model")
            .is_some());
    }

    #[test]
    fn later_turns_stay_on_the_first_turn_backend() {
        let pool = two_backend_pool();
        let affinity = BackendConversationAffinity::new(true, 8, 1_200);

        // Turn 1: both idle → least-connections picks b1.
        let key = affinity.key_for_chat_request(&turn(0), "model");
        let (url, guard) = affinity
            .select_url(&pool, key, "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b1:8000/v1/chat/completions");
        drop(guard);

        // b1 is now busier than b2 but within the imbalance bound: turn 2 must
        // still follow the prefix cache on b1 (least-connections alone would
        // have moved it to b2).
        pool.backends()[0].active_conns.store(5, Ordering::Relaxed);
        let key = affinity.key_for_chat_request(&turn(1), "model");
        let (url, _guard) = affinity
            .select_url(&pool, key, "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b1:8000/v1/chat/completions");
        assert_eq!(
            affinity.assignment(&key.unwrap()).as_deref(),
            Some("http://b1:8000")
        );
    }

    #[test]
    fn new_conversations_balance_by_least_connections() {
        let pool = two_backend_pool();
        let affinity = BackendConversationAffinity::new(true, 8, 1_200);
        pool.backends()[0].active_conns.store(3, Ordering::Relaxed);

        let other = serde_json::json!({
            "messages": [{"role": "user", "content": "a different conversation"}]
        });
        let key = affinity.key_for_chat_request(&other, "model");
        let (url, _guard) = affinity
            .select_url(&pool, key, "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b2:8000/v1/chat/completions");
        assert_eq!(
            affinity.assignment(&key.unwrap()).as_deref(),
            Some("http://b2:8000")
        );
    }

    #[test]
    fn overloaded_pin_is_rebalanced_and_repinned() {
        let pool = two_backend_pool();
        let affinity = BackendConversationAffinity::new(true, 4, 1_200);

        let key = affinity.key_for_chat_request(&turn(0), "model").unwrap();
        let (url, guard) = affinity
            .select_url(&pool, Some(key), "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b1:8000/v1/chat/completions");
        drop(guard);

        // b1 has 5 more in-flight requests than b2 (> max_imbalance 4).
        pool.backends()[0].active_conns.store(5, Ordering::Relaxed);
        let (url, guard) = affinity
            .select_url(&pool, Some(key), "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b2:8000/v1/chat/completions");
        assert_eq!(affinity.assignment(&key).as_deref(), Some("http://b2:8000"));
        drop(guard);

        // Once re-pinned, the conversation follows its new home even after b1
        // becomes idle again.
        pool.backends()[0].active_conns.store(0, Ordering::Relaxed);
        pool.backends()[1].active_conns.store(2, Ordering::Relaxed);
        let (url, _guard) = affinity
            .select_url(&pool, Some(key), "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b2:8000/v1/chat/completions");
    }

    #[test]
    fn unhealthy_pin_falls_back_to_a_healthy_backend() {
        let pool = two_backend_pool();
        let affinity = BackendConversationAffinity::new(true, 8, 1_200);

        let key = affinity.key_for_chat_request(&turn(0), "model").unwrap();
        let (url, guard) = affinity
            .select_url(&pool, Some(key), "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b1:8000/v1/chat/completions");
        drop(guard);

        pool.backends()[0].healthy.store(false, Ordering::Relaxed);
        let (url, _guard) = affinity
            .select_url(&pool, Some(key), "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b2:8000/v1/chat/completions");
        assert_eq!(affinity.assignment(&key).as_deref(), Some("http://b2:8000"));
    }

    #[test]
    fn departed_pin_is_rehomed_after_membership_change() {
        let pool = two_backend_pool();
        let affinity = BackendConversationAffinity::new(true, 8, 1_200);

        let key = affinity.key_for_chat_request(&turn(0), "model").unwrap();
        let (url, guard) = affinity
            .select_url(&pool, Some(key), "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b1:8000/v1/chat/completions");
        drop(guard);

        // Discovery replaces b1 with b3: the pin still names b1 (departed), so
        // the next turn goes least-connections and is re-pinned there.
        pool.set_backends(vec![
            ("http://b2:8000".to_string(), true),
            ("http://b3:8000".to_string(), true),
        ]);
        pool.backends()[0].active_conns.store(1, Ordering::Relaxed);
        let (url, _guard) = affinity
            .select_url(&pool, Some(key), "/v1/chat/completions")
            .unwrap();
        assert_eq!(url, "http://b3:8000/v1/chat/completions");
        assert_eq!(affinity.assignment(&key).as_deref(), Some("http://b3:8000"));
    }

    #[test]
    fn without_a_key_selection_is_plain_least_connections() {
        let pool = two_backend_pool();
        let affinity = BackendConversationAffinity::new(true, 8, 1_200);
        pool.backends()[0].active_conns.store(1, Ordering::Relaxed);
        let (url, _guard) = affinity.select_url(&pool, None, "/v1/completions").unwrap();
        assert_eq!(url, "http://b2:8000/v1/completions");
    }

    #[test]
    fn empty_pool_is_a_typed_error() {
        let pool = BackendPool::new(Vec::new());
        let affinity = BackendConversationAffinity::new(true, 8, 1_200);
        let key = affinity.key_for_chat_request(&turn(0), "model");
        assert!(matches!(
            affinity.select_url(&pool, key, "/v1/chat/completions"),
            Err(AppError::NoBackendsAvailable)
        ));
        assert!(matches!(
            affinity.select_url(&pool, None, "/v1/chat/completions"),
            Err(AppError::NoBackendsAvailable)
        ));
    }
}
