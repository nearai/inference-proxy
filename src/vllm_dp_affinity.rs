//! Stable conversation affinity for vLLM data-parallel replicas.
//!
//! vLLM keeps a separate prefix cache in every data-parallel engine. Its
//! default load balancer is intentionally free to send consecutive requests
//! to different ranks, so append-only chat turns can repeatedly miss an
//! otherwise reusable prefix. When configured, the proxy hashes the stable
//! beginning of a conversation and pins every turn to the same vLLM rank via
//! `X-data-parallel-rank`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use moka::sync::Cache;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Header understood by vLLM's OpenAI serving layer.
pub const DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

const MAX_AFFINITY_ASSIGNMENTS: u64 = 100_000;

/// Bounded, process-salted digest-only mapping from a conversation prefix to a
/// vLLM rank.
///
/// New conversations are assigned round-robin to preserve load balance. Later
/// turns reuse the cached assignment. The map stores no prompt content and
/// expires at the same cadence as other proxy chat state.
pub struct VllmDpAffinity {
    data_parallel_size: Option<usize>,
    assignments: Cache<[u8; 32], usize>,
    next_rank: AtomicUsize,
    affinity_salt: [u8; 32],
}

impl VllmDpAffinity {
    pub fn new(data_parallel_size: Option<usize>, ttl_secs: u64) -> Self {
        let assignments = Cache::builder()
            .max_capacity(MAX_AFFINITY_ASSIGNMENTS)
            .time_to_idle(Duration::from_secs(ttl_secs.max(1)))
            .build();
        Self {
            data_parallel_size: data_parallel_size.filter(|size| *size > 0),
            assignments,
            next_rank: AtomicUsize::new(0),
            affinity_salt: rand::random(),
        }
    }

    pub fn rank_for_chat_request(
        &self,
        request: &Value,
        deployed_model_name: &str,
    ) -> Option<usize> {
        let data_parallel_size = self.data_parallel_size?;
        let key = conversation_key(request, deployed_model_name, &self.affinity_salt)?;
        let existing = self.assignments.get(&key);
        let rank = existing.unwrap_or_else(|| {
            self.assignments.get_with(key, || {
                self.next_rank.fetch_add(1, Ordering::Relaxed) % data_parallel_size
            })
        });
        metrics::counter!(
            "vllm_dp_affinity_assignment_lookups_total",
            "outcome" => if existing.is_some() { "hit" } else { "miss" }
        )
        .increment(1);
        metrics::gauge!("vllm_dp_affinity_assignments").set(self.assignments.entry_count() as f64);
        Some(rank)
    }
}

/// Return a stable digest for an append-only chat conversation.
///
/// The affinity key contains only prompt-shaping fields that should remain
/// stable between turns: the deployed model name, leading system/developer
/// messages through the first user message, and optional tool/template inputs.
/// Later assistant and user turns are deliberately excluded. The digest uses a
/// process-random salt, stays in memory only, and is never logged or persisted.
fn conversation_key(
    request: &Value,
    deployed_model_name: &str,
    affinity_salt: &[u8; 32],
) -> Option<[u8; 32]> {
    let messages = request.get("messages")?.as_array()?;
    let first_user_index = messages.iter().position(|message| {
        message
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|role| role.eq_ignore_ascii_case("user"))
    })?;

    let mut key = Map::new();
    key.insert(
        "model".to_string(),
        Value::String(deployed_model_name.to_string()),
    );
    key.insert(
        "messages".to_string(),
        Value::Array(messages[..=first_user_index].to_vec()),
    );

    // These fields can affect the rendered prompt before the first user turn.
    // Include them when present so unrelated prompt templates do not share an
    // affinity bucket merely because their initial message text matches.
    // `tool_choice` is deliberately excluded: it selects how the model may
    // answer but does not change the reusable messages/tools prefix, and agent
    // loops commonly change it between otherwise append-only turns.
    for field in [
        "tools",
        "chat_template",
        "chat_template_kwargs",
        "documents",
        "mm_processor_kwargs",
    ] {
        if let Some(value) = request.get(field) {
            key.insert(field.to_string(), value.clone());
        }
    }

    let encoded = serde_json::to_vec(&Value::Object(key)).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(affinity_salt);
    hasher.update(encoded);
    let digest = hasher.finalize();
    Some(digest.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_only_turns_keep_the_same_rank() {
        let affinity = VllmDpAffinity::new(Some(4), 1_200);
        let first = serde_json::json!({
            "model": "client-alias",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Find the weather"}
            ]
        });
        let later = serde_json::json!({
            "model": "another-client-alias",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Find the weather"},
                {"role": "assistant", "content": "Sunny"},
                {"role": "user", "content": "And tomorrow?"}
            ]
        });

        assert_eq!(
            affinity.rank_for_chat_request(&first, "deployed-model"),
            affinity.rank_for_chat_request(&later, "deployed-model")
        );
    }

    #[test]
    fn prompt_shaping_fields_change_the_affinity_key() {
        let base = serde_json::json!({
            "messages": [{"role": "user", "content": "same prompt"}],
            "tools": [{"type": "function", "function": {"name": "alpha"}}]
        });
        let mut changed = base.clone();
        changed["tools"] = serde_json::json!([
            {"type": "function", "function": {"name": "beta"}}
        ]);

        assert_ne!(
            conversation_key(&base, "model", &[0; 32]),
            conversation_key(&changed, "model", &[0; 32])
        );
    }

    #[test]
    fn tool_choice_changes_keep_the_same_affinity_key() {
        let base = serde_json::json!({
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        });
        let mut automatic = base.clone();
        automatic["tool_choice"] = serde_json::json!("auto");
        let mut forced = base.clone();
        forced["tool_choice"] = serde_json::json!({
            "type": "function",
            "function": {"name": "lookup"}
        });

        let base_key = conversation_key(&base, "model", &[0; 32]);
        assert_eq!(base_key, conversation_key(&automatic, "model", &[0; 32]));
        assert_eq!(base_key, conversation_key(&forced, "model", &[0; 32]));
    }

    #[test]
    fn disabled_or_non_chat_requests_are_not_pinned() {
        let chat = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        let disabled = VllmDpAffinity::new(None, 1_200);
        let invalid = VllmDpAffinity::new(Some(0), 1_200);
        let enabled = VllmDpAffinity::new(Some(4), 1_200);
        assert_eq!(disabled.rank_for_chat_request(&chat, "model"), None);
        assert_eq!(invalid.rank_for_chat_request(&chat, "model"), None);
        assert_eq!(
            enabled.rank_for_chat_request(&serde_json::json!({}), "model"),
            None
        );
        assert_eq!(
            enabled.rank_for_chat_request(
                &serde_json::json!({"messages": [{"role": "assistant", "content": "hi"}]}),
                "model"
            ),
            None
        );
    }

    #[test]
    fn diverse_initial_prompts_spread_across_ranks() {
        let affinity = VllmDpAffinity::new(Some(4), 1_200);
        let mut counts = [0usize; 4];
        for index in 0..4_000 {
            let request = serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": format!("synthetic conversation {index}")
                }]
            });
            let rank = affinity.rank_for_chat_request(&request, "model").unwrap();
            counts[rank] += 1;
        }

        assert_eq!(counts, [1_000; 4]);
    }

    #[test]
    fn concurrent_identical_first_turns_get_one_assignment() {
        let affinity = std::sync::Arc::new(VllmDpAffinity::new(Some(4), 1_200));
        let request = serde_json::json!({
            "messages": [{"role": "user", "content": "same conversation"}]
        });
        let handles: Vec<_> = (0..32)
            .map(|_| {
                let affinity = affinity.clone();
                let request = request.clone();
                std::thread::spawn(move || {
                    affinity.rank_for_chat_request(&request, "model").unwrap()
                })
            })
            .collect();
        let ranks: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert!(ranks.iter().all(|rank| *rank == ranks[0]));
    }
}
