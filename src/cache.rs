use std::time::Duration;

use moka::sync::Cache;

/// In-memory cache with TTL for chat signatures.
pub struct ChatCache {
    inner: Cache<String, String>,
    /// Pre-computed key prefix: "{model_name}:chat:"
    key_prefix: String,
}

impl ChatCache {
    pub fn new(model_name: &str, ttl_secs: u64) -> Self {
        let inner = Cache::builder()
            .max_capacity(10_000)
            .time_to_live(Duration::from_secs(ttl_secs))
            .build();

        ChatCache {
            inner,
            key_prefix: format!("{model_name}:chat:"),
        }
    }

    fn make_key(&self, chat_id: &str) -> String {
        let mut key = String::with_capacity(self.key_prefix.len() + chat_id.len());
        key.push_str(&self.key_prefix);
        key.push_str(chat_id);
        key
    }

    pub fn set_chat(&self, chat_id: &str, value: &str) {
        let key = self.make_key(chat_id);
        self.inner.insert(key, value.to_string());
        // moka::sync::Cache::entry_count() is O(1) (atomic counter), safe to call per insert
        metrics::gauge!("cache_size").set(self.inner.entry_count() as f64);
    }

    pub fn get_chat(&self, chat_id: &str) -> Option<String> {
        let key = self.make_key(chat_id);
        let result = self.inner.get(&key);
        if result.is_some() {
            metrics::counter!("cache_hits_total").increment(1);
        } else {
            metrics::counter!("cache_misses_total").increment(1);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_set_and_get() {
        let cache = ChatCache::new("test-model", 1200);

        cache.set_chat("chat-123", r#"{"text":"hello"}"#);
        let result = cache.get_chat("chat-123");

        assert_eq!(result, Some(r#"{"text":"hello"}"#.to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let cache = ChatCache::new("test-model", 1200);
        assert_eq!(cache.get_chat("nonexistent"), None);
    }

    #[test]
    fn test_cache_key_format() {
        let cache = ChatCache::new("my-model", 1200);

        // Verify key is namespaced
        let key = cache.make_key("abc");
        assert_eq!(key, "my-model:chat:abc");
    }

    #[test]
    fn test_cache_overwrite() {
        let cache = ChatCache::new("test-model", 1200);

        cache.set_chat("chat-1", "value1");
        cache.set_chat("chat-1", "value2");

        assert_eq!(cache.get_chat("chat-1"), Some("value2".to_string()));
    }

    #[test]
    fn test_cache_different_keys() {
        let cache = ChatCache::new("test-model", 1200);

        cache.set_chat("chat-1", "value1");
        cache.set_chat("chat-2", "value2");

        assert_eq!(cache.get_chat("chat-1"), Some("value1".to_string()));
        assert_eq!(cache.get_chat("chat-2"), Some("value2".to_string()));
    }

    #[test]
    fn test_cache_ttl_expiry() {
        // Use a 1-second TTL
        let cache = ChatCache::new("test-model", 1);

        cache.set_chat("chat-1", "value");
        assert!(cache.get_chat("chat-1").is_some());

        // Sleep past TTL
        std::thread::sleep(Duration::from_secs(2));

        // moka may need a sync to evict
        cache.inner.run_pending_tasks();

        assert_eq!(cache.get_chat("chat-1"), None);
    }

    #[test]
    fn test_cache_model_isolation() {
        let cache_a = ChatCache::new("model-a", 1200);
        let cache_b = ChatCache::new("model-b", 1200);

        cache_a.set_chat("chat-1", "from-a");
        // cache_b should not see cache_a's entries (different key namespace)
        assert_eq!(cache_b.get_chat("chat-1"), None);
    }
}
