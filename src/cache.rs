use std::sync::Arc;
use std::time::Duration;

use moka::sync::Cache;

use crate::auth::{AuthPath, RequireAuth};

/// Organization and workspace a cached signature belongs to.
///
/// Recorded for responses served to a direct customer key so that only a key
/// of the same organization and workspace can read the signature back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureOwner {
    org_id: String,
    workspace_id: String,
}

impl SignatureOwner {
    /// `None` when either value is missing or empty: an owner is only
    /// recorded when both identify the caller.
    pub fn new(org_id: Option<&str>, workspace_id: Option<&str>) -> Option<Self> {
        let org_id = org_id.filter(|value| !value.is_empty())?;
        let workspace_id = workspace_id.filter(|value| !value.is_empty())?;
        Some(Self {
            org_id: org_id.to_string(),
            workspace_id: workspace_id.to_string(),
        })
    }

    /// Owner of a direct customer key request. `None` for any other caller.
    pub fn from_auth(auth: &RequireAuth) -> Option<Self> {
        if auth.request_source.auth_path != AuthPath::CloudApiKey {
            return None;
        }
        Self::new(auth.org_id.as_deref(), auth.workspace_id.as_deref())
    }
}

/// Who is reading a cached signature.
#[derive(Debug, Clone, Copy)]
pub enum SignatureReader<'a> {
    /// A configured `TOKEN` caller. Reads every entry.
    Trusted,
    /// A direct customer key. Reads only entries recorded for the same owner;
    /// entries without an owner are refused.
    Owner(&'a SignatureOwner),
}

impl<'a> SignatureReader<'a> {
    /// The reader for an authenticated request. `None` for a direct key whose
    /// owner cannot be determined; such a caller reads nothing.
    pub fn for_auth(auth: &RequireAuth, owner: Option<&'a SignatureOwner>) -> Option<Self> {
        match auth.request_source.auth_path {
            AuthPath::TrustedConfigToken => Some(Self::Trusted),
            AuthPath::CloudApiKey => owner.map(Self::Owner),
        }
    }
}

struct CachedSignature {
    owner: Option<SignatureOwner>,
    value: String,
}

/// In-memory cache with TTL for chat signatures.
pub struct ChatCache {
    inner: Cache<String, Arc<CachedSignature>>,
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

    /// Store a signature. `owner` is the direct-key owner of the response, or
    /// `None` for a response served to a configured `TOKEN` caller.
    pub fn set_chat(&self, chat_id: &str, value: &str, owner: Option<SignatureOwner>) {
        let key = self.make_key(chat_id);
        self.inner.insert(
            key,
            Arc::new(CachedSignature {
                owner,
                value: value.to_string(),
            }),
        );
        // moka::sync::Cache::entry_count() is O(1) (atomic counter), safe to call per insert
        metrics::gauge!("cache_size").set(self.inner.entry_count() as f64);
    }

    /// Read a signature for `reader`. A refused entry is reported exactly
    /// like a missing one so a caller cannot tell the two apart.
    pub fn get_chat(&self, chat_id: &str, reader: SignatureReader<'_>) -> Option<String> {
        let key = self.make_key(chat_id);
        let Some(entry) = self.inner.get(&key) else {
            metrics::counter!("cache_misses_total").increment(1);
            return None;
        };
        let allowed = match (reader, entry.owner.as_ref()) {
            (SignatureReader::Trusted, _) => true,
            (SignatureReader::Owner(reader), Some(owner)) => {
                if reader == owner {
                    true
                } else {
                    metrics::counter!(
                        "signature_lookup_denied_total",
                        "reason" => "owner_mismatch"
                    )
                    .increment(1);
                    false
                }
            }
            (SignatureReader::Owner(_), None) => {
                metrics::counter!("signature_lookup_denied_total", "reason" => "unowned")
                    .increment(1);
                false
            }
        };
        if allowed {
            metrics::counter!("cache_hits_total").increment(1);
            Some(entry.value.clone())
        } else {
            metrics::counter!("cache_misses_total").increment(1);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner(org: &str, workspace: &str) -> SignatureOwner {
        SignatureOwner::new(Some(org), Some(workspace)).unwrap()
    }

    #[test]
    fn test_cache_set_and_get() {
        let cache = ChatCache::new("test-model", 1200);

        cache.set_chat("chat-123", r#"{"text":"hello"}"#, None);
        let result = cache.get_chat("chat-123", SignatureReader::Trusted);

        assert_eq!(result, Some(r#"{"text":"hello"}"#.to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let cache = ChatCache::new("test-model", 1200);
        assert_eq!(
            cache.get_chat("nonexistent", SignatureReader::Trusted),
            None
        );
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

        cache.set_chat("chat-1", "value1", None);
        cache.set_chat("chat-1", "value2", None);

        assert_eq!(
            cache.get_chat("chat-1", SignatureReader::Trusted),
            Some("value2".to_string())
        );
    }

    #[test]
    fn test_cache_different_keys() {
        let cache = ChatCache::new("test-model", 1200);

        cache.set_chat("chat-1", "value1", None);
        cache.set_chat("chat-2", "value2", None);

        assert_eq!(
            cache.get_chat("chat-1", SignatureReader::Trusted),
            Some("value1".to_string())
        );
        assert_eq!(
            cache.get_chat("chat-2", SignatureReader::Trusted),
            Some("value2".to_string())
        );
    }

    #[test]
    fn test_cache_ttl_expiry() {
        // Use a 1-second TTL
        let cache = ChatCache::new("test-model", 1);

        cache.set_chat("chat-1", "value", None);
        assert!(cache.get_chat("chat-1", SignatureReader::Trusted).is_some());

        // Sleep past TTL
        std::thread::sleep(Duration::from_secs(2));

        // moka may need a sync to evict
        cache.inner.run_pending_tasks();

        assert_eq!(cache.get_chat("chat-1", SignatureReader::Trusted), None);
    }

    #[test]
    fn test_cache_model_isolation() {
        let cache_a = ChatCache::new("model-a", 1200);
        let cache_b = ChatCache::new("model-b", 1200);

        cache_a.set_chat("chat-1", "from-a", None);
        // cache_b should not see cache_a's entries (different key namespace)
        assert_eq!(cache_b.get_chat("chat-1", SignatureReader::Trusted), None);
    }

    #[test]
    fn owner_reads_its_own_entry() {
        let cache = ChatCache::new("test-model", 1200);
        let a = owner("org-a", "ws-a");
        cache.set_chat("chat-1", "value", Some(a.clone()));

        assert_eq!(
            cache.get_chat("chat-1", SignatureReader::Owner(&a)),
            Some("value".to_string())
        );
    }

    #[test]
    fn other_owner_is_refused() {
        let cache = ChatCache::new("test-model", 1200);
        cache.set_chat("chat-1", "value", Some(owner("org-a", "ws-a")));

        let other_org = owner("org-b", "ws-a");
        let other_workspace = owner("org-a", "ws-b");
        assert_eq!(
            cache.get_chat("chat-1", SignatureReader::Owner(&other_org)),
            None
        );
        assert_eq!(
            cache.get_chat("chat-1", SignatureReader::Owner(&other_workspace)),
            None
        );
    }

    #[test]
    fn unowned_entry_is_refused_to_an_owner() {
        let cache = ChatCache::new("test-model", 1200);
        cache.set_chat("chat-1", "value", None);

        let a = owner("org-a", "ws-a");
        assert_eq!(cache.get_chat("chat-1", SignatureReader::Owner(&a)), None);
    }

    #[test]
    fn trusted_reader_reads_owned_and_unowned_entries() {
        let cache = ChatCache::new("test-model", 1200);
        cache.set_chat("owned", "value-owned", Some(owner("org-a", "ws-a")));
        cache.set_chat("unowned", "value-unowned", None);

        assert_eq!(
            cache.get_chat("owned", SignatureReader::Trusted),
            Some("value-owned".to_string())
        );
        assert_eq!(
            cache.get_chat("unowned", SignatureReader::Trusted),
            Some("value-unowned".to_string())
        );
    }

    #[test]
    fn owner_requires_both_values() {
        assert!(SignatureOwner::new(Some("org"), Some("ws")).is_some());
        assert!(SignatureOwner::new(None, Some("ws")).is_none());
        assert!(SignatureOwner::new(Some("org"), None).is_none());
        assert!(SignatureOwner::new(Some(""), Some("ws")).is_none());
        assert!(SignatureOwner::new(Some("org"), Some("")).is_none());
    }
}
