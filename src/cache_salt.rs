//! Engine prompt-cache namespace (`cache_salt`) for chat/completions bodies.
//!
//! The engine only reuses a cached prefix between requests that carry the same
//! `cache_salt`. The value the engine sees must therefore be chosen by a
//! trusted party, never by the client:
//!
//! - A caller authenticated with the proxy's own config token (`TOKEN`) is a
//!   trusted gateway that sets its own value. Its body is left exactly as it
//!   arrived: no field is added, removed or rewritten.
//! - A caller authenticated with a Cloud API key (`sk-`) always gets a
//!   `cache_salt` set by the proxy, whether or not it sent one. The value is a
//!   SHA-256 digest over a fixed domain string, the authenticated organization
//!   and workspace, and the client's own `cache_salt` if any. The client can
//!   still split its own traffic into separate namespaces, but every value it
//!   can reach is bound to its own organization and workspace.
//!
//! Nothing here is logged or recorded as a metric: neither the identity, nor
//! the client value, nor the digest.

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::auth::{AuthPath, RequireAuth};
use crate::error::AppError;

/// Body field the engine reads.
pub const CACHE_SALT_FIELD: &str = "cache_salt";

/// Domain separation for the digest. Bump the version to move every direct
/// caller to a fresh namespace.
const DOMAIN: &[u8] = b"nearai.inference-proxy.direct-cache-salt.v1";

/// Tags for the client's own value, so absent, a string and any other JSON
/// value never hash alike.
const CLIENT_ABSENT: u8 = 0;
const CLIENT_STRING: u8 = 1;
const CLIENT_OTHER_JSON: u8 = 2;

/// Set `cache_salt` on a chat/completions body according to who sent it.
///
/// Trusted config-token callers: no-op. Cloud API key callers: the field is
/// overwritten with [`direct_cache_salt`]. A key without an organization or a
/// workspace is refused with [`AppError::BillingUnavailable`], as for usage
/// accounting; the proxy never falls back to a shared namespace.
pub fn apply(request_json: &mut Value, auth: &RequireAuth) -> Result<(), AppError> {
    // Pass through only when both trust signals agree; anything else is
    // treated as a direct caller.
    if auth.request_source.auth_path == AuthPath::TrustedConfigToken && auth.cloud_api_key.is_none()
    {
        return Ok(());
    }
    let (Some(org), Some(workspace)) = (
        non_empty(auth.org_id.as_deref()),
        non_empty(auth.workspace_id.as_deref()),
    ) else {
        return Err(AppError::BillingUnavailable);
    };
    let Some(body) = request_json.as_object_mut() else {
        return Ok(());
    };
    let salt = direct_cache_salt(org, workspace, body.get(CACHE_SALT_FIELD));
    body.insert(CACHE_SALT_FIELD.to_string(), Value::String(salt));
    Ok(())
}

/// Hex SHA-256 over the domain, the organization, the workspace and the
/// client's own value. Each variable-length input is length-prefixed so no two
/// different inputs share an encoding. A JSON `null` counts as absent.
pub fn direct_cache_salt(org: &str, workspace: &str, client: Option<&Value>) -> String {
    let mut hasher = Sha256::new();
    write_field(&mut hasher, DOMAIN);
    write_field(&mut hasher, org.as_bytes());
    write_field(&mut hasher, workspace.as_bytes());
    match client {
        None | Some(Value::Null) => hasher.update([CLIENT_ABSENT]),
        Some(Value::String(salt)) => {
            hasher.update([CLIENT_STRING]);
            write_field(&mut hasher, salt.as_bytes());
        }
        Some(other) => {
            hasher.update([CLIENT_OTHER_JSON]);
            write_field(&mut hasher, other.to_string().as_bytes());
        }
    }
    hex::encode(hasher.finalize())
}

fn write_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{IngressRouteKind, RequestSource, CALLER_API_KEY, CALLER_UNLABELED};
    use serde_json::json;

    fn direct(org: Option<&str>, workspace: Option<&str>) -> RequireAuth {
        RequireAuth {
            cloud_api_key: Some("sk-test-synthetic".to_string()),
            org_id: org.map(str::to_string),
            workspace_id: workspace.map(str::to_string),
            api_key_id: Some("key-1".to_string()),
            request_id: None,
            request_source: RequestSource {
                auth_path: AuthPath::CloudApiKey,
                ingress_route: IngressRouteKind::Missing,
                caller: CALLER_API_KEY,
            },
        }
    }

    fn trusted() -> RequireAuth {
        RequireAuth {
            cloud_api_key: None,
            org_id: None,
            workspace_id: None,
            api_key_id: None,
            request_id: None,
            request_source: RequestSource {
                auth_path: AuthPath::TrustedConfigToken,
                ingress_route: IngressRouteKind::Missing,
                caller: CALLER_UNLABELED,
            },
        }
    }

    fn applied(body: Value, auth: &RequireAuth) -> Value {
        let mut body = body;
        apply(&mut body, auth).unwrap();
        body
    }

    #[test]
    fn direct_caller_always_gets_a_proxy_value() {
        let auth = direct(Some("org-a"), Some("ws-a"));
        let without = applied(json!({"model": "m"}), &auth);
        let salt = without[CACHE_SALT_FIELD].as_str().unwrap();
        assert_eq!(salt.len(), 64);
        assert!(salt.chars().all(|c| c.is_ascii_hexdigit()));

        let with = applied(json!({"model": "m", "cache_salt": "mine"}), &auth);
        assert_ne!(with[CACHE_SALT_FIELD], "mine");
        assert_ne!(with[CACHE_SALT_FIELD], without[CACHE_SALT_FIELD]);
        // Other fields are untouched.
        assert_eq!(with["model"], "m");
    }

    #[test]
    fn same_client_value_lands_in_different_namespaces_per_tenant() {
        let body = json!({"cache_salt": "shared"});
        let a = applied(body.clone(), &direct(Some("org-a"), Some("ws-1")));
        let b = applied(body.clone(), &direct(Some("org-b"), Some("ws-1")));
        let c = applied(body.clone(), &direct(Some("org-a"), Some("ws-2")));
        assert_ne!(a[CACHE_SALT_FIELD], b[CACHE_SALT_FIELD]);
        assert_ne!(a[CACHE_SALT_FIELD], c[CACHE_SALT_FIELD]);
        assert_ne!(b[CACHE_SALT_FIELD], c[CACHE_SALT_FIELD]);
        // Without a client value too.
        let a = applied(json!({}), &direct(Some("org-a"), Some("ws-1")));
        let b = applied(json!({}), &direct(Some("org-b"), Some("ws-1")));
        assert_ne!(a[CACHE_SALT_FIELD], b[CACHE_SALT_FIELD]);
    }

    #[test]
    fn value_is_stable_for_one_tenant_and_one_client_value() {
        let auth = direct(Some("org-a"), Some("ws-a"));
        let first = applied(json!({"cache_salt": "s1"}), &auth);
        let again = applied(json!({"cache_salt": "s1"}), &auth);
        let other = applied(json!({"cache_salt": "s2"}), &auth);
        assert_eq!(first[CACHE_SALT_FIELD], again[CACHE_SALT_FIELD]);
        assert_ne!(first[CACHE_SALT_FIELD], other[CACHE_SALT_FIELD]);
        // Re-applying the derivation to its own output does not return it.
        let twice = applied(first.clone(), &auth);
        assert_ne!(twice[CACHE_SALT_FIELD], first[CACHE_SALT_FIELD]);
    }

    #[test]
    fn inputs_are_unambiguous() {
        // Moving bytes across a boundary changes the digest.
        assert_ne!(
            direct_cache_salt("ab", "c", None),
            direct_cache_salt("a", "bc", None)
        );
        assert_ne!(
            direct_cache_salt("a", "b", Some(&json!("c"))),
            direct_cache_salt("a", "bc", None)
        );
        // Absent and null are the same; an empty string, a number and its
        // string form are each distinct.
        let absent = direct_cache_salt("o", "w", None);
        assert_eq!(absent, direct_cache_salt("o", "w", Some(&Value::Null)));
        let empty = direct_cache_salt("o", "w", Some(&json!("")));
        let number = direct_cache_salt("o", "w", Some(&json!(1)));
        let number_string = direct_cache_salt("o", "w", Some(&json!("1")));
        assert_ne!(absent, empty);
        assert_ne!(number, number_string);
        assert_ne!(number, absent);
    }

    #[test]
    fn non_string_client_values_are_replaced_not_forwarded() {
        let auth = direct(Some("org-a"), Some("ws-a"));
        for client in [
            json!(null),
            json!(7),
            json!(["x"]),
            json!({"k": "v"}),
            json!(true),
        ] {
            let body = applied(json!({"cache_salt": client.clone()}), &auth);
            let salt = body[CACHE_SALT_FIELD].as_str().unwrap();
            assert_eq!(salt.len(), 64, "{client}");
        }
    }

    #[test]
    fn direct_caller_without_an_owner_is_refused() {
        for auth in [
            direct(None, Some("ws")),
            direct(Some("org"), None),
            direct(Some(""), Some("ws")),
            direct(Some("org"), Some("")),
        ] {
            let mut body = json!({"cache_salt": "mine"});
            assert!(matches!(
                apply(&mut body, &auth),
                Err(AppError::BillingUnavailable)
            ));
        }
    }

    #[test]
    fn trusted_caller_body_is_untouched() {
        for body in [
            json!({"model": "m", "cache_salt": "gateway-derived"}),
            json!({"model": "m"}),
            json!({"model": "m", "cache_salt": null}),
        ] {
            assert_eq!(applied(body.clone(), &trusted()), body);
        }
    }

    #[test]
    fn mixed_trust_signals_are_treated_as_direct() {
        // A config-token path that somehow carries an API key is not trusted.
        let mut auth = trusted();
        auth.cloud_api_key = Some("sk-test-synthetic".to_string());
        let mut body = json!({"cache_salt": "mine"});
        assert!(matches!(
            apply(&mut body, &auth),
            Err(AppError::BillingUnavailable)
        ));
        auth.org_id = Some("org-a".to_string());
        auth.workspace_id = Some("ws-a".to_string());
        let body = applied(json!({"cache_salt": "mine"}), &auth);
        assert_eq!(
            body[CACHE_SALT_FIELD],
            direct_cache_salt("org-a", "ws-a", Some(&json!("mine")))
        );
    }

    #[test]
    fn non_object_body_is_left_alone() {
        let mut body = json!([1, 2]);
        apply(&mut body, &direct(Some("o"), Some("w"))).unwrap();
        assert_eq!(body, json!([1, 2]));
    }
}
