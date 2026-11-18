//! The one step every route that forwards a client JSON body to an engine
//! runs on that body, after decryption and before dispatch:
//!
//! 1. the request field rules (`request_guard.rs`): engine-internal fields
//!    stripped, bounds, the structured-output walker, the media URL scheme,
//!    token id bounds;
//! 2. media inlining (`media_inline.rs`): with `VLLM_PROXY_INLINE_MEDIA` on,
//!    every remote image or audio URL becomes a `data:` URL, and anything
//!    that cannot be inlined is refused, so no URL reaches the engine.
//!
//! Both halves find media with the same walker (`media_refs.rs`). The chat
//! route runs the two halves at separate points (inlining after its admission
//! precheck, so a request the lane cannot take fetches nothing); every other
//! route calls [`prepare`]. Which routes do what is declared in
//! `routes::POST_ROUTES`.

use serde_json::Value;

use crate::config::Config;
use crate::error::AppError;

/// Which field rules a body gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rules {
    /// Chat, text completions and tokenize: every rule, sampling included.
    Completion,
    /// Embeddings, rerank, score, image generation and privacy classify: no
    /// sampling rules (`n` there is the image count, bounded by the image
    /// rules instead).
    Passthrough,
}

/// Run the field rules and media inlining on `body`. Returns whether the body
/// changed, in which case the caller forwards the re-serialized body instead
/// of the bytes it received.
pub async fn prepare(config: &Config, rules: Rules, body: &mut Value) -> Result<bool, AppError> {
    let guard = &config.request_guard;
    let stripped = match rules {
        Rules::Completion => guard.apply_completion_body(body)?,
        Rules::Passthrough => guard.apply_passthrough_body(body)?,
    };
    let inlined = crate::media_inline::inline_remote_media(body, &config.media_inline()).await?;
    Ok(stripped || inlined)
}
