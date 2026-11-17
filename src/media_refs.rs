//! Where media references live in a request body.
//!
//! One walker, shared by the URL-scheme rule (`request_guard.rs`) and media
//! inlining (`media_inline.rs`), so both see the same items on every route.
//!
//! The engines accept media in several shapes, and the walker finds all of
//! them by key rather than by a part's `type` (vLLM picks a part's media from
//! its keys when `type` is missing or a `uuid` is set):
//!
//! - chat content parts in `messages[*].content` (also batched lists of
//!   conversations);
//! - embeddings `input` holding chat messages, lists of them, or SGLang items
//!   such as `{"text": ..., "image": ..., "video": ...}`;
//! - scoring inputs (`query`, `queries`, `documents`, `items`, `text_1`,
//!   `text_2`, `data_1`, `data_2`) holding `{"content": [parts]}`, a list of
//!   parts, or lists of those.
//!
//! Only these input fields are walked, and inside a message only `content`:
//! a tool schema or tool-call arguments with a property named `image_url` are
//! not media.

use serde_json::{Map, Value};

/// Top-level fields that carry model input. Everything else in a body
/// (sampling fields, tools, response formats) is never treated as media.
pub const INPUT_FIELDS: &[&str] = &[
    "messages",
    "input",
    "prompt",
    "query",
    "queries",
    "documents",
    "items",
    "text_1",
    "text_2",
    "data_1",
    "data_2",
];

/// Keys that hold a media reference inside a content part or input item.
pub const MEDIA_KEYS: &[(&str, MediaKind)] = &[
    ("image_url", MediaKind::Image),
    ("input_image", MediaKind::Image),
    ("image", MediaKind::Image),
    ("image_pil", MediaKind::Image),
    ("audio_url", MediaKind::Audio),
    ("video_url", MediaKind::Video),
    ("video", MediaKind::Video),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Image,
    Audio,
    Video,
}

impl MediaKind {
    pub fn label(self) -> &'static str {
        match self {
            MediaKind::Image => "image",
            MediaKind::Audio => "audio",
            MediaKind::Video => "video",
        }
    }
}

/// How a media value is written in the body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaValue {
    /// `"key": "<url>"`.
    Bare(String),
    /// `"key": {"url": "<url>", ...}`.
    Object(String),
    /// Present but neither a string nor an object with a string `url`.
    Malformed,
}

impl MediaValue {
    pub fn url(&self) -> Option<&str> {
        match self {
            MediaValue::Bare(u) | MediaValue::Object(u) => Some(u),
            MediaValue::Malformed => None,
        }
    }
}

/// One media reference found in a body.
#[derive(Debug, Clone)]
pub struct MediaRef {
    /// JSON pointer to the content part (or input item) that holds it.
    pub part: String,
    /// The key under which the reference sits in that part.
    pub key: &'static str,
    pub kind: MediaKind,
    pub value: MediaValue,
    /// The part's `type`, when it is a string.
    pub part_type: Option<String>,
}

impl MediaRef {
    /// JSON pointer to the media value itself.
    pub fn pointer(&self) -> String {
        format!("{}/{}", self.part, self.key)
    }

    /// Whether a rewritten bare URL is written as `{"url": ...}`: only for a
    /// part whose `type` names its key (SGLang takes only the object form
    /// there). Every other bare URL keeps the caller's string form, which is
    /// what vLLM's untyped and `input_image` parts and SGLang's embedding
    /// items take.
    pub fn rewrite_as_object(&self) -> bool {
        matches!(self.key, "image_url" | "audio_url" | "video_url")
            && self.part_type.as_deref() == Some(self.key)
    }
}

/// A content part or input item found in a body, with its JSON pointer.
pub struct Part<'a> {
    pub pointer: String,
    pub obj: &'a Map<String, Value>,
}

/// The input fields nest deeper than any engine shape; such a body is
/// refused rather than walked partway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooDeep;

/// Every object under the input fields of `body` that can hold media: list
/// items, chat messages, `{"content": [...]}` inputs, and the parts and items
/// inside them. Inside an object only `content` is descended into.
pub fn parts(body: &Value) -> Result<Vec<Part<'_>>, TooDeep> {
    let mut out = Vec::new();
    let Some(obj) = body.as_object() else {
        return Ok(out);
    };
    for field in INPUT_FIELDS {
        if let Some(v) = obj.get(*field) {
            walk_input(v, format!("/{}", escape(field)), 0, &mut out)?;
        }
    }
    Ok(out)
}

/// Every media reference under the input fields of `body`.
pub fn media_refs(body: &Value) -> Result<Vec<MediaRef>, TooDeep> {
    let mut out = Vec::new();
    for part in parts(body)? {
        let part_type = part
            .obj
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string);
        for (key, kind) in MEDIA_KEYS {
            let value = match part.obj.get(*key) {
                None | Some(Value::Null) => continue,
                Some(Value::String(s)) => MediaValue::Bare(s.clone()),
                Some(Value::Object(m)) => match m.get("url") {
                    Some(Value::String(s)) => MediaValue::Object(s.clone()),
                    _ => MediaValue::Malformed,
                },
                Some(_) => MediaValue::Malformed,
            };
            out.push(MediaRef {
                part: part.pointer.clone(),
                key,
                kind: *kind,
                value,
                part_type: part_type.clone(),
            });
        }
    }
    Ok(out)
}

/// Deepest nesting walked. The deepest engine shape (batched conversations in
/// embeddings `input`) is five levels.
const MAX_DEPTH: usize = 16;

fn walk_input<'a>(
    v: &'a Value,
    pointer: String,
    depth: usize,
    out: &mut Vec<Part<'a>>,
) -> Result<(), TooDeep> {
    if depth > MAX_DEPTH {
        return Err(TooDeep);
    }
    match v {
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                walk_input(item, format!("{pointer}/{i}"), depth + 1, out)?;
            }
        }
        Value::Object(obj) => {
            // A content part, an embedding item (`{"text", "image",
            // "video"}`), a chat message or a `{"content": [...]}` input. The
            // object itself may carry a media key (the engines ignore keys
            // they do not know, so a `role` or `content` next to one does
            // not hide it), and its `content` may hold parts.
            if let Some(content) = obj.get("content") {
                walk_input(content, format!("{pointer}/content"), depth + 1, out)?;
            }
            out.push(Part { pointer, obj });
        }
        _ => {}
    }
    Ok(())
}

/// RFC 6901 escaping of one pointer segment.
fn escape(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn urls(body: Value) -> Vec<(String, &'static str, Option<String>)> {
        media_refs(&body)
            .unwrap()
            .into_iter()
            .map(|r| (r.pointer(), r.key, r.value.url().map(str::to_string)))
            .collect()
    }

    #[test]
    fn finds_media_in_every_engine_shape() {
        let u = "http://example.com/a";
        let cases = [
            // Chat parts, by type, without type, with a uuid, Responses style.
            (
                json!({"messages": [{"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": u}},
                    {"image_url": u},
                    {"type": "text", "text": "x", "uuid": "1", "video_url": {"url": u}},
                    {"type": "input_image", "image_url": u},
                    {"type": "audio_url", "audio_url": {"url": u}}
                ]}]}),
                vec![
                    "/messages/0/content/1/image_url",
                    "/messages/0/content/2/image_url",
                    "/messages/0/content/3/video_url",
                    "/messages/0/content/4/image_url",
                    "/messages/0/content/5/audio_url",
                ],
            ),
            // Batched conversations.
            (
                json!({"messages": [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}}]}]]}),
                vec!["/messages/0/0/content/0/image_url"],
            ),
            // vLLM embeddings: messages in `input`, single and batched.
            (
                json!({"input": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}}]}]}),
                vec!["/input/0/content/0/image_url"],
            ),
            (
                json!({"input": [[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": u}}]}]]}),
                vec!["/input/0/0/content/0/video_url"],
            ),
            // SGLang embeddings items.
            (
                json!({"input": [{"text": "a", "image": u}, {"video": u}]}),
                vec!["/input/0/image", "/input/1/video"],
            ),
            // vLLM scoring: `{"content": [parts]}`, alone and in lists.
            (
                json!({"query": {"content": [{"type": "image_url", "image_url": {"url": u}}]},
                       "documents": [{"content": [{"type": "video_url", "video_url": {"url": u}}]}, "plain"]}),
                vec![
                    "/query/content/0/image_url",
                    "/documents/0/content/0/video_url",
                ],
            ),
            (
                json!({"data_1": "a", "data_2": [{"content": [{"type": "image_url", "image_url": {"url": u}}]}],
                       "queries": [{"content": [{"image_url": u}]}], "items": [{"content": [{"image_url": u}]}],
                       "text_1": {"content": [{"image_url": u}]}, "text_2": [{"content": [{"image_url": u}]}]}),
                vec![
                    "/queries/0/content/0/image_url",
                    "/items/0/content/0/image_url",
                    "/text_1/content/0/image_url",
                    "/text_2/0/content/0/image_url",
                    "/data_2/0/content/0/image_url",
                ],
            ),
            // A media key next to `role` or `content` is still found.
            (
                json!({"input": [{"image": u, "content": "x"}],
                       "documents": [[{"type": "image_url", "image_url": {"url": u}, "role": "x"}]]}),
                vec!["/input/0/image", "/documents/0/0/image_url"],
            ),
            // SGLang rerank: lists of parts.
            (
                json!({"query": [{"type": "image_url", "image_url": {"url": u}}],
                       "documents": [[{"type": "text", "text": "d"}, {"type": "video_url", "video_url": {"url": u}}]]}),
                vec!["/query/0/image_url", "/documents/0/1/video_url"],
            ),
        ];
        for (body, expected) in cases {
            let found: Vec<String> = urls(body.clone()).into_iter().map(|(p, _, _)| p).collect();
            let mut want: Vec<String> = expected.into_iter().map(String::from).collect();
            let mut got = found.clone();
            want.sort();
            got.sort();
            assert_eq!(got, want, "{body}");
        }
    }

    #[test]
    fn tools_and_tool_arguments_are_not_media() {
        let body = json!({
            "messages": [{"role": "assistant", "content": null, "tool_calls": [{"function":
                {"name": "f", "arguments": {"image_url": "http://example.com/a"}}}]}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters":
                {"type": "object", "properties": {"image_url": {"type": "string"}}}}}],
            "response_format": {"type": "json_schema", "json_schema": {"schema":
                {"properties": {"image": {"type": "string"}}}}}
        });
        assert!(media_refs(&body).unwrap().is_empty());
    }

    #[test]
    fn malformed_values_are_reported() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"detail": "low"}},
            {"type": "image_url", "image_url": 5},
            {"type": "image_url", "image_url": null}
        ]}]});
        let refs = media_refs(&body).unwrap();
        assert_eq!(refs.len(), 2);
        assert!(refs.iter().all(|r| r.value == MediaValue::Malformed));
    }

    #[test]
    fn nesting_beyond_any_engine_shape_is_refused() {
        let mut deep = json!({"image": "http://example.com/a"});
        for _ in 0..20 {
            deep = json!([deep]);
        }
        assert_eq!(media_refs(&json!({ "input": deep })).unwrap_err(), TooDeep);
        let mut ok = json!("text");
        for _ in 0..10 {
            ok = json!([ok]);
        }
        assert!(media_refs(&json!({ "input": ok })).is_ok());
    }

    #[test]
    fn pointer_segments_are_escaped() {
        assert_eq!(escape("a/b~c"), "a~1b~0c");
    }
}
