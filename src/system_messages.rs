//! Pre-dispatch merge of `system` messages, for a model whose chat template
//! takes one, as the first message.
//!
//! Some chat templates refuse a `system` message anywhere but at the start of
//! the conversation: the engine answers `400 "System message must be at the
//! beginning."` to `[system, system, user]`, to `[system, user, system]` and
//! to `[user, system]` alike. Callers send all three. Agent frameworks keep a
//! second system prompt next to the first or add one late in the history, and
//! other providers take them, so the request fails for that model only.
//!
//! A model of a gateway list can ask for its requests to be put in the shape
//! its template takes (`merge_system_messages`, `model_list`). A request with
//! a `system` message after the first position is then rewritten to hold
//! exactly one, first: its content is the text of every `system` message in
//! the order they were sent, a blank line between two of them, and every
//! other message keeps its place among the others. No other role is read as
//! a system message, `developer` included.
//!
//! The rewrite never drops anything, and when it cannot keep everything it
//! does not happen: the request goes out as the caller wrote it and the
//! engine answers. That is the case when
//!
//! - a `system` message's content is not text: anything but a string or an
//!   array of text parts (an image part, a part of an unknown type, a part
//!   without a string `text`, `null`, no content at all);
//! - a text part carries a key other than `type` and `text` (a
//!   `cache_control` breakpoint, say): a string cannot carry it, and whether
//!   the engine reads it is not for this code to decide;
//! - two `system` messages give a field other than `role` and `content` (a
//!   `name`, say) different values. Such a field is otherwise kept on the
//!   merged message, where it then covers the whole of it.
//!
//! A request without a `system` message after the first position is not
//! touched at all, whatever its first message holds.
//!
//! List mode only: the one model of a process without a list has no such
//! setting and this module is never called for it. A rewritten request is
//! counted in `inference_proxy_model_system_messages_merged_total{model}`;
//! nothing about the messages is logged or labelled.

use serde_json::{Map, Value};

use crate::model_metrics::{model_counter, ModelLabel};

/// What goes between the texts of two `system` messages: a blank line.
const SEPARATOR: &str = "\n\n";

/// Rewrite `messages` so that it holds one `system` message, first, with the
/// text of all of them (module docs). Returns whether the request was
/// rewritten; `false` leaves it byte-for-byte unchanged.
///
/// `model` is the configured id of the model the request is served as
/// (`ModelView::label`), the only label of the counter.
pub fn merge_system_messages(request_json: &mut Value, model: ModelLabel) -> bool {
    let Some(messages) = request_json
        .get_mut("messages")
        .and_then(Value::as_array_mut)
    else {
        return false;
    };
    // More than one `system` message, or one that is not first, is exactly a
    // `system` message after the first position.
    if !messages.iter().skip(1).any(is_system) {
        return false;
    }
    let Some(merged) = merged_system_message(messages) else {
        return false;
    };
    let rest = std::mem::take(messages)
        .into_iter()
        .filter(|message| !is_system(message));
    *messages = std::iter::once(merged).chain(rest).collect();
    model_counter!(model, "inference_proxy_model_system_messages_merged_total").increment(1);
    true
}

/// A message whose `role` is exactly `system`, as the engine matches it.
fn is_system(message: &Value) -> bool {
    message.get("role").and_then(Value::as_str) == Some("system")
}

/// The one message that stands for every `system` message of `messages`, or
/// `None` when something would be lost by merging them.
fn merged_system_message(messages: &[Value]) -> Option<Value> {
    let mut merged = Map::new();
    let mut content = String::new();
    for message in messages.iter().filter(|message| is_system(message)) {
        let message = message.as_object()?;
        let before = content.len();
        if before > 0 {
            content.push_str(SEPARATOR);
        }
        let with_separator = content.len();
        push_text(&mut content, message.get("content")?)?;
        // An empty content adds nothing, not even the blank line.
        if content.len() == with_separator {
            content.truncate(before);
        }
        for (key, value) in message {
            if key == "role" || key == "content" {
                continue;
            }
            match merged.get(key) {
                // Two values for one field: one of them would be dropped.
                Some(kept) if kept != value => return None,
                Some(_) => {}
                None => {
                    merged.insert(key.clone(), value.clone());
                }
            }
        }
    }
    merged.insert("role".to_string(), Value::String("system".to_string()));
    merged.insert("content".to_string(), Value::String(content));
    Some(Value::Object(merged))
}

/// Append the text of a message's `content` to `text`: a string, or the text
/// parts of an array one after the other. `None` for any other content.
fn push_text(text: &mut String, content: &Value) -> Option<()> {
    match content {
        Value::String(string) => text.push_str(string),
        Value::Array(parts) => {
            for part in parts {
                text.push_str(text_of_part(part)?);
            }
        }
        _ => return None,
    }
    Some(())
}

/// The text of a `{"type": "text", "text": "..."}` part, and of nothing
/// else: a part with any other key holds more than a string can carry.
fn text_of_part(part: &Value) -> Option<&str> {
    let part = part.as_object()?;
    if part.len() != 2 || part.get("type")?.as_str()? != "text" {
        return None;
    }
    part.get("text")?.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MODEL: &str = "example/alpha";
    const COUNTER: &str = "inference_proxy_model_system_messages_merged_total";

    /// Merge `messages` as a request of `MODEL`: what they became, and
    /// whether the request was rewritten.
    fn merged(messages: Value) -> (Value, bool) {
        let mut request = json!({"model": MODEL, "messages": messages});
        let rewritten = merge_system_messages(&mut request, Some(MODEL));
        (request["messages"].take(), rewritten)
    }

    /// `request` is left exactly as it was, down to its serialisation.
    fn assert_untouched(request: Value) {
        let mut after = request.clone();
        let bytes = serde_json::to_vec(&request).unwrap();
        assert!(!merge_system_messages(&mut after, Some(MODEL)), "{request}");
        assert_eq!(after, request);
        assert_eq!(serde_json::to_vec(&after).unwrap(), bytes, "{request}");
    }

    fn assert_messages_untouched(messages: Value) {
        assert_untouched(json!({"model": MODEL, "messages": messages}));
    }

    #[test]
    fn the_three_refused_shapes_become_one_leading_system_message() {
        let user = json!({"role": "user", "content": "hello"});
        let both = json!([{"role": "system", "content": "first\n\nsecond"}, user]);
        for messages in [
            json!([
                {"role": "system", "content": "first"},
                {"role": "system", "content": "second"},
                user
            ]),
            json!([
                {"role": "system", "content": "first"},
                user,
                {"role": "system", "content": "second"}
            ]),
        ] {
            let sent = messages.clone();
            assert_eq!(merged(messages), (both.clone(), true), "{sent}");
        }
        // One system message, not first: it moves, and nothing is added.
        assert_eq!(
            merged(json!([user, {"role": "system", "content": "second"}])),
            (json!([{"role": "system", "content": "second"}, user]), true)
        );
    }

    #[test]
    fn system_messages_are_joined_in_the_order_they_were_sent() {
        let (messages, rewritten) = merged(json!([
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "one"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "two"},
            {"role": "system", "content": "three"},
            {"role": "user", "content": "u2"},
            {"role": "system", "content": "four"}
        ]));
        assert!(rewritten);
        assert_eq!(
            messages,
            json!([
                {"role": "system", "content": "one\n\ntwo\n\nthree\n\nfour"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"}
            ])
        );
    }

    #[test]
    fn strings_and_text_parts_merge_into_one_string() {
        let (messages, rewritten) = merged(json!([
            {"role": "system", "content": "a string"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "system", "content": [
                {"type": "text", "text": "part one, "},
                {"type": "text", "text": "part two"}
            ]},
            {"role": "system", "content": [{"type": "text", "text": "a single part"}]}
        ]));
        assert!(rewritten);
        assert_eq!(
            messages,
            json!([
                {"role": "system", "content": "a string\n\npart one, part two\n\na single part"},
                // A user message's parts are not the merge's business.
                {"role": "user", "content": [{"type": "text", "text": "hello"}]}
            ])
        );

        // One misplaced system message made of parts becomes a string too.
        assert_eq!(
            merged(json!([
                {"role": "user", "content": "hello"},
                {"role": "system", "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"}
                ]}
            ])),
            (
                json!([
                    {"role": "system", "content": "ab"},
                    {"role": "user", "content": "hello"}
                ]),
                true
            )
        );
    }

    #[test]
    fn an_empty_content_adds_no_blank_line() {
        for (contents, expected) in [
            (vec![json!(""), json!("b")], "b"),
            (vec![json!("a"), json!("")], "a"),
            (vec![json!("a"), json!([]), json!("c")], "a\n\nc"),
            (
                vec![
                    json!("a"),
                    json!([{"type": "text", "text": ""}]),
                    json!("c"),
                ],
                "a\n\nc",
            ),
            (vec![json!(""), json!([])], ""),
            // Whitespace is content like any other.
            (vec![json!("a"), json!(" "), json!("c")], "a\n\n \n\nc"),
        ] {
            let messages: Vec<Value> = contents
                .iter()
                .map(|content| json!({"role": "system", "content": content}))
                .collect();
            let (messages, rewritten) = merged(Value::Array(messages));
            assert!(rewritten, "{contents:?}");
            assert_eq!(
                messages,
                json!([{"role": "system", "content": expected}]),
                "{contents:?}"
            );
        }
    }

    #[test]
    fn a_system_message_that_is_not_text_leaves_the_request_untouched() {
        let image = json!({"type": "image_url", "image_url": {"url": "https://img.example/a.png"}});
        for content in [
            // A part that is not text, alone or next to text.
            json!([image]),
            json!([{"type": "text", "text": "look at this"}, image]),
            json!([{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]),
            // Parts of a shape this code does not know.
            json!([{"type": "input_text", "text": "another API's text part"}]),
            json!([{"type": "text"}]),
            json!([{"type": "text", "text": 7}]),
            json!([{"type": "text", "text": null}]),
            json!([{"text": "no type"}]),
            // A text part with more on it than a string can carry.
            json!([{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}]),
            json!([{"type": "text", "text": "annotated", "annotations": []}]),
            json!(["a bare string"]),
            json!([null]),
            // Contents that are neither a string nor an array.
            json!(null),
            json!(7),
            json!(true),
            json!({"type": "text", "text": "a part, not an array of them"}),
        ] {
            for messages in [
                json!([
                    {"role": "system", "content": "first"},
                    {"role": "system", "content": content}
                ]),
                json!([
                    {"role": "system", "content": content},
                    {"role": "user", "content": "hello"},
                    {"role": "system", "content": "second"}
                ]),
                json!([
                    {"role": "user", "content": "hello"},
                    {"role": "system", "content": content}
                ]),
            ] {
                assert_messages_untouched(messages);
            }
        }
        // No content at all.
        assert_messages_untouched(json!([
            {"role": "system", "content": "first"},
            {"role": "system"}
        ]));
    }

    #[test]
    fn other_roles_are_never_merged_moved_past_each_other_or_changed() {
        let developer = json!({"role": "developer", "content": "developer instructions"});
        let user = json!({"role": "user", "content": [
            {"type": "text", "text": "what is on this picture?"},
            {"type": "image_url", "image_url": {"url": "https://img.example/a.png"}}
        ]});
        let assistant = json!({"role": "assistant", "content": null, "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"q\":\"a\"}"}
        }]});
        let tool = json!({"role": "tool", "tool_call_id": "call_1", "content": "found"});
        let second_developer = json!({"role": "developer", "content": "more of them"});
        let last = json!({"role": "user", "content": "and now?", "name": "someone"});
        let (messages, rewritten) = merged(json!([
            developer,
            {"role": "system", "content": "first"},
            user,
            assistant,
            tool,
            {"role": "system", "content": "second"},
            second_developer,
            last
        ]));
        assert!(rewritten);
        assert_eq!(
            messages,
            json!([
                {"role": "system", "content": "first\n\nsecond"},
                developer,
                user,
                assistant,
                tool,
                second_developer,
                last
            ])
        );

        // Only `system`, spelled exactly so, is a system message: several
        // messages of any other role are not a reason to rewrite.
        assert_messages_untouched(json!([developer, second_developer, user]));
        assert_messages_untouched(json!([
            {"role": "system", "content": "first"},
            developer,
            {"role": "System", "content": "another role, as far as an engine goes"},
            {"role": "SYSTEM", "content": "and another"},
            {"role": ["system"], "content": "not a role"},
            {"content": "no role"},
            "not a message",
            null,
            user
        ]));
    }

    #[test]
    fn a_request_the_template_takes_is_left_byte_for_byte_alone() {
        for messages in [
            json!([]),
            json!([{"role": "user", "content": "hello"}]),
            json!([
                {"role": "system", "content": "first"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "again"}
            ]),
            // One leading system message is never rewritten, whatever it
            // holds: parts stay parts, and nothing is judged.
            json!([
                {"role": "system", "content": [
                    {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "b"}
                ], "name": "policy"},
                {"role": "user", "content": "hello"}
            ]),
            json!([
                {"role": "system", "content": [
                    {"type": "image_url", "image_url": {"url": "https://img.example/a.png"}}
                ]},
                {"role": "user", "content": "hello"}
            ]),
            json!([{"role": "system", "content": null}]),
            json!([{"role": "system"}, {"role": "developer", "content": "d"}]),
        ] {
            assert_messages_untouched(messages);
        }
        // Bodies without a list of messages.
        for request in [
            json!({"model": MODEL}),
            json!({"model": MODEL, "messages": null}),
            json!({"model": MODEL, "messages": "system"}),
            json!({"model": MODEL, "messages": {"role": "system", "content": "a"}}),
            json!({"model": MODEL, "prompt": "hello"}),
            json!([{"role": "user"}, {"role": "system"}]),
            json!("not an object"),
        ] {
            assert_untouched(request);
        }
    }

    #[test]
    fn other_fields_of_a_system_message_are_kept_and_never_chosen_between() {
        // On one message, on several with the same value, on different ones.
        let (messages, rewritten) = merged(json!([
            {"role": "system", "content": "first", "name": "policy"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "second", "name": "policy"},
            {"role": "system", "content": "third", "cache_control": {"type": "ephemeral"}}
        ]));
        assert!(rewritten);
        assert_eq!(
            messages,
            json!([
                {
                    "role": "system",
                    "content": "first\n\nsecond\n\nthird",
                    "name": "policy",
                    "cache_control": {"type": "ephemeral"}
                },
                {"role": "user", "content": "hello"}
            ])
        );

        // A single system message that only moves takes its fields along.
        assert_eq!(
            merged(json!([
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "late", "name": "policy"}
            ])),
            (
                json!([
                    {"role": "system", "content": "late", "name": "policy"},
                    {"role": "user", "content": "hello"}
                ]),
                true
            )
        );

        // Two values for one field: neither is dropped, so nothing is merged.
        assert_messages_untouched(json!([
            {"role": "system", "content": "first", "name": "policy"},
            {"role": "system", "content": "second", "name": "persona"}
        ]));
        assert_messages_untouched(json!([
            {"role": "system", "content": "first", "name": "policy"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "second", "name": null}
        ]));
    }

    #[test]
    fn a_rewritten_request_is_counted_under_its_model_and_nothing_else_is() {
        const SECRET: &str = "system-prompt-sentinel";
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            for messages in [
                // Rewritten: three.
                json!([{"role": "system", "content": SECRET}, {"role": "system", "content": "b"}]),
                json!([{"role": "user", "content": SECRET}, {"role": "system", "content": "b"}]),
                json!([
                    {"role": "system", "content": "a", "name": SECRET},
                    {"role": "user", "content": "u"},
                    {"role": "system", "content": [{"type": "text", "text": SECRET}]}
                ]),
                // Not rewritten: already in place, and not mergeable.
                json!([{"role": "system", "content": SECRET}, {"role": "user", "content": "u"}]),
                json!([{"role": "system", "content": SECRET}, {"role": "system", "content": 7}]),
            ] {
                merged(messages);
            }
            // Another model's requests are another series.
            let mut request = json!({"messages": [
                {"role": "user", "content": "u"},
                {"role": "system", "content": "s"}
            ]});
            assert!(merge_system_messages(&mut request, Some("example/beta")));
        });
        let rendered = handle.render();
        let samples: Vec<&str> = rendered
            .lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .collect();
        assert_eq!(
            samples.len(),
            2,
            "one series per model and no other: {rendered}"
        );
        for expected in [
            format!("{COUNTER}{{model=\"example/alpha\"}} 3"),
            format!("{COUNTER}{{model=\"example/beta\"}} 1"),
        ] {
            assert!(samples.contains(&expected.as_str()), "{rendered}");
        }
        assert!(!rendered.contains(SECRET), "{rendered}");
    }
}
