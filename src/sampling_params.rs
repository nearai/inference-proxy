//! Pre-dispatch bounds on sampling parameters that crash the engine.
//!
//! Each value refused here took down SGLang scheduler processes when
//! forwarded (GLM-5.3, tee-bench smoke cases C08–C12): one request, the
//! whole replica gone. vLLM refuses most of them itself, but not all (an
//! out-of-`int32` `top_k` is a 200 there). The proxy refuses them with a
//! `400` naming the field before any backend is chosen, on the body as it
//! will be forwarded (after decryption). Nothing is clamped: a caller who
//! sent a value learns it was not used.
//!
//! * chat `top_logprobs`, completions `logprobs`: an integer in
//!   `0..=MAX_LOGPROBS`. 20 is OpenAI's limit and vLLM's default
//!   `max_logprobs`; a huge value is an allocation the scheduler dies on
//!   (sglang#40076, sglang#31597).
//! * `top_k`: `-1` (disabled) or an integer in `0..=i32::MAX`; the engines
//!   keep it in an `int32` tensor (sglang#17646). `0` is forwarded as it
//!   always was, the engine decides what it means.
//! * `stop_token_ids`: an array of integers in `0..=u32::MAX`; a `null` or
//!   non-integer element kills the scheduler (sglang#22144). An id past the
//!   vocabulary (sglang#44535) is not caught: the proxy does not know the
//!   vocabulary size.
//!
//! A `null` field is the field left out and passes. A number with an exact
//! integer value (`5.0`) counts as that integer, as the engines' pydantic
//! models read it; the body is forwarded untouched.

use serde_json::Value;

use crate::error::AppError;

/// Upper bound for chat `top_logprobs` and completions `logprobs`.
pub const MAX_LOGPROBS: i128 = 20;

/// Refuse the chat-completions parameters the engine cannot survive.
pub fn validate_chat(request: &Value) -> Result<(), AppError> {
    check_range(request, "top_logprobs", 0, MAX_LOGPROBS)?;
    validate_common(request)
}

/// Refuse the (legacy) completions parameters the engine cannot survive.
pub fn validate_completion(request: &Value) -> Result<(), AppError> {
    check_range(request, "logprobs", 0, MAX_LOGPROBS)?;
    validate_common(request)
}

fn validate_common(request: &Value) -> Result<(), AppError> {
    check_top_k(request)?;
    check_stop_token_ids(request)
}

/// The value of an integer-valued JSON number, `None` for anything else.
fn integer(value: &Value) -> Option<i128> {
    let Value::Number(number) = value else {
        return None;
    };
    if let Some(i) = number.as_i64() {
        return Some(i.into());
    }
    if let Some(u) = number.as_u64() {
        return Some(u.into());
    }
    // An integral float: within `i128` only up to 2^127, and every bound
    // here is far below that, so anything larger is simply out of range.
    let f = number.as_f64()?;
    (f.is_finite() && f.fract() == 0.0 && f.abs() < 1e38).then_some(f as i128)
}

/// The field, unless it is absent or `null`.
fn present<'a>(request: &'a Value, field: &str) -> Option<&'a Value> {
    request.get(field).filter(|value| !value.is_null())
}

fn check_range(request: &Value, field: &str, min: i128, max: i128) -> Result<(), AppError> {
    match present(request, field) {
        None => Ok(()),
        Some(value) => match integer(value) {
            Some(i) if (min..=max).contains(&i) => Ok(()),
            _ => Err(AppError::BadRequest(format!(
                "`{field}` must be an integer between {min} and {max}"
            ))),
        },
    }
}

fn check_top_k(request: &Value) -> Result<(), AppError> {
    match present(request, "top_k").map(integer) {
        None => Ok(()),
        Some(Some(k)) if k == -1 || (0..=i128::from(i32::MAX)).contains(&k) => Ok(()),
        Some(_) => Err(AppError::BadRequest(format!(
            "`top_k` must be -1 (disabled) or an integer between 0 and {}",
            i32::MAX
        ))),
    }
}

fn check_stop_token_ids(request: &Value) -> Result<(), AppError> {
    let Some(value) = present(request, "stop_token_ids") else {
        return Ok(());
    };
    let refused = || {
        AppError::BadRequest(format!(
            "`stop_token_ids` must be an array of integers between 0 and {}",
            u32::MAX
        ))
    };
    let ids = value.as_array().ok_or_else(refused)?;
    let in_range =
        |id: &Value| integer(id).is_some_and(|i| (0..=i128::from(u32::MAX)).contains(&i));
    if ids.iter().all(in_range) {
        Ok(())
    } else {
        Err(refused())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn refused_field(result: Result<(), AppError>) -> String {
        match result {
            Err(AppError::BadRequest(message)) => message,
            other => panic!("expected a 400, got {other:?}"),
        }
    }

    #[test]
    fn test_chat_tee_bench_crash_cases_are_refused() {
        let cases = [
            // C08
            (
                json!({"logprobs": true, "top_logprobs": 1_000_000}),
                "`top_logprobs`",
            ),
            // C10 (the out-of-vocab id itself is not caught; this one is past u32)
            (
                json!({"stop_token_ids": [4_294_967_296u64], "min_tokens": 4}),
                "`stop_token_ids`",
            ),
            // C11
            (json!({"stop_token_ids": [null]}), "`stop_token_ids`"),
            // C12
            (json!({"top_k": 2_147_483_648u64}), "`top_k`"),
        ];
        for (body, field) in cases {
            let message = refused_field(validate_chat(&body));
            assert!(message.contains(field), "{body}: {message}");
        }
    }

    #[test]
    fn test_completion_tee_bench_crash_case_is_refused() {
        // C09
        let message = refused_field(validate_completion(&json!({"logprobs": 2_000_000_000})));
        assert!(message.contains("`logprobs`"), "{message}");
    }

    #[test]
    fn test_logprobs_bounds() {
        for n in [0, 1, 20] {
            validate_chat(&json!({"logprobs": true, "top_logprobs": n})).unwrap();
            validate_completion(&json!({"logprobs": n})).unwrap();
        }
        for bad in [
            json!(21),
            json!(-1),
            json!(2.5),
            json!("5"),
            json!(true),
            json!([1]),
        ] {
            refused_field(validate_chat(&json!({"top_logprobs": bad})));
            refused_field(validate_completion(&json!({"logprobs": bad})));
        }
        // Integral floats read as the integer.
        validate_chat(&json!({"top_logprobs": 20.0})).unwrap();
        refused_field(validate_chat(&json!({"top_logprobs": 21.0})));
        // Chat `logprobs` is the boolean switch, not a count.
        validate_chat(&json!({"logprobs": true})).unwrap();
    }

    #[test]
    fn test_top_k_bounds() {
        for k in [
            json!(-1),
            json!(0),
            json!(1),
            json!(40),
            json!(i32::MAX),
            json!(50.0),
        ] {
            validate_chat(&json!({"top_k": k})).unwrap();
            validate_completion(&json!({"top_k": k})).unwrap();
        }
        let too_big = i64::from(i32::MAX) + 1;
        for bad in [
            json!(-2),
            json!(too_big),
            json!(1e300),
            json!(0.5),
            json!("40"),
        ] {
            let message = refused_field(validate_chat(&json!({"top_k": bad})));
            assert!(message.contains("`top_k`"), "{message}");
            refused_field(validate_completion(&json!({"top_k": bad})));
        }
    }

    #[test]
    fn test_stop_token_ids_bounds() {
        for ids in [json!([]), json!([0, 151_329, u32::MAX])] {
            validate_chat(&json!({"stop_token_ids": ids})).unwrap();
            validate_completion(&json!({"stop_token_ids": ids})).unwrap();
        }
        let too_big = u64::from(u32::MAX) + 1;
        for bad in [
            json!([null]),
            json!([1, null]),
            json!([-1]),
            json!([too_big]),
            json!([1.5]),
            json!(["2"]),
            json!([[2]]),
            json!(2),
            json!("2"),
        ] {
            refused_field(validate_chat(&json!({"stop_token_ids": bad})));
            refused_field(validate_completion(&json!({"stop_token_ids": bad})));
        }
    }

    #[test]
    fn test_absent_and_null_fields_pass() {
        let body = json!({"messages": [], "max_tokens": 16});
        validate_chat(&body).unwrap();
        validate_completion(&body).unwrap();
        let nulls =
            json!({"top_logprobs": null, "logprobs": null, "top_k": null, "stop_token_ids": null});
        validate_chat(&nulls).unwrap();
        validate_completion(&nulls).unwrap();
    }
}
