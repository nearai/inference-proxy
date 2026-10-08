//! The `model` label of per-model metric series.
//!
//! A process that serves one model (every CVM proxy, a single-model gateway)
//! emits its admission, pool, tier, engine-load and usage series without a
//! `model` label, and dashboards and alerts do vector matching on exactly
//! those label sets. A gateway that serves a model list (`model_list.rs`)
//! emits the same series once per model, told apart by `model`.
//!
//! The macros below are the one place that difference lives. With `None` they
//! expand to the very `metrics::*!` invocation single-model code has always
//! made, so those series cannot change; with `Some(id)` the same series
//! carries `model` as its last label.
//!
//! Two series exist in list mode only (`ModelRequest`): the responses and the
//! failed streams of each model. A process that serves one model emits
//! neither. (Two more list-only series count requests a per-model setting
//! rewrote: `reasoning::apply_effort_map`'s for a `reasoning_effort_map`, and
//! `system_messages::merge_system_messages`'s for `merge_system_messages`.)

use axum::response::{IntoResponse, Response};

use crate::error::AppError;

/// `None` when the process serves one model; the configured model id in list
/// mode. `'static` because the ids are fixed at startup for the life of the
/// process (`model_list::ServedModel` leaks one small string per model), which
/// keeps the label `Copy` and free to attach.
pub type ModelLabel = Option<&'static str>;

macro_rules! model_counter {
    ($model:expr, $name:literal $(, $key:literal => $value:expr)* $(,)?) => {
        match $model {
            None => ::metrics::counter!($name $(, $key => $value)*),
            Some(model) => ::metrics::counter!($name $(, $key => $value)*, "model" => model),
        }
    };
}

macro_rules! model_gauge {
    ($model:expr, $name:literal $(, $key:literal => $value:expr)* $(,)?) => {
        match $model {
            None => ::metrics::gauge!($name $(, $key => $value)*),
            Some(model) => ::metrics::gauge!($name $(, $key => $value)*, "model" => model),
        }
    };
}

macro_rules! model_histogram {
    ($model:expr, $name:literal $(, $key:literal => $value:expr)* $(,)?) => {
        match $model {
            None => ::metrics::histogram!($name $(, $key => $value)*),
            Some(model) => ::metrics::histogram!($name $(, $key => $value)*, "model" => model),
        }
    };
}

pub(crate) use {model_counter, model_gauge, model_histogram};

/// `labels` followed by `model`, for the series whose label set is built as a
/// slice rather than written out at the call site.
pub(crate) fn with_model(
    labels: &[(&'static str, &'static str)],
    model: &'static str,
) -> Vec<metrics::Label> {
    labels
        .iter()
        .map(metrics::Label::from)
        .chain(std::iter::once(metrics::Label::new("model", model)))
        .collect()
}

/// A chat/completions request of a model list, as the two list-only series
/// label it: the configured model id (never the request's own `model`
/// string) and the route.
#[derive(Clone, Copy, Debug)]
pub struct ModelRequest {
    model: &'static str,
    endpoint: &'static str,
}

impl ModelRequest {
    /// `None` for the single model (`ModelView::label`): nothing is counted
    /// for it, and the request path pays one branch.
    pub(crate) fn of(model: ModelLabel, endpoint: &'static str) -> Option<Self> {
        model.map(|model| Self { model, endpoint })
    }
}

/// `inference_proxy_model_requests_total{endpoint, status, model}`: one per
/// response of a request whose model was resolved, under the status the
/// client is sent, whoever decided it: the engine, an upstream failure (502,
/// 504) or the gateway's own refusal (429, 503). Counted when the status goes
/// out, like `http_requests_total`, so a stream counts once its 200 is
/// committed and a client that leaves before any response is not counted.
///
/// The routes call this on what their handler returns, which makes it the one
/// exit every path after model selection goes through.
pub(crate) fn count_response(
    request: Option<ModelRequest>,
    result: Result<Response, AppError>,
) -> Result<Response, AppError> {
    let Some(request) = request else {
        return result;
    };
    // An error has no status until it is rendered. This is the call axum
    // makes on the handler's result anyway, so it still happens exactly once.
    let response = result.unwrap_or_else(IntoResponse::into_response);
    let status = response.status().as_u16().to_string();
    model_counter!(
        Some(request.model),
        "inference_proxy_model_requests_total",
        "endpoint" => request.endpoint,
        "status" => status,
    )
    .increment(1);
    Ok(response)
}

/// `inference_proxy_model_stream_errors_total{endpoint, model}`: one per
/// stream that failed after its 200 was sent. The status series above can
/// only say 200 for those.
pub(crate) fn count_stream_error(request: Option<ModelRequest>) {
    let Some(request) = request else {
        return;
    };
    model_counter!(
        Some(request.model),
        "inference_proxy_model_stream_errors_total",
        "endpoint" => request.endpoint,
    )
    .increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(model: ModelLabel) -> String {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            model_counter!(model, "t_refusals_total", "reason" => "budget").increment(1);
            model_gauge!(model, "t_inflight").set(3.0);
            model_histogram!(model, "t_seconds").record(0.5);
            let labels = [("outcome", "accepted")];
            match model {
                None => metrics::counter!("t_reports_total", &labels),
                Some(model) => metrics::counter!("t_reports_total", with_model(&labels, model)),
            }
            .increment(1);
        });
        handle.render()
    }

    #[test]
    fn without_a_model_the_series_are_the_plain_ones() {
        let rendered = rendered(None);
        assert!(
            rendered.contains("t_refusals_total{reason=\"budget\"} 1"),
            "{rendered}"
        );
        assert!(rendered.contains("t_inflight 3"), "{rendered}");
        assert!(rendered.contains("t_seconds_count 1"), "{rendered}");
        assert!(
            rendered.contains("t_reports_total{outcome=\"accepted\"} 1"),
            "{rendered}"
        );
        assert!(!rendered.contains("model="), "{rendered}");
    }

    #[test]
    fn a_single_model_response_goes_out_untouched_and_uncounted() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let request = ModelRequest::of(None, "/v1/route");
            assert!(request.is_none());
            // Still the error, for axum to render: not rendered here.
            let result = count_response(request, Err(AppError::RateLimited));
            assert!(matches!(result, Err(AppError::RateLimited)));
            count_stream_error(request);
        });
        let rendered = handle.render();
        assert!(!rendered.contains("inference_proxy_model_"), "{rendered}");
        assert!(!rendered.contains("http_errors_total"), "{rendered}");
    }

    #[test]
    fn a_list_model_response_is_counted_under_the_status_it_goes_out_with() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let request = ModelRequest::of(Some("org/model-a"), "/v1/route");
            let refused = count_response(
                request,
                Err(AppError::Overloaded {
                    reason: "budget",
                    retry_after_secs: 2,
                }),
            )
            .expect("an error is rendered to count its status");
            assert_eq!(refused.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(refused.headers().get("retry-after").unwrap(), "2");
            let served = count_response(request, Ok(axum::http::StatusCode::OK.into_response()))
                .expect("a response stays one");
            assert_eq!(served.status(), axum::http::StatusCode::OK);
            count_stream_error(request);
        });
        let rendered = handle.render();
        for expected in [
            "inference_proxy_model_requests_total{endpoint=\"/v1/route\",status=\"429\",model=\"org/model-a\"} 1",
            "inference_proxy_model_requests_total{endpoint=\"/v1/route\",status=\"200\",model=\"org/model-a\"} 1",
            "inference_proxy_model_stream_errors_total{endpoint=\"/v1/route\",model=\"org/model-a\"} 1",
            // Rendered once: here, and not again by axum.
            "http_errors_total{error_type=\"overloaded\"} 1",
        ] {
            assert!(rendered.contains(expected), "{expected}: {rendered}");
        }
    }

    #[test]
    fn with_a_model_every_series_carries_it_last() {
        let rendered = rendered(Some("org/model-a"));
        assert!(
            rendered.contains("t_refusals_total{reason=\"budget\",model=\"org/model-a\"} 1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("t_inflight{model=\"org/model-a\"} 3"),
            "{rendered}"
        );
        assert!(
            rendered.contains("t_seconds_count{model=\"org/model-a\"} 1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("t_reports_total{outcome=\"accepted\",model=\"org/model-a\"} 1"),
            "{rendered}"
        );
    }
}
