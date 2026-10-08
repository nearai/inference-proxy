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
