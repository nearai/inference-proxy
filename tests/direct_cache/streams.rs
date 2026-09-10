use super::*;
use bytes::Bytes;
use tokio::sync::mpsc;

#[tokio::test]
async fn direct_cache_named_and_fallback_completed_and_incomplete_streams() {
    for route in ["/v1/chat/completions", "/v1/cache-fixture"] {
        for completed in [true, false] {
            let backend = MockServer::start().await;
            let cloud = cloud_fixture().await;
            let mut sse =
                first_frame() + &frame(json!({"prompt_tokens": 1858, "completion_tokens": 8}));
            if completed {
                sse.push_str("data: [DONE]\n\n");
            }
            Mock::given(method("POST"))
                .and(path(route))
                .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
                .expect(1)
                .mount(&backend)
                .await;
            let app = build_test_app_with_cloud_api(&backend.uri(), &cloud.uri());
            let response = app.clone().oneshot(request(route, true)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(
                String::from_utf8_lossy(&bytes).contains("[DONE]"),
                completed
            );
            assert_report(&cloud, 1856, 8).await;
            if !completed {
                assert_no_signature(app).await;
            }
            backend.verify().await;
        }
    }
}

#[tokio::test]
async fn direct_cache_named_and_fallback_empty_usage_retains_cached_count() {
    for route in ["/v1/chat/completions", "/v1/cache-fixture"] {
        let backend = MockServer::start().await;
        let cloud = cloud_fixture().await;
        let sse = frame(json!({"prompt_tokens": 1858, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 1856}}))
            + &frame(json!({}))
            + "data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path(route))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&backend)
            .await;
        let app = build_test_app_with_cloud_api(&backend.uri(), &cloud.uri());

        let response = app.oneshot(request(route, true)).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&bytes).contains("[DONE]"));
        assert_report(&cloud, 1856, 3).await;
        backend.verify().await;
    }
}

#[tokio::test]
async fn direct_cache_downstream_disconnect_after_usage_reports_once() {
    for route in ["/v1/chat/completions", "/v1/cache-fixture"] {
        let cloud = cloud_fixture().await;
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(1);
        let stream = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let upstream = axum::Router::new().route(
            route,
            axum::routing::post(move || {
                let stream = stream.clone();
                async move {
                    let rx = stream.lock().await.take().expect("one upstream request");
                    axum::response::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from_stream(
                            tokio_stream::wrappers::ReceiverStream::new(rx),
                        ))
                        .unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let mut tasks = tokio::task::JoinSet::new();
        tasks.spawn(async move {
            axum::serve(listener, upstream).await.unwrap();
        });
        let app = build_test_app_with_cloud_api(&url, &cloud.uri());
        let response = app.clone().oneshot(request(route, true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        tx.send(Ok(Bytes::from(first_frame()))).await.unwrap();
        let mut body = response.into_body();
        let first = tokio::time::timeout(Duration::from_secs(2), body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(first.data_ref().unwrap()).contains("cached_tokens"));
        drop(body);
        assert_report(&cloud, 1856, 1).await;
        assert_no_signature(app).await;
        tokio::time::timeout(Duration::from_secs(2), tx.closed())
            .await
            .unwrap();
        tasks.shutdown().await;
    }
}
