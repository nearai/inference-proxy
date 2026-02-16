use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sha2::{Digest, Sha256};

// Re-use the test key constants from the codebase
const TEST_ECDSA_KEY: [u8; 32] = [
    0xac, 0x09, 0x74, 0xbe, 0xc3, 0x9a, 0x17, 0xe3, 0x6b, 0xa4, 0xa6, 0xb4, 0xd2, 0x38, 0xff, 0x94,
    0x4b, 0xac, 0xb3, 0x5e, 0x5d, 0xc4, 0xaf, 0x0f, 0x33, 0x47, 0xe5, 0x87, 0x31, 0x79, 0x67, 0x0f,
];

const TEST_ED25519_KEY: [u8; 32] = [
    0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c, 0xc4,
    0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae, 0x7f, 0x60,
];

/// Typical signed text: "{sha256_hex}:{sha256_hex}" = 129 chars
fn typical_sign_text() -> String {
    let req_hash = hex::encode(Sha256::digest(b"test request body"));
    let resp_hash = hex::encode(Sha256::digest(b"test response body"));
    format!("{req_hash}:{resp_hash}")
}

fn bench_ecdsa_sign(c: &mut Criterion) {
    use vllm_proxy_rs::signing::EcdsaContext;

    let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
    let text = typical_sign_text();

    c.bench_function("ecdsa_sign", |b| {
        b.iter(|| ctx.sign(black_box(&text)).unwrap())
    });
}

fn bench_ed25519_sign(c: &mut Criterion) {
    use vllm_proxy_rs::signing::Ed25519Context;

    let ctx = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
    let text = typical_sign_text();

    c.bench_function("ed25519_sign", |b| {
        b.iter(|| ctx.sign(black_box(&text)).unwrap())
    });
}

fn bench_sign_chat(c: &mut Criterion) {
    use vllm_proxy_rs::signing::{EcdsaContext, Ed25519Context, SigningPair};

    let ecdsa = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
    let ed25519 = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
    let pair = SigningPair { ecdsa, ed25519 };
    let text = typical_sign_text();

    c.bench_function("sign_chat", |b| {
        b.iter(|| pair.sign_chat(black_box(&text)).unwrap())
    });
}

fn bench_sha256_hex(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha256_hex");

    for size in [64, 256, 1024, 4096, 65536] {
        let data = vec![0xABu8; size];
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| hex::encode(Sha256::digest(black_box(data))))
        });
    }

    group.finish();
}

fn bench_hex_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("hex_encode");

    // SHA256 output = 32 bytes → 64 hex chars
    let sha256_output = [0xABu8; 32];
    group.bench_function("sha256_32bytes", |b| {
        b.iter(|| hex::encode(black_box(&sha256_output)))
    });

    // ECDSA signature = 65 bytes → 130 hex chars
    let ecdsa_sig = [0xABu8; 65];
    group.bench_function("ecdsa_65bytes", |b| {
        b.iter(|| hex::encode(black_box(&ecdsa_sig)))
    });

    // Ed25519 signature = 64 bytes → 128 hex chars
    let ed25519_sig = [0xABu8; 64];
    group.bench_function("ed25519_64bytes", |b| {
        b.iter(|| hex::encode(black_box(&ed25519_sig)))
    });

    group.finish();
}

fn bench_sse_parser(c: &mut Criterion) {
    use vllm_proxy_rs::proxy::SseParser;

    let mut group = c.benchmark_group("sse_parser");

    // Single chunk with ID + DONE
    let single_chunk = b"data: {\"id\":\"chatcmpl-abc123\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: [DONE]\n\n";
    group.bench_function("single_chunk_with_done", |b| {
        b.iter(|| {
            let mut parser = SseParser::new();
            parser.process_chunk(black_box(single_chunk));
        })
    });

    // Many small chunks (typical streaming)
    let chunks: Vec<Vec<u8>> = (0..50)
        .map(|i| {
            format!(
                "data: {{\"id\":\"chatcmpl-abc123\",\"choices\":[{{\"delta\":{{\"content\":\"word{i}\"}}}}]}}\n\n"
            )
            .into_bytes()
        })
        .collect();
    group.bench_function("50_streaming_chunks", |b| {
        b.iter(|| {
            let mut parser = SseParser::new();
            for chunk in &chunks {
                parser.process_chunk(black_box(chunk));
            }
        })
    });

    // Chunk split across boundaries
    let part1 = b"data: {\"id\":\"chatcmpl-ab";
    let part2 = b"c123\",\"choices\":[]}\n\ndata: [DO";
    let part3 = b"NE]\n\n";
    group.bench_function("split_chunks", |b| {
        b.iter(|| {
            let mut parser = SseParser::new();
            parser.process_chunk(black_box(part1.as_slice()));
            parser.process_chunk(black_box(part2.as_slice()));
            parser.process_chunk(black_box(part3.as_slice()));
        })
    });

    group.finish();
}

fn bench_cache(c: &mut Criterion) {
    use vllm_proxy_rs::cache::ChatCache;

    let cache = ChatCache::new("test-model", 1200);

    c.bench_function("cache_set", |b| {
        let value = r#"{"text":"hash1:hash2","signature_ecdsa":"0xabc...","signing_address_ecdsa":"0xdef...","signature_ed25519":"123...","signing_address_ed25519":"456..."}"#;
        let mut i = 0u64;
        b.iter(|| {
            let key = format!("chatcmpl-{i}");
            cache.set_chat(black_box(&key), black_box(value));
            i += 1;
        })
    });

    // Pre-populate for get benchmark
    let cache = ChatCache::new("test-model", 1200);
    for i in 0..1000 {
        cache.set_chat(&format!("chatcmpl-{i}"), "cached-value");
    }

    c.bench_function("cache_get_hit", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let key = format!("chatcmpl-{}", i % 1000);
            let _ = cache.get_chat(black_box(&key));
            i += 1;
        })
    });

    c.bench_function("cache_get_miss", |b| {
        b.iter(|| {
            let _ = cache.get_chat(black_box("nonexistent-key"));
        })
    });
}

fn bench_eip191_prefix(c: &mut Criterion) {
    let text = typical_sign_text();

    c.bench_function("eip191_prefix_construction", |b| {
        b.iter(|| {
            let message = black_box(&text);
            let prefix = format!("\x19Ethereum Signed Message:\n{}", message.len());
            let mut prefixed = Vec::with_capacity(prefix.len() + message.len());
            prefixed.extend_from_slice(prefix.as_bytes());
            prefixed.extend_from_slice(message.as_bytes());
            black_box(prefixed);
        })
    });
}

fn bench_parse_upstream_error(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_upstream_error");

    let nested = br#"{"error":{"message":"model not found","type":"not_found"}}"#;
    group.bench_function("nested_format", |b| {
        b.iter(|| vllm_proxy_rs::proxy::parse_upstream_error(black_box(nested)))
    });

    let flat = br#"{"object":"error","message":"context length exceeded","type":"BadRequestError","code":400}"#;
    group.bench_function("flat_format", |b| {
        b.iter(|| vllm_proxy_rs::proxy::parse_upstream_error(black_box(flat)))
    });

    let invalid = b"not json at all";
    group.bench_function("invalid", |b| {
        b.iter(|| vllm_proxy_rs::proxy::parse_upstream_error(black_box(invalid)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ecdsa_sign,
    bench_ed25519_sign,
    bench_sign_chat,
    bench_sha256_hex,
    bench_hex_encode,
    bench_sse_parser,
    bench_cache,
    bench_eip191_prefix,
    bench_parse_upstream_error,
);
criterion_main!(benches);
