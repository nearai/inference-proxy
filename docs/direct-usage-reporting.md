# Direct usage reporting

This describes repository code behavior, not deployed runtime verification.

Direct Cloud API key requests report chat usage to `POST /v1/internal/usage`
with the service token and the organization, workspace, and API key identity
returned by key validation. Proxy-token requests do not produce these reports.

`input_tokens` is the total prompt count, including cached input.
`cache_read_tokens` is the cached subset taken only from the provider's
`usage.prompt_tokens_details.cached_tokens`. `output_tokens` is the completion
count. For example, 1858 prompt tokens with 1856 cached tokens and 8 completion
tokens report `input_tokens: 1858`, `cache_read_tokens: 1856`, and
`output_tokens: 8`. The proxy does not calculate prices or subtract cached input.

For a standalone JSON response, missing, null, or noninteger cached counts
report zero. Integers outside the signed 32-bit counter range also report zero.
Other integers are clamped to `0..max(input_tokens, 0)`. Existing validation
and no-billable-usage rules for the primary counters still apply. Embedding,
rerank, score, and image usage bodies retain their existing shapes.

Within an SSE stream, cache counts are cumulative. Missing, null, or noninteger
details preserve the previously observed count, clamped to the latest input
count; an explicit integer zero resets it. An out-of-range integer is invalid
zero. Each stream and agent-loop iteration starts without a cache observation.
Repeated cumulative chunks are never added together.

Both streaming proxy paths carry the observed cached count into their single
final usage report, including existing partial-billing paths for downstream
disconnects, incomplete EOF, and upstream errors. Interrupted responses remain
ineligible for signatures. Chat and text responses reconstructed from SSE
retain an earlier cached observation when later details omit it, while keeping
other provider detail fields. Without a prior observation, reconstruction does
not turn missing or null public cache details into zero; ordinary JSON and
forwarded SSE response shapes are also preserved.

Agent loops sum each iteration's final cached count once, alongside its prompt
and completion counts, including interrupted loop results. Fusion normalizes
each component's cache count before summing it and serializes the aggregate as
`usage.prompt_tokens_details.cached_tokens` for both response modes and usage
reporting. Existing Fusion failure-billing and selection policies are unchanged.
