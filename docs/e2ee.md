# Field encryption

E2EE encrypts supported request fields to the attested model key and response
fields to the client's reply key. Version 3 authenticates that reply key before
inference and makes the extended field policy mandatory.

## Version 3

Send `X-Signing-Algo`, `X-Client-Pub-Key`, and `X-Encryption-Version: 3`.
Each independently encrypted request field uses:

```text
AAD = UTF8("nearai-e2ee-request-v3") || suite_id_u8 || canonical_reply_public_key
```

| Algorithm | Suite ID | Canonical reply public key | AEAD |
| --- | --- | --- | --- |
| ECDSA | `0x01` | 65-byte uncompressed secp256k1 key, including `04` | AES-256-GCM |
| Ed25519 | `0x02` | 32-byte Ed25519 key, before conversion to X25519 | XChaCha20-Poly1305 |

ECDSA headers may contain the existing 64-byte form; the proxy adds `04` before
building AAD. Hex case does not change the AAD. The prefix is 22 UTF-8 bytes;
there are no separators, length fields, or JSON serialization in the AAD.
AAD is an AEAD parameter, not part of the prompt or ciphertext envelope.

The key agreement, HKDF-SHA256 labels (`ecdsa_encryption` / `ed25519_encryption`),
field shapes, and envelopes are unchanged. ECDSA envelopes contain the ephemeral
public key (65 bytes), nonce (12 bytes), and ciphertext with tag. Ed25519 uses an
X25519 ephemeral public key (32 bytes), nonce (24 bytes), and ciphertext with tag.
Use a fresh ephemeral key and nonce for each field, with the same request AAD.

The proxy validates every protected field before calling the model. Authentication
failure rejects the request; encrypted schemas must also decode as JSON. Optional
null values and empty strings retain their existing meaning; multimodal content
arrays and tool parameter schemas must be serialized and encrypted as strings.

V3 always enables the extended field policy, including when
`X-Encrypt-All-Fields` is absent or false:

- Chat requests: message content, reasoning, audio data, refusal, message names,
  function tool definitions (name, description, parameters), tool/function call
  names and arguments, and named tool/function choice.
- Chat responses and SSE: content, reasoning, audio data, refusal, tool/function
  names and arguments, tool result output, and logprob tokens and byte arrays,
  including top alternatives.
- Other endpoints retain their defined prompt/input/output transforms. For text
  completions, v3 additionally encrypts the entire non-null `choices[].logprobs`
  object as a JSON string, in both ordinary and streaming responses.

Responses use the validated reply key and existing encryption envelopes with
empty AAD. Clients must reject failed decryption and plaintext protected fields.
Protocol metadata and fields outside these transforms remain visible.

## Compatibility

V1 and v2 retain their existing field policy and wire formats. V3 requires server
support and must not fall back to an older version on error. Deploy proxy and
gateway support before enabling v3 clients.

Reply-key binding does not reject a replay with the original reply key; the new
response is still encrypted to that key. Replay prevention and authentication of
other request metadata are separate protocol properties.

The [shared test vectors](../tests/fixtures/e2ee-v3.json) contain public test keys
and independent AES-GCM/libsodium ciphertexts, also consumed by all four SDKs.
