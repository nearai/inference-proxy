# Field encryption

E2EE encrypts supported request fields to the attested model key and response
fields to the client's reply key. Version 3 authenticates the reply key and each
request field label before inference and applies the full protected field set
to each request and response.

## Version 3

### Request authentication

Send `X-Signing-Algo`, `X-Client-Pub-Key`, and `X-Encryption-Version: 3`.
Each independently encrypted request field uses:

```text
AAD = UTF8("nearai-e2ee-request-v3") || suite_id_u8 || canonical_reply_public_key
      || 0x00 || UTF8(field_label)
```

| Algorithm | Suite ID | Canonical reply public key | AEAD |
| --- | --- | --- | --- |
| ECDSA | `0x01` | 65-byte uncompressed secp256k1 key, including `04` | AES-256-GCM |
| Ed25519 | `0x02` | 32-byte Ed25519 key, before conversion to X25519 | XChaCha20-Poly1305 |

ECDSA headers may contain the 64-byte form without the `04` prefix; the proxy adds
it before building AAD. Hex case does not change the AAD. The domain prefix
`nearai-e2ee-request-v3` is 22 UTF-8 bytes. A single zero byte separates the
canonical key from the ASCII field label. AAD is an AEAD parameter, separate
from the prompt and ciphertext envelope.

Field labels use the paths below. `[]` is literal and applies to every array
position; for example, both `messages[0].content` and `messages[1].content` use
`messages[].content`.

| Request value | Field label |
| --- | --- |
| Message content, including serialized content arrays | `messages[].content` |
| Message reasoning, name, refusal | `messages[].reasoning_content`, `messages[].reasoning`, `messages[].name`, `messages[].refusal` |
| Message audio | `messages[].audio.data` |
| Message tool call name and arguments | `messages[].tool_calls[].function.name`, `messages[].tool_calls[].function.arguments` |
| Message legacy function call | `messages[].function_call.name`, `messages[].function_call.arguments` |
| Tool definition | `tools[].function.name`, `tools[].function.description`, `tools[].function.parameters` |
| Named tool/function choice | `tool_choice.function.name`, `function_call.name` |
| Other endpoint root fields, including each string in array inputs | `prompt`, `input`, `query`, `text_1`, `text_2` |
| Rerank document strings / document text | `documents[]` / `documents[].text` |

### Ciphertext format

Both suites derive the AEAD key with HKDF-SHA256, using `ecdsa_encryption` or
`ed25519_encryption` as the info label. Each field contains a hex-encoded envelope:
ECDSA uses an ephemeral secp256k1 public key (65 bytes), nonce (12 bytes), and
ciphertext with tag. Ed25519 uses an X25519 ephemeral public key (32 bytes),
nonce (24 bytes), and ciphertext with tag.
Use a fresh ephemeral key and nonce for each field, with its field-specific AAD.

Responses use the validated reply key and the same envelope layout with empty AAD.
Clients validate protected containers and field types, and reject authentication failures.
Missing or null optional containers and partial tool-call deltas remain valid.

### Field policy

V3 applies encryption to the following fields when present, independently of
`X-Encrypt-All-Fields`. Required fields are defined by each endpoint's API.

- Chat requests: message content, reasoning, audio data, refusal, message names,
  function tool definitions (name, description, parameters), tool/function call
  names and arguments, and named tool/function choice.
- Chat responses and SSE: content, reasoning, audio data, refusal, tool/function
  names and arguments, tool result output, and logprob tokens and byte arrays,
  including top alternatives.
- Text completions: request `prompt`, response `choices[].text`, and the entire
  non-null `choices[].logprobs` object serialized as JSON. Responses use the same
  field policy for JSON and SSE.

Other endpoint transforms:

| Endpoint | Request fields | Response fields |
| --- | --- | --- |
| Image generation/editing | `prompt` | `data[].b64_json`, `data[].revised_prompt` |
| Embeddings | `input` | `data[].embedding`, serialized as JSON |
| Audio transcription | `prompt` | `text` |
| Rerank | `query`, strings in `documents[]` or their `text` fields | `results[].document.text`, `results[].relevance_score` serialized as JSON |
| Score | `text_1`, `text_2` | `score`, serialized as JSON |

Rerank results are sorted by relevance inside the model proxy before scores are
encrypted. Gateways preserve that ordering.

Optional request fields may be omitted or `null`. Text fields may be empty
strings; non-empty protected text must authenticate and decrypt before inference.
Serialize multimodal content arrays and tool parameter schemas as JSON before
encrypting them; tool schemas must decrypt to valid JSON. Authentication failures
reject the request.

V3 chat and text completions reject enabled `return_token_ids` and
`return_prompt_text` options and any non-null `prompt_logprobs` (including `0`).
Chat and text-completion JSON/SSE responses retain only supported protocol
fields, with filtering inside nested containers. Engine extensions such as
`matched_stop`, `stop_reason`, token IDs, hidden states, and arbitrary metadata
are removed. Standard output logprob tokens and bytes remain encrypted.
Response IDs, model identifiers, usage counts, and completion controls remain visible.

For encrypted requests in every version, upstream HTTP and SSE errors use a
fixed message and safe status/type information.

## Compatibility

V1/v2 select the extended field policy with `X-Encrypt-All-Fields` and allow
plaintext in extended request fields for compatibility with legacy clients.
V3 requires support in the model proxy and any gateway on the request path.
Clients must not downgrade to an older protocol version on failure.

A replay with the original reply key can still be processed; its response is
encrypted to that key. Replay prevention and authentication of other request
metadata are separate protocol properties.

The [protocol test vectors](../tests/fixtures/e2ee-v3.json) contain public test keys
and independent AES-GCM/libsodium ciphertexts. Their format is shared with the
test fixtures in all four SDKs.
