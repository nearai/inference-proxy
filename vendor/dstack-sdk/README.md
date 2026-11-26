# Dstack SDK client-only patch

Source: crates.io `dstack-sdk` 0.1.3, upstream commit
`f67b4f67ebabef0a27795705121698280a1038dc` (`sdk/rust`).
Download: <https://crates.io/api/v1/crates/dstack-sdk/0.1.3/download>.

SHA-256 of the vendored client files:

- `src/dstack_client.rs`: `ef3d9f87ad7ec1b481cc19cf1b23183e00dae52864c862507a78a1d57169a89b`
- `src/tappd_client.rs`: `990c9f02a46fdb4e8a567137a554cbdcc1efe470eb4fff4cd2c39db521df8593`

`DstackClient::info_raw` decodes the `tcb_info` JSON object and preserves nullable
measurements on AMD SEV-SNP guests. The Tappd client uses the upstream source.

This local patch removes the unused Ethereum module and its unconditional Alloy
dependency, which brings in `crypto-bigint` 0.5.5 affected by RUSTSEC-2026-0387.
Examples and upstream tests are omitted; no application calls the Ethereum helpers. Original source copyright
and license notices are preserved.

Remove this patch when an upstream SDK release supports platform-neutral `Info` and
allows using the clients without the affected Ethereum dependency. Do not restore
Alloy until its dependency tree passes the security audit.
