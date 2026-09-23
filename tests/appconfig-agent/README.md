# Real AWS AppConfig Agent local test

Run `./run.sh` from this directory (or invoke it from the repository root as
`tests/appconfig-agent/run.sh`). The script builds the proxy with the
production `Dockerfile`, starts an official AWS AppConfig Agent in local
development mode, and always runs `docker compose down --volumes` on exit.

The test driver reaches the Agent only through the private Compose network. It
checks Agent `/ping`, the Agent's `Configuration-Version` response, baseline
admission at one request, a hot increase to three, a decrease to two while
three requests are active, invalid-document last-known-good retention, and a
rollback to one. It also checks `/v1/models` capacity, structured proxy log
version evidence, and that the proxy container ID and start time stay fixed.
The mock engine is a Python standard-library HTTP server with explicit hold,
release, and request-count controls; all configuration replacements use
`os.replace` so the Agent cannot read a partial JSON document.

AWS documents local development mode with `LOCAL_DEVELOPMENT_DIRECTORY`, the
`application:environment:profile` filename, and the Agent configuration URL:

- [Working with AWS AppConfig Agent local development mode](https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-agent-how-to-use-local-development.html)
- [Retrieving configuration data in ECS/EKS](https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-integration-containers-agent-retrieving-data.html)
- [AWS AppConfig Agent version history and `/ping`](https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-integration-lambda-extensions-versions.html)

The Compose image is pinned to the immutable amd64 manifest digest
`sha256:532da31adf639752ac919f703c6e0de3d82142132422e348624680b6156c1769`.
AWS Public ECR currently resolves the official `2.x` multi-architecture tag to
index digest
`sha256:5e23ec8b3eb883680cdf1e4224c8d0f99405e14eb67131a2def7316eaa418f1a`;
the platform manifest is used in Compose so Docker cannot retarget the test to
another architecture. The local probe recorded Agent `/ping` version
`2.0.210365` on 2026-09-22.

This is an opt-in Docker lane and is intentionally outside ordinary Rust CI;
the wiremock and gateway integration tests remain the pull-request checks.
