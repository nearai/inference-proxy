# Protected AWS AppConfig smoke test

`../appconfig_aws_smoke.sh` is an opt-in, real-AWS verification lane. It uses
the AWS CLI AppConfig APIs to create an application, environment, hosted
free-form profile, JSON Schema validator, hosted versions, and a zero-minute
deployment strategy. It starts the production proxy image beside the official
AWS AppConfig Agent and this deterministic backend fixture.

The Agent image is pinned in the script to the current `2.x` multi-architecture
index digest. The Agent receives only the three caller-supplied temporary AWS
credential variables. The proxy receives the Agent URL and AppConfig identity
only. The script never prints credential values.

The control-plane profile and account confirmation are deliberately mandatory.
Before running, use a dedicated test account and temporary Agent credentials
with `appconfig:StartConfigurationSession` and `appconfig:GetLatestConfiguration`
on the test application. The control-plane profile must be allowed to create
and delete the temporary AppConfig resources.

Example invocation:

```bash
AWS_PROFILE=appconfig-smoke \
AWS_REGION=us-west-2 \
AWS_ACCOUNT_ID=123456789012 \
APPCONFIG_SMOKE_CONFIRM=I_UNDERSTAND_REAL_AWS_APPCONFIG_SMOKE:123456789012 \
AWS_APPCONFIG_AGENT_ACCESS_KEY_ID=... \
AWS_APPCONFIG_AGENT_SECRET_ACCESS_KEY=... \
AWS_APPCONFIG_AGENT_SESSION_TOKEN=... \
PROXY_IMAGE=123456789012.dkr.ecr.us-west-2.amazonaws.com/inference-proxy@sha256:... \
scripts/appconfig_aws_smoke.sh
```

The script does not run in ordinary CI. Its bounded wait deadline defaults to
180 seconds and can be changed with `APPCONFIG_SMOKE_TIMEOUT_SECS` (maximum
600). `PROXY_IMAGE` must use an immutable digest. `docker`, `aws`, `jq`, and
`curl` are required.

The AWS contracts used here are documented in the [AWS CLI
`create-configuration-profile` reference](https://docs.aws.amazon.com/cli/latest/reference/appconfig/create-configuration-profile.html),
the [AWS CLI `create-hosted-configuration-version`
reference](https://docs.aws.amazon.com/cli/latest/reference/appconfig/create-hosted-configuration-version.html),
the [AWS CLI `start-deployment` reference](https://docs.aws.amazon.com/cli/latest/reference/appconfig/start-deployment.html),
and the [AppConfig Agent container configuration
reference](https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-integration-containers-agent-configuring.html).
