# SECURITY.md — TOVAH v14.2.6

TOVAH contains autonomous-kernel, tool-use, and patch-governance concepts. Treat
it as research software with potentially dangerous capabilities if connected to
real tools.

## Safe defaults for buyer evaluation

- Run pretraining/evals first, not live autonomy mode.
- Do not provide production credentials during evaluation.
- Disable or stub network/API tools unless needed.
- Keep patch promotion fail-closed.
- Use sandboxed environments for code-execution experiments.
- Run on disposable machines or containers for adversarial testing.

## High-risk surfaces

- live patch proposal/application
- shell/code execution tools
- external API tool access
- persistence migrations over important state
- autonomous loop with broad filesystem access

## Commercial handoff note

Security hardening should be treated as a buyer integration workstream before
production deployment.
