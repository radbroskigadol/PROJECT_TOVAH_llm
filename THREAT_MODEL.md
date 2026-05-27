# THREAT_MODEL.md — TOVAH v14.2.6

## Assets

- source/IP
- training corpus
- checkpoints
- API credentials
- host filesystem
- patch/promotion authority
- buyer cluster resources

## Threats

- generated patch weakens invariants
- accidental credential exposure through tools/logs
- untrusted corpus payloads influencing autonomous behavior
- checkpoint poisoning or stale resume
- runaway cluster jobs
- unsafe shell/code execution

## Controls currently present

- HoTT/formal patch certificate substrate
- fail-closed HoTT promotion behavior
- presale audit docs
- safe-mode runbook
- checkpoint manifests
- metadata-aware corpus training

## Controls still buyer-owned

- OS/container sandbox
- credential vaulting
- cluster quotas
- production authz/authn
- third-party security audit
