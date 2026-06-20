# Security Policy

## Secrets

Never commit real API keys, access tokens, SSH keys, cloud credentials, or
service account files.

This project uses SwanLab for optional experiment tracking. Keep all committed
configs at:

```yaml
experiment:
  swanlab_api_key: null
```

Use the environment instead:

```bash
export SWANLAB_API_KEY=...
```

`.env` files are ignored by git. `.env.example` is safe to commit because it
contains placeholders only.

## If A Secret Was Committed

1. Rotate or revoke the exposed key immediately.
2. Remove the secret from the current tree.
3. Consider history rewriting before publishing to a public remote.

## Artifacts

Model weights, adapters, checkpoints, and generated datasets can be large and
may contain derived data. They are excluded from git by default.
