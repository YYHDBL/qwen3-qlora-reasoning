# models/

This directory holds local Hugging Face model weights.

## Do NOT commit model weights

Large model weight files (`*.safetensors`, `*.bin`, `*.pt`, `*.pth`) are
excluded by `.gitignore`. This directory should contain only this README in the
repository.

## Download example

```bash
huggingface-cli download Qwen/Qwen3-1.7B-Base \
  --local-dir models/Qwen3-1.7B-Base \
  --local-dir-use-symlinks False
```

After downloading, update your config `model.id` to point at the local path:

```yaml
model:
  id: models/Qwen3-1.7B-Base
```

## Directory layout

```
models/
  README.md              <- this file (committed)
  Qwen3-1.7B-Base/       <- local model (gitignored)
  Qwen3-4B-Base/         <- local model (gitignored)
  hf_cache/              <- optional HF cache symlinks (gitignored)
```
