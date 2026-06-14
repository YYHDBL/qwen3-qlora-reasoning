# GPU Server Evaluation

The evaluation entry point is shared by:

- BF16 Base: `--model-mode bf16`
- NF4 4-bit Base: `--model-mode nf4`
- NF4 4-bit Base plus LoRA: `--model-mode lora`

All modes use:

- model: `Qwen/Qwen3-4B-Base`
- input format: `{prompt}\n\nAnswer:`
- `max_length=512`
- deterministic generation with `do_sample=False`
- the same answer parsers, metrics, and artifact schema

## Environment Setup

From the repository root:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-gpu.txt
```

Authenticate only when the server or Hugging Face rate limits require it:

```bash
export HF_TOKEN="..."
```

Do not put tokens in repository files or command history.

## First Validation Smoke Run

Run exactly three validation samples with the BF16 Base model:

```bash
.venv/bin/python -m src.evaluate \
  --model-id Qwen/Qwen3-4B-Base \
  --model-mode bf16 \
  --split validation \
  --limit 3 \
  --batch-size 1 \
  --max-length 512 \
  --max-new-tokens 64 \
  --output-dir outputs/validation-bf16-smoke-3
```

This creates:

```text
outputs/validation-bf16-smoke-3/
├── predictions.jsonl
├── error_cases.jsonl
├── metrics.json
└── run_config.json
```

## NF4 Base

Use the same command and change only the mode and output directory:

```bash
.venv/bin/python -m src.evaluate \
  --model-mode nf4 \
  --split validation \
  --limit 3 \
  --max-length 512 \
  --output-dir outputs/validation-nf4-smoke-3
```

## NF4 Base plus LoRA

```bash
.venv/bin/python -m src.evaluate \
  --model-mode lora \
  --adapter-path /path/to/lora-adapter \
  --split validation \
  --limit 3 \
  --max-length 512 \
  --output-dir outputs/validation-lora-smoke-3
```

## Protected Test Split

Development and model selection must use `validation`. The CLI rejects
`test` unless `--allow-test` is supplied explicitly. Do not inspect
individual test errors or use test metrics for tuning.

## Local Tests

The unit tests do not import or require PyTorch, bitsandbytes, PEFT, a GPU, or
model weights:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q
```
