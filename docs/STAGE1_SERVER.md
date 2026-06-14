# Stage 1 GPU Server Guide

Stage 1 converts `Qwen/Qwen3-4B-Base` into a general instruction model with
BF16 LoRA and `HuggingFaceH4/no_robots`.

This guide stops after the overfit test, smoke run, adapter reload, and dev
evaluation. Do not start the formal one-epoch run until those artifacts have
been reviewed.

## 1. Check The Server

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h .
```

The standard RTX 4090 has 24 GB VRAM. Use the reported value when adjusting
the micro-batch size.

## 2. Create The Conda Environment

```bash
conda create -n qwen3-lora python=3.11 -y
conda activate qwen3-lora
python -m pip install --upgrade pip
python -m pip install -r requirements-stage1-gpu.txt
python -m pip install -r requirements-dev.txt
```

Verify CUDA and BF16 before downloading data or model files:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("bf16:", torch.cuda.is_bf16_supported())
PY
```

## 3. Optional Hugging Face Mirror

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/with/at/least/40GB/huggingface
```

Do not store an access token in repository files:

```bash
export HF_TOKEN="your-token-if-required"
```

## 4. Run Local Unit Tests On The Server

```bash
python -m pytest -q
```

## 5. Prepare No Robots

This downloads the dataset only. It maps the Hub dataset's official `train`
split to train and official `test` split to validation.

```bash
python -m src.data_processing.instruction_data \
  --config configs/stage1_no_robots.yaml
```

Expected artifacts:

```text
data/instruction/stage1/
├── train.jsonl
├── validation.jsonl
├── dataset_manifest.json
└── dataset_report.json
```

## 6. Audit Tokens And Assistant-Only Labels

This downloads the tokenizer, not model weights:

```bash
python -m src.training.label_audit \
  --config configs/stage1_no_robots.yaml
```

The command fails if:

- the assistant mask is empty
- user or system tokens enter the supervised span
- truncation removes the assistant turn terminator
- `<|im_end|>` is not the final supervised token

It creates `token_report.json` and `batch_audit.json` in the prepared data
directory.

`<|endoftext|>` remains the tokenizer EOS/PAD token. `<|im_end|>` terminates
Qwen3 chat messages and is included in assistant labels. Generation accepts
both IDs as stopping tokens.

## 7. Evaluate The Unmodified Base On Instruction Dev

```bash
python -m src.evaluation.instruction_eval \
  --config configs/stage1_no_robots.yaml \
  --split dev \
  --output-dir outputs/stage1_no_robots/base-dev
```

Do not run `instruction_test.jsonl` during development.

## 8. Run The Sixteen-Example Overfit Gate

```bash
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml \
  --mode overfit
```

The gate requires the final logged loss to be below 80% of the first logged
loss. Inspect:

```text
outputs/stage1_no_robots/overfit/overfit_passed.json
```

## 9. Run The Smoke SFT

The default smoke run uses 256 records and 20 optimizer steps:

```bash
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml \
  --mode smoke
```

If 2048 tokens do not fit, reduce only the micro-batch size first:

```bash
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml \
  --mode smoke \
  --set training.per_device_train_batch_size=1 \
  --set training.gradient_accumulation_steps=32
```

Use a new `experiment.output_root` override when repeating an experiment.
Existing output directories are intentionally not overwritten.

## 10. Reload The Smoke Adapter In A Fresh Process

```bash
python -m src.training.verify_adapter \
  --config configs/stage1_no_robots.yaml \
  --adapter-path outputs/stage1_no_robots/smoke/adapter \
  --output outputs/stage1_no_robots/smoke/adapter_reload.json
```

Then run the unchanged dev evaluation:

```bash
python -m src.evaluation.instruction_eval \
  --config configs/stage1_no_robots.yaml \
  --split dev \
  --adapter-path outputs/stage1_no_robots/smoke/adapter \
  --output-dir outputs/stage1_no_robots/smoke/dev-eval
```

Stop here and review:

- Base versus smoke dev metrics
- overfit loss decrease
- `batch_audit.json`
- peak GPU memory and elapsed time
- adapter reload result

The formal command exists but remains blocked until all preflight artifacts
are present:

```bash
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml \
  --mode formal
```
