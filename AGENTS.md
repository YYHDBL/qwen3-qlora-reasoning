# AGENTS.md

## Project

BF16 LoRA instruction fine-tuning for `Qwen/Qwen3-4B-Base` (alt: `models/Qwen3-1.7B-Base`). Two-stage: Stage 1 (instruction-following, implemented) / Stage 2 (reasoning, planned). Mixed Chinese/English codebase.

## Commands

```bash
# Install (order matters — each extends the previous)
pip install -r requirements.txt
pip install -r requirements-stage1-gpu.txt   # training deps (torch, transformers, peft, trl, accelerate)
pip install -r requirements-gpu.txt           # adds bitsandbytes for eval

# Run all offline tests (no GPU needed)
python -m pytest src/ tests/ -x -q

# Run a single test
python -m pytest tests/test_label_audit.py -x -q

# Data prep (downloads HuggingFaceH4/no_robots → JSONL)
python -m src.data_processing.instruction_data
```

## Architecture

### Gated training pipeline (must pass in order)

| Mode | Samples | Steps | Gate |
|------|---------|-------|------|
| `overfit` | 16 | 40 | Loss must drop to <80% of initial |
| `smoke` | 256 | 20 | Requires overfit pass + adapter reload success |
| `formal` | all | 3 epochs | Requires all prior gates + audit artifacts present |

Run as: `python -m src.training.train_sft --config configs/stage1_no_robots.yaml --mode overfit`

Training produces: `outputs/{experiment_name}/overfit|smoke|formal/{adapter,checkpoints,swanlab,logs}/` plus `resolved_config.yaml`, `train_metrics.json`, `conclusion.md`, and audit snapshots.

### The lm_head bottleneck (critical)

LoRA `all-linear` + frozen `lm_head` prevents the model from learning `<|im_end|>` (stop token). The fix in every config: `lora.modules_to_save: ["lm_head"]`. Never remove this or the model won't learn to stop generating.

### Config validation (enforced at runtime)

`validate_stage1_config()` in `src/common/config.py` checks:
- Only allowed model IDs: `Qwen/Qwen3-4B-Base` or `models/Qwen3-1.7B-Base`
- 4-bit/8-bit quantization is forbidden (`bf16` must be `true`)
- `assistant_only_loss` must be `true`
- `lora.target_modules` must be `all-linear`
- `dataset_id` must be `HuggingFaceH4/no_robots`
- Dev/test eval paths must differ

### Data processing paths

- **Path A (default):** `instruction_data.py` — downloads from HuggingFace Hub → `data/instruction/stage1/{train,validation}.jsonl` (messages format)
- **Path B (custom):** `prepare_dataset.py` — reads `data/raw/train.csv` (columns: `id, prompt, answer`) → classifies into 6 task types → stratified 80/10/10 split → `data/processed/*.jsonl`

### Chat templates

Two different Jinja2 templates for different purposes:
- `QWEN3_BASE_CHAT_TEMPLATE` — inference/generation
- `QWEN3_TRAINING_CHAT_TEMPLATE` — must contain `{% generation %}` markers; this is how `assistant_only_loss` identifies the assistant's turn to compute loss on

### Evaluation

Two separate systems:
- `instruction_eval.py` — format/stop/continuation metrics (YAML config driven)
- `evaluate.py` — reasoning answer accuracy with BF16/NF4/LoRA modes

## Conventions

- **Imports:** relative within `src/` (`from ..common.config import ...`), absolute `src.*` from `scripts/`
- **Lazy imports:** heavy libs (`torch`, `transformers`, `peft`, `trl`) imported inside functions in non-training modules to avoid import overhead
- **File I/O:** always write to `.tmp` then `os.replace()` for atomicity
- **Status logging:** `[HH:MM:SS] [stageN] message` to stderr
- **Type hints:** `from __future__ import annotations` in every file, typed APIs throughout
- **Seed:** 42 everywhere (configurable via YAML)
- **Config overrides:** CLI dot-notation: `--set training.max_length=1024`

## GPU requirements

BF16 training requires NVIDIA GPUs that support BF16 (RTX 3090/4090, A100, etc.). 4-bit evaluation additionally needs `bitsandbytes`.
