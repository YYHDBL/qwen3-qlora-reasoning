# AGENTS.md

## Project

Staged BF16 LoRA post-training for `Qwen3-1.7B-Base`, with optional 4B
configuration support. The repository covers:

1. Stage 1 No Robots instruction following
2. Stage 1.5 strict-format / stop behavior
3. Stage 2 thinking protocol warmup
4. Stage 3 Wonderland cold-start SFT data and experiment scripts

The codebase is mixed Chinese/English and intentionally keeps experiment
reports alongside runnable scripts.

## Commands

```bash
# CPU/offline development
pip install -r requirements-dev.txt

# GPU training stack
pip install -r requirements-stage1-gpu.txt
pip install -r requirements-gpu.txt

# Focused tests
python -m pytest tests/test_stage3_wonderland_cold_start.py \
  tests/test_stage2_thinking_data.py \
  tests/test_stage1_5_strict_data.py -q

# Full tests
python -m pytest -q
```

## Architecture

### Adapter Chain

```text
Qwen3-1.7B-Base
  -> Stage 1 No Robots
  -> Stage 1.5 strict-format
  -> Stage 2 thinking warmup
  -> Stage 3 Wonderland cold-start
```

Every formal stage writes to a new output directory. Do not overwrite previous
adapters.

### Gated Training Pipeline

| Mode | Purpose | Gate |
|------|---------|------|
| `overfit` | Prove the setup can learn a tiny batch | Loss decrease |
| `smoke` | Short end-to-end check | Prior gate + reload |
| `formal` | Full configured run | Prior gates + audits |

Training produces `outputs/{experiment_name}/{mode}/...`, plus resolved config,
metrics, logs, and adapter artifacts.

### lm_head Bottleneck

LoRA `all-linear` with a frozen `lm_head` was not enough for reliable
`<|im_end|>` learning. Keep:

```yaml
lora:
  modules_to_save: ["lm_head"]
```

Removing this is a known regression risk.

### Data Generation Rules

- Stage 3 SFT may only read `stage3_sft_pool` from
  `splits/wonderland_split_seed42.json`.
- Wonderland validation/test must not be read for SFT generation.
- Long solver traces may be saved to debug files, but training completions use
  compressed traces only.
- Assistant completions must not contain handwritten `<|im_end|>` or
  `\boxed{}` residue.
- Qwen3 tokenizer length checks are required for Stage 3 generation.

## Conventions

- Use `rg` / `rg --files` for searching.
- Preserve existing stage outputs and adapters unless explicitly asked.
- Write files atomically when scripts generate artifacts.
- Keep generated data under ignored paths such as `data/instruction/` and
  `outputs/`.
- Do not commit secrets. Config `swanlab_api_key` values must remain `null`;
  use `SWANLAB_API_KEY` from the environment.
