# Qwen3-1.7B LoRA Reasoning Lab

> A staged post-training research project for turning `Qwen3-1.7B-Base` into a
> chat-capable, strict-format, thinking-protocol model and cold-starting it on
> Wonderland rule induction.

This repository is an end-to-end engineering record of a small-model
post-training pipeline. It covers data construction, Qwen3 chat-template
handling, assistant-only label audits, BF16 LoRA SFT, adapter reload checks,
strict-format regression, thinking-protocol evaluation, and Wonderland
reasoning diagnostics.

The goal is not to ship a general-purpose instruction model. The goal is to
build a reproducible experimental chain and expose the real failure modes:
stop-token calibration, strict short-answer control, `<think>` protocol
stability, solver-distilled compressed CoT, and the arithmetic / algorithmic
limits of a 1.7B model.

## Highlights

| Area | What This Repo Implements |
|---|---|
| Base model | `models/Qwen3-1.7B-Base` local path, with YAML support for `Qwen/Qwen3-4B-Base` experiments |
| Training | BF16 LoRA SFT, `all-linear`, `r=32`, `alpha=64`, `modules_to_save=["lm_head"]` |
| Loss | Assistant-only supervision through Qwen3-compatible training chat templates |
| Stage 1 | No Robots instruction following and stop-token calibration |
| Stage 1.5 | Strict-format / answer-only / JSON-only / stop behavior replay |
| Stage 2 | Thinking protocol warmup with no-think replay |
| Stage 3 | Wonderland cold-start data generation, smoke/formal runs, per-task diagnostics |
| Audits | Prompt split leakage, token length, final-answer parse, `<think>` closure, `\boxed{}` residue, `<|im_end|>` residue |

## Training Roadmap

```text
Qwen3-1.7B-Base
  -> Stage 1   No Robots instruction-following adapter
  -> Stage 1.5 strict-format / stop adapter
  -> Stage 2   thinking warmup adapter
  -> Stage 3   Wonderland cold-start adapter
```

Current conclusion: Stage 3 SFT establishes formatting and improves simple
pattern-matching tasks, especially numeral conversion. It does not reliably
solve hard Wonderland tasks such as gravity, unit conversion, cipher, bit
manipulation, or symbolic equations on a 1.7B model. See the reports below for
the full analysis.

## Results Snapshot

| Stage | Main Result |
|---|---|
| Stage 1 | LoRA-only learned answer intent but failed to stop; saving `lm_head` made `<|im_end|>` rank top-1 in the stop overfit setup |
| Stage 1.5 | Strict-format stop accuracy reached about 99%; answer-only and JSON-only behavior became stable |
| Stage 2 | Thinking protocol success reached about 92% while no-think prompts stayed clean |
| Stage 3 v0_1 | Format cold-start worked, but reasoning accuracy remained low |
| Stage 3 v0_2 | Enriched CoT lifted numeral accuracy to a peak of 82.1% |
| Stage 3 v0_3 | Median-based traces and shorter cipher traces improved format stability but did not solve arithmetic tasks |

For the full narrative, read:

- [Qwen3-1.7B post-training experiment report](docs/reports/qwen3_1_7b_post_training_experiment_report.md)
- [Stage 3 Wonderland experiment report](docs/reports/stage3_wonderland_experiment_report.md)
- [Stage 3 data audit](reports/stage3_bit_data_audit.md)

## Repository Layout

```text
configs/                         YAML configs for staged experiments
data/
  raw/                           Wonderland train CSV source
  processed/                     Prepared train/validation/test JSONL
  eval/                          Small instruction-eval fixtures
docs/
  INDEX.md                       Documentation map
  reports/                       Human-readable experiment reports
reports/                         Machine/data audit reports
scripts/
  generate_stage1_5_strict_data.py
  generate_stage2_thinking_data.py
  generate_stage3_wonderland_cold_start.py
  generate_stage3_v0_2.py
  generate_stage3_v0_3.py
  run_stage3_*.py
  stage3_diagnostic.py
  stage3_trace_audit.py
  stage3_reasoners/              Local deterministic reasoner wrappers
src/
  common/                        Config, experiment metadata, prompt formatting
  data_processing/               Dataset preparation and splitters
  evaluation/                    Instruction and answer evaluation
  training/                      Qwen3 template, LoRA loading, SFT, label audit
  models/                        Local chat / inspection tools
tests/                           Offline unit tests
```

Generated artifacts are intentionally ignored:

- `outputs/`
- `data/instruction/`
- model weights and adapters
- Hugging Face caches
- local `.env` files

## Installation

For CPU-only development and tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

For training / evaluation on a BF16-capable NVIDIA GPU:

```bash
pip install -r requirements-stage1-gpu.txt
pip install -r requirements-gpu.txt
```

The project expects the local model path for 1.7B runs:

```text
models/Qwen3-1.7B-Base
```

`models/` is gitignored except for `models/README.md`.

## Secrets

Do not commit API keys. Config files keep:

```yaml
experiment:
  swanlab_api_key: null
```

Set the key through the environment instead:

```bash
cp .env.example .env
export SWANLAB_API_KEY=...
```

If a key was ever committed before, rotate it before publishing or using the
repository publicly.

## Core Workflows

### Stage 1: No Robots Data

```bash
python -m src.data_processing.instruction_data \
  --config configs/stage1_no_robots_qwen3_1_7b_local.yaml
```

### Label Audit

```bash
python -m src.training.label_audit \
  --config configs/stage1_no_robots_qwen3_1_7b_local.yaml
```

The audit verifies assistant-only masks, supervised token counts, and
`<|im_end|>` boundary supervision.

### SFT Training

Training is gated:

| Mode | Purpose | Gate |
|---|---|---|
| `overfit` | Prove the pipeline can learn a tiny set | Loss must drop enough |
| `smoke` | Short end-to-end sanity run | Adapter reload and metrics check |
| `formal` | Full configured run | Requires previous gates and audits |

```bash
python -m src.training.train_sft \
  --config configs/stage2_thinking_warmup.yaml \
  --mode smoke
```

### Stage 3 Data Generation

The current cold-start generator only reads IDs listed in
`splits/wonderland_split_seed42.json` under `stage3_sft_pool`.

```bash
python scripts/generate_stage3_wonderland_cold_start.py
```

It writes:

```text
data/instruction/stage3_wonderland_cold_start/train.jsonl
data/instruction/stage3_wonderland_cold_start/dev.jsonl
data/instruction/stage3_wonderland_cold_start/report.json
data/instruction/stage3_wonderland_cold_start/audit.md
data/instruction/stage3_wonderland_cold_start/manual_review.md
data/instruction/stage3_wonderland_cold_start/debug/raw_traces.jsonl
```

The generator requires a real Qwen3 tokenizer for token-length checks. It does
not fall back to character estimates.

### Stage 3 Experiment Scripts

```bash
python scripts/run_stage3_smoke.py
python scripts/run_stage3_formal_v0_1.py
python scripts/run_stage3_formal_v0_2.py
python scripts/stage3_diagnostic.py
python scripts/stage3_trace_audit.py
```

These scripts are research utilities; inspect the script headers and output
paths before launching GPU jobs.

## Evaluation

Instruction following:

```bash
python -m src.evaluation.instruction_eval \
  --config configs/stage1_no_robots_qwen3_1_7b_local.yaml \
  --split dev \
  --output-dir outputs/eval/instruction-dev
```

Wonderland binary / protocol probes:

```bash
python scripts/eval_wonderland_binary.py
python scripts/eval_stage2_thinking.py
```

## Testing

Offline tests:

```bash
python -m pytest tests/test_stage3_wonderland_cold_start.py \
  tests/test_stage2_thinking_data.py \
  tests/test_stage1_5_strict_data.py -q
```

Full suite:

```bash
python -m pytest -q
```

The full suite requires the Python dependencies in `requirements-dev.txt`.

## Engineering Notes

- `modules_to_save=["lm_head"]` is not optional for this project. It is the
  observed fix for stop-token calibration under LoRA SFT.
- The training template and inference template intentionally differ. Training
  requires generation markers for assistant-only loss.
- Stage 3 compressed CoT is intentionally short. Long solver traces are kept
  only in debug files and are not used as training completions.
- Wonderland validation/test must not be used for SFT data generation.

## Project Hygiene

- Contribution workflow: [CONTRIBUTING.md](CONTRIBUTING.md)
- Secret handling: [SECURITY.md](SECURITY.md)
- Documentation map: [docs/INDEX.md](docs/INDEX.md)

## Status

This is a research codebase with preserved experiment scripts and reports. It
is suitable for reproducing the staged training workflow, auditing data
generation, and studying small-model failure modes. It is not packaged as a
library and does not include model weights or adapters.
