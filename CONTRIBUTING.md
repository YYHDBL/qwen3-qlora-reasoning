# Contributing

This is a research repository. Contributions should preserve reproducibility
and avoid mixing generated artifacts with source code.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

GPU training requires:

```bash
pip install -r requirements-stage1-gpu.txt
pip install -r requirements-gpu.txt
```

## Before Committing

Run focused offline tests for the stage you touched. For Stage 1.5/2/3 data
generation:

```bash
python -m pytest tests/test_stage3_wonderland_cold_start.py \
  tests/test_stage2_thinking_data.py \
  tests/test_stage1_5_strict_data.py -q
```

For Python syntax checks:

```bash
python -m py_compile scripts/generate_stage3_wonderland_cold_start.py
```

## Repository Rules

- Do not commit model weights, LoRA adapters, checkpoints, SwanLab logs, or
  generated instruction JSONL files.
- Do not commit API keys or local credentials.
- Keep configs reproducible: use explicit paths, seeds, and output roots.
- Preserve previous stage adapters and outputs. New experiments should write to
  new directories.
- Wonderland validation/test splits must not be used for SFT data generation.

## Documentation

Human-readable experiment summaries belong under `docs/reports/`. Machine audit
outputs can live under `reports/` when they are small and useful for review.
