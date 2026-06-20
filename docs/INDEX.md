# Documentation Index

This repository keeps code, configs, and experiment reports together. Start
with the README, then use this index for deeper references.

## Project Reports

- [Qwen3-1.7B post-training experiment report](reports/qwen3_1_7b_post_training_experiment_report.md)
- [Stage 3 Wonderland experiment report](reports/stage3_wonderland_experiment_report.md)
- [Stage 3 data audit](../reports/stage3_bit_data_audit.md)

## Operations

- [Stage 1 server guide](STAGE1_SERVER.md)
- [Server evaluation guide](SERVER_EVALUATION.md)
- [Experiment notes](experiment-notes.md)

## Design Plans

- [Three-stage BF16 LoRA roadmap](plans/2026-06-14-three-stage-bf16-lora-roadmap.md)
- [Stage 1 No Robots implementation](plans/2026-06-14-stage1-no-robots-implementation.md)
- [Evaluation framework](plans/2026-06-14-evaluation-framework.md)

## Generated Artifacts Policy

Training outputs, adapters, model weights, SwanLab logs, Hugging Face caches,
and generated instruction JSONL files are ignored by git. Human-readable
reports that explain the experiment are stored under `docs/reports/`.
