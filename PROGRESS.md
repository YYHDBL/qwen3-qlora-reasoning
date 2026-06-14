# Project Progress

## Completed: Dataset Preparation

- Validated and classified 9,500 records into six task types.
- Created deterministic stratified splits:
  - train: 7,600
  - validation: 950
  - test: 950
- Preserved the internal test split for final evaluation.
- Generated split manifests, hashes, and dataset validation reports.

## Completed: Tokenizer Length Analysis

- Model tokenizer: `Qwen/Qwen3-4B-Base`
- Candidate training text:

  ```text
  {prompt}

  Answer:
  {answer}{eos}
  ```

- No chat template or role tokens were used.
- No model weights, GPU code, QLoRA code, training code, or generation
  evaluation were used.
- Input JSONL files were not modified; before/after SHA256 hashes match.

### Full Sequence Token Distribution

| Scope | Count | Mean | Median | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 9,500 | 122.25 | 108 | 231 | 250 | 269 | 269 |
| Train | 7,600 | 122.44 | 108 | 231 | 250 | 269 | 269 |
| Validation | 950 | 121.17 | 108 | 231 | 250 | 269 | 269 |
| Test | 950 | 121.85 | 108 | 231 | 250 | 269 | 269 |

### max_length Decision

- Recommended `max_length`: **512**
- Basis: train and validation only
- Train + validation samples over 512: **0 / 8,550**
- All samples over 512: **0 / 9,500**
- Maximum observed full sequence: **269 tokens**

`512` is the smallest allowed candidate that fully covers train and
validation. Larger candidates add sequence-length memory and compute cost
without increasing coverage for this dataset, which matters for later
training on a 24GB RTX 4090.

### Artifacts

- `src/evaluation/analyze_tokens.py`
- `tests/test_analyze_tokens.py`
- `data/processed/tokenizer_report.json`
- `data/processed/tokenizer_report.md`

## Completed: Shared Evaluation Framework

- Added one prompt-format module shared by tokenizer analysis and evaluation.
- Added task-aware parsing and comparison for:
  - bit manipulation
  - gravity
  - unit conversion
  - numeral conversion
  - cipher text
  - symbolic transformation
- Numeric comparisons use `Decimal`, never `float`.
- Symbolic answers are preserved exactly.
- Metrics include overall and per-task parse, format, primary, strict, and
  normalized accuracy.
- Added one evaluation entry point for:
  - BF16 Base
  - NF4 4-bit Base
  - NF4 4-bit Base plus LoRA
- Evaluation outputs:
  - `predictions.jsonl`
  - `error_cases.jsonl`
  - `metrics.json`
  - `run_config.json`
- Model libraries are delayed imports, so local unit tests require no GPU.
- The protected test split requires explicit `--allow-test`.

### Server Preparation

- `requirements.txt`
- `requirements-dev.txt`
- `requirements-gpu.txt`
- `docs/SERVER_EVALUATION.md`

## Completed: Source Reorganization

- Grouped dataset preparation code under `src/data_processing/`
- Grouped tokenizer analysis and evaluation code under `src/evaluation/`
- Moved shared prompt formatting into `src/common/`
- Updated tests and documentation to use the package-based module paths

## Current Stop Point

## Implemented: Stage 1 No Robots Infrastructure

- Added the fixed `configs/stage1_no_robots.yaml` experiment configuration.
- Added official No Robots `train` and `test` preparation with:
  - exact `messages` preservation
  - role-order and field validation
  - resolved Hub revision and Arrow fingerprints
  - deterministic JSONL hashes and manifests
- Added separate repository-owned instruction evaluation sets:
  - `data/eval/instruction_dev.jsonl`
  - `data/eval/instruction_test.jsonl`
- Added deterministic exact, regex, JSON, line-count, and contains
  validators.
- The frozen instruction test split requires explicit opt-in and does not
  emit individual error cases by default.
- Added TRL's training-compatible Qwen3 Chat Template integration.
- Added assistant-only label audits that require the trailing
  `<|im_end|>` token to remain supervised after truncation.
- Added shared BF16 Base and LoRA loading with delayed GPU imports.
- Added overfit, smoke, and formal SFT modes with independent output
  directories.
- Added exact supervised and total training-token counters, peak GPU memory,
  elapsed time, environment metadata, and Adapter artifacts.
- Added fresh-process Adapter reload verification.
- Added `requirements-stage1-gpu.txt` without bitsandbytes while preserving
  the older NF4 evaluation environment separately.
- Added Conda server commands in `docs/STAGE1_SERVER.md`.

## Current Stop Point

Stage 1 code is ready for server-side data preparation, label audit, Base dev
evaluation, sixteen-example overfit, smoke SFT, and Adapter reload. No model
weights were downloaded and no local inference or training was run. Stop
after the smoke artifacts are reviewed; do not start the formal one-epoch
run yet.
