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

- `src/analyze_tokens.py`
- `tests/test_analyze_tokens.py`
- `data/processed/tokenizer_report.json`
- `data/processed/tokenizer_report.md`

## Current Stop Point

Tokenizer analysis is complete. Base model evaluation has not started.
