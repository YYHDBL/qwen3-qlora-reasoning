# Stage 3 Wonderland Cold-Start Data Audit

## Data Generation
- **Input CSV**: `data/raw/train.csv` (9500 rows)
- **Split Manifest**: `splits/wonderland_split_seed42.json`
- **Split Ratios**: {'stage3_sft_pool': 0.7, 'rl_train_pool': 0.15, 'rl_dev_pool': 0.15}
- **Pool Used**: `stage3_sft_pool` (6650 IDs total, 1121 bit_manipulation)
- **Max Prompts Sampled**: 512

## Data Statistics
- **Sample Total**: 949
- **Answer Only**: 512
- **Compressed CoT**: 437
- **Task Type Distribution**: {'bit_manipulation': 949}
- **Sample Type Distribution**: {'answer_only': 512, 'compressed_cot': 437}

### Duplicate Checks
- **Duplicate Source IDs**: 437 (extra occurrences: 437)
- **Duplicate Prompt Hashes**: 0 (extra occurrences: 0)

### Data Quality
- **`\\boxed` Residuals**: 0
- **`<|im_end|>` Residuals**: 0
- **`<think>` Open Tags**: 437
- **`</think>` Close Tags**: 437
- **Malformed Think Tags (CoT)**: 0

### Pool Boundary Check
- **All Source IDs in stage3_sft_pool**: True
- **Unique Source IDs in Output**: 512
- **Pool Size**: 6650

## Token Statistics (Qwen3 Tokenizer)
- **User Token Length**: min=210, max=276, mean=243.32, p50=248, p95=276
- **Assistant Token Length**: min=8, max=135, mean=56.07, p50=8, p95=126
- **Total Token Length**: min=218, max=411, mean=299.38, p50=275, p95=389
- **Exceeds 1024 Tokens**: 0

## Constraints Validation
- **read_validation_or_test**: False
- **trained_adapter**: False
- **data_source**: stage3_sft_pool (6650 IDs from splits/wonderland_split_seed42.json)
