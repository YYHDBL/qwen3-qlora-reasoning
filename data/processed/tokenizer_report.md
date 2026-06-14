# Tokenizer Length Analysis

- Model ID: `Qwen/Qwen3-4B-Base`
- Tokenizer: `Qwen2Tokenizer`
- Vocabulary size: `151643`
- EOS: `<|endoftext|>` (ID `151643`)
- PAD: `<|endoftext|>` (ID `151643`)
- BOS: `None` (ID `None`)
- Revision: `906bfd4b4dc7f14ee4320094d8b41684abff8539`
- Transformers: `5.12.0`
- Executed: `2026-06-13T13:23:01.238025+00:00`

## Candidate Training Format

```text
{prompt}

Answer:
{answer}{eos}
```

## Overall Statistics

| Group | Metric | Count | Min | Mean | Median | P90 | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | prompt_tokens | 9500 | 49 | 116.74 | 102.0 | 222.0 | 241.0 | 260.0 | 260 |
| all | answer_tokens | 9500 | 2 | 5.51 | 6.0 | 9.0 | 9.0 | 9.0 | 9 |
| all | full_sequence_tokens | 9500 | 52 | 122.25 | 108.0 | 231.0 | 250.0 | 269.0 | 269 |

## Statistics by Split

| Group | Metric | Count | Min | Mean | Median | P90 | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | prompt_tokens | 950 | 51 | 116.33 | 102.0 | 222.0 | 241.0 | 260.0 | 260 |
| test | answer_tokens | 950 | 2 | 5.53 | 6.0 | 9.0 | 9.0 | 9.0 | 9 |
| test | full_sequence_tokens | 950 | 54 | 121.85 | 108.0 | 231.0 | 250.0 | 269.0 | 269 |
| train | prompt_tokens | 7600 | 49 | 116.92 | 102.0 | 222.0 | 241.0 | 260.0 | 260 |
| train | answer_tokens | 7600 | 2 | 5.51 | 6.0 | 9.0 | 9.0 | 9.0 | 9 |
| train | full_sequence_tokens | 7600 | 52 | 122.44 | 108.0 | 231.0 | 250.0 | 269.0 | 269 |
| validation | prompt_tokens | 950 | 51 | 115.67 | 102.0 | 222.0 | 241.0 | 260.0 | 260 |
| validation | answer_tokens | 950 | 2 | 5.51 | 6.0 | 9.0 | 9.0 | 9.0 | 9 |
| validation | full_sequence_tokens | 950 | 54 | 121.17 | 108.0 | 231.0 | 250.0 | 269.0 | 269 |

## Statistics by Task Type

| Group | Metric | Count | Min | Mean | Median | P90 | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bit_manipulation | prompt_tokens | 1602 | 203 | 231.56 | 241.0 | 260.0 | 260.0 | 260.0 | 260 |
| bit_manipulation | answer_tokens | 1602 | 9 | 9 | 9.0 | 9.0 | 9.0 | 9.0 | 9 |
| bit_manipulation | full_sequence_tokens | 1602 | 212 | 240.56 | 250.0 | 269.0 | 269.0 | 269.0 | 269 |
| cipher | prompt_tokens | 1576 | 82 | 117.37 | 117.0 | 141.0 | 145.0 | 155.0 | 168 |
| cipher | answer_tokens | 1576 | 4 | 5.31 | 5.0 | 6.0 | 7.0 | 7.0 | 8 |
| cipher | full_sequence_tokens | 1576 | 87 | 122.69 | 122.0 | 146.0 | 151.0 | 160.0 | 174 |
| gravity | prompt_tokens | 1597 | 103 | 128.57 | 128 | 148.0 | 149.0 | 151.0 | 152 |
| gravity | answer_tokens | 1597 | 4 | 6.05 | 6 | 7.0 | 7.0 | 7.0 | 7 |
| gravity | full_sequence_tokens | 1597 | 109 | 134.62 | 134 | 155.0 | 156.0 | 157.0 | 159 |
| numeral | prompt_tokens | 1576 | 49 | 60.65 | 61.0 | 68.0 | 69.0 | 71.0 | 72 |
| numeral | answer_tokens | 1576 | 2 | 3.25 | 3.0 | 4.0 | 5.0 | 5.0 | 5 |
| numeral | full_sequence_tokens | 1576 | 52 | 63.9 | 64.0 | 71.0 | 73.0 | 74.25 | 76 |
| symbolic_transform | prompt_tokens | 1555 | 51 | 71.56 | 70 | 91.0 | 93.0 | 96.0 | 98 |
| symbolic_transform | answer_tokens | 1555 | 2 | 3.47 | 3 | 5.0 | 5.0 | 5.0 | 5 |
| symbolic_transform | full_sequence_tokens | 1555 | 54 | 75.03 | 73 | 95.0 | 97.0 | 100.0 | 103 |
| unit_conversion | prompt_tokens | 1594 | 71 | 88.38 | 89.0 | 103.0 | 104.0 | 104.0 | 104 |
| unit_conversion | answer_tokens | 1594 | 5 | 5.91 | 6.0 | 6.0 | 6.0 | 6.0 | 6 |
| unit_conversion | full_sequence_tokens | 1594 | 76 | 94.28 | 95.0 | 109.0 | 110.0 | 110.0 | 110 |

## Recommended max_length

- Recommended: `512`
- Overflow count: `0`
- Overflow ratio: `0.0`
- Basis: `train, validation`
- Reason: 512 is the smallest allowed candidate that fully covers train and validation. Choosing the smallest complete candidate limits sequence-length memory and compute cost on a 24GB RTX 4090.

## Longest 20 Samples

| ID | Split | Task type | Prompt | Answer | Full sequence |
|---|---|---|---:|---:|---:|
| 0031df9c | train | bit_manipulation | 260 | 9 | 269 |
| 00890aff | train | bit_manipulation | 260 | 9 | 269 |
| 008b52fd | validation | bit_manipulation | 260 | 9 | 269 |
| 009a74b6 | train | bit_manipulation | 260 | 9 | 269 |
| 012fb81b | train | bit_manipulation | 260 | 9 | 269 |
| 016c474c | train | bit_manipulation | 260 | 9 | 269 |
| 030479a6 | train | bit_manipulation | 260 | 9 | 269 |
| 04c44df4 | train | bit_manipulation | 260 | 9 | 269 |
| 04d492a9 | train | bit_manipulation | 260 | 9 | 269 |
| 0520a6ec | train | bit_manipulation | 260 | 9 | 269 |
| 0528d502 | train | bit_manipulation | 260 | 9 | 269 |
| 069dbaab | train | bit_manipulation | 260 | 9 | 269 |
| 08615ada | test | bit_manipulation | 260 | 9 | 269 |
| 0a195a74 | train | bit_manipulation | 260 | 9 | 269 |
| 0a50c4a8 | train | bit_manipulation | 260 | 9 | 269 |
| 0a5e80bf | train | bit_manipulation | 260 | 9 | 269 |
| 0abfab8b | train | bit_manipulation | 260 | 9 | 269 |
| 0b3cf93f | train | bit_manipulation | 260 | 9 | 269 |
| 0c7acd69 | test | bit_manipulation | 260 | 9 | 269 |
| 0cb88778 | train | bit_manipulation | 260 | 9 | 269 |
