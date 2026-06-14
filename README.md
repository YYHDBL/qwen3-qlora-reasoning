# qwen3-qlora-reasoning

Data preparation, tokenizer analysis, and a shared evaluation framework for
`Qwen/Qwen3-4B-Base`.

- Fixed sequence length: `512`
- Evaluation prompt: `{prompt}\n\nAnswer:`
- Model modes: BF16 Base, NF4 Base, and NF4 Base plus LoRA
- GPU server guide: [`docs/SERVER_EVALUATION.md`](docs/SERVER_EVALUATION.md)
