# qwen3-qlora-reasoning

Data preparation, evaluation, and staged BF16 LoRA instruction tuning for
`Qwen/Qwen3-4B-Base`.

- Fixed sequence length: `512`
- Evaluation prompt: `{prompt}\n\nAnswer:`
- Model modes: BF16 Base, NF4 Base, and NF4 Base plus LoRA
- Code layout:
  - `src/data_processing/`: dataset preparation and split generation
  - `src/evaluation/`: tokenizer analysis and model evaluation
  - `src/common/`: shared prompt formatting
  - `src/training/`: Qwen3 Chat Template audits and BF16 LoRA SFT
- GPU server guide: [`docs/SERVER_EVALUATION.md`](docs/SERVER_EVALUATION.md)
- Stage 1 guide: [`docs/STAGE1_SERVER.md`](docs/STAGE1_SERVER.md)

Stage 1 uses `HuggingFaceH4/no_robots`, Qwen3's Chat Template,
assistant-only loss, and independent overfit/smoke/formal output
directories. Local tests do not load model weights or require a GPU.
