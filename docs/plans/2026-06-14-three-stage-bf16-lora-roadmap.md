# Three-Stage BF16 LoRA Roadmap

**Goal:** Turn `Qwen/Qwen3-4B-Base` into a general instruction model, compare
instruction-data mixtures, and then specialize the selected instruction
adapter for Wonderland reasoning without overwriting any prior experiment.

**Architecture:** Use one YAML-driven training entry point and shared modules
for model/tokenizer loading, conversational data validation, assistant-only
labels, LoRA construction, experiment manifests, callbacks, adapter reload,
and evaluation. All GPU work runs on the server. The local repository only
runs unit tests against lightweight fixtures and never downloads model
weights.

**Tech Stack:** Python 3.11, PyTorch CUDA, Transformers, Datasets, PEFT, TRL
`SFTTrainer`, Accelerate, BF16 LoRA, PyYAML, TensorBoard, pytest.

---

## 1. Target Repository Layout

```text
configs/
├── stage1_no_robots.yaml
├── stage2_mixed_instruction.yaml
└── stage3_wonderland.yaml

src/
├── common/
│   ├── config.py
│   ├── experiment.py
│   └── prompt_format.py
├── data_processing/
│   ├── instruction_data.py
│   ├── dolly.py
│   ├── wonderland_chat.py
│   └── validation.py
├── training/
│   ├── model_loader.py
│   ├── lora.py
│   ├── label_audit.py
│   ├── callbacks.py
│   └── train_sft.py
└── evaluation/
    ├── instruction_eval.py
    ├── wonderland_eval.py
    └── evaluate.py

scripts/
├── prepare_stage1.py
├── prepare_stage2.py
├── prepare_stage3.py
├── inspect_batch.py
├── overfit_smoke.py
└── verify_adapter_reload.py

data/
├── instruction/
│   ├── stage1/
│   ├── stage2/
│   └── replay/
└── eval/
    ├── instruction_dev.jsonl
    └── instruction_test.jsonl

outputs/
├── stage1_no_robots/
├── stage2_mixed_instruction/
└── stage3_wonderland/
```

Every output directory contains:

```text
resolved_config.yaml
environment.json
dataset_manifest.json
token_report.json
batch_audit.json
checkpoints/
adapter/
trainer_state.json
train_metrics.json
eval_metrics.json
logs/
conclusion.md
```

Existing output directories cause a hard failure unless an explicit resume
flag points to a checkpoint inside that same experiment.

## 2. Shared Training Contract

- Base model: `Qwen/Qwen3-4B-Base`.
- Precision: BF16 model weights and BF16 compute; no 4-bit loading.
- Chat formatting: tokenizer's Qwen3 Chat Template.
- Conversation schema: `messages: list[{role, content}]`.
- Loss: assistant messages only.
- Tokenizer EOS and chat-turn terminator are separate concepts:
  - `<|endoftext|>` is the tokenizer-level EOS/PAD token in the published
    `Qwen/Qwen3-4B-Base` tokenizer configuration.
  - `<|im_end|>` terminates each message emitted by the Qwen3 Chat Template.
  - Assistant SFT labels include the assistant content and its trailing
    `<|im_end|>`, but not user/system messages.
  - Chat generation explicitly stops on `<|im_end|>` and also accepts
    `<|endoftext|>` as a fallback EOS. Token IDs are resolved from the loaded
    tokenizer and recorded rather than hard-coded.
  - Dataset preparation must not append an extra `<|endoftext|>` after every
    chat message; the Chat Template owns message boundaries.
- LoRA: `target_modules="all-linear"`, `bias="none"`.
- Initial LoRA candidate:
  - `r=32`
  - `lora_alpha=64`
  - `lora_dropout=0.05`
- Initial optimizer candidate:
  - `adamw_torch_fused`
  - learning rate `1e-4`
  - cosine schedule
  - warmup ratio `0.03`
- Gradient checkpointing enabled; model `use_cache=False`.
- TF32 enabled on Ampere/Ada GPUs.
- Packing disabled for the first smoke tests to keep label audits simple.
- Formal packing is enabled only after assistant masks are verified on packed
  and unpacked batches.
- Seed, dataset revisions, model revision, tokenizer revision, package
  versions, Git commit, and CUDA/GPU details are recorded.

The existing Wonderland `max_length=512` is not reused blindly for
instruction data. Stage 1 first computes Chat Template token distributions;
the Stage 1 maximum length is selected from that report, initially capped at
2048 for the smoke run.

## 3. Mandatory Preflight Gates

No formal training starts until all gates pass:

1. Dataset schema and role-order validation.
2. No empty message content or unsupported roles.
3. Dataset ID uniqueness and fixed split hashes.
4. Chat Template rendering inspection.
5. Token-length and truncation report.
6. Assistant-token mask is nonempty.
7. User/system/control-token labels are `-100`.
8. Assistant answer and terminating `<|im_end|>` tokens carry labels.
9. One decoded token/label example is saved for human inspection.
10. Sixteen-example overfit test shows a clear loss decrease.
11. Smoke adapter saves and reloads successfully.
12. Reloaded adapter generates with the same evaluation code.

TRL supports `assistant_only_loss=True` for conversational datasets and
automatically patches known Qwen3 templates, but the project still verifies
the actual assistant mask rather than trusting the flag.

## 4. Fixed Instruction Evaluation Dev/Test Sets

Create two small repository-owned JSONL evaluation sets that never enter
training:

- `instruction_dev.jsonl`: visible during development and used for validator,
  prompt, decoding, checkpoint, and Stage 1 versus Stage 2 model-selection
  decisions.
- `instruction_test.jsonl`: frozen after creation, excluded from tuning and
  routine error inspection, and run only after the experiment configuration
  and selected instruction adapter are fixed.

The two files use disjoint prompt intents and paraphrases while covering the
same evaluation categories:

- direct instruction following
- exact output strings
- JSON-only output
- fixed item counts
- extraction and classification
- rewriting with explicit constraints
- system-role adherence
- multi-turn context retention
- stop behavior
- prompts designed to expose ordinary text continuation

Each record stores:

```json
{
  "id": "...",
  "category": "...",
  "messages": [{"role": "user", "content": "..."}],
  "validator": {"type": "exact|regex|json|line_count|contains"},
  "expected": {}
}
```

Base, Stage 1, and Stage 2 use identical per-split prompts, decoding
parameters, and validators. Development reports include instruction success,
format success, EOS/stop success, generated token length, and
continuation-style failure count. Test reports expose aggregate and
per-category metrics; routine development must not repeatedly inspect
individual test failures. Open-ended quality is sampled from dev for manual
review; no paid LLM judge is needed for the first iteration.

Stage 1 versus Stage 2 selection uses training validation loss, the No Robots
validation split, and `instruction_dev.jsonl`. After the winning instruction
adapter and decoding configuration are frozen, run the Base model and that
single selected adapter once on `instruction_test.jsonl` for the final
comparison. Persist the test file hash and test-run count in the experiment
manifest.

## 5. Stage 1: No Robots

Dataset: `HuggingFaceH4/no_robots`.

- Preserve official `messages` exactly.
- Preserve `prompt_id` and `category`.
- Add `source="HuggingFaceH4/no_robots"`.
- Use the dataset configuration's official fixed splits:
  - `train`: 9,500 training records
  - `test`: 500 validation records
- Record Hub revision and Arrow fingerprints.
- Do not merge the official validation split into training.

Stage 1 sequence:

1. Prepare and validate the dataset on the GPU server.
2. Generate token-length and assistant-mask reports.
3. Run the fixed instruction dev evaluation on the unmodified Base model.
4. Overfit 16 examples for approximately 30-50 optimizer steps.
5. Run a smoke SFT on 128-256 training examples for 10-20 optimizer steps.
6. Save `instruction-no-robots` smoke adapter.
7. Start a fresh process and reload the adapter.
8. Evaluate the reloaded adapter on a small fixed dev subset.
9. Stop and request confirmation.

The formal one-epoch Stage 1 training starts only after this checkpoint is
approved.

## 6. Stage 2: No Robots + Dolly Ablation

Stage 2 always starts again from the original Qwen3 Base model.

- The Stage 2 mixed-data pool includes all 9,500 No Robots training records.
- Target an initial No Robots:Dolly ratio of approximately 2:1 by effective
  supervised assistant tokens, not by row count. Select approximately 5,000
  Dolly rows, then adjust the deterministic Dolly sample count after
  tokenization to approach that token ratio while retaining every No Robots
  row.
- Sample Dolly deterministically and stratify by `category`.
- Convert Dolly to messages:
  - user content is `instruction`
  - append a clearly delimited `context` block only when nonempty
  - assistant content is `response`
- Preserve `source`, `category`, original row index, and sample seed.
- Hold out a separate stratified Dolly validation subset before sampling the
  training mixture.
- Use the same No Robots validation and fixed instruction dev/test sets as
  Stage 1.

Stage 2 uses a matched training-token control:

1. Tokenize the complete mixed pool after formatting and truncation are
   fixed. Define `B_mix` as the exact number of non-`-100` assistant label
   tokens in one deterministic pass over that pool.
2. Train the mixed Stage 2 adapter from the original Base for `B_mix`
   effective supervised tokens.
3. Train a separate No-Robots-only control adapter from the same original
   Base for the same `B_mix` budget within a documented tolerance, initially
   0.5%. Cycle the deterministic No Robots training order only after all
   9,500 rows have been consumed once.
4. Use audited token schedules and derived `max_steps`; do not compare one
   Stage 1 epoch directly with one larger Stage 2 mixed epoch as if their
   compute were equal.
5. Keep the original Stage 1 one-epoch adapter as the learning milestone.
   The primary Stage 2 data-mixture ablation is mixed adapter versus the
   Stage 2 matched-budget No-Robots-only control.
6. Report effective supervised tokens, total input tokens, row repetitions,
   optimizer steps, and elapsed GPU time for all three instruction adapters.

The Stage 2 mixed run and its matched control use the same model, LoRA,
optimizer, effective batch target, decoding, and evaluation settings. Their
intended independent variable is the instruction-data mixture, not additional
training compute.

## 7. Stage 3: Wonderland Specialization

Select the best instruction adapter using validation results, not test data.

- Stage 3 is continued PEFT training: load the original Base model, attach the
  selected Stage 1 or Stage 2 instruction adapter with `is_trainable=True`,
  and initialize Stage 3 from those existing LoRA weights.
- Start a new optimizer and scheduler for Stage 3. Do not merge the selected
  adapter into the Base model and do not initialize a second unrelated
  adapter unless a later explicit ablation requires it.
- Save Stage 3 into a new directory; never modify the selected adapter.
- Convert Wonderland records to messages:
  - user: original puzzle prompt
  - assistant: answer only
- Start with an 80:20 Wonderland:instruction replay mixture controlled by
  effective supervised assistant tokens, not row count.
- Enforce the ratio over each optimizer update window, which may span several
  micro-batches under gradient accumulation. The sampler selects homogeneous
  source batches where practical and tracks cumulative source tokens.
- Permit a small configurable deviation per update window, then correct the
  running ratio in subsequent windows. Report the achieved ratio by
  supervised tokens, total tokens, rows, micro-batches, and optimizer steps.
- Do not rely on random concatenation alone: different answer lengths would
  make an 80:20 row ratio materially different from the intended loss-token
  ratio.
- Preserve Wonderland task type and instruction source/category metadata.
- Evaluate both:
  - Wonderland validation with overall and per-task metrics
  - the unchanged fixed instruction dev set during model selection
  - the frozen instruction test set only after Stage 3 settings are fixed
- Record instruction-retention deltas relative to the selected Stage 1/2
  adapter.

The protected Wonderland test split remains untouched until all training and
model-selection decisions are frozen.

## 8. Initial Configuration Strategy

Safe smoke defaults before confirming actual GPU memory:

```yaml
model:
  id: Qwen/Qwen3-4B-Base
  dtype: bfloat16
  attn_implementation: sdpa

lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: all-linear

training:
  max_length: 2048
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  effective_batch_size: 32
  gradient_checkpointing: true
  assistant_only_loss: true
  packing: false
  learning_rate: 1.0e-4
  num_train_epochs: 1
  bf16: true
  tf32: true
```

Batch size is adjusted only after recording peak allocated/reserved memory
from the smoke run. Flash Attention is optional and excluded initially to
reduce installation and compatibility risk. `num_train_epochs: 1` is the
Stage 1 formal-run default; both Stage 2 runs replace the epoch stop condition
with audited `max_steps` schedules targeting `B_mix`.

## 9. Dependencies

### Shared/local validation

- Python 3.11
- PyYAML
- pytest
- jsonschema

### GPU training

- PyTorch CUDA build matching the server driver
- Transformers
- Datasets
- PEFT
- TRL
- Accelerate
- safetensors
- TensorBoard
- sentencepiece/tokenizers as resolved by Transformers

Do not install bitsandbytes for these experiments. After the Stage 1
compatibility smoke succeeds, export the exact environment to
`requirements-lock.txt` or `conda-lock.yml`; do not rely on broad version
ranges for later stages.

## 10. Information Needed From the GPU Server

Before implementing the runnable Stage 1 commands, collect:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
df -h .
```

The standard RTX 4090 has 24 GB VRAM, so the reported 48 GB must be verified.
The result determines smoke batch size and whether 2048-token batches fit
comfortably.

Also confirm:

- repository absolute path on the server
- model cache path and at least 40 GB free disk
- whether `hf-mirror.com` must remain configured
- whether TensorBoard can expose a port or logs should be downloaded

## 11. Stage 1 Implementation Stop Point

The next implementation batch creates only:

- shared YAML/config and experiment utilities
- No Robots preparation and reports
- fixed instruction dev/test sets and validators
- Chat Template and assistant-label audits
- BF16 LoRA smoke/overfit runner
- adapter save/reload verification
- Stage 1 server commands and tests

It does not run the formal one-epoch job and does not implement Stage 2 or
Stage 3 training. After the Stage 1 smoke artifacts are reviewed, execution
stops for confirmation.
