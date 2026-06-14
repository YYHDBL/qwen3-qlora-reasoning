# Stage 1 No Robots Implementation Plan

> **For Codex:** Implement task-by-task with test-driven development.

**Goal:** Add a server-ready BF16 LoRA Stage 1 pipeline for
`HuggingFaceH4/no_robots` without downloading weights or running training
locally.

**Architecture:** Keep dataset validation, configuration, instruction
validators, manifests, and label audits as dependency-light Python modules.
Delay imports of PyTorch, Datasets, Transformers, PEFT, TRL, and Accelerate
until a GPU command runs. Use TRL's Qwen3 training-compatible Chat Template
and verify the actual assistant mask before training.

**Tech Stack:** Python 3.11, PyYAML, Transformers, Datasets, PEFT, TRL
`SFTTrainer`, Accelerate, PyTorch CUDA, pytest.

---

### Task 1: Configuration And Experiment Safety

**Files:**
- Create: `src/common/config.py`
- Create: `src/common/experiment.py`
- Create: `configs/stage1_no_robots.yaml`
- Test: `tests/test_stage1_config.py`

1. Test loading and validating the Stage 1 YAML.
2. Test that an existing output directory fails without explicit resume.
3. Implement configuration loading, dotted overrides, and atomic JSON/YAML
   artifact writing.
4. Record Git, package, CUDA, dataset, model, and tokenizer metadata when
   available.

### Task 2: No Robots Preparation

**Files:**
- Create: `src/data_processing/instruction_data.py`
- Test: `tests/test_instruction_data.py`

1. Test schema and role-order validation with local fixtures.
2. Test exact message preservation and deterministic hashes.
3. Test that official `train` and `test` map to train and validation.
4. Implement delayed `datasets.load_dataset` loading and write immutable
   JSONL, manifest, and report artifacts.

### Task 3: Instruction Dev/Test Evaluation

**Files:**
- Create: `data/eval/instruction_dev.jsonl`
- Create: `data/eval/instruction_test.jsonl`
- Create: `src/evaluation/instruction_eval.py`
- Test: `tests/test_instruction_eval.py`

1. Test exact, regex, JSON, line-count, and contains validators.
2. Test dev/test access policy and deterministic aggregate metrics.
3. Implement Qwen3 Chat Template generation, explicit stop token IDs, and
   predictions/errors/metrics/run-config artifacts.
4. Keep test prompts frozen and suppress individual test-error artifacts by
   default.

### Task 4: Chat Template And Assistant Label Audit

**Files:**
- Create: `src/training/chat_template.py`
- Create: `src/training/label_audit.py`
- Test: `tests/test_label_audit.py`

1. Test actual `assistant_masks` consumption with a fake tokenizer.
2. Test user/system labels are `-100`.
3. Test assistant content and trailing `<|im_end|>` are supervised.
4. Test truncation and empty-assistant-mask failures.
5. Implement token reports and decoded label inspection artifacts.

### Task 5: BF16 LoRA Trainer And Reload Verification

**Files:**
- Create: `src/training/model_loader.py`
- Create: `src/training/lora.py`
- Create: `src/training/train_sft.py`
- Create: `src/training/verify_adapter.py`
- Test: `tests/test_stage1_training.py`

1. Test modules import without GPU packages.
2. Test overfit/smoke/formal run limits and SFT argument construction.
3. Implement BF16 model loading, `all-linear` LoRA, assistant-only
   `SFTTrainer`, progress logging, token accounting, peak-memory reporting,
   adapter saving, and fresh-process reload checks.
4. Refuse formal mode unless all preflight artifact gates pass.

### Task 6: Server Commands And Project Status

**Files:**
- Create: `docs/STAGE1_SERVER.md`
- Modify: `requirements.txt`
- Modify: `requirements-dev.txt`
- Modify: `requirements-gpu.txt`
- Modify: `README.md`
- Modify: `PROGRESS.md`

1. Document Conda setup, mirror configuration, data preparation, label audit,
   Base dev evaluation, overfit, smoke, and adapter reload commands.
2. Run all local tests.
3. Verify all new CLI modules provide `--help` without importing GPU
   libraries.
4. Stop before formal one-epoch training.
