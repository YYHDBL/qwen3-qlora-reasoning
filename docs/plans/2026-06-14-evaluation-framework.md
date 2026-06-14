# Evaluation Framework Implementation Plan

**Goal:** Build a shared, locally testable evaluation framework for BF16,
NF4, and NF4 plus LoRA runs on `Qwen/Qwen3-4B-Base`.

**Architecture:** Keep prompt formatting, answer parsing, metrics, and result
writing independent from model loading. `evaluate.py` owns orchestration and
uses delayed imports only when a real Hugging Face generator is requested.
All modes consume the same records and produce the same artifact schema.

**Tech Stack:** Python standard library, Transformers, PyTorch, Accelerate,
bitsandbytes, PEFT, pytest.

---

### Task 1: Unified prompt formatting

**Files:**
- Create: `src/prompt_format.py`
- Modify: `src/analyze_tokens.py`
- Test: `tests/test_prompt_format.py`

1. Write failing tests for exact evaluation and training strings.
2. Implement `format_evaluation_prompt` and `format_training_text`.
3. Re-export the existing training function from `analyze_tokens`.
4. Run prompt and tokenizer-analysis tests.

### Task 2: Task-aware answer evaluation

**Files:**
- Create: `src/answer_evaluation.py`
- Test: `tests/test_answer_evaluation.py`

1. Write failing tests for all six task types.
2. Cover clean outputs, accepted answer labels, malformed outputs, Decimal
   comparison, text normalization, and symbolic character preservation.
3. Implement parsed, strict, normalized, and primary correctness fields.
4. Run answer evaluation tests.

### Task 3: Metrics aggregation

**Files:**
- Create: `src/evaluation_metrics.py`
- Test: `tests/test_evaluation_metrics.py`

1. Write failing tests for overall and per-task counts and rates.
2. Implement deterministic aggregation without model dependencies.
3. Verify JSON serialization.

### Task 4: Unified evaluation CLI

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

1. Write failing orchestration tests using an injected fake generator.
2. Implement validation JSONL loading, prompt generation, predictions,
   errors, metrics, and run configuration artifacts.
3. Add delayed BF16, NF4, and NF4 plus LoRA model loading.
4. Require explicit opt-in before evaluating the protected test split.
5. Run evaluation tests without importing GPU libraries.

### Task 5: Server packaging and documentation

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `requirements-gpu.txt`
- Create: `docs/SERVER_EVALUATION.md`
- Modify: `PROGRESS.md`

1. Document environment setup and the first three-sample validation run.
2. Record supported model modes and output artifacts.
3. State that no model inference was run locally.

### Task 6: Verification

1. Run the entire pytest suite.
2. Compile all Python modules.
3. Verify existing split hashes did not change.
4. Verify importing evaluation modules does not import `torch`,
   `transformers`, `bitsandbytes`, or `peft`.
5. Stop before Base model evaluation.
