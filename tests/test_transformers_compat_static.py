from pathlib import Path


def test_core_pipeline_generation_prompt_keeps_dialog_history():
    source = Path("src/training/_core_pipeline.py").read_text(encoding="utf-8")

    assert "gen_messages = messages[:-1]" in source
    assert "role'] != \"assistant\"" not in source
    assert 'role"] != "assistant"' not in source


def test_inspect_model_tokenizes_rendered_chat_without_duplicate_special_tokens():
    source = Path("src/models/inspect_model.py").read_text(encoding="utf-8")

    assert 'tokenizer(prompt, return_tensors="pt", add_special_tokens=False)' in source
