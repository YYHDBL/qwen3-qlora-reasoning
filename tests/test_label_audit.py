import pytest

from src.training.label_audit import audit_conversation, build_audit_reports


class FakeTokenizer:
    im_end_id = 99
    eos_token = "<|endoftext|>"
    eos_token_id = 100
    pad_token = None
    pad_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is True
        assert kwargs["return_assistant_tokens_mask"] is True
        return {
            "input_ids": [10, 11, 12, 20, 21, self.im_end_id],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "assistant_masks": [0, 0, 0, 1, 1, 1],
        }

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return self.im_end_id

    def decode(self, token_ids, **kwargs):
        return " ".join(str(value) for value in token_ids)


class BoundaryTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is True
        assert kwargs["return_assistant_tokens_mask"] is True
        return {
            "input_ids": [10, 20, 21, 99],
            "attention_mask": [1, 1, 1, 1],
            "assistant_masks": [0, 1, 1, 0],
        }


class MultiSpanTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is True
        assert kwargs["return_assistant_tokens_mask"] is True
        return {
            "input_ids": [10, 20, 21, 99, 11, 30, 31, 99],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
            "assistant_masks": [0, 1, 1, 0, 0, 1, 1, 0],
        }


def conversation():
    return {
        "id": "sample",
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    }


def test_audit_masks_non_assistant_and_trains_im_end():
    result = audit_conversation(conversation(), FakeTokenizer(), max_length=32)

    assert result["labels"] == [-100, -100, -100, 20, 21, 99]
    assert result["supervised_tokens"] == 3
    assert result["im_end_supervised"] is True
    assert result["truncated"] is False


def test_audit_accepts_im_end_as_boundary_without_supervising_it():
    result = audit_conversation(conversation(), BoundaryTokenizer(), max_length=32)

    assert result["labels"] == [-100, 20, 21, -100]
    assert result["supervised_tokens"] == 2
    assert result["im_end_supervised"] is False
    assert result["im_end_follows_supervised_span"] is True


def test_audit_rejects_empty_assistant_mask():
    tokenizer = FakeTokenizer()

    def empty_mask(messages, **kwargs):
        result = FakeTokenizer.apply_chat_template(tokenizer, messages, **kwargs)
        result["assistant_masks"] = [0] * 6
        return result

    tokenizer.apply_chat_template = empty_mask

    with pytest.raises(ValueError, match="no supervised assistant tokens"):
        audit_conversation(conversation(), tokenizer, max_length=32)


def test_audit_rejects_truncation_that_removes_im_end():
    with pytest.raises(ValueError, match="terminating <\\|im_end\\|> boundary"):
        audit_conversation(conversation(), BoundaryTokenizer(), max_length=3)


def test_audit_accepts_multiple_supervised_assistant_spans():
    result = audit_conversation(conversation(), MultiSpanTokenizer(), max_length=32)

    assert result["supervised_tokens"] == 4
    assert result["im_end_supervised"] is False
    assert result["im_end_follows_supervised_span"] is True
    assert result["supervised_span_count"] == 2
    assert result["ends_with_supervised_token"] is False


def test_audit_rejects_truncation_inside_supervised_span():
    with pytest.raises(ValueError, match="cuts through a supervised assistant span"):
        audit_conversation(conversation(), MultiSpanTokenizer(), max_length=6)


def test_audit_rejects_missing_im_end_boundary_after_complete_span():
    with pytest.raises(ValueError, match="terminating <\\|im_end\\|> boundary"):
        audit_conversation(conversation(), MultiSpanTokenizer(), max_length=7)


def test_batch_audit_uses_im_end_boundary_check_not_im_end_supervision(tmp_path):
    audited = audit_conversation(conversation(), BoundaryTokenizer(), max_length=32)
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")
    validation_path.write_text("{}\n", encoding="utf-8")
    token_report, batch_audit = build_audit_reports(
        audited_by_split={"train": [audited], "validation": [audited]},
        input_paths={"train": train_path, "validation": validation_path},
        model_id="Qwen/Qwen3-4B-Base",
        tokenizer=BoundaryTokenizer(),
    )

    assert token_report["im_end_token"] == "<|im_end|>"
    assert batch_audit["status"] == "passed"
    assert batch_audit["checks"]["im_end_boundaries_present"] is True
