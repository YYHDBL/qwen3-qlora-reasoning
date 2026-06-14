import pytest

from src.training.label_audit import audit_conversation


class FakeTokenizer:
    im_end_id = 99

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
    with pytest.raises(ValueError, match="terminating <\\|im_end\\|>"):
        audit_conversation(conversation(), FakeTokenizer(), max_length=5)
