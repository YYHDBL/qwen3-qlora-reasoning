import sys
import types

import pytest

from src.training.chat_template import (
    QWEN3_BASE_CHAT_TEMPLATE,
    QWEN3_TRAINING_CHAT_TEMPLATE,
    configure_training_chat_template,
)


class FakeTokenizer:
    def __init__(self, chat_template):
        self.chat_template = chat_template

    def get_chat_template(self):
        return self.chat_template


def install_fake_trl(get_training_chat_template):
    module = types.ModuleType("trl.chat_template_utils")
    module.get_training_chat_template = get_training_chat_template
    package = types.ModuleType("trl")
    package.chat_template_utils = module
    sys.modules["trl"] = package
    sys.modules["trl.chat_template_utils"] = module


def uninstall_fake_trl():
    sys.modules.pop("trl.chat_template_utils", None)
    sys.modules.pop("trl", None)


def test_configure_training_chat_template_uses_trl_patch_when_available():
    tokenizer = FakeTokenizer("original")
    install_fake_trl(
        lambda processing_class: "{%- generation %}patched{%- endgeneration %}"
    )

    try:
        result = configure_training_chat_template(tokenizer)
    finally:
        uninstall_fake_trl()

    assert result == "{%- generation %}patched{%- endgeneration %}"
    assert tokenizer.chat_template == "{%- generation %}patched{%- endgeneration %}"


def test_configure_training_chat_template_falls_back_for_qwen3_base():
    tokenizer = FakeTokenizer(QWEN3_BASE_CHAT_TEMPLATE)

    def raising_patch(processing_class):
        raise ValueError("patching is not supported for this template")

    install_fake_trl(raising_patch)
    try:
        result = configure_training_chat_template(tokenizer)
    finally:
        uninstall_fake_trl()

    assert result == QWEN3_TRAINING_CHAT_TEMPLATE
    assert tokenizer.chat_template == QWEN3_TRAINING_CHAT_TEMPLATE
    assert "{%- generation %}" in result


def test_configure_training_chat_template_rejects_unknown_template():
    tokenizer = FakeTokenizer("unknown-template")

    def raising_patch(processing_class):
        raise ValueError("patching is not supported for this template")

    install_fake_trl(raising_patch)
    try:
        with pytest.raises(ValueError, match="not training-compatible"):
            configure_training_chat_template(tokenizer)
    finally:
        uninstall_fake_trl()
