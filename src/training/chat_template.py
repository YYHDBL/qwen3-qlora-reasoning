"""带有延迟 TRL 导入的 Qwen3 聊天模板辅助函数。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


IM_END_TOKEN = "<|im_end|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"

# Matches the published tokenizer chat template for Qwen/Qwen3-4B-Base.
QWEN3_BASE_CHAT_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""

# Matches the official TRL qwen3_training.jinja patch for assistant-only loss.
QWEN3_TRAINING_CHAT_TEMPLATE = """{#- Qwen3 聊天模板的训练变体（原始版本见 qwen3.jinja）。
     与原版的差异：
     - {%- if '</think>' in content %} -> {%- if '<think>' in content and '</think>' in content %}
       同时检查两个标签，避免模型只生成了一个标签时的边界情况。
     - 移除了 loop.index0 > ns.last_query_index 条件判断，始终包含 thinking 块。
       这使得模板在 [user, assistant] -> [user, assistant, tool] 转换时保持前缀不变。
     - 在 assistant 消息输出周围添加了 {% generation %} / {% endgeneration %}，
       以支持 SFT 训练中的 assistant-only loss 掩码。
-#}
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '<think>' in content and '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\\n' }}
        {%- generation %}
        {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
        {%- endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""


def _is_generation_compatible(chat_template: str) -> bool:
    return "{% generation" in chat_template or "{%- generation" in chat_template


def _fallback_training_template(tokenizer: Any, error: Exception) -> str | None:
    """当 TRL 的 ``get_training_chat_template`` 失败时，尝试从内置模板回退。

    触发条件：
    1. tokenizer 的原生 chat_template 与 QWEN3_BASE_CHAT_TEMPLATE 一致
       → 说明使用的是 Qwen3 官方模板，直接替换为内置的训练变体
    2. 错误消息明确说明模板"不兼容训练"或"不支持补丁"
       → 这种情况下没有本地回退方案，返回 None 调用方会抛异常
    """
    chat_template = tokenizer.get_chat_template()
    # 条件 1: 原生模板匹配 Qwen3 Base 模板 → 用内置训练模板替换
    if chat_template.strip() == QWEN3_BASE_CHAT_TEMPLATE.strip():
        return QWEN3_TRAINING_CHAT_TEMPLATE
    message = str(error)
    # 条件 2: TRL 明确报告不可补丁 → 返回 None，由调用方抛出错误
    if "not training-compatible" in message or "patching is not supported" in message:
        return None
    return None


def configure_training_chat_template(tokenizer: Any) -> str:
    """Attach a training-compatible Qwen3 chat template.

    回退机制说明：
    - 优先使用 TRL 的 ``get_training_chat_template``：它会从 HuggingFace Hub
      下载对应模型的训练变体模板（如果发布者提供了的话）
    - 如果 TRL 不可用（ImportError）或模板不兼容（ValueError），则尝试本地回退：
      * 如果是 Qwen3 Base 官方模板 → 替换为内置的 QWEN3_TRAINING_CHAT_TEMPLATE
      * 如果明确不可补丁 → 抛出 ValueError 阻止训练
    - 最终验证模板是否包含 {% generation %} 标记，确保 assistant-only loss 可用
    """
    template: str | None = None
    try:
        # 第 1 优先: TRL 库提供的训练模板（从 Hub 下载或自动补丁）
        from trl.chat_template_utils import get_training_chat_template

        template = get_training_chat_template(processing_class=tokenizer)
    except (ValueError, ModuleNotFoundError, ImportError) as exc:
        # 第 2 优先: 本地回退（内置的 Qwen3 训练模板变体）
        template = _fallback_training_template(tokenizer, exc)
        if template is None:
            raise ValueError(
                "The chat template is not training-compatible and this "
                "repository does not provide a fallback patch for it."
            ) from exc
    if template is not None:
        tokenizer.chat_template = template
    active_template = tokenizer.get_chat_template()
    # 最终验证: 训练模板必须包含 {% generation %} 标记
    #   - TRL 使用此标记来生成 assistant_tokens_mask（决定哪些 token 参与 loss 计算）
    #   - 没有此标记则 assistant_only_loss 无法工作
    if not _is_generation_compatible(active_template):
        raise ValueError("training chat template has no generation markers")
    # 保存到私有属性，供后续审计使用
    tokenizer._stage1_training_chat_template = active_template
    return active_template


def render_generation_prompt(
    tokenizer: Any, messages: Sequence[Mapping[str, str]]
) -> str:
    """将对话历史渲染为用于生成的 prompt 文本。

    - add_generation_prompt=True: 末尾追加 "<|im_start|>assistant\\n"，指示模型开始补全
    - enable_thinking=False: 不要求模型输出 <think> 推理过程，直接给出最终答案
      （评估阶段关注最终答案质量，thinking 会引入额外 token 和不确定性）
    """
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def resolve_stop_token_ids(tokenizer: Any) -> dict[str, int]:
    """解析用于生成停止条件的 token ID 映射。

    返回 {"im_end": 151645, "endoftext": 151643} 形式的字典。
    两个停止 token 的用途：
    - im_end (<|im_end|>): Qwen3 聊天格式的回合结束标记，正常结束的生成会输出此 token
    - endoftext (<|endoftext|>): EOS token，作为后备停止条件
    """
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    eos_id = tokenizer.eos_token_id
    unk_id = getattr(tokenizer, "unk_token_id", None)
    # 如果 im_end 无法识别（返回 None 或等于 UNK token），说明 tokenizer 未正确加载
    if im_end_id is None or im_end_id == unk_id:
        raise ValueError(f"tokenizer does not define {IM_END_TOKEN}")
    if eos_id is None:
        raise ValueError("tokenizer does not define an EOS token")
    return {"im_end": int(im_end_id), "endoftext": int(eos_id)}
