"""带有延迟 PEFT 导入的 LoRA 配置构造。"""

from __future__ import annotations

from typing import Any, Mapping


def build_lora_config(config: Mapping[str, Any]) -> Any:
    """从 YAML ``lora`` 段构造 PEFT ``LoraConfig``。

    参数说明：
    - r (rank): LoRA 低秩矩阵的秩。典型值 8-64。r 越大适配器容量越大但也越慢/越吃显存。
      秩为 16 时每个 adapter 大约增加 0.1% 的参数量。
    - lora_alpha: 缩放因子，实际缩放为 alpha/r。通常设为 2*r（如 r=16, alpha=32）。
      这样初始时 adapter 贡献接近零，训练稳定。
    - lora_dropout: LoRA 层的 dropout 率，正则化手段。典型值 0.05-0.1。
    - target_modules: 要应用 LoRA 的模块列表。Qwen3 常用 ["q_proj","k_proj","v_proj","o_proj",
      "gate_proj","up_proj","down_proj"]，覆盖所有线性层。
    - bias: "none" 表示不训练 bias；"all" 训练所有 bias；"lora_only" 只训练 LoRA 附带的 bias。
    - task_type: TaskType.CAUSAL_LM 表示因果语言模型任务。
    """
    from peft import LoraConfig, TaskType

    value = config["lora"]
    # 将 YAML 中的字符串 "CAUSAL_LM" 转为 PEFT 枚举值 TaskType.CAUSAL_LM
    task_type = getattr(TaskType, str(value["task_type"]))
    return LoraConfig(
        # r: 低秩分解的秩，控制 adapter 的参数量
        r=int(value["r"]),
        # lora_alpha: adapter 输出的缩放系数，实际缩放因子 = lora_alpha / r
        lora_alpha=int(value["lora_alpha"]),
        # lora_dropout: 在 adapter 层中使用的 dropout 概率
        lora_dropout=float(value["lora_dropout"]),
        # target_modules: 应用 LoRA 的目标层名称列表
        target_modules=value["target_modules"],
        # bias: 控制原始模型 bias 参数的训练方式
        bias=str(value["bias"]),
        # task_type: 任务类型，CAUSAL_LM 对应因果语言模型（自回归生成）
        task_type=task_type,
    )
