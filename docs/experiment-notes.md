# 实验学习笔记

## Qwen3-1.7B Base + LoRA SFT 后无法停止问题排查总结

### 1. 问题现象

使用：

- Qwen3-1.7B-Base
- No Robots
- BF16 LoRA SFT
- assistant-only loss

训练通用指令跟随能力。

训练过程本身正常：train_loss 正常下降、eval_loss 没有崩、grad_norm 稳定、mean_token_accuracy 有提升、LoRA adapter 能保存和加载。

但在生成评测中出现严重问题：

- instruction accuracy 很低
- stop accuracy = 0%
- continuation failure = 100%
- average generated tokens = max_new_tokens

模型不是完全不会答，而是：能开头答对，但不会停止，一直生成到 max_new_tokens 被截断。

典型例子：

```
User: Reply with exactly BLUE and nothing else.

Model:
BLUE 魔龙令牌
𫟦

I am happy to help with that.
I am here for you.
I am here for you.
...
```

```
User: Return only JSON with key sum and value 7.

Model:
Here is a JSON object ...
{
  "sum": 7
}
I hope this helps!
...
```

真实问题不是"模型完全没学"，而是：

1. 模型学到了一点回答意图
2. 但没有学会 strict format
3. 更严重的是没有学会输出 `<|im_end|>` 停止

### 2. 初始猜测

**猜测 1：数据分布不匹配。** No Robots 大部分是 helpful assistant 长回复数据，而 dev eval 是严格短答任务（如只输出 BLUE、只输出 JSON、输出 DONE 并停止、输出三行）。模型学到的是回答后继续解释、补充说明、礼貌收尾。这能解释模型为什么喜欢继续说。

**猜测 2：generation 参数不合适。** 手动测试使用 temperature=0.7, top_p=0.9, max_new_tokens=512，会放大低概率 token、废话、循环生成和奇怪字符。严格评测应使用 do_sample=false, temperature=0, max_new_tokens=32/64。

**猜测 3：prompt 构造错误。** 发现某条调试链路把完整训练 messages 传给了 generate，导致 user + assistant 标准答案 + assistant 新一轮生成。这需要修，但不是 instruction_dev 上 stop=0 的唯一解释。

**猜测 4：数据污染。** 输出里出现"魔龙令牌""𫟦"，但 grep 后这些文本只出现在模型输出里，不在训练数据、配置或源码中，基本排除。

### 3. 关键排查

#### 3.1 检查训练 label

No Robots 样本经过 training chat template 后：

```
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
<think>

</think>

assistant answer<|im_end|>
```

Mask 结果：

- user/system token: mask=0, label=-100
- assistant token: mask=1, label=token_id

`<|im_end|>` 确实在 assistant labels 里。结论：assistant-only loss 基本正确，`<|im_end|>` 没有被 mask 掉，训练目标里确实监督了结束符。

#### 3.2 检查 Base vs LoRA 手动生成

Base 模型对 "Reply with exactly BLUE in uppercase and nothing else." 输出完全不像回答，基本是乱码或重复外文 token。LoRA 后模型至少能输出 "BLUE / All caps is BLUE..."，说明 LoRA 已经让模型从纯续写模式变成了有一定 assistant 回答意图。Stage 1 不是完全失败。

#### 3.3 构造 stop overfit 数据

构造了小型 stop overfit 数据集（38 条训练 / 9 条验证），覆盖 single-word、JSON、classification、extract、stop_behavior。典型样本：

```
User: Reply with exactly BLUE and nothing else.
Assistant: BLUE
```

目标是验证：如果数据专门强调短答和停止，LoRA 能不能学会 `<|im_end|>`。

#### 3.4 三轮实验

**实验 1：** formatting_func + 无 assistant_only_loss → loss 降到 1.6，stop rate = 0%，模型输出完答案后继续循环。

**实验 2：** 直接传 dataset，让 SFTTrainer 使用 `{% generation %}` 做 assistant-only masking → loss 降到 2.7，stop rate 仍然 = 0%，循环模式变化但本质没变。

**实验 3：** LoRA config 加入 `modules_to_save=["lm_head"]` → loss 降到 0.00004，stop rate = 100%，instruction_eval 上 stop accuracy 从 0% 跳到 100%。这一步是关键证据。

#### 3.5 `<|im_end|>` logit probe

在答案末尾，下一 token 应该是 `<|im_end|>` 时，检查模型对 `<|im_end|>` 的预测排名：

- **Base:** rank ≈ 111000, prob ≈ 0, top-1 token = `<|endoftext|>`
- **LoRA all-linear:** rank ≈ 1960, prob ≈ 0.0004, top-1 token = 空格
- **LoRA + lm_head:** rank = 1, prob ≈ 0.9999, top-1 token = `<|im_end|>`

关键发现：LoRA all-linear 不是完全没学，它已经把 `<|im_end|>` 从 11 万名推到约 2000 名；但 rank=2000 仍然远远不够生成时被选中。加入 lm_head 后，`<|im_end|>` 直接变成 top-1。

### 4. 最终结论

核心不是 No Robots 数据完全不行，也不是 LoRA 完全没有学到东西，而是：

- Qwen3-1.7B-Base 原本没有稳定 chat stop 行为
- LoRA all-linear 能推动 hidden state，但无法充分把 `<|im_end|>` 推到 top-1
- **冻结的 lm_head 成为 stop token 学习的关键瓶颈**
- 训练 lm_head 后，模型才能稳定把答案末尾映射成 `<|im_end|>`

严谨表述：在当前 Qwen3-1.7B-Base + LoRA all-linear + No Robots / stop data 设置下，冻结 lm_head 是 stop failure 的主要瓶颈。LoRA 能学习回答意图，也能显著提升 `<|im_end|>` 排名，但不足以让它成为 top-1。加入 `modules_to_save=["lm_head"]` 后，stop 行为恢复，说明输出层校准对 Base → Instruct 阶段非常关键。

### 5. 工程教训

#### 教训 1：loss 下降不等于生成行为正确

训练时 train_loss 下降、eval_loss 稳定、mean_token_accuracy 提升，只能说明 teacher-forcing 下预测 token 变好了。但生成时还要看是否回答用户、是否遵守格式、是否正常停止、是否打满 max_new_tokens、是否重复生成。**必须做 generation eval**。

#### 教训 2：Base → Instruct 不只是学内容，还要学 chat 协议

Base 模型需要学的不只是"问题应该怎么回答"，还包括：什么时候开始 assistant 回复、什么时候结束、如何输出 `<|im_end|>`、如何遵守 only / exactly / JSON 这类格式约束。这些能力在 Base 模型里可能并不稳定。

#### 教训 3：LoRA all-linear 对普通语义适配有效，但对特殊 token 输出可能不够

LoRA 改的是中间层，理论上可以通过改变 hidden state 间接提高 `<|im_end|>` logit。但在当前实验中，LoRA all-linear 只能把 `<|im_end|>` rank 从 111000 推到 1960，无法推到 top-1，所以生成时仍然不会停。说明特殊 token 的输出层校准可能需要更直接的训练。

#### 教训 4：严格短答数据和 helpful 长答数据要分开看

No Robots 更偏 helpful 长回复（解释、总结、推荐、开放 QA），能训练回答意图，但不一定训练 exact output、JSON only、single-word answer、immediate stop。Stage 1 可以学"像 assistant 一样回答"，但要学"短、准、停"，还需要 Stage 1.5 strict-format / stop 数据。

### 6. 后续方案

**主线：** Qwen3-1.7B-Base + LoRA all-linear + `modules_to_save=["lm_head"]` + No Robots，先保证模型能回答、能停止、能进入后续 Wonderland。然后追加 Stage 1.5 strict-format / stop SFT，数据包括 exact answer、JSON only、yes/no、classification、short extraction、DONE and stop。

**副线：** LoRA-only + strict stop data，研究如果不训练 lm_head，只靠更强 stop 数据和更大 LoRA rank，能不能学会停止。但从当前 probe 看，工程上更稳的是 LoRA + lm_head。

---

## 关于"指令跟随是不是本来就不适合用 LoRA？"

### 1. LoRA 可以做指令微调，但更适合"已有 Instruct 模型的领域适配"

LoRA 很适合已有 Instruct 模型做垂直领域适配、风格调整、工具调用格式适配、小数据业务微调。例如 Qwen3-Instruct + 医疗问答 LoRA、客服回复 LoRA、Wonderland 推理 LoRA。这种场景下 Base 模型已经有 chat 格式、assistant 行为、stop 能力、基本指令跟随，LoRA 只需要改变一部分行为。

### 2. 但 Base → Instruct 是更底层的后训练，不一定适合纯 LoRA-only

Base → 通用指令模型比领域 LoRA 更难。不仅要学任务知识，还要学 chat template、role awareness、assistant-only answer、stop token、format following、safety / refusal style、helpfulness 风格。工业里通常不会只依赖很小的 LoRA adapter，更常见的是全参数 SFT，或至少解冻更多关键模块。

### 3. 合理定位

这个项目不是失败，而是非常有价值地暴露了一个真实后训练问题：**Base 模型不是简单喂几千条 instruction + LoRA 就自然变成 Instruct**；chat protocol、stop token、lm_head 校准、strict format 都需要单独验证。

项目路线可以调整为：

1. Qwen3-1.7B-Base → No Robots LoRA + lm_head：建立基础 assistant 行为
2. Strict Stop / Format SFT：补齐短答和停止能力
3. Wonderland SFT：垂直推理能力

最终学到的是：LoRA 不只是"能不能训"，还要看训哪些参数、训什么数据、评测什么行为、生成时是否真的可用。
