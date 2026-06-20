# Qwen3-1.7B-Base 后训练实验报告：从 Instruction-Following 到 Wonderland 推理冷启动

## 1. 项目目标

本项目围绕 `Qwen3-1.7B-Base` 进行一条完整的后训练学习实验链路，目标不是训练工业级通用模型，而是系统掌握从 Base 模型到指令跟随、严格格式控制、thinking protocol，再到领域推理冷启动的完整过程。

最终任务为 Wonderland 规则推理数据集。每个样本给出若干 `input -> output` 示例，模型需要归纳隐藏规则，并对 query 输出最终答案。任务类型包括 bit manipulation、cipher、gravity、unit conversion、numeral、symbolic equation 等。最终答案通常非常短，例如 8-bit binary、Roman numeral、数值字符串或符号串。因此模型不仅需要会推理，还必须稳定满足：

* 只输出最终答案；
* 按要求输出 `<think>...</think>`；
* 不要求 thinking 时不能乱输出 `<think>`；
* 能正常输出 `<|im_end|>` 并停止；
* final answer 能被自动解析；
* 为后续可能的 RL reward 判分提供稳定格式。

整体 adapter 链路为：

```text
Qwen3-1.7B-Base
→ Stage 1 No Robots adapter
→ Stage 1.5 strict-format adapter
→ Stage 2 thinking warmup adapter
→ Stage 3 Wonderland cold-start adapter
```

本项目最终停在 Stage 3 SFT 阶段，暂不继续 RL。原因是 Stage 3 已经完成了格式冷启动和部分任务类目的推理提升，但 hard 类任务仍存在明显能力瓶颈，直接进入 RL 会面临 reward 稀疏和探索效率低的问题。

---

## 2. 训练基础设置

基础模型使用：

```text
models/Qwen3-1.7B-Base
```

训练方式为 BF16 LoRA SFT，不使用 QLoRA，不使用 4-bit 量化。

核心配置为：

```text
LoRA r = 32
LoRA alpha = 64
dropout = 0.05
target_modules = all-linear
modules_to_save = ["lm_head"]
assistant_only_loss = true
Base 主干冻结
LoRA + lm_head 可训练
```

所有阶段都在同一条 adapter 链上继续训练，不重新从 Base 初始化 LoRA，也不叠加多个 LoRA。

---

## 3. Stage 1：No Robots 基础指令跟随

### 3.1 阶段目标

Stage 1 的目标是让纯 Base 模型具备基础 assistant 行为，包括：

* 能理解 chat template；
* 能根据用户问题生成回答；
* 能输出 assistant 内容；
* 能在回答后正常停止；
* 初步具备 helpful assistant 风格。

训练数据为 No Robots 风格指令数据：

```text
train: 9500
validation: 500
```

### 3.2 核心问题：LoRA-only 学不好 stop token

早期实验发现，Base 模型几乎不能遵守简单 strict 指令，例如：

```text
Reply exactly BLUE
```

只训练 LoRA all-linear 时，模型能学到回答意图，能输出 `BLUE` 或 JSON 内容，但不会停止，通常会继续生成直到 `max_new_tokens` 打满。

早期 strict eval 结果为：

```text
stop accuracy = 0%
continuation failure = 100%
mean generated tokens = 256
```

随后做了 stop overfit 和 logit probe：

1. 只训练 LoRA、不训练 lm_head 时：

```text
<|im_end|> rank 从约 111000 提升到约 1960
但仍不是 top-1
stop rate 仍为 0%
```

2. 加入：

```text
modules_to_save = ["lm_head"]
```

后：

```text
<|im_end|> rank = 1
stop overfit rate = 100%
strict eval stop accuracy 从 0% 跳到 100%
```

### 3.3 阶段结论

Stage 1 证明：在当前 `Qwen3-1.7B-Base + LoRA all-linear` 设置下，LoRA 可以学习回答意图，但冻结 lm_head 会成为 stop token 校准瓶颈。加入 `modules_to_save=["lm_head"]` 后，模型才能稳定学习 `<|im_end|>` 输出。

Stage 1 后模型具备基础 assistant 行为，但 strict-format 能力仍不足，经常在 JSON-only、answer-only、exactly 等指令下输出多余解释。

---

## 4. Stage 1.5：Strict-format / Stop 补课

### 4.1 阶段目标

Wonderland 任务最终答案通常很短，模型必须严格遵守格式。因此 Stage 1.5 专门训练：

* only means only；
* exactly means exactly；
* JSON-only；
* binary-only；
* answer-only；
* 输出答案后立即停止。

### 4.2 数据构成

Stage 1.5 使用 2000 条 train、200 条 validation。数据由 80% 规则合成 strict-format 样本和 20% No Robots replay 组成。

主要类别包括：

```text
exact_output
binary_only
wonderland_like_binary
json_only
yes_no
classification
extraction
line_count
stop_behavior
no_robots_replay
```

其中 `wonderland_like_binary` 使用程序生成简单 8-bit transformation，例如 NOT、ROTL、ROTR、XOR mask、AND/OR mask 等，但不使用官方 Wonderland 数据。

### 4.3 结果

Stage 1.5 训练 1 epoch，约 63 steps，结果为：

```text
train_loss final: 1.146
eval_loss best: 1.115
eval_mean_token_accuracy: 0.746
stop_accuracy: 99.38%
instruction adherence: 91.88%
```

### 4.4 阶段结论

Stage 1.5 证明：小规模规则合成数据能够有效补齐 strict-format 能力。模型学会了短答、JSON-only、binary-only、answer-only 和稳定停止。

此阶段为后续 Wonderland 和 RL reward parse 打下了关键基础。

---

## 5. Stage 2：Thinking Format Warmup

### 5.1 阶段目标

Stage 2 的目标不是解决 Wonderland 推理，而是让模型学会 thinking protocol：

当用户明确要求 think / reason / think briefly 时，输出：

```text
<think>
简短推理
</think>
最终答案
```

当用户要求 only / exactly / no explanation 时，不能乱输出 `<think>`。

### 5.2 数据构成

Stage 2 使用 2000 条 train、200 条 validation、200 条 protocol_test。数据分布为：

```text
cot_collection_short: 500
programmatic_thinking: 700
stage1_5_strict_replay: 500
no_robots_open_replay: 300
```

thinking 与 no-thinking 比例约为：

```text
thinking: 60%
no-thinking: 40%
```

未使用 OpenThoughts，因为其 DeepSeek-R1 风格长 CoT 中位 reasoning 长度约 4051 tokens，不适合当前 1.7B 模型和 short-thinking warmup。

### 5.3 关键问题与修复

Stage 2 中发现两个关键模板问题。

第一个问题是评测时 `enable_thinking=False` 被写死，导致 Qwen3 generation prompt 自动插入空 think：

```text
<think>

</think>
```

这等于告诉模型“思考已经结束，直接回答”，导致 think_tag_success 初始为 0%。修复方法是在评测 prompt 渲染时显式传入 `enable_thinking=True`。

第二个问题是训练模板无条件给 assistant 包空 think。对于 no-thinking 数据，标签变成：

```text
<think>

</think>

答案
```

模型因此学会“空 think + 答案”。修复方法是：只有当 `reasoning_content` 非空时才插入 `<think>` 块。

### 5.4 结果

Stage 2 训练 1 epoch，63 steps，结果为：

```text
train_loss: 2.255 → 1.138
eval_loss: 1.210 → 0.773
eval_mean_token_accuracy: 58.3% → 80.9%
显存峰值: 18.7GB
```

protocol_test 结果：

```text
think_tag_success: 110/120 = 91.7%
final_answer_parse: 110/120 = 91.7%
no_think_when_not_requested: 80/80 = 100%
stop_success overall: 192/200 = 96.0%
mean_generated_tokens: 66.2
overlong_thinking_rate: 20.0%
```

strict 回归：

```text
exact_match: 41/50 = 82.0%
no_spurious_think: 50/50 = 100%
stop_success: 50/50 = 100%
```

### 5.5 阶段结论

Stage 2 成功完成 thinking protocol warmup。模型已经学会在显式要求 think 时输出 `<think>...</think>\nanswer`，并且在 no-think prompt 中基本不误触发 thinking。

但 Stage 2 不解决真正推理正确性。它只解决“会不会按协议思考”和“思考格式是否稳定”。

---

## 6. Stage 3：Wonderland Cold-start

### 6.1 阶段目标

Stage 3 的目标是让模型开始学习 Wonderland 任务的规则归纳能力。

与 Stage 2 不同，Stage 3 不再只是训练 `<think>` 协议，而是训练模型如何根据 examples 归纳转换规则，并输出最终答案。

为了保留后续 RL 数据，没有直接使用全部 Wonderland train 做 SFT，而是先从 train 内部切分出 Stage 3 SFT pool，并保留大部分 train prompts 给后续 RL 可能使用。

---

## 7. Stage 3 数据生成管线

### 7.1 初始设计

Stage 3 数据由四部分组成：

```text
Wonderland answer-only
Wonderland compressed CoT
Stage 1.5 strict replay
Stage 2 thinking replay
```

基本思路是：

```text
外部 deterministic reasoner / solver
→ 生成原始 reasoning trace
→ 压缩为短 CoT
→ 校验 final answer == gold
→ 通过后进入 SFT
```

如果 reasoner 返回 None，或者 answer 与 gold 不一致，则不生成 CoT，样本降级为 answer-only 或跳过 CoT。

### 7.2 借鉴外部 reasoner

参考了 `tonghuikang/nemotron` 仓库中的 Wonderland/Nemotron 相关 reasoners。bit manipulation reasoner 最早即基于其思路实现，后续也参考了 gravity、unit、cipher、numeral、symbolic equation 等任务的求解方式。

但没有直接使用其长 trace，而是重写为适合 Qwen3-1.7B 的 compressed CoT。原因是：

* 当前模型只有 1.7B，不适合长调试日志；
* 长 CoT 容易导致 `<think>` 不闭合；
* final answer 需要非常干净；
* 后续 reward parse 依赖稳定格式。

### 7.3 数据审计

所有 Stage 3 数据都经过以下检查：

```text
source split leakage = 0
validation/test 未读取
final_answer 可解析
final_answer == gold
<think> 与 </think> 闭合
</think> 后只有最终答案
无 \boxed{} 残留
无 <|im_end|> 手写残留
Qwen3 tokenizer token length <= 1024
assistant-only labels 正常
<|im_end|> 在监督区
```

---

## 8. Stage 3 Smoke 与 Mini 实验

### 8.1 Smoke-300

Smoke 使用 300 条样本，20 steps，目标是验证训练链路，而不是追求准确率。

数据分布：

```text
Wonderland answer-only: 120
Wonderland compressed-CoT: 120
Stage 1.5 strict replay: 30
Stage 2 thinking replay: 30
```

结果：

```text
train_loss: 1.92 → 0.93
eval_loss: 1.17
adapter reload: PASS
Stage 2 protocol 基本保持
strict 能力没有明显崩坏
```

Wonderland dev exact 约 14.17%，与 Stage 2 baseline 接近。结论是：训练链路通过，但 20 steps 不足以学习任务。

### 8.2 Mini-1000

Mini 使用 1000 条数据，但由于仍使用 smoke 模式 `max_steps=20`，实际只覆盖约 64% 数据，不能作为完整训练结论。

结果显示：

```text
Wonderland exact: 14.17%
Parse rate: 70.83%
Protocol think 回到 83.33%
```

结论：mini_1000 本质仍是 smoke，不足以判断 Stage 3 数据有效性。

---

## 9. Stage 3 formal_v0_1：格式冷启动成功，推理冷启动不足

### 9.1 数据与训练

formal_v0_1 使用全量 Stage 3 数据，约 2495 条 train，1 epoch，从 Stage 2 adapter 重新开始训练。

训练结果：

```text
global_step: 78
train_loss avg: 0.813
final eval_loss: 0.510
CoT parse rate: 98.21%
Stop success: 100%
Mean generated tokens: 31.8
```

### 9.2 评估结果

同一份 116 条 wonderland-only dev 上：

```text
Stage 2 baseline: 5.17%
formal_v0_1: 10.34%
```

任务分布：

```text
numeral: 35.71%
unit_conversion: 7.14%
bit_manipulation: 6.25%
gravity: 0%
cipher: 0%
symbolic_equation: 0%
```

### 9.3 问题定位

formal_v0_1 证明：模型学会了格式控制，但没有真正学会大多数任务的解题过程。

错误分析发现，gravity/unit 中模型可以写出正确公式结构，但数值系数错误。根因是 v0_1 trace 只是声明：

```text
Use g = 2*d/t^2
Apply the coefficient
```

但没有展示具体中间计算。模型学会了“该说什么”，没有学会“怎么算”。

因此 v0_1 属于成功的格式冷启动，但不是成功的推理冷启动。

---

## 10. Stage 3 v0_2：加入具体中间计算

### 10.1 数据改造

v0_2 的核心修改是把 CoT 从 declarative 改为 demonstrative。

例如 gravity 从：

```text
Use g = 2*d/t^2. The examples give a consistent g around 18.61.
Then compute d = 0.5*g*t^2.
```

改成：

```text
Use g = 2*d/t^2.
Example: for t=1.59, d=20.54,
g = 2*20.54/1.59^2 ≈ 16.2478.
For query t=4.47,
d = 0.5 * 16.2478 * 4.47^2 = 162.32.
```

unit conversion 加入具体 ratio 与 query multiplication；bit 加入 explicit rule vector；cipher 加入部分 mapping；numeral 加入逐步 Roman numeral 分解。

### 10.2 formal_v0_2 结果

formal_v0_2 使用 v0_2 数据，1 epoch，从 Stage 2 adapter 重新开始训练。

核心结果：

```text
Overall exact: 20.69%
Parse rate: 83.93%
Stop success: 93.10%
Mean tokens: 69.8
```

分任务：

```text
numeral: 82.14%
bit_manipulation: 6.25%
gravity: 0%
unit_conversion: 0%
cipher: 0%
symbolic_equation: 0%
```

### 10.3 结论

v0_2 最重要的发现是：**enriched CoT 有效**。Numeral 从 v0_1 的 35.7% 提升到 82.1%，说明“展示具体步骤”的 trace 对模型学习有明显帮助。

但 gravity/unit 仍为 0%。进一步分析发现，v0_2 中展示的是“从单个 example 计算 g/ratio”，而 gold answer 实际使用的是所有 examples 的 median g / median coefficient。也就是说，trace 演示路径与 gold 生成路径不一致。

cipher 也出现新问题：mapping 条目过多，模型在生成时重复 mapping，导致 `<think>` 不闭合。parse rate 从 98.21% 下降到 83.93%。

---

## 11. Stage 3 v0_3：修复 median 路径与 cipher 循环

### 11.1 数据改造

v0_3 针对 v0_2 的问题做了三处主要修改：

第一，gravity 从单 example 改为：

```text
all g values → median g → query substitution
```

示例：

```text
g values: 16.19, 16.31, 16.33.
median g = 16.31.
query t=3.99:
d = 0.5 * 16.31 * 3.99^2 = 130.09.
```

第二，unit conversion 从单 ratio 改为：

```text
all coefficients → median coefficient → query multiplication
```

第三，cipher 限制 mapping 条目数量：

```text
mapping entries <= 10
explicit Decrypt query "..." -> "..."
```

避免模型重复输出长 mapping，导致 `<think>` 不闭合。

同时保留 numeral 的 v0_2 格式，bit 保留 explicit rule vector，symbolic 继续采用保守策略：没有具体 rule_name 的样本降级 answer-only。

### 11.2 formal_v0_3 结果

formal_v0_3 训练完成后，四版对比如下：

```text
Stage 2 overall: 5.17%
v0_1 overall: 9.48%
v0_2 overall: 20.69%
v0_3 overall: 17.24%
```

关键指标：

```text
parse rate: 89.29%
stop success: 94.83%
unclosed think: 6
numeral: 67.9%
bit: 6.25%
gravity: 0%
unit: 0%
cipher: 0%
```

### 11.3 v0_3 结论

v0_3 修复了部分生成稳定性问题：

```text
unclosed think: 12 → 6
parse rate: 83.93% → 89.29%
stop success: 93.10% → 94.83%
```

但 overall accuracy 从 v0_2 的 20.69% 回落到 17.24%，主要原因是 numeral 从 82.1% 回落到 67.9%。gravity/unit/cipher 仍然为 0。

最终分析认为：

* v0_2 / v0_3 已经证明 enriched CoT 对简单、模式明确的任务有效；
* numeral 是最清晰的成功案例；
* bit manipulation 有轻微非零提升；
* gravity/unit 的失败不再主要是 trace 格式问题，而是 1.7B 模型在自由生成中无法可靠从 prompt 提取数值、排序取 median、完成乘法/平方/四舍五入；
* cipher 任务即使缩短 mapping，仍然没有学会稳定字符替换；
* symbolic equation 数据可学习信号较稀疏，短期内没有明显收益。

---

## 12. 最终结论

本项目完整走通了从 Base 模型到 Stage 3 Wonderland cold-start 的后训练流程，并得到以下核心结论。

### 12.1 关于 instruction tuning

Base 模型不会天然变成 assistant。Stage 1 证明，No Robots 风格数据可以教会基础指令跟随，但不能保证 strict-format 和停止行为。

### 12.2 关于 stop token

只训练 LoRA all-linear 不训练 lm_head 时，模型能学到回答意图，但 `<|im_end|>` 难以成为 top-1。加入 `modules_to_save=["lm_head"]` 后，stop token 校准显著改善。

### 12.3 关于 strict-format

Stage 1.5 证明，小规模规则合成数据可以有效补齐 answer-only、binary-only、JSON-only、exactly、stop 等能力。这对后续自动判分和 RL 冷启动非常关键。

### 12.4 关于 thinking protocol

Stage 2 证明，模型不会天然理解 CoT 协议。必须显式训练 `<think>...</think>\nanswer`，同时保留 no-think replay，防止模型在普通 prompt 中乱输出 `<think>`。

### 12.5 关于 Wonderland cold-start

Stage 3 证明，solver-distilled compressed CoT 可以显著提升模型的格式稳定性和部分任务能力。尤其是 numeral 类任务，在展示 step-by-step greedy decomposition 后，从约 21% 提升到最高 82%。

### 12.6 关于小模型推理边界

Gravity/unit/cipher/symbolic 等任务暴露了 1.7B 模型的明显边界。即使 CoT 中展示了公式、median、query substitution，模型仍难以在自由生成时稳定完成数值抽取、排序、中位数、乘法、平方、字符映射等操作。

这说明：对 1.7B 模型来说，SFT CoT 可以教格式和部分模式匹配，但不能可靠教会复杂算法执行。若要继续提升，需要更强的底座、更高质量更密集的过程监督、更强的 deterministic verifier，或者进入更复杂的 RL / rejection sampling / tool-augmented 推理路径。

---

## 13. 为什么暂不继续 RL

虽然 Stage 3 已经把 final answer parse rate 和 stop success 提升到了较可用水平，但 hard 类任务准确率仍然接近 0。此时直接进入 RL 会面临几个问题：

1. reward 过于稀疏，大多数采样答案错误；
2. 模型还没有形成稳定的可奖励行为；
3. 1.7B 模型在数值计算和规则归纳上存在固有限制；
4. RL 可能主要优化格式，而不是带来真实推理突破；
5. 当前项目目标是学习完整后训练流程，而不是冲击竞赛高分。

因此，本项目在 Stage 3 SFT 后停止，不继续 RL。当前结果已经足以展示完整后训练链路中的关键工程问题：模板适配、label mask、stop token、strict-format、thinking protocol、solver-distilled CoT、数据审计、错误归因与迭代式数据改造。

---

## 14. 最终成果

最终得到的成果包括：

```text
Stage 1 No Robots adapter
Stage 1.5 strict-format adapter
Stage 2 thinking warmup adapter
Stage 3 formal_v0_1 / v0_2 / v0_3 adapters
多版本 Stage 3 数据集
多任务 reasoner wrapper
compressed CoT 数据生成管线
label audit 工具
split / leakage audit
strict regression eval
thinking protocol eval
Wonderland per-task eval
错误分析与 trace quality audit
```

最有价值的工程经验是：

```text
不要只看 loss；
不要只看 overall accuracy；
必须同时看 format、parse、stop、strict regression、thinking regression、per-task accuracy；
必须持续审计数据路径是否和 gold 生成路径一致；
对小模型而言，CoT 不是越长越好，而是必须短、具体、闭合、可执行。
```

本项目最终虽然没有把 Wonderland hard tasks 训练到高准确率，但完整复现了一个真实后训练项目中从问题发现、数据构造、训练、评估、错误分析到迭代修复的工程闭环。
