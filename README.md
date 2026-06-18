# qwen3-qlora-reasoning

基于 `Qwen/Qwen3-4B-Base` 的 BF16 LoRA 指令微调项目，支持两阶段训练：

- **Stage 1**：指令跟随微调（当前仓库实现）
- **Stage 2**：推理能力训练（开发中）

## 数据流概览

```
data/raw/train.csv              ← 原始 CSV（id, prompt, answer 三列）
        │
        ▼
prepare_dataset.py              ← 校验、分类（unit_conversion/cipher/...）、80/10/10 分片
        │
data/processed/train.jsonl      ← 处理后的 JSONL（id, task_type, prompt, answer）
data/processed/validation.jsonl
data/processed/test.jsonl
        │
        ▼
analyze_tokens.py               ← 分析 token 长度分布，确定 max_length
        │
        ▼
label_audit.py                  ← 审计助理标签是否正确（-100 mask），输出排除列表
        │
        ▼
train_sft.py                    ← SFT 训练（overfit → smoke → formal 三级门控）
        │
        ▼
outputs/stage1_no_robots/       ← 训练产物（adapter 权重、metrics、环境快照）
```

## 项目结构

```
├── configs/
│   ├── stage1_no_robots.yaml            # Stage 1 主配置（No Robots 数据集）
│   └── stage1_no_robots_qwen3_1_7b_local.yaml  # Qwen3-1.7B 本地模型变体
├── scripts/
│   ├── probe_im_end_rank.py             # 探索 lm_head 是否需要解冻
│   ├── generate_stop_overfit_data.py    # 生成短回答停止行为测试数据
│   └── stop_overfit_train.py            # 独立停止行为 overfit 训练脚本
├── data/
│   ├── raw/                    # 原始数据（CSV）
│   │   └── train.csv
│   ├── processed/              # 处理后的 JSONL
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── eval/                   # 指令跟随评估样本
│   │   ├── instruction_dev.jsonl
│   │   └── instruction_test.jsonl
│   └── instruction/            # 数据审计产物
│       └── stage1/
│           ├── train.jsonl
│           ├── validation.jsonl
│           ├── token_report.json
│           └── batch_audit.json
├── src/
│   ├── common/                 # 通用工具
│   │   ├── config.py           # YAML 加载、覆盖、校验
│   │   ├── experiment.py       # 文件 I/O、SHA-256、环境快照
│   │   └── prompt_format.py    # prompt/answer 格式化
│   ├── data_processing/        # 数据处理
│   │   ├── classifier.py       # 按 prompt 前缀对推理任务分类
│   │   ├── splitters.py        # 确定性分层 80/10/10 划分
│   │   ├── prepare_dataset.py  # CSV → 校验 → 分类 → 分片 → JSONL
│   │   └── instruction_data.py # No Robots 对话集下载与校验
│   ├── training/               # Stage 1 训练
│   │   ├── chat_template.py    # Qwen3 对话模板（推理/训练变体）
│   │   ├── model_loader.py     # tokenizer / BF16 模型 / LoRA 加载
│   │   ├── lora.py             # LoRA 配置构造
│   │   ├── label_audit.py      # 训练前助理标签审计
│   │   ├── train_sft.py        # SFT 训练入口（overfit/smoke/formal）
│   │   ├── verify_adapter.py   # 重载 adapter + 单条生成验证
│   │   └── _core_pipeline.py   # 核心流程 demo（一条数据走到底）
│   ├── evaluation/             # 评估
│   │   ├── evaluate.py         # 统一评估入口
│   │   ├── instruction_eval.py # 指令跟随评估
│   │   ├── analyze_tokens.py   # tokenizer 长度分布分析
│   │   ├── answer_evaluation.py# 任务感知答案解析与比较
│   │   └── metrics.py          # 指标聚合
│   └── models/
│       └── chat_demo.py        # 聊天交互 demo
├── tests/                      # 单元测试（离线、无 GPU）
│   ├── test_evaluate.py
│   ├── test_instruction_eval.py
│   └── test_transformers_compat_static.py
└── requirements*.txt           # 依赖文件
```

## 环境安装

```bash
# 基础依赖
pip install -r requirements.txt

# Stage 1 训练（GPU 必需）
pip install -r requirements-stage1-gpu.txt

# 完整评估（含 bitsandbytes 4-bit 量化）
pip install -r requirements-gpu.txt
```

## 两种数据来源

### 方式 A：HuggingFace No Robots（默认）

```bash
python -m src.data_processing.instruction_data \
  --config configs/stage1_no_robots.yaml
```

生成 `data/instruction/stage1/` 下的 JSONL，天然含 `messages` 字段。

### 方式 B：自己的 CSV（`data/raw/train.csv`）

```bash
python -m src.data_processing.prepare_dataset
```

要求 CSV 列名为 `id, prompt, answer`，输出 `data/processed/{train,validation,test}.jsonl`。

然后配置 `stage1_no_robots.yaml` 的 `data.output_dir` 指向 `data/processed`。

> **注意**：自定义数据格式为 `prompt`/`answer`，进入 chat template 前需先转为 `messages` 格式。详见 `_core_pipeline.py`。

## 训练前分析

```bash
# 分析 token 长度分布，辅助确定 max_length
python -m src.evaluation.analyze_tokens \
  --config configs/stage1_no_robots.yaml
```

输出各分片 token 长度统计（min/max/mean/p95）和 token 数阈值建议。

## 标签审计

```bash
# 验证 assistant-only mask 的正确性
python -m src.training.label_audit \
  --config configs/stage1_no_robots.yaml
```

- 每条样本的 assistant token 数量
- 边界检查（每个 assistant span 后必须有 `<|im_end|>`）
- 输出 `token_report.json`（排除的 ID 列表供训练过滤）

## 核心流程演示

```bash
python -m src.training._core_pipeline
```

用 `data/processed/train.jsonl` 的第一条数据走完完整流程：

```
原始 JSONL → messages 格式 → 训练模板渲染 → tokenize + mask → labels
```

## 训练

三种运行模式：

| 模式 | 用途 | 数据量 | 步数 | 门控 |
|------|------|--------|------|------|
| `overfit` | 验证 loss 能下降到 80% 以下 | 16 条 | 40 | 自动检查 loss 下降 |
| `smoke` | 小规模冒烟测试 | 256 条 | 20 | 需 overfit 通过 |
| `formal` | 正式完整训练 | 全部 | 3 epoch | 需所有前置门控通过 |

```bash
# overfit
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml --mode overfit

# smoke
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml --mode smoke

# formal
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml --mode formal
```

### lm_head 解冻

配置中 `lora.modules_to_save: ["lm_head"]` 使 `lm_head` 层**全量参与训练**（不通过 LoRA）。

用途：LoRA 对 Q/K/V/O 和 FFN 层的低秩干预不足以让 `lm_head` 学会在正确位置生成 `<|im_end|>`。解冻 `lm_head` 后其权重可直接更新，改善停止 token 的学习。

### 停止行为 overfit 测试

`scripts/` 下提供独立测试脚本，用于快速验证模型能否学会输出 `<|im_end|>`：

```bash
# 第 1 步：生成短回答训练数据
python scripts/generate_stop_overfit_data.py

# 第 2 步：运行停止 overfit 训练
python scripts/stop_overfit_train.py
```

```bash
# 探测 lm_head 是否需要解冻（分析 logit 排名）
python scripts/probe_im_end_rank.py
```

## Adapter 验证

训练后重载 LoRA adapter 并生成一次，验证 stop token 触发：

```bash
python -m src.training.verify_adapter \
  --config configs/stage1_no_robots.yaml \
  --adapter-path outputs/stage1_no_robots/overfit/adapter \
  --output outputs/stage1_no_robots/overfit/adapter_reload.json
```

## 评估

```bash
# 评估 base 模型指令跟随
python -m src.evaluation.instruction_eval \
  --config configs/stage1_no_robots.yaml \
  --split dev \
  --output-dir outputs/stage1_no_robots/base-dev

# 评估 LoRA adapter
python -m src.evaluation.instruction_eval \
  --config configs/stage1_no_robots.yaml \
  --split dev \
  --adapter-path outputs/stage1_no_robots/formal/adapter \
  --output-dir outputs/stage1_no_robots/lora-dev
```

### 统一评估入口（支持 BF16/NF4/LoRA）

```bash
python -m src.evaluation.evaluate \
  --config configs/stage1_no_robots.yaml \
  --split test \
  --output-dir outputs/stage1_no_robots/eval
```

## 评估指标

### 指令跟随

| 指标 | 含义 |
|------|------|
| `instruction_accuracy` | 完全遵循指令的比例 |
| `format_accuracy` | 输出格式正确的比例 |
| `stop_accuracy` | 在 256 token 内正确停止的比例 |
| `continuation_failure_rate` | 达到 max_tokens 被截断的比例 |

### 推理答案评估

| 指标 | 含义 |
|------|------|
| `primary_accuracy` | 主要正确率（各类别定义不同） |
| `strict_accuracy` | 严格匹配 gold answer |
| `normalized_accuracy` | 标准化后匹配 |
| `parse_success_rate` | 能解析出有效答案的比例 |

## SwanLab 实验追踪

训练过程自动上报到 [SwanLab](https://swanlab.cn)。配置位置：

```yaml
experiment:
  name: stage1_no_robots
  swanlab_project: "stage1-no-robots"
  swanlab_workspace: null
  swanlab_api_key: null        # 或从环境变量 SWANLAB_API_KEY 读取
```

## 配置说明

`configs/stage1_no_robots.yaml` 包含段：

- **experiment**：实验名、输出目录、SwanLab 配置
- **model**：模型 ID、dtype、attention 实现、cache 目录
- **data**：数据集来源（Hub 或本地）、output_dir
- **lora**：rank / alpha / dropout / target_modules / modules_to_save
- **training**：max_length / batch / 学习率 / epoch / 门控参数
- **generation**：max_new_tokens、do_sample
- **evaluation**：dev/test JSONL 路径

## GPU 要求

- BF16 训练需要支持 BF16 的 NVIDIA GPU（RTX 3090/4090, A100 等）
- 4-bit 评估需 `bitsandbytes`

## 本地测试

无需 GPU 的模块级测试：

```bash
python -m pytest src/ tests/ -x -q
```
