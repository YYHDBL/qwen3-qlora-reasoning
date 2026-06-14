# qwen3-qlora-reasoning

基于 `Qwen/Qwen3-4B-Base` 的 BF16 LoRA 指令微调项目，分 Stage 1（指令跟随）和后续推理阶段。

## 项目结构

```
├── configs/                    # YAML 配置文件
│   └── stage1_no_robots.yaml  # Stage 1 主配置
├── src/
│   ├── common/                 # 通用工具
│   │   ├── config.py           # YAML 加载、覆盖、校验
│   │   ├── experiment.py       # 文件 I/O、SHA-256、环境信息
│   │   └── prompt_format.py    # prompt/answer 格式化
│   ├── data_processing/        # 数据处理
│   │   ├── classifier.py       # 按 prompt 前缀对推理任务分类
│   │   ├── splitters.py        # 确定性分层 80/10/10 划分
│   │   ├── prepare_dataset.py  # CSV → 校验 → 分类 → 分片 → 落盘
│   │   └── instruction_data.py # No Robots 对话集下载与校验
│   ├── training/               # Stage 1 训练
│   │   ├── chat_template.py    # Qwen3 对话模板（含训练变体）
│   │   ├── model_loader.py     # tokenizer / BF16 模型 / LoRA 加载
│   │   ├── lora.py             # LoRA 配置构造
│   │   ├── label_audit.py      # 训练前 assistant-only 标签审计
│   │   ├── train_sft.py        # SFT 训练入口（overfit/smoke/formal）
│   │   └── verify_adapter.py   # LoRA adapter 重载验证
│   └── evaluation/             # 评估
│       ├── evaluate.py         # 统一评估入口（BF16/NF4/LoRA）
│       ├── instruction_eval.py # 指令跟随评估
│       ├── analyze_tokens.py   # tokenizer 长度分析
│       ├── answer_evaluation.py# 任务感知的答案解析与比较
│       └── metrics.py          # 指标聚合
├── requirements*.txt           # 依赖（基础 / Stage1 GPU / 完整 GPU）
└── data/                       # 数据制品目录
```

## 环境安装

```bash
# 基础依赖（YAML + safetensors）
pip install -r requirements.txt

# Stage 1 训练（GPU 必需）
pip install -r requirements-stage1-gpu.txt

# 完整评估（含 bitsandbytes 4-bit 量化）
pip install -r requirements-gpu.txt
```

## Stage 1：指令跟随微调

### 数据准备

```bash
# 从 HuggingFace Hub 下载 No Robots 数据集，生成 train/validation JSONL
python -m src.data_processing.instruction_data \
  --config configs/stage1_no_robots.yaml
```

### 标签审计

```bash
# 在校验 chat template 生成的 assistant-only 标签，输出 token_report.json
python -m src.training.label_audit \
  --config configs/stage1_no_robots.yaml
```

### 训练

三种模式：

| 模式 | 用途 | 数据量 | 步数 |
|------|------|--------|------|
| `overfit` | 快速验证 loss 能否下降 | 16 条 | 40 |
| `smoke` | 小规模功能验证 | 256 条 | 20 |
| `formal` | 正式完整训练 | 全部 | 1 epoch |

```bash
# 快速过拟合测试
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml --mode overfit

# 冒烟测试
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml --mode smoke

# 正式训练
python -m src.training.train_sft \
  --config configs/stage1_no_robots.yaml --mode formal
```

### Adapter 验证

```bash
python -m src.training.verify_adapter \
  --config configs/stage1_no_robots.yaml \
  --adapter-path outputs/stage1_no_robots/overfit/adapter \
  --output outputs/stage1_no_robots/overfit/adapter_reload.json
```

### 指令跟随评估

```bash
# 评估 base 模型
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

## SwanLab 实验追踪

训练过程自动上报到 [SwanLab](https://swanlab.cn)，包括 loss 曲线、GPU 利用率、token 统计。

在 `configs/stage1_no_robots.yaml` 的 `experiment` 段配置：

```yaml
experiment:
  name: stage1_no_robots
  swanlab_project: "stage1-no-robots"   # 项目名（必填）
  swanlab_workspace: null               # 团队空间，null=个人空间
  swanlab_api_key: null                 # API key，null=从环境变量 SWANLAB_API_KEY 读取
```

## 评估指标说明

### 指令跟随评估（instruction_eval.py）

| 指标 | 含义 |
|------|------|
| `instruction_accuracy` | 完全遵循指令的比例 |
| `format_accuracy` | 输出格式正确的比例 |
| `stop_accuracy` | 在 256 token 内正确停止的比例 |
| `continuation_failure_rate` | 达到 max_tokens 被截断的比例 |

### 推理答案评估（evaluate.py）

| 指标 | 含义 |
|------|------|
| `primary_accuracy` | 主要正确率（各类别定义不同） |
| `strict_accuracy` | 严格匹配 gold answer |
| `normalized_accuracy` | 标准化后匹配 |
| `parse_success_rate` | 能解析出有效答案的比例 |

## 配置说明

`configs/stage1_no_robots.yaml` 包含以下段：

- **experiment**：实验名、输出目录、SwanLab 配置
- **model**：模型 ID、dtype、attention 实现、cache 目录
- **data**：数据集 ID、split 名称、输出目录
- **lora**：rank、alpha、dropout、target_modules、bias、task_type
- **training**：max_length、batch_size、梯度累积、学习率、warmup、epoch/step、overfit/smoke/formal 限制
- **generation**：max_new_tokens、do_sample
- **evaluation**：dev/test JSONL 路径

## GPU 要求

- BF16 训练需要支持 BF16 的 NVIDIA GPU（RTX 3090/4090, A100 等）
- 4-bit 评估需 `bitsandbytes`

## 本地测试

不含模型权重、不需要 GPU 的模块级测试：

```bash
python -m pytest src/ -x -q
```
