# PaddleOCR 图纸识别 MLOps 平台

基于 Amazon SageMaker 的 PaddleOCR 图纸识别 MLOps 解决方案，支持模型版本管理、增量训练、数据标注和图形相似性检索。

## 项目结构

```
paddleocr-mlops/
│
├── README.md                              # 项目说明文档（本文件）
│
├── docs/                                  # 文档目录
│   ├── 方案文档-PaddleOCR图纸识别MLOps平台.md    # 完整技术方案
│   ├── PaddleOCR-SageMaker-部署指南.md          # 部署操作指南
│   └── PPT大纲-PaddleOCR-MLOps方案.md           # 演示 PPT 大纲
│
├── demos/                                 # 演示代码目录
│   ├── 01_ground_truth_labeling.py        # Ground Truth 数据标注
│   ├── 02_model_registry.py               # Model Registry 版本管理
│   ├── 03_incremental_training_pipeline.py # 增量训练 Pipeline
│   ├── 04_image_similarity_search.py      # 图形相似性检索
│   ├── 05_end_to_end_inference.py         # 端到端推理演示
│   └── 06_cost_estimation.py              # 成本估算工具
│
└── data/                                  # 数据处理目录
    ├── 07_dataset_download_preprocess.py  # Dataset-P&ID 下载预处理
    └── 07b_alternative_datasets.py        # 备用数据集下载
```

## 文件说明

### 文档 (docs/)

| 文件 | 说明 |
|-----|------|
| `方案文档-PaddleOCR图纸识别MLOps平台.md` | 完整的技术方案文档，包含架构设计、流程说明、成本估算、实施路线图 |
| `PaddleOCR-SageMaker-部署指南.md` | 详细的部署操作指南，包含 Docker 构建、训练、推理、增量训练步骤 |
| `PPT大纲-PaddleOCR-MLOps方案.md` | 客户演示 PPT 大纲，用于准备演示材料 |

### 演示代码 (demos/)

| 文件 | 功能 | 演示重点 |
|-----|------|---------|
| `01_ground_truth_labeling.py` | SageMaker Ground Truth 数据标注 | 标注任务创建、自定义模板、导出 PaddleOCR 格式 |
| `02_model_registry.py` | Model Registry 模型版本管理 | 版本注册、对比、审批流程、部署 |
| `03_incremental_training_pipeline.py` | SageMaker Pipeline 增量训练 | 自动化流程、条件判断、防止灾难性遗忘 |
| `04_image_similarity_search.py` | 图形相似性检索 | ResNet 特征提取、OpenSearch kNN、向量检索 |
| `05_end_to_end_inference.py` | 端到端推理演示 | OCR 识别、结构化提取、版本对比 |
| `06_cost_estimation.py` | 成本估算工具 | 不同规模场景成本计算、优化建议 |

### 数据处理 (data/)

| 文件 | 功能 | 使用场景 |
|-----|------|---------|
| `07_dataset_download_preprocess.py` | Dataset-P&ID 数据集下载与预处理 | 下载 500 张 P&ID 工程图纸数据集 |
| `07b_alternative_datasets.py` | 备用数据集下载 | FUNSD、SROIE、合成数据等替代方案 |

## 快速开始

### 1. 准备演示数据

```bash
# 方式一：下载 Dataset-P&ID（推荐）
python data/07_dataset_download_preprocess.py

# 方式二：生成合成数据（最快）
python data/07b_alternative_datasets.py --synthetic 100

# 方式三：下载所有备用数据集
python data/07b_alternative_datasets.py --all
```

### 2. 运行演示

```bash
# 成本估算（无需 AWS 环境）
python demos/06_cost_estimation.py

# 其他演示需要配置 AWS 环境
# 请先修改代码中的 BUCKET_NAME、ACCOUNT_ID 等配置
```

### 3. 演示顺序建议

1. **成本估算** (`06_cost_estimation.py`) - 让客户了解投入
2. **数据标注** (`01_ground_truth_labeling.py`) - 展示数据准备流程
3. **模型版本管理** (`02_model_registry.py`) - 核心功能演示
4. **增量训练** (`03_incremental_training_pipeline.py`) - 重点演示
5. **端到端推理** (`05_end_to_end_inference.py`) - 效果展示
6. **图形检索** (`04_image_similarity_search.py`) - 扩展功能

## 客户需求覆盖

| 需求 | 覆盖文件 | 状态 |
|-----|---------|------|
| 模型版本管理与增量训练 | `02`, `03` | ✅ |
| 训练数据清洗、标注流程 | `01`, `07`, `07b` | ✅ |
| 通用 OCR 演示（PaddleOCR） | `05` | ✅ |
| 图形相似性识别 | `04` | ✅ |

## 技术栈

- **OCR 框架**: PaddleOCR (PP-OCRv4)
- **ML 平台**: Amazon SageMaker
- **数据标注**: SageMaker Ground Truth
- **流程编排**: SageMaker Pipelines
- **模型管理**: SageMaker Model Registry
- **向量检索**: Amazon OpenSearch (kNN)
- **存储**: Amazon S3
- **容器**: Amazon ECR

## 预估成本

| 场景 | 月成本 (USD) |
|-----|-------------|
| POC | ~$174 |
| 开发环境 | ~$350 |
| 生产环境 | ~$850 |

详细成本分析请运行 `python demos/06_cost_estimation.py`

## 注意事项

1. 演示代码中的 AWS 资源配置（bucket、account_id 等）需要替换为实际值
2. 部分功能需要 GPU 实例（如 ml.p3.2xlarge）
3. 建议先在小规模数据上验证流程
4. 生产部署前请进行安全审查

## 联系方式

如有问题，请联系项目团队。
