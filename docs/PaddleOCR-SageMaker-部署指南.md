# PaddleOCR 部署到 Amazon SageMaker 指南

本文档介绍如何将 PaddleOCR 框架部署到 Amazon SageMaker，实现 OCR 文字识别服务。

## 架构概览

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SageMaker      │    │  SageMaker      │    │  SageMaker      │
│  Studio         │───▶│  Training Job   │───▶│  Endpoint       │
│  (开发环境)      │    │  (模型训练)      │    │  (推理服务)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Amazon ECR     │    │  Amazon S3      │    │  CloudWatch     │
│  (容器镜像)      │    │  (数据/模型)     │    │  (日志监控)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 前置条件

- AWS 账户及相应 IAM 权限
- Amazon SageMaker Studio 环境
- 基本的 Docker 和 Python 知识
- 训练数据集（标注好的 OCR 数据）

## 步骤一：创建 SageMaker Project

1. 打开 SageMaker Studio
2. 选择 **Projects** → **Create project**
3. 选择 MLOps 模板：**MLOps template for image building, model building, and model deployment**
4. 填写项目名称和描述，点击创建

项目会自动创建：
- 代码仓库（用于版本控制）
- CI/CD 管道（用于构建容器镜像）
- SageMaker Pipeline（用于 MLOps 工作流）

## 步骤二：构建自定义训练容器

### 2.1 创建 Dockerfile

```dockerfile
# Dockerfile for PaddleOCR Training
FROM python:3.8-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 PaddlePaddle 和 PaddleOCR
RUN pip install --no-cache-dir \
    paddlepaddle \
    paddleocr \
    opencv-python-headless \
    sagemaker-training

# 设置工作目录
WORKDIR /opt/ml/code

# 复制训练脚本
COPY train.py /opt/ml/code/

ENV SAGEMAKER_PROGRAM train.py

### 2.2 创建训练脚本 train.py

```python
import os
import json
import paddle
from paddleocr import PaddleOCR

def train():
    # 读取超参数
    hyperparameters_path = '/opt/ml/input/config/hyperparameters.json'
    with open(hyperparameters_path, 'r') as f:
        hyperparameters = json.load(f)
    
    # 数据路径
    training_data_path = '/opt/ml/input/data/training'
    model_output_path = '/opt/ml/model'
    
    # 配置训练参数
    epochs = int(hyperparameters.get('epochs', 100))
    batch_size = int(hyperparameters.get('batch_size', 32))
    learning_rate = float(hyperparameters.get('learning_rate', 0.001))
    
    # 初始化预训练模型
    # 使用 ch_ppocr_mobile_v2.0 作为基础模型进行微调
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        rec_model_dir=os.path.join(training_data_path, 'pretrained_model')
    )
    
    # 训练逻辑（根据实际需求实现）
    # ...
    
    # 保存模型
    paddle.save(ocr.state_dict(), os.path.join(model_output_path, 'model.pdparams'))
    print("Training completed!")

if __name__ == '__main__':
    train()
```

### 2.3 构建并推送镜像到 ECR

```bash
# 设置变量
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REPO_NAME=paddleocr-training

# 创建 ECR 仓库
aws ecr create-repository --repository-name ${REPO_NAME}

# 登录 ECR
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# 构建镜像
docker build -t ${REPO_NAME}:latest .

# 标记镜像
docker tag ${REPO_NAME}:latest ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest

# 推送镜像
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest
```

## 步骤三：准备训练数据

### 3.1 数据格式

PaddleOCR 训练数据格式：

```
train_data/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── labels/
    └── train.txt
```

`train.txt` 格式：
```
images/img_001.jpg	识别文本内容1
images/img_002.jpg	识别文本内容2
```

### 3.2 上传数据到 S3

```bash
aws s3 cp ./train_data s3://your-bucket/paddleocr/train_data/ --recursive
```

## 步骤四：启动 SageMaker 训练任务

```python
import sagemaker
from sagemaker.estimator import Estimator

# 配置
role = sagemaker.get_execution_role()
session = sagemaker.Session()
account_id = session.boto_session.client('sts').get_caller_identity()['Account']
region = session.boto_region_name

# 训练镜像 URI
training_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/paddleocr-training:latest"

# 创建 Estimator
estimator = Estimator(
    image_uri=training_image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU 实例
    output_path='s3://your-bucket/paddleocr/output/',
    hyperparameters={
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001
    }
)

# 指定训练数据
training_input = sagemaker.inputs.TrainingInput(
    s3_data='s3://your-bucket/paddleocr/train_data/',
    content_type='application/x-image'
)

# 启动训练
estimator.fit({'training': training_input})
```

## 步骤五：构建推理容器

### 5.1 推理 Dockerfile

```dockerfile
FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    paddlepaddle \
    paddleocr \
    flask \
    gunicorn

COPY serve.py /opt/ml/code/
COPY predictor.py /opt/ml/code/

WORKDIR /opt/ml/code

ENTRYPOINT ["python", "serve.py"]
```

### 5.2 推理脚本 predictor.py

```python
import os
import json
import base64
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import io

class OCRPredictor:
    def __init__(self):
        model_dir = '/opt/ml/model'
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            rec_model_dir=model_dir
        )
    
    def predict(self, image_data):
        """
        执行 OCR 推理
        :param image_data: base64 编码的图像或图像字节
        :return: OCR 识别结果
        """
        # 解码图像
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # 执行 OCR
        result = self.ocr.ocr(image_np, cls=True)
        
        # 格式化结果
        output = []
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]
            output.append({
                'box': box,
                'text': text,
                'confidence': float(confidence)
            })
        
        return output
```

### 5.3 服务脚本 serve.py

```python
from flask import Flask, request, jsonify
from predictor import OCRPredictor
import os

app = Flask(__name__)
predictor = None

@app.route('/ping', methods=['GET'])
def ping():
    """健康检查端点"""
    return '', 200

@app.route('/invocations', methods=['POST'])
def invoke():
    """推理端点"""
    global predictor
    if predictor is None:
        predictor = OCRPredictor()
    
    # 获取请求数据
    if request.content_type == 'application/json':
        data = request.get_json()
        image_data = data.get('image')
    else:
        image_data = request.data
    
    # 执行预测
    result = predictor.predict(image_data)
    
    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## 步骤六：部署模型到 SageMaker Endpoint

```python
from sagemaker.model import Model

# 推理镜像 URI
inference_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/paddleocr-inference:latest"

# 创建模型
model = Model(
    image_uri=inference_image_uri,
    model_data=estimator.model_data,  # 训练输出的模型路径
    role=role,
    sagemaker_session=session
)

# 部署端点
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='paddleocr-endpoint'
)
```

## 步骤七：调用推理端点

```python
import boto3
import json
import base64

# 创建 SageMaker Runtime 客户端
runtime = boto3.client('sagemaker-runtime')

# 读取测试图像
with open('test_image.jpg', 'rb') as f:
    image_bytes = f.read()

# 编码为 base64
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# 调用端点
response = runtime.invoke_endpoint(
    EndpointName='paddleocr-endpoint',
    ContentType='application/json',
    Body=json.dumps({'image': image_base64})
)

# 解析结果
result = json.loads(response['Body'].read().decode())
print(json.dumps(result, indent=2, ensure_ascii=False))
```

输出示例：
```json
{
  "predictions": [
    {
      "box": [[24, 36], [304, 36], [304, 76], [24, 76]],
      "text": "识别出的文字内容",
      "confidence": 0.9876
    }
  ]
}
```

## 步骤八：PaddleOCR 增量训练（Incremental Training）

增量训练允许基于已有模型继续训练，无需从头开始，可以节省训练时间并利用之前学到的特征。

### 8.1 增量训练概述

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  基础模型 V1     │    │  新增训练数据    │    │  增量训练后     │
│  (已训练)       │ +  │  (新场景/字体)   │ ─▶ │  模型 V2        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

适用场景：
- 新增特定领域数据（如医疗单据、金融票据）
- 支持新字体或手写体
- 提升特定场景识别准确率
- 持续学习，模型迭代优化

### 8.2 准备基础模型

将已训练好的模型上传到 S3：

```bash
# 打包已有模型
cd /path/to/trained_model
tar -czvf model.tar.gz .

# 上传到 S3
aws s3 cp model.tar.gz s3://your-bucket/paddleocr/base-model/model.tar.gz
```

### 8.3 增量训练脚本 incremental_train.py

```python
import os
import json
import yaml
import paddle
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.data import build_dataloader
from ppocr.utils.save_load import load_model

def incremental_train():
    # 读取超参数
    hyperparameters_path = '/opt/ml/input/config/hyperparameters.json'
    with open(hyperparameters_path, 'r') as f:
        hyperparameters = json.load(f)
    
    # 路径配置
    base_model_path = '/opt/ml/input/data/model'  # 基础模型路径
    training_data_path = '/opt/ml/input/data/training'  # 新增训练数据
    model_output_path = '/opt/ml/model'
    
    # 训练参数
    epochs = int(hyperparameters.get('epochs', 50))
    batch_size = int(hyperparameters.get('batch_size', 32))
    learning_rate = float(hyperparameters.get('learning_rate', 0.0001))  # 增量训练用较小学习率
    
    # 加载配置文件
    config_path = os.path.join(base_model_path, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config['Train']['dataset']['data_dir'] = training_data_path
    config['Train']['loader']['batch_size_per_card'] = batch_size
    config['Optimizer']['lr']['learning_rate'] = learning_rate
    config['Global']['epoch_num'] = epochs
    
    # 构建模型
    model = build_model(config['Architecture'])
    
    # 加载基础模型权重（关键步骤）
    base_model_weights = os.path.join(base_model_path, 'best_accuracy.pdparams')
    if os.path.exists(base_model_weights):
        load_model(config, model, optimizer=None, model_type='rec')
        print(f"Loaded base model from: {base_model_weights}")
    else:
        print("Warning: Base model not found, training from scratch")
    
    # 构建优化器
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=epochs,
        step_each_epoch=1000,
        model=model
    )
    
    # 构建数据加载器
    train_dataloader = build_dataloader(config, 'Train', device='gpu', logger=None)
    
    # 构建损失函数
    loss_class = build_loss(config['Loss'])
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch[0]
            labels = batch[1]
            
            # 前向传播
            preds = model(images)
            loss = loss_class(preds, labels)
            
            # 反向传播
            loss['loss'].backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss['loss'].item():.4f}")
        
        # 每个 epoch 保存检查点
        checkpoint_path = os.path.join(model_output_path, f'epoch_{epoch+1}.pdparams')
        paddle.save(model.state_dict(), checkpoint_path)
    
    # 保存最终模型
    final_model_path = os.path.join(model_output_path, 'best_accuracy.pdparams')
    paddle.save(model.state_dict(), final_model_path)
    
    # 保存配置文件
    config_output_path = os.path.join(model_output_path, 'config.yml')
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f)
    
    print("Incremental training completed!")

if __name__ == '__main__':
    incremental_train()
```

### 8.4 启动增量训练任务

```python
import sagemaker
from sagemaker.estimator import Estimator

role = sagemaker.get_execution_role()
session = sagemaker.Session()
account_id = session.boto_session.client('sts').get_caller_identity()['Account']
region = session.boto_region_name

# 训练镜像
training_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/paddleocr-training:latest"

# 创建增量训练 Estimator
incremental_estimator = Estimator(
    image_uri=training_image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    output_path='s3://your-bucket/paddleocr/incremental-output/',
    hyperparameters={
        'epochs': 50,              # 增量训练通常需要较少 epoch
        'batch_size': 32,
        'learning_rate': 0.0001    # 使用较小学习率避免遗忘
    }
)

# 配置输入数据通道
# model 通道：已有的基础模型
# training 通道：新增的训练数据
inputs = {
    'model': sagemaker.inputs.TrainingInput(
        s3_data='s3://your-bucket/paddleocr/base-model/',
        content_type='application/x-tar'
    ),
    'training': sagemaker.inputs.TrainingInput(
        s3_data='s3://your-bucket/paddleocr/new-train-data/',
        content_type='application/x-image'
    )
}

# 启动增量训练
incremental_estimator.fit(inputs)
```

### 8.5 使用 SageMaker model_uri 参数（简化方式）

SageMaker 原生支持通过 `model_uri` 参数进行增量训练：

```python
from sagemaker.estimator import Estimator

# 基础模型 S3 路径
base_model_s3_uri = 's3://your-bucket/paddleocr/base-model/model.tar.gz'

# 创建 Estimator，指定 model_uri
estimator = Estimator(
    image_uri=training_image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    output_path='s3://your-bucket/paddleocr/incremental-output/',
    model_uri=base_model_s3_uri,  # 关键参数：指向基础模型
    hyperparameters={
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.0001
    }
)

# 训练脚本中，基础模型会被解压到 /opt/ml/input/data/model/
training_input = sagemaker.inputs.TrainingInput(
    s3_data='s3://your-bucket/paddleocr/new-train-data/',
    content_type='application/x-image'
)

estimator.fit({'training': training_input})
```

### 8.6 增量训练最佳实践

| 参数 | 初始训练 | 增量训练 | 说明 |
|-----|---------|---------|------|
| learning_rate | 0.001 | 0.0001 | 增量训练用较小学习率 |
| epochs | 100+ | 20-50 | 增量训练需要较少轮次 |
| batch_size | 32 | 32 | 保持一致 |
| warmup | 是 | 可选 | 增量训练可跳过 warmup |

防止灾难性遗忘的策略：
- 混合新旧数据训练（推荐比例 7:3）
- 使用较小学习率
- 冻结部分网络层
- 使用正则化技术

### 8.7 冻结部分网络层进行增量训练

```python
import paddle

def freeze_backbone(model):
    """冻结骨干网络，只训练识别头"""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.trainable = False
            print(f"Frozen: {name}")
        else:
            param.trainable = True
            print(f"Trainable: {name}")

# 在训练前调用
model = build_model(config['Architecture'])
load_model(config, model, optimizer=None, model_type='rec')
freeze_backbone(model)

# 只优化可训练参数
trainable_params = [p for p in model.parameters() if p.trainable]
optimizer = paddle.optimizer.Adam(
    learning_rate=0.0001,
    parameters=trainable_params
)
```

### 8.8 多轮增量训练 Pipeline

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString

# 定义参数
base_model_uri = ParameterString(
    name="BaseModelUri",
    default_value="s3://your-bucket/paddleocr/base-model/model.tar.gz"
)

new_data_uri = ParameterString(
    name="NewDataUri",
    default_value="s3://your-bucket/paddleocr/new-train-data/"
)

# 增量训练步骤
incremental_training_step = TrainingStep(
    name="PaddleOCR-IncrementalTraining",
    estimator=Estimator(
        image_uri=training_image_uri,
        role=role,
        instance_count=1,
        instance_type='ml.p3.2xlarge',
        output_path='s3://your-bucket/paddleocr/incremental-output/',
        model_uri=base_model_uri,
        hyperparameters={
            'epochs': 50,
            'learning_rate': 0.0001
        }
    ),
    inputs={
        "training": sagemaker.inputs.TrainingInput(s3_data=new_data_uri)
    }
)

# 创建 Pipeline
incremental_pipeline = Pipeline(
    name="PaddleOCR-Incremental-Pipeline",
    parameters=[base_model_uri, new_data_uri],
    steps=[incremental_training_step],
    sagemaker_session=session
)

# 提交 Pipeline
incremental_pipeline.upsert(role_arn=role)

# 执行增量训练（可多次执行，每次使用上一轮的输出作为输入）
execution = incremental_pipeline.start(
    parameters={
        "BaseModelUri": "s3://your-bucket/paddleocr/model-v1/model.tar.gz",
        "NewDataUri": "s3://your-bucket/paddleocr/new-data-batch1/"
    }
)
```

## 步骤九：配置 SageMaker Pipeline（可选）

用于自动化 MLOps 工作流：

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, CreateModelStep
from sagemaker.workflow.model_step import ModelStep

# 定义训练步骤
training_step = TrainingStep(
    name="PaddleOCR-Training",
    estimator=estimator,
    inputs={"training": training_input}
)

# 定义模型创建步骤
model_step = CreateModelStep(
    name="PaddleOCR-CreateModel",
    model=model,
    inputs=sagemaker.inputs.CreateModelInput(
        instance_type="ml.m5.xlarge"
    )
)

# 创建 Pipeline
pipeline = Pipeline(
    name="PaddleOCR-Pipeline",
    steps=[training_step, model_step],
    sagemaker_session=session
)

# 提交 Pipeline
pipeline.upsert(role_arn=role)

# 执行 Pipeline
execution = pipeline.start()
```

## 成本优化建议

| 阶段 | 推荐实例 | 说明 |
|-----|---------|------|
| 开发调试 | ml.t3.medium | 低成本，适合代码调试 |
| 模型训练 | ml.p3.2xlarge | GPU 加速，训练效率高 |
| 推理服务 | ml.m5.xlarge | 平衡性能和成本 |
| 批量推理 | ml.m5.4xlarge | 高吞吐量场景 |

## 常见问题

### Q1: 如何支持多语言 OCR？
修改 PaddleOCR 初始化参数：
```python
ocr = PaddleOCR(lang='en')  # 英文
ocr = PaddleOCR(lang='ch')  # 中文
ocr = PaddleOCR(lang='japan')  # 日文
```

### Q2: 如何提高识别准确率？
- 使用更多标注数据进行微调
- 调整图像预处理参数
- 使用更大的预训练模型（如 PP-OCRv3）
- 使用增量训练针对特定场景优化（参考步骤八）

### Q4: 增量训练和从头训练如何选择？
| 场景 | 推荐方式 |
|-----|---------|
| 全新领域，数据量大 | 从头训练 |
| 已有模型，新增少量数据 | 增量训练 |
| 特定场景优化 | 增量训练 + 冻结骨干网络 |
| 持续迭代优化 | 增量训练 Pipeline |

### Q3: 如何处理大批量图像？
使用 SageMaker Batch Transform：
```python
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.xlarge'
)
transformer.transform(
    data='s3://bucket/batch-images/',
    content_type='application/x-image'
)
```

## 参考资源

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [AWS 官方博客 - PaddleOCR on SageMaker](https://aws.amazon.com/blogs/machine-learning/onboard-paddleocr-with-amazon-sagemaker-projects-for-mlops-to-perform-optical-character-recognition-on-identity-documents/)
- [SageMaker 自定义容器文档](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)
- [SageMaker Pipelines 文档](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
