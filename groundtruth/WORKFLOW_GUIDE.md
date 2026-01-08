# Ground Truth 标注完整工作流程指南

## 📋 目录
1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [配置 Ground Truth](#3-配置-ground-truth)
4. [创建标注工作](#4-创建标注工作)
5. [执行标注](#5-执行标注)
6. [下载标注结果](#6-下载标注结果)
7. [转换为训练格式](#7-转换为训练格式)
8. [训练检测模型](#8-训练检测模型)
9. [大模型识别文字](#9-大模型识别文字)
10. [故障排查](#10-故障排查)

---

## 1. 环境准备

### 1.1 安装依赖
```bash
pip install boto3 python-dotenv Pillow tqdm
```

### 1.2 配置 AWS 凭证
```bash
aws configure
# 输入 Access Key ID
# 输入 Secret Access Key
# 输入 Region (如 us-east-2)
```

### 1.3 配置环境变量
```bash
cd groundtruth
cp .env.example .env
# 编辑 .env 文件，填入你的配置
```

---

## 2. 数据准备

### 2.1 下载数据集
```bash
python data/07_dataset_download_preprocess.py
```

**输出：**
- `dataset_pid/processed/images/` - 处理后的图片
- `dataset_pid/paddleocr_format/` - PaddleOCR 格式标签

### 2.2 验证数据
```bash
ls dataset_pid/processed/images/ | wc -l
# 应该显示图片数量
```

---

## 3. 配置 Ground Truth

### 3.1 设置 S3 CORS
```bash
python groundtruth/setup_s3_cors.py --bucket your-bucket-name
```

**验证：**
```bash
python groundtruth/setup_s3_cors.py --bucket your-bucket-name --check
```

### 3.2 创建标注团队（首次使用）

**方式一：AWS Console**
1. SageMaker → Ground Truth → Labeling workforces
2. Private → Create private team
3. 添加标注员邮箱
4. 记录 Workteam ARN

**方式二：AWS CLI**
```bash
aws sagemaker create-workteam \
  --workteam-name my-labeling-team \
  --member-definitions \
    CognitoMemberDefinition={UserPool=xxx,UserGroup=xxx,ClientId=xxx} \
  --description "P&ID 标注团队"
```

### 3.3 更新 .env 配置
```bash
# 编辑 groundtruth/.env
WORKTEAM_ARN=arn:aws:sagemaker:region:account:workteam/private-crowd/your-team
```

---

## 4. 创建标注工作

### 4.1 测试流程（5张图片）
```bash
python groundtruth/create_labeling_job.py create \
  --images dataset_pid/processed/images \
  --max-images 5 \
  --template simple
```

### 4.2 生产环境（所有图片）
```bash
python groundtruth/create_labeling_job.py create \
  --images dataset_pid/processed/images \
  --template simple
```

**输出：**
```
标注工作已创建!
  工作名称: pid-ocr-labeling-20260108-061433
  ARN: arn:aws:sagemaker:...
```

### 4.3 查看工作状态
```bash
python groundtruth/create_labeling_job.py status \
  --job-name pid-ocr-labeling-20260108-061433
```

---

## 5. 执行标注

### 5.1 标注员登录
1. 标注员收到邮件邀请
2. 点击链接设置密码
3. 登录标注平台

### 5.2 开始标注
1. 选择待标注任务
2. 框选对象（符号或文字）
3. 选择正确的类别
4. 提交标注

### 5.3 标注规范

**符号标注（10类）：**
- valve, pump, tank, heat_exchanger, compressor
- filter, instrument, reducer, flange, pipe

**文字标注（5类）：**
- text_english - 英文
- text_chinese - 中文
- text_number - 纯数字
- text_mixed - 混合（如 P-101, DN50）
- text_symbol - 符号（如 Φ50, ±0.1）

### 5.4 监控进度
```bash
# 定期检查状态
python groundtruth/create_labeling_job.py status \
  --job-name pid-ocr-labeling-20260108-061433
```

---

## 6. 下载标注结果

### 6.1 等待完成
```bash
# 状态变为 Completed 后继续
python groundtruth/create_labeling_job.py status \
  --job-name pid-ocr-labeling-20260108-061433
```

### 6.2 下载输出
```bash
python groundtruth/convert_output.py \
  --bucket your-bucket-name \
  --job-name pid-ocr-labeling-20260108-061433 \
  --output-dir ./labeled_data
```

**输出结构：**
```
labeled_data/
├── temp/
│   └── output.manifest          # Ground Truth 原始输出
└── paddleocr_format/
    ├── label_train.txt          # 训练集标签
    ├── label_val.txt            # 验证集标签
    └── label_test.txt           # 测试集标签
```

### 6.3 验证输出
```bash
# 查看标注数量
wc -l labeled_data/paddleocr_format/label_train.txt
wc -l labeled_data/paddleocr_format/label_val.txt
wc -l labeled_data/paddleocr_format/label_test.txt

# 查看标注内容
head -n 2 labeled_data/paddleocr_format/label_train.txt
```

---

## 7. 转换为训练格式

### 7.1 检测模型格式（YOLO/COCO）

标注结果已包含：
- 边界框坐标
- 类别标签
- 图片路径

**转换为 YOLO 格式：**
```python
# 创建转换脚本
python training/convert_to_yolo.py \
  --input labeled_data/paddleocr_format \
  --output training_data/yolo_format
```

### 7.2 文字识别数据准备

**裁剪文字区域：**
```python
python training/crop_text_regions.py \
  --images dataset_pid/processed/images \
  --labels labeled_data/paddleocr_format/label_train.txt \
  --output training_data/text_crops
```

---

## 8. 训练检测模型

### 8.1 准备训练环境
```bash
# 安装 YOLOv8
pip install ultralytics

# 或使用 SageMaker Training Job
```

### 8.2 训练检测模型
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练
results = model.train(
    data='training_data/yolo_format/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='pid_detection'
)
```

### 8.3 评估模型
```python
# 验证
metrics = model.val()

# 推理测试
results = model.predict('test_image.jpg')
```

---

## 9. 大模型识别文字

### 9.1 使用检测模型定位文字
```python
# 检测所有对象
results = model.predict('pid_image.jpg')

# 筛选文字区域
text_regions = [
    box for box in results[0].boxes 
    if box.cls in ['text_english', 'text_chinese', 'text_mixed']
]
```

### 9.2 裁剪文字区域
```python
from PIL import Image

for i, box in enumerate(text_regions):
    x1, y1, x2, y2 = box.xyxy[0]
    crop = image.crop((x1, y1, x2, y2))
    crop.save(f'text_crop_{i}.jpg')
```

### 9.3 调用大模型识别
```python
import boto3

# 使用 AWS Bedrock
bedrock = boto3.client('bedrock-runtime')

# 或使用 OpenAI GPT-4V
import openai

response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "识别图片中的文字"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]
)

text_content = response.choices[0].message.content
```

### 9.4 结果融合
```python
# 合并检测和识别结果
final_results = {
    "symbols": symbol_detections,
    "texts": [
        {
            "bbox": box,
            "type": text_type,
            "content": recognized_text
        }
        for box, text_type, recognized_text in zip(...)
    ]
}
```

---

## 10. 故障排查

### 10.1 CORS 错误
**问题：** 标注界面白屏，提示 CORS 错误

**解决：**
```bash
python groundtruth/setup_s3_cors.py --bucket your-bucket-name
```

### 10.2 Lambda ARN 错误
**问题：** 创建工作失败，Lambda ARN 格式错误

**解决：** 检查 `create_labeling_job.py` 中的区域配置

### 10.3 权限错误
**问题：** 无法访问 S3 或创建工作

**解决：**
1. 检查 SageMaker 执行角色权限
2. 确保角色有 S3 读写权限
3. 确保角色有 Ground Truth 权限

### 10.4 标注失败
**问题：** 状态显示 "Complete with labeling errors"

**解决：**
1. 检查模板语法
2. 验证标签配置文件
3. 查看 CloudWatch 日志

---

## 📊 完整流程图

```
数据准备 → 配置环境 → 创建标注工作
    ↓
标注员标注 → 下载结果 → 转换格式
    ↓
训练检测模型 → 推理定位 → 大模型识别
    ↓
结果融合 → 完整的图纸理解
```

---

## 🎯 快速开始（5分钟测试）

```bash
# 1. 准备数据（5张图片）
python data/07_dataset_download_preprocess.py

# 2. 设置 CORS
python groundtruth/setup_s3_cors.py --bucket your-bucket

# 3. 创建标注工作
python groundtruth/create_labeling_job.py create \
  --images dataset_pid/processed/images \
  --max-images 5 \
  --template simple

# 4. 标注（手动）
# 访问标注平台，完成 5 张图片的标注

# 5. 下载结果
python groundtruth/convert_output.py \
  --bucket your-bucket \
  --job-name <job-name> \
  --output-dir ./test_output

# 6. 查看结果
cat test_output/paddleocr_format/label_train.txt
```

---

## 📚 相关文档

- [Ground Truth 官方文档](https://docs.aws.amazon.com/sagemaker/latest/dg/sms.html)
- [PaddleOCR 文档](https://github.com/PaddlePaddle/PaddleOCR)
- [YOLOv8 文档](https://docs.ultralytics.com/)

---

## 💡 最佳实践

1. **先小规模测试** - 用 5-10 张图片验证流程
2. **标注规范统一** - 制定详细的标注指南
3. **定期检查质量** - 抽查标注结果
4. **增量训练** - 逐步增加数据量
5. **版本管理** - 记录每次标注和训练的版本

---

## 🔗 下一步

完成标注后，可以：
1. 训练更大规模的检测模型
2. 集成到 MLOps 流程
3. 部署为 SageMaker 端点
4. 构建完整的图纸理解系统