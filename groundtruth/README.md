# Ground Truth OCR 标注工具

用于 P&ID 图纸文字标注的 AWS Ground Truth 自定义模板和工具集。

## 文件说明

```
groundtruth/
├── ocr_labeling_template.html  # 自定义标注模板 (Box + Text)
├── create_labeling_job.py      # 创建标注工作脚本
├── convert_output.py           # 转换输出为 PaddleOCR 格式
└── README.md
```

## 快速开始

### 1. 配置 AWS 凭证

```bash
aws configure
```

### 2. 修改配置

编辑 `create_labeling_job.py` 中的 `Config` 类：

```python
class Config:
    REGION = "us-west-2"           # 你的区域
    BUCKET = "your-bucket-name"    # 你的 S3 bucket
    ROLE_ARN = "arn:aws:iam::..."  # SageMaker 执行角色
    WORKTEAM_ARN = "arn:aws:..."   # 标注团队 ARN
```

### 3. 创建标注工作

```bash
python create_labeling_job.py create --images ./your-images-folder
```

### 4. 查看进度

```bash
python create_labeling_job.py status --job-name pid-ocr-labeling-20240101-120000
```

### 5. 转换输出

```bash
python convert_output.py \
  --bucket your-bucket \
  --job-name pid-ocr-labeling-20240101-120000 \
  --output-dir ./labeled_data
```

## 标注模板功能

- ✅ 矩形框选文字区域
- ✅ 输入文字内容
- ✅ 选择文字类别 (text/equipment_id/dimension/tag/note)
- ✅ 选择文字类型 (中文/英文/数字/混合)
- ✅ 实时统计标注数量

## 输出格式

转换后生成 PaddleOCR 标准格式：

```
label_train.txt
label_val.txt  
label_test.txt
```

每行格式：
```
image.jpg	[{"transcription": "P-101", "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}]
```
