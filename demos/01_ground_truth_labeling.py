"""
Demo 1: SageMaker Ground Truth 数据标注流程
用于图纸 OCR 标注任务创建和管理
"""

import boto3
import json
from datetime import datetime

# 初始化客户端
sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')

# 配置参数
BUCKET_NAME = 'your-ocr-project-bucket'
PROJECT_NAME = 'drawing-ocr'
REGION = 'us-east-1'


def create_labeling_job():
    """
    创建 Ground Truth 标注任务
    用于图纸文字和符号的 Bounding Box 标注
    """
    
    job_name = f"{PROJECT_NAME}-labeling-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 标注任务配置
    labeling_job_config = {
        'LabelingJobName': job_name,
        'LabelAttributeName': 'ocr-labels',
        
        # 输入数据配置
        'InputConfig': {
            'DataSource': {
                'S3DataSource': {
                    'ManifestS3Uri': f's3://{BUCKET_NAME}/{PROJECT_NAME}/input/manifest.json'
                }
            },
            'DataAttributes': {
                'ContentClassifiers': ['FreeOfPersonallyIdentifiableInformation']
            }
        },
        
        # 输出配置
        'OutputConfig': {
            'S3OutputPath': f's3://{BUCKET_NAME}/{PROJECT_NAME}/output/'
        },
        
        # 标注人员配置（私有团队）
        'HumanTaskConfig': {
            'WorkteamArn': f'arn:aws:sagemaker:{REGION}:YOUR_ACCOUNT:workteam/private-crowd/your-team',
            'UiConfig': {
                'UiTemplateS3Uri': f's3://{BUCKET_NAME}/{PROJECT_NAME}/templates/ocr-template.html'
            },
            'PreHumanTaskLambdaArn': f'arn:aws:lambda:{REGION}:YOUR_ACCOUNT:function:pre-labeling',
            'TaskTitle': '图纸 OCR 标注任务',
            'TaskDescription': '请标注图纸中的文字区域，并输入对应的文字内容',
            'NumberOfHumanWorkersPerDataObject': 1,
            'TaskTimeLimitInSeconds': 600,
            'AnnotationConsolidationConfig': {
                'AnnotationConsolidationLambdaArn': f'arn:aws:lambda:{REGION}:YOUR_ACCOUNT:function:consolidation'
            }
        },
        
        # IAM 角色
        'RoleArn': f'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerGroundTruthRole',
        
        # 标签类别
        'LabelCategoryConfigS3Uri': f's3://{BUCKET_NAME}/{PROJECT_NAME}/config/label-categories.json'
    }
    
    response = sagemaker.create_labeling_job(**labeling_job_config)
    print(f"标注任务已创建: {job_name}")
    return job_name


def create_manifest_file(image_list):
    """
    创建输入数据清单文件
    """
    manifest_lines = []
    for image_path in image_list:
        manifest_lines.append(json.dumps({
            'source-ref': f's3://{BUCKET_NAME}/{PROJECT_NAME}/images/{image_path}'
        }))
    
    manifest_content = '\n'.join(manifest_lines)
    
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f'{PROJECT_NAME}/input/manifest.json',
        Body=manifest_content
    )
    print(f"清单文件已创建，包含 {len(image_list)} 张图像")


def create_label_categories():
    """
    创建标签类别配置
    针对图纸场景的标签类型
    """
    categories = {
        'document-version': '2018-11-28',
        'labels': [
            {'label': 'text', 'attributes': [
                {'name': 'content', 'type': 'string'},
                {'name': 'text_type', 'type': 'string', 
                 'enum': ['title', 'dimension', 'note', 'symbol', 'other']}
            ]},
            {'label': 'symbol', 'attributes': [
                {'name': 'symbol_type', 'type': 'string',
                 'enum': ['tolerance', 'surface', 'weld', 'other']}
            ]},
            {'label': 'table', 'attributes': []},
            {'label': 'title_block', 'attributes': []}
        ]
    }
    
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f'{PROJECT_NAME}/config/label-categories.json',
        Body=json.dumps(categories, ensure_ascii=False)
    )
    print("标签类别配置已创建")


def create_ui_template():
    """
    创建自定义标注界面模板
    """
    template = '''
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<crowd-form>
  <crowd-bounding-box
    name="annotations"
    src="{{ task.input.source-ref | grant_read_access }}"
    header="请标注图纸中的文字和符号区域"
    labels="['text', 'symbol', 'table', 'title_block']"
  >
    <full-instructions header="标注说明">
      <p>请仔细标注图纸中的以下内容：</p>
      <ul>
        <li><strong>text</strong>: 所有文字内容（标题、尺寸、注释等）</li>
        <li><strong>symbol</strong>: 特殊符号（公差符号、表面粗糙度等）</li>
        <li><strong>table</strong>: 表格区域</li>
        <li><strong>title_block</strong>: 标题栏</li>
      </ul>
    </full-instructions>

    <short-instructions>
      标注图纸中的文字、符号、表格和标题栏
    </short-instructions>
  </crowd-bounding-box>
  
  <!-- 文字内容输入 -->
  <div style="margin-top: 20px;">
    <h3>请输入标注区域的文字内容（如适用）</h3>
    <crowd-text-area name="text_content" placeholder="输入识别的文字内容"></crowd-text-area>
  </div>
</crowd-form>
'''
    
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f'{PROJECT_NAME}/templates/ocr-template.html',
        Body=template
    )
    print("标注界面模板已创建")


def get_labeling_job_status(job_name):
    """
    获取标注任务状态
    """
    response = sagemaker.describe_labeling_job(LabelingJobName=job_name)
    
    status = {
        'job_name': job_name,
        'status': response['LabelingJobStatus'],
        'created_time': str(response['CreationTime']),
        'labeled_count': response.get('LabelCounters', {}).get('TotalLabeled', 0),
        'total_count': response.get('LabelCounters', {}).get('MachineLabeled', 0) + 
                       response.get('LabelCounters', {}).get('HumanLabeled', 0)
    }
    
    print(f"任务状态: {status}")
    return status


def export_to_paddleocr_format(output_manifest_path, output_dir):
    """
    将 Ground Truth 输出转换为 PaddleOCR 训练格式
    """
    # 下载输出清单
    response = s3.get_object(Bucket=BUCKET_NAME, Key=output_manifest_path)
    manifest_content = response['Body'].read().decode('utf-8')
    
    train_labels = []
    
    for line in manifest_content.strip().split('\n'):
        record = json.loads(line)
        image_path = record['source-ref'].split('/')[-1]
        
        # 解析标注结果
        if 'ocr-labels' in record:
            annotations = record['ocr-labels']['annotations']
            
            for ann in annotations:
                if ann['label'] == 'text':
                    # 提取边界框坐标
                    box = [
                        [ann['left'], ann['top']],
                        [ann['left'] + ann['width'], ann['top']],
                        [ann['left'] + ann['width'], ann['top'] + ann['height']],
                        [ann['left'], ann['top'] + ann['height']]
                    ]
                    
                    # 获取文字内容
                    text_content = record.get('text_content', '')
                    
                    train_labels.append({
                        'image': image_path,
                        'box': box,
                        'text': text_content
                    })
    
    # 生成 PaddleOCR 格式的标签文件
    label_lines = []
    for item in train_labels:
        label_lines.append(f"{item['image']}\t{item['text']}")
    
    # 保存标签文件
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f'{output_dir}/train.txt',
        Body='\n'.join(label_lines)
    )
    
    print(f"已导出 {len(train_labels)} 条训练数据到 PaddleOCR 格式")
    return train_labels


# ============ 演示流程 ============

if __name__ == '__main__':
    print("=" * 50)
    print("Demo 1: Ground Truth 数据标注流程")
    print("=" * 50)
    
    # Step 1: 准备配置文件
    print("\n[Step 1] 创建标签类别配置...")
    create_label_categories()
    
    # Step 2: 创建标注界面模板
    print("\n[Step 2] 创建标注界面模板...")
    create_ui_template()
    
    # Step 3: 创建输入清单
    print("\n[Step 3] 创建输入数据清单...")
    sample_images = [
        'drawing_001.png',
        'drawing_002.png',
        'drawing_003.png'
    ]
    create_manifest_file(sample_images)
    
    # Step 4: 创建标注任务
    print("\n[Step 4] 创建标注任务...")
    # job_name = create_labeling_job()  # 实际执行时取消注释
    
    # Step 5: 查看任务状态
    print("\n[Step 5] 查看任务状态...")
    # get_labeling_job_status(job_name)  # 实际执行时取消注释
    
    # Step 6: 导出为 PaddleOCR 格式
    print("\n[Step 6] 导出训练数据...")
    # export_to_paddleocr_format(
    #     'output/manifest.json',
    #     'training-data'
    # )
    
    print("\n演示完成！")
