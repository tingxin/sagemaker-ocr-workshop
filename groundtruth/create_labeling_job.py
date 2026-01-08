"""
Ground Truth OCR 标注工作创建脚本
用于创建 P&ID 图纸文字标注任务
"""

import boto3
import json
import os
from datetime import datetime
from pathlib import Path


# ============ 配置 ============

class Config:
    # AWS 配置
    REGION = "us-west-2"  # 修改为你的区域
    
    # S3 配置
    BUCKET = "your-bucket-name"  # 修改为你的 bucket
    INPUT_PREFIX = "groundtruth/input"
    OUTPUT_PREFIX = "groundtruth/output"
    TEMPLATE_PREFIX = "groundtruth/templates"
    
    # 标注工作配置
    JOB_NAME_PREFIX = "pid-ocr-labeling"
    
    # IAM Role (需要 SageMaker 执行权限)
    ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
    
    # 标注团队 (私有团队 ARN)
    WORKTEAM_ARN = "arn:aws:sagemaker:REGION:ACCOUNT:workteam/private-crowd/your-team"
    
    # 任务配置
    TASK_TITLE = "P&ID 图纸文字标注"
    TASK_DESCRIPTION = "框选图纸中的文字区域并输入对应的文字内容"
    TASK_TIME_LIMIT = 900  # 15 分钟
    WORKERS_PER_OBJECT = 1  # 每张图片的标注人数


# ============ 工具函数 ============

def get_clients():
    """获取 AWS 客户端"""
    return {
        's3': boto3.client('s3', region_name=Config.REGION),
        'sagemaker': boto3.client('sagemaker', region_name=Config.REGION)
    }


def upload_template(s3_client, template_type: str = "detection"):
    """上传标注模板到 S3"""
    template_files = {
        "ocr": "ocr_labeling_template.html",
        "mixed": "pid_mixed_labeling_template.html", 
        "detection": "pid_detection_template.html"
    }
    
    template_file = template_files.get(template_type, "pid_detection_template.html")
    template_path = Path(__file__).parent / template_file
    
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    
    s3_key = f"{Config.TEMPLATE_PREFIX}/{template_file}"
    
    print(f"上传模板 ({template_type}): s3://{Config.BUCKET}/{s3_key}")
    s3_client.upload_file(str(template_path), Config.BUCKET, s3_key)
    
    return f"s3://{Config.BUCKET}/{s3_key}"


def create_manifest(s3_client, image_folder: str, output_path: str = "input.manifest", max_images: int = None):
    """
    创建输入 manifest 文件
    
    Args:
        image_folder: 本地图片文件夹路径
        output_path: manifest 输出路径
        max_images: 最大图片数量限制（用于测试）
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    manifest_lines = []
    
    image_folder = Path(image_folder)
    
    # 获取所有图片文件
    all_images = []
    for img_file in image_folder.iterdir():
        if img_file.suffix.lower() in image_extensions:
            all_images.append(img_file)
    
    # 限制图片数量
    if max_images and max_images < len(all_images):
        print(f"📊 限制图片数量: {max_images} / {len(all_images)}")
        all_images = all_images[:max_images]
    else:
        print(f"📊 处理所有图片: {len(all_images)}")
    
    for img_file in all_images:
        # 上传图片到 S3
        s3_key = f"{Config.INPUT_PREFIX}/images/{img_file.name}"
        print(f"  上传: {img_file.name}")
        s3_client.upload_file(str(img_file), Config.BUCKET, s3_key)
        
        # 添加到 manifest
        manifest_lines.append(json.dumps({
            "source-ref": f"s3://{Config.BUCKET}/{s3_key}"
        }))
    
    # 保存 manifest
    manifest_content = "\n".join(manifest_lines)
    with open(output_path, 'w') as f:
        f.write(manifest_content)
    
    # 上传 manifest 到 S3
    manifest_s3_key = f"{Config.INPUT_PREFIX}/input.manifest"
    s3_client.upload_file(output_path, Config.BUCKET, manifest_s3_key)
    
    print(f"\nManifest 已创建: {len(manifest_lines)} 张图片")
    print(f"S3 路径: s3://{Config.BUCKET}/{manifest_s3_key}")
    
    return f"s3://{Config.BUCKET}/{manifest_s3_key}"


def create_labeling_job(sagemaker_client, manifest_uri: str, template_uri: str):
    """创建 Ground Truth 标注工作"""
    
    # 生成唯一的工作名称
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{Config.JOB_NAME_PREFIX}-{timestamp}"
    
    print(f"\n创建标注工作: {job_name}")
    
    response = sagemaker_client.create_labeling_job(
        LabelingJobName=job_name,
        LabelAttributeName="ocr-annotations",
        
        InputConfig={
            'DataSource': {
                'S3DataSource': {
                    'ManifestS3Uri': manifest_uri
                }
            }
        },
        
        OutputConfig={
            'S3OutputPath': f"s3://{Config.BUCKET}/{Config.OUTPUT_PREFIX}/"
        },
        
        RoleArn=Config.ROLE_ARN,
        
        HumanTaskConfig={
            'WorkteamArn': Config.WORKTEAM_ARN,
            
            'UiConfig': {
                'UiTemplateS3Uri': template_uri
            },
            
            'PreHumanTaskLambdaArn': get_pre_human_task_lambda_arn(),
            
            'TaskTitle': Config.TASK_TITLE,
            'TaskDescription': Config.TASK_DESCRIPTION,
            'NumberOfHumanWorkersPerDataObject': Config.WORKERS_PER_OBJECT,
            'TaskTimeLimitInSeconds': Config.TASK_TIME_LIMIT,
            
            'AnnotationConsolidationConfig': {
                'AnnotationConsolidationLambdaArn': get_consolidation_lambda_arn()
            }
        },
        
        Tags=[
            {'Key': 'Project', 'Value': 'PaddleOCR-MLOps'},
            {'Key': 'Task', 'Value': 'OCR-Labeling'}
        ]
    )
    
    print(f"标注工作已创建!")
    print(f"  工作名称: {job_name}")
    print(f"  ARN: {response['LabelingJobArn']}")
    
    return job_name


def get_pre_human_task_lambda_arn():
    """获取预处理 Lambda ARN"""
    # 使用 AWS 内置的 BoundingBox 预处理函数
    return f"arn:aws:lambda:{Config.REGION}:aws:function:PRE-BoundingBox"


def get_consolidation_lambda_arn():
    """获取标注合并 Lambda ARN"""
    # 使用 AWS 内置的 BoundingBox 合并函数
    return f"arn:aws:lambda:{Config.REGION}:aws:function:ACS-BoundingBox"


def check_job_status(sagemaker_client, job_name: str):
    """检查标注工作状态"""
    response = sagemaker_client.describe_labeling_job(
        LabelingJobName=job_name
    )
    
    status = response['LabelingJobStatus']
    counters = response['LabelCounters']
    
    print(f"\n标注工作状态: {job_name}")
    print(f"  状态: {status}")
    print(f"  总数: {counters.get('TotalLabeled', 0)}")
    print(f"  已完成: {counters.get('HumanLabeled', 0)}")
    print(f"  机器标注: {counters.get('MachineLabeled', 0)}")
    print(f"  失败: {counters.get('FailedNonRetryableError', 0)}")
    
    return response


def list_labeling_jobs(sagemaker_client):
    """列出所有标注工作"""
    response = sagemaker_client.list_labeling_jobs(
        NameContains=Config.JOB_NAME_PREFIX,
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=10
    )
    
    print("\n最近的标注工作:")
    print("-" * 60)
    
    for job in response['LabelingJobSummaryList']:
        print(f"  {job['LabelingJobName']}")
        print(f"    状态: {job['LabelingJobStatus']}")
        print(f"    创建时间: {job['CreationTime']}")
        print()
    
    return response['LabelingJobSummaryList']


# ============ 主函数 ============

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ground Truth OCR 标注工作管理')
    parser.add_argument('action', choices=['create', 'status', 'list'],
                        help='操作类型: create=创建工作, status=查看状态, list=列出工作')
    parser.add_argument('--images', type=str, help='图片文件夹路径 (create 时需要)')
    parser.add_argument('--max-images', type=int, help='最大图片数量限制 (用于测试)')
    parser.add_argument('--job-name', type=str, help='工作名称 (status 时需要)')
    parser.add_argument('--bucket', type=str, help='S3 bucket 名称')
    parser.add_argument('--region', type=str, help='AWS 区域')
    parser.add_argument('--template', type=str, choices=['ocr', 'mixed', 'detection'], 
                        default='detection', help='标注模板类型')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.bucket:
        Config.BUCKET = args.bucket
    if args.region:
        Config.REGION = args.region
    
    # 获取客户端
    clients = get_clients()
    
    if args.action == 'create':
        if not args.images:
            print("错误: 创建工作需要指定 --images 参数")
            return
        
        print("=" * 60)
        print("创建 Ground Truth OCR 标注工作")
        print("=" * 60)
        
        # 1. 上传模板
        print("\n[Step 1] 上传标注模板...")
        template_uri = upload_template(clients['s3'], args.template)
        
        # 2. 创建 manifest
        print("\n[Step 2] 创建输入 manifest...")
        manifest_uri = create_manifest(clients['s3'], args.images, max_images=args.max_images)
        
        # 3. 创建标注工作
        print("\n[Step 3] 创建标注工作...")
        job_name = create_labeling_job(clients['sagemaker'], manifest_uri, template_uri)
        
        print("\n" + "=" * 60)
        print("标注工作创建完成!")
        print("=" * 60)
        print(f"\n下一步:")
        print(f"  1. 通知标注团队开始工作")
        print(f"  2. 使用 'python {__file__} status --job-name {job_name}' 查看进度")
        print(f"  3. 完成后运行 convert_output.py 转换标注结果")
        
    elif args.action == 'status':
        if not args.job_name:
            print("错误: 查看状态需要指定 --job-name 参数")
            return
        check_job_status(clients['sagemaker'], args.job_name)
        
    elif args.action == 'list':
        list_labeling_jobs(clients['sagemaker'])


if __name__ == '__main__':
    main()
