"""
Demo 2: SageMaker Model Registry 模型版本管理
用于 PaddleOCR 模型的版本追踪和管理
"""

import boto3
import sagemaker
from sagemaker.model import ModelPackage
from datetime import datetime
import json

# 初始化
sm_client = boto3.client('sagemaker')
session = sagemaker.Session()

# 配置
MODEL_PACKAGE_GROUP_NAME = 'paddleocr-drawing-models'
BUCKET_NAME = 'your-ocr-project-bucket'


def create_model_package_group():
    """
    创建模型包组（用于管理同一模型的多个版本）
    """
    try:
        response = sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageGroupDescription='PaddleOCR 图纸识别模型版本管理',
            Tags=[
                {'Key': 'Project', 'Value': 'DrawingOCR'},
                {'Key': 'Framework', 'Value': 'PaddleOCR'},
            ]
        )
        print(f"模型包组已创建: {MODEL_PACKAGE_GROUP_NAME}")
        return response['ModelPackageGroupArn']
    except sm_client.exceptions.ResourceInUseException:
        print(f"模型包组已存在: {MODEL_PACKAGE_GROUP_NAME}")
        return f"arn:aws:sagemaker:{session.boto_region_name}:{boto3.client('sts').get_caller_identity()['Account']}:model-package-group/{MODEL_PACKAGE_GROUP_NAME}"


def register_model_version(
    model_data_url,
    image_uri,
    model_metrics,
    description,
    approval_status='PendingManualApproval'
):
    """
    注册新的模型版本
    
    Args:
        model_data_url: S3 模型路径
        image_uri: 推理容器镜像
        model_metrics: 模型评估指标
        description: 版本描述
        approval_status: 审批状态
    """
    
    # 模型指标报告
    model_metrics_report = {
        'ModelQuality': {
            'Statistics': {
                'ContentType': 'application/json',
                'S3Uri': f's3://{BUCKET_NAME}/metrics/model_quality.json'
            }
        }
    }
    
    # 保存指标到 S3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key='metrics/model_quality.json',
        Body=json.dumps(model_metrics)
    )
    
    # 注册模型版本
    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelPackageDescription=description,
        
        # 推理规格
        InferenceSpecification={
            'Containers': [
                {
                    'Image': image_uri,
                    'ModelDataUrl': model_data_url,
                    'Framework': 'PYTORCH',
                    'FrameworkVersion': '1.12',
                }
            ],
            'SupportedContentTypes': ['application/json', 'image/png', 'image/jpeg'],
            'SupportedResponseMIMETypes': ['application/json'],
            'SupportedRealtimeInferenceInstanceTypes': [
                'ml.m5.large', 'ml.m5.xlarge', 'ml.g4dn.xlarge'
            ],
            'SupportedTransformInstanceTypes': [
                'ml.m5.xlarge', 'ml.m5.2xlarge'
            ]
        },
        
        # 模型指标
        ModelMetrics=model_metrics_report,
        
        # 审批状态
        ModelApprovalStatus=approval_status,
        
        # 元数据
        CustomerMetadataProperties={
            'TrainingDate': datetime.now().strftime('%Y-%m-%d'),
            'Framework': 'PaddleOCR',
            'ModelType': 'PP-OCRv4',
        }
    )
    
    model_package_arn = response['ModelPackageArn']
    version = model_package_arn.split('/')[-1]
    
    print(f"模型版本已注册: v{version}")
    print(f"ARN: {model_package_arn}")
    print(f"审批状态: {approval_status}")
    
    return model_package_arn


def list_model_versions():
    """
    列出所有模型版本
    """
    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        SortBy='CreationTime',
        SortOrder='Descending'
    )
    
    print(f"\n{'='*60}")
    print(f"模型版本列表: {MODEL_PACKAGE_GROUP_NAME}")
    print(f"{'='*60}")
    
    versions = []
    for pkg in response['ModelPackageSummaryList']:
        version_info = {
            'version': pkg['ModelPackageArn'].split('/')[-1],
            'status': pkg['ModelApprovalStatus'],
            'created': str(pkg['CreationTime']),
            'arn': pkg['ModelPackageArn']
        }
        versions.append(version_info)
        
        print(f"\n版本: v{version_info['version']}")
        print(f"  状态: {version_info['status']}")
        print(f"  创建时间: {version_info['created']}")
    
    return versions


def compare_model_versions(version1_arn, version2_arn):
    """
    比较两个模型版本的指标
    """
    v1 = sm_client.describe_model_package(ModelPackageName=version1_arn)
    v2 = sm_client.describe_model_package(ModelPackageName=version2_arn)
    
    print(f"\n{'='*60}")
    print("模型版本对比")
    print(f"{'='*60}")
    
    # 获取指标
    s3 = boto3.client('s3')
    
    def get_metrics(model_package):
        if 'ModelMetrics' in model_package:
            metrics_uri = model_package['ModelMetrics'].get('ModelQuality', {}).get('Statistics', {}).get('S3Uri', '')
            if metrics_uri:
                bucket = metrics_uri.split('/')[2]
                key = '/'.join(metrics_uri.split('/')[3:])
                response = s3.get_object(Bucket=bucket, Key=key)
                return json.loads(response['Body'].read().decode('utf-8'))
        return {}
    
    metrics1 = get_metrics(v1)
    metrics2 = get_metrics(v2)
    
    print(f"\n{'指标':<20} {'版本1':<15} {'版本2':<15} {'变化':<10}")
    print("-" * 60)
    
    all_keys = set(list(metrics1.keys()) + list(metrics2.keys()))
    for key in all_keys:
        val1 = metrics1.get(key, 'N/A')
        val2 = metrics2.get(key, 'N/A')
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            change = f"{((val2 - val1) / val1 * 100):+.2f}%" if val1 != 0 else "N/A"
        else:
            change = "-"
        
        print(f"{key:<20} {str(val1):<15} {str(val2):<15} {change:<10}")


def approve_model_version(model_package_arn):
    """
    审批通过模型版本
    """
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus='Approved'
    )
    print(f"模型已审批通过: {model_package_arn}")


def reject_model_version(model_package_arn, reason):
    """
    拒绝模型版本
    """
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus='Rejected',
        ApprovalDescription=reason
    )
    print(f"模型已拒绝: {model_package_arn}")
    print(f"原因: {reason}")


def deploy_approved_model(model_package_arn, endpoint_name):
    """
    部署已审批的模型版本
    """
    from sagemaker.model import ModelPackage
    
    model = ModelPackage(
        role=sagemaker.get_execution_role(),
        model_package_arn=model_package_arn,
        sagemaker_session=session
    )
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        endpoint_name=endpoint_name
    )
    
    print(f"模型已部署到端点: {endpoint_name}")
    return predictor


def get_latest_approved_model():
    """
    获取最新的已审批模型
    """
    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if response['ModelPackageSummaryList']:
        latest = response['ModelPackageSummaryList'][0]
        print(f"最新已审批模型: {latest['ModelPackageArn']}")
        return latest['ModelPackageArn']
    else:
        print("没有已审批的模型")
        return None


# ============ 演示流程 ============

if __name__ == '__main__':
    print("=" * 60)
    print("Demo 2: Model Registry 模型版本管理")
    print("=" * 60)
    
    # Step 1: 创建模型包组
    print("\n[Step 1] 创建模型包组...")
    group_arn = create_model_package_group()
    
    # Step 2: 注册模型 v1
    print("\n[Step 2] 注册模型 v1（基础模型）...")
    v1_metrics = {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.87,
        'f1_score': 0.85,
        'training_samples': 10000
    }
    # v1_arn = register_model_version(
    #     model_data_url='s3://bucket/models/v1/model.tar.gz',
    #     image_uri='your-ecr-image:v1',
    #     model_metrics=v1_metrics,
    #     description='基础模型 - 初始训练'
    # )
    
    # Step 3: 注册模型 v2（增量训练后）
    print("\n[Step 3] 注册模型 v2（增量训练后）...")
    v2_metrics = {
        'accuracy': 0.91,
        'precision': 0.89,
        'recall': 0.93,
        'f1_score': 0.91,
        'training_samples': 15000
    }
    # v2_arn = register_model_version(
    #     model_data_url='s3://bucket/models/v2/model.tar.gz',
    #     image_uri='your-ecr-image:v2',
    #     model_metrics=v2_metrics,
    #     description='增量训练 - 新增图纸数据'
    # )
    
    # Step 4: 列出所有版本
    print("\n[Step 4] 列出所有模型版本...")
    # versions = list_model_versions()
    
    # Step 5: 版本对比
    print("\n[Step 5] 对比 v1 和 v2...")
    # compare_model_versions(v1_arn, v2_arn)
    
    # Step 6: 审批模型
    print("\n[Step 6] 审批 v2 模型...")
    # approve_model_version(v2_arn)
    
    # Step 7: 部署已审批模型
    print("\n[Step 7] 部署已审批模型...")
    # deploy_approved_model(v2_arn, 'paddleocr-endpoint-v2')
    
    print("\n演示完成！")
