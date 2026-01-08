#!/usr/bin/env python3
"""
Demo 07: PaddleOCR 训练 Pipeline
演示如何使用 SageMaker Pipeline 训练 PaddleOCR 检测模型
"""

import os
import sys
import boto3
import sagemaker
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sagemaker_pipelines.paddleocr.pid_pipeline import get_pid_detection_pipeline


def main():
    """主函数"""
    
    print("=" * 60)
    print("Demo 07: PaddleOCR 训练 Pipeline")
    print("=" * 60)
    
    # AWS 配置
    region = "us-east-2"
    bucket = "tx-ocr-ml-workshop2"
    role_arn = "arn:aws:iam::515491257789:role/service-role/AmazonSageMaker-ExecutionRole-20260105T221128"
    
    # 检查训练数据是否存在
    training_data_path = project_root / "pid_training_data"
    if not training_data_path.exists():
        print("❌ 错误: 未找到训练数据目录")
        print(f"   请确保 {training_data_path} 存在")
        print("   运行 Ground Truth 标注并转换数据后再试")
        return
    
    paddleocr_format_path = training_data_path / "paddleocr_format"
    if not paddleocr_format_path.exists():
        print("❌ 错误: 未找到 PaddleOCR 格式数据")
        print(f"   请确保 {paddleocr_format_path} 存在")
        print("   运行数据转换脚本后再试")
        return
    
    # 检查标签文件
    train_labels = paddleocr_format_path / "label_train.txt"
    val_labels = paddleocr_format_path / "label_val.txt"
    
    if not train_labels.exists():
        print(f"❌ 错误: 未找到训练标签文件 {train_labels}")
        return
    
    if not val_labels.exists():
        print(f"❌ 错误: 未找到验证标签文件 {val_labels}")
        return
    
    print(f"✅ 找到训练数据: {train_labels}")
    print(f"✅ 找到验证数据: {val_labels}")
    
    # 统计数据
    with open(train_labels, 'r', encoding='utf-8') as f:
        train_count = len(f.readlines())
    
    with open(val_labels, 'r', encoding='utf-8') as f:
        val_count = len(f.readlines())
    
    print(f"📊 训练图片数量: {train_count}")
    print(f"📊 验证图片数量: {val_count}")
    
    if train_count == 0:
        print("❌ 错误: 训练数据为空")
        return
    
    # 创建 SageMaker session
    print("\n🔧 创建 SageMaker Pipeline...")
    
    try:
        # 获取 Pipeline
        pipeline = get_pid_detection_pipeline(
            region=region,
            role=role_arn,
            default_bucket=bucket,
            model_package_group_name="PIDDetectionPackageGroup",
            pipeline_name="PIDDetectionPipeline",
            base_job_prefix="PIDDetection",
            project_id="PIDDetectionProject"
        )
        
        print(f"✅ Pipeline 创建成功: {pipeline.name}")
        
        # 上传训练数据到 S3
        print("\n📤 上传训练数据到 S3...")
        
        session = sagemaker.Session()
        
        # 上传整个训练数据目录
        s3_input_path = session.upload_data(
            path=str(training_data_path),
            bucket=bucket,
            key_prefix="pid-training-data"
        )
        
        print(f"✅ 数据已上传到: {s3_input_path}")
        
        # 创建或更新 Pipeline
        print("\n🚀 部署 Pipeline...")
        
        pipeline.upsert(role_arn=role_arn)
        print(f"✅ Pipeline 已部署: {pipeline.name}")
        
        # 启动 Pipeline 执行
        print("\n▶️  启动 Pipeline 执行...")
        
        execution = pipeline.start(
            parameters={
                "InputDataUrl": s3_input_path,
                "TrainingInstanceType": "ml.g4dn.xlarge",  # GPU 实例
                "Epochs": 30,  # 减少训练轮数用于测试
                "BatchSize": 4,  # 减少批次大小
                "LearningRate": 0.001
            }
        )
        
        print(f"✅ Pipeline 执行已启动")
        print(f"📋 执行 ARN: {execution.arn}")
        print(f"🔗 控制台链接: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/{pipeline.name}/executions/{execution.arn.split('/')[-1]}")
        
        print("\n" + "=" * 60)
        print("Pipeline 执行信息:")
        print("=" * 60)
        print(f"Pipeline 名称: {pipeline.name}")
        print(f"执行 ID: {execution.arn.split('/')[-1]}")
        print(f"输入数据: {s3_input_path}")
        print(f"训练实例: ml.g4dn.xlarge")
        print(f"训练参数: epochs=30, batch_size=4, lr=0.001")
        
        print("\n📝 后续步骤:")
        print("1. 在 SageMaker 控制台监控 Pipeline 执行状态")
        print("2. 查看训练日志和指标")
        print("3. 训练完成后检查模型注册情况")
        print("4. 如果模型性能满足要求，可以部署推理端点")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()