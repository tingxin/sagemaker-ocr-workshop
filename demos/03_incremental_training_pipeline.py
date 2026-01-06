"""
Demo 3: SageMaker Pipeline 增量训练流程
自动化 PaddleOCR 模型增量训练和版本管理
"""

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep, CreateModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.estimator import Estimator
from sagemaker.processing import ScriptProcessor
from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role

# 初始化
session = sagemaker.Session()
role = get_execution_role()
region = session.boto_region_name
account_id = boto3.client('sts').get_caller_identity()['Account']

# 配置
BUCKET_NAME = 'your-ocr-project-bucket'
PROJECT_NAME = 'paddleocr-drawing'
MODEL_PACKAGE_GROUP = 'paddleocr-drawing-models'


def create_incremental_training_pipeline():
    """
    创建增量训练 Pipeline
    """
    
    # ============ 定义 Pipeline 参数 ============
    
    # 基础模型路径（用于增量训练）
    base_model_uri = ParameterString(
        name="BaseModelUri",
        default_value=f"s3://{BUCKET_NAME}/{PROJECT_NAME}/models/base/model.tar.gz"
    )
    
    # 新增训练数据路径
    training_data_uri = ParameterString(
        name="TrainingDataUri",
        default_value=f"s3://{BUCKET_NAME}/{PROJECT_NAME}/data/incremental/"
    )
    
    # 训练参数
    epochs = ParameterInteger(name="Epochs", default_value=50)
    batch_size = ParameterInteger(name="BatchSize", default_value=32)
    learning_rate = ParameterString(name="LearningRate", default_value="0.0001")
    
    # 模型审批阈值
    accuracy_threshold = ParameterString(name="AccuracyThreshold", default_value="0.85")
    
    # 实例类型
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.p3.2xlarge"
    )
    
    # ============ Step 1: 数据预处理 ============
    
    processing_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/paddleocr-processing:latest"
    
    processor = ScriptProcessor(
        role=role,
        image_uri=processing_image_uri,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        command=["python3"],
        sagemaker_session=session
    )
    
    preprocessing_step = ProcessingStep(
        name="DataPreprocessing",
        processor=processor,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=training_data_uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{BUCKET_NAME}/{PROJECT_NAME}/processed/train/"
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{BUCKET_NAME}/{PROJECT_NAME}/processed/validation/"
            )
        ],
        code="preprocessing.py"
    )
    
    # ============ Step 2: 增量训练 ============
    
    training_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/paddleocr-training:latest"
    
    estimator = Estimator(
        image_uri=training_image_uri,
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"s3://{BUCKET_NAME}/{PROJECT_NAME}/output/",
        model_uri=base_model_uri,  # 关键：指向基础模型进行增量训练
        hyperparameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "freeze_backbone": "true",  # 冻结骨干网络
            "mixed_data_ratio": "0.3"   # 混合 30% 旧数据
        },
        sagemaker_session=session
    )
    
    training_step = TrainingStep(
        name="IncrementalTraining",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
            "validation": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
            )
        }
    )
    
    # ============ Step 3: 模型评估 ============
    
    evaluation_processor = ScriptProcessor(
        role=role,
        image_uri=processing_image_uri,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        command=["python3"],
        sagemaker_session=session
    )
    
    # 评估结果文件
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    evaluation_step = ProcessingStep(
        name="ModelEvaluation",
        processor=evaluation_processor,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            sagemaker.processing.ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{BUCKET_NAME}/{PROJECT_NAME}/evaluation/"
            )
        ],
        code="evaluation.py",
        property_files=[evaluation_report]
    )
    
    # ============ Step 4: 条件判断（是否注册模型） ============
    
    # 从评估报告中获取准确率
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy"
        ),
        right=float(accuracy_threshold.default_value)
    )
    
    # ============ Step 5: 模型注册 ============
    
    inference_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/paddleocr-inference:latest"
    
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json", "image/png", "image/jpeg"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge", "ml.g4dn.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="PendingManualApproval",
        model_metrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": f"s3://{BUCKET_NAME}/{PROJECT_NAME}/evaluation/evaluation.json"
                }
            }
        }
    )
    
    # 条件步骤：只有准确率达标才注册模型
    condition_step = ConditionStep(
        name="CheckAccuracy",
        conditions=[accuracy_condition],
        if_steps=[register_step],
        else_steps=[]  # 不达标则不注册
    )
    
    # ============ 创建 Pipeline ============
    
    pipeline = Pipeline(
        name=f"{PROJECT_NAME}-incremental-pipeline",
        parameters=[
            base_model_uri,
            training_data_uri,
            epochs,
            batch_size,
            learning_rate,
            accuracy_threshold,
            training_instance_type
        ],
        steps=[
            preprocessing_step,
            training_step,
            evaluation_step,
            condition_step
        ],
        sagemaker_session=session
    )
    
    return pipeline


def run_incremental_training(
    base_model_path,
    new_data_path,
    epochs=50,
    learning_rate="0.0001"
):
    """
    执行增量训练
    """
    pipeline = create_incremental_training_pipeline()
    
    # 提交 Pipeline
    pipeline.upsert(role_arn=role)
    
    # 启动执行
    execution = pipeline.start(
        parameters={
            "BaseModelUri": base_model_path,
            "TrainingDataUri": new_data_path,
            "Epochs": epochs,
            "LearningRate": learning_rate
        }
    )
    
    print(f"Pipeline 执行已启动: {execution.arn}")
    return execution


def monitor_pipeline_execution(execution):
    """
    监控 Pipeline 执行状态
    """
    execution.wait()
    
    # 获取执行结果
    steps = execution.list_steps()
    
    print("\n" + "=" * 60)
    print("Pipeline 执行结果")
    print("=" * 60)
    
    for step in steps:
        print(f"\n步骤: {step['StepName']}")
        print(f"  状态: {step['StepStatus']}")
        if 'FailureReason' in step:
            print(f"  失败原因: {step['FailureReason']}")
    
    return steps


# ============ 辅助脚本 ============

PREPROCESSING_SCRIPT = '''
"""
preprocessing.py - 数据预处理脚本
"""
import os
import json
import random
from pathlib import Path

def main():
    input_dir = "/opt/ml/processing/input"
    train_output = "/opt/ml/processing/output/train"
    val_output = "/opt/ml/processing/output/validation"
    
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    
    # 读取所有数据
    all_data = []
    for file in Path(input_dir).glob("*.json"):
        with open(file) as f:
            all_data.extend(json.load(f))
    
    # 数据清洗
    cleaned_data = []
    for item in all_data:
        # 过滤无效数据
        if item.get("text") and item.get("image"):
            cleaned_data.append(item)
    
    # 划分训练集和验证集 (8:2)
    random.shuffle(cleaned_data)
    split_idx = int(len(cleaned_data) * 0.8)
    
    train_data = cleaned_data[:split_idx]
    val_data = cleaned_data[split_idx:]
    
    # 保存
    with open(f"{train_output}/train.json", "w") as f:
        json.dump(train_data, f)
    
    with open(f"{val_output}/validation.json", "w") as f:
        json.dump(val_data, f)
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")

if __name__ == "__main__":
    main()
'''

EVALUATION_SCRIPT = '''
"""
evaluation.py - 模型评估脚本
"""
import os
import json
import tarfile
from paddleocr import PaddleOCR

def main():
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    output_dir = "/opt/ml/processing/evaluation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 解压模型
    with tarfile.open(f"{model_dir}/model.tar.gz", "r:gz") as tar:
        tar.extractall(model_dir)
    
    # 加载模型
    ocr = PaddleOCR(rec_model_dir=model_dir)
    
    # 加载测试数据
    with open(f"{test_dir}/validation.json") as f:
        test_data = json.load(f)
    
    # 评估
    correct = 0
    total = len(test_data)
    
    for item in test_data:
        result = ocr.ocr(item["image"])
        predicted_text = " ".join([line[1][0] for line in result[0]]) if result[0] else ""
        
        if predicted_text.strip() == item["text"].strip():
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # 保存评估结果
    metrics = {
        "metrics": {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_samples": correct
        }
    }
    
    with open(f"{output_dir}/evaluation.json", "w") as f:
        json.dump(metrics, f)
    
    print(f"评估完成: 准确率 = {accuracy:.4f}")

if __name__ == "__main__":
    main()
'''


# ============ 演示流程 ============

if __name__ == '__main__':
    print("=" * 60)
    print("Demo 3: 增量训练 Pipeline")
    print("=" * 60)
    
    # Step 1: 创建 Pipeline
    print("\n[Step 1] 创建增量训练 Pipeline...")
    pipeline = create_incremental_training_pipeline()
    print(f"Pipeline 定义完成: {pipeline.name}")
    
    # Step 2: 查看 Pipeline 结构
    print("\n[Step 2] Pipeline 结构:")
    print(json.dumps(pipeline.definition(), indent=2, default=str)[:500] + "...")
    
    # Step 3: 提交 Pipeline
    print("\n[Step 3] 提交 Pipeline 到 SageMaker...")
    # pipeline.upsert(role_arn=role)
    
    # Step 4: 执行增量训练
    print("\n[Step 4] 启动增量训练...")
    # execution = run_incremental_training(
    #     base_model_path="s3://bucket/models/v1/model.tar.gz",
    #     new_data_path="s3://bucket/data/new-drawings/",
    #     epochs=50,
    #     learning_rate="0.0001"
    # )
    
    # Step 5: 监控执行
    print("\n[Step 5] 监控 Pipeline 执行...")
    # monitor_pipeline_execution(execution)
    
    print("\n演示完成！")
