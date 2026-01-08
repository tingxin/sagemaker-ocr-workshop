"""
P&ID 检测模型训练 Pipeline
适配我们的 Ground Truth 标注数据
"""

import os
import boto3
import logging
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)


def get_session(region, default_bucket):
    """获取 SageMaker session"""
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pid_detection_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="PIDDetectionPackageGroup",
    pipeline_name="PIDDetectionPipeline",
    base_job_prefix="PIDDetection",
    project_id="PIDDetectionProject"
):
    """
    创建 P&ID 检测模型训练 Pipeline
    
    Args:
        region: AWS 区域
        role: IAM 角色
        default_bucket: S3 bucket
        model_package_group_name: 模型包组名称
        pipeline_name: Pipeline 名称
        base_job_prefix: 作业前缀
        project_id: 项目 ID
    
    Returns:
        Pipeline 实例
    """
    
    sagemaker_session = get_session(region, default_bucket)
    if not default_bucket:
        default_bucket = sagemaker_session.default_bucket()
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    # Pipeline 参数
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", 
        default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", 
        default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", 
        default_value="ml.g4dn.xlarge"  # GPU 实例用于训练
    )
    inference_instance_type = ParameterString(
        name="InferenceInstanceType", 
        default_value="ml.g4dn.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", 
        default_value="PendingManualApproval"
    )
    
    # 输入数据路径
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/pid-training-data/"
    )
    
    # 训练超参数
    epochs = ParameterInteger(name="Epochs", default_value=50)
    batch_size = ParameterInteger(name="BatchSize", default_value=8)
    learning_rate = ParameterFloat(name="LearningRate", default_value=0.001)
    
    # 数据预处理步骤
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/pid-data-preprocessing",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    step_process = ProcessingStep(
        name="PreprocessPIDData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/train",
                destination=f"s3://{default_bucket}/{base_job_prefix}/processed-data/train"
            ),
            ProcessingOutput(
                output_name="validation_data", 
                source="/opt/ml/processing/validation",
                destination=f"s3://{default_bucket}/{base_job_prefix}/processed-data/validation"
            )
        ],
        code=os.path.join(BASE_DIR, "pid_preprocess.py"),
    )
    
    # 训练步骤
    model_path = f"s3://{default_bucket}/{base_job_prefix}/model-artifacts"
    
    # 使用 PyTorch 容器进行 PaddleOCR 训练
    paddleocr_estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve(
            framework="pytorch",
            region=region,
            version="1.12.0",
            py_version="py38",
            instance_type=training_instance_type,
            accelerator_type="gpu"
        ),
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=model_path,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{base_job_prefix}/paddleocr-detection-training",
        entry_point="pid_train.py",
        source_dir=BASE_DIR,
        
        # 超参数
        hyperparameters={
            "epochs": epochs,
            "batch-size": batch_size,
            "learning-rate": learning_rate,
        },
        
        # 指标定义 - PaddleOCR 检测指标
        metric_definitions=[
            {"Name": "train:loss", "Regex": "loss: ([0-9\\.]+)"},
            {"Name": "validation:hmean", "Regex": "Validation hmean: ([0-9\\.]+)"},
            {"Name": "validation:precision", "Regex": "Validation precision: ([0-9\\.]+)"},
            {"Name": "validation:recall", "Regex": "Validation recall: ([0-9\\.]+)"},
        ]
    )
    
    step_train = TrainingStep(
        name="TrainPIDDetectionModel",
        estimator=paddleocr_estimator,
        inputs={
            "training": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                content_type="application/x-parquet"
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation_data"
                ].S3Output.S3Uri,
                content_type="application/x-parquet"
            )
        }
    )
    
    # 模型注册步骤
    step_register = RegisterModel(
        name="RegisterPIDDetectionModel",
        estimator=paddleocr_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[inference_instance_type],
        transform_instances=[inference_instance_type],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=None,  # 可以添加模型指标
    )
    
    # 条件步骤：只有当 hmean >= 0.6 时才注册模型
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=step_train.properties.FinalMetricDataList[1].Value,  # hmean
        right=0.6,
    )
    
    step_cond = ConditionStep(
        name="CheckModelPerformance",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )
    
    # 创建 Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            inference_instance_type,
            model_approval_status,
            input_data,
            epochs,
            batch_size,
            learning_rate,
        ],
        steps=[step_process, step_train, step_cond],
        sagemaker_session=sagemaker_session,
    )
    
    return pipeline