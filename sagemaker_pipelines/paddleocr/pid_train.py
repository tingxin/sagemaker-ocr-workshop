"""
P&ID PaddleOCR 检测模型训练脚本
在 SageMaker 训练作业中运行
"""

import os
import json
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def install_paddleocr_dependencies():
    """安装 PaddleOCR 训练依赖"""
    
    logger.info("安装 PaddleOCR 训练依赖...")
    
    # 安装 PaddlePaddle GPU 版本
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "paddlepaddle-gpu==2.5.2", 
        "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
    ])
    
    # 安装 PaddleOCR
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "paddleocr==2.7.3",
        "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
    ])
    
    # 安装其他依赖
    packages = [
        "opencv-python-headless",
        "Pillow",
        "PyYAML", 
        "tqdm",
        "visualdl",
        "lmdb",
        "imgaug"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def prepare_paddleocr_data(train_dir: str, val_dir: str, output_dir: str):
    """
    准备 PaddleOCR 检测训练数据
    
    Args:
        train_dir: 训练数据目录
        val_dir: 验证数据目录
        output_dir: 输出目录
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 PaddleOCR 目录结构
    (output_path / "train_data").mkdir(exist_ok=True)
    (output_path / "val_data").mkdir(exist_ok=True)
    
    # 查找 paddleocr_format 目录
    paddleocr_dir = None
    for root, dirs, files in os.walk(train_dir):
        if 'paddleocr_format' in dirs:
            paddleocr_dir = Path(root) / 'paddleocr_format'
            break
    
    if not paddleocr_dir or not paddleocr_dir.exists():
        logger.error(f"未找到 paddleocr_format 目录在 {train_dir}")
        # 尝试直接使用训练目录
        paddleocr_dir = Path(train_dir)
    
    logger.info(f"使用数据目录: {paddleocr_dir}")
    
    # 处理训练和验证数据
    for split in ['train', 'val']:
        label_file = paddleocr_dir / f"label_{split}.txt"
        if not label_file.exists():
            logger.warning(f"标签文件不存在: {label_file}")
            continue
        
        output_label_file = output_path / f"det_gt_{split}.txt"
        
        # 直接复制标签文件，因为格式已经是 PaddleOCR 兼容的
        logger.info(f"复制标签文件: {label_file} -> {output_label_file}")
        
        with open(label_file, 'r', encoding='utf-8') as f_in, \
             open(output_label_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                
                image_name = parts[0]
                annotations_str = parts[1]
                
                try:
                    annotations = json.loads(annotations_str)
                    
                    # 转换为 PaddleOCR 检测格式
                    det_annotations = []
                    for ann in annotations:
                        points = ann.get('points', [])
                        transcription = ann.get('transcription', '')
                        
                        if len(points) == 4:
                            det_annotations.append({
                                "transcription": transcription,
                                "points": points
                            })
                    
                    if det_annotations:
                        # PaddleOCR 检测标签格式：image_path \t json_annotations
                        f_out.write(f"{image_name}\t{json.dumps(det_annotations, ensure_ascii=False)}\n")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"解析标注失败: {e}")
                    continue
        
        logger.info(f"转换完成: {output_label_file}")
    
    return str(output_path)


def create_paddleocr_config(data_dir: str, model_dir: str, epochs: int, batch_size: int, learning_rate: float):
    """
    创建 PaddleOCR 检测训练配置
    
    Args:
        data_dir: 数据目录
        model_dir: 模型输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    
    config = {
        "Global": {
            "debug": False,
            "use_gpu": True,
            "epoch_num": epochs,
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": model_dir,
            "save_epoch_step": max(10, epochs // 5),
            "eval_batch_step": [0, 1000],
            "cal_metric_during_train": True,
            "pretrained_model": None,
            "checkpoints": None,
            "save_inference_dir": None,
            "use_visualdl": False,
            "infer_img": None,
            "save_res_path": f"{model_dir}/det_results.txt"
        },
        
        "Architecture": {
            "model_type": "det",
            "algorithm": "DB",
            "Transform": None,
            "Backbone": {
                "name": "MobileNetV3",
                "scale": 0.5,
                "model_name": "large",
                "disable_se": False
            },
            "Neck": {
                "name": "DBFPN",
                "out_channels": 256
            },
            "Head": {
                "name": "DBHead",
                "k": 50
            }
        },
        
        "Loss": {
            "name": "DBLoss",
            "balance_loss": True,
            "main_loss_type": "DiceLoss",
            "alpha": 5,
            "beta": 10,
            "ohem_ratio": 3
        },
        
        "Optimizer": {
            "name": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "name": "Cosine",
                "learning_rate": learning_rate,
                "warmup_epoch": 2
            },
            "regularizer": {
                "name": "L2",
                "factor": 5e-05
            }
        },
        
        "PostProcess": {
            "name": "DBPostProcess",
            "thresh": 0.3,
            "box_thresh": 0.6,
            "max_candidates": 1000,
            "unclip_ratio": 1.5
        },
        
        "Metric": {
            "name": "DetMetric",
            "main_indicator": "hmean"
        },
        
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": data_dir,
                "label_file_list": ["det_gt_train.txt"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"DetLabelEncode": None},
                    {"IaaAugment": {
                        "augmenter_args": [
                            {"type": "Fliplr", "args": {"p": 0.5}},
                            {"type": "Affine", "args": {"rotate": [-10, 10]}},
                            {"type": "Resize", "args": {"size": [0.5, 3]}}
                        ]
                    }},
                    {"EastRandomCropData": {
                        "size": [960, 960],
                        "max_tries": 50,
                        "keep_ratio": True
                    }},
                    {"MakeBorderMap": {
                        "shrink_ratio": 0.4,
                        "thresh_min": 0.3,
                        "thresh_max": 0.7
                    }},
                    {"MakeShrinkMap": {
                        "shrink_ratio": 0.4,
                        "min_text_size": 8
                    }},
                    {"NormalizeImage": {
                        "scale": "1./255.",
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "order": "hwc"
                    }},
                    {"ToCHWImage": None},
                    {"KeepKeys": {
                        "keep_keys": ["image", "threshold_map", "threshold_mask", "shrink_map", "shrink_mask"]
                    }}
                ]
            },
            "loader": {
                "shuffle": True,
                "drop_last": False,
                "batch_size_per_card": batch_size,
                "num_workers": 4
            }
        },
        
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": data_dir,
                "label_file_list": ["det_gt_val.txt"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"DetLabelEncode": None},
                    {"DetResizeForTest": None},
                    {"NormalizeImage": {
                        "scale": "1./255.",
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "order": "hwc"
                    }},
                    {"ToCHWImage": None},
                    {"KeepKeys": {
                        "keep_keys": ["image", "shape", "polys", "ignore_tags"]
                    }}
                ]
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": 1,
                "num_workers": 2
            }
        }
    }
    
    # 保存配置文件
    config_path = Path(data_dir) / "det_config.yml"
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    logger.info(f"PaddleOCR 配置已保存: {config_path}")
    return str(config_path)


def train_paddleocr_model(config_path: str, model_dir: str):
    """
    训练 PaddleOCR 检测模型
    
    Args:
        config_path: 配置文件路径
        model_dir: 模型输出目录
    """
    
    logger.info("开始训练 PaddleOCR 检测模型...")
    
    # 克隆 PaddleOCR 仓库
    if not os.path.exists("PaddleOCR"):
        subprocess.check_call([
            "git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git"
        ])
    
    os.chdir("PaddleOCR")
    
    # 安装 PaddleOCR 依赖
    if os.path.exists("requirements.txt"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
    
    # 开始训练
    train_cmd = [
        sys.executable, "tools/train.py",
        "-c", config_path,
        "-o", f"Global.save_model_dir={model_dir}"
    ]
    
    try:
        result = subprocess.run(train_cmd, capture_output=True, text=True, check=True)
        logger.info("训练输出:")
        logger.info(result.stdout)
        
        # 解析训练指标
        if "best metric" in result.stdout:
            import re
            hmean_match = re.search(r'hmean:([\d\.]+)', result.stdout)
            precision_match = re.search(r'precision:([\d\.]+)', result.stdout)
            recall_match = re.search(r'recall:([\d\.]+)', result.stdout)
            
            if hmean_match:
                logger.info(f"Validation hmean: {hmean_match.group(1)}")
            if precision_match:
                logger.info(f"Validation precision: {precision_match.group(1)}")
            if recall_match:
                logger.info(f"Validation recall: {recall_match.group(1)}")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"训练失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        raise
    
    logger.info("PaddleOCR 训练完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PaddleOCR 检测模型训练")
    
    # SageMaker 环境变量
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', './validation'))
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PaddleOCR 检测模型训练开始")
    logger.info("=" * 60)
    logger.info(f"训练数据: {args.train}")
    logger.info(f"验证数据: {args.validation}")
    logger.info(f"模型输出: {args.model_dir}")
    logger.info(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # 安装依赖
    logger.info("安装 PaddleOCR 依赖...")
    install_paddleocr_dependencies()
    
    # 准备训练数据
    logger.info("准备 PaddleOCR 训练数据...")
    data_dir = prepare_paddleocr_data(args.train, args.validation, "./paddleocr_data")
    
    # 创建配置文件
    logger.info("创建 PaddleOCR 配置...")
    config_path = create_paddleocr_config(
        data_dir, 
        args.model_dir, 
        args.epochs, 
        args.batch_size, 
        args.learning_rate
    )
    
    # 训练模型
    logger.info("开始训练 PaddleOCR 模型...")
    train_paddleocr_model(config_path, args.model_dir)
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()