"""
P&ID 数据预处理脚本
将 Ground Truth 标注数据转换为训练格式
"""

import os
import json
import argparse
import logging
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def process_pid_data(input_dir: str, train_output: str, val_output: str):
    """
    处理 P&ID 标注数据
    
    Args:
        input_dir: 输入数据目录
        train_output: 训练数据输出目录
        val_output: 验证数据输出目录
    """
    
    logger.info(f"处理输入数据: {input_dir}")
    
    # 创建输出目录
    Path(train_output).mkdir(parents=True, exist_ok=True)
    Path(val_output).mkdir(parents=True, exist_ok=True)
    
    # 查找标注文件
    paddleocr_dir = Path(input_dir) / "paddleocr_format"
    
    if not paddleocr_dir.exists():
        logger.error(f"未找到 paddleocr_format 目录: {paddleocr_dir}")
        return
    
    # 处理训练和验证数据
    for split in ['train', 'val']:
        label_file = paddleocr_dir / f"label_{split}.txt"
        
        if not label_file.exists():
            logger.warning(f"标签文件不存在: {label_file}")
            continue
        
        output_dir = train_output if split == 'train' else val_output
        
        # 读取标签文件
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        processed_data = []
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            image_name = parts[0]
            annotations_str = parts[1]
            
            try:
                annotations = json.loads(annotations_str)
                
                # 转换为训练格式
                for ann in annotations:
                    transcription = ann.get('transcription', '')
                    points = ann.get('points', [])
                    
                    if len(points) == 4:
                        # 计算边界框
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        processed_data.append({
                            'image': image_name,
                            'class': transcription,
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max,
                            'width': x_max - x_min,
                            'height': y_max - y_min
                        })
            
            except json.JSONDecodeError as e:
                logger.warning(f"解析标注失败: {line[:50]}... 错误: {e}")
                continue
        
        # 保存为 Parquet 格式
        if processed_data:
            df = pd.DataFrame(processed_data)
            output_file = Path(output_dir) / f"{split}_annotations.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"保存 {split} 数据: {len(processed_data)} 条 -> {output_file}")
        
        # 复制原始标签文件
        shutil.copy(label_file, Path(output_dir) / f"label_{split}.txt")
    
    logger.info("数据预处理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P&ID 数据预处理")
    parser.add_argument("--input-data", type=str, required=True, help="输入数据路径")
    
    args = parser.parse_args()
    
    # SageMaker 处理作业路径
    base_dir = "/opt/ml/processing"
    input_dir = f"{base_dir}/input"
    train_output = f"{base_dir}/train"
    val_output = f"{base_dir}/validation"
    
    logger.info("开始 P&ID 数据预处理")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"训练输出: {train_output}")
    logger.info(f"验证输出: {val_output}")
    
    process_pid_data(input_dir, train_output, val_output)