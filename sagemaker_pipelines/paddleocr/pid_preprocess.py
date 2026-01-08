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
    
    # 处理训练数据
    train_label_file = paddleocr_dir / "label_train.txt"
    val_label_file = paddleocr_dir / "label_val.txt"
    
    if not train_label_file.exists():
        logger.error(f"未找到训练标签文件: {train_label_file}")
        return
    
    # 读取训练数据
    with open(train_label_file, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    
    logger.info(f"训练数据: {len(train_lines)} 张图片")
    
    # 如果没有验证数据，从训练数据中分割
    if not val_label_file.exists() or os.path.getsize(val_label_file) == 0:
        logger.info("未找到验证数据，从训练数据中分割 20% 作为验证集")
        
        import random
        random.shuffle(train_lines)
        
        # 80% 训练，20% 验证
        split_idx = int(len(train_lines) * 0.8)
        actual_train_lines = train_lines[:split_idx]
        val_lines = train_lines[split_idx:]
        
        logger.info(f"分割后 - 训练: {len(actual_train_lines)}, 验证: {len(val_lines)}")
    else:
        with open(val_label_file, 'r', encoding='utf-8') as f:
            val_lines = f.readlines()
        actual_train_lines = train_lines
        logger.info(f"使用现有验证数据: {len(val_lines)} 张图片")
    
    # 处理训练和验证数据
    for split, lines, output_dir in [
        ('train', actual_train_lines, train_output), 
        ('val', val_lines, val_output)
    ]:
        
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
        
        # 保存标签文件
        label_output_file = Path(output_dir) / f"label_{split}.txt"
        with open(label_output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        logger.info(f"保存 {split} 标签文件: {label_output_file}")
    
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