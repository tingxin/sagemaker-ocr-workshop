"""
Demo 7: Dataset-P&ID 数据集下载与预处理
用于 PaddleOCR 图纸识别演示的数据准备

数据集来源:
- 论文: Digitize-PID: Automatic Digitization of Piping and Instrumentation Diagrams
- Hugging Face: https://huggingface.co/datasets/hamzas/digitize-pid-yolo
- 包含 500 张合成 P&ID 工程图纸，带有符号和文字标注
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import sys


# ============ 配置 ============

DATASET_NAME = "hamzas/digitize-pid-yolo"
OUTPUT_DIR = "dataset_pid"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 符号类别映射（P&ID 常见符号）
SYMBOL_CLASSES = {
    0: "valve",
    1: "pump", 
    2: "tank",
    3: "heat_exchanger",
    4: "compressor",
    5: "filter",
    6: "instrument",
    7: "reducer",
    8: "flange",
    9: "text",
    # 更多类别根据实际数据集调整
}


# ============ 依赖安装 ============

def install_dependencies():
    """安装必要的依赖包"""
    packages = [
        "huggingface_hub",
        "datasets",
        "Pillow",
        "tqdm",
        "numpy"
    ]
    
    print("正在检查并安装依赖包...")
    for package in packages:
        try:
            __import__(package.replace("-", "_").split("[")[0])
        except ImportError:
            print(f"  安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    
    print("依赖包检查完成！\n")


# ============ 数据集下载 ============

def download_dataset_from_huggingface():
    """
    从 Hugging Face 下载 Dataset-P&ID 数据集
    """
    from huggingface_hub import snapshot_download
    
    print("=" * 60)
    print("从 Hugging Face 下载 Dataset-P&ID 数据集")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 下载数据集
        print(f"\n正在下载数据集: {DATASET_NAME}")
        print("这可能需要几分钟，请耐心等待...\n")
        
        local_dir = snapshot_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            local_dir=os.path.join(OUTPUT_DIR, "raw"),
            local_dir_use_symlinks=False
        )
        
        print(f"\n数据集已下载到: {local_dir}")
        return local_dir
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n尝试备用下载方式...")
        return download_dataset_alternative()


def download_dataset_alternative():
    """
    备用下载方式：使用 datasets 库
    """
    from datasets import load_dataset
    
    print("使用 datasets 库下载...")
    
    try:
        dataset = load_dataset(DATASET_NAME)
        
        # 保存到本地
        raw_dir = os.path.join(OUTPUT_DIR, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        dataset.save_to_disk(raw_dir)
        print(f"数据集已保存到: {raw_dir}")
        return raw_dir
        
    except Exception as e:
        print(f"备用下载也失败: {e}")
        print("\n请手动下载数据集:")
        print(f"  1. 访问 https://huggingface.co/datasets/{DATASET_NAME}")
        print(f"  2. 下载文件并解压到 {OUTPUT_DIR}/raw 目录")
        return None


# ============ 数据预处理 ============

def explore_dataset_structure(raw_dir: str):
    """
    探索数据集结构
    """
    print("\n" + "=" * 60)
    print("探索数据集结构")
    print("=" * 60)
    
    for root, dirs, files in os.walk(raw_dir):
        level = root.replace(raw_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        
        # 只显示前 10 个文件
        subindent = ' ' * 2 * (level + 1)
        for i, file in enumerate(files[:10]):
            print(f'{subindent}{file}')
        if len(files) > 10:
            print(f'{subindent}... 还有 {len(files) - 10} 个文件')


def convert_yolo_to_paddleocr(
    image_path: str,
    label_path: str,
    image_width: int,
    image_height: int
) -> List[Dict]:
    """
    将 YOLO 格式标注转换为 PaddleOCR 格式
    
    YOLO 格式: class_id x_center y_center width height (归一化)
    PaddleOCR 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence
    """
    annotations = []
    
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height
        
        # 计算四个角点坐标
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center - height / 2
        x3 = x_center + width / 2
        y3 = y_center + height / 2
        x4 = x_center - width / 2
        y4 = y_center + height / 2
        
        # 获取类别名称
        class_name = SYMBOL_CLASSES.get(class_id, f"symbol_{class_id}")
        
        annotations.append({
            "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "text": class_name,
            "class_id": class_id,
            "confidence": 1.0
        })
    
    return annotations


def process_dataset(raw_dir: str):
    """
    处理数据集，转换为 PaddleOCR 训练格式
    """
    from PIL import Image
    from tqdm import tqdm
    
    print("\n" + "=" * 60)
    print("处理数据集，转换为 PaddleOCR 格式")
    print("=" * 60)
    
    # 创建输出目录
    processed_dir = os.path.join(OUTPUT_DIR, "processed")
    images_dir = os.path.join(processed_dir, "images")
    labels_dir = os.path.join(processed_dir, "labels")
    
    for d in [processed_dir, images_dir, labels_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 查找图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(raw_dir).rglob(f'*{ext}'))
        image_files.extend(Path(raw_dir).rglob(f'*{ext.upper()}'))
    
    print(f"\n找到 {len(image_files)} 张图像")
    
    if len(image_files) == 0:
        print("未找到图像文件，请检查数据集结构")
        return None
    
    # 处理每张图像
    all_annotations = []
    
    for img_path in tqdm(image_files, desc="处理图像"):
        img_path = Path(img_path)
        
        # 查找对应的标注文件
        label_path = img_path.with_suffix('.txt')
        if not label_path.exists():
            # 尝试在 labels 目录查找
            label_path = img_path.parent.parent / 'labels' / img_path.with_suffix('.txt').name
        
        try:
            # 读取图像尺寸
            with Image.open(img_path) as img:
                width, height = img.size
            
            # 转换标注
            annotations = convert_yolo_to_paddleocr(
                str(img_path),
                str(label_path),
                width,
                height
            )
            
            # 复制图像到处理目录
            new_img_name = f"pid_{len(all_annotations):04d}{img_path.suffix}"
            new_img_path = os.path.join(images_dir, new_img_name)
            shutil.copy(img_path, new_img_path)
            
            # 保存标注
            all_annotations.append({
                "image": new_img_name,
                "image_path": new_img_path,
                "width": width,
                "height": height,
                "annotations": annotations
            })
            
        except Exception as e:
            print(f"\n处理 {img_path} 时出错: {e}")
            continue
    
    print(f"\n成功处理 {len(all_annotations)} 张图像")
    
    # 保存完整标注
    annotations_path = os.path.join(processed_dir, "annotations.json")
    with open(annotations_path, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"标注已保存到: {annotations_path}")
    
    return all_annotations


def split_dataset(annotations: List[Dict]):
    """
    划分训练集、验证集、测试集
    """
    print("\n" + "=" * 60)
    print("划分数据集")
    print("=" * 60)
    
    # 随机打乱
    random.seed(42)
    random.shuffle(annotations)
    
    total = len(annotations)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_data = annotations[:train_end]
    val_data = annotations[train_end:val_end]
    test_data = annotations[val_end:]
    
    print(f"训练集: {len(train_data)} 张 ({TRAIN_RATIO*100:.0f}%)")
    print(f"验证集: {len(val_data)} 张 ({VAL_RATIO*100:.0f}%)")
    print(f"测试集: {len(test_data)} 张 ({TEST_RATIO*100:.0f}%)")
    
    # 保存划分结果
    processed_dir = os.path.join(OUTPUT_DIR, "processed")
    
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    for split_name, split_data in splits.items():
        split_path = os.path.join(processed_dir, f"{split_name}.json")
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}.json 已保存")
    
    return splits


def generate_paddleocr_label_files(splits: Dict):
    """
    生成 PaddleOCR 格式的标签文件
    
    格式: image_path \t json_annotations
    """
    print("\n" + "=" * 60)
    print("生成 PaddleOCR 标签文件")
    print("=" * 60)
    
    processed_dir = os.path.join(OUTPUT_DIR, "processed")
    paddleocr_dir = os.path.join(OUTPUT_DIR, "paddleocr_format")
    os.makedirs(paddleocr_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        label_lines = []
        
        for item in split_data:
            image_path = item["image"]
            
            # PaddleOCR 检测标签格式
            det_annotations = []
            for ann in item["annotations"]:
                box = ann["box"]
                # 转换为整数坐标
                points = [[int(p[0]), int(p[1])] for p in box]
                det_annotations.append({
                    "transcription": ann["text"],
                    "points": points
                })
            
            if det_annotations:
                label_line = f"{image_path}\t{json.dumps(det_annotations, ensure_ascii=False)}"
                label_lines.append(label_line)
        
        # 保存标签文件
        label_file = os.path.join(paddleocr_dir, f"label_{split_name}.txt")
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(label_lines))
        
        print(f"  {label_file} ({len(label_lines)} 条记录)")
    
    # 生成文件列表
    for split_name in splits.keys():
        file_list = os.path.join(paddleocr_dir, f"file_list_{split_name}.txt")
        with open(file_list, 'w') as f:
            for item in splits[split_name]:
                f.write(f"images/{item['image']}\n")
        print(f"  {file_list}")
    
    return paddleocr_dir


def generate_statistics(annotations: List[Dict]):
    """
    生成数据集统计信息
    """
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    
    total_images = len(annotations)
    total_annotations = sum(len(item["annotations"]) for item in annotations)
    
    # 统计各类别数量
    class_counts = {}
    for item in annotations:
        for ann in item["annotations"]:
            class_name = ann["text"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n总图像数: {total_images}")
    print(f"总标注数: {total_annotations}")
    print(f"平均每张图像标注数: {total_annotations/total_images:.1f}")
    
    print(f"\n各类别统计:")
    print("-" * 40)
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count}")
    
    # 保存统计信息
    stats = {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "avg_annotations_per_image": total_annotations / total_images,
        "class_distribution": class_counts
    }
    
    stats_path = os.path.join(OUTPUT_DIR, "dataset_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n统计信息已保存到: {stats_path}")
    
    return stats


# ============ S3 上传（可选） ============

def upload_to_s3(local_dir: str, bucket: str, prefix: str):
    """
    上传处理后的数据集到 S3
    """
    import boto3
    
    print("\n" + "=" * 60)
    print("上传数据集到 S3")
    print("=" * 60)
    
    s3 = boto3.client('s3')
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{prefix}/{relative_path}"
            
            print(f"  上传: {relative_path}")
            s3.upload_file(local_path, bucket, s3_key)
    
    print(f"\n数据集已上传到: s3://{bucket}/{prefix}/")


# ============ 主流程 ============

def main():
    """
    主函数：下载并预处理 Dataset-P&ID 数据集
    """
    print("=" * 60)
    print("Dataset-P&ID 数据集下载与预处理工具")
    print("=" * 60)
    print(f"\n数据集: {DATASET_NAME}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"划分比例: 训练 {TRAIN_RATIO*100:.0f}% / 验证 {VAL_RATIO*100:.0f}% / 测试 {TEST_RATIO*100:.0f}%")
    
    # Step 1: 安装依赖
    print("\n[Step 1/6] 检查依赖...")
    install_dependencies()
    
    # Step 2: 下载数据集
    print("\n[Step 2/6] 下载数据集...")
    raw_dir = download_dataset_from_huggingface()
    
    if raw_dir is None:
        print("\n数据集下载失败，请手动下载后重试")
        return
    
    # Step 3: 探索数据集结构
    print("\n[Step 3/6] 探索数据集结构...")
    explore_dataset_structure(raw_dir)
    
    # Step 4: 处理数据集
    print("\n[Step 4/6] 处理数据集...")
    annotations = process_dataset(raw_dir)
    
    if annotations is None or len(annotations) == 0:
        print("\n数据处理失败")
        return
    
    # Step 5: 划分数据集
    print("\n[Step 5/6] 划分数据集...")
    splits = split_dataset(annotations)
    
    # Step 6: 生成 PaddleOCR 格式
    print("\n[Step 6/6] 生成 PaddleOCR 格式...")
    paddleocr_dir = generate_paddleocr_label_files(splits)
    
    # 生成统计信息
    generate_statistics(annotations)
    
    # 完成
    print("\n" + "=" * 60)
    print("数据集准备完成！")
    print("=" * 60)
    print(f"\n输出目录结构:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── raw/                    # 原始数据")
    print(f"  ├── processed/              # 处理后数据")
    print(f"  │   ├── images/             # 图像文件")
    print(f"  │   ├── annotations.json    # 完整标注")
    print(f"  │   ├── train.json          # 训练集")
    print(f"  │   ├── val.json            # 验证集")
    print(f"  │   └── test.json           # 测试集")
    print(f"  ├── paddleocr_format/       # PaddleOCR 格式")
    print(f"  │   ├── label_train.txt     # 训练标签")
    print(f"  │   ├── label_val.txt       # 验证标签")
    print(f"  │   └── label_test.txt      # 测试标签")
    print(f"  └── dataset_statistics.json # 统计信息")
    
    print("\n下一步:")
    print("  1. 检查 dataset_statistics.json 了解数据分布")
    print("  2. 使用 paddleocr_format/ 目录进行 PaddleOCR 训练")
    print("  3. 可选：运行 upload_to_s3() 上传到 S3")


# ============ 演示用快速数据生成 ============

def generate_demo_subset(num_samples: int = 50):
    """
    生成演示用的小规模数据子集
    用于快速验证 MLOps 流程
    """
    print("\n" + "=" * 60)
    print(f"生成演示数据子集 ({num_samples} 张)")
    print("=" * 60)
    
    processed_dir = os.path.join(OUTPUT_DIR, "processed")
    demo_dir = os.path.join(OUTPUT_DIR, "demo_subset")
    
    # 检查是否已处理
    annotations_path = os.path.join(processed_dir, "annotations.json")
    if not os.path.exists(annotations_path):
        print("请先运行 main() 下载并处理完整数据集")
        return
    
    # 加载标注
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # 随机选择子集
    random.seed(42)
    demo_samples = random.sample(annotations, min(num_samples, len(annotations)))
    
    # 创建演示目录
    demo_images_dir = os.path.join(demo_dir, "images")
    os.makedirs(demo_images_dir, exist_ok=True)
    
    # 复制图像和标注
    demo_annotations = []
    for item in demo_samples:
        src_path = item["image_path"]
        dst_path = os.path.join(demo_images_dir, item["image"])
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            demo_annotations.append(item)
    
    # 保存演示标注
    demo_annotations_path = os.path.join(demo_dir, "annotations.json")
    with open(demo_annotations_path, 'w', encoding='utf-8') as f:
        json.dump(demo_annotations, f, ensure_ascii=False, indent=2)
    
    # 划分演示数据
    train_end = int(len(demo_annotations) * 0.7)
    val_end = train_end + int(len(demo_annotations) * 0.15)
    
    demo_splits = {
        "train": demo_annotations[:train_end],
        "val": demo_annotations[train_end:val_end],
        "test": demo_annotations[val_end:]
    }
    
    for split_name, split_data in demo_splits.items():
        split_path = os.path.join(demo_dir, f"{split_name}.json")
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n演示数据子集已生成:")
    print(f"  目录: {demo_dir}")
    print(f"  训练集: {len(demo_splits['train'])} 张")
    print(f"  验证集: {len(demo_splits['val'])} 张")
    print(f"  测试集: {len(demo_splits['test'])} 张")
    
    return demo_dir


# ============ 入口 ============

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset-P&ID 数据集下载与预处理')
    parser.add_argument('--demo', action='store_true', help='仅生成演示子集')
    parser.add_argument('--demo-size', type=int, default=50, help='演示子集大小')
    parser.add_argument('--upload-s3', action='store_true', help='上传到 S3')
    parser.add_argument('--bucket', type=str, default='your-bucket', help='S3 bucket 名称')
    parser.add_argument('--prefix', type=str, default='dataset-pid', help='S3 前缀')
    
    args = parser.parse_args()
    
    # 运行主流程
    main()
    
    # 生成演示子集
    if args.demo:
        generate_demo_subset(args.demo_size)
    
    # 上传到 S3
    if args.upload_s3:
        upload_to_s3(
            os.path.join(OUTPUT_DIR, "processed"),
            args.bucket,
            args.prefix
        )
