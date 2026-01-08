"""
专门转换你的 P&ID 标注输出
根据实际的 manifest 格式进行转换
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import random


def convert_pid_manifest_to_training_format(manifest_path: str, output_dir: str):
    """转换 P&ID manifest 为训练格式"""
    
    print("=" * 60)
    print("转换 P&ID 标注数据为训练格式")
    print("=" * 60)
    
    # 读取 manifest 文件
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"读取到 {len(manifest_data)} 条标注记录")
    
    # 分析类别映射
    all_class_maps = {}
    for item in manifest_data:
        metadata = item.get('pid-label-job-metadata', {})
        class_map = metadata.get('class-map', {})
        all_class_maps.update(class_map)
    
    print(f"类别映射: {all_class_maps}")
    
    # 转换数据
    detection_data = []
    
    for item in manifest_data:
        source_ref = item.get('source-ref', '')
        image_name = source_ref.split('/')[-1]
        
        # 获取标注数据
        annotation_data = item.get('pid-label-job', {})
        annotations = annotation_data.get('annotations', [])
        image_size = annotation_data.get('image_size', [{}])[0]
        
        img_width = image_size.get('width', 7168)
        img_height = image_size.get('height', 4562)
        
        # 获取类别映射
        metadata = item.get('pid-label-job-metadata', {})
        class_map = metadata.get('class-map', all_class_maps)
        
        # 转换每个标注
        detections = []
        for ann in annotations:
            class_id = ann.get('class_id')
            left = ann.get('left', 0)
            top = ann.get('top', 0)
            width = ann.get('width', 0)
            height = ann.get('height', 0)
            
            # 获取类别名称
            class_name = class_map.get(str(class_id), f"class_{class_id}")
            
            # 转换为四点坐标 (PaddleOCR 格式)
            x1, y1 = int(left), int(top)
            x2, y2 = int(left + width), int(top)
            x3, y3 = int(left + width), int(top + height)
            x4, y4 = int(left), int(top + height)
            
            detections.append({
                "transcription": class_name,
                "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "class": class_name,
                "class_id": class_id,
                "bbox": [left, top, width, height]
            })
        
        if detections:
            detection_data.append({
                "image": image_name,
                "width": img_width,
                "height": img_height,
                "detections": detections
            })
    
    print(f"转换完成: {len(detection_data)} 张图片")
    
    # 保存转换后的数据
    save_training_formats(detection_data, output_dir, all_class_maps)
    
    return detection_data


def save_training_formats(detection_data: List[Dict], output_dir: str, class_map: Dict):
    """保存多种训练格式"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 随机打乱数据
    random.seed(42)
    random.shuffle(detection_data)
    
    # 划分数据集
    total = len(detection_data)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    splits = {
        'train': detection_data[:train_end],
        'val': detection_data[train_end:val_end],
        'test': detection_data[val_end:]
    }
    
    print(f"\n数据集划分:")
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} 张图片")
    
    # 1. 保存 PaddleOCR 格式
    paddleocr_dir = output_dir / "paddleocr_format"
    paddleocr_dir.mkdir(exist_ok=True)
    
    for split_name, split_data in splits.items():
        paddleocr_lines = []
        
        for item in split_data:
            image_name = item['image']
            
            # PaddleOCR 标注格式
            ocr_anns = []
            for det in item['detections']:
                ocr_anns.append({
                    "transcription": det['transcription'],
                    "points": det['points']
                })
            
            if ocr_anns:
                line = f"{image_name}\t{json.dumps(ocr_anns, ensure_ascii=False)}"
                paddleocr_lines.append(line)
        
        # 保存 PaddleOCR 格式文件
        paddleocr_file = paddleocr_dir / f"label_{split_name}.txt"
        with open(paddleocr_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(paddleocr_lines))
        
        print(f"  PaddleOCR {split_name}: {len(paddleocr_lines)} 条 -> {paddleocr_file}")
    
    # 2. 保存 YOLO 格式
    yolo_dir = output_dir / "yolo_format"
    yolo_dir.mkdir(exist_ok=True)
    
    # 创建类别名称列表
    unique_classes = set()
    for item in detection_data:
        for det in item['detections']:
            unique_classes.add(det['class'])
    
    class_names = sorted(list(unique_classes))
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    # 保存 YOLO 数据配置
    yolo_config = {
        'path': str(yolo_dir.absolute()),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(yolo_dir / 'data.yaml', 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(yolo_config, f, allow_unicode=True)
    
    for split_name, split_data in splits.items():
        split_dir = yolo_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        for item in split_data:
            image_name = item['image']
            img_width = item['width']
            img_height = item['height']
            
            # YOLO 标注格式
            yolo_lines = []
            for det in item['detections']:
                bbox = det['bbox']
                class_id = class_to_id[det['class']]
                
                # 转换为 YOLO 格式 (归一化)
                x_center = (bbox[0] + bbox[2]/2) / img_width
                y_center = (bbox[1] + bbox[3]/2) / img_height
                norm_width = bbox[2] / img_width
                norm_height = bbox[3] / img_height
                
                yolo_lines.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")
            
            # 保存 YOLO 标注文件
            if yolo_lines:
                label_file = split_dir / f"{Path(image_name).stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
    
    print(f"  YOLO 格式已保存到: {yolo_dir}")
    
    # 3. 生成统计信息
    generate_statistics(detection_data, output_dir, class_map)


def generate_statistics(detection_data: List[Dict], output_dir: Path, class_map: Dict):
    """生成数据统计"""
    
    class_counts = {}
    total_detections = 0
    
    for item in detection_data:
        for det in item['detections']:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_detections += 1
    
    stats = {
        "total_images": len(detection_data),
        "total_detections": total_detections,
        "avg_detections_per_image": total_detections / len(detection_data),
        "class_mapping": class_map,
        "class_distribution": class_counts
    }
    
    # 保存统计信息
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 数据统计:")
    print(f"  总图片数: {stats['total_images']}")
    print(f"  总检测数: {stats['total_detections']}")
    print(f"  平均每图: {stats['avg_detections_per_image']:.1f}")
    print(f"\n类别分布:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count}")
    
    print(f"\n统计信息已保存到: {stats_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='转换 P&ID 标注输出')
    parser.add_argument('--manifest', required=True, help='manifest 文件路径')
    parser.add_argument('--output-dir', default='./pid_training_data', help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.manifest):
        print(f"❌ 文件不存在: {args.manifest}")
        return
    
    # 转换数据
    detection_data = convert_pid_manifest_to_training_format(args.manifest, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 转换完成!")
    print("=" * 60)
    print(f"\n输出目录: {args.output_dir}")
    print(f"  📁 paddleocr_format/")
    print(f"     ├── label_train.txt")
    print(f"     ├── label_val.txt")
    print(f"     └── label_test.txt")
    print(f"  📁 yolo_format/")
    print(f"     ├── data.yaml")
    print(f"     ├── train/ (标注文件)")
    print(f"     ├── val/ (标注文件)")
    print(f"     └── test/ (标注文件)")
    print(f"  📄 statistics.json")
    
    print(f"\n🚀 下一步:")
    print(f"  1. 训练检测模型:")
    print(f"     python training/train_detection_model.py --data {args.output_dir}/yolo_format/data.yaml")
    print(f"  2. 或使用 PaddleOCR 格式进行其他训练")


if __name__ == '__main__':
    main()