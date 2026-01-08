"""
Ground Truth Bounding Box 任务输出转换器
专门处理纯 bounding box 标注任务的输出
"""

import boto3
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random


class BoundingBoxConverter:
    """Ground Truth Bounding Box 输出转换器"""
    
    def __init__(self, region: str = "us-east-2"):
        self.s3 = boto3.client('s3', region_name=region)
    
    def download_output(self, bucket: str, job_name: str, local_dir: str) -> str:
        """下载 Ground Truth 输出文件"""
        # 尝试不同的输出路径格式
        possible_prefixes = [
            f"groundtruth/output/{job_name}",
            f"{job_name}",
            f"output/{job_name}"
        ]
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = None
        
        for prefix in possible_prefixes:
            try:
                response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                
                for obj in response.get('Contents', []):
                    key = obj['Key']
                    if 'output.manifest' in key:
                        local_path = local_dir / 'output.manifest'
                        self.s3.download_file(bucket, key, str(local_path))
                        manifest_path = str(local_path)
                        print(f"✅ 下载: {key} -> {local_path}")
                        return manifest_path
            except Exception as e:
                print(f"尝试前缀 {prefix} 失败: {e}")
                continue
        
        print("❌ 未找到输出 manifest 文件")
        return None

    def analyze_manifest_structure(self, manifest_path: str):
        """分析 manifest 文件结构"""
        print("\n🔍 分析 manifest 结构...")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                
                item = json.loads(line)
                print(f"\n📄 记录 {i+1}:")
                print(f"  Keys: {list(item.keys())}")
                
                # 查找标注数据
                for key, value in item.items():
                    if key != 'source-ref' and isinstance(value, dict):
                        print(f"  标注字段 '{key}': {list(value.keys())}")
                        if 'annotations' in value:
                            annotations = value['annotations']
                            if annotations:
                                print(f"    标注数量: {len(annotations)}")
                                print(f"    第一个标注: {annotations[0]}")
                
                if i >= 2:  # 只分析前3条记录
                    break

    def convert_bounding_box_to_detection_format(self, manifest_data: List[Dict]) -> List[Dict]:
        """将 bounding box 输出转换为检测格式"""
        detection_data = []
        
        for item in manifest_data:
            source_ref = item.get('source-ref', '')
            image_name = source_ref.split('/')[-1]
            
            # 查找标注数据字段
            annotation_field = None
            for key, value in item.items():
                if key != 'source-ref' and isinstance(value, dict) and 'annotations' in value:
                    annotation_field = key
                    break
            
            if not annotation_field:
                print(f"⚠️  {image_name}: 未找到标注数据")
                continue
            
            annotation_data = item[annotation_field]
            annotations = annotation_data.get('annotations', [])
            image_size = annotation_data.get('image_size', [{}])[0]
            
            img_width = image_size.get('width', 1200)
            img_height = image_size.get('height', 800)
            
            # 转换每个标注
            detections = []
            for ann in annotations:
                # 获取边界框坐标
                left = ann.get('left', 0)
                top = ann.get('top', 0)
                width = ann.get('width', 0)
                height = ann.get('height', 0)
                
                # 获取类别标签
                class_name = ann.get('class_id', 'unknown')
                
                # 转换为四点坐标 (PaddleOCR 格式)
                x1, y1 = int(left), int(top)
                x2, y2 = int(left + width), int(top)
                x3, y3 = int(left + width), int(top + height)
                x4, y4 = int(left), int(top + height)
                
                detections.append({
                    "transcription": class_name,  # 使用类别名作为 transcription
                    "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                    "class": class_name,
                    "bbox": [left, top, width, height]
                })
            
            if detections:
                detection_data.append({
                    "image": image_name,
                    "width": img_width,
                    "height": img_height,
                    "detections": detections
                })
        
        return detection_data

    def save_detection_format(self, detection_data: List[Dict], output_dir: str):
        """保存为检测格式"""
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
        
        # 保存每个分割
        for split_name, split_data in splits.items():
            # PaddleOCR 格式
            paddleocr_lines = []
            
            # YOLO 格式准备
            yolo_annotations = []
            
            for item in split_data:
                image_name = item['image']
                img_width = item['width']
                img_height = item['height']
                
                # PaddleOCR 格式
                ocr_anns = []
                yolo_anns = []
                
                for det in item['detections']:
                    # PaddleOCR 标注
                    ocr_anns.append({
                        "transcription": det['transcription'],
                        "points": det['points']
                    })
                    
                    # YOLO 格式标注
                    bbox = det['bbox']
                    x_center = (bbox[0] + bbox[2]/2) / img_width
                    y_center = (bbox[1] + bbox[3]/2) / img_height
                    norm_width = bbox[2] / img_width
                    norm_height = bbox[3] / img_height
                    
                    yolo_anns.append({
                        "class": det['class'],
                        "bbox_norm": [x_center, y_center, norm_width, norm_height]
                    })
                
                if ocr_anns:
                    paddleocr_lines.append(f"{image_name}\t{json.dumps(ocr_anns, ensure_ascii=False)}")
                
                if yolo_anns:
                    yolo_annotations.append({
                        "image": image_name,
                        "annotations": yolo_anns
                    })
            
            # 保存 PaddleOCR 格式
            paddleocr_file = output_dir / f"label_{split_name}.txt"
            with open(paddleocr_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(paddleocr_lines))
            
            # 保存 YOLO 格式
            yolo_file = output_dir / f"yolo_{split_name}.json"
            with open(yolo_file, 'w', encoding='utf-8') as f:
                json.dump(yolo_annotations, f, ensure_ascii=False, indent=2)
            
            print(f"  {split_name}: {len(paddleocr_lines)} 张图片")
        
        # 统计类别分布
        self.generate_class_statistics(detection_data, output_dir)
        
        return output_dir

    def generate_class_statistics(self, detection_data: List[Dict], output_dir: Path):
        """生成类别统计"""
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


def convert_bounding_box_output(
    bucket: str,
    job_name: str,
    output_dir: str,
    region: str = "us-east-2"
):
    """转换 bounding box 输出"""
    print("=" * 60)
    print("Ground Truth Bounding Box 输出转换")
    print("=" * 60)
    
    converter = BoundingBoxConverter(region)
    
    # 1. 下载输出
    print("\n[Step 1] 下载 Ground Truth 输出...")
    temp_dir = Path(output_dir) / "temp"
    manifest_path = converter.download_output(bucket, job_name, str(temp_dir))
    
    if not manifest_path:
        return None
    
    # 2. 分析结构
    converter.analyze_manifest_structure(manifest_path)
    
    # 3. 解析数据
    print("\n[Step 2] 解析标注数据...")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"解析完成: {len(manifest_data)} 条记录")
    
    # 4. 转换格式
    print("\n[Step 3] 转换为检测格式...")
    detection_data = converter.convert_bounding_box_to_detection_format(manifest_data)
    
    # 5. 保存文件
    print("\n[Step 4] 保存标签文件...")
    detection_dir = Path(output_dir) / "detection_format"
    converter.save_detection_format(detection_data, str(detection_dir))
    
    print("\n" + "=" * 60)
    print("✅ 转换完成!")
    print("=" * 60)
    print(f"\n输出目录: {detection_dir}")
    print(f"  - label_*.txt: PaddleOCR 格式")
    print(f"  - yolo_*.json: YOLO 格式")
    print(f"  - statistics.json: 数据统计")
    
    return detection_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Ground Truth Bounding Box 输出转换')
    parser.add_argument('--bucket', required=True, help='S3 bucket 名称')
    parser.add_argument('--job-name', required=True, help='标注工作名称')
    parser.add_argument('--output-dir', default='./bbox_output', help='输出目录')
    parser.add_argument('--region', default='us-east-2', help='AWS 区域')
    
    args = parser.parse_args()
    
    convert_bounding_box_output(
        bucket=args.bucket,
        job_name=args.job_name,
        output_dir=args.output_dir,
        region=args.region
    )