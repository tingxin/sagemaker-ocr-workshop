"""
Ground Truth 标注结果转换为 PaddleOCR 格式
"""

import boto3
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple


class GroundTruthConverter:
    """Ground Truth 输出转 PaddleOCR 格式转换器"""
    
    def __init__(self, region: str = "us-west-2"):
        self.s3 = boto3.client('s3', region_name=region)
    
    def download_output(self, bucket: str, job_name: str, local_dir: str) -> str:
        """下载 Ground Truth 输出文件"""
        output_prefix = f"groundtruth/output/{job_name}"
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # 列出输出文件
        response = self.s3.list_objects_v2(Bucket=bucket, Prefix=output_prefix)
        
        manifest_path = None
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('output.manifest'):
                local_path = local_dir / 'output.manifest'
                self.s3.download_file(bucket, key, str(local_path))
                manifest_path = str(local_path)
                print(f"下载: {key} -> {local_path}")
        
        return manifest_path

    def parse_manifest(self, manifest_path: str) -> List[Dict]:
        """解析 Ground Truth 输出 manifest"""
        results = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                results.append(item)
        
        print(f"解析完成: {len(results)} 条记录")
        return results
    
    def convert_to_paddleocr(
        self, 
        manifest_data: List[Dict],
        label_attribute: str = "ocr-annotations",
        image_width: int = None,
        image_height: int = None
    ) -> List[str]:
        """
        转换为 PaddleOCR 格式
        
        PaddleOCR 格式: image_path\t[{"transcription": "text", "points": [[x1,y1],...]}]
        """
        paddleocr_lines = []
        
        for item in manifest_data:
            # 获取图像路径
            source_ref = item.get('source-ref', '')
            image_name = source_ref.split('/')[-1]
            
            # 获取标注数据
            annotation_data = item.get(label_attribute, {})
            
            # 获取 bounding boxes
            bboxes = []
            if isinstance(annotation_data, dict):
                bboxes = annotation_data.get('annotations', [])
                img_size = annotation_data.get('image_size', [{}])[0]
                image_width = img_size.get('width', image_width)
                image_height = img_size.get('height', image_height)
            
            # 获取文字内容
            transcriptions = item.get('transcriptions', {})
            if isinstance(transcriptions, str):
                transcriptions = json.loads(transcriptions)
            
            # 转换每个标注
            ocr_annotations = []
            for idx, bbox in enumerate(bboxes):
                # 获取坐标
                left = bbox.get('left', 0)
                top = bbox.get('top', 0)
                width = bbox.get('width', 0)
                height = bbox.get('height', 0)
                
                # 转换为四点坐标
                x1, y1 = int(left), int(top)
                x2, y2 = int(left + width), int(top)
                x3, y3 = int(left + width), int(top + height)
                x4, y4 = int(left), int(top + height)
                
                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                
                # 获取文字内容
                text = transcriptions.get(str(idx), '')
                if not text:
                    text = transcriptions.get(idx, '')
                if not text:
                    text = bbox.get('label', 'text')
                
                ocr_annotations.append({
                    "transcription": text,
                    "points": points
                })
            
            if ocr_annotations:
                line = f"{image_name}\t{json.dumps(ocr_annotations, ensure_ascii=False)}"
                paddleocr_lines.append(line)
        
        return paddleocr_lines

    def save_paddleocr_format(
        self,
        paddleocr_lines: List[str],
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, str]:
        """保存为 PaddleOCR 格式文件"""
        import random
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 随机打乱
        random.seed(42)
        random.shuffle(paddleocr_lines)
        
        # 划分数据集
        total = len(paddleocr_lines)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': paddleocr_lines[:train_end],
            'val': paddleocr_lines[train_end:val_end],
            'test': paddleocr_lines[val_end:]
        }
        
        output_files = {}
        for split_name, lines in splits.items():
            output_path = output_dir / f"label_{split_name}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            output_files[split_name] = str(output_path)
            print(f"  {split_name}: {len(lines)} 条 -> {output_path}")
        
        return output_files


def convert_groundtruth_output(
    bucket: str,
    job_name: str,
    output_dir: str,
    region: str = "us-west-2"
):
    """完整转换流程"""
    print("=" * 60)
    print("Ground Truth 输出转换为 PaddleOCR 格式")
    print("=" * 60)
    
    converter = GroundTruthConverter(region)
    
    # 1. 下载输出
    print("\n[Step 1] 下载 Ground Truth 输出...")
    temp_dir = Path(output_dir) / "temp"
    manifest_path = converter.download_output(bucket, job_name, str(temp_dir))
    
    if not manifest_path:
        print("错误: 未找到输出 manifest 文件")
        return
    
    # 2. 解析 manifest
    print("\n[Step 2] 解析标注数据...")
    manifest_data = converter.parse_manifest(manifest_path)
    
    # 3. 转换格式
    print("\n[Step 3] 转换为 PaddleOCR 格式...")
    paddleocr_lines = converter.convert_to_paddleocr(manifest_data)
    
    # 4. 保存文件
    print("\n[Step 4] 保存标签文件...")
    paddleocr_dir = Path(output_dir) / "paddleocr_format"
    output_files = converter.save_paddleocr_format(paddleocr_lines, str(paddleocr_dir))
    
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
    print(f"\n输出目录: {paddleocr_dir}")
    
    return output_files


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Ground Truth 输出转换')
    parser.add_argument('--bucket', required=True, help='S3 bucket 名称')
    parser.add_argument('--job-name', required=True, help='标注工作名称')
    parser.add_argument('--output-dir', default='./gt_output', help='输出目录')
    parser.add_argument('--region', default='us-west-2', help='AWS 区域')
    
    args = parser.parse_args()
    
    convert_groundtruth_output(
        bucket=args.bucket,
        job_name=args.job_name,
        output_dir=args.output_dir,
        region=args.region
    )
