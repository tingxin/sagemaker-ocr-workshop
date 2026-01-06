"""
Demo 7b: 备用数据集下载方案
当 Dataset-P&ID 无法获取时，使用其他公开数据集进行演示

支持的数据集:
1. ICDAR 2015 - 场景文字检测（需注册）
2. FUNSD - 表单理解数据集（直接下载）
3. SROIE - 票据 OCR 数据集（直接下载）
4. 合成 P&ID 数据生成
"""

import os
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


# ============ 配置 ============

OUTPUT_DIR = "dataset_ocr"


# ============ FUNSD 数据集 ============

def download_funsd():
    """
    下载 FUNSD 数据集
    表单理解数据集，199 张扫描表单图像
    适合模拟图纸标题栏场景
    
    下载链接: https://guillaumejaume.github.io/FUNSD/
    """
    print("=" * 60)
    print("下载 FUNSD 数据集")
    print("=" * 60)
    
    funsd_dir = os.path.join(OUTPUT_DIR, "funsd")
    os.makedirs(funsd_dir, exist_ok=True)
    
    # FUNSD 官方下载链接
    url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    zip_path = os.path.join(funsd_dir, "dataset.zip")
    
    print(f"\n下载地址: {url}")
    print(f"保存路径: {zip_path}")
    
    try:
        # 使用 curl 下载
        subprocess.run([
            "curl", "-L", "-o", zip_path, url
        ], check=True)
        
        # 解压
        subprocess.run([
            "unzip", "-o", zip_path, "-d", funsd_dir
        ], check=True)
        
        print(f"\nFUNSD 数据集已下载到: {funsd_dir}")
        return funsd_dir
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n请手动下载:")
        print(f"  1. 访问 https://guillaumejaume.github.io/FUNSD/")
        print(f"  2. 下载 dataset.zip")
        print(f"  3. 解压到 {funsd_dir}")
        return None


def convert_funsd_to_paddleocr(funsd_dir: str):
    """
    将 FUNSD 格式转换为 PaddleOCR 格式
    """
    print("\n转换 FUNSD 为 PaddleOCR 格式...")
    
    output_dir = os.path.join(OUTPUT_DIR, "funsd_paddleocr")
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ["training_data", "testing_data"]:
        split_dir = os.path.join(funsd_dir, "dataset", split)
        if not os.path.exists(split_dir):
            continue
        
        annotations_dir = os.path.join(split_dir, "annotations")
        images_dir = os.path.join(split_dir, "images")
        
        label_lines = []
        
        for ann_file in Path(annotations_dir).glob("*.json"):
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            image_name = ann_file.stem + ".png"
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                continue
            
            # 提取文字标注
            ocr_annotations = []
            for item in data.get("form", []):
                box = item.get("box", [])
                text = item.get("text", "")
                
                if len(box) == 4 and text:
                    x1, y1, x2, y2 = box
                    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    ocr_annotations.append({
                        "transcription": text,
                        "points": points
                    })
            
            if ocr_annotations:
                label_line = f"{image_name}\t{json.dumps(ocr_annotations, ensure_ascii=False)}"
                label_lines.append(label_line)
        
        # 保存标签文件
        split_name = "train" if "training" in split else "test"
        label_file = os.path.join(output_dir, f"label_{split_name}.txt")
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(label_lines))
        
        print(f"  {label_file}: {len(label_lines)} 条记录")
    
    return output_dir


# ============ SROIE 数据集 ============

def download_sroie():
    """
    下载 SROIE 数据集
    票据 OCR 数据集，约 1000 张收据图像
    适合演示结构化信息提取
    """
    print("=" * 60)
    print("下载 SROIE 数据集")
    print("=" * 60)
    
    sroie_dir = os.path.join(OUTPUT_DIR, "sroie")
    os.makedirs(sroie_dir, exist_ok=True)
    
    # 从 GitHub 克隆
    repo_url = "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"
    
    print(f"\n克隆仓库: {repo_url}")
    
    try:
        subprocess.run([
            "git", "clone", "--depth", "1", repo_url, sroie_dir
        ], check=True)
        
        print(f"\nSROIE 数据集已下载到: {sroie_dir}")
        return sroie_dir
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n请手动下载:")
        print(f"  git clone {repo_url} {sroie_dir}")
        return None


def convert_sroie_to_paddleocr(sroie_dir: str):
    """
    将 SROIE 格式转换为 PaddleOCR 格式
    """
    print("\n转换 SROIE 为 PaddleOCR 格式...")
    
    output_dir = os.path.join(OUTPUT_DIR, "sroie_paddleocr")
    os.makedirs(output_dir, exist_ok=True)
    
    # SROIE 数据结构
    data_dir = os.path.join(sroie_dir, "data")
    
    for split in ["train", "test"]:
        images_dir = os.path.join(data_dir, split, "img")
        labels_dir = os.path.join(data_dir, split, "box")
        
        if not os.path.exists(images_dir):
            continue
        
        label_lines = []
        
        for img_file in Path(images_dir).glob("*.jpg"):
            label_file = os.path.join(labels_dir, img_file.stem + ".txt")
            
            if not os.path.exists(label_file):
                continue
            
            ocr_annotations = []
            
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 9:
                        # x1,y1,x2,y2,x3,y3,x4,y4,text
                        coords = [int(p) for p in parts[:8]]
                        text = ','.join(parts[8:])
                        
                        points = [
                            [coords[0], coords[1]],
                            [coords[2], coords[3]],
                            [coords[4], coords[5]],
                            [coords[6], coords[7]]
                        ]
                        
                        ocr_annotations.append({
                            "transcription": text,
                            "points": points
                        })
            
            if ocr_annotations:
                label_line = f"{img_file.name}\t{json.dumps(ocr_annotations, ensure_ascii=False)}"
                label_lines.append(label_line)
        
        # 保存标签文件
        label_file = os.path.join(output_dir, f"label_{split}.txt")
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(label_lines))
        
        print(f"  {label_file}: {len(label_lines)} 条记录")
    
    return output_dir


# ============ 合成 P&ID 数据生成 ============

def generate_synthetic_pid_data(num_images: int = 100):
    """
    生成合成的 P&ID 风格数据
    用于演示目的，包含常见的工程图纸元素
    """
    from PIL import Image, ImageDraw, ImageFont
    
    print("=" * 60)
    print(f"生成合成 P&ID 数据 ({num_images} 张)")
    print("=" * 60)
    
    output_dir = os.path.join(OUTPUT_DIR, "synthetic_pid")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # P&ID 常见文字元素
    pid_texts = {
        "equipment": ["P-101", "P-102", "V-201", "V-202", "E-301", "E-302", "T-401", "C-501"],
        "dimensions": ["DN50", "DN100", "DN150", "DN200", "2\"", "4\"", "6\"", "8\""],
        "materials": ["CS", "SS304", "SS316", "PTFE", "PVC", "HDPE"],
        "tags": ["FIC-101", "TIC-201", "PIC-301", "LIC-401", "FT-101", "TT-201"],
        "notes": ["NOTE 1", "NOTE 2", "SEE DWG", "TYP.", "REF."],
        "values": ["100°C", "150°C", "10 bar", "25 bar", "50 m³/h", "100 m³/h"]
    }
    
    all_annotations = []
    
    for i in range(num_images):
        # 创建白色背景图像
        width, height = 1200, 800
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
            font_large = font
        
        annotations = []
        
        # 绘制标题栏
        title_box = [50, 50, 400, 100]
        draw.rectangle(title_box, outline='black', width=2)
        title_text = f"P&ID-{i+1:03d}"
        draw.text((60, 60), title_text, fill='black', font=font_large)
        annotations.append({
            "box": [[50, 50], [400, 50], [400, 100], [50, 100]],
            "text": title_text,
            "type": "title"
        })
        
        # 随机添加设备标签
        num_elements = random.randint(5, 15)
        used_positions = []
        
        for j in range(num_elements):
            # 随机选择文字类型
            text_type = random.choice(list(pid_texts.keys()))
            text = random.choice(pid_texts[text_type])
            
            # 随机位置（避免重叠）
            for _ in range(10):
                x = random.randint(100, width - 200)
                y = random.randint(150, height - 100)
                
                # 检查是否与已有位置重叠
                overlap = False
                for px, py in used_positions:
                    if abs(x - px) < 100 and abs(y - py) < 50:
                        overlap = True
                        break
                
                if not overlap:
                    used_positions.append((x, y))
                    break
            
            # 绘制文字
            draw.text((x, y), text, fill='black', font=font)
            
            # 计算边界框
            bbox = draw.textbbox((x, y), text, font=font)
            x1, y1, x2, y2 = bbox
            
            annotations.append({
                "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "text": text,
                "type": text_type
            })
        
        # 绘制一些线条（模拟管道）
        for _ in range(random.randint(3, 8)):
            x1 = random.randint(50, width - 50)
            y1 = random.randint(150, height - 50)
            
            if random.random() > 0.5:
                # 水平线
                x2 = x1 + random.randint(100, 300)
                y2 = y1
            else:
                # 垂直线
                x2 = x1
                y2 = y1 + random.randint(100, 200)
            
            draw.line([(x1, y1), (x2, y2)], fill='black', width=2)
        
        # 保存图像
        img_name = f"pid_{i+1:04d}.png"
        img_path = os.path.join(images_dir, img_name)
        img.save(img_path)
        
        all_annotations.append({
            "image": img_name,
            "width": width,
            "height": height,
            "annotations": annotations
        })
    
    # 保存标注
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    # 生成 PaddleOCR 格式
    paddleocr_dir = os.path.join(output_dir, "paddleocr_format")
    os.makedirs(paddleocr_dir, exist_ok=True)
    
    # 划分数据集
    random.shuffle(all_annotations)
    train_end = int(len(all_annotations) * 0.8)
    val_end = train_end + int(len(all_annotations) * 0.1)
    
    splits = {
        "train": all_annotations[:train_end],
        "val": all_annotations[train_end:val_end],
        "test": all_annotations[val_end:]
    }
    
    for split_name, split_data in splits.items():
        label_lines = []
        for item in split_data:
            ocr_anns = []
            for ann in item["annotations"]:
                points = [[int(p[0]), int(p[1])] for p in ann["box"]]
                ocr_anns.append({
                    "transcription": ann["text"],
                    "points": points
                })
            if ocr_anns:
                label_lines.append(f"{item['image']}\t{json.dumps(ocr_anns, ensure_ascii=False)}")
        
        label_file = os.path.join(paddleocr_dir, f"label_{split_name}.txt")
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(label_lines))
        
        print(f"  {split_name}: {len(label_lines)} 张图像")
    
    print(f"\n合成数据已生成到: {output_dir}")
    print(f"  图像目录: {images_dir}")
    print(f"  PaddleOCR 格式: {paddleocr_dir}")
    
    return output_dir


# ============ ICDAR 2015 下载指南 ============

def print_icdar2015_guide():
    """
    打印 ICDAR 2015 数据集下载指南
    """
    print("=" * 60)
    print("ICDAR 2015 数据集下载指南")
    print("=" * 60)
    print("""
ICDAR 2015 是场景文字检测的标准数据集，需要注册后下载。

下载步骤:
1. 访问 https://rrc.cvc.uab.es/
2. 注册账号并登录
3. 进入 "Challenges" -> "ICDAR 2015"
4. 下载以下文件:
   - ch4_training_images.zip (训练图像)
   - ch4_training_localization_transcription_gt.zip (训练标注)
   - ch4_test_images.zip (测试图像)

数据集特点:
- 1000 张训练图像 + 500 张测试图像
- 场景文字（街景、招牌等）
- 四点标注格式

下载后解压到: {output_dir}/icdar2015/
""".format(output_dir=OUTPUT_DIR))


# ============ 主函数 ============

def main():
    """
    主函数：提供多种数据集下载选项
    """
    print("=" * 60)
    print("OCR 演示数据集下载工具")
    print("=" * 60)
    print("""
可用数据集:
  1. FUNSD      - 表单理解数据集 (199 张，直接下载)
  2. SROIE      - 票据 OCR 数据集 (~1000 张，GitHub)
  3. Synthetic  - 合成 P&ID 数据 (自定义数量)
  4. ICDAR 2015 - 场景文字数据集 (需注册)
  
推荐组合:
  - 快速演示: Synthetic (100 张)
  - 完整演示: FUNSD + Synthetic
  - 生产级别: ICDAR 2015 + FUNSD + Synthetic
""")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成合成数据（最简单的方式）
    print("\n[自动生成] 合成 P&ID 数据...")
    generate_synthetic_pid_data(100)
    
    # 下载 FUNSD
    print("\n[可选] 下载 FUNSD 数据集...")
    user_input = input("是否下载 FUNSD? (y/n): ").strip().lower()
    if user_input == 'y':
        funsd_dir = download_funsd()
        if funsd_dir:
            convert_funsd_to_paddleocr(funsd_dir)
    
    # 下载 SROIE
    print("\n[可选] 下载 SROIE 数据集...")
    user_input = input("是否下载 SROIE? (y/n): ").strip().lower()
    if user_input == 'y':
        sroie_dir = download_sroie()
        if sroie_dir:
            convert_sroie_to_paddleocr(sroie_dir)
    
    # ICDAR 2015 指南
    print("\n[参考] ICDAR 2015 下载指南...")
    print_icdar2015_guide()
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print(f"\n输出目录: {OUTPUT_DIR}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR 演示数据集下载工具')
    parser.add_argument('--synthetic', type=int, default=100, help='合成数据数量')
    parser.add_argument('--funsd', action='store_true', help='下载 FUNSD')
    parser.add_argument('--sroie', action='store_true', help='下载 SROIE')
    parser.add_argument('--all', action='store_true', help='下载所有数据集')
    
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成合成数据
    if args.synthetic > 0:
        generate_synthetic_pid_data(args.synthetic)
    
    # 下载 FUNSD
    if args.funsd or args.all:
        funsd_dir = download_funsd()
        if funsd_dir:
            convert_funsd_to_paddleocr(funsd_dir)
    
    # 下载 SROIE
    if args.sroie or args.all:
        sroie_dir = download_sroie()
        if sroie_dir:
            convert_sroie_to_paddleocr(sroie_dir)
    
    # 如果没有指定任何参数，运行交互式主函数
    if not (args.funsd or args.sroie or args.all):
        if args.synthetic == 100:  # 默认值，说明用户没有指定
            main()
