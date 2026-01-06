"""
Demo 5: 端到端推理演示
PaddleOCR 图纸识别完整流程
"""

import boto3
import json
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 配置
ENDPOINT_NAME = 'paddleocr-drawing-endpoint'
BUCKET_NAME = 'your-ocr-project-bucket'

# 初始化
runtime = boto3.client('sagemaker-runtime')
s3 = boto3.client('s3')


def invoke_ocr_endpoint(image_path):
    """
    调用 PaddleOCR 推理端点
    
    Args:
        image_path: 本地图像路径或 S3 路径
    
    Returns:
        OCR 识别结果
    """
    # 读取图像
    if image_path.startswith('s3://'):
        bucket = image_path.split('/')[2]
        key = '/'.join(image_path.split('/')[3:])
        response = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
    else:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    
    # 编码为 base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # 调用端点
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps({'image': image_base64})
    )
    
    # 解析结果
    result = json.loads(response['Body'].read().decode())
    return result


def visualize_ocr_result(image_path, ocr_result, output_path=None):
    """
    可视化 OCR 识别结果
    """
    # 读取图像
    if image_path.startswith('s3://'):
        bucket = image_path.split('/')[2]
        key = '/'.join(image_path.split('/')[3:])
        response = s3.get_object(Bucket=bucket, Key=key)
        image = Image.open(io.BytesIO(response['Body'].read()))
    else:
        image = Image.open(image_path)
    
    # 创建图形
    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.imshow(image)
    
    # 绘制检测框和文字
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, item in enumerate(ocr_result.get('predictions', [])):
        box = item['box']
        text = item['text']
        confidence = item['confidence']
        
        # 绘制边界框
        color = colors[i % len(colors)]
        polygon = patches.Polygon(
            box,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(polygon)
        
        # 添加文字标签
        ax.text(
            box[0][0], box[0][1] - 5,
            f'{text} ({confidence:.2f})',
            fontsize=8,
            color='white',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )
    
    ax.axis('off')
    plt.title(f'OCR 识别结果 - 共检测到 {len(ocr_result.get("predictions", []))} 个文本区域')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"结果已保存: {output_path}")
    
    plt.show()
    return fig


def extract_structured_data(ocr_result):
    """
    从 OCR 结果中提取结构化数据
    针对图纸场景的特定字段提取
    """
    structured_data = {
        'title_block': {},
        'dimensions': [],
        'notes': [],
        'symbols': [],
        'raw_text': []
    }
    
    for item in ocr_result.get('predictions', []):
        text = item['text']
        box = item['box']
        confidence = item['confidence']
        
        # 保存原始文本
        structured_data['raw_text'].append({
            'text': text,
            'position': box,
            'confidence': confidence
        })
        
        # 简单规则提取（实际场景需要更复杂的逻辑）
        
        # 尺寸标注（包含数字和单位）
        if any(unit in text.lower() for unit in ['mm', 'cm', 'm', 'inch', '°']):
            structured_data['dimensions'].append({
                'value': text,
                'position': box
            })
        
        # 标题栏字段
        if '图号' in text or '名称' in text or '材料' in text or '比例' in text:
            key = text.split(':')[0] if ':' in text else text.split('：')[0] if '：' in text else text
            value = text.split(':')[1] if ':' in text else text.split('：')[1] if '：' in text else ''
            structured_data['title_block'][key.strip()] = value.strip()
        
        # 技术要求/注释
        if '注' in text or '要求' in text or '说明' in text:
            structured_data['notes'].append(text)
    
    return structured_data


def batch_process_drawings(image_list, output_dir):
    """
    批量处理图纸
    """
    results = []
    
    for i, image_path in enumerate(image_list):
        print(f"处理中 [{i+1}/{len(image_list)}]: {image_path}")
        
        try:
            # OCR 识别
            ocr_result = invoke_ocr_endpoint(image_path)
            
            # 提取结构化数据
            structured_data = extract_structured_data(ocr_result)
            
            results.append({
                'image_path': image_path,
                'status': 'success',
                'ocr_result': ocr_result,
                'structured_data': structured_data
            })
            
        except Exception as e:
            results.append({
                'image_path': image_path,
                'status': 'error',
                'error': str(e)
            })
    
    # 保存结果
    output_file = f"{output_dir}/batch_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n批量处理完成，结果已保存: {output_file}")
    return results


def compare_model_versions(image_path, endpoint_v1, endpoint_v2):
    """
    对比不同模型版本的识别效果
    """
    print(f"测试图像: {image_path}")
    print("=" * 50)
    
    # V1 模型结果
    print("\n[V1 模型结果]")
    # result_v1 = invoke_ocr_endpoint_custom(image_path, endpoint_v1)
    result_v1 = {
        'predictions': [
            {'text': '齿轮零件图', 'confidence': 0.85},
            {'text': '材料: 45#钢', 'confidence': 0.78},
            {'text': 'φ50', 'confidence': 0.72}
        ]
    }
    for item in result_v1['predictions']:
        print(f"  - {item['text']} (置信度: {item['confidence']:.2f})")
    
    # V2 模型结果（增量训练后）
    print("\n[V2 模型结果 - 增量训练后]")
    # result_v2 = invoke_ocr_endpoint_custom(image_path, endpoint_v2)
    result_v2 = {
        'predictions': [
            {'text': '齿轮零件图', 'confidence': 0.95},
            {'text': '材料: 45#钢', 'confidence': 0.92},
            {'text': 'φ50±0.02', 'confidence': 0.89},
            {'text': '表面粗糙度 Ra1.6', 'confidence': 0.85}
        ]
    }
    for item in result_v2['predictions']:
        print(f"  - {item['text']} (置信度: {item['confidence']:.2f})")
    
    # 对比
    print("\n[对比分析]")
    print(f"  V1 检测数量: {len(result_v1['predictions'])}")
    print(f"  V2 检测数量: {len(result_v2['predictions'])}")
    
    avg_conf_v1 = sum(p['confidence'] for p in result_v1['predictions']) / len(result_v1['predictions'])
    avg_conf_v2 = sum(p['confidence'] for p in result_v2['predictions']) / len(result_v2['predictions'])
    
    print(f"  V1 平均置信度: {avg_conf_v1:.2f}")
    print(f"  V2 平均置信度: {avg_conf_v2:.2f}")
    print(f"  置信度提升: {((avg_conf_v2 - avg_conf_v1) / avg_conf_v1 * 100):+.1f}%")


# ============ 演示流程 ============

if __name__ == '__main__':
    print("=" * 60)
    print("Demo 5: 端到端推理演示")
    print("=" * 60)
    
    # Step 1: 单张图纸识别
    print("\n[Step 1] 单张图纸 OCR 识别...")
    test_image = "test_drawing.png"
    
    # 模拟 OCR 结果
    mock_ocr_result = {
        'predictions': [
            {
                'box': [[100, 50], [300, 50], [300, 80], [100, 80]],
                'text': '齿轮零件图',
                'confidence': 0.95
            },
            {
                'box': [[100, 100], [250, 100], [250, 130], [100, 130]],
                'text': '图号: DWG-2024-001',
                'confidence': 0.92
            },
            {
                'box': [[100, 150], [200, 150], [200, 180], [100, 180]],
                'text': '材料: 45#钢',
                'confidence': 0.89
            },
            {
                'box': [[300, 200], [350, 200], [350, 230], [300, 230]],
                'text': 'φ50±0.02',
                'confidence': 0.87
            },
            {
                'box': [[400, 250], [500, 250], [500, 280], [400, 280]],
                'text': 'Ra1.6',
                'confidence': 0.85
            }
        ]
    }
    
    print("\nOCR 识别结果:")
    print("-" * 40)
    for item in mock_ocr_result['predictions']:
        print(f"  文本: {item['text']}")
        print(f"  置信度: {item['confidence']:.2f}")
        print()
    
    # Step 2: 提取结构化数据
    print("\n[Step 2] 提取结构化数据...")
    structured = extract_structured_data(mock_ocr_result)
    
    print("\n结构化数据:")
    print("-" * 40)
    print(f"标题栏: {json.dumps(structured['title_block'], ensure_ascii=False, indent=2)}")
    print(f"尺寸标注: {structured['dimensions']}")
    print(f"注释: {structured['notes']}")
    
    # Step 3: 模型版本对比
    print("\n[Step 3] 模型版本对比...")
    compare_model_versions(
        test_image,
        'paddleocr-endpoint-v1',
        'paddleocr-endpoint-v2'
    )
    
    # Step 4: 输出 JSON 结果
    print("\n[Step 4] 完整 JSON 输出...")
    final_output = {
        'image': test_image,
        'ocr_result': mock_ocr_result,
        'structured_data': structured,
        'model_version': 'v2',
        'processing_time_ms': 156
    }
    print(json.dumps(final_output, ensure_ascii=False, indent=2))
    
    print("\n演示完成！")
