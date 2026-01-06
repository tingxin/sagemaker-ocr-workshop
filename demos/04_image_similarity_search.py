"""
Demo 4: 图形相似性检索
基于深度学习特征提取 + OpenSearch kNN 向量检索
"""

import boto3
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import sagemaker
from sagemaker.pytorch import PyTorchModel
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# 配置
BUCKET_NAME = 'your-ocr-project-bucket'
OPENSEARCH_ENDPOINT = 'your-opensearch-domain.us-east-1.es.amazonaws.com'
INDEX_NAME = 'drawing-vectors'
REGION = 'us-east-1'

# 初始化
session = sagemaker.Session()
s3 = boto3.client('s3')
credentials = boto3.Session().get_credentials()


# ============ 特征提取模型 ============

FEATURE_EXTRACTOR_CODE = '''
"""
inference.py - 图像特征提取推理脚本
使用 ResNet50 提取图像特征向量
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import base64
import numpy as np

# 全局模型
model = None
transform = None

def model_fn(model_dir):
    """加载预训练模型"""
    global model, transform
    
    # 使用 ResNet50 作为特征提取器
    model = models.resnet50(pretrained=True)
    # 移除最后的分类层，保留特征提取部分
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return model

def input_fn(request_body, request_content_type):
    """处理输入"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        image_data = base64.b64decode(data['image'])
    else:
        image_data = request_body
    
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_fn(input_data, model):
    """提取特征向量"""
    with torch.no_grad():
        features = model(input_data)
        # 展平为一维向量
        features = features.squeeze().numpy()
        # L2 归一化
        features = features / np.linalg.norm(features)
    return features

def output_fn(prediction, response_content_type):
    """格式化输出"""
    return json.dumps({
        'vector': prediction.tolist(),
        'dimension': len(prediction)
    })
'''


def deploy_feature_extractor():
    """
    部署特征提取模型到 SageMaker Endpoint
    """
    from sagemaker.pytorch import PyTorchModel
    
    model = PyTorchModel(
        model_data=f's3://{BUCKET_NAME}/models/feature-extractor/model.tar.gz',
        role=sagemaker.get_execution_role(),
        framework_version='1.12',
        py_version='py38',
        entry_point='inference.py'
    )
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        endpoint_name='drawing-feature-extractor'
    )
    
    print("特征提取模型已部署")
    return predictor


# ============ OpenSearch 向量索引 ============

def get_opensearch_client():
    """
    获取 OpenSearch 客户端
    """
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        'es',
        session_token=credentials.token
    )
    
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    
    return client


def create_vector_index():
    """
    创建向量索引（支持 kNN 检索）
    """
    client = get_opensearch_client()
    
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "drawing_id": {"type": "keyword"},
                "drawing_name": {"type": "text"},
                "category": {"type": "keyword"},
                "s3_path": {"type": "keyword"},
                "feature_vector": {
                    "type": "knn_vector",
                    "dimension": 2048,  # ResNet50 特征维度
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "upload_date": {"type": "date"}
                    }
                }
            }
        }
    }
    
    # 创建索引
    if not client.indices.exists(INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=index_body)
        print(f"索引已创建: {INDEX_NAME}")
    else:
        print(f"索引已存在: {INDEX_NAME}")


def extract_features(image_path, endpoint_name='drawing-feature-extractor'):
    """
    调用 SageMaker Endpoint 提取图像特征
    """
    runtime = boto3.client('sagemaker-runtime')
    
    # 读取图像
    if image_path.startswith('s3://'):
        bucket = image_path.split('/')[2]
        key = '/'.join(image_path.split('/')[3:])
        response = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
    else:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    
    # 调用端点
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps({
            'image': base64.b64encode(image_bytes).decode('utf-8')
        })
    )
    
    result = json.loads(response['Body'].read().decode())
    return result['vector']


def index_drawing(drawing_id, drawing_name, s3_path, category='general', metadata=None):
    """
    将图纸索引到 OpenSearch
    """
    client = get_opensearch_client()
    
    # 提取特征向量
    feature_vector = extract_features(s3_path)
    
    # 构建文档
    document = {
        'drawing_id': drawing_id,
        'drawing_name': drawing_name,
        'category': category,
        's3_path': s3_path,
        'feature_vector': feature_vector,
        'metadata': metadata or {}
    }
    
    # 索引文档
    response = client.index(
        index=INDEX_NAME,
        id=drawing_id,
        body=document
    )
    
    print(f"图纸已索引: {drawing_id}")
    return response


def search_similar_drawings(query_image_path, top_k=5, category_filter=None):
    """
    搜索相似图纸
    """
    client = get_opensearch_client()
    
    # 提取查询图像特征
    query_vector = extract_features(query_image_path)
    
    # 构建 kNN 查询
    query_body = {
        "size": top_k,
        "query": {
            "knn": {
                "feature_vector": {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }
    
    # 添加类别过滤
    if category_filter:
        query_body["query"] = {
            "bool": {
                "must": [
                    {"knn": {"feature_vector": {"vector": query_vector, "k": top_k * 2}}}
                ],
                "filter": [
                    {"term": {"category": category_filter}}
                ]
            }
        }
    
    # 执行搜索
    response = client.search(index=INDEX_NAME, body=query_body)
    
    # 解析结果
    results = []
    for hit in response['hits']['hits']:
        results.append({
            'drawing_id': hit['_source']['drawing_id'],
            'drawing_name': hit['_source']['drawing_name'],
            's3_path': hit['_source']['s3_path'],
            'similarity_score': hit['_score'],
            'category': hit['_source'].get('category')
        })
    
    return results


def batch_index_drawings(drawings_list):
    """
    批量索引图纸
    """
    client = get_opensearch_client()
    
    bulk_body = []
    for drawing in drawings_list:
        # 提取特征
        feature_vector = extract_features(drawing['s3_path'])
        
        # 添加索引操作
        bulk_body.append({
            "index": {"_index": INDEX_NAME, "_id": drawing['drawing_id']}
        })
        bulk_body.append({
            'drawing_id': drawing['drawing_id'],
            'drawing_name': drawing['drawing_name'],
            'category': drawing.get('category', 'general'),
            's3_path': drawing['s3_path'],
            'feature_vector': feature_vector,
            'metadata': drawing.get('metadata', {})
        })
    
    # 批量索引
    response = client.bulk(body=bulk_body)
    
    print(f"批量索引完成: {len(drawings_list)} 张图纸")
    return response


# ============ Lambda API 网关 ============

LAMBDA_HANDLER_CODE = '''
"""
lambda_handler.py - 图形相似性检索 API
"""
import json
import boto3
import base64
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

def lambda_handler(event, context):
    """
    API Gateway 触发的 Lambda 函数
    """
    # 解析请求
    body = json.loads(event.get('body', '{}'))
    action = body.get('action', 'search')
    
    if action == 'search':
        # 相似性搜索
        image_base64 = body.get('image')
        top_k = body.get('top_k', 5)
        category = body.get('category')
        
        results = search_similar(image_base64, top_k, category)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'results': results,
                'count': len(results)
            })
        }
    
    elif action == 'index':
        # 索引新图纸
        drawing_id = body.get('drawing_id')
        drawing_name = body.get('drawing_name')
        s3_path = body.get('s3_path')
        
        index_drawing(drawing_id, drawing_name, s3_path)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Indexed successfully'})
        }
    
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid action'})
        }
'''


# ============ 演示流程 ============

if __name__ == '__main__':
    print("=" * 60)
    print("Demo 4: 图形相似性检索")
    print("=" * 60)
    
    # Step 1: 创建向量索引
    print("\n[Step 1] 创建 OpenSearch 向量索引...")
    # create_vector_index()
    
    # Step 2: 部署特征提取模型
    print("\n[Step 2] 部署特征提取模型...")
    # deploy_feature_extractor()
    
    # Step 3: 索引示例图纸
    print("\n[Step 3] 索引示例图纸...")
    sample_drawings = [
        {
            'drawing_id': 'DWG-001',
            'drawing_name': '齿轮零件图',
            's3_path': f's3://{BUCKET_NAME}/drawings/gear_001.png',
            'category': 'mechanical'
        },
        {
            'drawing_id': 'DWG-002',
            'drawing_name': '轴承装配图',
            's3_path': f's3://{BUCKET_NAME}/drawings/bearing_001.png',
            'category': 'mechanical'
        },
        {
            'drawing_id': 'DWG-003',
            'drawing_name': '电路板布局图',
            's3_path': f's3://{BUCKET_NAME}/drawings/pcb_001.png',
            'category': 'electrical'
        }
    ]
    # batch_index_drawings(sample_drawings)
    
    # Step 4: 相似性搜索演示
    print("\n[Step 4] 相似性搜索演示...")
    query_image = f's3://{BUCKET_NAME}/query/test_gear.png'
    # results = search_similar_drawings(query_image, top_k=5)
    
    print("\n搜索结果（模拟）:")
    print("-" * 40)
    mock_results = [
        {'drawing_id': 'DWG-001', 'drawing_name': '齿轮零件图', 'similarity_score': 0.95},
        {'drawing_id': 'DWG-005', 'drawing_name': '齿轮组装图', 'similarity_score': 0.87},
        {'drawing_id': 'DWG-012', 'drawing_name': '传动齿轮', 'similarity_score': 0.82},
    ]
    for i, r in enumerate(mock_results, 1):
        print(f"{i}. {r['drawing_name']} (相似度: {r['similarity_score']:.2f})")
    
    print("\n演示完成！")
