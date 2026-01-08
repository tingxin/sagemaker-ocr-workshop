"""
设置 S3 CORS 配置以支持 Ground Truth 标注
"""

import boto3
import json
from pathlib import Path


def setup_s3_cors(bucket_name: str):
    """为 S3 bucket 设置 CORS 配置"""
    
    s3_client = boto3.client('s3')
    
    # Ground Truth 需要的 CORS 配置
    cors_configuration = {
        'CORSRules': [
            {
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET', 'HEAD'],
                'AllowedOrigins': ['*'],
                'ExposeHeaders': [
                    'ETag',
                    'x-amz-meta-custom-header'
                ],
                'MaxAgeSeconds': 3000
            }
        ]
    }
    
    try:
        # 应用 CORS 配置
        s3_client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration=cors_configuration
        )
        
        print(f"✅ S3 CORS 配置已成功应用到 bucket: {bucket_name}")
        
        # 验证配置
        response = s3_client.get_bucket_cors(Bucket=bucket_name)
        print(f"📋 当前 CORS 配置:")
        print(json.dumps(response['CORSRules'], indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"❌ 设置 CORS 配置失败: {e}")
        return False


def check_s3_cors(bucket_name: str):
    """检查 S3 bucket 的 CORS 配置"""
    
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.get_bucket_cors(Bucket=bucket_name)
        print(f"✅ Bucket {bucket_name} 已有 CORS 配置:")
        print(json.dumps(response['CORSRules'], indent=2, ensure_ascii=False))
        return True
        
    except s3_client.exceptions.NoSuchCORSConfiguration:
        print(f"❌ Bucket {bucket_name} 没有 CORS 配置")
        return False
        
    except Exception as e:
        print(f"❌ 检查 CORS 配置失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    # 加载环境变量
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / '.env'
        load_dotenv(env_path)
        import os
        bucket_name = os.getenv('S3_BUCKET')
    except:
        bucket_name = None
    
    parser = argparse.ArgumentParser(description='设置 S3 CORS 配置')
    parser.add_argument('--bucket', type=str, default=bucket_name, 
                        help='S3 bucket 名称')
    parser.add_argument('--check', action='store_true', 
                        help='只检查当前配置，不修改')
    
    args = parser.parse_args()
    
    if not args.bucket:
        print("❌ 请指定 bucket 名称: --bucket your-bucket-name")
        return
    
    print("=" * 60)
    print("S3 CORS 配置工具 - Ground Truth 支持")
    print("=" * 60)
    print(f"Bucket: {args.bucket}")
    
    if args.check:
        print("\n🔍 检查当前 CORS 配置...")
        check_s3_cors(args.bucket)
    else:
        print("\n🔧 设置 CORS 配置...")
        if setup_s3_cors(args.bucket):
            print("\n✅ 配置完成！现在可以正常使用 Ground Truth 标注了")
        else:
            print("\n❌ 配置失败，请检查 AWS 权限")


if __name__ == '__main__':
    main()