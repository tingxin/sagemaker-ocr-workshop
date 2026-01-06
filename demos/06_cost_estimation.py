"""
Demo 6: 成本估算工具
帮助客户估算 MLOps 平台运行成本
"""

# ============ AWS 服务定价参考（us-east-1，2024年参考价格） ============

PRICING = {
    # SageMaker Ground Truth 标注
    'ground_truth': {
        'per_object': 0.08,  # 每个标注对象（使用 Amazon Mechanical Turk）
        'per_object_private': 0.00,  # 私有团队（只收取 SageMaker 费用）
    },
    
    # SageMaker 训练实例（按小时）
    'training_instances': {
        'ml.m5.large': 0.115,
        'ml.m5.xlarge': 0.23,
        'ml.m5.2xlarge': 0.461,
        'ml.p3.2xlarge': 3.825,  # GPU
        'ml.p3.8xlarge': 14.688,
        'ml.g4dn.xlarge': 0.736,
        'ml.g4dn.2xlarge': 1.053,
    },
    
    # SageMaker 推理实例（按小时）
    'inference_instances': {
        'ml.t2.medium': 0.065,
        'ml.m5.large': 0.134,
        'ml.m5.xlarge': 0.269,
        'ml.m5.2xlarge': 0.538,
        'ml.g4dn.xlarge': 0.7364,
    },
    
    # SageMaker Processing（按小时）
    'processing_instances': {
        'ml.m5.large': 0.134,
        'ml.m5.xlarge': 0.269,
        'ml.m5.2xlarge': 0.538,
    },
    
    # S3 存储（每 GB/月）
    's3': {
        'standard': 0.023,
        'intelligent_tiering': 0.0125,
    },
    
    # OpenSearch（按小时）
    'opensearch': {
        't3.small.search': 0.036,
        't3.medium.search': 0.073,
        'm5.large.search': 0.142,
        'r5.large.search': 0.186,
    },
    
    # Lambda（每百万请求）
    'lambda': {
        'requests': 0.20,
        'duration_per_gb_second': 0.0000166667,
    },
    
    # API Gateway（每百万请求）
    'api_gateway': {
        'rest_api': 3.50,
    }
}


def estimate_labeling_cost(num_images, labels_per_image=5, use_private_team=True):
    """
    估算数据标注成本
    """
    if use_private_team:
        # 私有团队：只需要 SageMaker 处理费用
        cost = num_images * 0.01  # 估算处理费用
    else:
        # 使用 Mechanical Turk
        cost = num_images * labels_per_image * PRICING['ground_truth']['per_object']
    
    return {
        'service': 'Ground Truth',
        'num_images': num_images,
        'labels_per_image': labels_per_image,
        'use_private_team': use_private_team,
        'estimated_cost': round(cost, 2)
    }


def estimate_training_cost(
    instance_type='ml.p3.2xlarge',
    training_hours=10,
    num_training_jobs=4
):
    """
    估算模型训练成本
    """
    hourly_rate = PRICING['training_instances'].get(instance_type, 3.825)
    cost = hourly_rate * training_hours * num_training_jobs
    
    return {
        'service': 'SageMaker Training',
        'instance_type': instance_type,
        'hourly_rate': hourly_rate,
        'training_hours': training_hours,
        'num_jobs': num_training_jobs,
        'estimated_cost': round(cost, 2)
    }


def estimate_inference_cost(
    instance_type='ml.m5.xlarge',
    hours_per_day=24,
    days_per_month=30,
    num_instances=1
):
    """
    估算推理服务成本
    """
    hourly_rate = PRICING['inference_instances'].get(instance_type, 0.269)
    monthly_hours = hours_per_day * days_per_month
    cost = hourly_rate * monthly_hours * num_instances
    
    return {
        'service': 'SageMaker Inference',
        'instance_type': instance_type,
        'hourly_rate': hourly_rate,
        'monthly_hours': monthly_hours,
        'num_instances': num_instances,
        'estimated_cost': round(cost, 2)
    }


def estimate_storage_cost(data_size_gb=100, model_size_gb=10):
    """
    估算存储成本
    """
    total_gb = data_size_gb + model_size_gb
    cost = total_gb * PRICING['s3']['standard']
    
    return {
        'service': 'S3 Storage',
        'data_size_gb': data_size_gb,
        'model_size_gb': model_size_gb,
        'total_gb': total_gb,
        'estimated_cost': round(cost, 2)
    }


def estimate_opensearch_cost(
    instance_type='t3.small.search',
    num_instances=2,
    hours_per_month=720
):
    """
    估算 OpenSearch 成本（用于图形相似性检索）
    """
    hourly_rate = PRICING['opensearch'].get(instance_type, 0.036)
    cost = hourly_rate * hours_per_month * num_instances
    
    return {
        'service': 'OpenSearch',
        'instance_type': instance_type,
        'num_instances': num_instances,
        'estimated_cost': round(cost, 2)
    }


def estimate_total_monthly_cost(config):
    """
    估算总月度成本
    
    Args:
        config: 配置字典，包含各项参数
    """
    costs = []
    
    # 标注成本（如果有新数据）
    if config.get('labeling'):
        costs.append(estimate_labeling_cost(**config['labeling']))
    
    # 训练成本
    if config.get('training'):
        costs.append(estimate_training_cost(**config['training']))
    
    # 推理成本
    if config.get('inference'):
        costs.append(estimate_inference_cost(**config['inference']))
    
    # 存储成本
    if config.get('storage'):
        costs.append(estimate_storage_cost(**config['storage']))
    
    # OpenSearch 成本
    if config.get('opensearch'):
        costs.append(estimate_opensearch_cost(**config['opensearch']))
    
    # 汇总
    total = sum(c['estimated_cost'] for c in costs)
    
    return {
        'breakdown': costs,
        'total_monthly_cost': round(total, 2)
    }


def print_cost_report(cost_result):
    """
    打印成本报告
    """
    print("\n" + "=" * 60)
    print("月度成本估算报告")
    print("=" * 60)
    
    print(f"\n{'服务':<25} {'配置':<30} {'成本 (USD)':<15}")
    print("-" * 70)
    
    for item in cost_result['breakdown']:
        service = item['service']
        
        if service == 'Ground Truth':
            config = f"{item['num_images']} 张图像"
        elif service == 'SageMaker Training':
            config = f"{item['instance_type']} x {item['training_hours']}h x {item['num_jobs']}次"
        elif service == 'SageMaker Inference':
            config = f"{item['instance_type']} x {item['num_instances']}台 x 24/7"
        elif service == 'S3 Storage':
            config = f"{item['total_gb']} GB"
        elif service == 'OpenSearch':
            config = f"{item['instance_type']} x {item['num_instances']}台"
        else:
            config = "-"
        
        print(f"{service:<25} {config:<30} ${item['estimated_cost']:<15.2f}")
    
    print("-" * 70)
    print(f"{'总计':<55} ${cost_result['total_monthly_cost']:<15.2f}")
    print("=" * 60)


def generate_cost_scenarios():
    """
    生成不同规模的成本场景
    """
    scenarios = {
        'POC（小规模验证）': {
            'labeling': {'num_images': 1000, 'use_private_team': True},
            'training': {'instance_type': 'ml.g4dn.xlarge', 'training_hours': 5, 'num_training_jobs': 2},
            'inference': {'instance_type': 'ml.m5.large', 'hours_per_day': 8, 'days_per_month': 22},
            'storage': {'data_size_gb': 20, 'model_size_gb': 5},
        },
        '开发环境': {
            'labeling': {'num_images': 5000, 'use_private_team': True},
            'training': {'instance_type': 'ml.p3.2xlarge', 'training_hours': 10, 'num_training_jobs': 4},
            'inference': {'instance_type': 'ml.m5.xlarge', 'hours_per_day': 12, 'days_per_month': 30},
            'storage': {'data_size_gb': 50, 'model_size_gb': 10},
            'opensearch': {'instance_type': 't3.small.search', 'num_instances': 2},
        },
        '生产环境': {
            'labeling': {'num_images': 10000, 'use_private_team': True},
            'training': {'instance_type': 'ml.p3.2xlarge', 'training_hours': 20, 'num_training_jobs': 8},
            'inference': {'instance_type': 'ml.m5.xlarge', 'hours_per_day': 24, 'days_per_month': 30, 'num_instances': 2},
            'storage': {'data_size_gb': 200, 'model_size_gb': 20},
            'opensearch': {'instance_type': 't3.medium.search', 'num_instances': 3},
        }
    }
    
    print("\n" + "=" * 70)
    print("不同规模场景成本对比")
    print("=" * 70)
    
    results = {}
    for scenario_name, config in scenarios.items():
        result = estimate_total_monthly_cost(config)
        results[scenario_name] = result
        
        print(f"\n【{scenario_name}】")
        print(f"  月度总成本: ${result['total_monthly_cost']:.2f}")
    
    # 对比表格
    print("\n" + "-" * 70)
    print(f"{'场景':<20} {'标注':<12} {'训练':<12} {'推理':<12} {'存储':<12} {'总计':<12}")
    print("-" * 70)
    
    for scenario_name, result in results.items():
        breakdown = {item['service']: item['estimated_cost'] for item in result['breakdown']}
        print(f"{scenario_name:<20} "
              f"${breakdown.get('Ground Truth', 0):<11.0f} "
              f"${breakdown.get('SageMaker Training', 0):<11.0f} "
              f"${breakdown.get('SageMaker Inference', 0):<11.0f} "
              f"${breakdown.get('S3 Storage', 0):<11.0f} "
              f"${result['total_monthly_cost']:<11.0f}")
    
    return results


# ============ 演示 ============

if __name__ == '__main__':
    print("=" * 60)
    print("Demo 6: 成本估算工具")
    print("=" * 60)
    
    # 场景 1: 客户典型配置
    print("\n[场景 1] 客户典型配置估算")
    
    customer_config = {
        'labeling': {
            'num_images': 10000,
            'labels_per_image': 5,
            'use_private_team': True
        },
        'training': {
            'instance_type': 'ml.p3.2xlarge',
            'training_hours': 10,
            'num_training_jobs': 4  # 每月 4 次增量训练
        },
        'inference': {
            'instance_type': 'ml.m5.xlarge',
            'hours_per_day': 24,
            'days_per_month': 30,
            'num_instances': 1
        },
        'storage': {
            'data_size_gb': 100,
            'model_size_gb': 10
        },
        'opensearch': {
            'instance_type': 't3.small.search',
            'num_instances': 2
        }
    }
    
    result = estimate_total_monthly_cost(customer_config)
    print_cost_report(result)
    
    # 场景 2: 不同规模对比
    print("\n[场景 2] 不同规模场景对比")
    generate_cost_scenarios()
    
    # 成本优化建议
    print("\n" + "=" * 60)
    print("成本优化建议")
    print("=" * 60)
    print("""
1. 训练优化
   - 使用 Spot 实例可节省 60-90% 训练成本
   - 增量训练比全量训练节省 50%+ 时间和成本
   
2. 推理优化
   - 非高峰时段使用自动扩缩容
   - 考虑使用 Serverless Inference（按请求计费）
   
3. 存储优化
   - 使用 S3 Intelligent-Tiering 自动优化存储类别
   - 定期清理过期的模型版本和训练数据
   
4. 标注优化
   - 使用私有标注团队避免 MTurk 费用
   - 利用主动学习减少标注数据量
""")
    
    print("\n演示完成！")
