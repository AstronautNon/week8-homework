#库导入区
import pyarrow.parquet as pq
from functions import (generate_data_quality_report,
                       clean_data,
                       feature_engineering,
                       analyze_hour_distribution,
                       analyze_workday_weekend_distribution,
                       TOP10_PULocationID,
                       TOP10_DOLocationID,
                       hour_vs_fare,
                       hour_vs_tips,
                       passenger_vs_fare,
                       long_trip_profit_detail,
                       long_vs_normal_profit,
                       predict_demand_interactive,
                       predict_demand_interactive_rf,
                       show_model_visualization)


# ==================== 命令注册表（可扩展设计）====================
# 格式: {关键词列表: {'func': 函数对象, 'name': 显示名称, 'desc': 功能描述}}
COMMAND_REGISTRY = [
    # 时间分布分析
    {
        'keywords': ['小时分布', '时间分布', 'hourly', 'hour distribution', '出行需求时间'],
        'func': analyze_hour_distribution,
        'name': '小时分布分析',
        'desc': '分析出行需求时间分布',
        'category': '时间分布'
    },
    {
        'keywords': ['工作日', '周末', 'workday', 'weekend'],
        'func': analyze_workday_weekend_distribution,
        'name': '工作日周末分析',
        'desc': '分析工作日与周末出行需求分布',
        'category': '时间分布'
    },

    # 区域热度分析
    {
        'keywords': ['上客点', '上车点', 'pickup', 'pu location', '热度上客'],
        'func': TOP10_PULocationID,
        'name': 'TOP10上客点分析',
        'desc': '分析订单量最多的10个上客点',
        'category': '区域热度'
    },
    {
        'keywords': ['下客点', '下车点', 'dropoff', 'do location', '热度下客'],
        'func': TOP10_DOLocationID,
        'name': 'TOP10下客点分析',
        'desc': '分析订单量最多的10个下客点',
        'category': '区域热度'
    },

    # 车费影响因素
    {
        'keywords': ['时段车费', '小时车费', 'hour fare', '时间车费'],
        'func': hour_vs_fare,
        'name': '时段车费分析',
        'desc': '分析不同时段对车费的影响',
        'category': '车费分析'
    },
    {
        'keywords': ['时段小费', '小时小费', 'hour tip', 'tips', '时间小费'],
        'func': hour_vs_tips,
        'name': '时段小费分析',
        'desc': '分析不同时段对小费的影响',
        'category': '车费分析'
    },
    {
        'keywords': ['乘客车费', '人数车费', 'passenger fare', '乘客数量'],
        'func': passenger_vs_fare,
        'name': '乘客数量车费分析',
        'desc': '分析乘客数量对车费的影响',
        'category': '车费分析'
    },

    # 长途旅行分析
    {
        'keywords': ['长途详情', '长途收益', 'long trip detail', '每趟长途'],
        'func': long_trip_profit_detail,
        'name': '长途旅行详情分析',
        'desc': '分析每趟长途旅行的单位距离收益',
        'category': '长途分析'
    },
    {
        'keywords': ['长途对比', '长短途', 'long vs normal', '长途非长途'],
        'func': long_vs_normal_profit,
        'name': '长途vs非长途对比',
        'desc': '对比长途与非长途旅行的单位距离收益',
        'category': '长途分析'
    },

    # 模型预测
    {
        'keywords': ['神经网络预测', 'nn预测', 'neural network', 'nn predict', '深度学习预测'],
        'func': predict_demand_interactive,
        'name': '神经网络交互式预测',
        'desc': '使用训练好的神经网络进行订单量预测',
        'category': '模型预测',
        'need_data': False
    },
    {
        'keywords': ['随机森林预测', 'rf预测', 'random forest', 'rf predict', '森林预测'],
        'func': predict_demand_interactive_rf,
        'name': '随机森林交互式预测',
        'desc': '使用训练好的随机森林进行订单量预测',
        'category': '模型预测',
        'need_data': False
    },

    # 可视化展示
    {
        'keywords': ['所有图片', '全部可视化', 'all images', 'all visualization', '全部图片'],
        'func': lambda: show_model_visualization('all'),
        'name': '显示所有可视化图片',
        'desc': '展示所有模型训练相关的可视化结果',
        'category': '可视化',
        'need_data': False
    },
    {
        'keywords': ['训练曲线', 'loss曲线', 'training curve', 'loss', '损失曲线'],
        'func': lambda: show_model_visualization('loss'),
        'name': '神经网络训练曲线',
        'desc': '显示神经网络的Loss、MAE、RMSE训练曲线',
        'category': '可视化',
        'need_data': False
    },
    {
        'keywords': ['预测对比', 'prediction', '预测vs实际', '预测实际对比'],
        'func': lambda: show_model_visualization('prediction'),
        'name': '预测vs实际值对比图',
        'desc': '显示神经网络预测值与实际值的对比',
        'category': '可视化',
        'need_data': False
    },
    {
        'keywords': ['模型对比', '比较', 'comparison', 'neural vs random', '两种模型'],
        'func': lambda: show_model_visualization('comparison'),
        'name': '模型预测结果对比',
        'desc': '对比神经网络和随机森林的预测结果',
        'category': '可视化',
        'need_data': False
    },
    {
        'keywords': ['mae', 'rmse', '指标对比', 'metrics', '性能指标'],
        'func': lambda: show_model_visualization('mae_rmse'),
        'name': 'MAE/RMSE指标对比',
        'desc': '对比两种模型的MAE和RMSE指标',
        'category': '可视化',
        'need_data': False
    },
    {
        'keywords': ['误差分布', 'error distribution', '误差', 'error'],
        'func': lambda: show_model_visualization('error'),
        'name': '误差分布对比图',
        'desc': '显示两种模型的误差分布直方图',
        'category': '可视化',
        'need_data': False
    },
    {
        'keywords': ['散点图', 'scatter', '散点', 'scatter plot'],
        'func': lambda: show_model_visualization('scatter'),
        'name': '预测vs实际散点图',
        'desc': '显示两种模型预测值与实际值的散点图',
        'category': '可视化',
        'need_data': False
    },
]


def show_help():
    """显示帮助信息"""
    print("\n" + "=" * 60)
    print("可用命令列表")
    print("=" * 60)

    # 按类别分组显示
    categories = {}
    for cmd in COMMAND_REGISTRY:
        category = cmd.get('category', '其他')
        if category not in categories:
            categories[category] = []
        categories[category].append(cmd)

    for category, commands in categories.items():
        print(f"\n【{category}】")
        for cmd in commands:
            # 提取前两个关键词作为示例
            keywords = cmd['keywords'][:2]
            keyword_str = ' / '.join(keywords)
            print(f"  {keyword_str:<30} - {cmd['desc']}")

    print("\n【其他】")
    print(f"  {'help / 帮助':<30} - 显示此帮助信息")
    print(f"  {'quit / exit / q / 退出':<30} - 退出程序")
    print("=" * 60)


def match_command(query):
    """
    根据用户输入匹配对应的命令

    参数:
    query: str, 用户输入的自然语言查询

    返回:
    dict or None: 匹配到的命令配置，如果没有匹配则返回None
    """
    query_lower = query.lower().strip()

    # 遍历所有注册的命令
    for cmd in COMMAND_REGISTRY:
        # 检查是否包含任意一个关键词
        if any(keyword in query_lower for keyword in cmd['keywords']):
            return cmd

    return None


def auto_preprocessing(trips, original_trips):
    """
    自动执行数据预处理流程

    参数:
    trips: DataFrame, 原始数据
    original_trips: DataFrame, 原始数据副本

    返回:
    tuple: (清洗后的数据, 特征工程后的数据)
    """
    print("\n" + "=" * 60)
    print("开始自动数据预处理流程")
    print("=" * 60)

    # 询问是否需要生成数据质量报告
    while True:
        choice = input("\n是否生成数据质量报告？(yes/no) > ").strip().lower()
        if choice in ['yes', 'y', '是', '是的', '要']:
            print("\n正在生成数据质量报告...")
            generate_data_quality_report(trips)
            break
        elif choice in ['no', 'n', '否', '不要']:
            print("\n跳过数据质量报告")
            break
        else:
            print("请输入 yes 或 no")

    # 自动执行数据清洗
    print("\n正在执行数据清洗...")
    trips_cleaned, cleaning_stats = clean_data(trips, original_trips)
    print(f"✓ 数据清洗完成")

    # 自动执行特征工程
    print("\n正在执行特征工程...")
    trips_featured, feature_stats = feature_engineering(trips_cleaned)
    print(f"✓ 特征工程完成")

    print("\n" + "=" * 60)
    print("数据预处理流程完成！")
    print("=" * 60)

    return trips_cleaned, trips_featured


def main():
    """主函数"""
    print("=" * 60)
    print("出租车数据分析系统 - QA交互界面")
    print("=" * 60)

    # 读取文件
    print("\n正在加载数据...")
    trips = pq.read_table('data/yellow_tripdata_2023-01.parquet')
    trips = trips.to_pandas()
    original_trips = trips.copy()
    print(f"✓ 数据读取完成，共 {len(trips)} 条记录\n")

    # 自动执行数据预处理
    trips_cleaned, trips_featured = auto_preprocessing(trips, original_trips)

    # 显示欢迎信息和帮助
    print("\n欢迎使用数据分析功能！")
    print("输入 'help' 或 '帮助' 查看可用命令\n")

    # 主循环
    running = True
    while running:
        try:
            # 获取用户输入
            user_input = input("请输入您的问题 > ").strip()

            # 跳过空输入
            if not user_input:
                continue

            query_lower = user_input.lower().strip()

            # 退出命令
            if query_lower in ['quit', 'exit', 'q', '退出', '结束']:
                print("\n感谢使用，再见！")
                running = False
                continue

            # 帮助命令
            if query_lower in ['help', '帮助', 'h', '?', 'commands']:
                show_help()
                continue

            # 匹配命令
            matched_cmd = match_command(user_input)

            if matched_cmd:
                print(f"\n正在执行: {matched_cmd['name']}")
                print("-" * 60)

                # 检查是否需要数据参数
                need_data = matched_cmd.get('need_data', True)

                try:
                    if need_data:
                        # 需要传入数据的函数
                        matched_cmd['func'](trips_featured)
                    else:
                        # 不需要数据的函数（包括lambda包装的）
                        if callable(matched_cmd['func']):
                            # 检查是否是lambda或其他可调用对象
                            result = matched_cmd['func']()
                            if result is not None and not isinstance(result, (dict, list, tuple)):
                                # 如果返回值不是数据结构，可能是普通函数
                                pass
                        else:
                            matched_cmd['func'](trips_featured)

                    print("-" * 60)
                    print(f"✓ {matched_cmd['name']} 执行完成\n")

                except Exception as e:
                    print(f"\n✗ 执行出错: {e}")
                    print("请检查是否已完成数据预处理，或联系管理员\n")
            else:
                # 未识别的命令
                print(f"\n⚠ 未识别的命令: '{user_input}'")
                print("输入 'help' 或 '帮助' 查看可用命令\n")

        except KeyboardInterrupt:
            print("\n\n检测到中断信号")
            print("感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试或输入 'help' 查看帮助\n")


# 启动程序
if __name__ == '__main__':
    main()


