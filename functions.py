#库导入区
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
import subprocess
import platform

'''
函数存放区
'''


def generate_data_quality_report(trips):
    """
    生成数据质量报告

    参数:
    trips: DataFrame，包含出租车行程数据

    返回:
    dict: 包含各种异常掩码和统计信息的字典
    """
    print("=" * 60)
    print("数据质量报告")
    print("=" * 60)

    # 1. 基本信息
    print("\n【基本信息】")
    print(f"总记录数: {len(trips)}")
    print(f"总列数: {len(trips.columns)}")

    # 2. 缺失值分析
    print("\n" + "=" * 60)
    print("【缺失值分析】")
    print("=" * 60)

    missing_values = trips.isnull().sum()
    missing_percentage = (missing_values / len(trips) * 100).round(2)
    missing_df = pd.DataFrame({
        '缺失数量': missing_values,
        '缺失比例(%)': missing_percentage
    })
    missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values(by='缺失数量', ascending=False)

    if len(missing_df) > 0:
        print("\n存在缺失值的列：")
        for col, row in missing_df.iterrows():
            print(f"  {col}: {row['缺失数量']} ({row['缺失比例(%)']}%)")
        print(f"\n总计缺失值数量: {missing_values.sum()}")
    else:
        print("\n未发现缺失值")

    # 3. 异常值分析
    print("\n" + "=" * 60)
    print("【异常值分析】")
    print("=" * 60)

    # 3.1 时间异常：pickup_datetime >= dropoff_datetime
    time_mask = trips['tpep_pickup_datetime'] >= trips['tpep_dropoff_datetime']
    time_anomalies = trips[time_mask]
    print(f"\n(1) 时间异常 (上车时间 >= 下车时间): {len(time_anomalies)} 条")
    if len(time_anomalies) > 0:
        print(f"    占比: {(len(time_anomalies)/len(trips)*100):.2f}%")

    # 3.2 乘客数量异常：<=0 或 >6
    passenger_mask = (trips['passenger_count'] <= 0) | (trips['passenger_count'] > 6)
    passenger_anomalies = trips[passenger_mask]
    print(f"\n(2) 乘客数量异常 (<=0 或 >6): {len(passenger_anomalies)} 条")
    if len(passenger_anomalies) > 0:
        print(f"    占比: {(len(passenger_anomalies)/len(trips)*100):.2f}%")
        if 'passenger_count' in trips.columns:
            print(f"    最小值: {trips.loc[passenger_mask, 'passenger_count'].min()}")
            print(f"    最大值: {trips.loc[passenger_mask, 'passenger_count'].max()}")

    # 3.3 行驶距离异常：<=0 或 >=100
    distance_mask = (trips['trip_distance'] <= 0) | (trips['trip_distance'] >= 100)
    distance_anomalies = trips[distance_mask]
    print(f"\n(3) 行驶距离异常 (<=0 或 >=100): {len(distance_anomalies)} 条")
    if len(distance_anomalies) > 0:
        print(f"    占比: {(len(distance_anomalies)/len(trips)*100):.2f}%")
        if 'trip_distance' in trips.columns:
            print(f"    最小值: {trips.loc[distance_mask, 'trip_distance'].min()}")
            print(f"    最大值: {trips.loc[distance_mask, 'trip_distance'].max()}")

    # 3.4 车费金额异常：>=1000 或 <=0
    total_mask = (trips['total_amount'] >= 1000) | (trips['total_amount'] <= 0)
    total_anomalies = trips[total_mask]
    print(f"\n(4) 车费金额异常 (>=1000 或 <=0): {len(total_anomalies)} 条")
    if len(total_anomalies) > 0:
        print(f"    占比: {(len(total_anomalies)/len(trips)*100):.2f}%")
        if 'fare_amount' in trips.columns:
            print(f"    最小值: {trips.loc[total_mask, 'fare_amount'].min()}")
            print(f"    最大值: {trips.loc[total_mask, 'fare_amount'].max()}")

    # 4. 异常值汇总
    print("\n" + "=" * 60)
    print("【异常值汇总】")
    print("=" * 60)

    # 创建综合异常标记
    combined_anomaly_mask = time_mask | passenger_mask | distance_mask | total_mask
    total_anomalies_combined = trips[combined_anomaly_mask]

    print(f"\n存在至少一种异常的记录总数: {len(total_anomalies_combined)}")
    print(f"异常记录占比: {(len(total_anomalies_combined)/len(trips)*100):.2f}%")
    print(f"完全正常的记录数: {len(trips) - len(total_anomalies_combined)}")
    print(f"正常记录占比: {((len(trips) - len(total_anomalies_combined))/len(trips)*100):.2f}%")

    # 5. 详细异常示例
    print("\n" + "=" * 60)
    print("【异常数据示例】")
    print("=" * 60)

    if len(total_anomalies_combined) > 0:
        print("\n前10条异常记录：")
        display_cols = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                        'passenger_count', 'trip_distance', 'fare_amount']
        available_cols = [col for col in display_cols if col in total_anomalies_combined.columns]
        print(total_anomalies_combined[available_cols].head(10).to_string())

    print("\n" + "=" * 60)
    print("报告生成完毕")
    print("=" * 60)

    # 返回异常掩码和统计信息，供后续使用
    return {
        'time_mask': time_mask,
        'passenger_mask': passenger_mask,
        'distance_mask': distance_mask,
        'total_mask': total_mask,
        'combined_anomaly_mask': combined_anomaly_mask,
        'missing_values': missing_values,
        'total_records': len(trips),
        'anomaly_count': len(total_anomalies_combined)
    }


def clean_data(trips, original_trips=None):
    """
    数据清洗函数

    参数:
    trips: DataFrame，包含出租车行程数据
    original_trips: DataFrame，原始数据（用于计算删除记录数），可选

    返回:
    tuple: (清洗后的DataFrame, 清洗统计信息字典)
    """
    print("\n\n" + "=" * 60)
    print("数据清洗开始")
    print("=" * 60)
    print(f"原始数据记录数: {len(trips)}")

    if original_trips is None:
        original_count = len(trips)
    else:
        original_count = len(original_trips)

    # 1. 删除所有包含NaN的行
    print("\n(1) 删除包含NaN的行...")
    print("    理由：含NaN行的数据存在缺失，无法补回数据")
    before_count = len(trips)
    trips = trips.dropna()
    print(f"    删除了 {before_count - len(trips)} 条记录")
    print(f"    剩余记录数: {len(trips)}")

    # 2. 删除passenger_count异常的记录 (<=0 或 >6)
    print("\n(2) 删除passenger_count异常的记录 (<=0 或 >6)...")
    print("    理由：乘客数量超出物理限制，记录错误，无法找回真实值")
    before_count = len(trips)
    mask_passenger = (trips['passenger_count'] <= 0) | (trips['passenger_count'] > 6)
    trips = trips[~mask_passenger].copy()
    print(f"    删除了 {before_count - len(trips)} 条记录")
    print(f"    剩余记录数: {len(trips)}")

    # 3. 删除tpep_pickup_datetime >= tpep_dropoff_datetime的记录
    print("\n(3) 删除上车时间 >= 下车时间的记录...")
    print("    理由：上车时间晚于下车时间，登记错误故删除数据")
    before_count = len(trips)
    mask_time = trips['tpep_pickup_datetime'] >= trips['tpep_dropoff_datetime']
    trips = trips[~mask_time].copy()
    print(f"    删除了 {before_count - len(trips)} 条记录")
    print(f"    剩余记录数: {len(trips)}")

    # 4. 处理trip_distance
    print("\n(4) 处理trip_distance...")
    print("    理由：行程距离<0或>350均为异常记录，>100可能是长途行程，故添加标签方便后续研究")

    before_count = len(trips)
    mask_distance_zero = trips['trip_distance'] <= 0
    trips = trips[~mask_distance_zero].copy()
    print(f"    删除trip_distance <= 0的记录: {before_count - len(trips)} 条")

    before_count = len(trips)
    mask_distance_extreme = trips['trip_distance'] > 350
    trips = trips[~mask_distance_extreme].copy()
    print(f"    删除trip_distance > 350的记录: {before_count - len(trips)} 条")

    trips['long_trip'] = trips['trip_distance'].apply(lambda x: 1 if (x > 100 and x <= 350) else 0)
    long_trip_count = trips['long_trip'].sum()
    print(f"    添加long_trip标签: {long_trip_count} 条记录标记为长途行程")
    print(f"    剩余记录数: {len(trips)}")

    # 5. 处理total_amount：删除<=0和>=1000的记录
    print("\n(5) 处理total_amount...")
    print("    理由：>=1000的路费过高，甚至超过长途旅行范围，故删除")
    before_count = len(trips)
    mask_total_invalid = (trips['total_amount'] <= 0) | (trips['total_amount'] >= 1000)
    trips = trips[~mask_total_invalid].copy()
    print(f"    删除了 {before_count - len(trips)} 条记录 (total_amount <= 0 或 >= 1000)")
    print(f"    剩余记录数: {len(trips)}")

    print("\n" + "=" * 60)
    print("数据清洗完成")
    print("=" * 60)
    total_deleted = original_count - len(trips)
    print(f"清洗后记录数: {len(trips)}")
    print(f"总共删除记录数: {total_deleted}")
    print(f"删除比例: {(total_deleted / original_count * 100):.2f}%")

    cleaning_stats = {
        'original_count': original_count,
        'cleaned_count': len(trips),
        'total_deleted': total_deleted,
        'deletion_ratio': (total_deleted / original_count * 100),
        'long_trip_count': long_trip_count
    }

    return trips, cleaning_stats


def feature_engineering(trips):
    """
    特征工程函数

    参数:
    trips: DataFrame，清洗后的数据

    返回:
    tuple: (添加特征后的DataFrame, 特征工程统计信息字典)
    """
    print("\n\n" + "=" * 60)
    print("特征工程开始")
    print("=" * 60)

    # 1. 从tpep_pickup_datetime提取小时和星期
    print("\n(1) 提取上车时间的小时和星期...")
    trips['pickup_hour'] = trips['tpep_pickup_datetime'].dt.hour
    trips['pickup_day'] = trips['tpep_pickup_datetime'].dt.dayofweek + 1
    print(f"    pickup_hour范围: {trips['pickup_hour'].min()} - {trips['pickup_hour'].max()}")
    print(f"    pickup_day范围: {trips['pickup_day'].min()} (周一) - {trips['pickup_day'].max()} (周日)")

    # 2. 从tpep_dropoff_datetime提取小时
    print("\n(2) 提取下车时间的小时...")
    trips['dropoff_hour'] = trips['tpep_dropoff_datetime'].dt.hour
    print(f"    dropoff_hour范围: {trips['dropoff_hour'].min()} - {trips['dropoff_hour'].max()}")

    # 3. 创建is_peak列
    print("\n(3) 创建is_peak列（高峰期标记）...")
    trips['is_peak'] = (
            (trips['pickup_day'] <= 5) &
            ((trips['pickup_hour'] >= 7) & (trips['pickup_hour'] <= 9) |
             (trips['pickup_hour'] >= 17) & (trips['pickup_hour'] <= 19))
    ).astype(int)

    peak_count = trips['is_peak'].sum()
    print(f"    高峰期行程数量: {peak_count}")
    print(f"    非高峰期行程数量: {len(trips) - peak_count}")
    print(f"    高峰期占比: {(peak_count / len(trips) * 100):.2f}%")

    # 4. 创建pre_distance_profit列
    print("\n(4) 创建pre_distance_profit列（单位距离收益）...")
    trips['pre_distance_profit'] = trips['total_amount'] / trips['trip_distance']
    print(f"    pre_distance_profit统计信息:")
    print(f"    平均值: {trips['pre_distance_profit'].mean():.2f}")
    print(f"    最小值: {trips['pre_distance_profit'].min():.2f}")
    print(f"    最大值: {trips['pre_distance_profit'].max():.2f}")
    print(f"    中位数: {trips['pre_distance_profit'].median():.2f}")

    print("\n" + "=" * 60)
    print("特征工程完成")
    print("=" * 60)
    new_features = ['pickup_hour', 'pickup_day', 'dropoff_hour', 'is_peak', 'pre_distance_profit']
    print(f"新增列: {new_features}")
    print(f"当前总列数: {len(trips.columns)}")
    print(f"当前记录数: {len(trips)}")

    feature_stats = {
        'new_features': new_features,
        'total_columns': len(trips.columns),
        'total_records': len(trips),
        'peak_count': int(peak_count),
        'peak_ratio': float(peak_count / len(trips) * 100)
    }

    return trips, feature_stats


def analyze_hour_distribution(trips_data):
    """
    研究出行需求时间分布的函数

    参数:
    trips_data: DataFrame，包含行程数据，需要有'tpep_pickup_datetime'列

    返回:
    hourly_avg_orders: 每小时平均订单量的Series
    """
    print("\n\n" + "=" * 60)
    print("出行需求时间分布分析")
    print("=" * 60)

    # 确保数据中包含必要的列
    if 'tpep_pickup_datetime' not in trips_data.columns:
        raise ValueError("数据中缺少'tpep_pickup_datetime'列")

    # 提取小时信息
    trips_data = trips_data.copy()
    trips_data['pickup_hour'] = trips_data['tpep_pickup_datetime'].dt.hour

    # 计算每小时的订单量
    hourly_orders = trips_data.groupby('pickup_hour').size()

    # 计算平均订单量（按天数平均）
    unique_days = trips_data['tpep_pickup_datetime'].dt.date.nunique()
    if unique_days > 0:
        hourly_avg_orders = hourly_orders / unique_days
    else:
        hourly_avg_orders = hourly_orders

    # 输出小时平均订单量统计
    print("\n小时平均订单量统计:")
    print("-" * 30)
    for hour in range(24):
        if hour in hourly_avg_orders.index:
            avg_count = hourly_avg_orders[hour]
            print(f"  {hour:02d}:00 - 平均订单量: {avg_count:.1f}")
        else:
            print(f"  {hour:02d}:00 - 平均订单量: 0.0")

    # 图片路径
    image_path = 'output/hourly_order_distribution.png'

    # 检查图片是否存在
    if os.path.exists(image_path):
        print(f"\n图片相对路径: {image_path}")
        print(f"图片绝对路径: {os.path.abspath(image_path)}")

        # 打开图片
        try:
            system_platform = platform.system()
            if system_platform == "Darwin":  # macOS
                subprocess.call(['open', image_path])
            elif system_platform == "Windows":
                subprocess.call(['start', image_path], shell=True)
            else:  # Linux
                subprocess.call(['xdg-open', image_path])
            print(f"已尝试打开图片: {image_path}")
        except Exception as e:
            print(f"无法自动打开图片: {e}")
            print(f"请手动打开: {image_path}")
    else:
        print(f"\n警告: 图片文件不存在: {image_path}")
        print("请先运行 main.py 生成该图片")

    print("\n" + "=" * 60)
    print("出行需求时间分布分析完成")
    print("=" * 60)

    return hourly_avg_orders


def analyze_workday_weekend_distribution(trips_data):
    """
    研究工作日和周末出行需求分布的函数

    参数:
    trips_data: DataFrame，包含行程数据，需要有'pickup_day'列（1-7表示周一到周日）

    返回:
    dict: 包含工作日和周末平均订单量的字典
    """
    print("\n\n" + "=" * 60)
    print("工作日与周末出行需求分布分析")
    print("=" * 60)

    # 确保数据中包含必要的列
    if 'pickup_day' not in trips_data.columns:
        raise ValueError("数据中缺少'pickup_day'列，请先进行特征工程")

    # 创建is_weekend列
    trips_data = trips_data.copy()
    trips_data['is_weekend'] = trips_data['pickup_day'].apply(lambda x: 1 if x >= 6 else 0)

    # 计算每天类型的订单量
    daily_orders = trips_data.groupby(['pickup_day', 'is_weekend']).size().reset_index(name='order_count')

    # 计算工作日和周末的平均订单量（按天数平均）
    workday_days = 5  # 周一到周五
    weekend_days = 2  # 周六和周日

    workday_total = daily_orders[daily_orders['is_weekend'] == 0]['order_count'].sum()
    weekend_total = daily_orders[daily_orders['is_weekend'] == 1]['order_count'].sum()

    workday_avg = workday_total / workday_days
    weekend_avg = weekend_total / weekend_days

    # 输出统计信息
    print("\n工作日与周末平均订单量统计:")
    print("-" * 40)
    print(f"  工作日平均订单量: {workday_avg:.1f} 单/天")
    print(f"  周末平均订单量: {weekend_avg:.1f} 单/天")
    print(f"  差异: {abs(workday_avg - weekend_avg):.1f} 单/天")

    if workday_avg > weekend_avg:
        print(f"  结论: 工作日订单量比周末高 {(workday_avg - weekend_avg) / weekend_avg * 100:.1f}%")
    else:
        print(f"  结论: 周末订单量比工作日高 {(weekend_avg - workday_avg) / workday_avg * 100:.1f}%")

    # 图片路径
    image_path = 'output/workday_weekend_comparison.png'

    # 检查图片是否存在
    if os.path.exists(image_path):
        print(f"\n图片相对路径: {image_path}")
        print(f"图片绝对路径: {os.path.abspath(image_path)}")

        # 打开图片
        try:
            system_platform = platform.system()
            if system_platform == "Darwin":  # macOS
                subprocess.call(['open', image_path])
            elif system_platform == "Windows":
                subprocess.call(['start', image_path], shell=True)
            else:  # Linux
                subprocess.call(['xdg-open', image_path])
            print(f"已尝试打开图片: {image_path}")
        except Exception as e:
            print(f"无法自动打开图片: {e}")
            print(f"请手动打开: {image_path}")
    else:
        print(f"\n警告: 图片文件不存在: {image_path}")
        print("请先运行 main.py 生成该图片")

    print("\n" + "=" * 60)
    print("工作日与周末出行需求分布分析完成")
    print("=" * 60)

    return {
        'workday_avg': workday_avg,
        'weekend_avg': weekend_avg,
        'workday_total': workday_total,
        'weekend_total': weekend_total
    }


def TOP10_PULocationID(trips_data):
    """
    研究10个订单量最多的上客点的函数

    参数:
    trips_data: DataFrame，包含行程数据，需要有'PULocationID'和'is_peak'列

    返回:
    dict: 包含TOP10上客点统计信息的字典
    """
    print("\n\n" + "=" * 60)
    print("区域热度分析：TOP 10 上客点")
    print("=" * 60)

    # 确保数据中包含必要的列
    if 'PULocationID' not in trips_data.columns:
        raise ValueError("数据中缺少'PULocationID'列")
    if 'is_peak' not in trips_data.columns:
        raise ValueError("数据中缺少'is_peak'列，请先进行特征工程")

    # 获取前10个订单量最多的上客点
    top10_pickup = trips_data['PULocationID'].value_counts().head(10).index.tolist()

    # 筛选出这10个上客点的数据
    pickup_data = trips_data[trips_data['PULocationID'].isin(top10_pickup)]

    # 按上客点和是否高峰期统计订单量
    pickup_peak_stats = pickup_data.groupby(['PULocationID', 'is_peak']).size().unstack(fill_value=0)
    pickup_peak_stats.columns = ['off_peak', 'peak']  # 0=非高峰，1=高峰

    # 按总订单量排序
    pickup_peak_stats['total'] = pickup_peak_stats['peak'] + pickup_peak_stats['off_peak']
    pickup_peak_stats = pickup_peak_stats.sort_values(by='total', ascending=False)

    # 输出TOP10上客点统计信息
    print("\nTOP 10 热度最高上客点统计:")
    print("-" * 70)
    print(f"{'排名':<6} {'上客点ID':<12} {'总订单量':<12} {'高峰期订单量':<14} {'非高峰期订单量':<14}")
    print("-" * 70)

    result_dict = {}
    for rank, (location_id, row) in enumerate(pickup_peak_stats.iterrows(), 1):
        total = int(row['total'])
        peak = int(row['peak'])
        off_peak = int(row['off_peak'])

        print(f"{rank:<6} {int(location_id):<12} {total:<12} {peak:<14} {off_peak:<14}")

        result_dict[int(location_id)] = {
            'rank': rank,
            'total_orders': total,
            'peak_orders': peak,
            'off_peak_orders': off_peak
        }

    # 图片路径
    image_path = 'output/pickup_location_heatmap.png'

    # 检查图片是否存在
    if os.path.exists(image_path):
        print(f"\n图片相对路径: {image_path}")
        print(f"图片绝对路径: {os.path.abspath(image_path)}")

        # 打开图片
        try:
            system_platform = platform.system()
            if system_platform == "Darwin":  # macOS
                subprocess.call(['open', image_path])
            elif system_platform == "Windows":
                subprocess.call(['start', image_path], shell=True)
            else:  # Linux
                subprocess.call(['xdg-open', image_path])
            print(f"已尝试打开图片: {image_path}")
        except Exception as e:
            print(f"无法自动打开图片: {e}")
            print(f"请手动打开: {image_path}")
    else:
        print(f"\n警告: 图片文件不存在: {image_path}")
        print("请先运行 main.py 生成该图片")

    print("\n" + "=" * 60)
    print("TOP 10 上客点分析完成")
    print("=" * 60)

    return result_dict


def TOP10_DOLocationID(trips_data):
    """
    研究10个订单量最多的下客点的函数

    参数:
    trips_data: DataFrame，包含行程数据，需要有'DOLocationID'和'is_peak'列

    返回:
    dict: 包含TOP10下客点统计信息的字典
    """
    print("\n\n" + "=" * 60)
    print("区域热度分析：TOP 10 下客点")
    print("=" * 60)

    # 确保数据中包含必要的列
    if 'DOLocationID' not in trips_data.columns:
        raise ValueError("数据中缺少'DOLocationID'列")
    if 'is_peak' not in trips_data.columns:
        raise ValueError("数据中缺少'is_peak'列，请先进行特征工程")

    # 获取前10个订单量最多的下客点
    top10_dropoff = trips_data['DOLocationID'].value_counts().head(10).index.tolist()

    # 筛选出这10个下客点的数据
    dropoff_data = trips_data[trips_data['DOLocationID'].isin(top10_dropoff)]

    # 按下客点和是否高峰期统计订单量
    dropoff_peak_stats = dropoff_data.groupby(['DOLocationID', 'is_peak']).size().unstack(fill_value=0)
    dropoff_peak_stats.columns = ['off_peak', 'peak']  # 0=非高峰，1=高峰

    # 按总订单量排序
    dropoff_peak_stats['total'] = dropoff_peak_stats['peak'] + dropoff_peak_stats['off_peak']
    dropoff_peak_stats = dropoff_peak_stats.sort_values(by='total', ascending=False)

    # 输出TOP10下客点统计信息
    print("\nTOP 10 热度最高下客点统计:")
    print("-" * 70)
    print(f"{'排名':<6} {'下客点ID':<12} {'总订单量':<12} {'高峰期订单量':<14} {'非高峰期订单量':<14}")
    print("-" * 70)

    result_dict = {}
    for rank, (location_id, row) in enumerate(dropoff_peak_stats.iterrows(), 1):
        total = int(row['total'])
        peak = int(row['peak'])
        off_peak = int(row['off_peak'])

        print(f"{rank:<6} {int(location_id):<12} {total:<12} {peak:<14} {off_peak:<14}")

        result_dict[int(location_id)] = {
            'rank': rank,
            'total_orders': total,
            'peak_orders': peak,
            'off_peak_orders': off_peak
        }

    # 图片路径
    image_path = 'output/dropoff_location_heatmap.png'

    # 检查图片是否存在
    if os.path.exists(image_path):
        print(f"\n图片相对路径: {image_path}")
        print(f"图片绝对路径: {os.path.abspath(image_path)}")

        # 打开图片
        try:
            system_platform = platform.system()
            if system_platform == "Darwin":  # macOS
                subprocess.call(['open', image_path])
            elif system_platform == "Windows":
                subprocess.call(['start', image_path], shell=True)
            else:  # Linux
                subprocess.call(['xdg-open', image_path])
            print(f"已尝试打开图片: {image_path}")
        except Exception as e:
            print(f"无法自动打开图片: {e}")
            print(f"请手动打开: {image_path}")
    else:
        print(f"\n警告: 图片文件不存在: {image_path}")
        print("请先运行 main.py 生成该图片")

    print("\n" + "=" * 60)
    print("TOP 10 下客点分析完成")
    print("=" * 60)

    return result_dict


def hour_vs_fare(trips_data):
    """
    研究时段对车费影响的函数

    参数:
    trips_data: DataFrame，包含行程数据，需要有'pickup_hour'和'fare_amount'列

    返回:
    dict: 包含每个时段的平均车费统计信息
    """
    print("\n\n" + "=" * 60)
    print("车费影响因素分析：时段与车费关系")
    print("=" * 60)

    # 确保数据中包含必要的列
    if 'pickup_hour' not in trips_data.columns:
        raise ValueError("数据中缺少'pickup_hour'列，请先进行特征工程")
    if 'fare_amount' not in trips_data.columns:
        raise ValueError("数据中缺少'fare_amount'列")

    # 计算每个时段的平均车费
    hourly_fare_stats = trips_data.groupby('pickup_hour')['fare_amount'].agg(['mean', 'median', 'count']).round(2)
    hourly_fare_stats.columns = ['avg_fare', 'median_fare', 'trip_count']

    # 输出时段车费统计信息
    print("\n各时段车费统计:")
    print("-" * 55)
    print(f"{'时段':<8} {'平均车费($)':<12} {'中位数车费($)':<14} {'订单数量':<10}")
    print("-" * 55)

    result_dict = {}
    for hour in range(24):
        if hour in hourly_fare_stats.index:
            avg_fare = hourly_fare_stats.loc[hour, 'avg_fare']
            median_fare = hourly_fare_stats.loc[hour, 'median_fare']
            trip_count = int(hourly_fare_stats.loc[hour, 'trip_count'])

            print(f"{hour:02d}:00   {avg_fare:<12.2f} {median_fare:<14.2f} {trip_count:<10}")

            result_dict[hour] = {
                'avg_fare': float(avg_fare),
                'median_fare': float(median_fare),
                'trip_count': trip_count
            }
        else:
            print(f"{hour:02d}:00   {'N/A':<12} {'N/A':<14} {0:<10}")
            result_dict[hour] = {
                'avg_fare': None,
                'median_fare': None,
                'trip_count': 0
            }

    # 图片路径
    image_path = 'output/hour_vs_fare_scatter.png'

    # 检查图片是否存在
    if os.path.exists(image_path):
        print(f"\n图片相对路径: {image_path}")
        print(f"图片绝对路径: {os.path.abspath(image_path)}")

        # 打开图片
        try:
            system_platform = platform.system()
            if system_platform == "Darwin":  # macOS
                subprocess.call(['open', image_path])
            elif system_platform == "Windows":
                subprocess.call(['start', image_path], shell=True)
            else:  # Linux
                subprocess.call(['xdg-open', image_path])
            print(f"已尝试打开图片: {image_path}")
        except Exception as e:
            print(f"无法自动打开图片: {e}")
            print(f"请手动打开: {image_path}")
    else:
        print(f"\n警告: 图片文件不存在: {image_path}")
        print("请先运行 main.py 生成该图片")

    print("\n" + "=" * 60)
    print("时段与车费关系分析完成")
    print("=" * 60)

    return result_dict


def hour_vs_tips(trips_data):
    """
    研究时段对小费影响的函数

    参数:
    trips_data: DataFrame，包含行程数据，需要有'pickup_hour'和'tip_amount'列

    返回:
    dict: 包含每个时段的平均小费统计信息
    """
    print("\n\n" + "=" * 60)
    print("车费影响因素分析：时段与小费关系")
    print("=" * 60)

    # 确保数据中包含必要的列
    if 'pickup_hour' not in trips_data.columns:
        raise ValueError("数据中缺少'pickup_hour'列，请先进行特征工程")
    if 'tip_amount' not in trips_data.columns:
        raise ValueError("数据中缺少'tip_amount'列")

    # 计算每个时段的平均小费
    hourly_tip_stats = trips_data.groupby('pickup_hour')['tip_amount'].agg(['mean', 'median', 'count']).round(2)
    hourly_tip_stats.columns = ['avg_tip', 'median_tip', 'trip_count']

    # 输出时段小费统计信息
    print("\n各时段小费统计:")
    print("-" * 55)
    print(f"{'时段':<8} {'平均小费($)':<12} {'中位数小费($)':<14} {'订单数量':<10}")
    print("-" * 55)

    result_dict = {}
    for hour in range(24):
        if hour in hourly_tip_stats.index:
            avg_tip = hourly_tip_stats.loc[hour, 'avg_tip']
            median_tip = hourly_tip_stats.loc[hour, 'median_tip']
            trip_count = int(hourly_tip_stats.loc[hour, 'trip_count'])

            print(f"{hour:02d}:00   {avg_tip:<12.2f} {median_tip:<14.2f} {trip_count:<10}")

            result_dict[hour] = {
                'avg_tip': float(avg_tip),
                'median_tip': float(median_tip),
                'trip_count': trip_count
            }
        else:
            print(f"{hour:02d}:00   {'N/A':<12} {'N/A':<14} {0:<10}")
            result_dict[hour] = {
                'avg_tip': None,
                'median_tip': None,
                'trip_count': 0
            }

    # 图片路径
    image_path = 'output/hour_vs_tip_scatter.png'

    # 检查图片是否存在
    if os.path.exists(image_path):
        print(f"\n图片相对路径: {image_path}")
        print(f"图片绝对路径: {os.path.abspath(image_path)}")

        # 打开图片
        try:
            system_platform = platform.system()
            if system_platform == "Darwin":  # macOS
                subprocess.call(['open', image_path])
            elif system_platform == "Windows":
                subprocess.call(['start', image_path], shell=True)
            else:  # Linux
                subprocess.call(['xdg-open', image_path])
            print(f"已尝试打开图片: {image_path}")
        except Exception as e:
            print(f"无法自动打开图片: {e}")
            print(f"请手动打开: {image_path}")
    else:
        print(f"\n警告: 图片文件不存在: {image_path}")
        print("请先运行 main.py 生成该图片")

    print("\n" + "=" * 60)
    print("时段与小费关系分析完成")
    print("=" * 60)

    return result_dict