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



