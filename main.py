#库导入区
import pyarrow.parquet as pq
import pandas as pd

#主函数区

#读取文件
trips = pq.read_table('data/yellow_tripdata_2023-01.parquet')
trips = trips.to_pandas()



# 生成数据质量报告
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
total_anomalies = trips[combined_anomaly_mask]

print(f"\n存在至少一种异常的记录总数: {len(total_anomalies)}")
print(f"异常记录占比: {(len(total_anomalies)/len(trips)*100):.2f}%")
print(f"完全正常的记录数: {len(trips) - len(total_anomalies)}")
print(f"正常记录占比: {((len(trips) - len(total_anomalies))/len(trips)*100):.2f}%")

# 5. 详细异常示例
print("\n" + "=" * 60)
print("【异常数据示例】")
print("=" * 60)

if len(total_anomalies) > 0:
    print("\n前10条异常记录：")
    display_cols = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                    'passenger_count', 'trip_distance', 'fare_amount']
    available_cols = [col for col in display_cols if col in total_anomalies.columns]
    print(total_anomalies[available_cols].head(10).to_string())

print("\n" + "=" * 60)
print("报告生成完毕")
print("=" * 60)


# ==================== 数据清洗 ====================
print("\n\n" + "=" * 60)
print("数据清洗开始")
print("=" * 60)
print(f"原始数据记录数: {len(trips)}")

# 1. 删除所有包含NaN的行
#理由：含NaN行的数据存在缺失，无法补回数据
print("\n(1) 删除包含NaN的行...")
initial_count = len(trips)
trips = trips.dropna()
print(f"    删除了 {initial_count - len(trips)} 条记录")
print(f"    剩余记录数: {len(trips)}")

# 2. 删除passenger_count异常的记录 (<=0 或 >6)
#乘客数量超出物理限制，记录错误，无法找回真实值
print("\n(2) 删除passenger_count异常的记录 (<=0 或 >6)...")
initial_count = len(trips)
mask_passenger = (trips['passenger_count'] <= 0) | (trips['passenger_count'] > 6)
trips = trips[~mask_passenger].copy()
print(f"    删除了 {initial_count - len(trips)} 条记录")
print(f"    剩余记录数: {len(trips)}")

# 3. 删除tpep_pickup_datetime >= tpep_dropoff_datetime的记录
#上车时间晚于下车时间，登记错误顾删除数据
print("\n(3) 删除上车时间 >= 下车时间的记录...")
initial_count = len(trips)
mask_time = trips['tpep_pickup_datetime'] >= trips['tpep_dropoff_datetime']
trips = trips[~mask_time].copy()
print(f"    删除了 {initial_count - len(trips)} 条记录")
print(f"    剩余记录数: {len(trips)}")

# 4. 处理trip_distance
#行程距离<0或>350均为异常记录，>100可能是长途行程，故添加标签方便后续研究
print("\n(4) 处理trip_distance...")
# 4.1 删除trip_distance <= 0的记录
initial_count = len(trips)
mask_distance_zero = trips['trip_distance'] <= 0
trips = trips[~mask_distance_zero].copy()
print(f"    删除trip_distance <= 0的记录: {initial_count - len(trips)} 条")

# 4.2 删除trip_distance > 350的记录
initial_count = len(trips)
mask_distance_extreme = trips['trip_distance'] > 350
trips = trips[~mask_distance_extreme].copy()
print(f"    删除trip_distance > 350的记录: {initial_count - len(trips)} 条")

# 4.3 为100 < trip_distance <= 350添加long_trip标签
trips['long_trip'] = trips['trip_distance'].apply(lambda x: 1 if (x > 100 and x <= 350) else 0)
long_trip_count = trips['long_trip'].sum()
print(f"    添加long_trip标签: {long_trip_count} 条记录标记为长途行程")
print(f"    剩余记录数: {len(trips)}")

# 5. 处理total_amount：删除<=0和>=1000的记录
#>=1000的路费过高，甚至超过长途旅行范围，故删除
print("\n(5) 处理total_amount...")
initial_count = len(trips)
mask_total_invalid = (trips['total_amount'] <= 0) | (trips['total_amount'] >= 1000)
trips = trips[~mask_total_invalid].copy()
print(f"    删除了 {initial_count - len(trips)} 条记录 (total_amount <= 0 或 >= 1000)")
print(f"    剩余记录数: {len(trips)}")

# 清洗完成总结
print("\n" + "=" * 60)
print("数据清洗完成")
print("=" * 60)
print(f"清洗后记录数: {len(trips)}")
print(f"总共删除记录数: {len(pq.read_table('data/yellow_tripdata_2023-01.parquet').to_pandas()) - len(trips)}")

# 保存清洗后的数据
output_path = 'output/cleaned_yellow_tripdata_2023-01.parquet'
trips.to_parquet(output_path, index=False)
print(f"\n清洗后的数据已保存到: {output_path}")