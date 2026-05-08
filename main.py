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




# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False



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



# ==================== 特征工程 ====================
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

# 3. 创建is_peak列（高峰期：周一至周五的7-9点和17-19点）
print("\n(3) 创建is_peak列（高峰期标记）...")
# pickup_day: 1=周一, 2=周二, 3=周三, 4=周四, 5=周五, 6=周六, 7=周日
# 高峰期条件：工作日(1-5) 且 (7-9点 或 17-19点)
trips['is_peak'] = (
    (trips['pickup_day'] <= 5) &
    ((trips['pickup_hour'] >= 7) & (trips['pickup_hour'] <= 9) |
     (trips['pickup_hour'] >= 17) & (trips['pickup_hour'] <= 19))
).astype(int)

peak_count = trips['is_peak'].sum()
print(f"    高峰期行程数量: {peak_count}")
print(f"    非高峰期行程数量: {len(trips) - peak_count}")
print(f"    高峰期占比: {(peak_count/len(trips)*100):.2f}%")

# 4. 创建pre_distance_profit列（单位距离收益）
print("\n(4) 创建pre_distance_profit列（单位距离收益）...")
trips['pre_distance_profit'] = trips['total_amount'] / trips['trip_distance']
print(f"    pre_distance_profit统计信息:")
print(f"    平均值: {trips['pre_distance_profit'].mean():.2f}")
print(f"    最小值: {trips['pre_distance_profit'].min():.2f}")
print(f"    最大值: {trips['pre_distance_profit'].max():.2f}")
print(f"    中位数: {trips['pre_distance_profit'].median():.2f}")

# 特征工程完成总结
print("\n" + "=" * 60)
print("特征工程完成")
print("=" * 60)
print(f"新增列: pickup_hour, pickup_day, dropoff_hour, is_peak, pre_distance_profit")
print(f"当前总列数: {len(trips.columns)}")
print(f"当前记录数: {len(trips)}")

# 保存特征工程后的数据
output_path_feature = 'output/featured_yellow_tripdata_2023-01.parquet'
trips.to_parquet(output_path_feature, index=False)
print(f"\n特征工程后的数据已保存到: {output_path_feature}")

# 显示最终数据结构
print("\n" + "=" * 60)
print("最终数据结构预览")
print("=" * 60)
print("\n前5行数据：")
display_cols = ['pickup_hour', 'pickup_day', 'dropoff_hour', 'is_peak',
                'trip_distance', 'total_amount', 'pre_distance_profit']
print(trips[display_cols].head())

print("\n" + "=" * 60)
print("全部处理完成")
print("=" * 60)


'''
# ==================== 可视化分析：出行需求时间分布 ====================
print("\n\n" + "=" * 60)
print("可视化分析：出行需求时间分布")
print("=" * 60)

# 1. 按小时统计平均订单量
print("\n(1) 绘制小时平均订单量折线图...")
hourly_orders = trips.groupby('pickup_hour').size()
hourly_avg_orders = hourly_orders / len(trips['pickup_day'].unique())

plt.figure(figsize=(14, 6))
plt.plot(hourly_avg_orders.index, hourly_avg_orders.values, marker='o', linewidth=2,
         markersize=6, color='#2E86AB')

# 标注每个时间点的订单量
for x, y in zip(hourly_avg_orders.index, hourly_avg_orders.values):
    plt.annotate(f'{y:.1f}', xy=(x, y), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9, color='#A23B72')

plt.xlabel('小时 (Hour)', fontsize=12, fontweight='bold')
plt.ylabel('平均订单量 (Average Orders)', fontsize=12, fontweight='bold')
plt.title('每小时平均订单量分布', fontsize=14, fontweight='bold', pad=15)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/hourly_order_distribution.png', dpi=300, bbox_inches='tight')
print("    已保存: output/hourly_order_distribution.png")
plt.show()

# 2. 工作日和周末平均订单量对比
print("\n(2) 绘制工作日vs周末平均订单量柱状图...")
# pickup_day: 1-5为工作日，6-7为周末
trips['is_weekend'] = trips['pickup_day'].apply(lambda x: 1 if x >= 6 else 0)

# 计算每天类型的订单量
daily_orders = trips.groupby(['pickup_day', 'is_weekend']).size().reset_index(name='order_count')

# 计算工作日和周末的平均订单量（按天数平均）
workday_days = 5  # 周一到周五
weekend_days = 2  # 周六和周日

workday_total = daily_orders[daily_orders['is_weekend'] == 0]['order_count'].sum()
weekend_total = daily_orders[daily_orders['is_weekend'] == 1]['order_count'].sum()

workday_avg = workday_total / workday_days
weekend_avg = weekend_total / weekend_days

categories = ['工作日\n(Mon-Fri)', '周末\n(Sat-Sun)']
avg_orders = [workday_avg, weekend_avg]
colors = ['#2E86AB', '#F18F01']

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, avg_orders, color=colors, width=0.5, edgecolor='black', linewidth=1.5)

# 标注柱状图的数值
for bar, value in zip(bars, avg_orders):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.1f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#C73E1D')

plt.xlabel('日期类型 (Day Type)', fontsize=12, fontweight='bold')
plt.ylabel('平均订单量 (Average Orders)', fontsize=12, fontweight='bold')
plt.title('工作日与周末平均订单量对比', fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/workday_weekend_comparison.png', dpi=300, bbox_inches='tight')
print("    已保存: output/workday_weekend_comparison.png")
plt.show()

print("\n" + "=" * 60)
print("可视化分析完成")
print("=" * 60)
print("生成的图表文件:")
print("  1. output/hourly_order_distribution.png - 每小时平均订单量分布")
print("  2. output/workday_weekend_comparison.png - 工作日与周末对比")



# ==================== 可视化分析：区域热度分布 ====================
print("\n\n" + "=" * 60)
print("可视化分析：区域热度分布")
print("=" * 60)

# 1. 上客热度分布（PULocationID）
print("\n(1) 绘制上客热度分布堆叠柱状图...")
# 获取前10个订单量最多的上客点
top10_pickup = trips['PULocationID'].value_counts().head(10).index.tolist()

# 筛选出这10个上客点的数据
pickup_data = trips[trips['PULocationID'].isin(top10_pickup)]

# 按上客点和是否高峰期统计订单量
pickup_peak_stats = pickup_data.groupby(['PULocationID', 'is_peak']).size().unstack(fill_value=0)
pickup_peak_stats.columns = ['off_peak', 'peak']  # 0=非高峰，1=高峰

# 按总订单量排序
pickup_peak_stats['total'] = pickup_peak_stats['peak'] + pickup_peak_stats['off_peak']
pickup_peak_stats = pickup_peak_stats.sort_values(by='total', ascending=False)

# 绘制堆叠柱状图
plt.figure(figsize=(16, 8))
x_pos = range(len(pickup_peak_stats.index))
bar_width = 0.6

# 下半部分：非高峰期（蓝色系）
bars_off_peak = plt.bar(x_pos, pickup_peak_stats['off_peak'], bar_width,
                        label='非高峰期', color='#2E86AB', edgecolor='black', linewidth=0.5)

# 上半部分：高峰期（红色系）
bars_peak = plt.bar(x_pos, pickup_peak_stats['peak'], bar_width, bottom=pickup_peak_stats['off_peak'],
                    label='高峰期', color='#C73E1D', edgecolor='black', linewidth=0.5)

# 标注非高峰期的订单量
for bar in bars_off_peak:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                f'{int(height)}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# 标注高峰期的订单量
for bar in bars_peak:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                f'{int(height)}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# 标注每根柱子的总订单量
for i, (idx, row) in enumerate(pickup_peak_stats.iterrows()):
    plt.text(i, row['total'] + 50, f'{int(row["total"])}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1A1A2E')

plt.xlabel('上客点ID (Pickup Location ID)', fontsize=12, fontweight='bold')
plt.ylabel('订单量 (Order Count)', fontsize=12, fontweight='bold')
plt.title('上客热度分布 TOP 10', fontsize=14, fontweight='bold', pad=15)
plt.xticks(x_pos, [str(int(id)) for id in pickup_peak_stats.index], rotation=45, ha='right')
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/pickup_location_heatmap.png', dpi=300, bbox_inches='tight')
print("    已保存: output/pickup_location_heatmap.png")
plt.show()

# 2. 下客热度分布（DOLocationID）
print("\n(2) 绘制下客热度分布堆叠柱状图...")
# 获取前10个订单量最多的下客点
top10_dropoff = trips['DOLocationID'].value_counts().head(10).index.tolist()

# 筛选出这10个下客点的数据
dropoff_data = trips[trips['DOLocationID'].isin(top10_dropoff)]

# 按下客点和是否高峰期统计订单量
dropoff_peak_stats = dropoff_data.groupby(['DOLocationID', 'is_peak']).size().unstack(fill_value=0)
dropoff_peak_stats.columns = ['off_peak', 'peak']  # 0=非高峰，1=高峰

# 按总订单量排序
dropoff_peak_stats['total'] = dropoff_peak_stats['peak'] + dropoff_peak_stats['off_peak']
dropoff_peak_stats = dropoff_peak_stats.sort_values(by='total', ascending=False)

# 绘制堆叠柱状图
plt.figure(figsize=(16, 8))
x_pos = range(len(dropoff_peak_stats.index))
bar_width = 0.6

# 下半部分：非高峰期（蓝色系）
bars_off_peak = plt.bar(x_pos, dropoff_peak_stats['off_peak'], bar_width,
                        label='非高峰期', color='#2E86AB', edgecolor='black', linewidth=0.5)

# 上半部分：高峰期（红色系）
bars_peak = plt.bar(x_pos, dropoff_peak_stats['peak'], bar_width, bottom=dropoff_peak_stats['off_peak'],
                    label='高峰期', color='#C73E1D', edgecolor='black', linewidth=0.5)

# 标注非高峰期的订单量
for bar in bars_off_peak:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                f'{int(height)}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# 标注高峰期的订单量
for bar in bars_peak:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                f'{int(height)}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# 标注每根柱子的总订单量
for i, (idx, row) in enumerate(dropoff_peak_stats.iterrows()):
    plt.text(i, row['total'] + 50, f'{int(row["total"])}',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1A1A2E')

plt.xlabel('下客点ID (Dropoff Location ID)', fontsize=12, fontweight='bold')
plt.ylabel('订单量 (Order Count)', fontsize=12, fontweight='bold')
plt.title('下客热度分布 TOP 10', fontsize=14, fontweight='bold', pad=15)
plt.xticks(x_pos, [str(int(id)) for id in dropoff_peak_stats.index], rotation=45, ha='right')
plt.legend(loc='upper right', fontsize=10)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/dropoff_location_heatmap.png', dpi=300, bbox_inches='tight')
print("    已保存: output/dropoff_location_heatmap.png")
plt.show()

print("\n" + "=" * 60)
print("区域热度分布分析完成")
print("=" * 60)
print("生成的图表文件:")
print("  1. output/pickup_location_heatmap.png - 上客热度分布 TOP 10")
print("  2. output/dropoff_location_heatmap.png - 下客热度分布 TOP 10")



# ==================== 可视化分析：车费影响因素 ====================
print("\n\n" + "=" * 60)
print("可视化分析：车费影响因素")
print("=" * 60)

# 1. 时段-车费散点图
print("\n(1) 绘制时段-车费散点图...")
plt.figure(figsize=(14, 8))
plt.scatter(trips['pickup_hour'], trips['fare_amount'],
           alpha=0.3, s=20, color='#2E86AB', edgecolors='none')

plt.xlabel('上车时段 (Pickup Hour)', fontsize=12, fontweight='bold')
plt.ylabel('车费金额 (Fare Amount $)', fontsize=12, fontweight='bold')
plt.title('时段与车费关系散点图', fontsize=14, fontweight='bold', pad=15)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/hour_vs_fare_scatter.png', dpi=300, bbox_inches='tight')
print("    已保存: output/hour_vs_fare_scatter.png")
plt.show()

# 2. 时段-小费散点图
print("\n(2) 绘制时段-小费散点图...")
plt.figure(figsize=(14, 8))
plt.scatter(trips['pickup_hour'], trips['tip_amount'],
           alpha=0.3, s=20, color='#F18F01', edgecolors='none')

plt.xlabel('上车时段 (Pickup Hour)', fontsize=12, fontweight='bold')
plt.ylabel('小费金额 (Tip Amount $)', fontsize=12, fontweight='bold')
plt.title('时段与小费关系散点图', fontsize=14, fontweight='bold', pad=15)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/hour_vs_tip_scatter.png', dpi=300, bbox_inches='tight')
print("    已保存: output/hour_vs_tip_scatter.png")
plt.show()

# 3. 乘客数-车费散点图
print("\n(3) 绘制乘客数-车费散点图...")
plt.figure(figsize=(14, 8))
plt.scatter(trips['passenger_count'], trips['fare_amount'],
           alpha=0.3, s=20, color='#A23B72', edgecolors='none')

plt.xlabel('乘客数量 (Passenger Count)', fontsize=12, fontweight='bold')
plt.ylabel('车费金额 (Fare Amount $)', fontsize=12, fontweight='bold')
plt.title('乘客数量与车费关系散点图', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/passenger_vs_fare_scatter.png', dpi=300, bbox_inches='tight')
print("    已保存: output/passenger_vs_fare_scatter.png")
plt.show()

print("\n" + "=" * 60)
print("车费影响因素分析完成")
print("=" * 60)
print("生成的图表文件:")
print("  1. output/hour_vs_fare_scatter.png - 时段与车费关系")
print("  2. output/hour_vs_tip_scatter.png - 时段与小费关系")
print("  3. output/passenger_vs_fare_scatter.png - 乘客数与车费关系")



# ==================== 可视化分析：长途旅行单位距离收益 ====================
print("\n\n" + "=" * 60)
print("可视化分析：长途旅行单位距离收益")
print("=" * 60)

# 1. 每一趟长途旅行的单位距离收益
print("\n(1) 绘制每趟长途旅行的单位距离收益柱状图...")
long_trip_data = trips[trips['long_trip'] == 1].copy()

if len(long_trip_data) > 0:
    plt.figure(figsize=(16, 8))
    x_pos = range(len(long_trip_data))

    bars = plt.bar(x_pos, long_trip_data['pre_distance_profit'].values,
                   color='#A23B72', edgecolor='black', linewidth=0.5)

    # 标注每根柱子的数值（每隔一定数量标注，避免过于拥挤）
    step = max(1, len(long_trip_data) // 50)  # 最多显示50个标注
    for i, bar in enumerate(bars):
        if i % step == 0:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#C73E1D')

    plt.xlabel('长途旅行序号 (Long Trip Index)', fontsize=12, fontweight='bold')
    plt.ylabel('单位距离收益 ($/mile)', fontsize=12, fontweight='bold')
    plt.title(f'每趟长途旅行的单位距离收益 (共{len(long_trip_data)}趟)',
              fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('output/long_trip_profit_detail.png', dpi=300, bbox_inches='tight')
    print(f"    已保存: output/long_trip_profit_detail.png")
    plt.show()
else:
    print("    警告: 未找到长途旅行数据")

# 2. 长途与非长途平均单位距离收益对比
print("\n(2) 绘制长途与非长途平均单位距离收益对比柱状图...")
# 计算长途旅行的平均单位距离收益
long_trip_avg_profit = trips[trips['long_trip'] == 1]['pre_distance_profit'].mean()
# 计算非长途旅行的平均单位距离收益
normal_trip_avg_profit = trips[trips['long_trip'] == 0]['pre_distance_profit'].mean()

categories = ['长途旅行\n(Long Trip)', '非长途旅行\n(Normal Trip)']
avg_profits = [long_trip_avg_profit, normal_trip_avg_profit]
colors = ['#A23B72', '#2E86AB']

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, avg_profits, color=colors, width=0.5,
               edgecolor='black', linewidth=1.5)

# 标注柱状图的数值
for bar, value in zip(bars, avg_profits):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{value:.2f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#C73E1D')

plt.xlabel('旅行类型 (Trip Type)', fontsize=12, fontweight='bold')
plt.ylabel('平均单位距离收益 ($/mile)', fontsize=12, fontweight='bold')
plt.title('长途与非长途旅行平均单位距离收益对比', fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/long_vs_normal_profit_comparison.png', dpi=300, bbox_inches='tight')
print("    已保存: output/long_vs_normal_profit_comparison.png")
plt.show()

print("\n" + "=" * 60)
print("长途旅行单位距离收益分析完成")
print("=" * 60)
print("生成的图表文件:")
print("  1. output/long_trip_profit_detail.png - 每趟长途旅行的单位距离收益")
print("  2. output/long_vs_normal_profit_comparison.png - 长途与非长途平均收益对比")
print(f"\n统计信息:")
print(f"  长途旅行平均单位距离收益: ${long_trip_avg_profit:.2f}/mile")
print(f"  非长途旅行平均单位距离收益: ${normal_trip_avg_profit:.2f}/mile")
print(f"  差异: ${abs(long_trip_avg_profit - normal_trip_avg_profit):.2f}/mile")
'''

# ==================== 机器学习：出行需求预测模型 (PyTorch) ====================
print("\n\n" + "=" * 60)
print("机器学习：出行需求预测模型 (PyTorch)")
print("=" * 60)

# 1. 找出订单量最高的上客点
print("\n(1) 筛选订单量最高的上客点...")
pickup_counts = trips['PULocationID'].value_counts()
top_pickup_id = pickup_counts.index[0]
top_pickup_count = pickup_counts.iloc[0]
print(f"    订单量最高的上客点ID: {top_pickup_id}")
print(f"    总订单数: {top_pickup_count}")

# 筛选出该上客点的数据
top_pickup_data = trips[trips['PULocationID'] == top_pickup_id].copy()
print(f"    筛选后数据量: {len(top_pickup_data)}")

# 2. 提取日期信息并构建特征
print("\n(2) 构建特征数据集...")
top_pickup_data['date'] = top_pickup_data['tpep_pickup_datetime'].dt.date
top_pickup_data['day_of_month'] = top_pickup_data['tpep_pickup_datetime'].dt.day
top_pickup_data['hour'] = top_pickup_data['tpep_pickup_datetime'].dt.hour
top_pickup_data['day_of_week'] = top_pickup_data['tpep_pickup_datetime'].dt.dayofweek + 1

# 按天-小时聚合订单量
hourly_demand = top_pickup_data.groupby(['date', 'day_of_month', 'day_of_week', 'hour']).size().reset_index(
    name='order_count')
print(f"    聚合后的数据量: {len(hourly_demand)}")

# 3. 划分训练集和测试集（1.1-1.25为训练集，1.26-1.31为测试集）
print("\n(3) 划分训练集和测试集...")
train_data = hourly_demand[hourly_demand['day_of_month'] <= 25].copy()
test_data = hourly_demand[hourly_demand['day_of_month'] >= 26].copy()

print(f"    训练集大小: {len(train_data)} (1月1日-1月25日)")
print(f"    测试集大小: {len(test_data)} (1月26日-1月31日)")

# 4. 准备特征和标签
print("\n(4) 准备特征和标签...")
feature_columns = ['day_of_month', 'day_of_week', 'hour']
X_train = train_data[feature_columns].values
y_train = train_data['order_count'].values
X_test = test_data[feature_columns].values
y_test = test_data['order_count'].values

# 5. 特征标准化
print("\n(5) 特征标准化...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"    训练集特征形状: {X_train_scaled.shape}")
print(f"    测试集特征形状: {X_test_scaled.shape}")

# 6. 转换为PyTorch张量
print("\n(6) 转换为PyTorch张量...")
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled).unsqueeze(1)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"    训练批次数量: {len(train_loader)}")

# 7. 定义神经网络模型
print("\n(7) 定义神经网络模型...")


class DemandPredictionModel(nn.Module):
    def __init__(self, input_size=3):
        super(DemandPredictionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# 初始化模型
model = DemandPredictionModel(input_size=3)
print(model)

# 8. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 9. 训练模型
print("\n(8) 训练模型...")
num_epochs = 100
patience = 10
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
train_maes = []
val_maes = []
train_rmses = []
val_rmses = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0
    epoch_train_mae = 0
    epoch_train_mse = 0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算MAE和MSE
        mae = torch.mean(torch.abs(outputs - y_batch))
        mse = torch.mean((outputs - y_batch) ** 2)

        epoch_train_loss += loss.item()
        epoch_train_mae += mae.item()
        epoch_train_mse += mse.item()
        num_batches += 1

    avg_train_loss = epoch_train_loss / num_batches
    avg_train_mae = epoch_train_mae / num_batches
    avg_train_mse = epoch_train_mse / num_batches
    avg_train_rmse = np.sqrt(avg_train_mse)

    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_mae = torch.mean(torch.abs(val_outputs - y_test_tensor)).item()
        val_mse = torch.mean((val_outputs - y_test_tensor) ** 2).item()
        val_rmse = np.sqrt(val_mse)

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    train_maes.append(avg_train_mae)
    val_maes.append(val_mae)
    train_rmses.append(avg_train_rmse)
    val_rmses.append(val_rmse)

    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"    Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Train RMSE: {avg_train_rmse:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")

    if patience_counter >= patience:
        print(f"    早停触发于 epoch {epoch + 1}")
        break

# 恢复最佳模型
if best_model_state:
    model.load_state_dict(best_model_state)

actual_epochs = len(train_losses)
print(f"\n    实际训练轮数: {actual_epochs}")

# 10. 绘制Loss曲线
print("\n(9) 绘制Loss曲线...")
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss', linewidth=2, color='#2E86AB')
plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss', linewidth=2, color='#A23B72')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

plt.subplot(1, 3, 2)
plt.plot(range(1, actual_epochs + 1), train_maes, label='Training MAE', linewidth=2, color='#2E86AB')
plt.plot(range(1, actual_epochs + 1), val_maes, label='Validation MAE', linewidth=2, color='#A23B72')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('MAE', fontsize=12, fontweight='bold')
plt.title('Model MAE During Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

plt.subplot(1, 3, 3)
plt.plot(range(1, actual_epochs + 1), train_rmses, label='Training RMSE', linewidth=2, color='#2E86AB')
plt.plot(range(1, actual_epochs + 1), val_rmses, label='Validation RMSE', linewidth=2, color='#A23B72')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('RMSE', fontsize=12, fontweight='bold')
plt.title('Model RMSE During Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('output/model_training_curves.png', dpi=300, bbox_inches='tight')
print("    已保存: output/model_training_curves.png")
plt.show()

# 11. 在测试集上进行预测
print("\n(10) 在测试集上进行预测...")
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy().flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 确保预测值为非负
y_pred = np.maximum(y_pred, 0)

# 12. 计算评估指标（分别使用MAE和RMSE）
print("\n(11) 计算评估指标...")
# 使用MAE评估
mae = mean_absolute_error(y_test, y_pred)
# 使用RMSE评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n{'=' * 60}")
print("测试集评估报告")
print(f"{'=' * 60}")
print(f"  MAE (平均绝对误差): {mae:.2f} 订单/小时")
print(f"  RMSE (均方根误差): {rmse:.2f} 订单/小时")
print(f"  测试集实际订单量范围: [{y_test.min():.0f}, {y_test.max():.0f}]")
print(f"  测试集预测订单量范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

# 计算MAE和RMSE的相对误差
mae_percentage = (mae / y_test.mean()) * 100
rmse_percentage = (rmse / y_test.mean()) * 100
print(f"\n  相对MAE: {mae_percentage:.2f}%")
print(f"  相对RMSE: {rmse_percentage:.2f}%")

# 13. 绘制预测vs实际值对比图
print("\n(12) 绘制预测vs实际值对比图...")
plt.figure(figsize=(14, 6))
x_axis = range(len(y_test))
plt.plot(x_axis, y_test, marker='o', markersize=4, linewidth=2, label='实际值', color='#2E86AB')
plt.plot(x_axis, y_pred, marker='s', markersize=4, linewidth=2, label='预测值', color='#A23B72')
plt.xlabel('测试集样本序号', fontsize=12, fontweight='bold')
plt.ylabel('订单量 (Orders)', fontsize=12, fontweight='bold')
plt.title(f'测试集预测结果对比 (上客点ID: {top_pickup_id})', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("    已保存: output/prediction_vs_actual.png")
plt.show()

# 14. 预测1月30日17时的订单量
print("\n(13) 预测1月30日17时的订单量...")
# 1月30日是星期几
jan_30_date = pd.Timestamp('2023-01-30')
jan_30_day_of_week = jan_30_date.dayofweek + 1  # 转换为1-7
jan_30_day_of_month = 30
jan_30_hour = 17

print(f"    预测时间: 2023年1月30日 (星期{jan_30_day_of_week}) 17时")

# 准备预测数据
predict_input = np.array([[jan_30_day_of_month, jan_30_day_of_week, jan_30_hour]])
predict_input_scaled = scaler_X.transform(predict_input)
predict_input_tensor = torch.FloatTensor(predict_input_scaled)

model.eval()
with torch.no_grad():
    predicted_demand_scaled = model(predict_input_tensor).numpy().flatten()[0]
    predicted_demand = scaler_y.inverse_transform([[predicted_demand_scaled]])[0][0]
    predicted_demand = max(0, predicted_demand)  # 确保非负

print(f"\n{'=' * 60}")
print("预测结果")
print(f"{'=' * 60}")
print(f"  上客点ID: {top_pickup_id}")
print(f"  预测时间: 2023年1月30日 17:00")
print(f"  预测订单量: {predicted_demand:.2f} 单")
print(f"{'=' * 60}")

# 15. 保存模型
print("\n(14) 保存模型...")
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'model_config': {
        'input_size': 3,
        'hidden1': 64,
        'hidden2': 32
    }
}, 'output/demand_prediction_model.pth')
print("    模型已保存: output/demand_prediction_model.pth")

print("\n" + "=" * 60)
print("出行需求预测模型完成")
print("=" * 60)
print("生成的文件:")
print("  1. output/model_training_curves.png - 模型训练曲线")
print("  2. output/prediction_vs_actual.png - 预测vs实际值对比")
print("  3. output/demand_prediction_model.pth - 训练好的模型 (PyTorch)")