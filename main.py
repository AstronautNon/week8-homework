#库导入区
import pyarrow.parquet as pq

#主函数区

#读取文件
trips = pq.read_table('data/yellow_tripdata_2023-01.parquet')
trips = trips.to_pandas()

# 获取并打印表头信息
print("数据表的列名（表头）：")
print(trips.columns.tolist())

print("\n数据表结构信息：")
print(trips.info())

print("\n前5行数据预览：")
print(trips.head())
