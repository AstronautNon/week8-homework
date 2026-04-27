#库导入区
import pyarrow.parquet as pq

#主函数区

#读取文件
trips = pq.read_table('data/yellow_tripdata_2023-01.parquet')
trips = trips.to_pandas()

print(trips)
