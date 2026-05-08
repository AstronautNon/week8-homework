#库导入区
import pyarrow.parquet as pq
from functions import generate_data_quality_report


#主函数区

#读取文件
trips = pq.read_table('data/yellow_tripdata_2023-01.parquet')
trips = trips.to_pandas()

#试运行区

#调用函数
report = generate_data_quality_report(trips)
