#库导入区
import pyarrow.parquet as pq
from functions import (generate_data_quality_report,
                       clean_data, feature_engineering,
                       analyze_hour_distribution,
                       analyze_workday_weekend_distribution,
                       TOP10_PULocationID,
                       TOP10_DOLocationID,
                       hour_vs_fare,
                       hour_vs_tips)


#主函数区

#读取文件
trips = pq.read_table('data/yellow_tripdata_2023-01.parquet')
trips = trips.to_pandas()
original_trips = trips.copy()
print(f"数据读取完成，共 {len(trips)} 条记录\n")

#试运行区

#调用函数
#report = generate_data_quality_report(trips)
trips_cleaned, cleaning_stats = clean_data(trips, original_trips)
trips_featured, feature_stats = feature_engineering(trips_cleaned)
#hourly_avg = analyze_hour_distribution(trips_featured)
#workday_weekend_stats = analyze_workday_weekend_distribution(trips_featured)
#top10_pickup_stats = TOP10_PULocationID(trips_featured)
#top10_dropoff_stats = TOP10_DOLocationID(trips_featured)
#hour_fare_stats = hour_vs_fare(trips_featured)
hour_tip_stats = hour_vs_tips(trips_featured)