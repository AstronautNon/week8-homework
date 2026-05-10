#陈硕勋-25343004-人工智能编程期末作业

## 1.任务拆解与AI写协作策略
我先按照期末作业的四个模块拆分成四项大任务，然后在进行细化。
M1 数据处理：
1. 读取数据
2. 生成数据质量报告
3. 清洗数据
4. 特征工程

M2 分析可视化：
1. 出行需求时间规律
2. 区域热度分析
3. 车费影响因素分析
4. 长途旅行单位距离收益分析

M3 构建预测模型：
1. 使用深度学习方法构建神经网络
2. 绘制神经网络loss曲线和测试集报告
3. 使用随机森林构建模型
4. 对比随机森林和深度学习的测试集报告以及预测正确率

M4 问答接口：
1. 将main.py中的代码进行封装
2. 构建问答接口
3. 接入DeepSeek大模型对无法解决的问题进行解释


## 2.核心 Prompt 迭代记录
初代Prompt：
根据main.py制作预测模型部分，调用已训练好的神经网络，要求： 
1、能够让用户输入具体时间
2、调用模型预测用户输入的时间，并与真实值进行比较，得到绝对数量差和相对数量差
3、让用户输入前先输出一段例子，预测一遍1月30日17时的需求量，来展示输入的格式
4、只需要新生成的函数以及其调用方式，不需要生成已有的部分

AI的问题：
在保存神经网络模型时，采用整体保存的方式，用 weights_only=True 加载这个文件时，PyTorch 发现里面存在一个它不允许的全局对象 sklearn.preprocessing._data.StandardScaler，于是抛出 Unsupported global 错误，拒绝加载。

解决方案：
在神经网络构建预测模型后，保存模型时改为采用state_dict保存，同时单独保存 scaler（使用 joblib）


## 3. 人工代码审查
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


## 4.AI交互日志
### M1：
关键Prompt：
现在进行下一项操作。
1.新建pickup_hour和pickup_day和dropoff_hour列，从上车时间（tpep_pickup_datetime）中提取出小时、星期数据,从下车时间（tpep_dropoff_datetime）提取出小时存入这三列中；
2.新建is_peak列，记录是否高峰（高峰期为周一至周五的7am-9am,17am-19am）
3.新建pre_distance_profit列，通过total_amount/trip_distance得到
保留上文内容，生成实现该功能的程序

AI输出：

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

缺点： 星期用0-6表示，增加阅读难度

修改部分：

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

### M2：
关键Prompt：
第四个功能：研究长途旅行与非长途旅行的单位距离收益
要求：
1、绘制柱状图，描述每一趟长途旅行的单位距离收益，横轴为每一次长途旅行，纵轴为单位距离收益，依据'long_trip'和'pre_distance_profit'
2、绘制长途与非长途平均单位距离收益对比，算出长途和非长途旅行的平均单位距离收益，用柱状图表示。
以上柱状图都要求在柱子上方标注出具体的单位距离收益

AI输出：

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

### M3：
关键Prompt