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
在main.py神经网络构建预测模型后，保存模型时改为采用state_dict保存，同时单独保存 scaler（使用 joblib）


## 3. 人工代码审查
    # 1. 删除所有包含NaN的行
    #理由：含NaN行的数据存在缺失，无法补回数据
    print("\n(1) 删除包含NaN的行...")
    initial_count = len(trips)
    #删除了含NaN的行
    trips = trips.dropna()
    print(f"    删除了 {initial_count - len(trips)} 条记录")
    print(f"    剩余记录数: {len(trips)}")

    # 2. 删除passenger_count异常的记录 (<=0 或 >6)
    #乘客数量超出物理限制，记录错误，无法找回真实值
    print("\n(2) 删除passenger_count异常的记录 (<=0 或 >6)...")
    initial_count = len(trips)
    #添加标签标注出乘客数异常的行
    mask_passenger = (trips['passenger_count'] <= 0) | (trips['passenger_count'] > 6)
    #删除了带有mask_passenger标记的行
    trips = trips[~mask_passenger].copy()
    print(f"    删除了 {initial_count - len(trips)} 条记录")
    print(f"    剩余记录数: {len(trips)}")

    # 3. 删除tpep_pickup_datetime >= tpep_dropoff_datetime的记录
    #上车时间晚于下车时间，登记错误顾删除数据
    print("\n(3) 删除上车时间 >= 下车时间的记录...")
    initial_count = len(trips)
    #新增标签标注出上车时间晚于下车时间的行
    mask_time = trips['tpep_pickup_datetime'] >= trips['tpep_dropoff_datetime']
    #删除了带有mask_time标记的行
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
    #添加标签标注出付费金额数异常的行
    mask_total_invalid = (trips['total_amount'] <= 0) | (trips['total_amount'] >= 1000)
    #删除了带有mask_total_invalid标记的行
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
关键Prompt：
现在我也需要一个预测模型，预测1月30日17时该月订单量最高的上客点的出行需求量。 按照日期8:2，前面日期作为训练集，后面日期作为测试集（1.1-1.25为训练集，1.26-1.31为测试集）。 先筛选出订单量最高的上车点，使用tensorflow，用两个隐藏层学习出行需求规律，学会了“星期几 + 几点钟 + 几号”与订单量之间的关系，在模型训练完后绘制loss曲线，测试集报告 MAE 与 RMSE。

AI输出：

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
    hourly_demand = top_pickup_data.groupby(['date', 'day_of_month', 'day_of_week', 'hour']).size().reset_index(name='order_count')
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

    # 6. 构建神经网络模型
    print("\n(6) 构建神经网络模型...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # 7. 设置早停机制
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 8. 训练模型
    print("\n(7) 训练模型...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    print(f"    实际训练轮数: {len(history.history['loss'])}")

    # 9. 绘制Loss曲线
    print("\n(8) 绘制Loss曲线...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2, color='#2E86AB')
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='#A23B72')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('MAE', fontsize=12, fontweight='bold')
    plt.title('Model MAE During Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('output/model_training_curves.png', dpi=300, bbox_inches='tight')
    print("    已保存: output/model_training_curves.png")
    plt.show()

    # 10. 在测试集上进行预测
    print("\n(9) 在测试集上进行预测...")
    y_pred_scaled = model.predict(X_test_scaled).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 确保预测值为非负
    y_pred = np.maximum(y_pred, 0)

    # 11. 计算评估指标（分别使用MAE和RMSE）
    print("\n(10) 计算评估指标...")
    # 使用MAE评估
    mae = mean_absolute_error(y_test, y_pred)
    # 使用RMSE评估
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{'='*60}")
    print("测试集评估报告")
    print(f"{'='*60}")
    print(f"  MAE (平均绝对误差): {mae:.2f} 订单/小时")
    print(f"  RMSE (均方根误差): {rmse:.2f} 订单/小时")
    print(f"  测试集实际订单量范围: [{y_test.min():.0f}, {y_test.max():.0f}]")
    print(f"  测试集预测订单量范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

    # 计算MAE和RMSE的相对误差
    mae_percentage = (mae / y_test.mean()) * 100
    rmse_percentage = (rmse / y_test.mean()) * 100
    print(f"\n  相对MAE: {mae_percentage:.2f}%")
    print(f"  相对RMSE: {rmse_percentage:.2f}%")

    # 12. 绘制预测vs实际值对比图
    print("\n(11) 绘制预测vs实际值对比图...")
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

    # 13. 预测1月30日17时的订单量
    print("\n(12) 预测1月30日17时的订单量...")
    # 1月30日是星期几
    jan_30_date = pd.Timestamp('2023-01-30')
    jan_30_day_of_week = jan_30_date.dayofweek + 1  # 转换为1-7
    jan_30_day_of_month = 30
    jan_30_hour = 17

    print(f"    预测时间: 2023年1月30日 (星期{jan_30_day_of_week}) 17时")

    # 准备预测数据
    predict_input = np.array([[jan_30_day_of_month, jan_30_day_of_week, jan_30_hour]])
    predict_input_scaled = scaler_X.transform(predict_input)
    predicted_demand_scaled = model.predict(predict_input_scaled).flatten()[0]
    predicted_demand = scaler_y.inverse_transform([[predicted_demand_scaled]])[0][0]
    predicted_demand = max(0, predicted_demand)  # 确保非负

    print(f"\n{'='*60}")
    print("预测结果")
    print(f"{'='*60}")
    print(f"  上客点ID: {top_pickup_id}")
    print(f"  预测时间: 2023年1月30日 17:00")
    print(f"  预测订单量: {predicted_demand:.2f} 单")
    print(f"{'='*60}")

    # 14. 保存模型
    print("\n(13) 保存模型...")
    model.save('output/demand_prediction_model.keras')
    print("    模型已保存: output/demand_prediction_model.keras")

    print("\n" + "=" * 60)
    print("出行需求预测模型完成")
    print("=" * 60)
    print("生成的文件:")
    print("  1. output/model_training_curves.png - 模型训练曲线")
    print("  2. output/prediction_vs_actual.png - 预测vs实际值对比")
    print("  3. output/demand_prediction_model.keras - 训练好的模型")

缺陷：训练过程一直卡在在Epoch 1/10不动了，检查数据不存在 NaN/Inf，初步判定是“样本太少+控制台输出缓冲区造成的假死”，但是换显示模式或是强制刷新输出代码仍会卡住不往下运行

改进方式：因找不到解决方案，怀疑是无法使用tensorflow，转用PyTorch构建模型，并且成功构建预测模型

### M4：
关键Prompt：
接下来我想要接入DeepSeek，来解释为什么用户提的问题无法解决。
Prompt：
你是一个智慧交通助手的后端解释模块。
当前系统只能完成以下功能： {system_capabilities}
用户刚才的提问是：“{user_input}” 系统没能理解或无法执行。 请你用中文友好地告诉用户：
1、为什么当前不能回答（比如功能限制、数据范围等）
2、可以建议用户怎样提问才能得到帮助（引导到已有功能上）
3、尽量简短，不超过三句话。

AI输出：

    # ==================== DeepSeek API 配置 ====================
    DEEPSEEK_API_KEY = "your_api_key_here"  # 请替换为您的 DeepSeek API Key
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


    def get_system_capabilities():
        """
        从命令注册表中提取系统能力描述
    
        返回:
        str: 系统功能描述的文本
        """
        capabilities = []
    
        # 按类别分组
        categories = {}
        for cmd in COMMAND_REGISTRY:
            category = cmd.get('category', '其他')
            if category not in categories:
                categories[category] = []
            categories[category].append(cmd)
    
        for category, commands in categories.items():
            capabilities.append(f"\n【{category}】")
            for cmd in commands:
                keywords = cmd['keywords'][:2]
                keyword_str = ' / '.join(keywords)
                capabilities.append(f"- {cmd['desc']}（可输入：{keyword_str}）")
    
        return '\n'.join(capabilities)


    def explain_unsupported_query(user_input):
        """
        调用 DeepSeek API 生成友好的错误解释
    
        参数:
        user_input: str, 用户的原始输入
    
        返回:
        str: DeepSeek 生成的解释文本，如果调用失败则返回默认提示
        """
        # 获取系统能力描述
        system_capabilities = get_system_capabilities()
    
        # 构建 prompt
        prompt = f"""你是一个智慧交通助手的后端解释模块。
    当前系统只能完成以下功能：
    {system_capabilities}

    用户刚才的提问是："{user_input}"
    系统没能理解或无法执行。请你用中文友好地告诉用户：
    1. 为什么当前不能回答（比如功能限制、数据范围等）
    2. 可以建议用户怎样提问才能得到帮助（引导到已有功能上）
    3. 尽量简短，不超过三句话。"""
    
        try:
            # 构建请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            }
        
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个友好的智能助手，擅长用简洁清晰的语言解释问题。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
        
            # 发送请求
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
        
            # 解析响应
            result = response.json()
            explanation = result['choices'][0]['message']['content'].strip()
        
            return explanation
        
        except requests.exceptions.RequestException as e:
            # API 调用失败，返回默认提示
            print(f"\n[DEBUG] DeepSeek API 调用失败: {e}")
            return generate_fallback_explanation(user_input)
        except Exception as e:
            print(f"\n[DEBUG] 生成解释时出错: {e}")
            return generate_fallback_explanation(user_input)


    def generate_fallback_explanation(user_input):
        """
        当 API 调用失败时，生成默认的友好提示
    
        参数:
        user_input: str, 用户的原始输入
    
        返回:
        str: 默认的解释文本
        """
        fallback_messages = [
            f"抱歉，我暂时无法理解您的问题。您可以输入 'help' 查看我支持的功能列表，或者尝试使用更具体的关键词，如'小时分布'、'时段车费'等。",
            f"这个问题超出了我的能力范围哦～我是出租车数据分析助手，主要提供数据统计和可视化功能。试试输入 'help' 看看我能做什么吧！",
            f"我可能没听懂您的意思。作为智慧交通助手，我可以分析出租车出行数据、预测订单量等。输入 'help' 或 '帮助' 获取完整功能列表吧！"
        ]
    
        # 根据输入长度选择不同的提示
        import random
        return random.choice(fallback_messages)


## 5.反思
第一点，AI会放大我的错误。如果我提出了一个模糊的甚至错误的目标，它会自动往那个方向扩展，将错误放大，自动修正的概率不高。例如我一开始想要训练一个模型来预测最热门地区在高峰期的订单量，但是这个目标实际的样本量其实很小，很难去支撑一个神经网络的训练。但AI并不能实时根据样本量来提示这个目标的优劣，更不用说找出合适的改正方案。
第二点，AI有一定的自动扩展能力。当我让AI以一定的要求生成代码时，能够在一些细枝末节的部分做的很好，例如我在生成多幅散点图的时候他会在不同图片的点用不同的颜色，或是在关键词的定义上帮我自动做好，但并不完整，还需要后期的调整。再者，它在保存模型时是整体保存的，但是没有意识到保存模型的目的是再调用，而这种保存方式是无法再调用的，AI的目的与做法是矛盾的
所以，在人机协作编程的当下，人更多是一个管理者、决策者、指导者，而AI更多是一个执行者的角色，我们现在更重要的不再是学会这个语法怎么用，而是在于如何将一个任务细致的拆解。