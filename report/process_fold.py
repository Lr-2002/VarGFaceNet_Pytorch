import pandas as pd

# 创建空列表存储每个fold的 weighted avg 结果
weighted_avg_list = []

# 读取每个fold的csv文件
for i in range(10):
    file_name = f'vargface_{i}.csv'
    df = pd.read_csv(file_name)

    # 提取weighted avg行的precision, recall, f1-score
    weighted_avg = df.loc[df['Unnamed: 0'] == 'weighted avg', ['precision', 'recall', 'f1-score']]

    # 为行命名并添加到列表中
    weighted_avg.index = [f'fold_{i}']
    weighted_avg_list.append(weighted_avg)

# 将所有折的weighted avg信息合并为一个DataFrame
result_df = pd.concat(weighted_avg_list)

# 保存结果为CSV文件
result_df.to_csv('weighted_avg_summary.csv', index=True)
print("已成功保存weighted avg汇总表格到 weighted_avg_summary.csv")
