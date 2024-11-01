import pandas as pd

# 读取结果文件
result_df = pd.read_csv('result.csv')  # 预测结果文件
label_df = pd.read_csv('test_data/label.csv')    # 真实标签文件

# 将真实标签文件中的 id 列设置为索引，以便于快速查找
label_df.set_index('id', inplace=True)

# 将预测结果文件与真实标签文件合并，按照 id 列进行匹配
merged_df = result_df.merge(label_df, left_on='image_id', right_index=True)

# 计算预测正确的数量
correct_predictions = (merged_df['predicted_digit'] == merged_df['label']).sum()

# 计算总的样本数
total_samples = len(merged_df)

# 计算准确率
accuracy = correct_predictions / total_samples * 100

print(f'预测准确率为: {accuracy:.2f}%')
