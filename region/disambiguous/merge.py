import pandas as pd

# 读取两个csv文件
df1 = pd.read_csv('C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\region\\disambiguous\\templates_generate_neg_output.csv', encoding='utf-8-sig')
df2 = pd.read_csv('C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\region\\disambiguous\\templates_generate_nonneg_output.csv', encoding='utf-8-sig')

# 合并这两个dataframes
df = pd.concat([df1, df2])

# 重设索引，新的索引列名为'example_id'，并且我们希望索引从1开始，不是默认的从0开始
df = df.reset_index(drop=True)
df.index += 1
df = df.reset_index()

# 将列名'index'重命名为'example_id'
df = df.rename(columns={'index': 'example_id'})

# 保存合并后的DataFrame到csv文件中
df.to_csv('C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\region\\disambiguous\\disambiguous.csv', encoding='utf-8-sig', index=False)
