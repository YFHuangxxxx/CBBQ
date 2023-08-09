import pandas as pd
import re
import ast
import itertools

# 读取csv文件
df = pd.read_csv('C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\age\\templates_generate_neg.csv', encoding='utf-8-sig')  

# 新的DataFrame，用于存储结果
df_new = pd.DataFrame(columns=['question_index', 'question_polarity', 'context_condition', 'category', 'subcategory', 'bias_targeted_groups', 'source', 'context', 'question', 'ans0', 'ans1', 'ans2', 'label'])

# 遍历原始DataFrame的每一行
for index, row in df.iterrows():
    if not isinstance(row['NAME1'], str):
        row['NAME1'] = str(row['NAME1'])  # 转换为字符串类型
    row['NAME1'] = re.sub(r'\b(\w+)\b', r'"\1"', row['NAME1'])
    if not isinstance(row['NAME2'], str):
        row['NAME2'] = str(row['NAME2'])  # 转换为字符串类型
    row['NAME2'] = re.sub(r'\b(\w+)\b', r'"\1"', row['NAME2'])
    # 获取NAME1和NAME2的所有可能的组合
    for name1, name2 in itertools.product(ast.literal_eval(row['NAME1']), ast.literal_eval(row['NAME2'])):
        # 填充模板
        sentence1 = row['context'].replace('{{NAME1}}', name1).replace('{{NAME2}}', name2)
        sentence2 = row['context'].replace('{{NAME1}}', name2).replace('{{NAME2}}', name1)
        ans0 = row['ans0'].replace('{{NAME1}}', name1)
        ans1 = row['ans1'].replace('{{NAME2}}', name2)
        
        # 添加到新的DataFrame
        df_new = pd.concat([df_new, pd.DataFrame([{
            'question_index': row['question_index'],
            'question_polarity': row['question_polarity'],
            'context_condition': row['context_condition'],
            'category': row['category'],
            'subcategory': row['subcategory'], 
            'bias_targeted_groups': row['bias_targeted_groups'], 
            'source': row['source'], 
            'context': sentence1, 
            'question': row['question'],
            'ans0': ans0, 
            'ans1': ans1,
            'ans2': row['ans2'],
            'label': row['label']
        }], index=[0])], ignore_index=True)

        df_new = pd.concat([df_new, pd.DataFrame([{
            'question_index': row['question_index'],
            'question_polarity': row['question_polarity'],
            'context_condition': row['context_condition'],
            'category': row['category'],
            'subcategory': row['subcategory'], 
            'bias_targeted_groups': row['bias_targeted_groups'], 
            'source': row['source'], 
            'context': sentence2, 
            'question': row['question'],
            'ans0': ans0, 
            'ans1': ans1,
            'ans2': row['ans2'],
            'label': row['label']
        }], index=[0])], ignore_index=True)

# 写入新的csv文件
df_new.to_csv('C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\age\\templates_generate_neg_output.csv', index=False)
