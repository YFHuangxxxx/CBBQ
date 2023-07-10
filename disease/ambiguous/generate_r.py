import pandas as pd
import re
import ast
import itertools

# 读取csv文件
df = pd.read_csv('C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\disease\\templates_generate_neg.csv', encoding='utf-8-sig')  

# 新的DataFrame，用于存储结果
df_new = pd.DataFrame(columns=['question_index', 'question_polarity', 'context_condition', 'category', 'subcategory', 'bias_targeted_groups', 'source', 'context', 'question', 'ans0', 'ans1', 'ans2', 'label'])

# 遍历原始DataFrame的每一行
for index, row in df.iterrows():
    if not isinstance(row['NAME1'], str):
        row['NAME1'] = str(row['NAME1'])  # 转换为字符串类型
    row['NAME1'] = re.sub(r'\b(\w+)\b', r'"\1"', row['NAME1'])
    
    if pd.isnull(row['NAME2']):
        # 如果NAME2是空的，只填充NAME1
        name1_list = ast.literal_eval(row['NAME1'])
        for name1 in name1_list:
            sentence1 = row['context'].replace('{{NAME1}}', name1)
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
                'ans0': row['ans0'], 
                'ans1': row['ans1'],
                'ans2': row['ans2'],
                'label': row['label']
            }], index=[0])], ignore_index=True)
    else:
        # 如果NAME2不是空的，填充NAME1和NAME2
        if not isinstance(row['NAME2'], str):
            row['NAME2'] = str(row['NAME2'])  # 转换为字符串类型
        row['NAME2'] = re.sub(r'\b(\w+)\b', r'"\1"', row['NAME2'])
        for name1, name2 in itertools.product(ast.literal_eval(row['NAME1']), ast.literal_eval(row['NAME2'])):
            sentence1 = row['context'].replace('{{NAME1}}', name1).replace('{{NAME2}}', name2)
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
                'ans0': row['ans0'], 
                'ans1': row['ans1'],
                'ans2': row['ans2'],
                'label': row['label']
            }], index=[0])], ignore_index=True)

# 写入新的csv文件
df_new.to_csv('C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\disease\\templates_generate_neg_output.csv', index=False)
