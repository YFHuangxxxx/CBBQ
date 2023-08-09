import csv
import json

def csv_to_json(csv_path, json_path):
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        jsonfile.write(json.dumps(rows, separators=(',', ':'), ensure_ascii=False))

if __name__ == '__main__':
    csv_path = 'C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\region\\disambiguous\\disambiguous.csv'  # 输入的CSV文件路径
    json_path = 'C:\\Users\\14190\\Desktop\\bias_data\\bias_data\\data\\region\\disambiguous\\disambiguous.json'  # 输出的JSON文件路径
    csv_to_json(csv_path, json_path)


