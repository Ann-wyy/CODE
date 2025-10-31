import os
import pandas as pd

# ----------------------------
# 配置参数（请根据实际情况修改）
# ----------------------------
csv_file_path = '/data/truenas_B2/yyi/xray_records.csv'          # 输入的CSV文件路径
folder_path = '/data/truenas_B2/Dataset/001_6yXray'         # 要比对的文件夹路径（里面包含以条码命名的子文件夹）
barcode_column = '条码号'             # CSV中条码号的列名（可改为 'barcode' 等）
columns_to_save = ['条码号', '临床诊断']
output_excel = '/data/truenas_B2/yyi/6y_exist_xray.xlsx' # 输出的Excel文件名
# ----------------------------

# 1. 读取CSV文件
df = pd.read_csv(csv_file_path, encoding='utf-8')  # 如果是GBK编码，可改为 encoding='gbk'

# 检查条码列是否存在
if barcode_column not in df.columns:
    raise ValueError(f"CSV文件中没有找到列名 '{barcode_column}'，请检查列名是否正确。")

# 2. 获取文件夹下的所有子文件夹名（作为有效的条码号集合）
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"文件夹路径不存在: {folder_path}")

valid_barcodes = set()
for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if os.path.isdir(item_path):  # 只取文件夹
        valid_barcodes.add(item)

print(f"找到 {len(valid_barcodes)} 个有效文件夹（条码）")

# 3. 筛选CSV中条码号存在于文件夹中的行
# 注意：条码号在CSV中可能是数字或字符串，统一转为字符串比较
df[barcode_column] = df[barcode_column].astype(str)
matched_df = df[df[barcode_column].isin(valid_barcodes)]

print(f"匹配到 {len(matched_df)} 条记录")

# 4. 保存到Excel
matched_df.to_excel(output_excel, index=False, engine='openpyxl')

print(f"匹配结果已保存到: {output_excel}")