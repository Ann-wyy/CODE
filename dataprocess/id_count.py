import pandas as pd
from pathlib import Path

# 读取 CSV（假设文件名为 data.csv）
df = pd.read_csv('/home/yyi/data/data_6y/data_cancer.csv')  # ← 替换为你的实际文件名

# 方法1：使用 pathlib.Path 提取倒数第三部分（推荐，跨平台安全）
df['barcode'] = df['image_path'].apply(lambda x: Path(x).parts[-2])


# 统计去重条码数量
unique_count = df['barcode'].nunique()

print(f"去重条码号数量: {unique_count}")

# （可选）查看所有去重条码
# print(df['barcode'].unique())