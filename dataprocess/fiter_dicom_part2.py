import pandas as pd
import os

# 配置参数
input_csv = "/home/yyi/data/cancer.csv"          # 替换为你的输入 CSV 路径
output_csv = "/home/yyi/data/cancer_part_output.csv"        # 输出 CSV 路径
path_column = "image_path"                     # 你的路径列名（根据实际情况修改）

# 读取 CSV
df = pd.read_csv(input_csv, encoding='utf-8-sig')  # utf-8-sig 处理可能的 BOM

# 提取倒数第二个路径部分（即父文件夹名）
def extract_parent_dir(path):
    if pd.isna(path) or not isinstance(path, str):
        return ""
    # 规范化路径并分割
    parts = [p for p in os.path.normpath(path).split(os.sep) if p]
    if len(parts) >= 2:
        return parts[-3]  # 倒数第几
    else:
        return ""  # 路径太短，无法提取

df['parent_folder'] = df[path_column].apply(extract_parent_dir)

# 保存结果
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"✅ 已成功提取父文件夹名并保存至: {output_csv}")