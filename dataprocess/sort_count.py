import pandas as pd

# 1. 读取 CSV 文件
df = pd.read_csv('/home/yyi/data/bonecancer_png.csv')  # 替换为你的文件路径

# 2. 指定要统计的列名（例如 '诊断结果'）
column_name = '病理结果'  # ←←← 请根据你的实际列名修改这里！

# 3. 检查该列是否存在
if column_name not in df.columns:
    raise ValueError(f"列 '{column_name}' 不存在于 CSV 文件中。可用列：{list(df.columns)}")

# 4. 进行分类计数（自动忽略 NaN）
value_counts = df[column_name].value_counts(dropna=True)

# 5. 打印结果
print("分类统计结果：")
print(value_counts)

# 6. （可选）将统计结果保存为新的 CSV 文件
value_counts_df = value_counts.reset_index()
value_counts_df.columns = [column_name, '频次']  # 重命名列
value_counts_df.to_csv('分类统计结果.csv', index=False, encoding='utf-8-sig')

print("\n统计结果已保存到 '分类统计结果.csv'")