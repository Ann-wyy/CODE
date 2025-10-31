import pandas as pd

file1 = '/home/yyi/bone_cancer.csv'        # 临床数据
file2 = '/home/yyi/bone_cancer_png.csv'    # 图像路径数据
output_file = '/home/yyi/data/bone_cancer_png_path.csv'

# 读取
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("📁 file1 列:", df1.columns.tolist())
print("📁 file2 列:", df2.columns.tolist())

# 假设两个表可以通过 '条码号' 或 'PatientID' 关联
# 这里以 '条码号' 为例（你也可以用 'PatientID'，根据实际情况选）
on_column = '条码号'  # 或 'PatientID'

# 左连接：保留所有有 PNG 路径的记录，并补充临床信息
df_merged = pd.merge(df2, df1, on=on_column, how='left')

# 现在指定你最终想要的列（这些列可能来自 df1 或 df2）
final_columns = ['条码号', 'PatientID', '病理结果', '良恶性', 'PNG路径']

# 检查哪些列实际存在
available_cols = [col for col in final_columns if col in df_merged.columns]
missing_cols = [col for col in final_columns if col not in df_merged.columns]

if missing_cols:
    print(f"⚠️ 警告：以下列在合并后不存在，将跳过：{missing_cols}")

# 只保留存在的列，并按你指定的顺序（如果存在）
df_output = df_merged[available_cols]

# 保存
df_output.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ 合并完成！")
print(f"💾 已保存为：{output_file}")
print(f"📊 总行数：{len(df_output)}")
print(f"📋 输出列：{df_output.columns.tolist()}")