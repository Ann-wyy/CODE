import pandas as pd

# === 配置区（按你的需求修改）===
csv1_path = "/home/yyi/data/data_pretrain/isCancer_0.csv"
csv2_path = "/home/yyi/data/data_pretrain/原发转移骨肿瘤_匹配合并结果_含部位.csv"
output_path = "/home/yyi/data/data_pretrain/isCancer.csv"

columns_to_keep = ["DICOM文件", "年龄", "性别", "影像号","检查部位", "标签"]  # ← 修改为你需要的列名

# 指定在第二个 CSV 中要设为 1 的列名（通常是标签列）
label_column = "标签"           # ← 修改为你的标签列名

# === 主逻辑 ===
# 读取两个 CSV
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# 只保留指定的列（如果某列不存在会报错，可加容错）
df1 = df1[columns_to_keep]
for col in columns_to_keep:
    if col not in df2.columns:
        if col == label_column:
            df2[col] = 1 # 为第二个 CSV 新增 label=1

# 将第二个 CSV 的指定列设为 1
df2 = df2[columns_to_keep]

# 按行拼接（纵向合并）
merged_df = pd.concat([df1, df2], ignore_index=True)

# 保存结果
merged_df.to_csv(output_path, index=False)

print(f"✅ 合并完成！共 {len(merged_df)} 行，已保存到: {output_path}")