import pandas as pd

# 配置路径和列名（请根据实际情况修改）
csv1_path = "/path/to/csv1.csv"      # 第一个 CSV（主表）
csv2_path = "/path/to/csv2.csv"      # 第二个 CSV（含 body_part_examined）
output_path = "/path/to/merged_output.csv"

# 列名配置（请按你的实际列名调整）
barcode_col = "条码号"        # 两个 CSV 中都有的条码列（注意中文列名）
csv1_id_col = "id"           # CSV1 中的 ID（对应文件名无后缀）
csv2_filename_col = "file_name"  # CSV2 中的文件名（如 815665457.dcm）
body_part_col = "body_part_examined"

# 读取 CSV
df1 = pd.read_csv(csv1_path, encoding='utf-8-sig')
df2 = pd.read_csv(csv2_path, encoding='utf-8-sig')

# 从 CSV2 的 file_name 中提取无后缀的 ID（统一转为字符串处理）
df2['id_clean'] = df2[csv2_filename_col].astype(str).str.replace(r'\.dcm$', '', case=False, regex=True)

# 合并：基于 条码号 + id 匹配
merged = pd.merge(
    df1,
    df2[[barcode_col, 'id_clean', body_part_col]],
    left_on=[barcode_col, csv1_id_col],
    right_on=[barcode_col, 'id_clean'],
    how='left'
)

# 删除辅助列 'id_clean'
merged = merged.drop(columns=['id_clean'])

# 保存结果
merged.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 合并完成！结果已保存至: {output_path}")
print(f"📊 总行数: {len(merged)}，其中匹配到 body_part_examined 的有: {merged[body_part_col].notna().sum()} 行")