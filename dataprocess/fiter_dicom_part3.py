import pandas as pd

# ========== 配置 ==========
csv_path = "/home/yyi/data/bonecancer/bonecancer_part_output.csv"
excel_path = "/home/yyi/data/bonecancer/primary_merge_all.xlsx"
output_success_path = "/home/yyi/data/bonecancer/success_filtered.csv"
output_failed_path = "/home/yyi/data/bonecancer/failed.csv"

barcode_col = "条码号"
exam_item_col = "检查项目"
body_part_col = "body_part_examined"

# ========== 部位映射规则（请根据实际调整）==========
BODY_PART_MAPPING = {
    '胫腓骨': ['TIBIA', 'FIBULA', 'LEG'],
    '尺桡骨': ['RADIUS', 'ULNA', 'FOREARM'],
    '股骨': ['FEMUR'],
    '肱骨': ['HUMERUS'],
    '骨盆': ['PELVIS'],
    '脊柱': ['SPINE', 'LUMBAR', 'THORACIC', 'CERVICAL'],
    '手': ['HAND'],
    '足': ['FOOT'],
    '肩': ['SHOULDER'],
    '膝': ['KNEE'],
    '踝': ['ANKLE'],
    '肘': ['ELBOW'],
    '腕': ['WRIST'],
    '胸': ['CHEST'],
    '头': ['HEAD', 'SKULL'],
}

# ========== 读取数据 ==========
df_csv = pd.read_csv(csv_path, encoding='utf-8-sig')
df_excel = pd.read_excel(excel_path, dtype={barcode_col: str})

# 确保条码列为字符串
df_csv[barcode_col] = df_csv[barcode_col].astype(str)
df_excel[barcode_col] = df_excel[barcode_col].astype(str)

# ========== 合并（保留所有 CSV 行）==========
merged = pd.merge(
    df_csv,
    df_excel[[barcode_col, exam_item_col]],
    on=barcode_col,
    how='left',
    indicator=True  # 用于标记是否匹配
)

# ========== 匹配判断函数 ==========
def is_body_part_match(exam_item, body_part):
    if pd.isna(exam_item) or pd.isna(body_part):
        return False
    exam_item = str(exam_item)
    body_part = str(body_part).upper().strip()
    for chinese_key, standard_list in BODY_PART_MAPPING.items():
        if chinese_key in exam_item:
            if body_part in [s.upper() for s in standard_list]:
                return True
    return False

# ========== 标记各类失败原因 ==========
def classify_row(row):
    bp = str(row[body_part_col]).strip()
    exam_item = row.get(exam_item_col, None)
    merged_flag = row['_merge']
    
    # 规则1: body_part_examined == "1" → 失败
    if bp == "1":
        return "body_part_is_1"
    
    # 规则2: 条码号未在 Excel 中找到 → 失败
    if merged_flag == 'left_only':
        return "barcode_not_found_in_excel"
    
    # 规则3: 条码匹配但部位不一致 → 失败
    if not is_body_part_match(exam_item, bp):
        return "body_part_mismatch"
    
    # 否则成功
    return "success"

merged['status'] = merged.apply(classify_row, axis=1)

# ========== 分离成功与失败 ==========
success_df = merged[merged['status'] == 'success'].copy()
failed_df = merged[merged['status'] != 'success'].copy()

# 删除辅助列（可选）
success_df = success_df.drop(columns=['_merge', 'status'])
failed_df = failed_df.drop(columns=['_merge'])

# ========== 保存结果 ==========
success_df.to_csv(output_success_path, index=False, encoding='utf-8-sig')
failed_df.to_csv(output_failed_path, index=False, encoding='utf-8-sig')

# ========== 打印统计 ==========
print(f"✅ 处理完成！")
print(f"  总输入行数: {len(df_csv)}")
print(f"  成功匹配且部位一致: {len(success_df)}")
print(f"  失败总数: {len(failed_df)}")
print(f"    - body_part_examined == '1': {(failed_df['status'] == 'body_part_is_1').sum()}")
print(f"    - 条码号未在 Excel 中找到: {(failed_df['status'] == 'barcode_not_found_in_excel').sum()}")
print(f"    - 部位不匹配: {(failed_df['status'] == 'body_part_mismatch').sum()}")
print(f"\n💾 成功结果保存至: {output_success_path}")
print(f"💾 失败记录保存至: {output_failed_path}")