import os
import pandas as pd

# ========== 配置 ==========
excel_path = '/home/yyi/data/data_pretrain/原发转移骨肿瘤_匹配合并结果.xlsx'
barcode_column = 'DICOM文件'
folder_path = r'/data/truenas_B2/Dataset/001_6yXray/bone_cancer'
output_success = 'bone_cancer.csv'
output_failed = 'bone_cancer_failed.xlsx'

# ✅ 指定所有需要防止 .0 的 ID 列
id_columns = ['DICOM文件', '影像号']  # ← 根据你的数据调整，加更多列如 '申请单号' 等

save_columns = ['DICOM文件', '原发/转移', '原发良性1/中间型2/恶性3', '影像号']

# ========== 清洗函数 ==========
def clean_id_value(x):
    if pd.isna(x):
        return ''
    if isinstance(x, (int, float)):
        if x != x:
            return ''
        try:
            return str(int(x))
        except (ValueError, OverflowError, TypeError):
            return str(x).strip()
    s = str(x).strip()
    if s.lower() in ('', 'nan', 'none'):
        return ''
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        else:
            return s
    except ValueError:
        return s

# ========== 主程序 ==========
df = pd.read_excel(excel_path)

# 🔥 对所有 ID 列进行清洗
for col in id_columns:
    if col in df.columns:
        df[col] = df[col].apply(clean_id_value)

# 构建文件夹映射（基于条码号 exact match）
subfolder_paths = {
    name: os.path.join(folder_path, name)
    for name in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, name))
}

# 匹配（只用条码号）
df['文件夹路径'] = df[barcode_column].map(subfolder_paths)

# 分离
matched_df = df[df['文件夹路径'].notna()].copy()
failed_df = df[df['文件夹路径'].isna()].copy()

# 保存成功文件
if save_columns is not None:
    cols_to_keep = [col for col in save_columns if col in df.columns]
    cols_to_keep.append('文件夹路径')
    matched_df = matched_df[cols_to_keep]

matched_df.to_csv(output_success, index=False, encoding='utf-8-sig')
failed_df.to_excel(output_failed, index=False)

print(f"✅ 完成！成功: {len(matched_df)}，失败: {len(failed_df)}")