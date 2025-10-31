import os
import pandas as pd

# ========== 配置参数 ==========
excel_path = '/data/truenas_B2/yyi/data/primary.xlsx'          # Excel 文件路径
barcode_column = 'X线条码号'             # Excel 中包含条码号的列名
folder_path = r'/data/truenas_B2/Dataset/001_6yXray/bone_cancer'    # 要匹配的父文件夹路径
output_csv = 'bone_cancer.csv'        # 输出结果的 Excel 文件名
save_columns = ['X线条码号', '病理结果', '良性1/中间型2/恶性3']  # 若为 None，则保存整行；若指定列，如 ['条码号', '商品名称']，则只保存这些列

# ========== 主程序 ==========
# 1. 读取 Excel，条码列强制为字符串
df = pd.read_excel(excel_path, dtype={barcode_column: str})

# 2. 获取所有子文件夹名（仅一级），并构建“文件夹名 -> 完整路径”的映射
subfolder_paths = {
    name: os.path.join(folder_path, name)
    for name in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, name))
}

# 3. 清理条码列
df[barcode_column] = df[barcode_column].astype(str).str.strip()

# 4. 创建一个新列：文件夹路径，初始为 NaN
df['文件夹路径'] = df[barcode_column].map(subfolder_paths)  # 自动匹配，未匹配的为 NaN

# 5. 筛选出成功匹配的行（即“文件夹路径”非空）
matched_df = df.dropna(subset=['文件夹路径']).copy()

# 6. 如果指定了 save_columns，则只保留这些列 + “文件夹路径”
if save_columns is not None:
    # 确保“文件夹路径”被包含
    cols_to_keep = [col for col in save_columns if col in df.columns]
    cols_to_keep.append('文件夹路径')
    matched_df = matched_df[cols_to_keep]

# 7. 保存为 CSV
matched_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"✅ 匹配完成！共找到 {len(matched_df)} 个匹配项，结果已保存到 '{output_csv}'")