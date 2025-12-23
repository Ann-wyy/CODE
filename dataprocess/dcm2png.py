import os
import pandas as pd
import pydicom
import cv2
import numpy as np

# ==================== 配置 ====================
csv_path = '/home/yyi/data/data_pretrain/bonecancer_backup.csv'
png_root = '/data/truenas_B2/yyi/data/6y_bone_cancer'  # 已有 PNG 的根目录（也会在这里生成新 PNG）
dicom_col = 'DICOM文件'
patient_id_col = '影像号'
png_col = 'image_path'

# ==================================================

def extract_barcode_and_filebase(dicom_path):
    """从 DICOM 路径提取 barcode（倒数第二目录）和 file_base（文件名无扩展名）"""
    parts = os.path.normpath(dicom_path).split(os.sep)
    if len(parts) < 2:
        raise ValueError(f"路径太短: {dicom_path}")
    barcode = parts[-3]
    file_base = os.path.splitext(parts[-1])[0]
    return barcode, file_base

def convert_dicom_to_png(dicom_path, output_png_path):
    """将单个 DICOM 转为 PNG（复用你的逻辑）"""
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array

        if 'WindowCenter' in dicom_data and 'WindowWidth' in dicom_data:
            wc = dicom_data.WindowCenter
            ww = dicom_data.WindowWidth
            wc = wc[0] if hasattr(wc, '__len__') and not isinstance(wc, str) else wc
            ww = ww[0] if hasattr(ww, '__len__') and not isinstance(ww, str) else ww

            min_val = wc - ww / 2
            max_val = wc + ww / 2
            pixel_array = np.clip(pixel_array, min_val, max_val)
            pixel_array = ((pixel_array - min_val) / (max_val - min_val)) * 255.0
            pixel_array = pixel_array.astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(float)
            pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
            pixel_array = np.uint8(pixel_array)

        if getattr(dicom_data, 'PhotometricInterpretation', '') == "MONOCHROME1":
            pixel_array = cv2.bitwise_not(pixel_array)

        cv2.imwrite(output_png_path, pixel_array)
        return True
    except Exception as e:
        print(f"    ❌ 转换失败: {dicom_path} -> {e}")
        return False

# ==================== 主流程 ====================


# 读取 CSV
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 检查必要列
required_cols = [dicom_col, patient_id_col]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV 缺少必要列: {col}")

# 初始化 PNG 路径列
if png_col not in df.columns:
    df[png_col] = ''

total_matched = 0
total_generated = 0

for idx, row in df.iterrows():
    dicom_path = row[dicom_col]
    patient_id = row[patient_id_col]

    # 跳过无效行
    if pd.isna(dicom_path) or not str(dicom_path).strip():
        continue
    if pd.isna(patient_id):
        print(f"    ⚠️ 第 {idx} 行：PatientID 缺失，跳过")
        continue

    dicom_path = str(dicom_path).strip()
    patient_id = str(patient_id).strip()  # 确保是整数字符串（如 123 而非 123.0）

    # 提取 barcode 和 file_base
    try:
        barcode, file_base = extract_barcode_and_filebase(dicom_path)
    except Exception as e:
        print(f"    ⚠️ 第 {idx} 行：路径解析失败 - {dicom_path}: {e}")
        continue

    # 构造 PNG 文件名
    png_filename = f"{barcode}_{patient_id}_{file_base}.png"
    expected_png_path = os.path.join(png_root, png_filename)

    # 检查是否已存在
    if os.path.isfile(expected_png_path):
        df.at[idx, png_col] = expected_png_path
        total_matched += 1
        print(f"  ✔️ 已存在: {png_filename}")
    else:
        # 不存在 → 生成 PNG
        print(f"  ➕ 生成 PNG: {png_filename}")
        if convert_dicom_to_png(dicom_path, expected_png_path):
            df.at[idx, png_col] = expected_png_path
            total_generated += 1
        else:
            df.at[idx, png_col] = ''  # 转换失败留空

# 保存更新后的 CSV
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 处理完成！")
print(f"📁 已匹配: {total_matched}")
print(f"🆕 新生成: {total_generated}")
print(f"📄 CSV 已更新，PNG 路径列: '{png_col}'")
print(f"🖼️ PNG 根目录: {png_root}")