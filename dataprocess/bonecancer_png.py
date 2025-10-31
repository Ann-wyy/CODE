import pydicom
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path

def convert_dicom_to_png(dicom_path, output_png_path):
    """
    将单个DICOM文件转换为PNG图像。
    会尝试进行窗宽窗位处理，并处理 MONOCHROME1 反色。
    """
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array

        # 窗宽窗位处理
        if 'WindowCenter' in dicom_data and 'WindowWidth' in dicom_data:
            wc = dicom_data.WindowCenter
            ww = dicom_data.WindowWidth
            # 处理可能的多值（如列表）
            wc = wc[0] if hasattr(wc, '__len__') and not isinstance(wc, str) else wc
            ww = ww[0] if hasattr(ww, '__len__') and not isinstance(ww, str) else ww

            min_val = float(wc) - float(ww) / 2.0
            max_val = float(wc) + float(ww) / 2.0
            pixel_array = np.clip(pixel_array, min_val, max_val)
            pixel_array = ((pixel_array - min_val) / (max_val - min_val)) * 255.0
        else:
            # 全局 min-max 归一化（保留负值）
            pixel_array = pixel_array.astype(np.float32)
            p_min, p_max = pixel_array.min(), pixel_array.max()
            if p_max > p_min:
                pixel_array = (pixel_array - p_min) / (p_max - p_min) * 255.0
            else:
                pixel_array = np.zeros_like(pixel_array)

        pixel_array = pixel_array.astype(np.uint8)

        # 反色处理（MONOCHROME1 表示黑底白图，需反转）
        if hasattr(dicom_data, 'PhotometricInterpretation'):
            if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = cv2.bitwise_not(pixel_array)

        # 保存 PNG
        cv2.imwrite(output_png_path, pixel_array)
        return True
    except Exception as e:
        print(f"    ❌ 转换失败 {dicom_path}: {e}")
        return False


def safe_get_attr(ds, attr_name, default="UNKNOWN"):
    """安全获取 DICOM 属性，避免缺失字段报错"""
    try:
        val = getattr(ds, attr_name, None)
        if val is None:
            return default
        return str(val).strip()
    except Exception:
        return default


if __name__ == "__main__":
    # ========== 配置参数 ==========
    input_csv = '/home/yyi/bone_cancer.csv'      # 包含“条码号”和“文件夹路径”的 CSV
    barcode_column = '条码号'                    # 条码列名（作为标签）
    folder_path_column = '文件夹路径' 
    output_png_root = '/data/truenas_B2/yyi/data/6y_bone_cancer'  # 所有 PNG 保存的根目录
    output_csv = 'bone_cancer_png.csv'           # 输出的新 CSV：记录标签、DICOM路径、PNG路径等

    # 创建输出目录
    os.makedirs(output_png_root, exist_ok=True)

    # 读取输入 CSV
    df = pd.read_csv(input_csv, dtype={barcode_column: str})
    
    if barcode_column not in df.columns or folder_path_column not in df.columns:
        raise ValueError(f"CSV 必须包含列: {barcode_column} 和 {folder_path_column}")

    records = []  # 用于保存新 CSV 的记录
    total_converted = 0

    for _, row in df.iterrows():
        barcode = str(row[barcode_column]).strip() if pd.notna(row[barcode_column]) else "UNKNOWN_BARCODE"
        dicom_folder = row[folder_path_column]

        if not isinstance(dicom_folder, str) or not os.path.isdir(dicom_folder):
            print(f"⚠️ 跳过无效路径: 条码={barcode}, 路径={dicom_folder}")
            continue

        print(f"\n🔍 处理条码: {barcode} | 路径: {dicom_folder}")

        # 遍历该文件夹下所有 .dcm 文件（包括子目录）
        dcm_files = list(Path(dicom_folder).rglob("*.dcm")) + list(Path(dicom_folder).rglob("*.DCM"))
        
        if not dcm_files:
            print(f"  ⚠️ 未找到 DICOM 文件")
            continue

        for dcm_path in dcm_files:
            dcm_path = str(dcm_path)
            try:
                # 快速读取元数据（不加载像素）以获取 PatientID
                meta = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                patient_id = safe_get_attr(meta, 'PatientID', 'UNKNOWN_PATIENT')
            except Exception as e:
                print(f"    ⚠️ 无法读取 PatientID，使用 UNKNOWN_PATIENT: {dcm_path} | {e}")
                patient_id = "UNKNOWN_PATIENT"

            # 构建 PNG 文件名：条码号_PatientID_原始文件名（不含扩展名）
            dcm_stem = Path(dcm_path).stem
            png_name = f"{barcode}_{patient_id}_{dcm_stem}.png"
            png_path = os.path.join(output_png_root, png_name)

            # 避免重名（虽然概率极低，但保留保险机制）
            counter = 1
            base_png_path = png_path
            while os.path.exists(png_path):
                png_name = f"{barcode}_{patient_id}_{dcm_stem}_{counter}.png"
                png_path = os.path.join(output_png_root, png_name)
                counter += 1
                if counter > 10:
                    break  # 防止无限循环

            print(f"  🖼️ 正在转换: {os.path.basename(dcm_path)} -> {png_name}")
            if convert_dicom_to_png(dcm_path, png_path):
                records.append({
                    '条码号': barcode,
                    'PatientID': patient_id,
                    'DICOM路径': dcm_path,
                    'PNG路径': png_path
                })
                total_converted += 1

    # 保存映射 CSV
    if records:
        result_df = pd.DataFrame(records)
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ 转换完成！共转换 {total_converted} 张图像。")
        print(f"📊 映射信息已保存至: {output_csv}")
        print(f"📁 PNG 图像保存在: {output_png_root}")
    else:
        print("❌ 未转换任何图像。")


