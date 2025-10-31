import pydicom
import cv2
import numpy as np
import os

def convert_dicom_to_png(dicom_path, output_png_path):
    """
    将单个DICOM文件转换为PNG图像。
    会尝试进行简单的窗宽窗位处理以获得较好的可视化效果。
    """
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array

        # 尝试应用窗宽窗位
        if 'WindowCenter' in dicom_data and 'WindowWidth' in dicom_data:
            window_center = dicom_data.WindowCenter
            window_width = dicom_data.WindowWidth

            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]

            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2

            pixel_array = np.clip(pixel_array, min_val, max_val)
            pixel_array = ((pixel_array - min_val) / (max_val - min_val)) * 255.0
            pixel_array = pixel_array.astype(np.uint8)
        else:
            # 简单归一化
            pixel_array = pixel_array.astype(float)
            pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
            pixel_array = np.uint8(pixel_array)

        # 黑白反转
        if hasattr(dicom_data, 'PhotometricInterpretation') and dicom_data.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = cv2.bitwise_not(pixel_array)

        cv2.imwrite(output_png_path, pixel_array)
        return True
    except Exception as e:
        print(f"    错误：无法转换 {dicom_path}: {e}")
        return False


def convert_all_dicom_folders(dicom_root_folder, output_png_folder):
    """
    遍历指定 DICOM 根文件夹下的所有子文件夹，
    将所有 .dcm 文件转换为 PNG，并统一保存到输出文件夹。
    """
    if not os.path.exists(output_png_folder):
        os.makedirs(output_png_folder)
        print(f"已创建输出文件夹: {output_png_folder}")

    total_converted = 0

    for root, _, files in os.walk(dicom_root_folder):
        for file_name in files:
            if file_name.lower().endswith('.dcm'):
                dicom_path = os.path.join(root, file_name)
                png_file_name = os.path.splitext(file_name)[0] + ".png"
                output_png_path = os.path.join(output_png_folder, png_file_name)

                # 避免重复写入同名文件（可选）
                counter = 1
                while os.path.exists(output_png_path):
                    png_file_name = f"{os.path.splitext(file_name)[0]}_{counter}.png"
                    output_png_path = os.path.join(output_png_folder, png_file_name)
                    counter += 1

                print(f"  正在转换: {dicom_path} -> {output_png_path}")
                if convert_dicom_to_png(dicom_path, output_png_path):
                    total_converted += 1

    print(f"\n--- 转换完成 ---")
    print(f"总共转换了 {total_converted} 张 DICOM 图像为 PNG 格式。")
    print(f"所有 PNG 图像已保存至: {output_png_folder}")


# --- 配置你的路径 ---
dicom_root_folder = "/data/truenas_B2/Dataset/001_6yXray/A602978218/1"         # 替换为你的 DICOM 根文件夹路径
output_png_folder = "/home/yyi/dinov2/images"  # 替换为你想保存 PNG 的文件夹路径


# 执行转换
convert_all_dicom_folders(dicom_root_folder, output_png_folder)