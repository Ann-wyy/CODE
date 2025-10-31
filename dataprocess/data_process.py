# -*- coding: utf-8 -*-
import os
import numpy as np
import logging
import pandas as pd
from PIL import Image
import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm
import hashlib
from scipy.ndimage import zoom

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='skpi_image_processing.log',  # 指定日志文件名称
    filemode='a'  # 'a' 表示追加模式，如果文件存在，则在末尾添加日志
)
logger = logging.getLogger(__name__)


# --- 图像读取函数 (修改后) ---
def load_image_raw(image_path):
    if not isinstance(image_path, str):
        logger.error(f"无效的图像路径类型: {image_path}, type={type(image_path)}")
        return None
    if not os.path.exists(image_path):
        logger.warning(f"文件不存在: {image_path}")
        return None
    try:
        _, ext = os.path.splitext(image_path)
        ext = ext.lower()
        if ext in ['.dcm', '.dicom']:
            pixel_array = _load_dicom_image_raw(image_path)
            if pixel_array is None:
                return None

            # ========== ✅ 兼容多维数组但不转灰度 ==========
            if pixel_array.ndim == 3:
                # 情况1: (H, W, 3) 或 (H, W, 4) —— 彩色图，保留
                if pixel_array.shape[-1] in [3, 4]:
                    pass  # 保留原样
                # 情况2: (3, H, W) 或 (4, H, W) —— 通道在前，转置
                elif pixel_array.shape[0] in [3, 4] and pixel_array.shape[1] > 1 and pixel_array.shape[2] > 1:
                    pixel_array = np.moveaxis(pixel_array, 0, -1)  # (C, H, W) -> (H, W, C)

            # 如果处理后还不是 2D 或 3D(HWC)，跳过
            if pixel_array.ndim not in [2, 3]:
                logger.warning(f"无法处理的数组维度，跳过: {image_path}, 形状: {pixel_array.shape}")
                return None

            # ========== 归一化到 [0, 255] ==========
            img_min, img_max = pixel_array.min(), pixel_array.max()
            if img_max > img_min:
                pixel_array = (pixel_array - img_min) / (img_max - img_min) * 255.0
            else:
                pixel_array = np.zeros_like(pixel_array)
            return pixel_array.astype(np.uint8)

        elif ext in ['.png', '.jpg', '.jpeg']:
            # 不强制转灰度，保留原始模式
            img = Image.open(image_path)
            return np.array(img, dtype=np.uint8)  # 可能是 (H, W) 或 (H, W, 3)

        else:
            logger.warning(f"不支持格式: {image_path}")
            return None

    except Exception as e:
        logger.error(f"加载图像时发生未知错误: {image_path}, 错误: {e}", exc_info=True)
        return None


def _load_dicom_image_raw(image_path):
    """加载 DICOM 图像，不进行 [0, 255] 归一化"""
    try:
        dicom_data = pydicom.dcmread(image_path, force=True)
    except (InvalidDicomError, FileNotFoundError, OSError) as e:
        logger.warning(f"DICOM 读取失败（格式错误或损坏）: {image_path}, 错误: {e}")
        return None
    except Exception as e:
        logger.error(f"意外错误读取 DICOM: {image_path}, 错误: {e}", exc_info=True)
        return None
    if "PixelData" not in dicom_data:
        logger.warning(f"DICOM 文件缺少 PixelData: {image_path}")
        return None
    try:
        # 直接使用原始像素数据，通常为int16
        pixel_array = dicom_data.pixel_array
        # 窗宽窗位处理，但不对像素值进行归一化或缩放
        if 'WindowCenter' in dicom_data and 'WindowWidth' in dicom_data:
            try:
                wc = dicom_data.WindowCenter
                ww = dicom_data.WindowWidth
                if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
                if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
                min_val = float(wc) - float(ww) / 2
                max_val = float(wc) + float(ww) / 2
                pixel_array = np.clip(pixel_array, min_val, max_val)
                logger.debug(f"应用窗宽窗位: {image_path}")
            except Exception as e:
                logger.debug(f"应用窗宽窗位失败（返回原始值）: {image_path}, 错误: {e}")
        return pixel_array
    except Exception as e:
        logger.warning(f"无法解码 pixel_array（可能压缩或损坏）: {image_path}, 错误: {e}")
        return None


# --- 分批次处理函数 (使用新的加载函数) ---
def process_images_in_batches(csv_path, output_dir, batch_size=5000, start_batch=0, target_size=(518, 518)):
    """
    从 CSV 文件中读取图像路径，分批次处理，缩放并保存为 npz 文件。
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV 文件不存在: {csv_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    try:
        df = pd.read_csv(csv_path)
        if 'image_path' not in df.columns:
            logger.error("CSV 文件中未找到 'image_path' 列。")
            return
        image_paths = df['image_path'].tolist()
    except Exception as e:
        logger.error(f"读取 CSV 文件时发生错误: {e}", exc_info=True)
        return
    if not image_paths:
        logger.warning(f"在 {csv_path} 中没有找到任何图像路径。")
        return

    total_images = len(image_paths)
    total_batches = (total_images + batch_size - 1) // batch_size
    logger.info(f"总共找到 {total_images} 张图片，将分为 {total_batches} 批次处理，每批次 {batch_size} 张。")
    logger.info(f"所有图像将被缩放为 {target_size[0]}x{target_size[1]} 并以原始像素值保存。")

    
    output_csv_path = "/home/yyi/data/saved_npz_paths.csv"
    for i in range(start_batch, total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_images)
        batch_paths = image_paths[start_idx:end_idx]

        logger.info(f"--- 开始处理第 {i + 1}/{total_batches} 批次 ({len(batch_paths)} 张图片) ---")

        saved_paths_batch = [] # 用于存储成功保存的npz文件路径
        for image_path in tqdm(batch_paths, desc=f"处理批次 {i + 1}"):
            # 使用新的加载函数来获取原始像素数据
            img_np = load_image_raw(image_path)
            if img_np is None:
                logger.warning(f"跳过文件：{image_path}")
                continue

            if img_np.ndim not in [2, 3]:
                logger.warning(f"跳过不支持的数组维度: {image_path}, 形状为 {img_np.shape}")
                continue
            if img_np.ndim == 3 and img_np.shape[-1] not in [3, 4]:
                logger.warning(f"跳过不支持的通道数: {image_path}, 形状为 {img_np.shape}")
                continue

            
            # ========== ✅ 智能缩放：根据维度选择缩放方式 ==========
            try:
                original_shape = img_np.shape
                if img_np.ndim == 2:
                    # 2D 灰度图
                    img_pil = Image.fromarray(img_np, mode='L')
                    img_pil_resized = img_pil.resize(target_size, resample=Image.Resampling.LANCZOS)
                    img_np_resized = np.array(img_pil_resized)
                elif img_np.ndim == 3:
                    if img_np.shape[-1] in [3, 4]:
                        # (H, W, C) 彩色图，用 PIL
                        mode = 'RGB' if img_np.shape[2] == 3 else 'RGBA'
                        img_pil = Image.fromarray(img_np, mode=mode)
                        img_pil_resized = img_pil.resize(target_size, resample=Image.Resampling.LANCZOS)
                        img_np_resized = np.array(img_pil_resized)
                    else:
                        # (D, H, W) 多帧/3D 图像，用 scipy.zoom
                        # 计算缩放因子：只缩放 H, W，D 保持不变
                        zoom_factors = (
                            1.0,  # D 不缩放
                            target_size[1] / img_np.shape[1],  # H
                            target_size[0] / img_np.shape[2]   # W （注意 PIL 是 (W, H)）
                        )
                        img_np_resized = zoom(img_np, zoom_factors, order=1)  # 双线性插值
                        logger.debug(f"使用 scipy.zoom 缩放 3D 图像: {image_path}, 原形状: {original_shape} -> {img_np_resized.shape}")
                else:
                    logger.warning(f"跳过不支持的维度: {img_np.shape}, 路径: {image_path}")
                    continue

            except Exception as e:
                logger.error(f"缩放图像时出错: {image_path}, 形状: {img_np.shape}, 错误: {e}", exc_info=True)
                continue

            if img_np_resized is None or img_np_resized.size == 0:
                logger.warning(f"跳过文件，因为缩放后的图像数组为空：{image_path}")
                continue

            unique_id = hashlib.md5(image_path.encode('utf-8')).hexdigest()
            output_filepath = os.path.join(output_dir, f"{unique_id}.npz")
            try:
                np.savez_compressed(output_filepath, img=img_np_resized)
            except Exception as e:
                logger.error(f"保存npz文件时发生错误: {output_filepath}, 错误: {e}", exc_info=True)
                continue
            saved_paths_batch.append(output_filepath)
        logger.info(f"--- 批次 {i + 1} 处理完毕 ---")
        #  --- 新增逻辑：每批次保存一次 CSV ---
        if saved_paths_batch:
            df_batch = pd.DataFrame(saved_paths_batch, columns=['saved_npz_path'])
            try:
                # 检查文件是否存在，决定是否写入表头
                file_exists = os.path.exists(output_csv_path)
                df_batch.to_csv(output_csv_path, mode='a', header=not file_exists, index=False)
                logger.info(f"批次 {i+1} 的 {len(saved_paths_batch)} 个文件路径已追加保存到: {output_csv_path}")
            except Exception as e:
                logger.error(f"保存批次 {i+1} 文件路径CSV时发生错误: {e}", exc_info=True)
        else:
            logger.warning(f"批次 {i+1} 没有文件成功保存，未写入CSV。")
    logger.info("所有图像已处理完毕。")


if __name__ == "__main__":
    input_csv_path = '/home/yyi/skipped_paths_1.csv'
    output_npz_directory = '/data/truenas_B2/Dataset/001_6yXray/pretrain_1022'

    BATCH_SIZE = 5000
    START_BATCH = 0

    TARGET_SIZE = (1022, 1022)

    process_images_in_batches(input_csv_path, output_npz_directory,
                              batch_size=BATCH_SIZE,
                              start_batch=START_BATCH,
                              target_size=TARGET_SIZE)

