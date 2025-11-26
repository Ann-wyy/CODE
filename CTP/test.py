import nibabel as nib
import numpy as np
import os

def ctp_gz_to_zyx_list(file_path: str) -> list:
    """
    加载 NIfTI .gz 文件 (CTP数据)，并将其转换为 (T, Z, Y, X) 格式的 3D 图像列表。
    默认 NIfTI 原始数据顺序为 (X, Y, Z, T)。

    Args:
        file_path: NIfTI (.nii.gz) 文件的完整路径。

    Returns:
        list: 包含 T 个时间点 3D 图像 (Z, Y, X) 的 NumPy 数组列表。
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件未找到: {file_path}")
        return []

    try:
        # 1. 加载图像并获取数据
        img = nib.load(file_path)
        data_4d = img.get_fdata()
        
        original_shape = data_4d.shape
        original_ndim = data_4d.ndim
        
        print(f"✅ 文件加载成功。原始形状: {original_shape}")

        if original_ndim < 3:
            print("❌ 错误: 数据维度小于 3D。")
            return []
        
        # 2. 检查并确保有时间维度 (T)
        if original_ndim == 3:
            # 3D (X, Y, Z) -> 4D (X, Y, Z, 1)
            data_4d = data_4d[..., np.newaxis]
            T = 1
        elif original_ndim == 4:
            # 4D (X, Y, Z, T)
            T = data_4d.shape[-1]
        else:
            print(f"⚠️ 警告: 维度数 > 4D ({original_ndim}D)，仅使用前 4 维。")
            # 假设时间维度在最后
            data_4d = data_4d[..., :4] 
            T = data_4d.shape[-1]

        # 3. 维度重排：(X, Y, Z, T) -> (T, Z, Y, X)
        
        # a. 交换 T 和其他维度：(X, Y, Z, T) -> (T, X, Y, Z)
        data_4d_permuted = np.transpose(data_4d, (3, 0, 1, 2))
        
        # b. 拆分并转换为 (Z, Y, X)
        # 从 (T, X, Y, Z) 中取出每个 3D 图像 (X, Y, Z)
        image_list_xyz = [data_4d_permuted[t] for t in range(T)]
        
        # 将每个 (X, Y, Z) 转换为要求的 (Z, Y, X)
        image_list_zyx = [np.transpose(img_3d, (2, 1, 0)) for img_3d in image_list_xyz]
        
        # 4. 结果报告
        if image_list_zyx:
            final_shape = image_list_zyx[0].shape
            print(f"--- 转换完成 ---")
            print(f"总时间点 (T): {len(image_list_zyx)}")
            print(f"每个 3D 图像的形状 (Z, Y, X): {final_shape}")
            
        return image_list_zyx

    except nib.filebased.FileBasedImageError:
        print("❌ 错误: Nibabel 无法识别文件格式。它可能不是 NIfTI (.nii.gz)。")
        return []
    except Exception as e:
        print(f"❌ 发生意外错误: {e}")
        return []

# --- 🎯 使用示例 ---
# 替换为您的文件路径
ctp_file_path = '/data/truenas_36T/HeadStroke_Yan/a106369160/CTP/Head Volume Perfusion_5.0 x 5.0_301.nii.gz' 

# 运行函数
image_list = ctp_gz_to_zyx_list(ctp_file_path)

if image_list:
    # 示例: 访问第一个时间点的图像
    first_image_zyx = image_list[0]
    # 您现在可以将 first_image_zyx 用于进一步处理 (例如，切片、显示等)
    print(f"\n第一个时间点的图像类型: {type(first_image_zyx)}")