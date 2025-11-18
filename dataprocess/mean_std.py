import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 支持的图像扩展名（PIL 能处理的常见格式）
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')

def calculate_rgb_mean_and_std_safe(image_folder_path, target_size=None):
    """
    安全地计算大图像数据集的 R, G, B 三通道均值和标准差。
    采用增量计算，跳过无法打开的图像，避免内存溢出。
    
    Args:
        image_folder_path (str): 包含图像文件的文件夹路径。
        target_size (tuple or int, optional): 若为 tuple (H, W) 或 int（正方形缩放），则统一 resize。

    Returns:
        tuple: (rgb_mean_list, rgb_std_list)
    """
    
    # 收集所有支持格式的图像路径（不区分大小写）
    image_paths = []
    for root, _, files in os.walk(image_folder_path):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, file))
    
    # 去重（防止大小写重叠）
    image_paths = list(set(image_paths))
    image_paths.sort()  # 可选：保证顺序一致

    if not image_paths:
        print(f"调试：检查路径 {image_folder_path} 中的文件:")
        for root, _, files in os.walk(image_folder_path):
            for file in files[:5]:  # 只显示前5个文件
                print(f"  - {file} (扩展名: {Path(file).suffix})")
            break  # 只显示根目录
        print(f"\n在路径 {image_folder_path} 中没有找到支持的图像文件")
        return None, None

    transform_list = []
    if target_size is not None:
        transform_list.append(transforms.Resize(target_size))
    transform_list.append(transforms.ToTensor())
    data_transform = transforms.Compose(transform_list)

    # 初始化累加器（使用 float64 避免精度损失）
    sum_per_channel = torch.zeros(3, dtype=torch.float64)
    sum_sq_per_channel = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0  # 动态累计成功加载图像的总像素数（按通道）

    print(f"总共找到 {len(image_paths)} 个候选图像文件。开始处理...")
    
    valid_count = 0
    for path in tqdm(image_paths, desc="Processing images"):
        try:
            img = Image.open(path).convert('RGB')
            tensor = data_transform(img)  # shape: (C, H, W), dtype: float32 in [0,1]
            
            pixels_in_image = tensor.size(1) * tensor.size(2)
            total_pixels += pixels_in_image
            valid_count += 1

            sum_per_channel += tensor.sum(dim=[1, 2], dtype=torch.float64)
            sum_sq_per_channel += (tensor ** 2).sum(dim=[1, 2], dtype=torch.float64)

        except Exception as e:
            print(f"\n[Warning] 跳过无效图像: {os.path.basename(path)} | 错误: {e}")
            continue

    if total_pixels == 0:
        print("没有成功加载任何有效图像。")
        return None, None

    print(f"成功处理 {valid_count} / {len(image_paths)} 张图像。")

    # 计算均值和标准差
    mean_tensor = (sum_per_channel / total_pixels).to(torch.float32)
    var_tensor = (sum_sq_per_channel / total_pixels) - (mean_tensor ** 2)
    var_tensor = torch.clamp(var_tensor, min=0.0)
    std_tensor = torch.sqrt(var_tensor).to(torch.float32)

    return mean_tensor.tolist(), std_tensor.tolist()


# --- 示例用法 ---
image_dir = '/data/truenas_B2/Dataset/001_6yXray/bone_dataset/MURA-v1.1/train/'
target_size = 512  # 可为 int 或 (H, W)

dataset_means, dataset_stds = calculate_rgb_mean_and_std_safe(image_dir, target_size)

if dataset_means is not None:
    print("\n--- ✅ 结果 ---")
    print(f"归一化 RGB 均值: {dataset_means}")
    print(f"归一化 RGB 标准差: {dataset_stds}")
    
    print("\n--- DINOv3 配置建议 ---")
    print("rgb_mean:")
    for m in dataset_means:
        print(f"  - {m:.6f}")
    print("rgb_std:")
    for s in dataset_stds:
        print(f"  - {s:.6f}")