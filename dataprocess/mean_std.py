import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from tqdm import tqdm  # 用于显示进度条

def calculate_rgb_mean_and_std_safe(image_folder_path, target_size=None):
    """
    安全地计算大图像数据集的 R, G, B 三通道均值和标准差。
    采用迭代/增量计算，避免 CUDA/CPU 内存溢出。
    
    Args:
        image_folder_path (str): 包含图像文件的文件夹路径。
        target_size (tuple, optional): 如果需要将所有图像resize到统一大小，例如 (224, 224)。
                                        None表示不进行resize。

    Returns:
        tuple: (rgb_mean_list, rgb_std_list)
    """
    
    # 查找所有图像路径
    image_paths = glob(os.path.join(image_folder_path, '*.png'))
    image_paths.extend(glob(os.path.join(image_folder_path, '*.jpg')))
    
    if not image_paths:
        print(f"在路径 {image_folder_path} 中没有找到支持的图像文件。请检查路径和文件格式。")
        return None, None

    # 定义预处理步骤
    transform_list = []
    if target_size:
        # Resize 操作
        transform_list.append(transforms.Resize(target_size))
        
    # ToTensor() 会将图像转换为 (C, H, W) 格式，并将像素值缩放到 [0.0, 1.0]
    transform_list.append(transforms.ToTensor())
    
    data_transform = transforms.Compose(transform_list)

    # --- 初始化核心累加器 (使用 CPU) ---
    # R, G, B 三个通道的累加器，初始化为 0.0
    # 注意：使用 float64 (double) 以避免长时间累加可能出现的精度问题
    sum_per_channel = torch.zeros(3, dtype=torch.float64)
    sum_sq_per_channel = torch.zeros(3, dtype=torch.float64)
    total_pixels_per_channel = 0 

    print(f"总共找到 {len(image_paths)} 张图像。开始增量计算...")
    
    # 使用 tqdm 显示进度
    for path in tqdm(image_paths, desc="Calculating Stats"):
        try:
            # 1. 加载图像并确保是 RGB 模式
            img = Image.open(path).convert('RGB') 
            
            # 2. 应用转换，得到 [0, 1] 范围的张量 (C, H, W)
            # 注意：张量默认在 CPU 上创建，保持在 CPU 上以节省 GPU 内存
            tensor = data_transform(img)
            
            # 3. 累计统计量
            
            # (H * W) 是每张图片每个通道的像素数量
            pixels_in_image = tensor.size(1) * tensor.size(2)
            
            if total_pixels_per_channel == 0:
                 # 只需要累加一次总像素数量（因为每张图的像素数都是 H*W）
                 total_pixels_per_channel = pixels_in_image * len(image_paths)

            # 累计总和：在 H 和 W 维度上求和 (保留 C 维度)
            # 结果是一个 (3,) 的张量
            sum_per_channel += tensor.sum(dim=[1, 2])

            # 累计平方和：先平方，再在 H 和 W 维度上求和
            sum_sq_per_channel += (tensor ** 2).sum(dim=[1, 2])

        except Exception as e:
            # 打印错误，但不中断整个统计过程
            print(f"\n[Warning] 加载或处理图像 {os.path.basename(path)} 失败: {e}. 跳过此文件。")
            continue
            
    if total_pixels_per_channel == 0:
        print("所有图像加载失败或数据集中没有像素。")
        return None, None

    # --- 最终计算 ---
    
    # 均值 = 总和 / 总像素数
    # 转换为 float32 输出
    mean_tensor = (sum_per_channel / total_pixels_per_channel).to(torch.float32)
    
    # 标准差 = sqrt( (平方和 / 总像素数) - (均值 * 均值) )
    var_tensor = (sum_sq_per_channel / total_pixels_per_channel) - (mean_tensor ** 2)
    var_tensor = torch.clamp(var_tensor, min=0.0)  # 避免负值
    std_tensor = torch.sqrt(var_tensor).to(torch.float32)

    rgb_mean_list = mean_tensor.tolist()
    rgb_std_list = std_tensor.tolist()

    return rgb_mean_list, rgb_std_list

# --- 示例用法 ---

# !!! 请务必替换为你的图像文件夹路径 !!!
image_dir = '/data/truenas_B2/yyi/data/6y_bone_cancer' 

# 如果模型要求输入图像大小统一，这里可以设置。否则设置为 None
# 例如: target_size = (224, 224)
target_size = 512

# 执行计算
dataset_means, dataset_stds = calculate_rgb_mean_and_std_safe(image_dir, target_size)

if dataset_means is not None:
    print("\n--- ✅ 结果 ---")
    print(f"计算得到的**归一化 R G B 均值 (Mean)**: {dataset_means}")
    print(f"计算得到的**归一化 R G B 标准差 (Std)**: {dataset_stds}")
    
    print("\n--- DINOv3 配置更新建议 (请复制到你的 config.yaml) ---")
    print(f"rgb_mean:")
    print(f"  - {dataset_means[0]:.6f}")
    print(f"  - {dataset_means[1]:.6f}")
    print(f"  - {dataset_means[2]:.6f}")
    print(f"rgb_std:")
    print(f"  - {dataset_stds[0]:.6f}")
    print(f"  - {dataset_stds[1]:.6f}")
    print(f"  - {dataset_stds[2]:.6f}")