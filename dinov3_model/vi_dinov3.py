import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel



from collections import OrderedDict

def convert_dinov3_teacher_to_hf_state_dict(
    teacher_state_dict: dict, 
    model_dim: int = 1024
) -> OrderedDict:
    """
    将 DINOv3 训练代码（Teacher 模型）的状态字典键名 
    转换为 Hugging Face Transformers 库（DINOv3ViTModel）的键名。
    
    Args:
        teacher_state_dict: 从 .pth 文件中提取的 Teacher 模型的 PyTorch 状态字典。
        model_dim: 模型特征维度 (ViT-L 为 1024)。

    Returns:
        OrderedDict: 适用于 Hugging Face 模型的重命名状态字典。
    """
    state_dict_renamed = OrderedDict()

    for k, v in teacher_state_dict.items():
        # 1. 移除顶层前缀
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('teacher.'):
            k = k[8:] 
        
        # 2. 移除 'backbone.' 前缀并处理 Embedding 层的核心键名映射
        if k.startswith('backbone.'):
            k_clean = k[9:]
            
            # 2.1. Patch Embedding 投影层
            if k_clean.startswith('patch_embed.proj'):
                k = k_clean.replace('patch_embed.proj', 'embeddings.patch_embeddings')
            
            # 2.2. 特殊 Token 映射
            elif k_clean == 'cls_token':
                k = 'embeddings.cls_token'
            elif k_clean == 'mask_token':
                # --- 修复 Mask Token 维度差异 ---
                if v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == model_dim:
                    v = v.view(1, 1, model_dim)
                k = 'embeddings.mask_token'
                
            elif k_clean == 'storage_tokens':
                k = 'embeddings.register_tokens'
                
            # 2.3. Transformer Blocks: 'blocks.X' -> 'layer.X'
            elif k_clean.startswith('blocks.'):
                parts = k_clean.split('.')
                if parts[1].isdigit():
                    layer_index = parts[1]
                    new_prefix = f'layer.{layer_index}'
                    k = new_prefix + '.' + '.'.join(parts[2:])
                else:
                    k = k_clean 
            
            else:
                k = k_clean 
        
        # 3. Transformer Block 内部命名转换
        
        # 3.1. Attention QKV 权重的拆分和重命名 (保持不变)
        if 'attn.qkv.' in k:
            if 'weight' in k:
                dim = v.shape[0] // 3
                q, k_t, v_t = v.chunk(3, dim=0)
                k_base = k.replace('.attn.qkv.weight', '.attention')
                state_dict_renamed[k_base + '.q_proj.weight'] = q
                state_dict_renamed[k_base + '.k_proj.weight'] = k_t
                state_dict_renamed[k_base + '.v_proj.weight'] = v_t
                continue 
                
            elif 'bias' in k:
                dim = v.shape[0] // 3
                q_b, k_b, v_b = v.chunk(3, dim=0)
                k_base = k.replace('.attn.qkv.bias', '.attention')
                state_dict_renamed[k_base + '.q_proj.bias'] = q_b
                state_dict_renamed[k_base + '.k_proj.bias'] = k_b
                state_dict_renamed[k_base + '.v_proj.bias'] = v_b
                continue 

        # 3.2. Attention 输出投影层
        if 'attn.proj.' in k:
            k = k.replace('.attn.proj.', '.attention.o_proj.')
            
        # 3.3. MLP 层（前馈网络 FFN）
        if '.mlp.fc1.' in k:
            k = k.replace('.mlp.fc1.', '.mlp.up_proj.')
        if '.mlp.fc2.' in k:
            k = k.replace('.mlp.fc2.', '.mlp.down_proj.')
            
        # 3.4. Layer Scale 核心修正 (针对缺失的 lambdaX)
        # 原始 DINOv3 通常是 ls1/ls2 或 ls1.weight/ls2.weight
        # HF ViT 期望: layer_scale1.lambda1 / layer_scale2.lambda2
        
        # 修正 ls1 -> layer_scale1.lambda1
        if '.ls1' in k:
            # 移除 ls1 的可能后缀 (如 .weight)
            k_base = k.replace('.ls1.weight', '.ls1').replace('.ls1', '.layer_scale1.lambda1')
            k = k_base
            
        # 修正 ls2 -> layer_scale2.lambda2
        if '.ls2' in k:
            # 移除 ls2 的可能后缀 (如 .weight)
            k_base = k.replace('.ls2.weight', '.ls2').replace('.ls2', '.layer_scale2.lambda1') # 注意：HF 可能是 lambda1
            k = k_base

        # 3.5. 修复 Layer Norm 命名 (如果存在差异)
        # DINOv3: norm1/norm2
        # HF ViT: layernorm_before/layernorm_after
        # ViT 大多使用 norm1/norm2，这里主要确保它们没有被错误地当成 Layer Scale
        
        # 如果键没有被 QKV 逻辑跳过，则将其添加到重命名字典中
        state_dict_renamed[k] = v

    return state_dict_renamed
# --- 核心函数：合并所有步骤，不绘制对比图 ---
def run_dinov3_pca_pipeline(
    pretrained_model_name: str,
    image_paths: list[str],
    save_dir: str,
    image_size: int,
    local_name: str,
    local_weights: bool = False,
    local_weights_path: str = ""
):
    # 1. 配置和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output directory: {save_dir}")

    # 2. 模型加载
    print("Loading DINOv3 model and processor...")
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map=device if device.type == 'cuda' else None 
    )

    # 2.1. 加载本地权重 (如果需要)
    if local_weights:
        if os.path.exists(local_weights_path):
            print(f"Loading local weights from {local_weights_path}...")
            checkpoint = torch.load(local_weights_path, map_location='cpu')
            if 'teacher' in checkpoint:
                state_dict = checkpoint['teacher']
                print("🔑 Checkpoint found 'teacher' key. Using teacher model state dict.")
            # 兼容一些只有 'model' 键的结构
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("🔑 Checkpoint found 'model' key. Using model state dict.")
            else:
                state_dict = checkpoint
                print("🔑 Checkpoint is flat. Using top-level state dict.")
        
        state_dict_renamed = convert_dinov3_teacher_to_hf_state_dict(
                state_dict, 
                model_dim=1024 
            )

        try:
            # 使用 strict=False 允许忽略不需要的键（如分类头或注册token）
            load_info = model.load_state_dict(state_dict_renamed, strict=False)
            print("Local Teacher weights loaded successfully.")
            # 打印缺失和不匹配的键，用于调试
            if load_info.unexpected_keys:
                 print(f"⚠️ Warning: Unexpected keys (ignored): {load_info.unexpected_keys[:5]}...")
            if load_info.missing_keys:
                 print(f"⚠️ Warning: Missing keys (using HF weights): {load_info.missing_keys[:5]}...")
                 
        except Exception as e:
            print(f"❌ Error loading local state dict: {e}")
            print("请检查键名转换逻辑或模型结构是否匹配。")
            return
    
    model.to(device).eval()

    # 3. 图像加载和预处理
    original_images = []
    images_to_process = []
    image_filenames = []
    
    # 3.1. 批量加载图像和文件名
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            
            # 提取文件名（不含路径和扩展名）
            filename_without_ext = os.path.splitext(os.path.basename(path))[0]
            
            original_images.append(image)
            images_to_process.append(image)
            image_filenames.append(filename_without_ext)
        except FileNotFoundError:
            print(f"❌ Error: Image file not found at {path}")
        except Exception as e:
            print(f"❌ Error loading image {path}: {e}")

    if not images_to_process:
        print("No valid images to process. Exiting.")
        return

    # 3.2. 预处理 (修正 size 参数格式错误)
    print(f"Processing {len(images_to_process)} images to size {image_size}x{image_size}...")
    inputs = processor(
        images=images_to_process, 
        # 修正：使用字典格式 {"height": size, "width": size}
        size={"height": image_size, "width": image_size}, 
        return_tensors="pt"
    ).to(device)

    # 4. 特征提取
    all_features_list = []
    with torch.inference_mode():
        outputs = model(**inputs) 
        # DINOv3 ViT 模型跳过 CLS token (1) 和 Register tokens (4)
        patch_tokens = outputs.last_hidden_state[:, 5:] # [batch_size, N_patches, D_feature]
        
        for single_image_tokens in patch_tokens:
            # [N_patches, D_feature] -> numpy
            features = single_image_tokens.detach().cpu().numpy()
            all_features_list.append(features)
            
    # 5. PCA 计算和保存
    print("Computing PCA and saving results...")
    
    for i, (features_numpy, filename) in enumerate(tqdm(
        zip(all_features_list, image_filenames), # 不需要 original_image
        total=len(all_features_list), 
        desc="PCA and Saving"
    )):
        try:
            num_patches = features_numpy.shape[0]
            
            # 假设输入是 IMAGE_SIZE x IMAGE_SIZE，PATCH_SIZE=16
            side_patches = image_size // 16 
            h_patches, w_patches = side_patches, side_patches
            
            if h_patches * w_patches != num_patches:
                 # 非标准输入尺寸时的安全检查
                 side_patches = int(np.round(np.sqrt(num_patches)))
                 h_patches, w_patches = side_patches, side_patches
                 
                 if h_patches * w_patches != num_patches:
                     print(f"⚠️ Warning: Image {i+1} Patch count ({num_patches}) does not match expected {image_size//16}x{image_size//16}. Using best-guess square.")
                     
            if features_numpy.shape[0] < 3:
                print(f"⚠️ Warning: Image {i+1} has too few patches ({features_numpy.shape[0]}). Skipping PCA.")
                continue

            # 5.1. 拟合 PCA
            pca = PCA(n_components=3, whiten=True)
            projected_features = pca.fit_transform(features_numpy)
            
            # 5.2. 重塑和颜色增强
            # 从 numpy 转换到 torch
            projected_image = torch.from_numpy(projected_features).view(h_patches, w_patches, 3)
            # 颜色增强 (Sigmoid * 2.0)
            projected_image = torch.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1) # [3, h, w]

            # 5.3. 保存 PCA 图像
            pca_image_numpy = (projected_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pca_image_pil_low_res = Image.fromarray(pca_image_numpy)

            w_orig, h_orig = original_images[i].size
            # 缩放 PCA 图像到原图尺寸
            pca_image_pil_resized = pca_image_pil_low_res.resize(
                (w_orig, h_orig), 
                resample=Image.Resampling.BICUBIC # 使用高质量插值
            )
            # 构造保存路径
            pca_filename = f"{filename}_{local_name}_pca.png" 
            pca_image_path = os.path.join(save_dir, pca_filename)
            pca_image_pil_resized.save(pca_image_path)
            # 打印保存信息
            print(f"\n✅ Saved PCA to: {pca_image_path}")
            
        except Exception as e:
            print(f"\n❌ Error processing image {i+1} ({filename}): {e}")

    print("--- DINOv3 PCA Pipeline Finished ---")


# --- 主程序执行：使用整合后的函数 ---
if __name__ == "__main__":
    # --- 1. 配置和路径 ---
    pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    LOCAL_Weights = True
    local_weights_path = "/data/truenas_B2/yyi/bone_logs_512/eval/training_92999/teacher_checkpoint.pth"
    IMAGE_PATHS = [
        "/home/yyi/images/images/cat.jpg",
        "/home/yyi/images/images/chest.png",
        "/home/yyi/images/images/foot.png",
        "/home/yyi/images/images/hand.png",
        "/home/yyi/images/images/Knee.png",
        "/home/yyi/images/images/pelvis.jpeg",
        "/home/yyi/images/images/right_limb.png",
        "/home/yyi/images/images/spine.png"
    ]
    IMAGE_SIZE = 1024
    LOCAL_NAME = "boneDinov3_92999"
    SAVE_DIR = f"/home/yyi/images/pca_image/{LOCAL_NAME}"
    
    # 确保图像路径不是默认的占位符
    if any("path/to/your/" in p for p in IMAGE_PATHS):
        print("\n!!! 错误：请先将 IMAGE_PATHS 列表中的路径替换为您本地图像的实际路径 !!!\n")
    else:
        # 执行整合后的管道
        run_dinov3_pca_pipeline(
            pretrained_model_name=pretrained_model_name,
            image_paths=IMAGE_PATHS,
            save_dir=SAVE_DIR,
            image_size=IMAGE_SIZE,
            local_name=LOCAL_NAME,
            local_weights=LOCAL_Weights,
            local_weights_path=local_weights_path
        )