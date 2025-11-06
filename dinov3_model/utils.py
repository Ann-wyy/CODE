import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoImageProcessor, AutoModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import default_collate
import logging
import time
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T


def set_seed(seed):
    """设置所有必要的随机种子"""
    # Python 内建的随机数
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # GPU (CUDA) 种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
        # 强制 CUDA 禁用非确定性算法，确保结果完全一致
        # 但可能会轻微降低一些性能
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

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


