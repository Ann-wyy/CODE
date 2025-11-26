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
        
        # 修正 ls1 -> layer_scale1.lambda1
        if '.ls1' in k:
            # 移除 ls1 的可能后缀 (如 .weight)
            k_base = k.replace('.ls1.weight', '.ls1').replace('.ls1', '.layer_scale1.lambda1')
            k_base = k_base.replace('.gamma', '')
            if k_base != k:
                k = k_base
            
        # 修正 ls2 -> layer_scale2.lambda2
        if '.ls2' in k:
            # 移除 ls2 的可能后缀 (如 .weight)
            k_base = k.replace('.ls2.weight', '.ls2').replace('.ls2', '.layer_scale2.lambda1') # 注意：HF 可能是 lambda1
            k_base = k_base.replace('.gamma', '')
            if k_base != k:
                k = k_base

        
        # 如果键没有被 QKV 逻辑跳过，则将其添加到重命名字典中
        state_dict_renamed[k] = v

    return state_dict_renamed


def preprocess_labels_and_setup_datasets(TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH, LABEL_COLUMNS,IMAGE_PATH_COLUMN, logger):
    """
    加载CSV文件，对多任务标签进行编码，处理未知标签为-1，
    并为二分类任务将少数类编码为 1。
    """
    # 1. 加载数据
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        val_df = pd.read_csv(VAL_CSV_PATH)
        test_df = pd.read_csv(TEST_CSV_PATH)
    except FileNotFoundError as e:
        logger.error(f"加载CSV文件失败: {e}")
        raise

    for df in [train_df, val_df, test_df]:
        df.dropna(subset=[IMAGE_PATH_COLUMN], inplace=True)
    label_encoders = {}
    num_classes_dict = {}

    from sklearn.preprocessing import LabelEncoder
    
    for col in LABEL_COLUMNS:
        le = LabelEncoder()
        train_labels_str = train_df[col].astype(str).copy()
        ignore_mask_train = (train_labels_str == '-1')
        train_labels_for_fit = train_labels_str.loc[~ignore_mask_train]
        le.fit(train_labels_for_fit)
        encoded_labels_train = pd.Series(le.transform(train_labels_for_fit), 
                                         index=train_labels_for_fit.index)
        # 拟合训练集
        new_train_labels = pd.Series(-1, index=train_df.index, dtype=np.int64)
        new_train_labels.loc[~ignore_mask_train] = encoded_labels_train.values
        train_df[col] = new_train_labels.astype(int)

        # 转换 val/test
        for name, df in [("Val", val_df), ("Test", test_df)]:
            df_labels_str = df[col].astype(str).copy()
            # 找到要忽略的标签
            ignore_mask_df = (df_labels_str == '-1')
            
            # 找到需要转换的有效标签
            df_labels_to_transform = df_labels_str.loc[~ignore_mask_df]
            new_df_labels = pd.Series(-1, index=df.index,dtype=np.int64)
            try:
                encoded_labels_df = pd.Series(le.transform(df_labels_to_transform), 
                                             index=df_labels_to_transform.index)
                new_df_labels.loc[~ignore_mask_df] = encoded_labels_df
            except ValueError as e:
                # 如果遇到训练集未见过的标签，则进行映射
                logger.warning(f"{name} set contains unseen labels in '{col}': {e}. Mapping unknown to -1.")
                label_to_idx = {label: idx for idx, label in enumerate(le.classes_)}
                mapped_labels = df_labels_to_transform.map(label_to_idx).fillna(-1).astype(int)
                new_df_labels.loc[~ignore_mask_df] = mapped_labels
            df[col] = new_df_labels.astype(int)

        label_encoders[col] = le
        # 类别数是 fit_transform 之后 le.classes_ 的长度
        num_classes_dict[col] = len(le.classes_) 
        logger.info(f"任务 '{col}': 类别 {list(le.classes_)} → 编码 [0..{len(le.classes_)-1}], 共 {len(le.classes_)} 类")

        # --- 少数类反转逻辑 ---
        if num_classes_dict[col] == 2:
            encoded_labels = train_df[col].loc[train_df[col] != -1]
            if len(np.unique(encoded_labels)) < 2:
                 logger.warning(f"任务 '{col}' 的有效训练样本中少于 2 个类别，跳过少数类反转。")
            else:
                counts = encoded_labels.value_counts()
                
                if len(counts) == 2:
                    minority_encoded_value = counts.idxmin() # 频率最小的编码值
                    
                    if minority_encoded_value == 0:
                        logger.warning(f"任务 '{col}' 的少数类被编码为 0。正在反转标签 0 <-> 1...")
                        mapping = {0: 1, 1: 0}
                        
                        for df in [train_df, val_df, test_df]:
                            # 仅对 0 和 1 进行映射，-1 保持不变
                            df[col] = df[col].replace(mapping)
                    else:
                        logger.info(f"任务 '{col}' 的少数类已被正确编码为 1。无需反转。")
        # --- 少数类反转逻辑结束 ---
    return train_df, val_df, test_df, num_classes_dict