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


# --- 评估函数 ---
def calculate_metrics(
    all_labels: List[int],
    all_probs: List[List[float]],  # shape: [N, C]
    num_classes: int,
    task_name: str,
    mode: str,
    logger: logging.Logger
)  -> Dict[str, float]:
    """计算二分类或多分类模型的评估指标。"""
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    metrics = {}
    metrics['accuracy'] = float(accuracy_score(all_labels, all_preds))
    metrics['precision'] = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    metrics['recall'] = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    metrics['f1'] = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))

    logger.info(f"--- {mode.upper()} 结果 (类别数: {num_classes}) ---")
    logger.info(f"整体准确率 (Accuracy): {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"整体精确率 (Precision): {metrics['precision'] * 100:.2f}%")
    logger.info(f"整体召回率 (Recall): {metrics['recall'] * 100:.2f}%")
    logger.info(f"整体 F1-Score: {metrics['f1'] * 100:.2f}%")

    # --- AUROC ---
    try:
        if num_classes == 2:
            # 二分类：用正类概率
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
            auprc = average_precision_score(all_labels, all_probs[:, 1])
        else:
            auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            auprc = average_precision_score(all_labels, all_probs, average='macro')  # 或 'weighted'
        metrics['auroc'] = float(auroc)
        metrics['auprc'] = float(auprc)
        logger.info(f"整体 AUROC: {metrics['auroc'] * 100:.2f}%")
        logger.info(f"整体 AUPRC: {metrics['auprc'] * 100:.2f}%")
    except Exception as e:
        logger.warning(f"计算 AUROC/AUPRC 失败 ({task_name}, {mode}): {e}")
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0

    return metrics

        
# --- 辅助函数：评估流程 ---
def evaluate(model, data_loader, criterion_dict, task_names, num_classes_dict, device, mode, logger):
    model.eval()
    total_combined_loss = 0

    task_labels = {task: [] for task in task_names}
    task_probs = {task: [] for task in task_names} # <--- 新增
    task_counts = {task: 0 for task in task_names}

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            
            pixel_values, labels_dict, img_paths = batch
            batch_size = pixel_values.size(0)
            pixel_values = pixel_values.to(device)

            predictions_dict = model(pixel_values)
            combined_loss = 0

            for task_name in task_names:
                labels = labels_dict[task_name].to(device)
                predictions = predictions_dict[task_name]
                num_cls = num_classes_dict.get(task_name)
                task_criterion = criterion_dict[task_name]
                
                # 计算损失
                if num_cls and num_cls > 2:
                    # 多分类任务
                    target = labels.long()
                    probabilities = torch.softmax(predictions, dim=1)
                else:
                    # --- 二分类：BCEWithLogitsLoss ---
                    target = labels.float().view(-1, 1) 
                    probs_pos = torch.sigmoid(predictions).squeeze(1)
                    probabilities = torch.stack([1 - probs_pos, probs_pos], dim=1) # 形状 [N, 2]
                task_criterion = criterion_dict[task_name] # 建议为每个任务使用正确的 criterion
                task_loss = task_criterion(predictions, target)
                combined_loss += task_loss
                
                # 累加损失和样本数
                task_counts[task_name] += batch_size
                
                # 收集预测和标签
                # 对于 CrossEntropyLoss，预测是 argmax
                task_probs[task_name].extend(probabilities.cpu().tolist())
                preds = (predictions.squeeze(1) > 0).long()
                task_labels[task_name].extend(labels.cpu().tolist())
            total_combined_loss += combined_loss.item()


    # 计算评估指标 (例如：Accuracy, F1 Score)
    task_metrics = {}
     # 假设已安装 sklearn
    
    for task_name in task_names:
        # 使用无偏的、不加权的评估
        true_labels = task_labels[task_name]
        probabilities = task_probs[task_name]
        
        if len(true_labels) > 0:
            num_cls = num_classes_dict[task_name]
            metrics = calculate_metrics(
                all_labels=true_labels,
                all_probs=probabilities,
                num_classes=num_cls,
                task_name=task_name,
                mode=f'{mode}_{task_name}',
                logger=logger
            )
            task_metrics[task_name] = metrics
        else:
            task_metrics[task_name] = {'accuracy': 0.0, 'f1': 0.0, 'auroc': float('nan'), 'auprc': float('nan')}

    # 返回所有结果
    return task_metrics


# --- 辅助函数：TensorBoard 记录 ---
def log_metrics_to_tensorboard(
    writer: SummaryWriter, 
    metrics_dict: Dict[str, Dict[str, float]], 
    step: int, 
    stage: str, 
    logger: logging.Logger,
):
    """
    将所有指标（accuracy, precision, recall, f1, auroc, auprc）按任务类型聚合后写入 TensorBoard。
    """
    all_task_names = list(metrics_dict.keys()) 
    independent_tasks = [task for task in all_task_names]

    # 所有独立子任务（用于总体平均）
    all_individual_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auroc': [],
        'auprc': []
    }

    logger.info(f"--- {stage} Epoch {step} 任务摘要指标 ---")

    # 2. 处理独立任务
    for task_name in independent_tasks:
        metrics = metrics_dict.get(task_name, {})
        if not metrics:
            continue
        for key in all_individual_metrics:
            val = metrics.get(key, float('nan'))
            if not np.isnan(val):
                writer.add_scalar(f'{stage}_Summary/{key.upper()}_{task_name}', val, step)
                all_individual_metrics[key].append(val)
    
    average_metrics = {}
    for key in ['auroc', 'auprc', 'accuracy']:
        values = all_individual_metrics[key]
        if values:
            avg_val = np.mean(values)
            average_metrics[key] = avg_val
            writer.add_scalar(f'{stage}_Aggregated/AVERAGE_{key.upper()}', avg_val, step)
            logger.info(f"Average {key.upper()}: {avg_val:.4f}")
        else:
            logger.warning(f"No valid {key.upper()} values found for averaging.")
            average_metrics[key] = float('nan')
    

    