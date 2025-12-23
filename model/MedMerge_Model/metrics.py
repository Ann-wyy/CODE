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
    valid_indices = (all_labels >= 0)
    
    # 过滤数据
    valid_labels = all_labels[valid_indices]
    valid_preds = all_preds[valid_indices]
    valid_probs = all_probs[valid_indices]
    if len(valid_labels) == 0:
        logger.warning(f"任务 {task_name} 中没有有效标签 (>=0) 的样本，跳过指标计算。")
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auroc': 0.0, 'auprc': 0.0}
    metrics['accuracy'] = float(accuracy_score(valid_labels, valid_preds))
    target_names = list(range(num_classes))
    metrics['precision'] = float(precision_score(valid_labels, valid_preds, labels=target_names,average='macro', zero_division=0))
    metrics['recall'] = float(recall_score(valid_labels, valid_preds, labels=target_names,average='macro', zero_division=0))
    metrics['f1'] = float(f1_score(valid_labels, valid_preds, labels=target_names,average='macro', zero_division=0))

    logger.info(f"--- {mode.upper()} 结果 (类别数: {num_classes}) ---")
    logger.info(f"整体准确率 (Accuracy): {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"整体精确率 (Precision): {metrics['precision'] * 100:.2f}%")
    logger.info(f"整体召回率 (Recall): {metrics['recall'] * 100:.2f}%")
    logger.info(f"整体 F1-Score: {metrics['f1'] * 100:.2f}%")

    # --- AUROC ---
    try:
        if num_classes == 2:
            if len(np.unique(valid_labels)) < 2:
                raise ValueError("有效二分类样本只含有一个类别。")
            # 二分类：用正类概率
            auroc = roc_auc_score(valid_labels, valid_probs[:, 1]) # 使用过滤后的数据
            auprc = average_precision_score(valid_labels, valid_probs[:, 1])
        else:
            classes = list(range(num_classes))
            y_true_bin = label_binarize(valid_labels, classes=classes)
            auroc = roc_auc_score(y_true_bin, valid_probs, multi_class='ovr')
            auprc = average_precision_score(y_true_bin, valid_probs, average='macro')
        metrics['auprc'] = float(auprc)
        metrics['auroc'] = float(auroc)
        logger.info(f"整体 AUPRC: {metrics['auprc'] * 100:.2f}%")
        logger.info(f"整体 AUROC: {metrics['auroc'] * 100:.2f}%")
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
    task_probs = {task: [] for task in task_names} 
    
    device = torch.device(device)

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            
            # --- 适配点 1: 修改解包方式 (从元组改为字典) ---
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_dict = batch["labels"] # 后面会 individually to device
            
            
            predictions_dict = model(pixel_values, input_ids, attention_mask)
            
            combined_loss = 0

            for task_name in task_names:
                labels = labels_dict[task_name].to(device)
                predictions = predictions_dict[task_name]
                num_cls = num_classes_dict.get(task_name)
                task_criterion = criterion_dict[task_name]

                valid_mask = (labels != -1)
                valid_count = valid_mask.sum()
                
                # 计算 Loss
                if valid_count > 0:
                    valid_labels = labels[valid_mask]
                    valid_predictions = predictions[valid_mask]

                    if num_cls and num_cls > 2:
                        task_loss = task_criterion(valid_predictions, valid_labels.long())
                    else:
                        target = valid_labels.float().view(-1, 1)
                        task_loss = task_criterion(valid_predictions, target)
                    combined_loss += task_loss
                
                # --- 适配点 3: 概率计算逻辑微调 ---
                if num_cls and num_cls > 2:
                    probabilities = torch.softmax(predictions, dim=1)
                else:
                    # predictions 形状为 [N, 1]
                    probs_pos = torch.sigmoid(predictions).squeeze(1) # [N]
                    # 组合成 [N, 2] 以适配 calculate_metrics 的 all_probs[:, 1]
                    probabilities = torch.stack([1 - probs_pos, probs_pos], dim=1)

                task_probs[task_name].extend(probabilities.cpu().tolist())
                task_labels[task_name].extend(labels.cpu().tolist())
                
            total_combined_loss += combined_loss.item() if isinstance(combined_loss, torch.Tensor) else combined_loss

    # 计算评估指标 (使用你提供的 calculate_metrics)
    task_metrics = {}
    for task_name in task_names:
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
            task_metrics[task_name] = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auroc': 0.0, 'auprc': 0.0}

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
    for key in [ 'auprc','auroc', 'accuracy']:
        values = all_individual_metrics[key]
        if values:
            avg_val = np.mean(values)
            average_metrics[key] = avg_val
            writer.add_scalar(f'{stage}_Aggregated/AVERAGE_{key.upper()}', avg_val, step)
            logger.info(f"Average {key.upper()}: {avg_val:.4f}")
        else:
            logger.warning(f"No valid {key.upper()} values found for averaging.")
            average_metrics[key] = float('nan')
    

    