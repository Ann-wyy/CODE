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
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

# --- 配置参数 ---
MODEL_NAME = "facebook/dinov2-base" 
TARGET_IMAGE_SIZE = 518 # 图像目标尺寸
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
NUM_EPOCHS = 100
PATIENCE = 50 # 早停耐心值
Abstraction_dim = 1024
WEIGHT_DECAY = 1e-4

# 自动选择 GPU 设备，优先使用 cuda:0
DEVICE = "cuda:0"
if torch.cuda.is_available():
    # 使用用户定义的逻辑，但修正为可用设备
    device_id = 1 if torch.cuda.device_count() > 2 else 0
    DEVICE = f"cuda:{device_id}"
else:
    DEVICE = "cpu"


# 用户提供的文件路径
TRAIN_NAME = "BTXRD"
TRAIN_CSV_PATH = "/home/yyi/data/BTXRD_train.csv"
VAL_CSV_PATH = "/home/yyi/data/BTXRD_val.csv"
CSV_PATH = "/home/yyi/data/test_dataset/FracAtlas_dataset.csv" # 标签CSV文件路径
IMAGE_PATH_COLUMN = 'image_id' # CSV中包含图像相对路径的列名
LABEL_COLUMNS = ['tumor','benign','malignant'] # 您的所有标签列名
LOAD_LOCAL_CHECKPOINT = True
TEST_NAME = "boneDinov2_518"
TEST_NAME = f"{TEST_NAME}_{TRAIN_NAME}"
LOCAL_CHECKPOINT_PATH = "/home/yyi/weight/rad_dino.pth" # 替换为您的本地 .pth 文件路径
DATA_ROOT_CHECKPOINT = False
DATA_ROOT = "/data/truenas_B2/yyi/data/boneage-training-dataset" # 图像的根目录

# **新增：日志配置函数**
FILENAME = f"{TEST_NAME}_{TARGET_IMAGE_SIZE}_{time.strftime('%Y%m%d-%H%M%S')}"
LOG_FILENAME = os.path.join(f'logs/', f"{FILENAME}.log")


def setup_logging():
    """配置日志记录，输出到文件和控制台。"""
    if logging.getLogger().hasHandlers():
        return logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILENAME), # 写入文件
            logging.StreamHandler() # 输出到控制台
        ]
    )
    return logging.getLogger(__name__)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
logger = setup_logging() # 初始化全局日志记录器


# --- 自定义 PyTorch Dataset (处理多列分类标签) ---
class MultiTaskImageDataset(Dataset):
    """
    多任务图像数据集。支持多个标签列，并将每个多类别标签转换为 OvR (One-vs-Rest) 二分类任务。
    
    参数:
        root_dir (str): 图像文件所在的根目录。
        csv_path (str): 包含图像路径和标签的 CSV 文件路径。
        img_col (str): 图像路径在 CSV 中的列名。
        label_cols (List[str]): 原始标签列名列表。
        processor (AutoImageProcessor): Hugging Face 图像预处理器。
        size (int): 图像目标尺寸。
        logger (logging.Logger): 日志记录器。
        fitted_encoders (Dict[str, Any], optional): 预先拟合的编码器和任务信息，用于验证集。
                                                     键应包括 'label_encoders', 'ovr_tasks_map' 等。
    """
    def __init__(self, root_dir: str, csv_path: str, img_col: str, label_cols: List[str], 
                 processor: AutoImageProcessor, size: int, logger: logging.Logger, 
                 fitted_encoders: Dict[str, Any] = None, is_training: bool = False):
        
        self.root_dir = root_dir
        self.processor = processor
        self.size = size
        self.logger = logger
        self.label_cols = label_cols
        self.is_training = is_training
        
        try:
            self.df = pd.read_csv(csv_path)
            # 移除图像路径为空或缺失的行
            self.df.dropna(subset=[img_col], inplace=True)
        except Exception as e:
            logger.critical(f"无法读取或处理 CSV 文件 {csv_path}: {e}")
            raise

        self.img_col = img_col
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.ovr_tasks_map: Dict[str, List[str]] = {} # 原始任务名 -> 对应的 OvR 任务名列表
        self.num_classes_per_task: Dict[str, int] = {} # OvR 任务名 -> 类别数 (始终为 2)
        self.all_task_names: List[str] = [] # 所有 OvR 任务名

        if fitted_encoders is None:
            # 训练模式：拟合编码器并创建 OvR 任务
            self._fit_encoders()
        else:
            # 验证模式：使用拟合好的编码器和任务映射
            self.label_encoders = fitted_encoders['label_encoders']
            self.ovr_tasks_map = fitted_encoders['ovr_tasks_map']
            self.all_task_names = fitted_encoders['all_task_names']
            # 所有 OvR 任务都是二分类
            self.num_classes_per_task = {task: 2 for task in self.all_task_names}
            
            self._transform_labels()

        logger.info(f"数据集 {csv_path} 加载成功，总样本数: {len(self.df)}")
        logger.info(f"创建的 OvR 任务总数: {len(self.all_task_names)}")

        # 数据增强
        if self.is_training:
            # 训练集使用随机增强
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0,translate=(0.1, 0.1),shear=0),
            ])
            self.logger.info("✅ 训练集已启用数据增强。")
        else:
            # 验证集不使用随机增强
            self.transform = None
            self.logger.info("验证集未启用数据增强。")


    def _fit_encoders(self):
        """在训练集上拟合 LabelEncoder 并生成 OvR 任务。"""
        for col in self.label_cols:
            le = LabelEncoder()
            # 拟合并转换原始标签
            try:
                self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
            except Exception as e:
                self.logger.error(f"无法对列 {col} 进行 fit_transform: {e}")
                continue

            self.label_encoders[col] = le
            original_classes = le.classes_.tolist()
            ovr_tasks = []
            original_label_str = self.df[col].astype(str)
            num_unique_classes = len(original_classes)

            if num_unique_classes <= 2:
                ovr_task_name = col  # 保持原始任务名
                self.all_task_names.append(ovr_task_name)
                self.num_classes_per_task[ovr_task_name] = 2 # 仍是二分类
                self.ovr_tasks_map[col] = [col] # 映射到自身
                self.df[ovr_task_name] = self.df[col + '_encoded'] 
                self.logger.info(f"任务 '{col}' 为二分类 (类别数: {num_unique_classes})，跳过 OvR 转换。")
            else:
                for class_name in original_classes:
                    ovr_task_name = f"{col}_vs_{class_name}"
                    ovr_tasks.append(ovr_task_name)
                    
                    # 创建新的 OvR 标签列 (0 或 1)
                    # 1: 样本属于该类别， 0: 样本不属于该类别
                    self.df[ovr_task_name] = (original_label_str == class_name).astype(int)
                    
                    self.num_classes_per_task[ovr_task_name] = 2 # 始终为 2 (二分类)
                    self.all_task_names.append(ovr_task_name)
                
                self.ovr_tasks_map[col] = ovr_tasks
                self.logger.info(f"任务 '{col}' 为多分类 (类别数: {num_unique_classes})，已创建 {len(ovr_tasks)} 个 OvR 任务。")

            
        
        self.logger.info("编码器已拟合，OvR 任务已创建。")


    def _transform_labels(self):
        """在验证集上使用已拟合的 LabelEncoder 转换标签。"""
        for col in self.label_cols:
            if col not in self.label_encoders:
                self.logger.error(f"原始任务 {col} 在拟合编码器中缺失。跳过。")
                continue
            
            le = self.label_encoders[col]
            ovr_tasks = self.ovr_tasks_map.get(col, [])
            original_classes = le.classes_.tolist()
            num_unique_classes = len(original_classes)
            
            # 使用 transform 转换原始标签
            # 必须处理在训练集中未出现的类别（用 nan 或其他方式标记，通常 LabelEncoder 会报错）
            def transform_or_ignore(x):
                try:
                    return le.transform([x])[0]
                except ValueError:
                    # 如果验证集有训练集未见的类别，这里将其视为一个特殊的类别，但在 OvR 中它们都将是 0
                    return -1 
            
            self.df[col + '_encoded'] = self.df[col].astype(str).apply(transform_or_ignore)

            # 根据 OvR 任务映射创建 OvR 标签
            if num_unique_classes <= 2 and col in ovr_tasks:
                # 训练集将其视为二分类，则验证集直接使用编码后的标签
                self.df[col] = self.df[col + '_encoded']
            elif num_unique_classes > 2:
                # 多分类，根据 OvR 任务映射创建 OvR 标签
                original_label_str = self.df[col].astype(str)
                for class_name in original_classes:
                    ovr_task_name = f"{col}_vs_{class_name}"
                    if ovr_task_name in ovr_tasks:
                        # 1: 样本属于该类别， 0: 样本不属于该类别
                        self.df[ovr_task_name] = (original_label_str == class_name).astype(int)
        
        self.logger.info("标签已转换。")


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        """
        返回: 图像 Tensor, 标签字典 (OvR任务名 -> 标签Tensor), 图像文件路径
        注意: 为方便调试，返回路径，但 collate_fn 会过滤掉
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row[self.img_col])
        
        # 1. 尝试加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.logger.warning(f"无法加载或损坏的图像文件 {img_path}: {e}")
            # 返回 None 信号，由 custom_collate_fn 过滤
            return None, None, img_path 

        if self.transform is not None:
            image = self.transform(image)
        inputs = self.processor(images=image, size=self.size, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0) # [C, H, W]
        
        # 3. 提取 OvR 标签
        labels_dict = {}
        for task_name in self.all_task_names:
            # OvR 标签是 0 或 1，需要是 LongTensor
            label_value = row[task_name]
            labels_dict[task_name] = torch.tensor(label_value, dtype=torch.long)


        return pixel_values, labels_dict, img_path


# ====================================================================
# 2. custom_collate_fn 实现
# ====================================================================

def custom_collate_fn(batch: List[Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    自定义 collate_fn，用于处理 __getitem__ 返回 None 的情况 (如图像损坏)。
    过滤掉损坏的样本，然后将有效的样本打包成 Tensor。
    """
    # 过滤掉 None 样本 (由损坏图像引起)
    batch = [item for item in batch if item[0] is not None]
    
    if not batch:
        # 如果批次中所有样本都损坏，返回 None
        return None 

    # 1. 图像堆叠
    pixel_values = torch.stack([item[0] for item in batch])
    
    # 2. 标签字典处理
    # 提取第一个样本的标签字典中的所有任务名
    task_names = list(batch[0][1].keys())
    
    labels_dict = {}
    for task_name in task_names:
        # 收集该任务的所有标签并堆叠
        labels = [item[1][task_name] for item in batch]
        labels_dict[task_name] = torch.stack(labels).squeeze(0) # 堆叠后形状应为 [N]

    # 返回图像和标签字典
    img_paths = [item[2] for item in batch]
    return pixel_values, labels_dict, img_paths

# --- 自定义模型：DINOv3 + 多个分类头 ---

class DinoV3MultiTaskClassifier(nn.Module):
    """
    基于 DINOv3 主干网络，带有多任务分类头。
    """
    def __init__(self, model_name: str, task_num_classes: Dict[str, int]):
        super().__init__()

        self.task_names = list(task_num_classes.keys())

        # 1. 加载 DINOv3 主干网络并冻结
        self.backbone = AutoModel.from_pretrained(model_name)
        self.input_device = torch.device(DEVICE)
        feature_dim = self.backbone.config.hidden_size

        # ==================== 根据全局变量加载本地检查点 ====================
        global LOAD_LOCAL_CHECKPOINT, LOCAL_CHECKPOINT_PATH

        if LOAD_LOCAL_CHECKPOINT:
            if os.path.exists(LOCAL_CHECKPOINT_PATH):
                logger.info(f"Global flag LOAD_LOCAL_CHECKPOINT is True. Loading full model checkpoint from: {LOCAL_CHECKPOINT_PATH}")
                try:
                    checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location='cpu')

                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif isinstance(checkpoint, dict) and 'teacher' in checkpoint:
                        logger.info("Assuming DINOv2 official checkpoint format ('teacher' key).")
                        state_dict = checkpoint['teacher']
                    else:
                        state_dict = checkpoint

                    backbone_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('backbone.'):
                            new_key = k[len('backbone.'):]  # 去掉 'backbone.' 前缀
                            backbone_state_dict[new_key] = v
                    # 解决 mask_token 尺寸不匹配问题 (DINOv2/v3 特有)
                    if 'embeddings.mask_token' in backbone_state_dict:
                        mask_token = backbone_state_dict['embeddings.mask_token']
                        if mask_token.dim() == 2:  # [1, dim] -> 需要 [1, 1, dim]
                            logger.info("Reshaping mask_token from [1, dim] to [1, 1, dim]")
                            backbone_state_dict['embeddings.mask_token'] = mask_token.unsqueeze(1)
                        
                    missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
                    logger.info("✅ Backbone checkpoint loaded successfully.")

                except Exception as e:
                    logger.error(f"Error loading checkpoint at {LOCAL_CHECKPOINT_PATH}: {e}")
                    logger.warning("Continuing with the model initialized from Hugging Face and random classifiers.")
            else:
                logger.error(f"Warning: Global flag LOAD_LOCAL_CHECKPOINT is True, but file not found at: {LOCAL_CHECKPOINT_PATH}")
        else:
            logger.error("Global flag LOAD_LOCAL_CHECKPOINT is False. Initializing model from scratch (Hugging Face backbone + new classifiers).")

        '''
        abstraction_dropout = 0.5
        if Abstraction_dim >= feature_dim:
            # 如果降维维度大于或等于原始维度，则不降维，只保留一个Dropout层
            logger.warning(f"Abstraction dimension ({Abstraction_dim}) >= Feature dimension ({feature_dim}). Skipping feature abstraction/compression.")
            self.abstraction_layer = nn.Dropout(abstraction_dropout)
            final_feature_dim = feature_dim
        else:
            # 引入非线性降维和强正则化
            self.abstraction_layer = nn.Sequential(
                nn.Linear(feature_dim, Abstraction_dim),
                nn.GELU(),
                nn.Dropout(abstraction_dropout)
            )
            final_feature_dim = Abstraction_dim
            logger.info(f"Classifier Head uses abstraction: {feature_dim} -> {final_feature_dim} with Dropout {abstraction_dropout}")


        
        '''
        
        # 冻结主干网络
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 定义多个分类头
        self.classifiers = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            # self.classifiers[task_name] = nn.Sequential(nn.Dropout(0.5),nn.Linear(feature_dim, num_classes))
            self.classifiers[task_name] = nn.Linear(feature_dim, num_classes)
            for param in self.classifiers[task_name].parameters():
                 param.requires_grad = True
        
        # for param in self.abstraction_layer.parameters():
        #    param.requires_grad = True

        
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 运行主干网络（冻结）
        pixel_values = pixel_values.to(self.input_device)
        
        # 即使主干网络冻结，也要确保它在正确的设备上运行
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        # pooler_output 提取全局特征 (CLS Token)
        global_feature = outputs.last_hidden_state[:, 0, :]

        # 正则
        # abstracted_feature = self.abstraction_layer(global_feature)

        # 运行各个分类头
        logits = {}
        for task_name in self.task_names:
            logits[task_name] = self.classifiers[task_name](global_feature)

        return logits


# --- 评估函数 ---
def calculate_metrics_binarized(
        all_labels: List[int],
        all_preds: List[int],
        all_probs: List[List[float]],
        unique_labels: List[int],
        pos_label: Optional[int] = None,
        mode: str = 'test',
        logger: logging.Logger = logging.getLogger(__name__)
) -> Dict[str, float]:
    """计算二分类或多分类模型的评估指标。"""
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)
    n_classes = len(unique_labels)

    if len(all_labels_np) == 0:
        logger.warning(f"警告: {mode.upper()} 数据集为空，跳过指标计算。")
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auroc': float('nan'), 'auprc': float('nan')}

    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels_np, all_preds_np)
    metrics['precision'] = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)

    logger.info(f"--- {mode.upper()} 结果 (类别数: {n_classes}) ---")
    logger.info(f"整体准确率 (Accuracy): {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"整体精确率 (Precision): {metrics['precision'] * 100:.2f}%")
    logger.info(f"整体召回率 (Recall): {metrics['recall'] * 100:.2f}%")
    logger.info(f"整体 F1-Score: {metrics['f1'] * 100:.2f}%")

    # --- AUROC ---
    try:
        if n_classes == 2:
            # 二分类：需要知道哪个是正类 (pos_label=1 默认)
            pos_index = unique_labels.index(pos_label) if pos_label in unique_labels else 1
            y_score = all_probs_np[:, pos_index]
            metrics['auroc'] = roc_auc_score(all_labels_np, y_score)
        else:
            # 多分类：使用 OVR (One vs Rest) 和加权平均
            labels_bin = label_binarize(all_labels_np, classes=unique_labels)
            metrics['auroc'] = roc_auc_score(
                labels_bin, all_probs_np, average='weighted', multi_class='ovr'
            )
        logger.info(f"AUROC (weighted/binary): {metrics['auroc']:.4f}")
    except Exception as e:
        logger.warning(f"⚠️ AUROC 计算失败: {e}")
        metrics['auroc'] = float('nan')

    # --- AUPRC ---
    try:
        # AUPRC 在多分类中也使用标签二值化
        labels_bin = label_binarize(all_labels_np, classes=unique_labels)
        if n_classes == 2:
            # 二分类 AUPRC
            pos_label_int = 1
            pos_index = unique_labels.index(pos_label) if pos_label in unique_labels else 1
            y_score = all_probs_np[:, pos_index]
            metrics['auprc'] = average_precision_score(
                all_labels_np, 
                y_score, 
                pos_label=pos_label_int
            )
        else:
            # 多分类 AUPRC
            metrics['auprc'] = average_precision_score(
                labels_bin, all_probs_np, average='weighted'
            )
        logger.info(f"AUPRC (weighted/binary): {metrics['auprc']:.4f}")
    except Exception as e:
        logger.warning(f"⚠️ AUPRC 计算失败: {e}")
        metrics['auprc'] = float('nan')

    return metrics

        
# --- 辅助函数：评估流程 ---
def evaluate(model, data_loader, criterion, task_names, num_classes_dict, device, mode, logger):
    # ... 初始化 (与原代码一致) ...
    model.eval()
    total_combined_loss = 0

    task_preds = {task: [] for task in task_names}
    task_labels = {task: [] for task in task_names}
    task_probs = {task: [] for task in task_names} # <--- 新增
    task_counts = {task: 0 for task in task_names}
    # 保存错误列表
    misclassified_samples = {task: [] for task in task_names}

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
                target = labels.float().view(-1, 1)
                
                # 计算损失
                task_loss = criterion(predictions, target)
                combined_loss += task_loss
                
                # 累加损失和样本数
                task_counts[task_name] += batch_size
                if num_classes_dict.get(task_name) and num_classes_dict[task_name] > 1:
                    # 多分类任务 (如果存在)
                    probabilities = torch.softmax(predictions, dim=1)
                else:
                    probs_pos = torch.sigmoid(predictions).squeeze(1) # 形状 [N]
                    # 构造 N x 2 的概率 [P(0), P(1)]
                    probabilities = torch.stack([1 - probs_pos, probs_pos], dim=1) # 形状 [N, 2]
                
                # 2. 收集概率 (列表的列表: [样本数, 类别数])
                task_probs[task_name].extend(probabilities.cpu().tolist())
                
                # 收集预测和标签
                # 对于 CrossEntropyLoss，预测是 argmax
                preds = (predictions.squeeze(1) > 0).long()
                task_preds[task_name].extend(preds.cpu().tolist())
                task_labels[task_name].extend(labels.cpu().tolist())
                true_labels_cpu = labels.cpu().tolist()
                preds_cpu = preds.cpu().tolist()

                for path, true_label, pred_label in zip(img_paths, true_labels_cpu, preds_cpu):
                    if true_label != pred_label:
                        misclassified_samples[task_name].append({
                            'path': path,
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'task_name': task_name 
                        })

            total_combined_loss += combined_loss.item()
            
    # 计算平均总损失
    avg_combined_loss = total_combined_loss / len(data_loader)

    # 计算评估指标 (例如：Accuracy, F1 Score)
    task_metrics = {}
     # 假设已安装 sklearn
    
    for task_name in task_names:
        true_labels = task_labels[task_name]
        predictions = task_preds[task_name]
        probabilities = task_probs[task_name]
        current_mode = f'{mode}_{task_name}'
        
        if len(true_labels) > 0:
            unique_labels = sorted(list(set(true_labels)))
            pos_label = 1
            metrics = calculate_metrics_binarized(
                all_labels=true_labels,
                all_preds=predictions,
                all_probs=probabilities,
                unique_labels=unique_labels,
                pos_label=pos_label,
                mode=current_mode,
                logger=logger
            )
            task_metrics[task_name] = metrics
        else:
            task_metrics[task_name] = {'accuracy': 0.0, 'f1': 0.0, 'auroc': float('nan'), 'auprc': float('nan')}

    # 返回所有结果
    return avg_combined_loss, task_metrics, misclassified_samples


# --- 辅助函数：TensorBoard 记录 ---
def log_metrics_to_tensorboard(
    writer: SummaryWriter, 
    metrics_dict: Dict[str, Dict[str, float]], 
    total_loss: float,
    step: int, 
    stage: str, 
    logger: logging.Logger,
    ovr_tasks_map: Dict[str, List[str]] 
):
    """
    将所有指标（accuracy, precision, recall, f1, auroc, auprc）按任务类型聚合后写入 TensorBoard。
    """
    all_task_names = list(metrics_dict.keys())
    ovr_parent_tasks = list(ovr_tasks_map.keys()) 
    ovr_sub_tasks = [sub for subs in ovr_tasks_map.values() for sub in subs] 
    independent_tasks = [
        task for task in all_task_names 
        if task not in ovr_parent_tasks and task not in ovr_sub_tasks
    ]
    
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

    # 1. 处理 OvR 父任务：对其子任务求平均
    for parent_task in ovr_parent_tasks:
        sub_tasks = ovr_tasks_map.get(parent_task, [])
        parent_metrics = {k: [] for k in all_individual_metrics.keys()}

        for sub_task in sub_tasks:
            metrics = metrics_dict.get(sub_task, {})
            if not metrics:
                continue
            for key in all_individual_metrics:
                val = metrics.get(key, float('nan'))
                if not np.isnan(val):
                    parent_metrics[key].append(val)
                    all_individual_metrics[key].append(val)

        # 计算父任务平均并写入 TensorBoard
        for key, vals in parent_metrics.items():
            if vals:
                mean_val = np.mean(vals)
                writer.add_scalar(f'{stage}_Summary/{key.upper()}_{parent_task}', mean_val, step)
        
        # 日志打印（可选：只打印关键指标）
        mean_auroc = np.mean(parent_metrics['auroc']) if parent_metrics['auroc'] else float('nan')
        mean_f1 = np.mean(parent_metrics['f1']) if parent_metrics['f1'] else float('nan')
        logger.info(f"  OvR Parent '{parent_task}': AUROC={mean_auroc:.4f}, F1={mean_f1:.4f}")

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

    # 3. 记录总体平均指标（所有子任务 + 独立任务）
    for key, vals in all_individual_metrics.items():
        if vals:
            mean_val = np.mean(vals)
            writer.add_scalar(f'{stage}_Aggregated/Mean_{key.upper()}', mean_val, step)


# --- 训练函数 (新增日志和早停逻辑) ---
def train_multi_task_classifier(logger: logging.Logger, val_split_ratio: float = 0.2, random_seed: int = 42):
    # 1. 初始化预处理器
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # --- TENSORBOARD 初始化 ---
    # 使用日志文件名作为 log_dir 的一部分，确保每次运行的日志独立
    TENSORBOARD_LOG_DIR = os.path.join(f'runs/', FILENAME)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    logger.info(f"TensorBoard Writer initialized at: {TENSORBOARD_LOG_DIR}")
    

    # 2. 初始化整个数据集并进行分割
    try:
        train_dataset = MultiTaskImageDataset(
            root_dir=DATA_ROOT, csv_path=TRAIN_CSV_PATH, img_col=IMAGE_PATH_COLUMN,
            label_cols=LABEL_COLUMNS, processor=processor, size=TARGET_IMAGE_SIZE,
            logger=logger,is_training=True
        )
    except Exception as e:
        logger.critical(f"致命错误：训练数据集加载失败。请检查路径和 CSV 文件。")
        logger.critical(e)
        return

    task_names = train_dataset.all_task_names 
    num_classes_dict = {task: 1 for task in task_names} ## 强制将所有任务的输出类别数设置为 1，以适应 nn.BCEWithLogitsLoss
    fitted_encoders = {
        'label_encoders': train_dataset.label_encoders,
        'ovr_tasks_map': train_dataset.ovr_tasks_map,
        'all_task_names': train_dataset.all_task_names # 传递所有任务名以确保验证集结构一致
    }

    #  类别不平衡权重
    task_weights = {}
    total_samples = len(train_dataset)

    for task_name in task_names:
        # OvR 任务的标签列已经添加到 train_dataset.df 中 (值为 0 或 1)
        pos_count = train_dataset.df[task_name].sum()
        neg_count = total_samples - pos_count
        
        if pos_count > 0 and neg_count > 0:
            # 计算 pos_weight = 负样本 / 正样本
            weight = neg_count / pos_count
            task_weights[task_name] = torch.tensor([weight], dtype=torch.float32).to(DEVICE)
            logger.info(f"任务 '{task_name}' (正样本: {pos_count}, 负样本: {neg_count}) -> Pos Weight: {weight:.2f}")
        else:
            # 如果某个类别的样本数为0，则不进行加权 (权重为 1.0)
            task_weights[task_name] = torch.tensor([1.0], dtype=torch.float32).to(DEVICE)
            logger.warning(f"任务 '{task_name}' 样本数不足 (正: {pos_count})，不应用加权。")


    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"所有二分类任务列表 (含 OvR): {task_names}")


    # 加载验证数据集 (使用训练集编码器)
    try:
        val_dataset = MultiTaskImageDataset(
            root_dir=DATA_ROOT, csv_path=VAL_CSV_PATH, img_col=IMAGE_PATH_COLUMN,
            label_cols=LABEL_COLUMNS, processor=processor, size=TARGET_IMAGE_SIZE,
            logger=logger, 
            fitted_encoders=fitted_encoders,is_training=False
        )
    except Exception as e:
        logger.critical(f"致命错误：验证集加载失败。请检查路径和 CSV 文件 ({VAL_CSV_PATH})。")
        logger.critical(e)
        return
    
    logger.info(f"验证集大小: {len(val_dataset)}")
    train_ovr_tasks_map = train_dataset.ovr_tasks_map 
    val_ovr_tasks_map = val_dataset.ovr_tasks_map

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)

    # 初始化模型、损失函数和优化器
    model = DinoV3MultiTaskClassifier(MODEL_NAME, num_classes_dict).to(DEVICE)
    # 创建任务特定的加权损失函数字典
    criterion_dict = {}
    for task_name, weight in task_weights.items():
        criterion_dict[task_name] = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='mean')
    
    # 用于评估的损失函数通常不加权，以反映真实损失
    unweighted_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    # 仅优化分类头参数 (假设主干网络冻结)
    optimizer = torch.optim.AdamW(model.classifiers.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

    # 初始化 GradScaler
    scaler = torch.amp.GradScaler('cuda')

    logger.info(f"模型已加载，在设备 {DEVICE} 上训练...")

    best_val_score = -1.0
    patience_counter = 0

    best_epoch_val_misclassified_samples: Dict[str, List[Dict[str, Any]]] = {}
    best_epoch_train_misclassified_samples: Dict[str, List[Dict[str, Any]]] = {}
    best_epoch = -1

    # 4. 训练循环
    for epoch in range(NUM_EPOCHS):
        total_combined_loss = 0
        train_labels_all = {task: [] for task in task_names}
        train_preds_all = {task: [] for task in task_names}
        train_probs_all = {task: [] for task in task_names}
        train_paths_all = []
        model.train()

        # 训练步骤
        for step, batch in enumerate(train_loader):
            if batch is None:
                logger.warning("Received an empty batch after filtering corrupt files. Skipping step.")
                continue
            pixel_values, labels_dict, img_paths = batch
            batch_size = pixel_values.size(0)
            pixel_values = pixel_values.to(DEVICE)
            for task in labels_dict:
                labels_dict[task] = labels_dict[task].to(DEVICE)

            optimizer.zero_grad()
            combined_loss = 0.0
            train_paths_all.extend(img_paths)

            with torch.amp.autocast(device_type=DEVICE):
                predictions_dict = model(pixel_values)
                for task_name in model.task_names:
                    predictions = predictions_dict[task_name]  # shape: (batch_size, 1)
                    labels = labels_dict[task_name]            # shape: (batch_size)
                    target = labels.float().view(-1, 1)

                    # 1. 损失计算
                    task_criterion = criterion_dict[task_name]
                    task_loss = task_criterion(predictions, target)
                    combined_loss += task_loss # 累加总损失

                    # 2. 训练集指标积累 (所有任务)
                    predictions_logits = predictions.squeeze(1) # shape: (batch_size)
                    probs = torch.sigmoid(predictions_logits) 
                    
                    train_probs_all[task_name].extend(probs.cpu().tolist())
                    preds = (probs > 0.5).long()
                    train_preds_all[task_name].extend(preds.cpu().tolist())
                    safe_labels_list = labels.cpu().reshape(-1).tolist() 
                    train_labels_all[task_name].extend(safe_labels_list)

            scaler.scale(combined_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_combined_loss += combined_loss.item()
            
            # 记录迭代训练损失
            if step % 50 == 0 and step > 0:
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Step {step}/{len(train_loader)}, "
                        f"Total Train Loss: {combined_loss.item():.4f}")

        avg_train_loss = total_combined_loss / len(train_loader)

        # --- 训练评估
        logger.info(f"--------- Epoch {epoch + 1} 训练评估总结 --------")
        train_metrics = {}
        misclassified_train_samples = {task: [] for task in task_names}
        for task_name in task_names:
            # 这里的 lists 现在包含了所有任务的数据
            true_labels = train_labels_all[task_name]
            predictions = train_preds_all[task_name]
            probabilities = [[1-p, p] for p in train_probs_all[task_name]]
            metrics = calculate_metrics_binarized(
                all_labels=train_labels_all[task_name],
                all_preds=train_preds_all[task_name],
                all_probs=[[1-p, p] for p in train_probs_all[task_name]], 
                unique_labels=[0, 1],
                pos_label=1,
                mode=f'train_{task_name}',
                logger=logger
            )
            train_metrics[task_name] = metrics
            for path, true_label, pred_label in zip(train_paths_all, true_labels, predictions):
                    if true_label != pred_label:
                        misclassified_train_samples[task_name].append({
                            'path': path,
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'task_name': task_name 
                        })

        log_metrics_to_tensorboard(
            writer, 
            train_metrics, 
            avg_train_loss, # 传递总损失
            epoch + 1, 
            'Train', 
            logger,
            ovr_tasks_map=train_ovr_tasks_map
        )

        # --- Epoch 结束后的评估 ---
        logger.info(f"--------- Epoch {epoch + 1} 验证评估总结 --------")
        val_loss, val_metrics, misclassified_val_samples = evaluate(
            model, val_loader, unweighted_criterion, model.task_names, num_classes_dict, DEVICE, mode='val', logger=logger
        )
        log_metrics_to_tensorboard(
            writer, 
            val_metrics, 
            val_loss, # 传递总损失
            epoch + 1, 
            'Val', 
            logger,
            ovr_tasks_map=val_ovr_tasks_map
        )
        
        # --- 早停和模型保存逻辑更新 ---
        key_task = task_names[0]
        val_score = val_metrics[key_task].get('accuracy', val_metrics[key_task].get('f1', 0.0))
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_epoch = epoch + 1
            best_epoch_train_misclassified_samples = misclassified_train_samples 
            best_epoch_val_misclassified_samples = misclassified_val_samples
            logger.info(f"最佳模型auc分数: {best_val_score:.4f}")
        else:
            patience_counter += 1
            logger.info(f"🖤验证未改善。当前耐心值: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info(f"🛑 早停触发！在 Epoch {epoch + 1} 停止训练。")
                break

    logger.info("\n多任务训练完成！")
    if best_epoch > 0:
        MISCLASSIFIED_OUTPUT_DIR = os.path.join(TENSORBOARD_LOG_DIR, 'misclassified')
        os.makedirs(MISCLASSIFIED_OUTPUT_DIR , exist_ok=True)
        
        all_errors = []
        
        # --- 收集训练集错误 ---
        for task_name, errors in best_epoch_train_misclassified_samples.items():
            for error in errors:
                error['epoch'] = best_epoch 
                error['subset'] = 'train' # 标记为训练集
                all_errors.append(error)

        # --- 收集验证集错误 ---
        for task_name, errors in best_epoch_val_misclassified_samples.items():
            for error in errors:
                error['epoch'] = best_epoch 
                error['subset'] = 'val' # 标记为验证集
                all_errors.append(error)

        if all_errors:
            error_df = pd.DataFrame(all_errors)
            
            # ⚠️ 使用固定名称，实现覆盖 (如果路径存在)
            output_path = os.path.join(MISCLASSIFIED_OUTPUT_DIR, f'{best_epoch}_misclassified.csv')
            
            # 只保留关键列，并保存
            error_df[['subset', 'epoch', 'task_name', 'path', 'true_label', 'predicted_label']].to_csv(output_path, index=False)
            logger.info(f"🎉 最终保存：已将最佳 Epoch ({best_epoch}) 的 {len(all_errors)} 个 (Train+Val) 错误样本记录到 {output_path}")
        else:
             logger.info(f"最佳 Epoch ({best_epoch}) 没有记录到任何错误样本。")
    writer.close() # 确保所有数据写入日志文件
    return None


if __name__ == "__main__":
    # 初始化日志记录器
    main_logger = setup_logging()
    main_logger.info(f"日志文件已创建：{LOG_FILENAME}")
    main_logger.info(f"运行设备: {DEVICE}")
    main_logger.info(f"图像尺寸: {TARGET_IMAGE_SIZE}")
    main_logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    main_logger.info(f"LEARNING_RATE: {LEARNING_RATE}")

    trained_model = train_multi_task_classifier(main_logger)

    if trained_model:
        main_logger.info("\n最终模型已训练并加载。")
