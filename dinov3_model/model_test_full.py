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

from utils import set_seed, convert_dinov3_teacher_to_hf_state_dict, calculate_metrics_binarized, log_metrics_to_tensorboard, evaluate
 
# --- 配置参数 ---
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TARGET_IMAGE_SIZE = 256 # 图像目标尺寸
BATCH_SIZE = 256
LEARNING_RATE = 0.1
NUM_EPOCHS = 100
PATIENCE = 10 # 早停耐心值
RANDOM_SEED = 42

# 自动选择 GPU 设备，优先使用 cuda:0
DEVICE = "cuda:0"
if torch.cuda.is_available():
    # 使用用户定义的逻辑，但修正为可用设备
    device_id = 1 if torch.cuda.device_count() > 2 else 0
    DEVICE = f"cuda:{device_id}"
else:
    DEVICE = "cpu"


# 用户提供的文件路径
TRAIN_NAME = f"BTXRD_{TARGET_IMAGE_SIZE}_{LEARNING_RATE}_{RANDOM_SEED}"
CSV_PATH = "/home/yyi/data/test_dataset/BTXRD/BTXRD_dataset.csv" 
IMAGE_PATH_COLUMN = 'image_path' # CSV中包含图像相对路径的列名
LABEL_COLUMNS = ['tumor','benign','malignant'] # 您的所有标签列名
LOAD_LOCAL_CHECKPOINT = False
if LOAD_LOCAL_CHECKPOINT:
    TEST_NAME = "boneDinov3"
else:
    TEST_NAME = "Dinov3"
TEST_NAME = f"{TRAIN_NAME}_{TEST_NAME}"
LOCAL_CHECKPOINT_PATH = "/data/truenas_B2/yyi/bone_logs_512/eval/training_80999/teacher_checkpoint.pth" # 替换为您的本地 .pth 文件路径

# **新增：日志配置函数**
LOG_DIR = f"/data/truenas_B2/yyi/logs/{TEST_NAME}_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f"{TEST_NAME}.log")

set_seed(RANDOM_SEED) # 设置随机种子

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
    def __init__(self, csv_path: str, img_col: str, label_cols: List[str], 
                 processor: AutoImageProcessor, size: int, logger: logging.Logger, 
                 fitted_encoders: Dict[str, Any] = None, is_training: bool = False, indices: List[int] = None):
        self.processor = processor
        self.size = size
        self.logger = logger
        self.label_cols = label_cols
        self.is_training = is_training
        
        try:
            self.df = pd.read_csv(csv_path)
            # 移除图像路径为空或缺失的行
            self.df.dropna(subset=[img_col], inplace=True)
            if indices is not None:
                self.df = self.df.iloc[indices].reset_index(drop=True)
                self.logger.info(f"数据集已根据传入的 {len(indices)} 个索引进行过滤。")
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
                checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location='cpu')
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
                    state_dict = convert_dinov3_teacher_to_hf_state_dict(
                            state_dict, 
                            model_dim=1024 
                        )

                try:
                    # 使用 strict=False 允许忽略不需要的键（如分类头或注册token）
                    load_info = self.backbone.load_state_dict(state_dict, strict=False)
                    print("Local Teacher weights loaded successfully.")
                    # 打印缺失和不匹配的键，用于调试
                    if load_info.unexpected_keys:
                        print(f"⚠️ Warning: Unexpected keys (ignored): {load_info.unexpected_keys[:5]}...")
                    if load_info.missing_keys:
                        print(f"⚠️ Warning: Missing keys (using HF weights): {load_info.missing_keys[:5]}...")
                    logger.info("✅ Backbone checkpoint loaded successfully.")

                except Exception as e:
                    logger.error(f"Error loading checkpoint at {LOCAL_CHECKPOINT_PATH}: {e}")
                    logger.warning("Continuing with the model initialized from Hugging Face and random classifiers.")
            else:
                logger.error(f"Warning: Global flag LOAD_LOCAL_CHECKPOINT is True, but file not found at: {LOCAL_CHECKPOINT_PATH}")
        else:
            logger.info("Global flag LOAD_LOCAL_CHECKPOINT is False. Initializing model from scratch (Hugging Face backbone + new classifiers).")
        
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





# --- 训练函数 (新增日志和早停逻辑) ---
def train_multi_task_classifier(logger: logging.Logger, val_split_ratio: float = 0.2, random_seed: int = 42):
    random_seed = RANDOM_SEED
    # 1. 初始化预处理器
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # --- TENSORBOARD 初始化 ---
    # 使用日志文件名作为 log_dir 的一部分，确保每次运行的日志独立
    writer = SummaryWriter(log_dir=LOG_DIR)
    logger.info(f"TensorBoard Writer initialized at: {LOG_DIR}")
    best_model_path = os.path.join(LOG_DIR, "best_model.pth")
    print(f"模型将保存到: {best_model_path}")
    

    #  初始化整个数据集并进行分割
    try:
        full_dataset = MultiTaskImageDataset(csv_path=CSV_PATH, img_col=IMAGE_PATH_COLUMN,
            label_cols=LABEL_COLUMNS, processor=processor, size=TARGET_IMAGE_SIZE,
            logger=logger,is_training=True
        )
    except Exception as e:
        logger.critical(f"致命错误：训练数据集加载失败。请检查路径和 CSV 文件。")
        logger.critical(e)
        return
    total_size = len(full_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    logger.info(f"完整数据集大小: {total_size}")
    logger.info(f"划分比例: 训练集 {train_size} ({train_size}) / 验证集 {val_size} ({val_size}) / 测试集 {test_size} ({test_size})")

    task_names = full_dataset.all_task_names
    num_classes_dict = {task: 1 for task in task_names} ## 强制将所有任务的输出类别数设置为 1，以适应 nn.BCEWithLogitsLoss
    fitted_encoders = {
        'label_encoders': full_dataset.label_encoders,
        'ovr_tasks_map': full_dataset.ovr_tasks_map,
        'all_task_names': full_dataset.all_task_names # 传递所有任务名以确保验证集结构一致
    }

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    try:
        # random_split 返回的是 Subset 对象列表
        train_subset, val_subset, test_subset = random_split(
            full_dataset, 
            [train_size, val_size, test_size], 
            generator=generator
        )
    except Exception as e:
        logger.critical(f"致命错误：数据集划分失败。")
        logger.critical(e)
        return
    
    try:
        train_dataset = train_subset
        # 提取需要的元数据 (如 task_names, encoders) - 此时应从 full_dataset 获取
        task_names = full_dataset.all_task_names 
        num_classes_dict = {task: 1 for task in task_names}
        fitted_encoders = {
            'label_encoders': full_dataset.label_encoders, # 从完整数据集获取拟合的编码器
            'ovr_tasks_map': full_dataset.ovr_tasks_map,
            'all_task_names': full_dataset.all_task_names
        }
    except Exception as e:
        logger.critical(f"致命错误：训练集子集处理失败。")
        logger.critical(e)
        return
    
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices
    train_dataset = MultiTaskImageDataset(
        root_dir=DATA_ROOT, csv_path=CSV_PATH, img_col=IMAGE_PATH_COLUMN,
        label_cols=LABEL_COLUMNS, processor=processor, size=TARGET_IMAGE_SIZE,
        logger=logger,fitted_encoders=fitted_encoders, is_training=True,indices=train_indices
    )
    val_dataset = MultiTaskImageDataset(
        root_dir=DATA_ROOT, csv_path=CSV_PATH, img_col=IMAGE_PATH_COLUMN,
        label_cols=LABEL_COLUMNS, processor=processor, size=TARGET_IMAGE_SIZE,
        logger=logger,fitted_encoders=fitted_encoders, is_training=False,indices=val_indices
    )
    test_dataset = MultiTaskImageDataset(
        root_dir=DATA_ROOT, csv_path=CSV_PATH, img_col=IMAGE_PATH_COLUMN,
        label_cols=LABEL_COLUMNS, processor=processor, size=TARGET_IMAGE_SIZE,
        logger=logger,fitted_encoders=fitted_encoders, is_training=False,indices=test_indices
    )
    

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

    val_dataset = val_subset
    test_dataset = test_subset

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
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
    optimizer = torch.optim.AdamW(model.classifiers.parameters(), lr=LEARNING_RATE)

    # 初始化 GradScaler
    scaler = torch.amp.GradScaler('cuda')

    logger.info(f"模型已加载，在设备 {DEVICE} 上训练...")

    best_val_score = -1.0
    patience_counter = 0

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
            epoch + 1, 
            'Train', 
            logger,
            ovr_tasks_map=full_dataset.ovr_tasks_map
        )

        # --- Epoch 结束后的评估 ---
        logger.info(f"--------- Epoch {epoch + 1} 验证评估总结 --------")
        val_metrics = evaluate(
            model, val_loader, unweighted_criterion, model.task_names, num_classes_dict, DEVICE, mode='val', logger=logger
        )
        log_metrics_to_tensorboard(
            writer, 
            val_metrics, 
            epoch + 1, 
            'Val', 
            logger,
            ovr_tasks_map=full_dataset.ovr_tasks_map
        )
        
        # --- 早停和模型保存逻辑更新 ---
        key_task = task_names[0]
        val_score = val_metrics[key_task].get('auprc', val_metrics[key_task].get('auroc', 0.0))
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_epoch = epoch + 1       
            logger.info(f"最佳模型auprc分数: {best_val_score:.4f}")
            try:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_score': best_val_score,
                    'task_names': task_names
                }, best_model_path)
                logger.info(f"✅ 模型权重已保存到: {best_model_path}")
            except Exception as e:
                logger.error(f"保存模型权重失败: {e}")
        else:
            patience_counter += 1
            logger.info(f"🖤验证未改善。当前耐心值: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info(f"🛑 早停触发！在 Epoch {epoch + 1} 停止训练。")
                break

    logger.info("\n多任务训练完成！")
    # 加载最佳模型进行最终评估
    if os.path.exists(best_model_path):
        logger.info(f"正在从 {best_model_path} 加载最佳模型权重...")
        try:
            # 💥 关键修改：加载最佳模型
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            # 确保模型处于评估模式
            model.eval() 
            logger.info(f"模型已成功加载 (最佳 Epoch: {checkpoint['epoch']}, Score: {checkpoint['best_val_score']:.4f})")
        except Exception as e:
            logger.critical(f"加载最佳模型失败: {e}")
            return # 如果加载失败，则无法进行测试评估
    else:
        logger.warning("未找到最佳模型检查点，使用当前模型状态进行测试评估。")
        model.eval() # 切换到评估模式
    test_metrics = evaluate(
        model, test_loader, unweighted_criterion, model.task_names, num_classes_dict, DEVICE, mode='test', logger=logger
    )
    log_metrics_to_tensorboard(
        writer, 
        test_metrics, 
        epoch + 1, 
        'Test', 
        logger,
        ovr_tasks_map=full_dataset.ovr_tasks_map
    )
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
