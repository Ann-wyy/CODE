import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer,AutoImageProcessor, AutoModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ================================导入工具函数====================================
from utils.utils import set_seed, convert_dinov3_teacher_to_hf_state_dict, preprocess_labels_and_setup_datasets
from metrics import calculate_metrics, log_metrics_to_tensorboard, evaluate
# --- 配置参数 ---
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TARGET_IMAGE_SIZE = 256 # 图像目标尺寸
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
PATIENCE = 10 # 早停耐心值
RANDOM_SEED = 42 # 42, 100, 601, 1010, 2025

# 自动选择 GPU 设备，优先使用 cuda:0
DEVICE = "cuda:6"

# 用户提供的文件路径
TRAIN_NAME = "CancerBenign"
TRAIN_CSV_PATH = "/home/yyi/data/test_dataset/cancer_benign/CancerBenign_42_train.csv"
VAL_CSV_PATH = "/home/yyi/data/test_dataset/cancer_benign/CancerBenign_42_val.csv"
TEST_CSV_PATH = "/home/yyi/data/test_dataset/cancer_benign/CancerBenign_42_test.csv"
IMAGE_PATH_COLUMN = 'image_path' # CSV中包含图像相对路径的列名
LABEL_COLUMNS = ['label']  # 您的所有标签列名
TEXT_COLS = ['性别', 'age', 'BodyPart']
LOAD_LOCAL_CHECKPOINT = True # 是否加载本地检查点
if LOAD_LOCAL_CHECKPOINT:
    TEST_NAME = "xrayDinov3_ReLU"
else:
    TEST_NAME = "Dinov3"
TEST_NAME = f"{TEST_NAME}_{TRAIN_NAME}_{TARGET_IMAGE_SIZE}_{LEARNING_RATE}_{RANDOM_SEED}"
LOCAL_CHECKPOINT_PATH = "/data/truenas_B2/yyi/bone_logs_512/eval/training_186999/teacher_checkpoint.pth" # 替换为您的本地 .pth 文件路径
IGNORE_INDEX = -1

# **新增：日志配置函数**
LOG_DIR = f"/data/truenas_B2/yyi/logs/{TRAIN_NAME}/{TEST_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f"{TEST_NAME}_{time.strftime('%Y%m%d-%H%M%S')}.log")

set_seed(RANDOM_SEED) # 设置随机种子

# MedCPT
MEDCPT_NAME = "ncbi/MedCPT-Query-Encoder"
medcpt_tokenizer = AutoTokenizer.from_pretrained(MEDCPT_NAME)

def setup_logging():
    """配置日志记录，输出到文件和控制台。"""
    if logging.getLogger().hasHandlers():
        return logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(LOG_FILENAME), # 写入文件
            logging.StreamHandler() # 输出到控制台
        ]
    )
    return logging.getLogger(__name__)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
logger = setup_logging() # 初始化全局日志记录器
logger.info(f"随机种子: {RANDOM_SEED}")


# --- 自定义 PyTorch Dataset (处理多列分类标签) ---
class MultiModalMedicalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, text_cols: List[str],
                 label_cols: List[str], processor: AutoImageProcessor, 
                 tokenizer: Any, size: int, logger: logging.Logger, 
                 is_training: bool = False):
        self.df = df
        self.img_col = img_col
        self.text_cols = text_cols  # 新增：CSV中作为文本特征的列名列表
        self.label_cols = label_cols
        self.processor = processor
        self.tokenizer = tokenizer  # 新增：MedCPT 的 tokenizer
        self.size = size
        self.logger = logger

        if is_training:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.logger.warning(f"图像损坏或无法加载: {img_path}")
            return None, None, img_path

        if self.transform:
            image = self.transform(image)
        inputs = self.processor(images=image, size=self.size, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        # 文本处理
        combined_text = ", ".join([f"{col}: {str(row[col])}" for col in self.text_cols])
        text_inputs = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=128,  # MedCPT 建议长度
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        labels_dict = {}
        for task in self.label_cols:
            label_val = row[task]
            # 如果 label_val == -1（未知类别），可在此返回 None 或保留（后续 loss 忽略需特殊处理）
            labels_dict[task] = torch.tensor(label_val, dtype=torch.long)

        return pixel_values, input_ids, attention_mask, labels_dict, img_path



# ====================================================================
# 2. custom_collate_fn 实现
# ====================================================================

def custom_collate_fn(batch: List[Any]) -> Dict[str, Any]:
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None

    return {
        "pixel_values": torch.stack([item[0] for item in batch]),
        "input_ids": torch.stack([item[1] for item in batch]),
        "attention_mask": torch.stack([item[2] for item in batch]),
        "labels": {task: torch.stack([item[3][task] for item in batch]) 
                   for task in batch[0][3].keys()},
        "img_paths": [item[4] for item in batch]
    }

#  --- MedicalFusionClassifier---
class MedicalFusionClassifier(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, num_classes):
        super(MedicalFusionClassifier, self).__init__()
        
        # 文本支路投影
        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # 图像支路投影 (修正了原代码中的全角逗号和定义错误)
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # 融合后的分类层 (512 + 512 = 1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image_features, text_features):
        t_feat = self.text_projection(text_features)
        i_feat = self.image_projection(image_features)
        combined = torch.cat((i_feat, t_feat), dim=1)
        logits = self.classifier(combined)
        return logits



# --- 自定义模型：DINOv3 + 多个分类头 ---

class DinoV3MultiTaskClassifier(nn.Module):
    """
    基于 DINOv3 主干网络，带有多任务分类头。
    """
    def __init__(self, image_model_name: str, task_num_classes: Dict[str, int]):
        super().__init__()

        self.task_names = list(task_num_classes.keys())
        self.device = torch.device(DEVICE)

        # 1. 加载 DINOv3 图像主干
        self.image_backbone = AutoModel.from_pretrained(image_model_name)
        image_feature_dim = self.image_backbone.config.hidden_size

        # 2. 加载 MedCPT 文本主干
        self.text_backbone = AutoModel.from_pretrained(MEDCPT_NAME)
        text_feature_dim = 768  # MedCPT 固定维度

        # ==================== 根据全局变量加载本地检查点 ====================
        global LOAD_LOCAL_CHECKPOINT, LOCAL_CHECKPOINT_PATH
        if LOAD_LOCAL_CHECKPOINT and os.path.exists(LOCAL_CHECKPOINT_PATH):
            logger.info(f"Loading local checkpoint: {LOCAL_CHECKPOINT_PATH}")
            checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location='cpu')
            state_dict = checkpoint.get('teacher', checkpoint.get('model', checkpoint))
            state_dict = convert_dinov3_teacher_to_hf_state_dict(state_dict, model_dim=1024)
            self.image_backbone.load_state_dict(state_dict, strict=False)
            logger.info("✅ Image backbone loaded successfully.")
        # 冻结主干网络参数
        for param in self.image_backbone.parameters():
            param.requires_grad = False
        for param in self.text_backbone.parameters():
            param.requires_grad = False
        # 定义融合分类头
        self.classifiers = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            # 确定输出维度
            output_dim = 1 if num_classes == 2 else num_classes

            self.classifiers[task_name] = MedicalFusionClassifier(
                image_feature_dim=image_feature_dim,
                text_feature_dim=text_feature_dim,
                num_classes=output_dim
            )
            
        # 确保分类头参数是可训练的
        for param in self.classifiers[task_name].parameters():
            param.requires_grad = True
        

        
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 确保数据在正确的设备上
        pixel_values = pixel_values.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 1. 提取图像特征 (冻结模式)
        with torch.no_grad():
            img_outputs = self.image_backbone(pixel_values=pixel_values)
            img_feature = img_outputs.last_hidden_state[:, 0, :]  # CLS Token

        # 2. 提取文本特征 (冻结模式)
        with torch.no_grad():
            text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
            text_feature = text_outputs.last_hidden_state[:, 0, :]  # CLS Token

        # 3. 运行各个任务的融合分类头
        logits = {}
        for task_name in self.task_names:
            # 这里的 classifier 是 MedicalFusionClassifier 实例
            logits[task_name] = self.classifiers[task_name](img_feature, text_feature)

        return logits
    



# --- 训练函数 (新增日志和早停逻辑) ---
def train_multi_task_classifier(logger: logging.Logger):
    # 1. 初始化预处理器
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    if LOAD_LOCAL_CHECKPOINT:
        processor.image_mean = [0.351, 0.35, 0.351]
        processor.image_std = [0.297, 0.298, 0.298]
        logger.info(f"图像处理参数已修改: Mean={processor.image_mean}, Std={processor.image_std}")
    # --- TENSORBOARD 初始化 ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    logger.info(f"TensorBoard Writer initialized at: {LOG_DIR}")
    best_model_path = os.path.join(LOG_DIR, "best_model.pth")

    # 读取数据集
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=[IMAGE_PATH_COLUMN], inplace=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(f"数据集已加载 -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    label_encoders = {}
    num_classes_dict = {}

    train_df, val_df, test_df, num_classes_dict = preprocess_labels_and_setup_datasets(TRAIN_CSV_PATH, 
        VAL_CSV_PATH, TEST_CSV_PATH, LABEL_COLUMNS, IMAGE_PATH_COLUMN, logger)
    
    
    def create_dataset(df, is_train=False):
        return MultiModalMedicalDataset(
            df=df,
            img_col=IMAGE_PATH_COLUMN,
            text_cols=TEXT_COLS,
            label_cols=LABEL_COLUMNS,
            processor=processor,
            tokenizer=medcpt_tokenizer,
            size=TARGET_IMAGE_SIZE,
            logger=logger,
            is_training=is_train
        )
    
    train_dataset = create_dataset(train_df, is_train=True)
    val_dataset = create_dataset(val_df, is_train=False)
    test_dataset = create_dataset(test_df, is_train=False)  # 用于最终测试

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=8, collate_fn=custom_collate_fn, pin_memory=True)

    #  类别不平衡权重
    task_weights = {}

    for task in LABEL_COLUMNS:
        num_cls = num_classes_dict[task]
        labels = train_df[task].values
        # 过滤掉 -1（未知类别）
        valid_labels = labels[labels != -1]

        if len(np.unique(valid_labels)) < num_cls:
            logger.warning(f"训练集中任务 '{task}' 未覆盖所有类别")

        if len(valid_labels) == 0:
            weight = torch.ones(num_cls, device=DEVICE)
        else:
            try:
                classes = np.arange(num_cls)
                weights = compute_class_weight('balanced', classes=classes, y=valid_labels)
                weight = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
            except ValueError:
                weight = torch.ones(num_cls, device=DEVICE)
        task_weights[task] = weight

    # 初始化模型、损失函数和优化器
    model = DinoV3MultiTaskClassifier(MODEL_NAME, num_classes_dict).to(DEVICE)
    # 创建任务特定的加权损失函数字典
    criterion_dict = {}
    for task in LABEL_COLUMNS:
        weight = task_weights[task]  # shape: (num_classes,)
        num_cls = num_classes_dict[task]
        
        if num_cls == 2:
            # --- 二分类 (C=2) ---
            w_neg = weight[0].item()
            w_pos = weight[1].item()
            
            # 确保 w_neg 不为零以防除零错误
            if w_neg > 0:
                # pos_weight 必须是标量，表示正类的权重
                pos_weight_scalar = torch.tensor(w_pos / w_neg, dtype=torch.float32, device=DEVICE)
                logger.info(f"任务 '{task}' 的 BCEWithLogitsLoss 使用 pos_weight={pos_weight_scalar.item():.4f}")
            else:
                # 如果负类权重为零 (极少数情况)，则使用默认值 1.0 或 w_pos
                pos_weight_scalar = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
            
            criterion_dict[task] = nn.BCEWithLogitsLoss(pos_weight=pos_weight_scalar)
            
        elif num_cls > 2:
            criterion_dict[task] = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
            logger.info(f"任务 '{task}' 的 CrossEntropyLoss 使用 class weights: {weight.cpu().numpy()}")
            
        else:
            logger.error(f"任务 '{task}' 的类别数 {num_cls} 无效。使用默认 CrossEntropyLoss。")
            criterion_dict[task] = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    # 仅优化分类头参数 (假设主干网络冻结)
    optimizer = torch.optim.AdamW(model.classifiers.parameters(), lr=LEARNING_RATE)

    # 初始化 GradScaler
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',                 # 监控分数，所以使用 'max'
        factor=0.1,                 # 每次降低为原来的 1/2
        patience=10,               # 多少个 epoch 不改善后触发
        min_lr=1e-4             # 学习率下限
    )
    logger.info(f"学习率调度器 ReduceLROnPlateau 已初始化，监控模式: max, 降低耐心值: 10")

    logger.info(f"模型已加载，在设备 {DEVICE} 上训练...")

    best_val_score = -1.0
    patience_counter = 0

    best_epoch = -1

    # 4. 训练循环
    task_names = LABEL_COLUMNS
    for epoch in range(NUM_EPOCHS):
        total_combined_loss = 0
        train_labels_all = {task: [] for task in task_names}
        train_probs_all = {task: [] for task in task_names}
        train_paths_all = []
        model.train()

        # 训练步骤
        for step, batch in enumerate(train_loader):
            if batch is None:
                logger.warning("Received an empty batch after filtering corrupt files. Skipping step.")
                continue
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_dict = batch["labels"] 
            img_paths = batch["img_paths"]
            batch_size = pixel_values.size(0)
            for task in labels_dict:
                labels_dict[task] = labels_dict[task].to(DEVICE)

            optimizer.zero_grad()
            combined_loss = 0.0
            train_paths_all.extend(img_paths)

            with torch.amp.autocast(device_type=DEVICE):
                predictions_dict = model(pixel_values, input_ids, attention_mask)
                for task_name in model.task_names:
                    logits = predictions_dict[task_name]  # shape: (batch_size, 1)
                    labels = labels_dict[task_name]   
                    num_cls = num_classes_dict[task_name]         # shape: (batch_size)
                    task_criterion = criterion_dict[task_name]
                    task_loss = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)

                    valid_mask = (labels != -1)
                    valid_count = valid_mask.sum()
                    if valid_count == 0:
                        continue

                    valid_logits = logits[valid_mask]
                    valid_labels = labels[valid_mask].long()      # [N_valid]

                    if num_cls == 2:
                        valid_mask = (labels != -1)
                        valid_count = valid_mask.sum()
                            
                        # BCEWithLogitsLoss 需要浮点型的 target，形状为 (N, 1)
                        target = valid_labels.float().view(-1, 1) 
                        
                        task_loss = task_criterion(valid_logits, target)
                        # 计算全部样本的概率 (用于指标统计)
                        probs_pos = torch.sigmoid(logits).squeeze(1) # [N]
                        # 重新构造 [1-p, p] 格式的概率，用于 metrics
                        probabilities = torch.stack([1 - probs_pos, probs_pos], dim=1) # [N, 2]
                    else:
                        target = valid_labels.long() 
                        task_loss = task_criterion(valid_logits , target)
                        probabilities = torch.softmax(logits , dim=1) 

                    combined_loss += task_loss # 累加总损失
                    train_probs_all[task_name].extend(probabilities.cpu().tolist())
                    train_labels_all[task_name].extend(labels.cpu().tolist())

            scaler.scale(combined_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_combined_loss += combined_loss.item()
            
            # 记录迭代训练损失
            if step % 50 == 0 and step > 0:
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Step {step}/{len(train_loader)}, "
                        f"Total Train Loss: {combined_loss.item():.4f}")

        # --- 训练评估
        logger.info(f"--------- Epoch {epoch + 1} 训练评估总结 --------")
        train_metrics = {}
        for task_name in task_names:
            num_cls = num_classes_dict[task_name]
            metrics = calculate_metrics(
                all_labels=train_labels_all[task_name],
                all_probs=train_probs_all[task_name],
                num_classes=num_cls,
                task_name=task_name,
                mode=f'train_{task_name}',
                logger=logger
            )
            train_metrics[task_name] = metrics

        log_metrics_to_tensorboard(
            writer, 
            train_metrics, 
            epoch + 1, 
            'Train', 
            logger
        )

        # --- Epoch 结束后的评估 ---
        logger.info(f"--------- Epoch {epoch + 1} 验证评估总结 --------")
        val_metrics = evaluate(
            model, val_loader, criterion_dict, model.task_names, num_classes_dict, DEVICE, mode='val', logger=logger
        )
        log_metrics_to_tensorboard(
            writer, 
            val_metrics, 
            epoch + 1, 
            'Val', 
            logger
        )
        key_task = model.task_names[0]
        val_score = val_metrics[key_task]['auroc']
        # === 新增：学习率调度器步进 ===
        scheduler.step(val_score) 
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"当前学习率 (LR): {current_lr:.4f}")
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_epoch = epoch + 1
            logger.info(f"最佳模型auroc分数: {best_val_score:.4f}")
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
    # 加载最佳模型权重
    if os.path.exists(best_model_path):
        logger.info(f"正在从 {best_model_path} 加载最佳模型权重...")
        try:
            # 💥 关键修改：加载最佳模型
            checkpoint = torch.load(best_model_path, map_location=DEVICE,weights_only=False)
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
        model, test_loader, criterion_dict, model.task_names, num_classes_dict, DEVICE, mode='test', logger=logger
    )
    log_metrics_to_tensorboard(
            writer, 
            val_metrics, 
            epoch + 1, 
            'Test', 
            logger
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
