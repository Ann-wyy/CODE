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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ================================导入工具函数====================================
from utils.utils import set_seed, convert_dinov3_teacher_to_hf_state_dict, preprocess_labels_and_setup_datasets
from utils.metrics import calculate_metrics, log_metrics_to_tensorboard, evaluate
 
# --- 配置参数 ---
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TARGET_IMAGE_SIZE = 256 # 图像目标尺寸
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 30 # 早停耐心值
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
LOAD_LOCAL_CHECKPOINT = False # 是否加载本地检查点
if LOAD_LOCAL_CHECKPOINT:
    TEST_NAME = "xrayDinov3_ReLU"
else:
    TEST_NAME = "Dinov3_ReLU"
TEST_NAME = f"{TEST_NAME}_{TRAIN_NAME}_{TARGET_IMAGE_SIZE}_{LEARNING_RATE}_{RANDOM_SEED}"
LOCAL_CHECKPOINT_PATH = "/data/truenas_B2/yyi/bone_logs_512/eval/training_186999/teacher_checkpoint.pth" # 替换为您的本地 .pth 文件路径
IGNORE_INDEX = -1

# **新增：日志配置函数**
LOG_DIR = f"/data/truenas_B2/yyi/logs/{TRAIN_NAME}/{TEST_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f"{TEST_NAME}_{time.strftime('%Y%m%d-%H%M%S')}.log")

set_seed(RANDOM_SEED) # 设置随机种子

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

# clinical
class ClinicalEncoder:
    def __init__(self, df, text_cols):
        # 性别映射
        self.gender_map = {val: i for i, val in enumerate(df['性别'].unique())}
        # 部位映射
        self.body_part_map = {val: i for i, val in enumerate(df['BodyPart'].unique())}
        # 记录维度
        self.clinical_dim = 1 + len(self.gender_map) + len(self.body_part_map)

    def encode(self, row):
        # 1. 年龄归一化 (假设最大100岁)
        age = torch.tensor([float(row['age']) / 100.0], dtype=torch.float32)
        # 2. 性别 One-hot
        gender = torch.zeros(len(self.gender_map))
        gender[self.gender_map[row['性别']]] = 1.0
        # 3. 部位 One-hot
        body = torch.zeros(len(self.body_part_map))
        body[self.body_part_map[row['BodyPart']]] = 1.0
        
        return torch.cat([age, gender, body])


# --- 自定义 PyTorch Dataset (处理多列分类标签) ---
class MultiTaskImageDatasetFromDataFrame(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, 
                 label_cols: List[str], processor: AutoImageProcessor, 
                 size: int, logger: logging.Logger,clinical_encoder, is_training: bool = False):
        self.df = df
        self.img_col = img_col
        self.label_cols = label_cols
        self.processor = processor
        self.size = size
        self.logger = logger
        self.clinical_encoder = clinical_encoder

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
        clinical_values = self.clinical_encoder.encode(row)

        labels_dict = {}
        for task in self.label_cols:
            label_val = row[task]
            # 如果 label_val == -1（未知类别），可在此返回 None 或保留（后续 loss 忽略需特殊处理）
            labels_dict[task] = torch.tensor(label_val, dtype=torch.long)

        return pixel_values, clinical_values, labels_dict, img_path



# ====================================================================
# 2. custom_collate_fn 实现
# ====================================================================

def custom_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None

    pixel_values = torch.stack([item[0] for item in batch])
    clinical_values = torch.stack([item[1] for item in batch]) # 新增
    
    # 保持 labels 处理逻辑不变
    task_names = list(batch[0][2].keys())
    labels_dict = {name: torch.stack([item[2][name] for item in batch]) for name in task_names}
    img_paths = [item[3] for item in batch]
    
    return pixel_values, clinical_values, labels_dict, img_paths


# ---- Gated ---
class GatedFusionHead(nn.Module):
    def __init__(self, image_dim, clinical_dim, output_dim):
        super().__init__()
        # 投影层：将不同模态对齐到同一维度
        self.img_proj = nn.Sequential(nn.Linear(image_dim, 512), nn.ReLU())
        self.cli_proj = nn.Sequential(nn.Linear(clinical_dim, 512), nn.ReLU())
        
        # 门控网络：学习一个权重来平衡两者的重要性
        self.gate = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, img_feat, cli_feat):
        i_p = self.img_proj(img_feat)
        c_p = self.cli_proj(cli_feat)
        
        # 计算门控值
        g = self.gate(torch.cat([i_p, c_p], dim=1))
        
        # 融合：如果g趋近1则偏向图像，趋近0则偏向临床
        fused = g * i_p + (1 - g) * c_p
        return self.classifier(fused)

# --- 自定义模型：DINOv3 + 多个分类头 ---

class DinoV3MultiTaskClassifier(nn.Module):
    """
    基于 DINOv3 主干网络，带有多任务分类头。
    """
    def __init__(self, model_name: str, task_num_classes: Dict[str, int], clinical_dim):
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
                
                # 转换键名以匹配 Hugging Face 模型
                state_dict = convert_dinov3_teacher_to_hf_state_dict(
                        state_dict, 
                        model_dim=1024
                )

                try:
                    # 使用 strict=False 允许忽略不需要的键（如分类头或注册token）
                    load_info = self.backbone.load_state_dict(state_dict, strict=False)
                    logger.info("Local Teacher weights loaded successfully.")
                    # 打印缺失和不匹配的键，用于调试
                    if load_info.unexpected_keys:
                        logger.warning(f"⚠️ Warning: Unexpected keys (ignored): {load_info.unexpected_keys[:5]}...")
                    if load_info.missing_keys:
                        logger.warning(f"⚠️ Warning: Missing keys (using HF weights): {load_info.missing_keys[:5]}...")
                    logger.info("✅ Backbone checkpoint loaded successfully.")

                except Exception as e:
                    logger.error(f"Error loading checkpoint at {LOCAL_CHECKPOINT_PATH}: {e}")
                    logger.warning("Continuing with the model initialized from Hugging Face and random classifiers.")
            else:
                logger.error(f"Warning: Global flag LOAD_LOCAL_CHECKPOINT is True, but file not found at: {LOCAL_CHECKPOINT_PATH}")
        else:
            logger.info("Global flag LOAD_LOCAL_CHECKPOINT is False. Initializing model from scratch (Hugging Face backbone + new classifiers).")
        
        # 冻结主干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 定义多个分类头
        self.classifiers = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            out_dim = 1 if num_classes == 2 else num_classes
            self.classifiers[task_name] = GatedFusionHead(feature_dim, clinical_dim, out_dim)

        

        
    def forward(self, pixel_values: torch.Tensor, clinical_values):
        # 运行主干网络（冻结）
        pixel_values = pixel_values.to(self.input_device)
        clinical_values = clinical_values.to(DEVICE)
        
        # 即使主干网络冻结，也要确保它在正确的设备上运行
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        # pooler_output 提取全局特征 (CLS Token)
        global_feature = outputs.last_hidden_state[:, 0, :]

        # 运行各个分类头
        logits = {}
        for task_name in self.task_names:
            logits[task_name] = self.classifiers[task_name](global_feature, clinical_values)

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
    # 初始化编码器 (自动计算维度)
    clinical_encoder = ClinicalEncoder(train_df, TEXT_COLS)
    clinical_dim = clinical_encoder.clinical_dim
    logger.info(f"临床特征维度: {clinical_dim}")

    train_df, val_df, test_df, num_classes_dict = preprocess_labels_and_setup_datasets(TRAIN_CSV_PATH, 
        VAL_CSV_PATH, TEST_CSV_PATH, LABEL_COLUMNS, IMAGE_PATH_COLUMN, logger)
    
    def create_dataset(df, is_train=False):
        return MultiTaskImageDatasetFromDataFrame(
            df=df,
            img_col=IMAGE_PATH_COLUMN,
            label_cols=LABEL_COLUMNS,
            processor=processor,
            size=TARGET_IMAGE_SIZE,
            logger=logger,
            clinical_encoder=clinical_encoder,
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
    model = DinoV3MultiTaskClassifier(MODEL_NAME, num_classes_dict, clinical_dim).to(DEVICE)
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
            pixel_values, clinical_values, labels_dict, img_paths = batch
            batch_size = pixel_values.size(0)
            pixel_values = pixel_values.to(DEVICE)
            clinical_values = clinical_values.to(DEVICE)
            for task in labels_dict:
                labels_dict[task] = labels_dict[task].to(DEVICE)

            optimizer.zero_grad()
            combined_loss = 0.0
            train_paths_all.extend(img_paths)

            with torch.amp.autocast(device_type=DEVICE):
                predictions_dict = model(pixel_values, clinical_values)
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
