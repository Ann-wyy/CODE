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
from utils import set_seed, convert_dinov3_teacher_to_hf_state_dict
from metrics import calculate_metrics, log_metrics_to_tensorboard, evaluate
 
# --- 配置参数 ---
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TARGET_IMAGE_SIZE = 256 # 图像目标尺寸
BATCH_SIZE = 256
LEARNING_RATE = 0.1
NUM_EPOCHS = 1
PATIENCE = 50 # 早停耐心值
RANDOM_SEED = 1

# 自动选择 GPU 设备，优先使用 cuda:0
DEVICE = "cuda:1"

# 用户提供的文件路径
TRAIN_NAME = f"BTXRD"
CSV_PATH = "/home/yyi/data/test_dataset/BTXRD_dataset.csv" # 标签CSV文件路径
IMAGE_PATH_COLUMN = 'image_path' # CSV中包含图像相对路径的列名
LABEL_COLUMNS = ['tumor','benign','malignant'] # 您的所有标签列名
LOAD_LOCAL_CHECKPOINT = False
if LOAD_LOCAL_CHECKPOINT:
    TEST_NAME = "boneDinov3"
else:
    TEST_NAME = "Dinov3"
TEST_NAME = f"{TEST_NAME}_{TRAIN_NAME}_{TARGET_IMAGE_SIZE}_{LEARNING_RATE}_{RANDOM_SEED}"
LOCAL_CHECKPOINT_PATH = "/data/truenas_B2/yyi/bone_logs_512/eval/training_92999/teacher_checkpoint.pth" # 替换为您的本地 .pth 文件路径

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
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILENAME), # 写入文件
            logging.StreamHandler() # 输出到控制台
        ]
    )
    return logging.getLogger(__name__)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
logger = setup_logging() # 初始化全局日志记录器


# --- 自定义 PyTorch Dataset (处理多列分类标签) ---
class MultiTaskImageDatasetFromDataFrame(Dataset):
    def __init__(self, df: pd.DataFrame, img_col: str, 
                 label_cols: List[str], processor: AutoImageProcessor, 
                 size: int, logger: logging.Logger, is_training: bool = False):
        self.df = df
        self.img_col = img_col
        self.label_cols = label_cols
        self.processor = processor
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

        labels_dict = {}
        for task in self.label_cols:
            label_val = row[task]
            # 如果 label_val == -1（未知类别），可在此返回 None 或保留（后续 loss 忽略需特殊处理）
            labels_dict[task] = torch.tensor(label_val, dtype=torch.long)

        return pixel_values, labels_dict, img_path



# ====================================================================
# 2. custom_collate_fn 实现
# ====================================================================

def custom_collate_fn(batch: List[Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None

    pixel_values = torch.stack([item[0] for item in batch])
    
    task_names = list(batch[0][1].keys())
    labels_dict = {}
    for task_name in task_names:
        labels = [item[1][task_name] for item in batch]
        labels_dict[task_name] = torch.stack(labels)  # shape: [N]

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
        
        # 冻结主干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 定义多个分类头
        self.classifiers = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            if num_classes == 2:
                output_dim = 1
            elif num_classes > 2:
                output_dim = num_classes
            else:
                logger.warning(f"警告：任务 '{task_name}' 的类别数 {num_classes} 无效。设置为 1。")
                output_dim = 1
            self.classifiers[task_name] = nn.Linear(feature_dim, output_dim)
            # 确保分类头参数是可训练的
            for param in self.classifiers[task_name].parameters():
                 param.requires_grad = True
        

        
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 运行主干网络（冻结）
        pixel_values = pixel_values.to(self.input_device)
        
        # 即使主干网络冻结，也要确保它在正确的设备上运行
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        # pooler_output 提取全局特征 (CLS Token)
        global_feature = outputs.last_hidden_state[:, 0, :]

        # 运行各个分类头
        logits = {}
        for task_name in self.task_names:
            logits[task_name] = self.classifiers[task_name](global_feature)

        return logits



# --- 训练函数 (新增日志和早停逻辑) ---
def train_multi_task_classifier(logger: logging.Logger):
    # 1. 初始化预处理器
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    # --- TENSORBOARD 初始化 ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    logger.info(f"TensorBoard Writer initialized at: {LOG_DIR}")
    best_model_path = os.path.join(LOG_DIR, "best_model.pth")

    # 读取数据集
    total_df = pd.read_csv(CSV_PATH)
    total_df.dropna(subset=[IMAGE_PATH_COLUMN], inplace=True)
    logger.info(f"总数据集（去除无效路径后）: {len(total_df)}")
    total_df = total_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_val_df, test_df = train_test_split(
        total_df, test_size=0.2, stratify=total_df[LABEL_COLUMNS[0]], random_state=RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, stratify=train_val_df[LABEL_COLUMNS[0]], random_state=RANDOM_SEED
    )
    logger.info(f"划分完成 -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    label_encoders = {}
    num_classes_dict = {}

    for col in LABEL_COLUMNS:
        le = LabelEncoder()
        # 拟合训练集
        train_labels_str = train_df[col].astype(str)
        train_df[col] = le.fit_transform(train_labels_str)

        # 转换 val/test
        for name, df in [("Val", val_df), ("Test", test_df)]:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError as e:
                logger.warning(f"{name} set contains unseen labels in '{col}': {e}. Mapping unknown to -1.")
                # 构建映射字典，未知设为 -1
                label_to_idx = {label: idx for idx, label in enumerate(le.classes_)}
                df[col] = df[col].astype(str).map(label_to_idx).fillna(-1).astype(int)

        label_encoders[col] = le
        num_classes_dict[col] = len(le.classes_)
        logger.info(f"任务 '{col}': 类别 {list(le.classes_)} → 编码 [0..{len(le.classes_)-1}], 共 {len(le.classes_)} 类")
    
    def create_dataset(df, is_train=False):
        return MultiTaskImageDatasetFromDataFrame(
            df=df,
            img_col=IMAGE_PATH_COLUMN,
            label_cols=LABEL_COLUMNS,
            processor=processor,
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
            else:
                # 如果负类权重为零 (极少数情况)，则使用默认值 1.0 或 w_pos
                pos_weight_scalar = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
            
            criterion_dict[task] = nn.BCEWithLogitsLoss(pos_weight=pos_weight_scalar)
            
        elif num_cls > 2:
            criterion_dict[task] = nn.CrossEntropyLoss(weight=weight)
            
        else:
            logger.error(f"任务 '{task}' 的类别数 {num_cls} 无效。使用默认 BCEWithLogitsLoss。")
            criterion_dict[task] = nn.BCEWithLogitsLoss()
    
    # 仅优化分类头参数 (假设主干网络冻结)
    optimizer = torch.optim.AdamW(model.classifiers.parameters(), lr=LEARNING_RATE)

    # 初始化 GradScaler
    scaler = torch.amp.GradScaler('cuda')

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
                    logits = predictions_dict[task_name]  # shape: (batch_size, 1)
                    labels = labels_dict[task_name]   
                    num_cls = num_classes_dict[task_name]         # shape: (batch_size)

                    # 1. 损失计算
                    task_criterion = criterion_dict[task_name]
                    if num_cls == 2:
                        target = labels.float().view(-1, 1) 
                        probs_pos = torch.sigmoid(logits).squeeze(1) # [N]
                        probabilities = torch.stack([1 - probs_pos, probs_pos], dim=1) # [N, 2]
                        preds = (logits.squeeze(1) > 0).long()
                    else:
                        target = labels.long() 
                        probabilities = torch.softmax(logits, dim=1) 
                        preds = torch.argmax(logits, dim=1)

                    task_loss = task_criterion(logits, target)
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
        test_metrics, 
        epoch + 1, 
        'Test', 
        logger,
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
