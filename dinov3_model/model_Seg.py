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
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.functional import accuracy

# ================================导入工具函数====================================
from utils import set_seed, convert_dinov3_teacher_to_hf_state_dict
from metrics import log_metrics_to_tensorboard, evaluate
 
# --- 配置参数 ---
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TARGET_IMAGE_SIZE = 256 # 图像目标尺寸
BATCH_SIZE = 4
LEARNING_RATE = 0.1
NUM_EPOCHS = 1
PATIENCE = 30 # 早停耐心值
RANDOM_SEED = 42 # 42, 100, 600, 1000, 2025

# 自动选择 GPU 设备，优先使用 cuda:0
DEVICE = "cuda:7"

# 用户提供的文件路径
TRAIN_NAME = f"BoneFracture"
BASE_DATA_DIR = "/data/truenas_B2/yyi/data/BoneFractureYolo8"
NUM_CLASSES = 2 # 分割类别数（包括背景）
LOAD_LOCAL_CHECKPOINT = False # 是否加载本地检查点
if LOAD_LOCAL_CHECKPOINT:
    TEST_NAME = "boneDinov3"
else:
    TEST_NAME = "Dinov3"
TEST_NAME = f"{TEST_NAME}_{TRAIN_NAME}_{TARGET_IMAGE_SIZE}_{LEARNING_RATE}_{RANDOM_SEED}"
LOCAL_CHECKPOINT_PATH = "/data/truenas_B2/yyi/bone_logs_512/eval/training_92499_1/teacher_checkpoint.pth" # 替换为您的本地 .pth 文件路径

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



#  ----配对文件夹-----
def get_image_mask_pairs(image_dir: str, mask_dir: str, logger: logging.Logger):
    """
    根据 images/ 和 labels/ 文件夹，返回配对的 (image_path, mask_path) 列表。
    要求：两个文件夹下文件名一一对应（不含扩展名）。
    """
    image_files = {os.path.splitext(f)[0]: os.path.join(image_dir, f) 
                   for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))}
    mask_files = {os.path.splitext(f)[0]: os.path.join(mask_dir, f) 
                  for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.txt'))}

    common_names = set(image_files.keys()) & set(mask_files.keys())
    if len(common_names) == 0:
        logger.warning("⚠️ 未找到任何 image-mask 配对文件！请检查文件夹结构。")
    else:
        logger.info(f"✅ 找到 {len(common_names)} 个配对样本。")

    pairs = [(image_files[name], mask_files[name]) for name in sorted(common_names)]
    return pairs

# --- 自定义 PyTorch Dataset (处理多列分类标签) ---
# --- 1. 分割 PyTorch Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self,image_mask_pairs: List[Tuple[str, str]],processor: AutoImageProcessor,
        size: int,is_training: bool = False, logger: Optional[logging.Logger] = None
    ):
        self.pairs = image_mask_pairs
        self.processor = processor
        self.size = size
        self.is_training = is_training
        self.logger = logger
        
        # 定义一个只用于 Mask 的 Resize/ToTensor 变换
        self.mask_transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST), # Mask 必须使用 NEAREST 插值
            T.ToTensor(), # 将 PIL Image 转换为 [1, H, W] 的 Long Tensor
        ])
        

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L") # 灰度图模式打开
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading image/mask at index {idx}: {e}")
            return None  # 返回 None 以便在 collate_fn 中处理   
        
        # 图像预处理
        resize_transform = T.Resize((self.size, self.size))
        image = resize_transform(image)
        mask = resize_transform(mask)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [C, H, W]
        mask_tensor = T.ToTensor()(mask).squeeze(0).long() 
        return pixel_values, mask_tensor, img_path

# --- 2. Collate Function 修改 ---
# 分割 Mask 的尺寸通常是相同的 (TARGET_IMAGE_SIZE x TARGET_IMAGE_SIZE)
def custom_segmentation_collate_fn(batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    # 过滤掉加载失败的项
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None # 返回 None 处理空批次

    pixel_values = torch.stack([item[0] for item in batch]) # [N, C, H, W]
    
    # 分割 Mask
    # item[1] 是 [H, W] 的 Long Tensor
    segmentation_masks = torch.stack([item[1] for item in batch]) # [N, H, W]

    img_paths = [item[2] for item in batch]
    return pixel_values, segmentation_masks, img_paths

# --- 自定义模型：DINOv3 + 多个分类头 ---

class DinoV3SegmentationModel(nn.Module):
    """
    基于 DINOv3 主干网络，带有多任务分类头。
    """
    def __init__(self, model_name: str, num_classes: int, size: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.input_device = torch.device(DEVICE)
        self.num_classes = NUM_CLASSES
        self.size = TARGET_IMAGE_SIZE # TARGET_IMAGE_SIZE

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
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.config.hidden_size, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(self.size, self.size), mode='bilinear', align_corners=False),
            nn.Conv2d(256, self.num_classes, kernel_size=1)
        )
        #   卡
        
        # 确保解码器参数是可训练的
        for param in self.decoder.parameters():
             param.requires_grad = True
        

        
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 运行主干网络（冻结）
        pixel_values = pixel_values.to(self.input_device)
        
        # 即使主干网络冻结，也要确保它在正确的设备上运行
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        patch_tokens = outputs.last_hidden_state[:, 1:, :] # (Batch, Num_Tokens, Feature_Dim)

        # 2. 重塑为特征图
        # Num_Tokens = (H/P)^2
        num_patches_per_side = int(patch_tokens.size(1)**0.5)
        
        # (Batch, Feature_Dim, H/P, W/P)
        feature_map = patch_tokens.permute(0, 2, 1).reshape(
            -1, self.backbone.config.hidden_size, num_patches_per_side, num_patches_per_side
        )

        # 3. 运行解码器
        logits = self.decoder(feature_map) # (Batch, Num_Classes, H, W)

        return logits

def calculate_segmentation_metrics(pred_masks, true_masks, num_classes, logger):
    """
    计算 mIoU 和 Pixel Accuracy。
    :param pred_masks: 预测的类别 ID (N, H, W)
    :param true_masks: 真实的类别 ID (N, H, W)
    """
    # 过滤掉 -1（未知/忽略）像素
    valid_pixels = (true_masks != -1)
    
    # 计算 mIoU
    jaccard = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=-1, average='macro').to(pred_masks.device)
    miou = jaccard(pred_masks, true_masks).item()
    
    # 计算 Pixel Accuracy (PA)
    pa = accuracy(pred_masks[valid_pixels], true_masks[valid_pixels], task="multiclass", num_classes=num_classes).item()

    metrics = {
        'miou': miou,
        'pixel_accuracy': pa
    }
    
    logger.info(f"评估指标: mIoU={miou:.4f}, PA={pa:.4f}")
    return metrics

# --- 训练函数 (新增日志和早停逻辑) ---
def train_multi_task_segmentation(logger: logging.Logger):
    # 1. 初始化预处理器
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME,do_resize=False)
    writer = SummaryWriter(log_dir=LOG_DIR)
    # --- TENSORBOARD 初始化 ---
    logger.info(f"TensorBoard Writer initialized at: {LOG_DIR}")
    best_model_path = os.path.join(LOG_DIR, "best_model.pth")

    # 读取数据集
    print("Train image dir exists?", os.path.exists(os.path.join(BASE_DATA_DIR, "train", "images")))
    train_pairs = get_image_mask_pairs(
        image_dir=os.path.join(BASE_DATA_DIR, "train", "images"),
        mask_dir=os.path.join(BASE_DATA_DIR, "train", "masks"),
        logger=logger
    )
    val_pairs = get_image_mask_pairs(
        image_dir=os.path.join(BASE_DATA_DIR, "valid", "images"),
        mask_dir=os.path.join(BASE_DATA_DIR, "valid", "masks"),
        logger=logger
    )
    test_pairs = get_image_mask_pairs(
        image_dir=os.path.join(BASE_DATA_DIR, "test", "images"),
        mask_dir=os.path.join(BASE_DATA_DIR, "test", "masks"),
        logger=logger
    )
    

    logger.info(f"数据集大小 -> Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    # 创建 Dataset 和 DataLoader
    train_dataset = SegmentationDataset(train_pairs, processor, TARGET_IMAGE_SIZE, is_training=True, logger=logger)
    val_dataset = SegmentationDataset(val_pairs, processor, TARGET_IMAGE_SIZE, is_training=False, logger=logger)
    test_dataset = SegmentationDataset(test_pairs, processor, TARGET_IMAGE_SIZE, is_training=False, logger=logger)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, collate_fn=custom_segmentation_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, collate_fn=custom_segmentation_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, collate_fn=custom_segmentation_collate_fn, pin_memory=True)
    
    # 初始化模型、损失函数和优化器
    model = DinoV3SegmentationModel(MODEL_NAME, num_classes=NUM_CLASSES, size=TARGET_IMAGE_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # 仅优化分类头参数 (假设主干网络冻结)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=LEARNING_RATE)

    # 初始化 GradScaler
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-4
    )
    logger.info(f"学习率调度器 ReduceLROnPlateau 已初始化，监控模式: max, 降低耐心值: 10")

    logger.info(f"模型已加载，在设备 {DEVICE} 上训练...")
    best_val_miou = -1.0
    patience_counter = 0

    # 4. 训练循环
    for epoch in range(NUM_EPOCHS):
        total_combined_loss = 0
        model.train()
        total_loss = 0.0
        num_batches = 0

        # 训练步骤
        for step, batch in enumerate(train_loader):
            if batch is None:
                logger.warning("Received an empty batch after filtering corrupt files. Skipping step.")
                continue
            pixel_values, true_masks, img_paths = batch  # ← 注意：不是 labels_dict
            pixel_values = pixel_values.to(DEVICE)
            true_masks = true_masks.to(DEVICE)  # [N, H, W], LongTensor

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE):
                logits = model(pixel_values)  # [N, num_classes, H, W]
                loss = criterion(logits, true_masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_combined_loss += loss.item()

        # 测试阶段
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None or batch[0] is None:
                    continue
                pixel_values, true_masks, _ = batch
                pixel_values = pixel_values.to(DEVICE)
                true_masks = true_masks.to(DEVICE)
                
                logits = model(pixel_values)
                preds = torch.argmax(logits, dim=1)  # [N, H, W]
                
                all_preds.append(preds.cpu())
                all_targets.append(true_masks.cpu())
        
        if all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            val_metrics = calculate_segmentation_metrics(all_preds, all_targets, NUM_CLASSES, logger)
            val_miou = val_metrics['miou']
            writer.add_scalar('mIoU/Val', val_miou, epoch + 1)
            writer.add_scalar('PixelAccuracy/Val', val_metrics['pixel_accuracy'], epoch + 1)

            # === 学习率调度 & 早停 & 模型保存 ===
            scheduler.step(val_miou)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"当前学习率: {current_lr:.6f}")

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_val_miou,
                }, best_model_path)
                logger.info(f"✅ 新最佳模型 (mIoU={best_val_miou:.4f}) 已保存至: {best_model_path}")
            else:
                patience_counter += 1
                logger.info(f"💔 验证 mIoU 未提升。早停计数: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                logger.info("🛑 触发早停！提前结束训练。")
                break
        else:
            logger.warning("验证阶段无有效批次。")
        writer.close()
        # === 最终：在 test 集上评估最佳模型 ===
        logger.info("🔍 开始在测试集上评估最佳模型...")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                if batch is None or batch[0] is None:
                    continue
                pixel_values, true_masks, _ = batch
                pixel_values = pixel_values.to(DEVICE)
                true_masks = true_masks.to(DEVICE)
                logits = model(pixel_values)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(true_masks.cpu())

        if all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            test_metrics = calculate_segmentation_metrics(all_preds, all_targets, NUM_CLASSES, logger)
            logger.info("🎉 测试集最终结果:")
            for k, v in test_metrics.items():
                logger.info(f"  {k.upper()}: {v:.4f}")
            # 可选：保存测试结果到文件
            with open(os.path.join(LOG_DIR, "test_results.txt"), "w") as f:
                f.write(f"miou: {test_metrics['miou']:.6f}\n")
                f.write(f"pixel_accuracy: {test_metrics['pixel_accuracy']:.6f}\n")
        else:
            logger.error("测试集评估失败：无有效数据。")

    return None


if __name__ == "__main__":
    # 初始化日志记录器
    main_logger = setup_logging()
    main_logger.info(f"日志文件已创建：{LOG_FILENAME}")
    main_logger.info(f"运行设备: {DEVICE}")
    main_logger.info(f"图像尺寸: {TARGET_IMAGE_SIZE}")
    main_logger.info(f"类别数: {NUM_CLASSES}")
    main_logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    main_logger.info(f"LEARNING_RATE: {LEARNING_RATE}")

    trained_model = train_multi_task_segmentation(main_logger)

    if trained_model:
        main_logger.info("\n最终模型已训练并加载。")
