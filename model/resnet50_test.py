import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Any, Tuple
import time

#--------------self------
from utils.utils import set_seed, convert_dinov3_teacher_to_hf_state_dict, preprocess_labels_and_setup_datasets
from utils.metrics import calculate_metrics, log_metrics_to_tensorboard, evaluate


# --- 1. 配置和超参数 (修改此处!) ---
class Config:
    TRAIN_CSV = "/home/yyi/data/test_dataset/BoneCancer/bone_cancer_train.csv"
    VAL_CSV = "/home/yyi/data/test_dataset/BoneCancer/bone_cancer_val.csv"
    TEST_CSV = "/home/yyi/data/test_dataset/BoneCancer/bone_cancer_test.csv"
    IMAGE_PATH_COLUMN = 'image_path' # CSV中包含图像相对路径的列名
    LABEL_COLUMNS = ['原发/转移','良恶性']  # 您的所有标签列名
    
    # *** 关键：定义您的所有分类任务及其类别数 ***
    # 假设您的 CSV 中有 'tumor_type', 'aggressiveness', 'calcium_score' 三列作为标签
    TASK_CONFIG = {
        '原发/转移': 2,  # 2 类别二分类
        '良恶性': 3    # 3 类别多分类
    }
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 80
    IMAGE_SIZE = 224
    DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

cfg = Config()

RANDOM_SEED = 42
TRAIN_NAME = f"bonecancer"
TEST_NAME = f"{TRAIN_NAME}_ResNet50_{cfg.IMAGE_SIZE}_{cfg.LEARNING_RATE}_{RANDOM_SEED}"
LOG_DIR = f"/data/truenas_B2/yyi/logs/{TRAIN_NAME}/{TEST_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f"{TEST_NAME}_{time.strftime('%Y%m%d-%H%M%S')}.log")
# 设置日志
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

# --- 2. 自定义数据集类 (加载多个标签) ---
class MultiTaskImageDataset(Dataset):
    # 删除了 img_root_dir 参数，并接收路径列名 path_column
    def __init__(self, csv_file, task_config, path_column: str, transform=None):
        self.data_frame = csv_file
        self.transform = transform
        self.task_names = list(task_config.keys())
        self.path_column = path_column # 存储 CSV 中路径列的名称
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx][self.path_column] 
        
        # 加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # 记录错误时直接使用路径
            logger.error(f"Error loading image: {img_path}. Skipping. Error: {e}")
            raise IndexError("Image load error")

        if self.transform:
            image = self.transform(image)
            
        # 加载所有任务的标签 (保持不变)
        labels = {}
        for task_name in self.task_names:
            labels[task_name] = torch.tensor(self.data_frame.iloc[idx][task_name], dtype=torch.long)
            
        return image, labels, img_path

# --- 3. 多头 ResNet 模型 ---
class MultiTaskResNet(nn.Module):
    """带有多任务分类头的 ResNet-50 模型"""
    def __init__(self, task_config: Dict[str, int]):
        super().__init__()
        # 1. 加载预训练 ResNet-50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 2. 移除原分类头，获得特征向量
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() 

        # 3. 创建多任务分类头
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_config.items():
            # 每个任务一个独立的线性层
            out_dim = 1 if num_classes == 2 else num_classes 
            self.task_heads[task_name] = nn.Linear(num_ftrs, out_dim)
            
    def forward(self, x):
        # 提取特征
        features = self.resnet(x)
        
        # 通过各自的分类头得到输出 (logits)
        output = {}
        for task_name, head in self.task_heads.items():
            output[task_name] = head(features)
        return output



# --- 5. 训练和评估循环 (多任务适应) ---

def train_one_epoch(model, dataloader, criteria, optimizer, device, task_config):
    

    model.train()
    total_loss = 0.0
    
    for inputs, labels_dict, _ in dataloader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs_dict = model(inputs)
        
        # 计算总损失 (所有任务损失的加权和，这里使用等权重)
        loss = 0
        for task_name, output in outputs_dict.items():
            labels = labels_dict[task_name].to(device)
            num_cls = task_config[task_name]
            
            # 过滤无效标签，您的评估函数假设无效标签为 -1
            valid_mask = (labels != -1)
            valid_count = valid_mask.sum()

            if valid_count > 0:
                valid_labels = labels[valid_mask]
                valid_predictions = output[valid_mask]

                if num_cls == 2:
                    # 二分类：使用 BCEWithLogitsLoss，需要 FloatTensor，形状为 [N, 1]
                    target = valid_labels.float().view(-1, 1)
                    task_loss = criteria[task_name](valid_predictions, target)
                else:
                    # 多分类：使用 CrossEntropyLoss，需要 LongTensor
                    target = valid_labels.long() 
                    task_loss = criteria[task_name](valid_predictions, target)
                
                loss += task_loss
        
        if loss != 0:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss


# --- 6. 主程序执行 ---
if __name__ == '__main__':
    logger.info(f"TensorBoard Writer initialized at: {LOG_DIR}")
    best_model_path = os.path.join(LOG_DIR, "best_model.pth")
    
    # 数据加载器的配置
    train_transforms = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_df, val_df, test_df, final_num_classes_dict = preprocess_labels_and_setup_datasets(cfg.TRAIN_CSV, cfg.VAL_CSV, cfg.TEST_CSV, cfg.LABEL_COLUMNS, cfg.IMAGE_PATH_COLUMN, logger)
    cfg.TASK_CONFIG = final_num_classes_dict

    train_dataset = MultiTaskImageDataset(train_df, cfg.TASK_CONFIG,cfg.IMAGE_PATH_COLUMN, train_transforms)
    val_dataset = MultiTaskImageDataset(val_df, cfg.TASK_CONFIG,cfg.IMAGE_PATH_COLUMN, train_transforms)
    test_dataset = MultiTaskImageDataset(test_df, cfg.TASK_CONFIG,cfg.IMAGE_PATH_COLUMN, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    logger.info(f"数据集已加载。任务列表: {list(cfg.TASK_CONFIG.keys())}")
    logger.info(f"数据集 -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 模型、优化器和损失函数
    model = MultiTaskResNet(cfg.TASK_CONFIG).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    criteria = {}
    for task_name, num_cls in cfg.TASK_CONFIG.items():
        if num_cls == 2:
            # 二分类任务使用 BCEWithLogitsLoss
            criteria[task_name] = nn.BCEWithLogitsLoss()
        else:
            # 多分类任务使用 CrossEntropyLoss
            criteria[task_name] = nn.CrossEntropyLoss()

    # 主训练循环
    logger.info(f"开始在 {cfg.DEVICE} 上训练...")
    best_main_task_acc = 0.0 # 可以选择一个主要任务来保存最佳模型
    main_task_name = list(cfg.TASK_CONFIG.keys())[0]

    for epoch in range(cfg.NUM_EPOCHS):
        # 训练
        logger.info(f"\n======== Epoch {epoch+1}/{cfg.NUM_EPOCHS} ========")
        logger.info(f"训练总损失: {train_loss:.4f}")
        train_loss = train_one_epoch(model, train_loader, criteria, optimizer, cfg.DEVICE, cfg.TASK_CONFIG)
        
        # 验证
        val_metrics = evaluate(model=model, data_loader=val_loader, criterion_dict=criteria, task_names=list(cfg.TASK_CONFIG.keys()), 
            num_classes_dict=cfg.TASK_CONFIG, device=cfg.DEVICE, mode='Validation', logger=logger)
        
        # 使用第一个任务（如 'tumor_type'）的准确率来保存模型
        if main_task_name in val_metrics:
            current_acc = val_metrics[main_task_name].get('auroc', 0.0)
            if current_acc > best_main_task_acc:
                best_main_task_acc = current_acc
                torch.save(model.state_dict(), best_model_path)
                logger.info(">>> 改进, 保存模型.-------------------------------------------------------------------------------------------")

    logger.info(f"训练完成. 最佳 {main_task_name} AUROC: {best_main_task_acc:.4f}")

    # 最终测试和性能报告
    logger.info("\n--- 开始最终测试集评估 ---")
    model.load_state_dict(torch.load('best_multitask_resnet_model.pth'))
    test_metrics = evaluate(model=model, data_loader=test_loader, criterion_dict=criteria, 
        task_names=list(cfg.TASK_CONFIG.keys()), num_classes_dict=cfg.TASK_CONFIG, device=cfg.DEVICE, mode='Test', logger=logger)

    for task_name, metrics in test_metrics.items():
        logger.info(f"\n--- 任务: {task_name.upper()} 最终指标 ---")
        for key, value in metrics.items():
            if key in ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc']:
                logger.info(f"{key.upper()}: {value*100:.2f}%")
            else:
                logger.info(f"{key.upper()}: {value}")