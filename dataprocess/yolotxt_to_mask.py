import os
import cv2
import numpy as np
from pathlib import Path

# ================== 【请根据你的实际情况修改以下路径和参数】 ==================
IMAGES_DIR = "/data/truenas_B2/yyi/data/BoneFractureYolo8/valid/images"
LABELS_DIR = "/data/truenas_B2/yyi/data/BoneFractureYolo8/valid/labels"
OUTPUT_MASKS_DIR = "/data/truenas_B2/yyi/data/BoneFractureYolo8/valid/masks"

# 如果你的数据是多类别（如 class 0=肱骨骨折, 1=股骨骨折），设 num_classes > 1
# 如果只是二分类（有骨折/无骨折），设 num_classes = 1（所有框都标为 1）
NUM_CLASSES = 1  # 1 表示二值掩码；>1 表示保留原始类别（像素值 = class_id + 1）

# 支持的图像扩展名
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
# ========================================================================

def yolo_to_mask(txt_path, img_shape, num_classes=1):
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if not os.path.exists(txt_path):
        return mask

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_w = float(parts[3])
            box_h = float(parts[4])

            x1 = int((x_center - box_w / 2) * W)
            y1 = int((y_center - box_h / 2) * H)
            x2 = int((x_center + box_w / 2) * W)
            y2 = int((y_center + box_h / 2) * H)

            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))

            pixel_value = 1 if num_classes == 1 else (class_id + 1)
            mask[y1:y2, x1:x2] = pixel_value

        except Exception as e:
            print(f"警告：解析 {txt_path} 时出错: {e}")
            continue

    return mask

def convert_yolo_to_masks(images_dir, labels_dir, output_masks_dir, num_classes=1, image_exts=IMAGE_EXTS):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_masks_dir = Path(output_masks_dir)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_exts]
    print(f"🔍 找到 {len(image_files)} 张图像")

    for img_path in image_files:
        txt_path = labels_dir / (img_path.stem + ".txt")
        mask_path = output_masks_dir / (img_path.stem + ".png")

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ 无法读取图像: {img_path}")
                continue
            img_shape = img.shape
        except Exception as e:
            print(f"⚠️ 读取图像失败 {img_path}: {e}")
            continue

        mask = yolo_to_mask(txt_path, img_shape, num_classes=num_classes)
        cv2.imwrite(str(mask_path), mask)
        print(f"✅ 保存掩码: {mask_path}")

    print(f"\n🎉 转换完成！掩码已保存至: {output_masks_dir}")

if __name__ == "__main__":
    convert_yolo_to_masks(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_masks_dir=OUTPUT_MASKS_DIR,
        num_classes=NUM_CLASSES
    )