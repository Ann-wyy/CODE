import os
import pandas as pd

# 设置文件夹路径（请根据你的实际路径修改）
non_fractured_dir = '/data/truenas_B2/Dataset/001_6yXray/bone_dataset/FracAtlas/images/Non_fractured'
fractured_dir = '/data/truenas_B2/Dataset/001_6yXray/bone_dataset/FracAtlas/images/Fractured'

# 支持的图像扩展名（可根据需要增删）
extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

def get_image_paths_and_labels(folder_path, label):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                full_path = os.path.abspath(os.path.join(root, file))
                data.append((full_path, label))
    return data

# 收集两个类别的数据
data = []
data.extend(get_image_paths_and_labels(non_fractured_dir, 'Non_fractured'))
data.extend(get_image_paths_and_labels(fractured_dir, 'Fractured'))

# 转为 DataFrame
df = pd.DataFrame(data, columns=['image_path', 'Fractured'])

# 可选：打乱顺序（对训练集常用）
# df = df.sample(frac=1).reset_index(drop=True)

# 保存为 CSV
df.to_csv('fracture_dataset.csv', index=False, encoding='utf-8-sig')

print(f"✅ 已生成 CSV 文件，共 {len(df)} 条记录。")
print("前几行预览：")
print(df.head())