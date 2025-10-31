import os
import pandas as pd

# 支持的图像扩展名（可根据需要添加）
IMAGE_EXTENSIONS = {'.npz'}

def create_image_csv(image_folder, output_csv):
    # 用于保存图像路径的列表
    image_path = []

    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(image_folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(root, file)
                image_path.append(full_path)

    # 创建 DataFrame 并写入 CSV
    df = pd.DataFrame(image_path, columns=['image_path'])
    df.to_csv(output_csv, index=False)

    print(f"✅ 已生成 CSV 文件: {output_csv}")
    print(f"总共找到图像数量: {len(image_path)}")

# 示例调用
if __name__ == '__main__':
    image_folder = '/data/truenas_B2/Dataset/001_6yXray/pretrain_1022'     # 替换为你的图像文件夹路径
    output_csv = '/home/yyi/data/pretrain_1024.csv'  # 替换为你想保存的CSV路径

    create_image_csv(image_folder, output_csv)