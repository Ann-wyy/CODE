import pandas as pd

# 读取原始CSV
df = pd.read_csv('/home/yyi/data/pretrain_1022.csv')  # 替换为你的CSV文件名

# 定义前缀
prefix_main = '/data/dataserver02/public/data/004_XrayFM/6yuan_raw/usb1'
prefix_bone = '/data/dataserver02/public/data/004_XrayFM/hbrm'

# 筛选：属于bone_dataset的
df_bone = df[df['image_path'].str.startswith(prefix_bone)]

# 筛选：属于主目录但不属于bone_dataset的（避免重复）
df_main_only = df[
    df['image_path'].str.startswith(prefix_main) &
    ~df['image_path'].str.startswith(prefix_bone)
]

# 保存为两个CSV文件
df_main_only.to_csv('/home/yyi/data/6yuan_raw_dataset.csv', index=False)
df_bone.to_csv('/home/yyi/data/hbrm_dataset.csv', index=False)

print(f"主数据集行数: {len(df_main_only)}")
print(f"骨数据集行数: {len(df_bone)}")
print("✅ 已成功保存为 main_dataset.csv 和 bone_dataset.csv")