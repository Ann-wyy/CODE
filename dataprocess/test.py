import pandas as pd

# === 配置部分 ===
input_excel = "/home/yyi/data/dataset.xlsx"          # 输入的 Excel 文件路径
output_csv = "/home/yyi/data/BTXRD_dataset.csv"           # 输出的 CSV 文件路径
required_columns = ["image_id", "tumor", "benign", "malignant"]  # 你要保留的列名（按需修改）
image_prefix = "/data/truenas_B2/Dataset/001_6yXray/bone_dataset/BTXRD/images/"  # path列前缀

# === 读取 Excel ===
df = pd.read_excel(input_excel)

# === 检查所需列是否存在 ===
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"以下列在 Excel 中不存在: {missing_cols}")

# === 只保留需要的列 ===
df_filtered = df[required_columns].copy()

# === 给 imagepath 列添加前缀（处理 NaN 或空值）===
df_filtered["image_id"] = df_filtered["image_id"].fillna("").astype(str).apply(
    lambda x: image_prefix + x if x.strip() != "" else ""
)

# === 保存为 CSV ===
df_filtered.to_csv(output_csv, index=False, encoding="utf-8")

print(f"✅ 已成功将 {input_excel} 转换为 {output_csv}，仅保留列: {required_columns}")