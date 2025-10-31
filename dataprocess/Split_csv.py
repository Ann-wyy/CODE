import pandas as pd
from sklearn.model_selection import train_test_split

# ====== 在这里修改你的配置 ======
input_csv = "/home/yyi/data/BTXRD_dataset.csv"      # 输入的 CSV 文件路径
output_train = '/home/yyi/data/BTXRD_train.csv'
output_val = '/home/yyi/data/BTXRD_val.csv'
stratify_col = "tumor"           # 用于分层的列名（改成你自己的列名）
test_size = 0.2                  # 验证集比例（0.2 = 20%）
random_state = 42                # 随机种子，保证结果可复现
# ================================

# 读取数据
df = pd.read_csv(input_csv)

# 分层划分
train_df, val_df = train_test_split(
    df,
    test_size=test_size,
    stratify=df[stratify_col],
    random_state=random_state
)

# 保存结果
train_df.to_csv(output_train, index=False)
val_df.to_csv(output_val, index=False)

# 打印信息
print(f"原始数据量: {len(df)}")
print(f"训练集: {len(train_df)} → 已保存为 train.csv")
print(f"验证集: {len(val_df)} → 已保存为 val.csv")

# 可选：显示各类别分布
print("\n训练集类别分布:")
print(train_df[stratify_col].value_counts().sort_index())
print("\n验证集类别分布:")
print(val_df[stratify_col].value_counts().sort_index())