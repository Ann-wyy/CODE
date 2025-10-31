import pandas as pd
from sklearn.model_selection import train_test_split

# ==================== 配置 ====================
input_csv = '/home/yyi/data/bone_cancer_png_path.csv'
output_train = 'bone_cancer_train.csv'
output_val = 'bone_cancer_val.csv'
val_ratio = 0.2
random_state = 42

# 是否启用分层划分？
use_stratify = True

# 指定用于分层的列（必须是标签列之一）
stratify_column = '良恶性'  # ← 只用于分层，但不会影响保存哪些列

# 所有你想保存的列（包括图像路径 + 所有标签列）
# 格式：{'原始列名': '输出列名'}
output_columns = {
    'PNG路径': 'image_path',
    '条码号': 'barcode',
    '病理结果': '病理结果',
    '良恶性': '良恶性',
    # 可继续添加其他标签列...
}

# ==================================================

# 1. 读取数据
df = pd.read_csv(input_csv, encoding='utf-8-sig')

# 必需的基础列
required_cols = ['PNG路径', 'PatientID']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"CSV 必须包含列: {required_cols}")

# 检查 output_columns 中的所有原始列是否存在
missing_cols = [col for col in output_columns.keys() if col not in df.columns]
if missing_cols:
    raise ValueError(f"以下指定输出的原始列不存在于 CSV 中: {missing_cols}")

# 如果启用分层，检查 stratify_column 是否存在
if use_stratify:
    if stratify_column not in df.columns:
        raise ValueError(f"分层列 '{stratify_column}' 不存在于数据中！")
    if stratify_column not in output_columns:
        print(f"⚠️ 警告：分层列 '{stratify_column}' 不在 output_columns 中，但仍可用于分层。")

print(f"📊 总图像数: {len(df)}")
print(f"👥 总病人数: {df['PatientID'].nunique()}")

# 2. 按病人分层划分
if use_stratify:
    # 为每个病人确定一个分层标签（取众数）
    patient_stratify_map = df.groupby('PatientID')[stratify_column].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )
    all_patients = patient_stratify_map.index.tolist()
    all_stratify_labels = patient_stratify_map.tolist()

    print(f"🏷️  分层依据列 '{stratify_column}' 的病人级别分布:")
    print(pd.Series(all_stratify_labels).value_counts())

    try:
        train_patients, val_patients = train_test_split(
            all_patients,
            test_size=val_ratio,
            stratify=all_stratify_labels,
            random_state=random_state
        )
    except ValueError as e:
        print("⚠️ 分层划分失败（类别太少或不平衡），回退到随机划分")
        train_patients, val_patients = train_test_split(
            all_patients,
            test_size=val_ratio,
            random_state=random_state
        )
else:
    unique_patients = df['PatientID'].unique()
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=val_ratio,
        random_state=random_state
    )

# 3. 划分图像
train_pids = set(train_patients)
val_pids = set(val_patients)

def assign_split(pid):
    if pid in train_pids:
        return 'train'
    elif pid in val_pids:
        return 'val'
    else:
        return 'ignore'

df['split'] = df['PatientID'].apply(assign_split)

# 4. 拆分数据
train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'val'].copy()

# 5. 选择并重命名列（保留所有 output_columns 指定的列）
def apply_column_mapping(df_subset):
    return df_subset[list(output_columns.keys())].rename(columns=output_columns)

train_final = apply_column_mapping(train_df)
val_final = apply_column_mapping(val_df)

# 6. 保存
train_final.to_csv(output_train, index=False, encoding='utf-8-sig')
val_final.to_csv(output_val, index=False, encoding='utf-8-sig')

# 7. 输出统计
print(f"\n✅ 划分完成！")
print(f"📁 训练集: {len(train_final)} 张图像 | {len(train_pids)} 位病人 → {output_train}")
print(f"📁 验证集: {len(val_final)} 张图像 | {len(val_pids)} 位病人 → {output_val}")

# 如果分层列被保存了，打印其分布
if use_stratify and stratify_column in output_columns:
    out_name = output_columns[stratify_column]
    print(f"\n📊 训练集 '{out_name}' 分布:")
    print(train_final[out_name].value_counts().sort_index())
    print(f"\n📊 验证集 '{out_name}' 分布:")
    print(val_final[out_name].value_counts().sort_index())