import pandas as pd
import sys

def clean_and_replace(
    csv_input,
    remove_column,      # 要检查并删除 -1 的列名
    replace_column=None, # 要执行 3→2 替换的列名（若为 None，则与 remove_column 相同）
    csv_output=None
):
    if replace_column is None:
        replace_column = remove_column  # 默认对同一列操作

    if csv_output is None:
        csv_output = csv_input.replace('.csv', '_cleaned.csv')

    try:
        df = pd.read_csv(csv_input)
        print(f"✅ 读取文件：{csv_input}，共 {len(df)} 行")
    except Exception as e:
        print(f"❌ 读取失败：{e}")
        sys.exit(1)

    # 检查列是否存在
    for col in [remove_column, replace_column]:
        if col not in df.columns:
            print(f"⚠️ 列 '{col}' 不存在！可用列：{list(df.columns)}")
            sys.exit(1)

    # 🔹 步骤1：删除 remove_column 中值为 -1 的行（兼容 str/int/float）
    mask_remove = ~df[remove_column].astype(str).str.strip().isin(['-1', '-1.0'])
    df_cleaned = df[mask_remove].copy()
    removed_count = len(df) - len(df_cleaned)
    print(f"🗑️  删除 {removed_count} 行（{remove_column} 列值为 -1）")

    # 🔹 步骤2：将 replace_column 中所有 3 → 2（兼容数值型和字符串型 '3'）
    # 先尝试转换为数值（避免字符串'3'漏掉），但保留原始类型风格
    before_replace = (df_cleaned[replace_column] == 3).sum() + (df_cleaned[replace_column].astype(str) == '3').sum()
    
    # 安全替换：仅当值等于 3（数值或字符串）时替换为整数 2
    df_cleaned[replace_column] = df_cleaned[replace_column].replace({'3': 2, 3: 2})
    
    after_replace = (df_cleaned[replace_column] == 2).sum()
    replaced_count = before_replace  # 因为只有 3→2，且原2不变，所以替换数 = 原3的数量
    print(f"🔄 将 {replace_column} 列中 {replaced_count} 个 '3' 或 3 替换为 2")

    # 保存
    try:
        df_cleaned.to_csv(csv_output, index=False, encoding='utf-8-sig')
        print(f"✅ 处理完成！剩余 {len(df_cleaned)} 行 → 已保存为：{csv_output}")
    except Exception as e:
        print(f"❌ 保存失败：{e}")
        sys.exit(1)

# ——— 使用示例 ———
if __name__ == "__main__":
    INPUT_CSV = "/home/yyi/data/data_pretrain/原发转移骨肿瘤_匹配合并结果_含部位_0.csv"

    # ✅ 关键配置 ↓
    REMOVE_COL = "良性1/中间型2/恶性3"      # 删除该列中 -1 的行
    REPLACE_COL = "良性1/中间型2/恶性3"     # 将该列中 3 改为 2（可设为不同列，如 "class"）

    OUTPUT_CSV = "/home/yyi/data/data_pretrain/bonecancer_begin.csv"

    clean_and_replace(
        csv_input=INPUT_CSV,
        remove_column=REMOVE_COL,
        replace_column=REPLACE_COL,   # ← 可改为其他列名
        csv_output=OUTPUT_CSV
    )