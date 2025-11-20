import pandas as pd

excel_path = '/home/yyi/data/data_pretrain/原发转移骨肿瘤_匹配合并结果.xlsx'
output_csv = '/home/yyi/data/data_pretrain/bone_cancer.csv'

# ✅ 指定哪些列要强制转为整数（缺失值替换为 -1）
int_columns = ['原发/转移', '影像号','原发良性1/中间型2/恶性3']  # ← 请根据你的实际列名调整

df = pd.read_excel(excel_path)

# 对每列进行转换
for col in int_columns:
    if col in df.columns:
        # 1. 将非数字值（如文本）转为 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 2. 将 NaN 替换为 -1
        df[col] = df[col].fillna(-1)
        # 3. 强制转为整数（此时已无 NaN，全是数字）
        df[col] = df[col].astype('int64')

# 保存为 CSV
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"✅ 已将 {excel_path} 转换为 {output_csv}")