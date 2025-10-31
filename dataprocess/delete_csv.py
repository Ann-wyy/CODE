import pandas as pd
import os

def filter_by_barcode_count(file_path: str, count_threshold: int = 40):
    """
    Reads a CSV file, extracts a barcode from the 'image_path' column,
    and removes all rows for any barcode that appears more than a
    specified number of times.

    Args:
        file_path (str): The path to the input CSV file.
        count_threshold (int): The maximum allowed count for a barcode.
                               Barcodes with a count > this threshold will be removed.
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件不存在于 {file_path}")
        return

    print(f"正在读取文件：{file_path} ...")
    df = pd.read_csv(file_path)

    # 检查 'image_path' 列是否存在
    if 'image_path' not in df.columns:
        print("错误：CSV文件中未找到 'image_path' 列。")
        return
        
    # --- 1. 从 image_path 中提取条码 ---
    # 使用 apply 函数和 lambda 表达式，通过 '/' 分割路径并获取倒数第三个元素（即条码）
    print("正在从 image_path 中提取条码...")
    df['barcode'] = df['image_path'].apply(
        lambda x: x.split('/')[-3] if isinstance(x, str) and len(x.split('/')) > 3 else None
    )

    # --- 2. 统计每个条码的出现次数 ---
    barcode_counts = df['barcode'].value_counts()
    print("条码统计完成。")
    
    # --- 3. 找出需要删除的条码（出现次数 > 40） ---
    barcodes_to_remove = barcode_counts[barcode_counts > count_threshold].index.tolist()

    if not barcodes_to_remove:
        print(f"没有找到出现次数超过 {count_threshold} 次的条码。无需删除。")
        return
        
    print(f"\n找到以下需要删除的条码 (出现次数 > {count_threshold})：")
    for barcode in barcodes_to_remove:
        print(f"  - {barcode} (出现次数：{barcode_counts[barcode]})")
    '''
    # --- 4. 过滤 DataFrame，删除包含这些条码的行 ---
    original_shape = df.shape
    df_filtered = df[~df['barcode'].isin(barcodes_to_remove)].copy()
    
    print(f"\n原始 DataFrame 行数: {original_shape[0]}")
    print(f"过滤后 DataFrame 行数: {df_filtered.shape[0]}")
    
    # --- 5. 保存过滤后的数据到新文件 ---
    # 为了安全起见，保存到新文件，避免覆盖原始数据
    output_path = file_path.replace('.csv', '_filtered.csv')
    df_filtered.to_csv(output_path, index=False)
    print(f"过滤后的数据已成功保存到：{output_path}")
    '''

# --- 执行脚本 ---
# 定义文件路径
file_path = "/home/yyi/data/shukun_2T_s.csv"

# 调用函数执行操作
filter_by_barcode_count(file_path)