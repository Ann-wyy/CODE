import pandas as pd
import csv
import logging
import shutil
import os
from pathlib import Path

# ==================== 配置区（你只需改这里）====================
file1 = '/home/yyi/data/data_pretrain/bonecancer_原发.csv'          # 主表（含重复条码）
file2_original = '/home/yyi/data/data_pretrain/bonecancer_backup_part.csv'           # 原始临床表（可能格式不规范）
output_file = '/data/truenas_B2/yyi/datapath/bonecancer_trans.csv'

# 你最终要的列（按顺序！不存在的列会自动跳过）
final_columns = ['DICOM文件','检查项目','原发/转移','原发骨肿瘤病理结果','转移原发肿瘤','良性1/中间型2/恶性3','原发组织类型','转移标签','影像号','年龄','性别','BodyPart','image_path']

on_key = 'DICOM文件'  # 连接字段名
# ============================================================

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fix_csv_format(input_path, output_path):
    """使用 csv 模块安全读取并重新写入，自动处理跨行字段和引号"""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"源文件不存在: {input_path}")

    # 尝试读取（自动处理跨行字段）
    with open(input_path, 'r', encoding='utf-8', newline='') as f_in:
        reader = csv.reader(f_in)
        rows = list(reader)

    logging.info(f"成功读取 {len(rows)} 行（已处理跨行字段）")

    # 重新写入（自动加必要引号）
    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)

    logging.info(f"✅ 修复后的 CSV 已保存至: {output_path}")
    return str(output_path)

def safe_read_csv_with_encoding(file_path):
    """尝试 UTF-8，失败则回退到 GBK"""
    for enc in ['utf-8', 'gbk']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            logging.info(f"以 {enc} 编码成功读取: {file_path}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.warning(f"以 {enc} 读取失败: {e}")
    raise RuntimeError(f"无法以 UTF-8 或 GBK 读取 {file_path}")

def clean_key_series(series):
    """清洗连接键：统一为字符串、去空格、去不可见字符"""
    s = series.astype(str).str.strip()
    s = s.str.replace(r'[\s\u00A0\u200b\ufeff]+', '', regex=True)
    s = s.replace('nan', '')
    return s

def main():
    # 1. 修复临床表格式（关键！）
    file2_fixed = str(Path(file2_original).with_suffix('.fixed.csv'))
    fix_csv_format(file2_original, file2_fixed)

    # 2. 读取两个表
    df1 = safe_read_csv_with_encoding(file1)
    df2 = safe_read_csv_with_encoding(file2_fixed)

    logging.info(f"主表形状: {df1.shape}，列: {list(df1.columns)}")
    logging.info(f"临床表形状: {df2.shape}，列: {list(df2.columns)}")

    # 3. 检查连接键是否存在
    if on_key not in df1.columns:
        raise KeyError(f"主表缺少连接键 '{on_key}'，可用列: {list(df1.columns)}")
    if on_key not in df2.columns:
        raise KeyError(f"临床表缺少连接键 '{on_key}'，可用列: {list(df2.columns)}")

    # 4. 清洗连接键
    df1 = df1.copy()
    df2 = df2.copy()
    df1['_merge_key'] = clean_key_series(df1[on_key])
    df2['_merge_key'] = clean_key_series(df2[on_key])

    # 5. 左连接（保留主表所有行，包括重复）
    merged = pd.merge(df1, df2, on='_merge_key', how='left', suffixes=('', '_from_clinical'))

    # 6. 清理：删除辅助列和重复后缀列（保留主表的列）
    drop_cols = ['_merge_key'] + [col for col in merged.columns if col.endswith('_from_clinical')]
    merged = merged.drop(columns=drop_cols, errors='ignore')

    # 7. 按需选择输出列
    available_cols = [col for col in final_columns if col in merged.columns]
    missing_cols = [col for col in final_columns if col not in merged.columns]

    if missing_cols:
        logging.warning(f"以下列不存在，将跳过: {missing_cols}")
    if not available_cols:
        raise ValueError("没有一列是你指定的！请检查列名是否匹配。")

    output_df = merged[available_cols].copy()

    # 8. 保存结果
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f"✅ 合并完成！输出 {len(output_df)} 行")
    logging.info(f"💾 保存至: {output_file}")
    logging.info(f"📋 实际输出列: {list(output_df.columns)}")

    # 可选：清理临时文件
    try:
        os.remove(file2_fixed)
        logging.info(f"🧹 已删除临时文件: {file2_fixed}")
    except Exception as e:
        logging.warning(f"无法删除临时文件: {e}")

if __name__ == "__main__":
    main()