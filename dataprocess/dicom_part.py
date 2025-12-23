import pandas as pd
import pydicom
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_body_part_safe(dicom_path):
    """
    安全地从 DICOM 文件中提取 BodyPartExamined，不加载像素。
    若失败或字段缺失，返回 None。
    """
    try:
        # stop_before_pixels=True: 不加载像素；force=True: 容忍部分头信息缺失
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        body_part = getattr(ds, 'BodyPartExamined', None)
        return body_part.strip() if body_part and isinstance(body_part, str) else None
    except Exception as e:
        logging.warning(f"读取 DICOM 文件失败: {dicom_path} | 错误: {e}")
        return None

def add_body_part_to_csv(csv_path, dicom_column='DICOM文件', output_path=None):
    """
    从 CSV 中读取 DICOM 路径列，提取 BodyPartExamined，新增一列并保存。
    
    参数:
        csv_path: 输入 CSV 路径
        dicom_column: 包含 DICOM 文件路径的列名
        output_path: 输出 CSV 路径，若为 None 则覆盖原文件（建议先备份）
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    if dicom_column not in df.columns:
        raise ValueError(f"CSV 中未找到列 '{dicom_column}'，当前列: {list(df.columns)}")

    logging.info(f"开始处理 {len(df)} 行 DICOM 文件...")

    # 提取部位信息
    df['BodyPart'] = df[dicom_column].apply(
        lambda p: extract_body_part_safe(p) if pd.notna(p) and Path(p).exists() else None
    )

    # 设置输出路径
    if output_path is None:
        output_path = csv_path  # 默认覆盖（谨慎）

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"处理完成，结果已保存至: {output_path}")

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 请根据实际情况修改以下路径
    input_csv = "/home/yyi/data/data_pretrain/isCancer.csv"      # 输入 CSV
    output_csv = "/home/yyi/data/data_pretrain/isCancer_part.csv"  # 输出 CSV

    add_body_part_to_csv(
        csv_path=input_csv,
        dicom_column='DICOM文件',      # 确保列名与你的 CSV 一致
        output_path=output_csv
    )