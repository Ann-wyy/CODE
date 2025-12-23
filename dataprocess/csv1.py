import os
import pandas as pd
import logging
from pathlib import Path

# 配置日志（符合你的偏好）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_dicom_files(root_dir: str) -> list:
    """遍历目录，查找 DICOM 文件（.dcm 或无扩展名）"""
    dicom_paths = []
    root = Path(root_dir)

    if not root.exists():
        logging.error(f"根目录不存在: {root_dir}")
        return []

    for file_path in root.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            # 判断是否为 DICOM 文件
            if file_path.suffix.lower() == '.dcm' or file_path.suffix == '':
                dicom_paths.append(str(file_path.resolve()))
    logging.info(f"共找到 {len(dicom_paths)} 个 DICOM 文件")
    return dicom_paths

def extract_two_labels(file_path: str):
    """从路径提取倒数第二级目录 和 文件名（无扩展）"""
    p = Path(file_path)
    parts = p.parts  # 所有路径组件

    if len(parts) < 2:
        # 路径太短，无法提取两级
        return "", os.path.splitext(p.name)[0]

    label_dir = parts[-2]      # 倒数第二级目录
    label_file = p.stem        # 文件名（不含扩展名）
    return label_dir, label_file

def generate_csv_with_two_labels(dicom_paths: list, output_csv: str):
    records = []
    for path in dicom_paths:
        label_dir, label_file = extract_two_labels(path)
        records.append({
            'DICOM文件': path,
            'label_dir': label_dir,    # 倒数第二级目录
            'label_file': label_file   # 文件名（无扩展）
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logging.info(f"CSV 已保存至: {output_csv}")

# ===== 主程序 =====
if __name__ == "__main__":
    root_folder = "/data/dataserver02/public/data/004_XrayFM"
    output_csv_path = "/home/yyi/dicom_paths_with_label.csv"

    dicom_files = find_dicom_files(root_folder)
    if dicom_files:
        generate_csv_with_two_labels(dicom_files, output_csv_path)
    else:
        logging.warning("未找到任何 DICOM 文件")
