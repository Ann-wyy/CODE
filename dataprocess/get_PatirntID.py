import pydicom
import csv
import os
import tempfile
from pathlib import Path

def update_csv_with_accession_number(
    csv_path: str,
    dicom_path_column: str = 'image_path',
    accession_column_name: str = 'accession_number'
):
    """
    直接在原 CSV 文件中新增一列 accession_number（影像检查号），
    使用临时文件安全覆盖原文件。
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"❌ CSV 文件不存在: {csv_path}")

    # 读取原文件内容（自动处理 BOM）
    with open(csv_path, 'r',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=',')
        original_fieldnames = reader.fieldnames
        print("🔍 CSV 实际列名（请核对）:", original_fieldnames)

        if dicom_path_column not in original_fieldnames:
            raise ValueError(f"❌ CSV 必须包含列 '{dicom_path_column}'")

        # 若已存在该列，可选择跳过或覆盖（这里选择覆盖）
        rows = []
        total, success = 0, 0
        for row in reader:
            total += 1
            dicom_path = row.get(dicom_path_column, '').strip()
            accession = "Unknown"

            if dicom_path and os.path.isfile(dicom_path):
                try:
                    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                    accession = str(getattr(ds, 'PatientID', 'Unknown')).strip() or 'Unknown'
                    accession = str(accession).strip()
                    success += 1
                    print(f"✅ 读取成功 {dicom_path}: {accession}")
                except Exception as e:
                    print(f"⚠️ 读取失败 {dicom_path}: {str(e)[:150]}")
                    accession = "ReadError"
            else:
                print(f"⚠️ 路径无效或文件不存在: {dicom_path}")
                accession = "FileNotFound"

            row[accession_column_name] = accession
            rows.append(row)

    # 确保新列在最后（或按需调整顺序）
    if accession_column_name not in original_fieldnames:
        fieldnames = original_fieldnames + [accession_column_name]
    else:
        fieldnames = original_fieldnames  # 已存在，覆盖值

    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', delete=False) as tmpfile:
        writer = csv.DictWriter(tmpfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = tmpfile.name

    # 原子替换原文件（保留权限）
    try:
        os.replace(tmp_path, csv_path)
        print(f"\n✅ 成功更新原文件！共 {total} 行，成功读取 {success} 个 accession number")
        print(f"📁 文件已就地更新: {csv_path}")
    except Exception as e:
        os.unlink(tmp_path)
        raise RuntimeError(f"❌ 覆盖原文件失败: {e}")


# ==============================
# 使用示例
# ==============================
if __name__ == "__main__":
    # ⚠️ 直接修改原文件，请确保已备份重要数据！
    CSV_FILE = "/home/yyi/data/data_pretrain/fracture.csv"

    update_csv_with_accession_number(
        csv_path=CSV_FILE,
        dicom_path_column='DICOM文件',
        accession_column_name='影像号'
    )