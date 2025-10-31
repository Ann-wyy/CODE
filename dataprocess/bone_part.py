import pydicom
import csv
import os
from collections import defaultdict

def extract_dicom_metadata_to_csv(input_csv, output_csv, failed_csv=None, path_column='image_path'):
    """
    从 CSV 读取 DICOM 文件路径，提取 BodyPartExamined 等元数据，保存到新 CSV
    不加载像素数据，高效处理大量文件
    可选：将失败路径保存到 failed_csv
    """
    body_part_count = defaultdict(int)
    processed_count = 0
    error_count = 0

    # 打开输出文件（成功+失败）
    with open(output_csv, 'w', newline='', encoding='utf-8') as out_f:
        fieldnames = [
            'dicom_path',
            'body_part_examined',
            'modality',
            'study_description',
            'status'
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        # 可选：打开失败记录文件
        failed_writer = None
        if failed_csv:
            failed_f = open(failed_csv, 'w', newline='', encoding='utf-8')
            failed_fieldnames = ['dicom_path', 'error_message']
            failed_writer = csv.DictWriter(failed_f, fieldnames=failed_fieldnames)
            failed_writer.writeheader()

        # 读取输入 CSV
        with open(input_csv, 'r', encoding='utf-8-sig') as in_f:  # 修复BOM问题
            reader = csv.DictReader(in_f)
            if path_column not in reader.fieldnames:
                raise ValueError(f"❌ 输入 CSV 必须包含列 '{path_column}'")

            for row in reader:
                dicom_path = row[path_column].strip()
                result = {
                    'dicom_path': dicom_path,
                    'body_part_examined': 'Unknown',
                    'modality': 'Unknown',
                    'study_description': '',
                    'status': 'success'
                }

                try:
                    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

                    result['body_part_examined'] = (getattr(ds, 'BodyPartExamined', 'Unknown') or 'Unknown').strip()
                    result['modality'] = (getattr(ds, 'Modality', 'Unknown') or 'Unknown').strip()
                    result['study_description'] = (getattr(ds, 'StudyDescription', '') or '').strip()

                    body_part_count[result['body_part_examined']] += 1
                    processed_count += 1

                except Exception as e:
                    error_msg = str(e)[:200]  # 截断避免超长
                    result['status'] = f'read_error: {error_msg}'
                    error_count += 1
                    print(f"⚠️ 读取失败 {dicom_path}: {error_msg}")

                    # 记录到失败文件
                    if failed_writer:
                        failed_writer.writerow({
                            'dicom_path': dicom_path,
                            'error_message': error_msg
                        })

                writer.writerow(result)

        # 关闭失败文件（如果打开）
        if failed_csv:
            failed_f.close()

    # 打印统计
    print(f"\n✅ 处理完成!")
    print(f"   总文件数: {processed_count + error_count}")
    print(f"   成功: {processed_count} | 失败: {error_count}")
    print(f"\n📊 BodyPartExamined 统计:")
    for part, count in sorted(body_part_count.items(), key=lambda x: x[1], reverse=True):
        print(f"   {part}: {count}")

    print(f"\n💾 主结果已保存至: {output_csv}")
    if failed_csv:
        print(f"💾 失败记录已保存至: {failed_csv}")

    return body_part_count

# ==============================
# 使用示例
# ==============================
if __name__ == "__main__":
    INPUT_CSV = "/home/yyi/data/6yuan_raw_dataset.csv"     # 输入路径列表
    OUTPUT_CSV = "/home/yyi/data/6yuan_raw_count.csv"   # 输出带标签结果
    FAILED_CSV = "/home/yyi/data/6yuan_raw_failed_paths.csv"         # 新增：失败路径记录（可选）

    stats = extract_dicom_metadata_to_csv(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        failed_csv=FAILED_CSV,              # 启用失败记录
        path_column='image_path'
    )