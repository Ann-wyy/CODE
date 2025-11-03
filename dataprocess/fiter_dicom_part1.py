import pydicom
import csv
import os
from collections import defaultdict

def scan_dicom_folder_to_csv(root_dir, output_csv, failed_csv=None):
    """
    遍历 root_dir 下所有文件，识别 DICOM 文件（通过扩展名或尝试读取），
    提取 BodyPartExamined 等元数据，保存到 CSV。
    不加载像素数据，高效处理。
    """
    body_part_count = defaultdict(int)
    processed_count = 0
    error_count = 0

    # 收集所有可能的 DICOM 文件（简单策略：无扩展名或 .dcm）
    dicom_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            # 常见 DICOM 文件无扩展名或使用 .dcm
            if f.lower().endswith('.dcm') or '.' not in f:
                dicom_paths.append(os.path.join(dirpath, f))

    print(f"🔍 找到 {len(dicom_paths)} 个候选 DICOM 文件（基于扩展名/无扩展名）")

    with open(output_csv, 'w', newline='', encoding='utf-8') as out_f:
        fieldnames = [
            'file_name',
            'dicom_path',
            'body_part_examined',
            'modality',
            'study_description',
            'status'
        ]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        failed_writer = None
        if failed_csv:
            failed_f = open(failed_csv, 'w', newline='', encoding='utf-8')
            failed_writer = csv.DictWriter(failed_f, fieldnames=['dicom_path', 'error_message'])
            failed_writer.writeheader()

        for dicom_path in dicom_paths:
            file_name = os.path.basename(dicom_path)
            result = {
                'file_name': file_name,
                'dicom_path': dicom_path,
                'body_part_examined': 'Unknown',
                'modality': 'Unknown',
                'study_description': '',
                'status': 'success'
            }

            try:
                # stop_before_pixels=True 避免加载图像数据，加快速度
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

                result['body_part_examined'] = (getattr(ds, 'BodyPartExamined', 'Unknown') or 'Unknown').strip()
                result['modality'] = (getattr(ds, 'Modality', 'Unknown') or 'Unknown').strip()
                result['study_description'] = (getattr(ds, 'StudyDescription', '') or '').strip()

                body_part_count[result['body_part_examined']] += 1
                processed_count += 1

            except Exception as e:
                error_msg = str(e)[:200]
                result['status'] = f'read_error: {error_msg}'
                error_count += 1
                print(f"⚠️ 读取失败: {dicom_path} | {error_msg}")

                if failed_writer:
                    failed_writer.writerow({
                        'dicom_path': dicom_path,
                        'error_message': error_msg
                    })

            writer.writerow(result)

        if failed_csv:
            failed_f.close()

    # 打印统计
    print(f"\n✅ 扫描完成!")
    print(f"   总候选文件数: {len(dicom_paths)}")
    print(f"   成功解析: {processed_count} | 失败: {error_count}")
    print(f"\n📊 BodyPartExamined 统计:")
    for part, count in sorted(body_part_count.items(), key=lambda x: x[1], reverse=True):
        print(f"   {part}: {count}")

    print(f"\n💾 结果已保存至: {output_csv}")
    if failed_csv:
        print(f"💾 失败记录已保存至: {failed_csv}")

    return body_part_count


# ==============================
# 使用示例
# ==============================
if __name__ == "__main__":
    ROOT_DIR = "/data/truenas_B2/Dataset/001_6yXray/bone_cancer"  # 当前文件夹，可改为其他路径如 "/data/dicom_root"
    OUTPUT_CSV = "/home/yyi/data/dicom_inventory.csv"
    FAILED_CSV = "/home/yyi/data/dicom_read_failures.csv"

    stats = scan_dicom_folder_to_csv(
        root_dir=ROOT_DIR,
        output_csv=OUTPUT_CSV,
        failed_csv=FAILED_CSV
    )