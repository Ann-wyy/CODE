import csv
import random
import tempfile
from pathlib import Path
from collections import defaultdict

def downsample_csv_by_label(
    csv_path: str,
    label_column: str,
    max_samples_per_class: int = 1129,
    random_seed: int = 42
):
    """
    对 CSV 中指定列的每个标签（1,2,3）最多保留 max_samples_per_class 行。
    
    参数:
        csv_path (str): CSV 文件路径
        label_column (str): 标签列名（如 '标签'）
        max_samples_per_class (int): 每类最大保留行数
        random_seed (int): 随机种子（保证可复现）
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"❌ 文件不存在: {csv_path}")

    random.seed(random_seed)  # 保证随机可复现

    # 存储每类的行
    label_to_rows = defaultdict(list)
    fieldnames = None
    total_rows = 0

    # 读取所有行并按标签分组
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("❌ CSV 无表头")

        # 处理 BOM
        clean_fields = [col.lstrip('\ufeff') for col in fieldnames]
        if label_column in clean_fields:
            actual_label_col = fieldnames[clean_fields.index(label_column)]
        elif label_column in fieldnames:
            actual_label_col = label_column
        else:
            raise ValueError(f"❌ 列 '{label_column}' 不存在。可用列: {clean_fields}")

        for row in reader:
            total_rows += 1
            label_val = row.get(actual_label_col, '').strip()

            # 只处理 1, 2, 3（其他值保留？这里选择跳过或保留？）
            # 根据需求：只对 1/2/3 下采样，其他值全部保留
            if label_val in {'1', '2', '3'}:
                label_to_rows[label_val].append(row)

    # 构建保留的行
    kept_rows = []

    # 处理 1, 2, 3
    for label in ['1', '2', '3']:
        rows = label_to_rows[label]
        if len(rows) > max_samples_per_class:
            sampled = random.sample(rows, max_samples_per_class)
            kept_rows.extend(sampled)
            print(f"📌 标签 {label}: 原 {len(rows)} 行 → 保留 {max_samples_per_class} 行")
        else:
            kept_rows.extend(rows)
            print(f"📌 标签 {label}: {len(rows)} 行（≤{max_samples_per_class}，全部保留）")

    # 添加其他行（不采样）
    other_rows = label_to_rows['__other__']
    kept_rows.extend(other_rows)
    if other_rows:
        print(f"📌 其他值（非1/2/3）: 保留 {len(other_rows)} 行")

    # 打乱顺序（可选：避免同类聚集）
    # random.shuffle(kept_rows)

    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
        tmp_path = tmp.name

    # 原地替换
    try:
        csv_path.unlink()
        Path(tmp_path).rename(csv_path)
        print(f"\n✅ 完成！原 {total_rows} 行 → 新 {len(kept_rows)} 行，文件已更新: {csv_path}")
    except Exception as e:
        Path(tmp_path).unlink()
        raise RuntimeError(f"❌ 保存失败: {e}")


# ==============================
# 使用示例
# ==============================
if __name__ == "__main__":
    CSV_FILE = "/home/yyi/data/data_pretrain/原发转移骨肿瘤_匹配合并结果_含部位_0.csv"
    LABEL_COLUMN = "良性1/中间型2/恶性3"  # 替换为你的列名，如 'label', 'category'

    # ⚠️ 注意：此操作会修改原文件！建议先备份
    downsample_csv_by_label(
        csv_path=CSV_FILE,
        label_column=LABEL_COLUMN,
        max_samples_per_class=1129,
        random_seed=42
    )