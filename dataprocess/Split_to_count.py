import csv
import random
import tempfile
from pathlib import Path
from collections import defaultdict

def downsample_csv_by_label(
    csv_path: str,
    label_column: str,
    max_samples_per_class: int = 1129,
    target_labels=None,
    random_seed: int = 42
):
    """
    对 CSV 中指定列的指定标签（如 ['1','2','3']）最多保留 max_samples_per_class 行。
    
    参数:
        csv_path (str): CSV 文件路径
        label_column (str): 标签列名（如 '标签'）
        max_samples_per_class (int): 每类最大保留行数
        target_labels (list): 要下采样的标签列表，如 ['1','2','3']。默认为 ['1','2','0']
        random_seed (int): 随机种子（保证可复现）
    """
    if target_labels is None:
        target_labels =  ['0', '1', '2'] # 默认值
    
    target_set = set(str(x).strip() for x in target_labels)  # 转为字符串 + 去空格，确保匹配

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"❌ 文件不存在: {csv_path}")

    random.seed(random_seed)

    label_to_rows = defaultdict(list)
    fieldnames = None
    total_rows = 0

    # 读取并分组
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("❌ CSV 无表头")

        # 处理 B BOM
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

            if label_val in target_set:
                label_to_rows[label_val].append(row)
            else:
                label_to_rows['__other__'].append(row)

    # 构建保留行
    kept_rows = []

    # 处理目标标签
    for label in target_labels:  # 按用户指定顺序处理（便于日志）
        label_str = str(label).strip()
        rows = label_to_rows[label_str]
        if len(rows) > max_samples_per_class:
            sampled = random.sample(rows, max_samples_per_class)
            kept_rows.extend(sampled)
            print(f"📌 标签 '{label_str}': 原 {len(rows)} 行 → 保留 {max_samples_per_class} 行")
        else:
            kept_rows.extend(rows)
            print(f"📌 标签 '{label_str}': {len(rows)} 行（≤{max_samples_per_class}，全部保留）")

    # 添加非目标行
    other_rows = label_to_rows['__other__']
    kept_rows.extend(other_rows)
    if other_rows:
        print(f"📌 非目标标签（共 {len(other_rows)} 行）: 全部保留")

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
    CSV_FILE = "/home/yyi/data/data_pretrain/2_isCancer_frac.csv"
    LABEL_COLUMN = "split_label"

    # ✅ 现在可以任意指定要处理的标签！
    downsample_csv_by_label(
        csv_path=CSV_FILE,
        label_column=LABEL_COLUMN,
        max_samples_per_class=1886,
        target_labels=['0', '1', '2'],
        random_seed=42
    )