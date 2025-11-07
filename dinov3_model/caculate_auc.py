import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_max_value_from_tb(log_dir, tag):
    """从 TensorBoard 日志中提取指定 tag 的最大值"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    if tag not in event_acc.Tags()['scalars']:
        print(f"⚠️ Warning: Tag '{tag}' not found in {log_dir}. Skipping.")
        return np.nan
    values = [event.value for event in event_acc.Scalars(tag)]
    return max(values)

if __name__ == "__main__":
    # -------------------------------
    # 📁 日志目录（5 次独立运行）
    log_dirs = [
        "/data/truenas_B2/yyi/logs/BTXRD/Dinov3_BTXRD_256_0.1_1",
        "/data/truenas_B2/yyi/logs/BTXRD/Dinov3_BTXRD_256_0.1_42",
        "/data/truenas_B2/yyi/logs/BTXRD/Dinov3_BTXRD_256_0.1_123",
        "/data/truenas_B2/yyi/logs/BTXRD/Dinov3_BTXRD_256_0.1_1000",
        "/data/truenas_B2/yyi/logs/BTXRD/Dinov3_BTXRD_256_0.1_2025",
    ]

    # ✅ 定义你的所有子任务名称（必须与 TensorBoard 中的后缀一致）
    subtasks = ["tumor", "benign", "malignant"]  # ← 修改这里！

    # 自动生成所有指标 tag（假设格式为 Test_Summary/{METRIC}_{subtask}）
    metric_names_base = ["AUPRC", "AUROC", "ACCURACY"]
    metric_tags = {}
    for subtask in subtasks:
        for metric in metric_names_base:
            col_name = f"{metric}_{subtask}"  # 如 AUPRC_benign
            tb_tag = f"Test_Summary/{metric}_{subtask}"
            metric_tags[col_name] = tb_tag

    output_csv = "/home/yyi/data/test_dataset/Tset_Summary/BTXRD_boneDINOV3_TestSummary.csv"

    # -------------------------------
    # 📊 提取数据：每行一个 run，每列一个 (指标_子任务)
    data = []
    for i, log_dir in enumerate(log_dirs, 1):
        row = {"Run": f"Run_{i}", "Log_Dir": log_dir}
        for col_name, tb_tag in metric_tags.items():
            try:
                max_val = get_max_value_from_tb(log_dir, tb_tag)
                row[col_name] = max_val
            except Exception as e:
                print(f"❌ Error for {col_name} in {log_dir}: {e}")
                row[col_name] = np.nan
        data.append(row)

    df = pd.DataFrame(data)

    # -------------------------------
    # 📈 计算 MEAN 和 STD（按列）
    metric_cols = list(metric_tags.keys())
    mean_vals = df[metric_cols].mean()
    std_vals = df[metric_cols].std(ddof=1)

    # 添加汇总行
    df.loc[len(df)] = ["MEAN", ""] + mean_vals.tolist()
    df.loc[len(df)] = ["STD", ""] + std_vals.tolist()

    # -------------------------------
    # 💾 保存
    df.to_csv(output_csv, index=False)

    # -------------------------------
    # 🖨️ 打印结果（按子任务分组显示）
    print("\n" + "="*70)
    print("📊 Final Results (Mean ± Std) by Subtask:")
    for subtask in subtasks:
        print(f"\n🔹 Subtask: {subtask}")
        for metric in metric_names_base:
            col = f"{metric}_{subtask}"
            mean = mean_vals[col]
            std = std_vals[col]
            print(f"  {metric}: {mean:.6f} ± {std:.6f}")

    print(f"\n✅ Results saved to:\n  - CSV:  {output_csv}")