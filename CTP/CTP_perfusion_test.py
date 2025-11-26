import SimpleITK as sitk
import numpy as np
import os

def load_nii_to_array(path):
    """加载 NIfTI 文件并返回 NumPy 数组和 SimpleITK Image 对象。"""
    if not os.path.exists(path):
        print(f"错误: 文件不存在 -> {path}")
        return None, None
    img = sitk.ReadImage(path)
    # 将图像数据转换为 numpy 数组，SimpleITK 通常使用 (z, y, x) 顺序
    arr = sitk.GetArrayFromImage(img)
    return arr, img

def get_hemisphere_masks(mask_arr, mask_img):
    """
    根据 SimpleITK 图像的中线分割脑部掩膜为左右半球。
    注意: 假设图像是以常见的轴向（axial）方向存储（z是切片方向）。
    """
    # 假设中轴线在图像的 x 维度上
    x_dim = mask_arr.shape[-1]
    mid_x = x_dim // 2
    
    # SimpleITK GetArrayFromImage 顺序通常是 (Z, Y, X)
    
    # 创建左右半球掩膜
    left_mask = np.zeros_like(mask_arr, dtype=bool)
    right_mask = np.zeros_like(mask_arr, dtype=bool)
    
    # 假设 x 轴是左右方向。需要根据图像方向确定哪边是左/右
    # 这里我们只是简单地根据索引划分，实际应用可能需要检查 Image Direction
    
    # 假设索引 < mid_x 是左侧， > mid_x 是右侧
    # 注意：这里的左右定义依赖于 SimpleITK 的数组顺序，可能需要根据实际数据调整
    # 仅保留在原始脑部掩膜内的体素
    full_brain = mask_arr.astype(bool)
    
    # 左半球 (Left Hemisphere)
    left_mask[..., :mid_x] = full_brain[..., :mid_x]
    
    # 右半球 (Right Hemisphere)
    right_mask[..., mid_x:] = full_brain[..., mid_x:]
    
    return left_mask, right_mask

def get_voxel_volume(sitk_img):
    """根据 SimpleITK 图像获取体素体积（单位：mL）。"""
    # SimpleITK 空间间隔（体素大小）通常是 (x, y, z)
    spacing = sitk_img.GetSpacing()
    # 体积单位是 mm^3，除以 1000 转换为 mL (cc)
    return spacing[0] * spacing[1] * spacing[2] / 1000.0


def analyze_perfusion_maps(
    cbf_path, cbv_path, tmax_path, mtt_path, mask_path
):
    """
    执行灌注图的质量和临床指标测试。
    """
    # 1. 加载所有图像
    cbf_arr, cbf_img = load_nii_to_array(cbf_path)
    cbv_arr, _ = load_nii_to_array(cbv_path)
    tmax_arr, _ = load_nii_to_array(tmax_path)
    mtt_arr, _ = load_nii_to_array(mtt_path)
    mask_arr, mask_img = load_nii_to_array(mask_path)
    
    if any(x is None for x in [cbf_arr, cbv_arr, tmax_arr, mtt_arr, mask_arr]):
        print("无法加载所有必需的灌注图文件，分析中止。")
        return {}

    voxel_volume = get_voxel_volume(cbf_img)
    print(f"体素体积: {voxel_volume:.4f} mL")

    # 2. 划分半球并自动确定患侧/健侧
    left_mask, right_mask = get_hemisphere_masks(mask_arr, mask_img)
    
    # 计算左右半球 Tmax 的平均值（仅在脑部掩膜内）
    mean_tmax_left = np.mean(tmax_arr[left_mask]) if np.any(left_mask) else 0
    mean_tmax_right = np.mean(tmax_arr[right_mask]) if np.any(right_mask) else 0

    # 患侧是 Tmax 更高的一侧（血流延迟更严重）
    if mean_tmax_left > mean_tmax_right:
        lesion_side = "左侧"
        lesion_mask = left_mask
        reference_mask = right_mask
    else:
        lesion_side = "右侧"
        lesion_mask = right_mask
        reference_mask = left_mask

    print(f"\n--- 自动判断 ---")
    print(f"左侧平均 Tmax: {mean_tmax_left:.2f} s | 右侧平均 Tmax: {mean_tmax_right:.2f} s")
    print(f"患侧（病灶侧）被确定为: {lesion_side}")
    
    # 3. 计算健侧参考值
    ref_cbf_mean = np.mean(cbf_arr[reference_mask])
    ref_mtt_mean = np.mean(mtt_arr[reference_mask])

    print(f"健侧 CBF 参考均值: {ref_cbf_mean:.2f} | 健侧 MTT 参考均值: {ref_mtt_mean:.2f}")

    # 4. 定义阈值和生成掩膜
    
    # --- 梗死核心区 (Infarct Core) ---
    # ① CBV 绝对值 < 2.0 ml/100 g
    core_cbv_criterion = (cbv_arr < 2.0)
    # ② 相对 CBF < 30% 健侧 CBF
    core_cbf_criterion = (cbf_arr < (0.30 * ref_cbf_mean))
    
    # 核心区掩膜：任一标准满足，且必须在患侧脑区内
    core_mask = (core_cbv_criterion | core_cbf_criterion) & lesion_mask
    
    # --- 低灌注区 (Hypoperfusion Region) ---
    # ① Tmax 绝对值 > 6 s
    lesion_tmax_criterion = (tmax_arr > 6.0)
    # ② 相对 MTT > 145% 健侧 MTT
    lesion_mtt_criterion = (mtt_arr > (1.45 * ref_mtt_mean))
    
    # 低灌注区掩膜：任一标准满足，且必须在患侧脑区内
    lesion_mask_all = (lesion_tmax_criterion | lesion_mtt_criterion) & lesion_mask
    
    # --- 严重低灌注区 (Severe Hypoperfusion) ---
    severe_lesion_mask = (tmax_arr > 10.0) & lesion_mask
    
    # 5. 计算体积
    v_core = np.sum(core_mask) * voxel_volume
    v_lesion = np.sum(lesion_mask_all) * voxel_volume
    v_severe_lesion = np.sum(severe_lesion_mask) * voxel_volume
    
    # 6. 计算不匹配率
    mismatch_ratio = v_lesion / v_core if v_core > 0 else np.inf

    # 7. 输出结果和质量判断
    print("\n--- 灌注图质量和临床指标 ---")
    print(f"1. 核心梗死区体积 (V_Core): {v_core:.2f} mL")
    print(f"2. 低灌注区体积 (V_Lesion): {v_lesion:.2f} mL")
    print(f"3. 严重低灌注区体积 (Tmax > 10s): {v_severe_lesion:.2f} mL")
    print(f"4. 不匹配率 (V_Lesion / V_Core): {mismatch_ratio:.2f}")
    
    # 质量/临床判断
    quality_notes = []
    if v_core < 70:
        quality_notes.append("核心区体积较小 (< 70 mL)")
    else:
        quality_notes.append("核心区体积较大 (>= 70 mL)")
        
    if v_severe_lesion < 100:
        quality_notes.append("严重低灌注区体积较小 (< 100 mL)")
    else:
        quality_notes.append("严重低灌注区体积较大 (>= 100 mL)")
        
    if mismatch_ratio > 1.2:
        quality_notes.append("不匹配比例较大 (> 1.2), 提示存在缺血半暗带。")
    else:
        quality_notes.append("不匹配比例较小 (<= 1.2), 提示缺血半暗带有限。")

    print("\n[分析总结]")
    print(f"建议治疗适应症评估（参考标准）: {' & '.join(quality_notes)}")

    return {
        "V_Core": v_core,
        "V_Lesion": v_lesion,
        "V_Severe_Lesion": v_severe_lesion,
        "Mismatch_Ratio": mismatch_ratio,
        "Lesion_Side": lesion_side
    }

# --- 示例使用 ---
if __name__ == "__main__":
    # ⚠️ 请根据您的实际运行环境调整基础路径和文件名
    BASE_DIR = "/home/yyi/CT-and-MR-Perfusion-Tool/demo_data/sub-stroke0143/ses-01/perfusion-maps"
    
    # 确保这些文件是 core 函数生成的输出
    TEST_FILES = {
        "cbf_path": os.path.join(BASE_DIR, 'generated_cbf.nii.gz'),
        "cbv_path": os.path.join(BASE_DIR, 'generated_cbv.nii.gz'),
        "tmax_path": os.path.join(BASE_DIR, 'generated_tmax.nii.gz'),
        "mtt_path": os.path.join(BASE_DIR, 'generated_mtt.nii.gz'),
        # 假设脑部掩膜在上一级目录
        "mask_path": os.path.join(os.path.dirname(BASE_DIR), 'brain_mask.nii.gz'), 
    }

    print("--- 开始灌注图质量测试 ---")
    results = analyze_perfusion_maps(**TEST_FILES)
    
    if results:
        print("\n--- 原始结果字典 ---")
        print(results)