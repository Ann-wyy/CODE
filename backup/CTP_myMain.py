from utils.compare_to_reference_maps import *
from core import *
from config import *
import os
import glob
import logging

# ==============================
# 配置日志
# ==============================
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

"""
This main script loops over all patients in a perfusion imaging dataset.
For each patient, it generates perfusion maps (CBF, CBV, MTT, TTP, TMAX) from the raw 4D perfusion data.
It then compares the generated maps to provided reference maps.
In case reference maps are not available, the comparison step can be skipped by setting the `SHOW_COMPARISONS` flag to `True`.
The main options can be set in the config.py file.
Smaller adjustments such as thresholds and kernel sizes can be adjusted in the respective utility files.
PyPeT assumes a specific directory structure for the dataset. The required structure is described in the README file.
"""

root_dir = "/data/truenas_36T/HeadStroke_Yan"
logger.info(f"Started processing... root_dir = {root_dir}")

# Get all subdirectories that match pattern a* (e.g., a106369160)
patient_dirs = [
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('a')
]
logger.info(f"Found {len(patient_dirs)} patient directories starting with 'a'")

# Loop over all directories in the dataset
for patient_dir in patient_dirs:
    ctp_dir = os.path.join(patient_dir, "CTP")
    if not os.path.exists(ctp_dir):
        logger.warning(f"CTP directory not found: {ctp_dir}, skipping...")
        continue

    nii_files = glob.glob(os.path.join(ctp_dir, "*.nii.gz"))
    if not nii_files:
        logger.warning(f"No .nii.gz files in {ctp_dir}, skipping...")
        continue

    logger.info(f"Processing patient: {os.path.basename(patient_dir)}")
    logger.info(f"Found {len(nii_files)} scan(s): {[os.path.basename(f) for f in nii_files]}")

    for perf_path in nii_files:
        basename = os.path.basename(perf_path)
        if basename.endswith('.nii.gz'):
            scan_name = basename[:-7]  # remove '.nii.gz'
        else:
            scan_name = os.path.splitext(basename)[0]

        output_dir = os.path.join(ctp_dir, scan_name)
        logger.info(f"→ Processing: {basename}")
        logger.info(f"  Output to: {output_dir}")

        # Generate perfusion maps from inputted perfusion data
        if GENERATE_PERFUSION_MAPS:
            try:
                core(perf_path=perf_path, mask_path=None, output_dir=output_dir)
                logger.info("  ✅ Success")
            except Exception as e:
                logger.error(f"  ❌ Failed: {e}")
                continue

        # 注意：你原代码中 core 被调用了两次！这里只保留一次。
        # 如果你需要对比或其他后续步骤，可在此处添加，但不要重复 core()

    logger.info("Finished processing current patient!")

logger.info("All done!")