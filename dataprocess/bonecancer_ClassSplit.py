import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# 1. 配置（请修改）
# ----------------------------
INPUT_CSV = "/home/yyi/分类统计结果.csv"
OUTPUT_CSV = "/home/yyi/classified_output_fuzzy.csv"
TUMOR_COL = "原发骨肿瘤病理结果"  # 请替换为您的列名

# ----------------------------
# 2. 构建模糊匹配规则（按优先级顺序！）
# ----------------------------
CLASSIFICATION_RULES = [
    # (大类, 分类, 匹配关键词列表)
    ("软骨源性肿瘤", "良性", [ "甲下外生性骨疣","奇异性骨旁骨软骨瘤样增生","骨膜软骨瘤","内生软骨瘤","骨软骨瘤",
                      "软骨母细胞瘤","软骨母", "软骨黏液样纤维瘤", "骨软骨黏液瘤", "内生性软骨瘤","内生性软骨", 
                      "内生软骨", "多发性骨软骨病", "ollier"]),
    ("软骨源性肿瘤", "中间性", ["非典型软骨瘤", "软骨瘤病","非典型性软骨性肿瘤", "非典型软骨性肿瘤", "非典型软骨源性肿瘤",
                       "不典型软骨源性肿瘤", "不典型性软骨肿瘤", "高分化软骨性肿瘤","非典型性软骨粘液样纤维瘤", "act"]),
    ("软骨源性肿瘤", "恶性", ["软骨肉瘤","骨膜软骨肉瘤","透明细胞软骨肉瘤","间叶性软骨肉瘤","去分化软骨肉瘤","ollier", "肉瘤变"]),
    
    ("骨源性肿瘤", "良性", ["骨样骨瘤","骨瘤","骨疣", "外生骨疣", "甲下外生性骨疣", "外生性甲下骨疣",
                     "复发性甲下骨疣", "宽基底骨疣", "塔状骨疣"]),
    ("骨源性肿瘤", "中间性", ["骨母细胞瘤"]),
    ("骨源性肿瘤", "恶性", ["低级别中心性骨肉瘤","骨肉瘤","普通型骨肉瘤","毛细血管扩张型骨肉瘤","上皮样骨母",
                     "小细胞型骨肉瘤","骨旁骨肉瘤","骨膜骨肉瘤","高级别表面骨肉瘤","继发性骨肉瘤"]),

    ("纤维源性肿瘤", "中间性", ["促结缔组织增生性纤维瘤"]),
    ("纤维源性肿瘤", "恶性", ["纤维肉瘤"]),

     # 造血系统肿瘤
     ("骨的造血系统肿瘤", "恶性", ["骨的浆细胞瘤","恶性淋巴瘤非霍奇金型","霍奇金病","弥漫大B细胞淋巴瘤","滤泡性淋巴瘤",
                         "边缘区B细胞淋巴瘤","T细胞淋巴瘤","间变性大细胞淋巴瘤","恶性淋巴瘤","Burkitt", "淋巴瘤","伯基特淋巴瘤"
                         "朗格汉斯细胞组织细胞增生症","Erdheim-Chester","埃德海姆切斯特病","Rosai-Dorfman","罗萨伊多尔夫曼病"]),
    
    # 血管性肿瘤
    ("骨的血管性肿瘤", "良性", ["血管瘤"]),
    ("骨的血管性肿瘤", "中间性", ["上皮样血管瘤"]),
    ("骨的血管性肿瘤", "恶性", ["上皮样血管内皮瘤","血管肉瘤"]),
    
    
    ("富于破骨细胞样多核巨细胞的肿瘤", "良性", ["ABC", "动脉瘤样骨囊肿","非骨化性纤维瘤"]),
    ("富于破骨细胞样多核巨细胞的肿瘤", "中间性", ["骨巨", "骨巨细胞瘤", "GCT"]),
    ("富于破骨细胞样多核巨细胞的肿瘤", "恶性", ["恶性骨巨", "恶性骨巨细胞瘤", "GCT"]),

    ("脊索组织肿瘤", "良性", ["良性脊索样肿瘤"]),
    ("脊索组织肿瘤", "恶性", ["脊索瘤","软骨样脊索瘤","分化差的脊索瘤","低分化脊索瘤","退分化脊索瘤"]),
    
    ("骨的其他间叶性肿瘤", "良性", ["胸壁软骨间叶性错构瘤","单纯性骨囊肿","纤维结构不良", "骨性纤维结构不良","FD", "NOF","脂肪","冬眠",
                          "骨囊肿", "骨内腱鞘囊肿", "纤维骨皮质缺损", "良性囊肿性病变","良性骨病", "良性病变", "囊肿性病变", "表皮样囊肿",
                          "多发性磷酸盐尿性间叶性肿瘤", "骨纤维异常增殖症", "骨岛","富于细胞性神经纤维瘤"]),
    ("骨的其他间叶性肿瘤", "中间性", ["骨性纤维结构不良样釉质瘤","间叶瘤"]),
    ("骨的其他间叶性肿瘤", "恶性", ["长骨釉质瘤","退分化釉质瘤","平滑肌肉瘤","未分化多形性肉瘤","骨转移瘤"]),
    
    ("骨的其他肿瘤", "良性", []),
    ("骨的其他肿瘤", "中间性", []),
    ("骨的其他肿瘤", "恶性", ["尤文肉瘤"]),
]

# 标准化函数：统一小写、去除空格、符号
def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    # 去除常见符号
    text = re.sub(r'[,\(\)\.\-\s]+', '', text)
    return text

# ----------------------------
# 3. 主函数
# ----------------------------
def classify_tumor(name):
    if not name or name == "":
        return None, None
    
    norm_name = normalize(name)
    
    # 按规则顺序匹配（优先级从高到低）
    for major, subtype, keywords in CLASSIFICATION_RULES:
        for kw in keywords:
            norm_kw = normalize(kw)
            if norm_kw in norm_name:
                return major, subtype
    
    # 未匹配
    return "", ""

def main():
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"加载 {len(df)} 行数据")

    # 过滤空值
    df = df.dropna(subset=[TUMOR_COL], how='all')
    df = df[df[TUMOR_COL].astype(str).str.strip() != '']
    logger.info(f"过滤空值后剩余 {len(df)} 行")

    # 分类
    results = []
    unmatched = []

    for idx, row in df.iterrows():
        tumor = str(row[TUMOR_COL]).strip()
        major, subtype = classify_tumor(tumor)
        new_row = row.copy()
        new_row['大类'] = major
        new_row['分类'] = subtype
        
        if major == "":
            unmatched.append(new_row)
            logger.warning(f"未匹配: '{tumor}'")
        else:
            results.append(new_row)
    
    # 合并：先匹配的，后未匹配的
    final_df = pd.concat([pd.DataFrame(results), pd.DataFrame(unmatched)], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    logger.info(f"✅ 完成！匹配 {len(results)} 行，未匹配 {len(unmatched)} 行 → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()