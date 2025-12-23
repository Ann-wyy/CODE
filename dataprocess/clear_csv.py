import pandas as pd

# 读取 CSV 文件
clearname = '检查部位'
# clearname1 = '转移标签'
df = pd.read_csv('/home/yyi/data/data_pretrain/isCancer_part.csv', dtype={clearname: str})  # 先按字符串读取，避免 .0 被 float 处理丢失

# 处理「影像号」：去除末尾的 .0（若存在）
# df[clearname] = df[clearname].str.replace(r'\.0$', '', regex=True)
# df[clearname1] = df[clearname1].str.replace(r'\.0$', '', regex=True)


# 处理「年龄」：删除「岁」字，并转为纯数字（可选）
clearage = '年龄'
df[clearage] = df[clearage].str.replace('岁', '', regex=False)


#clearnan = '原发/转移'
#df[clearnan] = df[clearnan].str.strip().replace('', pd.NA)
#df = df.dropna(subset=[clearnan])

# 保存结果
df.to_csv('/home/yyi/data/data_pretrain/isCancer_part_0.csv', index=False, encoding='utf-8-sig')
