import pandas as pd
import numpy as np
import  pycountry
# 读取数据
df = pd.read_csv('2025_Problem_C_Data/NEW_MEDAL_TABLE.csv')
df_participants = pd.read_csv('2025_Problem_C_Data/each_country_each_year_count_entries.csv')

# 初始化 Participants 列
df['Participants'] = np.nan

# 自定义映射表
custom_mapping = {
    "DDR": "GDR",  # 东德
    "FRG": "FRG",  # 西德
    "URS": "URS",  # 苏联
    "TCH": "TCH",  # 捷克斯洛伐克
    "YUG": "YUG",  # 南斯拉夫
}

# 定义三位字母编码到 ISO alpha-3 的映射函数
def to_iso_alpha3(code, custom_mapping):
    # 检查自定义映射表
    if code in custom_mapping.values():
        # 如果输入的是 IOC 编码，返回对应的 ISO 编码
        return next(iso for iso, ioc in custom_mapping.items() if ioc == code)
    elif code in custom_mapping:
        # 如果输入的是 ISO 编码，返回对应的 IOC 编码
        return custom_mapping[code]

    # 尝试将编码作为 ISO alpha-3 编码查找
    country = pycountry.countries.get(alpha_3=code)
    if country:
        return country.alpha_3  # 返回 ISO alpha-3 编码

    # 尝试将编码作为 IOC 编码查找
    try:
        country = pycountry.countries.lookup(code)
        return country.alpha_3  # 返回 ISO alpha-3 编码
    except LookupError:
        return None  # 如果找不到对应的 ISO 编码，返回 None


# 定义搜索函数
def search_rows(year, noc_code):
    # 将 NOC 编码转换为 ISO alpha-3 编码
    iso_alpha3 = to_iso_alpha3(noc_code, custom_mapping)
    if not iso_alpha3:
        # print(f"未找到对应的 ISO alpha-3 编码: {noc_code}")
        return -1

    # 筛选符合条件的行
    result = df[(df['Year'] == year) & (df['NOC'].apply(lambda x: to_iso_alpha3(x, custom_mapping)) == iso_alpha3)]

    # 返回行号
    if not result.empty:
        return result.index.tolist()  # 返回行号列表
    else:
        # print("未找到符合条件的行: 年份 {year}, 国家代号 {noc_code}")
        return -1

# 遍历 df 的每一行
for i in range(df.shape[0]):
    row = df.iloc[i]
    year = row['Year']
    NOC = row['NOC']

    row_numbers = search_rows(year, NOC)
    if row_numbers!=-1:
        participants_value = df_participants.loc[row_numbers, 'Participants'].values[0]
        df.at[i, 'Participants'] = participants_value

df_tmp = df.dropna()
df_tmp.to_csv('NEW_MEDAL_TABLE_WITH_PARTICIPANTS.csv')
# 找到包含 NaN 值的行
# nan_rows = df[df.isna().any(axis=1)]
# print("包含 NaN 值的行：")
# print(nan_rows)
# nan_noc = nan_rows['NOC'].drop_duplicates()
# nan_noc.to_csv('tmp.csv')
# print(len(nan_noc))

# def ioc_to_iso(code):
#     # 自定义映射（处理特殊情况）
#     custom_mapping = {
#         "NED": "NL",  # 荷兰
#         "GBR": "GB",  # 英国
#     }
#     if code in custom_mapping:
#         return custom_mapping[code]
#
#     # 尝试将编码作为 IOC 编码查找
#     country = pycountry.countries.get(alpha_3=code)
#     if country:
#         return country.alpha_3  # 返回 ISO alpha-2 编码
#
#     # 如果不是 IOC 编码，尝试作为 ISO 编码查找
#     country = pycountry.countries.get(alpha_2=code)
#     if country:
#         return code  # 已经是 ISO alpha-2 编码，直接返回
#
#     return None  # 如果找不到对应的 ISO 编码，返回 None
# #
# # df['ISO'] = df['NOC'].apply(ioc_to_iso)
# # print(df)
