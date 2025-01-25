import pandas as pd
import pycountry
import numpy as np
maindf = pd.read_csv('2025_Problem_C_Data/NEW_MEDAL_TABLE_WITH_PARTICIPANTS.csv')
GDP = pd.read_csv('2025_Problem_C_Data/GDP.csv')
POP = pd.read_csv('2025_Problem_C_Data/Population.csv')
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
def search_rows(df, noc_code):
    iso_alpha3 = to_iso_alpha3(noc_code, custom_mapping)
    if not iso_alpha3:
        # print(f"未找到对应的 ISO alpha-3 编码: {noc_code}")
        return -1

    # 筛选符合条件的行
    result = df[df['NOC'].apply(lambda x: to_iso_alpha3(x, custom_mapping)) == iso_alpha3]

    # 返回行号
    if not result.empty:
        return result.index.tolist()  # 返回行号列表
    else:
        # print("未找到符合条件的行: 国家代号 {noc_code}")
        return -1

maindf['GDP'] = np.nan
maindf['Population'] = np.nan
for i in range(maindf.shape[0]):
    row = maindf.iloc[i]
    year = row['Year']
    noc_code = row['NOC']
    if year < 1960 or year > 2020:
        continue
    index = search_rows(GDP , noc_code)
    GDP_v = np.nan
    Pop_v = np.nan
    if index != -1:
        GDP_v = GDP.loc[index, str(year)]
        if isinstance(GDP_v, pd.Series):  # 如果 GDP_v 是 Series，取第一个值
            GDP_v = GDP_v.iloc[0]
    else:
        print(noc_code,"is not found (GDP)")
    index = search_rows(POP, noc_code)
    if index != -1:
        Pop_v = POP.loc[index, str(year)]
        if isinstance(Pop_v, pd.Series):  # 如果 Pop_v 是 Series，取第一个值
            Pop_v = Pop_v.iloc[0]
    else:
        print(noc_code,"is not found (Population)")
    # print(GDP_v)
    # print(Pop_v)
    maindf.loc[i, 'GDP'] =  GDP_v
    maindf.loc[i, 'Population'] = Pop_v

maindf.to_csv('NEW_MEDAL_TABLE_WITH_PARTICIPANTS.csv', index=False)
