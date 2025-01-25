import pandas as pd
import numpy as np
# 假设你的数据已经被加载到一个 DataFrame 中

df = pd.read_csv('Final_table.csv')

# 创建一个以 NOC 为索引的 DataFrame，并获取每个 NOC 对应的参赛年份
noc_years = df.groupby('NOC')['Year'].apply(list).reset_index()
# print(len(noc_years))
# 输出结果
windows = {}
def clean_list(mixed_list):
    int_list = []  # 创建一个新的列表用于存储转换后的整数
    for item in mixed_list:
        try:
            int_value = int(item)  # 尝试将每个元素转换为整数
            int_list.append(int_value)  # 添加到新列表中
        except (ValueError, TypeError):
            # 如果转换失败，跳过该项
            continue
    return int_list

for i in range(noc_years.shape[0]):
    year = noc_years.iloc[i]['Year']
    noc = noc_years.iloc[i]['NOC']
    if len(year) < 3:
        continue

    windows[noc] = []
    for j in range(3, len(year)-1):
        tmp_ls = []
        if year[j] == year[j-1]+4 and year[j] == year[j-2]+8 and year[j] == year[j-3]+12:
            for k in range(j-3, j):

                condition = (df['Year'] == year[k]) & (df['NOC'] == noc)
                find = df[condition]
                row = find.iloc[0].drop(['Year', 'NOC', 'Rank']).tolist()  # 将行数据转换为列表

                # 将 numpy 数据类型转换为 Python 原生类型

                tmp_ls += row

            k = j+1
            condition = (df['Year'] == year[k]) & (df['NOC'] == noc)
            find = df[condition]
            row = find.iloc[0].drop(['Year', 'NOC', 'Rank', 'Participants']).tolist()
            row = [item.item() if isinstance(item, np.generic) else item for item in row]
            tmp_ls = clean_list(tmp_ls)
            row = clean_list(row)
            tmp = {}
            tmp['Feats'] = tmp_ls
            tmp['label'] = row
            windows[noc].append(tmp)
        else:
            continue

print(windows)