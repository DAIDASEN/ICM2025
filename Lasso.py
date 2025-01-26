import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 读取数据
with open('data.json', 'r') as f:
    data = json.load(f)

# 转换为DataFrame
rows = []
for country, entries in data.items():
    for entry in entries:
        row = {
            'Country': country,
            'Year': entry['Feats'][-1],  # 预测年份是否是主办方（第19个特征）
            **{f'Feat_{i}': entry['Feats'][i] for i in range(18)},  # 前18个特征
            'Label': entry['label'][3]
        }
        rows.append(row)
df = pd.DataFrame(rows)

print(df)

# # 特征工程
preprocessor = ColumnTransformer(
    transformers=[
        ('country', OneHotEncoder(handle_unknown='ignore'), ['Country']),  # 国家编码
        ('num', StandardScaler(), [f'Feat_{i}' for i in range(18)]),        # 数值特征标准化
        ('host_year', 'passthrough', ['Year'])                               # 保留预测年份主办方状态
    ])

# 定义模型管道
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LassoCV(cv=5, max_iter=10000))
])

# 训练
X = df.drop('Label', axis=1)
y = df['Label']
model.fit(X, y)

# 预测示例
sample = pd.DataFrame([{
    'Country': 'USA',
    'Year': 1,  # 假设是主办方
    **{f'Feat_{i}': [10, 8, 6, 24, 500, 1][i%6] for i in range(18)}  # 模拟前三届特征
}])
pred = model.predict(sample)
print(f"预测奖牌数: {pred[0]:.1f}")