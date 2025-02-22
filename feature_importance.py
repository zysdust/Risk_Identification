import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 创建结果目录（如果不存在）
os.makedirs('results', exist_ok=True)

# 读取数据
df = pd.read_csv('Data/Tianchi/train.csv')  # Data/train.csv
X = df.drop('Label', axis=1)
y = df['Label']

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
lgb_model = lgb.LGBMClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42)
cat_model = CatBoost(params={'verbose': False, 'random_seed': 42})

# 训练模型
models = {
    'LightGBM': lgb_model,
    'XGBoost': xgb_model,
    'CatBoost': cat_model
}

feature_importance_dict = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    
    # 获取特征重要性
    if name == 'LightGBM':
        importance = model.feature_importances_
    elif name == 'XGBoost':
        importance = model.feature_importances_
    else:  # CatBoost
        importance = model.get_feature_importance()
    
    feature_importance_dict[name] = pd.Series(importance, index=X.columns)

# 计算平均特征重要性
avg_importance = pd.DataFrame(feature_importance_dict).mean(axis=1)
avg_importance = avg_importance.sort_values(ascending=False)

# 保存特征重要性到CSV
importance_df = pd.DataFrame(feature_importance_dict)
importance_df['Average'] = importance_df.mean(axis=1)
importance_df.sort_values('Average', ascending=False).to_csv('results/feature_importance.csv')

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
sns.barplot(x=avg_importance.values[:20], y=avg_importance.index[:20])
plt.title('Top 20 Feature Importance (Average of Multiple Models)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 为每个模型单独绘制特征重要性图
for name, importance in feature_importance_dict.items():
    plt.figure(figsize=(12, 8))
    importance = importance.sort_values(ascending=False)
    sns.barplot(x=importance.values[:20], y=importance.index[:20])
    plt.title(f'Top 20 Feature Importance - {name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(f'results/feature_importance_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("特征重要性分析完成！结果已保存在 results 文件夹中。")

