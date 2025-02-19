import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 全局配置变量
MAX_EPOCHS = 100  # 训练轮数
DATASET_PATH = 'Data/train.csv'  # 数据集路径

def calculate_specificity(y_true, y_pred):
    """计算特异度"""
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return true_negatives / (true_negatives + false_positives + 1e-7)

def calculate_metrics(y_true, y_pred, y_prob):
    """计算所有评估指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'specificity': calculate_specificity(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred)
    }

def load_and_preprocess_data(file_path):
    """
    加载并预处理数据
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 分离特征和标签
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 处理分类特征（如果有的话）
    categorical_features = X.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
    
    return X, y

class MetricsCallback:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_metrics = []
        self.test_metrics = []
        self.epochs = []
        self.times = []
        self.start_time = time.time()
        
        # 创建结果DataFrame
        self.results_df = pd.DataFrame(columns=[
            'epoch', 'time',
            'train_accuracy', 'train_precision', 'train_recall', 
            'train_specificity', 'train_auc', 'train_f1',
            'test_accuracy', 'test_precision', 'test_recall', 
            'test_specificity', 'test_auc', 'test_f1'
        ])

    def on_iteration(self, model, epoch):
        current_time = time.time() - self.start_time
        
        # 训练集评估
        train_pred = model.predict(self.X_train)
        train_prob = model.predict_proba(self.X_train)[:, 1]
        train_metrics = calculate_metrics(self.y_train, train_pred, train_prob)
        
        # 测试集评估
        test_pred = model.predict(self.X_test)
        test_prob = model.predict_proba(self.X_test)[:, 1]
        test_metrics = calculate_metrics(self.y_test, test_pred, test_prob)
        
        # 记录指标
        self.train_metrics.append(train_metrics)
        self.test_metrics.append(test_metrics)
        self.epochs.append(epoch)
        self.times.append(current_time)
        
        # 更新DataFrame
        new_row = {
            'epoch': epoch,
            'time': current_time,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_specificity': train_metrics['specificity'],
            'train_auc': train_metrics['auc'],
            'train_f1': train_metrics['f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_specificity': test_metrics['specificity'],
            'test_auc': test_metrics['auc'],
            'test_f1': test_metrics['f1']
        }
        self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])], ignore_index=True)

def train_ebm_model():
    """训练和评估EBM模型"""
    # 创建results目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 加载数据
    X, y = load_and_preprocess_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 初始化回调和模型
    callback = MetricsCallback(X_train, y_train, X_test, y_test)
    ebm = ExplainableBoostingClassifier(
        n_jobs=-1,
        interactions=10,
        random_state=42,
        max_rounds=MAX_EPOCHS,  # 使用全局变量设置训练轮数
        early_stopping_rounds=None  # 禁用早停
    )
    
    print("开始训练EBM模型...")
    print(f"训练轮数: {MAX_EPOCHS}")
    print(f"数据集路径: {DATASET_PATH}")
    
    # 训练过程中记录指标
    for epoch in range(MAX_EPOCHS):
        ebm.fit(X_train, y_train)
        callback.on_iteration(ebm, epoch + 1)
    
    # 保存训练结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存CSV格式的详细指标
    csv_file = f'results/ebm_metrics_{timestamp}.csv'
    callback.results_df.to_csv(csv_file, index=False)
    
    # 保存JSON格式的完整结果（用于绘图）
    results = {
        'epochs': callback.epochs,
        'times': callback.times,
        'train_metrics': callback.train_metrics,
        'test_metrics': callback.test_metrics
    }
    json_file = f'results/ebm_metrics_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f)
    
    # 特征重要性分析和图表保存
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': np.abs(ebm.feature_importances_)
    })
    feature_importance = feature_importance.sort_values('重要性', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['特征'][:10], feature_importance['重要性'][:10])
    plt.xticks(rotation=45, ha='right')
    plt.title('EBM模型 - 前10个最重要特征')
    plt.tight_layout()
    plt.savefig(f'results/ebm_feature_importance_{timestamp}.png')
    plt.close()
    
    return ebm, feature_importance, csv_file, json_file

if __name__ == "__main__":
    try:
        ebm_model, feature_imp, csv_file, json_file = train_ebm_model()
        print("\n前10个最重要特征:")
        print(feature_imp.head(10))
        print(f"\nEBM模型训练完成！")
        print(f"详细指标已保存至: {csv_file}")
        print(f"JSON结果已保存至: {json_file}")
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")

