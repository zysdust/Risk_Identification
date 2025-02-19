import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ngboost import NGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime

# 全局配置参数
N_ESTIMATORS = 100  # 训练轮数
DATA_PATH = 'Data/train.csv'  # 数据集路径

# 设置随机种子
np.random.seed(42)

# 创建results文件夹
if not os.path.exists('results'):
    os.makedirs('results')

# 计算特异度
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# 评估所有指标
def calculate_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': specificity_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba[:, 1]),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

class MetricsTracker:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_metrics_history = {
            'iterations': [],
            'time': [],
            'Accuracy': [], 'Precision': [], 'Recall': [],
            'Specificity': [], 'AUC': [], 'F1-Score': []
        }
        self.test_metrics_history = {
            'iterations': [],
            'time': [],
            'Accuracy': [], 'Precision': [], 'Recall': [],
            'Specificity': [], 'AUC': [], 'F1-Score': []
        }
        self.start_time = time.time()
        
        # 创建详细的CSV文件
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f'results/training_history_{self.timestamp}.csv'
        
        # 创建CSV文件头
        header = ['Iteration', 'Time', 
                 'Train_Accuracy', 'Train_Precision', 'Train_Recall', 
                 'Train_Specificity', 'Train_AUC', 'Train_F1-Score',
                 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 
                 'Test_Specificity', 'Test_AUC', 'Test_F1-Score']
        
        with open(self.results_file, 'w') as f:
            f.write(','.join(header) + '\n')

    def update(self, model, current_iter):
        current_time = time.time() - self.start_time

        # 计算训练集指标
        y_train_pred = model.predict(self.X_train)
        y_train_pred_proba = model.predict_proba(self.X_train)
        train_metrics = calculate_metrics(self.y_train, y_train_pred, y_train_pred_proba)

        # 计算测试集指标
        y_test_pred = model.predict(self.X_test)
        y_test_pred_proba = model.predict_proba(self.X_test)
        test_metrics = calculate_metrics(self.y_test, y_test_pred, y_test_pred_proba)

        # 记录历史
        self.train_metrics_history['iterations'].append(current_iter)
        self.test_metrics_history['iterations'].append(current_iter)
        self.train_metrics_history['time'].append(current_time)
        self.test_metrics_history['time'].append(current_time)

        for metric in train_metrics:
            self.train_metrics_history[metric].append(train_metrics[metric])
            self.test_metrics_history[metric].append(test_metrics[metric])

        # 将当前轮次的所有指标写入CSV文件
        metrics_row = [
            current_iter, current_time,
            train_metrics['Accuracy'], train_metrics['Precision'], 
            train_metrics['Recall'], train_metrics['Specificity'],
            train_metrics['AUC'], train_metrics['F1-Score'],
            test_metrics['Accuracy'], test_metrics['Precision'],
            test_metrics['Recall'], test_metrics['Specificity'],
            test_metrics['AUC'], test_metrics['F1-Score']
        ]
        
        with open(self.results_file, 'a') as f:
            f.write(','.join(map(str, metrics_row)) + '\n')

        # 打印进度
        print(f"\r训练进度: {current_iter}/{N_ESTIMATORS}轮 | 用时: {current_time:.2f}秒 | "
              f"测试集准确率: {test_metrics['Accuracy']:.4f}", end="")

def plot_metrics(train_history, test_history, x_axis='iterations'):
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC', 'F1-Score']
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'NGBoost模型评估指标 (基于{x_axis})')
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        
        x_train = train_history[x_axis]
        x_test = test_history[x_axis]
        
        axes[row, col].plot(x_train, train_history[metric], label='训练集')
        axes[row, col].plot(x_test, test_history[metric], label='测试集')
        
        axes[row, col].set_title(metric)
        axes[row, col].set_xlabel('训练轮数' if x_axis == 'iterations' else '训练时间(秒)')
        axes[row, col].set_ylabel('指标值')
        axes[row, col].grid(True)
        axes[row, col].legend()
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/metrics_{x_axis}_{timestamp}.png')
    plt.close()

# 加载数据
def load_data(file_path=DATA_PATH):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None

# 数据预处理
def preprocess_data(df):
    if df is None:
        return None, None, None, None
    
    # 分离特征和标签
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# 训练NGBoost模型
def train_ngboost(X_train, y_train, X_test, y_test):
    # 创建指标跟踪器
    tracker = MetricsTracker(X_train, X_test, y_train, y_test)
    
    print("开始训练NGBoost模型...")
    
    # 训练模型并在每轮结束后评估
    for i in range(1, N_ESTIMATORS + 1):
        # 训练一个新的模型到第i轮
        current_model = NGBClassifier(
            n_estimators=i,
            learning_rate=0.01,
            natural_gradient=True,
            verbose=False,
            random_state=42
        )
        
        # 训练当前模型
        current_model.fit(X_train, y_train)
        
        # 更新跟踪器
        tracker.update(current_model, i)
    
    # 返回最后一轮的模型
    return current_model, tracker

def softmax(x, axis=1):
    """计算softmax概率"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def main():
    print("正在加载数据...")
    df = load_data()
    
    if df is not None:
        print("正在预处理数据...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        print(f"正在训练NGBoost模型 (训练轮数: {N_ESTIMATORS})...")
        model, tracker = train_ngboost(X_train, y_train, X_test, y_test)
        
        print("\n正在生成评估图表...")
        # 基于训练轮数的图表
        plot_metrics(tracker.train_metrics_history, tracker.test_metrics_history, 'iterations')
        # 基于训练时间的图表
        plot_metrics(tracker.train_metrics_history, tracker.test_metrics_history, 'time')
        
        print("\n模型训练和评估完成！")
        print(f"结果已保存到 results 文件夹中")
        print(f"详细训练历史已保存到: {tracker.results_file}")
    else:
        print("由于数据加载错误，程序终止。")

if __name__ == "__main__":
    main()

