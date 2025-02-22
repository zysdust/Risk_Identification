import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

# 全局配置参数
MAX_ITERATIONS = 100  # 训练轮数
DATASET_PATH = 'Data/Tianchi/train.csv'  # 数据集路径

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 确保results文件夹存在
if not os.path.exists('results'):
    os.makedirs('results')

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Specificity': calculate_specificity(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'F1-Score': f1_score(y_true, y_pred)
    }

# 加载数据
def load_data(file_path=DATASET_PATH):
    print("正在加载数据...")
    df = pd.read_csv(file_path)
    return df

# 数据预处理
def preprocess_data(df):
    print("正在预处理数据...")
    # 分离特征和标签
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 处理缺失值
    X = X.fillna(X.mean())
    
    return X, y

def train_with_monitoring(X_train, y_train, X_test, y_test, max_iter=MAX_ITERATIONS):
    metrics_history = {
        'train_metrics_history': [],
        'test_metrics_history': [],
        'times': []
    }
    
    start_time = time.time()
    
    print("开始训练模型...")
    print(f"总训练轮数: {max_iter}")
    
    # 使用tqdm创建进度条
    for i in tqdm(range(max_iter), desc="训练进度"):
        # 训练一个新模型，迭代次数为i+1
        model = HistGradientBoostingClassifier(
            max_iter=i+1,
            learning_rate=0.1,
            max_depth=None,
            random_state=42,
            early_stopping=False,
            validation_fraction=None
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 记录当前时间
        current_time = time.time() - start_time
        
        # 计算训练集指标
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)
        
        # 计算测试集指标
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
        
        # 保存指标
        metrics_history['train_metrics_history'].append(train_metrics)
        metrics_history['test_metrics_history'].append(test_metrics)
        metrics_history['times'].append(current_time)
        
        # 保存到CSV
        save_metrics_to_csv(i, current_time, train_metrics, test_metrics)
    
    return model, metrics_history

def save_metrics_to_csv(iteration, current_time, train_metrics, test_metrics):
    metrics_dict = {
        'Iteration': iteration + 1,
        'Time': current_time,
        'Train_Accuracy': train_metrics['Accuracy'],
        'Train_Precision': train_metrics['Precision'],
        'Train_Recall': train_metrics['Recall'],
        'Train_Specificity': train_metrics['Specificity'],
        'Train_AUC': train_metrics['AUC'],
        'Train_F1-Score': train_metrics['F1-Score'],
        'Test_Accuracy': test_metrics['Accuracy'],
        'Test_Precision': test_metrics['Precision'],
        'Test_Recall': test_metrics['Recall'],
        'Test_Specificity': test_metrics['Specificity'],
        'Test_AUC': test_metrics['AUC'],
        'Test_F1-Score': test_metrics['F1-Score']
    }
    
    df = pd.DataFrame([metrics_dict])
    
    # 如果文件不存在，创建新文件并写入表头
    if not os.path.exists('results/training_metrics.csv'):
        df.to_csv('results/training_metrics.csv', index=False)
    else:
        # 如果文件存在，追加数据而不写入表头
        df.to_csv('results/training_metrics.csv', mode='a', header=False, index=False)

def plot_metrics(metrics_history, save_path='results'):
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC', 'F1-Score']
    iterations = range(1, len(metrics_history['train_metrics_history']) + 1)
    
    # 创建2x1的子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # 绘制基于训练轮数的指标
    for metric in metrics:
        train_values = [m[metric] for m in metrics_history['train_metrics_history']]
        test_values = [m[metric] for m in metrics_history['test_metrics_history']]
        ax1.plot(iterations, train_values, label=f'Train {metric}', linestyle='-')
        ax1.plot(iterations, test_values, label=f'Test {metric}', linestyle='--')
    
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('指标值')
    ax1.set_title('评估指标随训练轮数的变化')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    # 绘制基于训练时间的指标
    for metric in metrics:
        train_values = [m[metric] for m in metrics_history['train_metrics_history']]
        test_values = [m[metric] for m in metrics_history['test_metrics_history']]
        ax2.plot(metrics_history['times'], train_values, label=f'Train {metric}', linestyle='-')
        ax2.plot(metrics_history['times'], test_values, label=f'Test {metric}', linestyle='--')
    
    ax2.set_xlabel('训练时间 (秒)')
    ax2.set_ylabel('指标值')
    ax2.set_title('评估指标随训练时间的变化')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_plot.png'), bbox_inches='tight')
    plt.close()

# 模型训练和评估
def train_and_evaluate():
    # 加载数据
    df = load_data()
    X, y = preprocess_data(df)
    
    # 划分训练集和测试集
    print("正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型并监控过程
    model, metrics_history = train_with_monitoring(X_train, y_train, X_test, y_test, MAX_ITERATIONS)
    
    # 绘制并保存图表
    print("正在生成评估指标图表...")
    plot_metrics(metrics_history)
    
    return metrics_history['test_metrics_history'][-1], model

if __name__ == "__main__":
    try:
        # 训练模型并获取评估指标
        metrics, model = train_and_evaluate()
        
        # 打印最终评估指标
        print("\n=== 最终模型评估结果 ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print("\n结果已保存在 results 文件夹中")
        print(f"- 训练过程指标: results/training_metrics.csv")
        print(f"- 评估指标图表: results/metrics_plot.png")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")

