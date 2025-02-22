import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# 全局配置参数
N_ITERATIONS = 100  # 训练轮数
DATA_PATH = 'Data/Tianchi/train.csv'  # 数据集路径

# 设置随机种子以确保结果可复现
np.random.seed(42)

def calculate_specificity(y_true, y_pred):
    """计算特异度 (True Negative Rate)"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'iterations': [],
            'time': [],
            'train_accuracy': [],
            'train_precision': [],
            'train_recall': [],
            'train_specificity': [],
            'train_auc': [],
            'train_f1': [],
            'test_accuracy': [],
            'test_precision': [],
            'test_recall': [],
            'test_specificity': [],
            'test_auc': [],
            'test_f1': []
        }
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def update(self, iteration, model, train_pool, test_pool):
        current_time = time.time() - self.start_time
        
        # 获取预测结果
        train_pred = model.predict(train_pool)
        test_pred = model.predict(test_pool)
        
        # 获取概率预测用于AUC计算
        train_pred_proba = model.predict_proba(train_pool)[:, 1]
        test_pred_proba = model.predict_proba(test_pool)[:, 1]
        
        # 更新指标
        self.metrics['iterations'].append(iteration)
        self.metrics['time'].append(current_time)
        
        # 训练集指标
        self.metrics['train_accuracy'].append(accuracy_score(train_pool.get_label(), train_pred))
        self.metrics['train_precision'].append(precision_score(train_pool.get_label(), train_pred))
        self.metrics['train_recall'].append(recall_score(train_pool.get_label(), train_pred))
        self.metrics['train_specificity'].append(calculate_specificity(train_pool.get_label(), train_pred))
        self.metrics['train_auc'].append(roc_auc_score(train_pool.get_label(), train_pred_proba))
        self.metrics['train_f1'].append(f1_score(train_pool.get_label(), train_pred))
        
        # 测试集指标
        self.metrics['test_accuracy'].append(accuracy_score(test_pool.get_label(), test_pred))
        self.metrics['test_precision'].append(precision_score(test_pool.get_label(), test_pred))
        self.metrics['test_recall'].append(recall_score(test_pool.get_label(), test_pred))
        self.metrics['test_specificity'].append(calculate_specificity(test_pool.get_label(), test_pred))
        self.metrics['test_auc'].append(roc_auc_score(test_pool.get_label(), test_pred_proba))
        self.metrics['test_f1'].append(f1_score(test_pool.get_label(), test_pred))

def plot_metrics(metrics_tracker, save_dir):
    """绘制评估指标图"""
    metrics_names = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 训练轮数 vs 训练集指标
    ax = axes[0, 0]
    for metric in metrics_names:
        ax.plot(metrics_tracker.metrics['iterations'],
                metrics_tracker.metrics[f'train_{metric}'],
                label=f'Train {metric.upper()}')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('指标值')
    ax.set_title('训练轮数 vs 训练集指标')
    ax.legend()
    ax.grid(True)
    
    # 训练轮数 vs 测试集指标
    ax = axes[0, 1]
    for metric in metrics_names:
        ax.plot(metrics_tracker.metrics['iterations'],
                metrics_tracker.metrics[f'test_{metric}'],
                label=f'Test {metric.upper()}')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('指标值')
    ax.set_title('训练轮数 vs 测试集指标')
    ax.legend()
    ax.grid(True)
    
    # 训练时间 vs 训练集指标
    ax = axes[1, 0]
    for metric in metrics_names:
        ax.plot(metrics_tracker.metrics['time'],
                metrics_tracker.metrics[f'train_{metric}'],
                label=f'Train {metric.upper()}')
    ax.set_xlabel('训练时间（秒）')
    ax.set_ylabel('指标值')
    ax.set_title('训练时间 vs 训练集指标')
    ax.legend()
    ax.grid(True)
    
    # 训练时间 vs 测试集指标
    ax = axes[1, 1]
    for metric in metrics_names:
        ax.plot(metrics_tracker.metrics['time'],
                metrics_tracker.metrics[f'test_{metric}'],
                label=f'Test {metric.upper()}')
    ax.set_xlabel('训练时间（秒）')
    ax.set_ylabel('指标值')
    ax.set_title('训练时间 vs 测试集指标')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_dir, f'metrics_plot_{timestamp}.png'))
    plt.close()

    # 保存指标数据到CSV
    metrics_df = pd.DataFrame(metrics_tracker.metrics)
    metrics_df.to_csv(os.path.join(save_dir, f'metrics_data_{timestamp}.csv'), index=False)

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def prepare_data(data):
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建CatBoost数据池
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)
    
    return train_pool, test_pool

def train_model(train_pool, test_pool):
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    
    # 初始化指标跟踪器
    metrics_tracker = MetricsTracker()
    
    # 初始化模型
    model = CatBoostClassifier(
        iterations=1,  # 每次只训练1轮
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=True
    )
    
    # 开始计时
    print("开始训练模型...")
    metrics_tracker.start()
    
    # 逐轮训练并记录指标
    for iteration in range(1, N_ITERATIONS + 1):
        # 训练一轮
        if iteration == 1:
            model.fit(train_pool, eval_set=test_pool)
        else:
            model.fit(train_pool, eval_set=test_pool, init_model=model)
        
        # 记录当前轮次的指标
        metrics_tracker.update(iteration, model, train_pool, test_pool)
        
    # 绘制并保存结果
    plot_metrics(metrics_tracker, 'results')
    
    return model, metrics_tracker

def main():
    # 加载数据
    print("加载数据...")
    data = load_data(DATA_PATH)
    
    # 准备数据
    print("准备数据...")
    train_pool, test_pool = prepare_data(data)
    
    # 训练模型并记录指标
    model, metrics_tracker = train_model(train_pool, test_pool)
    
    print("\n训练完成！结果已保存到results目录")

if __name__ == "__main__":
    main()

