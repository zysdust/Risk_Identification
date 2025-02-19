import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor
import os
import time
from tqdm import tqdm

# 全局变量
NUM_EPOCHS = 5  # 训练轮数
DATASET_PATH = 'Data/train.csv'  # 数据集路径
RESULTS_DIR = 'results'  # 结果保存目录

def calculate_specificity(y_true, y_pred):
    """计算特异度"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def evaluate_metrics(y_true, y_pred, y_score=None):
    """计算所有评估指标"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Specificity': calculate_specificity(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    if y_score is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_score)
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path='results/svm_confusion_matrix.png'):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('SVM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics(epochs_metrics, time_metrics, save_dir='results'):
    """绘制评估指标随训练轮数和时间的变化"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC', 'F1-Score']
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('SVM Training Metrics', fontsize=16)
    
    # 绘制训练轮数-训练集指标图
    ax = axes[0, 0]
    for metric in metrics_names:
        if metric in epochs_metrics['train']:
            ax.plot(epochs_metrics['train'][metric], label=f'Train {metric}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metric Value')
    ax.set_title('Training Metrics vs Epochs (Train Set)')
    ax.legend()
    ax.grid(True)
    
    # 绘制训练轮数-测试集指标图
    ax = axes[0, 1]
    for metric in metrics_names:
        if metric in epochs_metrics['test']:
            ax.plot(epochs_metrics['test'][metric], label=f'Test {metric}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metric Value')
    ax.set_title('Training Metrics vs Epochs (Test Set)')
    ax.legend()
    ax.grid(True)
    
    # 绘制训练时间-训练集指标图
    ax = axes[1, 0]
    for metric in metrics_names:
        if metric in time_metrics['train']:
            ax.plot(time_metrics['times'], time_metrics['train'][metric], label=f'Train {metric}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Metric Value')
    ax.set_title('Training Metrics vs Time (Train Set)')
    ax.legend()
    ax.grid(True)
    
    # 绘制训练时间-测试集指标图
    ax = axes[1, 1]
    for metric in metrics_names:
        if metric in time_metrics['test']:
            ax.plot(time_metrics['times'], time_metrics['test'][metric], label=f'Test {metric}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Metric Value')
    ax.set_title('Training Metrics vs Time (Test Set)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'svm_training_metrics.png'))
    plt.close()

def save_metrics_to_csv(epochs_metrics, time_metrics, save_path='results/svm_metrics.csv'):
    """保存评估指标到CSV文件"""
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建结果数据框
    results = []
    for epoch in range(len(time_metrics['times'])):
        row = {
            'Epoch': epoch + 1,
            'Time': time_metrics['times'][epoch]
        }
        # 添加训练集指标
        for metric in epochs_metrics['train']:
            row[f'Train_{metric}'] = epochs_metrics['train'][metric][epoch]
        # 添加测试集指标
        for metric in epochs_metrics['test']:
            row[f'Test_{metric}'] = epochs_metrics['test'][metric][epoch]
        results.append(row)
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"评估指标已保存到 {save_path}")

def main():
    # 创建数据处理器实例
    processor = DataProcessor(DATASET_PATH)
    
    # 执行数据处理流程
    processor.load_data().preprocess_data().split_data(test_size=0.2)
    
    # 初始化评估指标记录器
    epochs_metrics = {'train': {}, 'test': {}}
    time_metrics = {
        'times': [],
        'train': {},
        'test': {}
    }
    for metric in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC', 'F1-Score']:
        epochs_metrics['train'][metric] = []
        epochs_metrics['test'][metric] = []
        time_metrics['train'][metric] = []
        time_metrics['test'][metric] = []
    
    # 创建SVM模型
    print("\n开始训练SVM模型...")
    svm = SVC(
        kernel='rbf',
        C=1.0,
        random_state=42,
        probability=True  # 启用概率估计以计算AUC
    )
    
    # 设置训练轮数（这里我们通过不同的C值来模拟多轮训练）
    C_values = np.logspace(-2, 2, NUM_EPOCHS)
    start_time = time.time()
    
    for epoch, C in enumerate(tqdm(C_values, desc="训练进度")):
        # 更新模型参数
        svm.C = C
        
        # 训练模型
        svm.fit(processor.X_train, processor.y_train)
        current_time = time.time() - start_time
        
        # 计算训练集指标
        y_train_pred = svm.predict(processor.X_train)
        y_train_score = svm.predict_proba(processor.X_train)[:, 1]
        train_metrics = evaluate_metrics(processor.y_train, y_train_pred, y_train_score)
        
        # 计算测试集指标
        y_test_pred = svm.predict(processor.X_test)
        y_test_score = svm.predict_proba(processor.X_test)[:, 1]
        test_metrics = evaluate_metrics(processor.y_test, y_test_pred, y_test_score)
        
        # 记录指标
        time_metrics['times'].append(current_time)
        for metric in train_metrics:
            epochs_metrics['train'][metric].append(train_metrics[metric])
            epochs_metrics['test'][metric].append(test_metrics[metric])
            time_metrics['train'][metric].append(train_metrics[metric])
            time_metrics['test'][metric].append(test_metrics[metric])
        
        # 打印当前轮次的指标
        print(f"\n轮次 {epoch+1}/{NUM_EPOCHS}, 训练时间: {current_time:.2f}秒")
        print("训练集指标:")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.4f}")
        print("\n测试集指标:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # 保存评估指标到CSV文件
    save_metrics_to_csv(epochs_metrics, time_metrics, f"{RESULTS_DIR}/svm_metrics.csv")
    
    # 绘制并保存评估指标图
    plot_metrics(epochs_metrics, time_metrics, RESULTS_DIR)
    print(f"\n评估指标图已保存到 {RESULTS_DIR}/svm_training_metrics.png")
    
    # 绘制并保存最终的混淆矩阵
    plot_confusion_matrix(processor.y_test, y_test_pred, f"{RESULTS_DIR}/svm_confusion_matrix.png")
    print(f"混淆矩阵已保存到 {RESULTS_DIR}/svm_confusion_matrix.png")

if __name__ == "__main__":
    main()

