import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import time
import os

# 全局变量
NUM_EPOCHS = 100  # 训练轮数
DATASET_PATH = 'Data/train.csv'  # 数据集路径

def calculate_specificity(y_true, y_pred):
    # 计算特异度 (True Negative Rate)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def evaluate_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Specificity': calculate_specificity(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    if y_pred_proba is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    return metrics

def plot_metrics(train_metrics, test_metrics, epochs, times, save_dir='results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC', 'F1-Score']
    fig, axes = plt.subplots(2, 1, figsize=(15, 20))
    
    # 绘制基于训练轮数的指标
    for metric in metrics:
        if metric in train_metrics:
            axes[0].plot(epochs, train_metrics[metric], label=f'Train {metric}', marker='o')
        if metric in test_metrics:
            axes[0].plot(epochs, test_metrics[metric], label=f'Test {metric}', marker='s', linestyle='--')
    
    axes[0].set_xlabel('训练轮数')
    axes[0].set_ylabel('指标值')
    axes[0].set_title('不同评价指标随训练轮数的变化')
    axes[0].grid(True)
    axes[0].legend()
    
    # 绘制基于训练时间的指标
    for metric in metrics:
        if metric in train_metrics:
            axes[1].plot(times, train_metrics[metric], label=f'Train {metric}', marker='o')
        if metric in test_metrics:
            axes[1].plot(times, test_metrics[metric], label=f'Test {metric}', marker='s', linestyle='--')
    
    axes[1].set_xlabel('训练时间 (秒)')
    axes[1].set_ylabel('指标值')
    axes[1].set_title('不同评价指标随训练时间的变化')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.close()

def save_metrics_to_csv(train_metrics, test_metrics, epochs, times, save_dir='results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建结果数据框
    results_data = {
        'Epoch': epochs,
        'Time': times
    }
    
    # 添加训练集指标
    for metric in train_metrics:
        results_data[f'Train_{metric}'] = train_metrics[metric]
    
    # 添加测试集指标
    for metric in test_metrics:
        results_data[f'Test_{metric}'] = test_metrics[metric]
    
    # 保存为CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

def train_and_evaluate():
    # 读取数据
    try:
        data = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"错误：未找到数据文件 {DATASET_PATH}")
        return
    
    # 确保Label列存在
    if 'Label' not in data.columns:
        print("错误：数据集中未找到'Label'列")
        return
    
    # 分离特征和标签
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 重新组合训练集和测试集
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # 初始化记录指标的字典
    train_metrics = {metric: [] for metric in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC', 'F1-Score']}
    test_metrics = {metric: [] for metric in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC', 'F1-Score']}
    epochs = []
    times = []
    start_time = time.time()
    
    # 设置保存路径
    save_path = 'autogluon_models'
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n开始第 {epoch + 1} 轮训练...")
        
        # 初始化TabularPredictor
        predictor = TabularPredictor(
            label='Label',
            path=f'{save_path}_epoch_{epoch + 1}',
            eval_metric='accuracy',
            problem_type='binary'  # 明确指定为二分类问题
        )
        
        # 训练模型
        predictor.fit(
            train_data,
            time_limit=600,  # 训练时间限制为600秒
            presets='best_quality',  # 使用最佳质量预设
            auto_stack=True,  # 启用模型堆叠
            hyperparameter_tune_kwargs=None  # 禁用早停
        )
        
        # 获取预测结果
        train_pred = predictor.predict(train_data)
        test_pred = predictor.predict(test_data)
        
        # 获取预测概率（直接获取正类的概率）
        train_pred_proba = predictor.predict_proba(train_data).iloc[:, 1]  # 获取第二列（正类）概率
        test_pred_proba = predictor.predict_proba(test_data).iloc[:, 1]  # 获取第二列（正类）概率
        
        # 计算并记录指标
        train_metrics_values = evaluate_metrics(y_train, train_pred, train_pred_proba)
        test_metrics_values = evaluate_metrics(y_test, test_pred, test_pred_proba)
        
        # 更新指标记录
        for metric in train_metrics:
            train_metrics[metric].append(train_metrics_values[metric])
            test_metrics[metric].append(test_metrics_values[metric])
        
        epochs.append(epoch + 1)
        times.append(time.time() - start_time)
        
        # 打印当前轮次的评估结果
        print(f"\n第 {epoch + 1} 轮训练评估结果：")
        print("\n训练集：")
        for metric, value in train_metrics_values.items():
            print(f"{metric}: {value:.4f}")
        print("\n测试集：")
        for metric, value in test_metrics_values.items():
            print(f"{metric}: {value:.4f}")
        
        # 保存每轮的详细结果
        save_metrics_to_csv(train_metrics, test_metrics, epochs, times)
    
    # 绘制并保存图表
    plot_metrics(train_metrics, test_metrics, epochs, times)
    
    # 输出最终模型的详细性能指标
    print("\n最终模型性能详情：")
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(leaderboard)
    leaderboard.to_csv(os.path.join('results', 'final_model_leaderboard.csv'))

if __name__ == "__main__":
    train_and_evaluate()

