import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def load_latest_results():
    """加载最新的实验结果文件"""
    results_files = glob.glob('results/ebm_metrics_*.json')
    if not results_files:
        raise FileNotFoundError("没有找到结果文件")
    
    latest_file = max(results_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_metrics(results_file='results/metrics_history.json'):
    # 加载实验结果
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 获取数据
    epochs = range(1, results['epochs'] + 1)
    times = results['times']
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
    
    # 设置图表样式
    plt.style.use('seaborn')
    
    # 创建两个大图（按轮数和时间）
    fig1, axes1 = plt.subplots(3, 2, figsize=(15, 18))
    fig2, axes2 = plt.subplots(3, 2, figsize=(15, 18))
    
    # 设置标题
    fig1.suptitle('评估指标随训练轮数的变化', fontsize=16)
    fig2.suptitle('评估指标随训练时间的变化', fontsize=16)
    
    # 绘制每个指标的图表
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        
        # 按轮数绘制
        ax1 = axes1[row, col]
        ax1.plot(epochs, results['train_metrics'][metric], label='训练集')
        ax1.plot(epochs, results['val_metrics'][metric], label='验证集')
        ax1.axhline(y=results['test_metrics'][metric], color='r', linestyle='--', label='测试集')
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel(metric.upper())
        ax1.set_title(f'{metric.upper()} vs 训练轮数')
        ax1.legend()
        ax1.grid(True)
        
        # 按时间绘制
        ax2 = axes2[row, col]
        ax2.plot(times, results['train_metrics'][metric], label='训练集')
        ax2.plot(times, results['val_metrics'][metric], label='验证集')
        ax2.axhline(y=results['test_metrics'][metric], color='r', linestyle='--', label='测试集')
        ax2.set_xlabel('训练时间 (秒)')
        ax2.set_ylabel(metric.upper())
        ax2.set_title(f'{metric.upper()} vs 训练时间')
        ax2.legend()
        ax2.grid(True)
    
    # 调整布局
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    fig1.savefig('results/metrics_vs_epochs.png', dpi=300, bbox_inches='tight')
    fig2.savefig('results/metrics_vs_time.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print("评估指标图表已保存到 results 文件夹中。")

if __name__ == "__main__":
    try:
        results = load_latest_results()
        plot_metrics()
    except Exception as e:
        print(f"绘图过程中发生错误: {str(e)}") 