import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fastai.tabular.all import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime

# 全局配置变量
NUM_EPOCHS = 3  # 训练轮数
DATASET_PATH = 'Data/train.csv'  # 数据集路径

# 设置随机种子以确保结果可复现
set_seed(42)

# 确保results文件夹存在
if not os.path.exists('results'):
    os.makedirs('results')

def calculate_metrics(preds, targets):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    if len(preds.shape) > 1 and preds.shape[1] > 1:
        preds_class = np.argmax(preds, axis=1)
        preds_prob = preds[:, 1]  # 用于计算AUC
    else:
        preds_class = (preds > 0.5).astype(int)
        preds_prob = preds
        
    accuracy = accuracy_score(targets, preds_class)
    precision = precision_score(targets, preds_class, average='binary')
    recall = recall_score(targets, preds_class, average='binary')
    tn, fp, fn, tp = confusion_matrix(targets, preds_class).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(targets, preds_prob)
    f1 = f1_score(targets, preds_class, average='binary')
    return accuracy, precision, recall, specificity, auc, f1

class CustomRecorder(Recorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics = {
            'epoch': [], 'time': [],
            'accuracy': [], 'precision': [], 'recall': [], 
            'specificity': [], 'auc': [], 'f1': []
        }
        self.valid_metrics = {
            'epoch': [], 'time': [],
            'accuracy': [], 'precision': [], 'recall': [], 
            'specificity': [], 'auc': [], 'f1': []
        }
        self.start_time = time.time()
        
        # 创建用于保存所有指标的DataFrame
        self.metrics_df = pd.DataFrame(columns=[
            'epoch', 'time',
            'train_accuracy', 'train_precision', 'train_recall', 
            'train_specificity', 'train_auc', 'train_f1',
            'test_accuracy', 'test_precision', 'test_recall', 
            'test_specificity', 'test_auc', 'test_f1'
        ])

    def after_epoch(self):
        """每个epoch结束后的操作"""
        super().after_epoch()
        try:
            # 计算训练集指标
            self.learn.model.eval()
            with torch.no_grad():
                print(f"\n计算第 {self.epoch} 轮训练集指标...")
                train_preds, train_targets = self.learn.get_preds(dl=self.learn.dls.train)
                train_metrics = calculate_metrics(train_preds, train_targets)
                
                print(f"计算第 {self.epoch} 轮验证集指标...")
                valid_preds, valid_targets = self.learn.get_preds(dl=self.learn.dls.valid)
                valid_metrics = calculate_metrics(valid_preds, valid_targets)
            
            current_time = time.time() - self.start_time
            
            # 记录训练集指标
            self.train_metrics['epoch'].append(self.epoch)
            self.train_metrics['time'].append(current_time)
            self.train_metrics['accuracy'].append(train_metrics[0])
            self.train_metrics['precision'].append(train_metrics[1])
            self.train_metrics['recall'].append(train_metrics[2])
            self.train_metrics['specificity'].append(train_metrics[3])
            self.train_metrics['auc'].append(train_metrics[4])
            self.train_metrics['f1'].append(train_metrics[5])
            
            # 记录验证集指标
            self.valid_metrics['epoch'].append(self.epoch)
            self.valid_metrics['time'].append(current_time)
            self.valid_metrics['accuracy'].append(valid_metrics[0])
            self.valid_metrics['precision'].append(valid_metrics[1])
            self.valid_metrics['recall'].append(valid_metrics[2])
            self.valid_metrics['specificity'].append(valid_metrics[3])
            self.valid_metrics['auc'].append(valid_metrics[4])
            self.valid_metrics['f1'].append(valid_metrics[5])
            
            # 将当前轮次的所有指标添加到DataFrame
            new_row = {
                'epoch': self.epoch,
                'time': current_time,
                'train_accuracy': train_metrics[0],
                'train_precision': train_metrics[1],
                'train_recall': train_metrics[2],
                'train_specificity': train_metrics[3],
                'train_auc': train_metrics[4],
                'train_f1': train_metrics[5],
                'test_accuracy': valid_metrics[0],
                'test_precision': valid_metrics[1],
                'test_recall': valid_metrics[2],
                'test_specificity': valid_metrics[3],
                'test_auc': valid_metrics[4],
                'test_f1': valid_metrics[5]
            }
            self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([new_row])], ignore_index=True)
            
            print(f"\n第 {self.epoch}/{NUM_EPOCHS} 轮训练完成:")
            print(f"训练集 - Accuracy: {train_metrics[0]:.4f}, Precision: {train_metrics[1]:.4f}, Recall: {train_metrics[2]:.4f}")
            print(f"验证集 - Accuracy: {valid_metrics[0]:.4f}, Precision: {valid_metrics[1]:.4f}, Recall: {valid_metrics[2]:.4f}")
            print(f"已用时间: {current_time:.2f}秒")
        except Exception as e:
            print(f"计算指标时出现错误: {str(e)}")
            raise e

def plot_metrics(recorder, save_path):
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
    fig, axes = plt.subplots(2, 1, figsize=(15, 20))
    
    # 绘制按轮数的指标
    ax = axes[0]
    for metric in metrics:
        ax.plot(recorder.train_metrics['epoch'], 
                recorder.train_metrics[metric], 
                label=f'Train {metric}', marker='o')
        ax.plot(recorder.valid_metrics['epoch'], 
                recorder.valid_metrics[metric], 
                label=f'Valid {metric}', marker='o', linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics Value')
    ax.set_title('Metrics by Epoch')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    # 绘制按时间的指标
    ax = axes[1]
    for metric in metrics:
        ax.plot(recorder.train_metrics['time'], 
                recorder.train_metrics[metric], 
                label=f'Train {metric}', marker='o')
        ax.plot(recorder.valid_metrics['time'], 
                recorder.valid_metrics[metric], 
                label=f'Valid {metric}', marker='o', linestyle='--')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Metrics Value')
    ax.set_title('Metrics by Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# 加载数据
def load_data():
    print("正在加载数据...")
    df = pd.read_csv(DATASET_PATH)
    return df

# 数据预处理
def preprocess_data(df):
    print("正在进行数据预处理...")
    y = df['Label']
    X = df.drop('Label', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_df = pd.concat([X_train, pd.Series(y_train, index=X_train.index, name='Label')], axis=1)
    test_df = pd.concat([X_test, pd.Series(y_test, index=X_test.index, name='Label')], axis=1)
    
    return train_df, test_df

def main():
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    df = load_data()
    train_df, test_df = preprocess_data(df)
    
    print("正在准备数据加载器...")
    cont_names = [col for col in train_df.columns if col != 'Label']
    cat_names = []
    
    procs = [Categorify, FillMissing, Normalize]
    splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
    
    to = TabularPandas(train_df, 
                       procs=procs,
                       cat_names=cat_names,
                       cont_names=cont_names,
                       y_names='Label',
                       splits=splits)
    
    dls = to.dataloaders(bs=64)
    
    print("正在创建模型...")
    learn = tabular_learner(dls, 
                           layers=[200,100],
                           metrics=accuracy)
    
    # 替换默认的Recorder
    learn.remove_cb(Recorder)
    recorder = CustomRecorder()
    learn.add_cb(recorder)
    
    # 移除早停回调（如果存在）
    learn.remove_cbs([callback for callback in learn.cbs if isinstance(callback, EarlyStoppingCallback)])
    
    print(f"开始训练模型... 总轮数: {NUM_EPOCHS}")
    try:
        with learn.no_bar():
            learn.fit_one_cycle(NUM_EPOCHS, 1e-2)
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return
    
    print("\n训练完成！正在评估最终模型性能...")
    
    try:
        test_dl = learn.dls.test_dl(test_df)
        preds, _ = learn.get_preds(dl=test_dl)
        preds = preds.cpu().numpy()
        predictions = np.argmax(preds, axis=1)
        
        # 计算并保存最终的测试集指标
        final_metrics = {
            'accuracy': accuracy_score(test_df['Label'], predictions),
            'precision': precision_score(test_df['Label'], predictions, average='binary'),
            'recall': recall_score(test_df['Label'], predictions, average='binary'),
            'auc': roc_auc_score(test_df['Label'], preds[:, 1]),
            'f1': f1_score(test_df['Label'], predictions, average='binary')
        }
        
        # 保存训练过程中的所有指标到CSV
        metrics_save_path = f'results/training_metrics_{timestamp}.csv'
        recorder.metrics_df.to_csv(metrics_save_path, index=False)
        print(f"训练过程指标已保存到: {metrics_save_path}")
        
        # 保存最终指标到CSV
        metrics_df = pd.DataFrame([final_metrics])
        metrics_df.to_csv(f'results/final_metrics_{timestamp}.csv', index=False)
        
        # 绘制并保存图表
        print("正在生成评估指标图表...")
        plot_metrics(recorder, f'results/metrics_plot_{timestamp}.png')
        
        print(f"\n最终测试集评估结果:")
        for metric, value in final_metrics.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()

