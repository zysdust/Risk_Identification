import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor
import time
import os

# 全局变量
N_EPOCHS = 100  # 训练轮数
DATASET_PATH = 'Data/Tianchi/train.csv'  # 数据集路径

class XGBoostClassifier:
    def __init__(self, params=None):
        self.params = params if params is not None else {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': N_EPOCHS,  # 使用全局变量
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = None
        self.training_history = {
            'epochs': [],
            'time': [],
            'train_metrics': {
                'accuracy': [], 'precision': [], 'recall': [], 
                'specificity': [], 'auc': [], 'f1': []
            },
            'test_metrics': {
                'accuracy': [], 'precision': [], 'recall': [], 
                'specificity': [], 'auc': [], 'f1': []
            }
        }
        
    def calculate_metrics(self, X, y_true):
        """计算所有评估指标"""
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # 计算混淆矩阵元素
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'auc': roc_auc_score(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred)
        }
        return metrics
        
    def train(self, X_train, y_train, X_test, y_test):
        """训练XGBoost模型并记录训练过程"""
        print("\n开始训练XGBoost模型...")
        
        # 创建results文件夹（如果不存在）
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # 创建DMatrix数据格式
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练模型
        for epoch in range(self.params['n_estimators']):
            # 训练一轮
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=1,
                xgb_model=self.model if self.model else None
            )
            
            # 记录当前轮次
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['time'].append(time.time() - start_time)
            
            # 计算并记录训练集指标
            train_metrics = self.calculate_metrics(X_train, y_train)
            for metric, value in train_metrics.items():
                self.training_history['train_metrics'][metric].append(value)
            
            # 计算并记录测试集指标
            test_metrics = self.calculate_metrics(X_test, y_test)
            for metric, value in test_metrics.items():
                self.training_history['test_metrics'][metric].append(value)
            
            # 每10轮打印一次进度
            if (epoch + 1) % 10 == 0:
                print(f"轮次 {epoch + 1}/{self.params['n_estimators']}")
                print(f"训练集准确率: {train_metrics['accuracy']:.4f}")
                print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
        
        print("模型训练完成")
        self.plot_training_history()
        self.save_metrics()
        return self
    
    def predict(self, X):
        """模型预测"""
        dtest = xgb.DMatrix(X)
        return (self.model.predict(dtest) > 0.5).astype(int)
    
    def predict_proba(self, X):
        """预测概率"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def plot_training_history(self):
        """绘制训练历史折线图"""
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 绘制按轮次的训练集指标
        ax = axes[0, 0]
        for metric in metrics:
            ax.plot(self.training_history['epochs'], 
                   self.training_history['train_metrics'][metric],
                   label=f'Train {metric.capitalize()}')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel('指标值')
        ax.set_title('训练集评价指标 vs 训练轮数')
        ax.legend()
        ax.grid(True)
        
        # 绘制按轮次的测试集指标
        ax = axes[0, 1]
        for metric in metrics:
            ax.plot(self.training_history['epochs'], 
                   self.training_history['test_metrics'][metric],
                   label=f'Test {metric.capitalize()}')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel('指标值')
        ax.set_title('测试集评价指标 vs 训练轮数')
        ax.legend()
        ax.grid(True)
        
        # 绘制按时间的训练集指标
        ax = axes[1, 0]
        for metric in metrics:
            ax.plot(self.training_history['time'], 
                   self.training_history['train_metrics'][metric],
                   label=f'Train {metric.capitalize()}')
        ax.set_xlabel('训练时间（秒）')
        ax.set_ylabel('指标值')
        ax.set_title('训练集评价指标 vs 训练时间')
        ax.legend()
        ax.grid(True)
        
        # 绘制按时间的测试集指标
        ax = axes[1, 1]
        for metric in metrics:
            ax.plot(self.training_history['time'], 
                   self.training_history['test_metrics'][metric],
                   label=f'Test {metric.capitalize()}')
        ax.set_xlabel('训练时间（秒）')
        ax.set_ylabel('指标值')
        ax.set_title('测试集评价指标 vs 训练时间')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_metrics.png')
        plt.close()
    
    def save_metrics(self):
        """保存训练指标到CSV文件"""
        # 创建包含所有指标的DataFrame
        metrics_df = pd.DataFrame({
            'epoch': self.training_history['epochs'],
            'time': self.training_history['time'],
            'train_accuracy': self.training_history['train_metrics']['accuracy'],
            'train_precision': self.training_history['train_metrics']['precision'],
            'train_recall': self.training_history['train_metrics']['recall'],
            'train_specificity': self.training_history['train_metrics']['specificity'],
            'train_auc': self.training_history['train_metrics']['auc'],
            'train_f1': self.training_history['train_metrics']['f1'],
            'test_accuracy': self.training_history['test_metrics']['accuracy'],
            'test_precision': self.training_history['test_metrics']['precision'],
            'test_recall': self.training_history['test_metrics']['recall'],
            'test_specificity': self.training_history['test_metrics']['specificity'],
            'test_auc': self.training_history['test_metrics']['auc'],
            'test_f1': self.training_history['test_metrics']['f1']
        })
        
        # 保存到CSV文件
        metrics_df.to_csv('results/training_history.csv', index=False)

if __name__ == "__main__":
    # 创建数据处理器实例
    processor = DataProcessor(DATASET_PATH)  # 使用全局变量
    
    # 执行数据处理流程
    processor.load_data().analyze_data().preprocess_data().split_data()
    
    # 创建并训练XGBoost模型
    xgb_clf = XGBoostClassifier()
    xgb_clf.train(processor.X_train, processor.y_train, 
                  processor.X_test, processor.y_test)

