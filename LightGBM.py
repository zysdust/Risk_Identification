import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 全局配置变量
CONFIG = {
    'NUM_ROUNDS': 100,  # 训练轮数
    'DATA_DIR': 'Data',  # 数据集目录
    'DATA_FILE': 'train.csv',  # 数据文件名
    'DATASET_NAME': 'train',  # 数据集名称
    'MAX_SAMPLES': float('inf')  # 使用的最大样本数，默认使用全部
}

class DataProcessor:
    def __init__(self, data_path, max_samples=float('inf')):
        self.data_path = data_path
        self.max_samples = max_samples
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """加载数据并进行基本检查"""
        print("正在加载数据...")
        # 读取指定数量的样本
        if self.max_samples < float('inf'):
            self.data = pd.read_csv(self.data_path, nrows=self.max_samples)
        else:
            self.data = pd.read_csv(self.data_path)
            
        print(f"数据集形状: {self.data.shape}")
        print("\n数据基本信息:")
        print(self.data.info())
        print("\n缺失值统计:")
        print(self.data.isnull().sum())
        return self
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n开始数据预处理...")
        
        # 分离特征和标签
        self.X = self.data.drop(['ID', 'Label'], axis=1)
        self.y = self.data['Label']
        
        # 处理异常值
        print("处理异常值...")
        for column in self.X.columns:
            Q1 = self.X[column].quantile(0.25)
            Q3 = self.X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.X[column] = self.X[column].clip(lower_bound, upper_bound)
        
        # 标准化特征
        print("标准化特征...")
        scaler = StandardScaler()
        self.X = pd.DataFrame(
            scaler.fit_transform(self.X),
            columns=self.X.columns
        )
        
        print("数据预处理完成")
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        print("\n划分训练集和测试集...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state,
            stratify=self.y
        )
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        return self

class LightGBMClassifier:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.metrics_history = {
            'train': {
                'iterations': [], 'time': [],
                'accuracy': [], 'precision': [], 'recall': [], 
                'specificity': [], 'auc': [], 'f1': []
            },
            'test': {
                'iterations': [], 'time': [],
                'accuracy': [], 'precision': [], 'recall': [], 
                'specificity': [], 'auc': [], 'f1': []
            }
        }
        self.start_time = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def save_metrics_to_csv(self):
        """保存评估指标到CSV文件"""
        # 创建结果数据框
        results = {
            'iteration': self.metrics_history['train']['iterations'],
            'time': self.metrics_history['train']['time'],
            'train_accuracy': self.metrics_history['train']['accuracy'],
            'train_precision': self.metrics_history['train']['precision'],
            'train_recall': self.metrics_history['train']['recall'],
            'train_specificity': self.metrics_history['train']['specificity'],
            'train_auc': self.metrics_history['train']['auc'],
            'train_f1': self.metrics_history['train']['f1'],
            'test_accuracy': self.metrics_history['test']['accuracy'],
            'test_precision': self.metrics_history['test']['precision'],
            'test_recall': self.metrics_history['test']['recall'],
            'test_specificity': self.metrics_history['test']['specificity'],
            'test_auc': self.metrics_history['test']['auc'],
            'test_f1': self.metrics_history['test']['f1']
        }
        
        # 创建数据框
        results_df = pd.DataFrame(results)
        
        # 保存到CSV文件
        results_path = f'results/metrics_{CONFIG["DATASET_NAME"]}.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n评估指标已保存到: {results_path}")
    
    def calculate_metrics(self, y_true, y_pred_proba):
        """计算所有评估指标"""
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 计算混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算各项指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        specificity = tn / (tn + fp)  # 特异度
        f1 = f1_score(y_true, y_pred)
        
        # 计算AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'auc': auc_score,
            'f1': f1
        }
    
    def train(self, X_train, y_train, X_test, y_test):
        """训练LightGBM模型"""
        print("开始训练LightGBM模型...")
        
        # 保存数据集
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # 创建results文件夹（如果不存在）
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 设置优化后的参数
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,  # 降低学习率
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,  # 增加最小叶子节点样本数
            'min_child_weight': 0.001,  # 设置最小叶子节点权重
            'min_split_gain': 0.0,  # 设置最小分裂增益
            'reg_alpha': 0.1,  # L1正则化
            'reg_lambda': 0.1,  # L2正则化
            'verbose': -1  # 减少警告信息
        }
        
        def eval_metrics(env):
            """评估回调函数"""
            iteration = env.iteration
            elapsed_time = time.time() - self.start_time
            
            # 使用当前模型进行预测
            y_train_pred = env.model.predict(X_train)
            y_test_pred = env.model.predict(X_test)
            
            # 计算训练集指标
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            
            # 计算测试集指标
            test_metrics = self.calculate_metrics(y_test, y_test_pred)
            
            # 记录训练集指标
            self.metrics_history['train']['iterations'].append(iteration)
            self.metrics_history['train']['time'].append(elapsed_time)
            for metric, value in train_metrics.items():
                self.metrics_history['train'][metric].append(value)
            
            # 记录测试集指标
            self.metrics_history['test']['iterations'].append(iteration)
            self.metrics_history['test']['time'].append(elapsed_time)
            for metric, value in test_metrics.items():
                self.metrics_history['test'][metric].append(value)
            
            # 每10轮打印一次进度
            if iteration % 10 == 0:
                print(f"\r训练进度: {iteration}/{CONFIG['NUM_ROUNDS']}轮 | "
                      f"耗时: {elapsed_time:.2f}秒 | "
                      f"训练集准确率: {train_metrics['accuracy']:.4f} | "
                      f"测试集准确率: {test_metrics['accuracy']:.4f}", end="")
        
        # 训练模型
        print("模型训练中...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=CONFIG['NUM_ROUNDS'],
            valid_sets=[train_data, test_data],
            callbacks=[eval_metrics]
        )
        
        print("\n模型训练完成！")
        
        # 获取特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        # 保存评估指标到CSV
        self.save_metrics_to_csv()
        
        # 绘制评估指标图
        self._plot_metrics_curves()
    
    def _plot_metrics_curves(self):
        """绘制评估指标曲线"""
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
        fig, axes = plt.subplots(2, 1, figsize=(15, 20))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制基于迭代次数的曲线
        ax = axes[0]
        for metric in metrics:
            ax.plot(self.metrics_history['train']['iterations'],
                   self.metrics_history['train'][metric],
                   label=f'训练集 {metric}')
            ax.plot(self.metrics_history['test']['iterations'],
                   self.metrics_history['test'][metric],
                   label=f'测试集 {metric}',
                   linestyle='--')
        
        ax.set_xlabel('训练轮数')
        ax.set_ylabel('评价指标值')
        ax.set_title('基于训练轮数的评价指标变化')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        # 绘制基于时间的曲线
        ax = axes[1]
        for metric in metrics:
            ax.plot(self.metrics_history['train']['time'],
                   self.metrics_history['train'][metric],
                   label=f'训练集 {metric}')
            ax.plot(self.metrics_history['test']['time'],
                   self.metrics_history['test'][metric],
                   label=f'测试集 {metric}',
                   linestyle='--')
        
        ax.set_xlabel('训练时间（秒）')
        ax.set_ylabel('评价指标值')
        ax.set_title('基于训练时间的评价指标变化')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/metrics_curves.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        print("\n模型评估:")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # 计算各种评估指标
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_binary))
        
        # 绘制ROC曲线
        self._plot_roc_curve(y_test, y_pred)
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(y_test, y_pred_binary)
        
        # 绘制特征重要性
        self._plot_feature_importance()
        
    def _plot_roc_curve(self, y_test, y_pred):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('接收者操作特征(ROC)曲线')
        plt.legend(loc="lower right")
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, y_test, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """绘制特征重要性"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', 
                   data=self.feature_importance.head(10))
        plt.title('特征重要性（前10个特征）')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # 构建数据文件路径
    data_path = os.path.join(CONFIG['DATA_DIR'], CONFIG['DATA_FILE'])
    
    # 数据预处理
    processor = DataProcessor(data_path, CONFIG['MAX_SAMPLES'])
    processor.load_data()
    processor.preprocess_data()
    processor.split_data(test_size=0.2)
    
    # 创建并训练模型
    classifier = LightGBMClassifier()
    classifier.train(processor.X_train, processor.y_train, 
                    processor.X_test, processor.y_test)
    
    # 评估模型
    classifier.evaluate(processor.X_test, processor.y_test)

if __name__ == "__main__":
    main()
