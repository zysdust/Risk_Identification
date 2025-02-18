import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import time
import os
from data_processor import DataProcessor

# 全局变量
N_EPOCHS = 100  # 训练轮数
DATASET_PATH = 'Data/train.csv'  # 数据集路径
BATCH_SIZE = 1024  # 批次大小

class TabNetClassifier:
    def __init__(self, feature_dims, num_classes=2):
        self.feature_dims = feature_dims
        self.num_classes = num_classes
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
        self._build_model()
    
    def _build_model(self):
        """构建TabNet模型"""
        # 特征处理层
        feature_input = tf.keras.layers.Input(shape=(self.feature_dims,))
        
        # TabNet的特征选择层
        num_features = 64
        feature_transformer = tf.keras.Sequential([
            tf.keras.layers.Dense(num_features, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_features, activation='relu'),
            tf.keras.layers.BatchNormalization()
        ])
        
        # 注意力机制层
        attention = tf.keras.layers.Dense(self.feature_dims, activation='sigmoid')(feature_input)
        attended_features = tf.keras.layers.Multiply()([feature_input, attention])
        
        # 特征转换
        transformed = feature_transformer(attended_features)
        
        # 输出层 - 修改为单个输出节点
        output = tf.keras.layers.Dense(1, activation='sigmoid')(transformed)
        
        # 构建模型
        self.model = tf.keras.Model(inputs=feature_input, outputs=output)
        
        # 编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """计算评估指标"""
        # 将概率转换为类别
        y_pred_class = (y_pred_proba.reshape(-1) > 0.5).astype(int)
        y_pred_proba = y_pred_proba.reshape(-1)
        
        # 计算混淆矩阵元素
        tn = np.sum((y_true == 0) & (y_pred_class == 0))
        fp = np.sum((y_true == 0) & (y_pred_class == 1))
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_class),
            'precision': precision_score(y_true, y_pred_class),
            'recall': recall_score(y_true, y_pred_class),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'auc': roc_auc_score(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred_class)
        }
        return metrics
    
    def train(self, X_train, y_train, X_test, y_test):
        """训练模型并记录训练过程"""
        print("\n开始训练TabNet模型...")
        
        # 创建results文件夹（如果不存在）
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # 记录开始时间
        start_time = time.time()
        
        # 转换数据格式
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)
        ).batch(BATCH_SIZE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (X_test, y_test)
        ).batch(BATCH_SIZE)
        
        # 训练循环
        for epoch in range(N_EPOCHS):
            # 训练一轮
            self.model.fit(
                train_dataset,
                epochs=1,
                verbose=0
            )
            
            # 记录当前轮次
            current_time = time.time() - start_time
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['time'].append(current_time)
            
            # 计算训练集指标
            train_pred = self.model.predict(X_train, verbose=0)
            train_metrics = self.calculate_metrics(y_train, train_pred, train_pred)
            for metric, value in train_metrics.items():
                self.training_history['train_metrics'][metric].append(value)
            
            # 计算测试集指标
            test_pred = self.model.predict(X_test, verbose=0)
            test_metrics = self.calculate_metrics(y_test, test_pred, test_pred)
            for metric, value in test_metrics.items():
                self.training_history['test_metrics'][metric].append(value)
            
            # 每10轮打印一次进度
            if (epoch + 1) % 10 == 0:
                print(f"轮次 {epoch + 1}/{N_EPOCHS}")
                print(f"训练集准确率: {train_metrics['accuracy']:.4f}")
                print(f"测试集准确率: {test_metrics['accuracy']:.4f}")
        
        print("模型训练完成")
        self.plot_training_history()
        self.save_metrics()
        return self
    
    def plot_training_history(self):
        """绘制训练历史折线图"""
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            
            # 绘制训练集指标
            plt.plot(
                self.training_history['time'],
                self.training_history['train_metrics'][metric],
                label=f'训练集 {metric}'
            )
            
            # 绘制测试集指标
            plt.plot(
                self.training_history['time'],
                self.training_history['test_metrics'][metric],
                label=f'测试集 {metric}'
            )
            
            plt.xlabel('时间 (秒)')
            plt.ylabel(f'{metric}')
            plt.title(f'{metric} vs 时间')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_metrics.png')
        plt.close()
    
    def save_metrics(self):
        """保存训练指标到CSV文件"""
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
        
        metrics_df.to_csv('results/training_history.csv', index=False)

if __name__ == "__main__":
    # 创建数据处理器实例
    processor = DataProcessor(DATASET_PATH)
    
    # 执行数据处理流程
    processor.load_data().analyze_data().preprocess_data().split_data()
    
    # 创建并训练TabNet模型
    tabnet = TabNetClassifier(feature_dims=processor.X_train.shape[1])
    tabnet.train(processor.X_train, processor.y_train, 
                 processor.X_test, processor.y_test)

