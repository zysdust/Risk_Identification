import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import time
import os

# 全局变量设置
NUM_EPOCHS = 100  # 训练轮数
DATASET_PATH = 'Data/train.csv'  # 数据集路径
BATCH_SIZE = 32  # 批次大小
RESULTS_DIR = 'results'  # 结果保存目录

class ObliviousDecisionLayer(layers.Layer):
    def __init__(self, num_trees, depth, **kwargs):
        super(ObliviousDecisionLayer, self).__init__(**kwargs)
        self.num_trees = num_trees
        self.depth = depth
        
    def build(self, input_shape):
        # 决策节点的分裂参数
        self.decision_weights = self.add_weight(
            name='decision_weights',
            shape=(self.num_trees, self.depth, input_shape[-1]),
            initializer='random_normal',
            trainable=True
        )
        
        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(self.num_trees, self.depth),
            initializer='zeros',
            trainable=True
        )
        
        # 叶子节点的输出值
        self.leaf_weights = self.add_weight(
            name='leaf_weights',
            shape=(self.num_trees, 2 ** self.depth),
            initializer='random_normal',
            trainable=True
        )
        
    def call(self, inputs):
        # 计算每个特征的决策
        decisions = tf.einsum('bj,tdk->btd', inputs, self.decision_weights)
        decisions = decisions + self.thresholds
        decisions = tf.sigmoid(decisions)  # 将决策转换为概率
        
        # 计算每个样本在每棵树中的路径
        paths = tf.ones((tf.shape(inputs)[0], self.num_trees, 1))
        
        for d in range(self.depth):
            decision_layer = decisions[:, :, d:d+1]
            paths_left = paths * decision_layer
            paths_right = paths * (1 - decision_layer)
            paths = tf.concat([paths_left, paths_right], axis=2)
        
        # 计算每棵树的输出
        outputs = tf.einsum('btl,tl->bt', paths, self.leaf_weights)
        
        # 汇总所有树的输出
        return tf.reduce_mean(outputs, axis=1, keepdims=True)

class NODEModel(Model):
    def __init__(self, num_trees=5, depth=3):
        super(NODEModel, self).__init__()
        self.num_trees = num_trees
        self.depth = depth
        
        # 模型层定义
        self.dense1 = layers.Dense(64, activation='relu')
        self.node_layer = ObliviousDecisionLayer(num_trees, depth)
        self.dense2 = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.node_layer(x)
        return self.dense2(x)

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, test_data):
        super(MetricsCallback, self).__init__()
        self.train_x, self.train_y = train_data
        self.test_x, self.test_y = test_data
        self.metrics_history = {
            'time': [],
            'epoch': [],
            'train_accuracy': [], 'test_accuracy': [],
            'train_precision': [], 'test_precision': [],
            'train_recall': [], 'test_recall': [],
            'train_specificity': [], 'test_specificity': [],
            'train_auc': [], 'test_auc': [],
            'train_f1': [], 'test_f1': []
        }
        self.start_time = time.time()

    def calculate_metrics(self, x, y_true):
        y_pred_prob = self.model.predict(x, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = np.array(y_true)
        
        # 计算所有指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # 修复数组计算方式
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 确保输入是正确的形状
        auc = roc_auc_score(y_true, y_pred_prob.flatten())
        f1 = f1_score(y_true, y_pred)
        
        return accuracy, precision, recall, specificity, auc, f1

    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time() - self.start_time
        
        # 计算训练集指标
        train_metrics = self.calculate_metrics(self.train_x, self.train_y)
        test_metrics = self.calculate_metrics(self.test_x, self.test_y)
        
        # 记录时间和轮次
        self.metrics_history['time'].append(current_time)
        self.metrics_history['epoch'].append(epoch)
        
        # 记录训练集和测试集的所有指标
        metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
        for i, name in enumerate(metric_names):
            self.metrics_history[f'train_{name}'].append(train_metrics[i])
            self.metrics_history[f'test_{name}'].append(test_metrics[i])

def plot_metrics(metrics_history, save_dir=RESULTS_DIR):
    """绘制所有评估指标的折线图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
    
    # 创建两个大图：基于轮次和基于时间的
    for x_metric in ['epoch', 'time']:
        plt.figure(figsize=(20, 12))
        
        for i, metric in enumerate(metric_names, 1):
            plt.subplot(2, 3, i)
            
            # 绘制训练集和测试集的曲线
            plt.plot(metrics_history[x_metric], 
                    metrics_history[f'train_{metric}'], 
                    label=f'训练集 {metric.upper()}')
            plt.plot(metrics_history[x_metric], 
                    metrics_history[f'test_{metric}'], 
                    label=f'测试集 {metric.upper()}')
            
            plt.title(f'{metric.upper()} 曲线')
            plt.xlabel('训练轮次' if x_metric == 'epoch' else '训练时间(秒)')
            plt.ylabel('指标值')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'metrics_vs_{x_metric}.png'))
        plt.close()

def save_metrics_to_csv(metrics_history, save_dir=RESULTS_DIR):
    """将评估指标保存为CSV文件"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建结果数据框
    results_df = pd.DataFrame()
    
    # 添加轮次和时间列
    results_df['epoch'] = metrics_history['epoch']
    results_df['time'] = metrics_history['time']
    
    # 添加所有指标列
    metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
    for metric in metric_names:
        # 训练集指标
        results_df[f'train_{metric}'] = metrics_history[f'train_{metric}']
        # 测试集指标
        results_df[f'test_{metric}'] = metrics_history[f'test_{metric}']
    
    # 保存为CSV文件
    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n评估指标已保存至: {csv_path}")

def train_node_model(data_path=DATASET_PATH, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    """
    训练NODE模型
    
    参数:
    data_path: 数据集路径
    epochs: 训练轮数
    batch_size: 批次大小
    """
    # 加载数据
    print("正在加载数据...")
    data = pd.read_csv(data_path)
    
    # 准备特征和标签
    X = data.drop(['ID', 'Label'], axis=1)
    y = data['Label']
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建模型
    model = NODEModel()
    
    # 创建指标回调
    metrics_callback = MetricsCallback(
        train_data=(X_train, y_train),
        test_data=(X_test, y_test)
    )
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[metrics_callback],
        verbose=1
    )
    
    # 绘制并保存所有评估指标图
    plot_metrics(metrics_callback.metrics_history)
    
    # 保存评估指标到CSV文件
    save_metrics_to_csv(metrics_callback.metrics_history)
    
    # 打印最终评估结果
    print("\n最终评估结果:")
    final_metrics = {
        metric: metrics_callback.metrics_history[f'test_{metric}'][-1]
        for metric in ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
    }
    
    for metric, value in final_metrics.items():
        print(f"测试集 {metric.upper()}: {value:.4f}")
    
    return model, metrics_callback.metrics_history

if __name__ == "__main__":
    # 训练模型
    model, metrics_history = train_node_model()

