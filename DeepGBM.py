import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import time
import os

# 全局变量
NUM_EPOCHS = 50  # 训练轮数
DATASET_PATH = 'Data/train.csv'  # 数据集路径

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        super(MetricsCallback, self).__init__()
        self.train_data = training_data
        self.val_data = validation_data
        self.train_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'specificity': [], 'auc': [], 'f1': []
        }
        self.val_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'specificity': [], 'auc': [], 'f1': []
        }
        self.times = []
        self.start_time = time.time()

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        # 计算特异度
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['f1'] = f1_score(y_true, y_pred)
        return metrics

    def on_epoch_end(self, epoch, logs=None):
        # 记录时间
        self.times.append(time.time() - self.start_time)
        
        # 训练集评估
        train_pred_proba = self.model.predict(self.train_data[0], verbose=0)
        train_pred = (train_pred_proba > 0.5).astype(int)
        train_metrics = self.calculate_metrics(
            self.train_data[1], train_pred.ravel(), train_pred_proba.ravel()
        )
        
        # 验证集评估
        val_pred_proba = self.model.predict(self.val_data[0], verbose=0)
        val_pred = (val_pred_proba > 0.5).astype(int)
        val_metrics = self.calculate_metrics(
            self.val_data[1], val_pred.ravel(), val_pred_proba.ravel()
        )
        
        # 保存指标
        for metric in self.train_metrics:
            self.train_metrics[metric].append(train_metrics[metric])
            self.val_metrics[metric].append(val_metrics[metric])

class DeepGBM:
    def __init__(self, num_trees=100, learning_rate=0.1, max_depth=3, 
                 hidden_units=[64, 32]):
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.hidden_units = hidden_units
        self.gbdt = None
        self.dnn = None
        self.scaler = StandardScaler()
        self.metrics_callback = None
        
    def _create_dnn(self, input_dim):
        model = tf.keras.Sequential()
        for units in self.hidden_units:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.2))
        # 修改输出层为单个节点
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model
    
    def fit(self, X, y, validation_data=None):
        # 确保y是二维数组
        y = np.array(y).reshape(-1, 1)
        
        # 训练GBDT
        self.gbdt = lgb.LGBMClassifier(
            n_estimators=self.num_trees,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth
        )
        self.gbdt.fit(X, y.ravel())
        
        # 获取GBDT的叶节点编码
        leaf_indices = self.gbdt.predict(X, pred_leaf=True)
        
        # 标准化原始特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 将原始特征和叶节点编码组合
        combined_features = np.hstack([X_scaled, leaf_indices])
        
        # 处理验证集
        if validation_data is not None:
            X_val, y_val = validation_data
            val_leaf_indices = self.gbdt.predict(X_val, pred_leaf=True)
            X_val_scaled = self.scaler.transform(X_val)
            val_combined_features = np.hstack([X_val_scaled, val_leaf_indices])
            validation_data = (val_combined_features, y_val)
        
        # 创建回调
        self.metrics_callback = MetricsCallback(
            training_data=(combined_features, y),
            validation_data=validation_data
        )
        
        # 创建和训练DNN
        self.dnn = self._create_dnn(combined_features.shape[1])
        self.dnn.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        
        self.dnn.fit(combined_features, y,
                     epochs=NUM_EPOCHS,
                     batch_size=256,
                     validation_data=validation_data,
                     callbacks=[self.metrics_callback],
                     verbose=1)
    
    def predict(self, X):
        # 获取GBDT的叶节点编码
        leaf_indices = self.gbdt.predict(X, pred_leaf=True)
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 组合特征
        combined_features = np.hstack([X_scaled, leaf_indices])
        
        # DNN预测
        predictions = self.dnn.predict(combined_features)
        # 确保输出是一维数组
        return predictions.ravel().round().astype(int)
    
    def predict_proba(self, X):
        leaf_indices = self.gbdt.predict(X, pred_leaf=True)
        X_scaled = self.scaler.transform(X)
        combined_features = np.hstack([X_scaled, leaf_indices])
        return self.dnn.predict(combined_features).ravel()

def plot_metrics(metrics_callback, save_dir='results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 设置图表样式
    plt.style.use('default')  # 使用默认样式
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
    epochs = range(1, len(metrics_callback.times) + 1)
    
    # 创建两个大图：按轮数和按时间
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    
    # 设置图表整体标题
    fig1.suptitle('各项指标随训练轮数的变化', fontsize=16)
    fig2.suptitle('各项指标随训练时间的变化', fontsize=16)
    
    metric_names = {
        'accuracy': '准确率',
        'precision': '精确率',
        'recall': '召回率',
        'specificity': '特异度',
        'auc': 'AUC',
        'f1': 'F1分数'
    }
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        
        # 按轮数绘图
        ax1 = axes1[row, col]
        ax1.plot(epochs, metrics_callback.train_metrics[metric], 'b-', label='训练集')
        ax1.plot(epochs, metrics_callback.val_metrics[metric], 'r-', label='测试集')
        ax1.set_title(metric_names[metric])
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('指标值')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 按时间绘图
        ax2 = axes2[row, col]
        ax2.plot(metrics_callback.times, metrics_callback.train_metrics[metric], 'b-', label='训练集')
        ax2.plot(metrics_callback.times, metrics_callback.val_metrics[metric], 'r-', label='测试集')
        ax2.set_title(metric_names[metric])
        ax2.set_xlabel('训练时间 (秒)')
        ax2.set_ylabel('指标值')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(os.path.join(save_dir, 'metrics_by_epoch.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(save_dir, 'metrics_by_time.png'), dpi=300, bbox_inches='tight')
    plt.close('all')
    
    # 保存指标数据
    metrics_data = {
        'epoch': epochs,
        'time': metrics_callback.times
    }
    
    # 添加训练集和测试集的所有指标
    for metric in metrics:
        metrics_data[f'train_{metric}'] = metrics_callback.train_metrics[metric]
        metrics_data[f'test_{metric}'] = metrics_callback.val_metrics[metric]
    
    # 保存为CSV文件
    pd.DataFrame(metrics_data).to_csv(
        os.path.join(save_dir, 'training_history.csv'), 
        index=False
    )

def main():
    # 加载数据
    data = pd.read_csv(DATASET_PATH)
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建和训练模型
    model = DeepGBM()
    model.fit(X_train, y_train, validation_data=(X_test, y_test))
    
    # 评估模型并绘制结果
    plot_metrics(model.metrics_callback)
    
    # 打印最终测试集结果
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("\n最终测试集评估结果：")
    print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test, y_pred):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test, y_pred):.4f}")
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"特异度 (Specificity): {specificity:.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

if __name__ == '__main__':
    main()

