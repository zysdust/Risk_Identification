import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time
import os
import json
import matplotlib.pyplot as plt

# 全局变量
NUM_EPOCHS = 100  # 训练轮数
DATASET_PATH = 'Data/Tianchi/train.csv'  # 数据集路径

# 定义GTN模型
class GatedTransformerNetwork(tf.keras.Model):
    def __init__(self, num_features, num_heads=4, ff_dim=32, num_transformer_blocks=2, mlp_units=[32], dropout=0.1):
        super(GatedTransformerNetwork, self).__init__()
        
        self.input_reshape = tf.keras.layers.Reshape((1, num_features))
        
        self.mlp_layers = []
        for dim in mlp_units:
            self.mlp_layers.extend([
                tf.keras.layers.Dense(dim, activation="relu"),
                tf.keras.layers.Dropout(dropout)
            ])
            
        self.transformer_blocks = []
        for _ in range(num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(num_features, num_heads, ff_dim, dropout)
            )
            
        self.gate = tf.keras.layers.Dense(num_features, activation='sigmoid')
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        # 确保输入维度正确
        x = self.input_reshape(inputs)
        
        # Gate mechanism
        gate_values = self.gate(x)
        x = x * gate_values
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
            
        # 压缩序列维度
        x = tf.squeeze(x, axis=1)
        
        # MLP layers
        for layer in self.mlp_layers:
            x = layer(x)
            
        return self.final_layer(x)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 修改 key_dim 计算方式
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=max(1, embed_dim // num_heads),
            value_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        
    def call(self, inputs):
        # 添加序列维度 [batch_size, 1, features]
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
        
        # 注意力机制
        attn_output = self.att(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=None
        )
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# 加载数据
def load_and_preprocess_data():
    # 读取数据
    data = pd.read_csv(DATASET_PATH)
    
    # 分离特征和标签
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

class MetricsHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MetricsHistory, self).__init__()
        self.times = []
        self.train_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'specificity': [], 'auc': [], 'f1': []
        }
        self.val_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'specificity': [], 'auc': [], 'f1': []
        }
        self.start_time = None
        self.metrics_df = pd.DataFrame()
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def calculate_metrics(self, y_true, y_pred_proba):
        y_pred = (y_pred_proba > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'specificity': tn / (tn + fp),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred)
        }
        return metrics
        
    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time() - self.start_time
        self.times.append(current_time)
        
        # 计算训练集指标
        train_pred = self.model.predict(self.model.train_data[0], verbose=0)
        train_metrics = self.calculate_metrics(self.model.train_data[1], train_pred)
        
        # 计算验证集指标
        val_pred = self.model.predict(self.model.val_data[0], verbose=0)
        val_metrics = self.calculate_metrics(self.model.val_data[1], val_pred)
        
        # 保存指标
        for metric in self.train_metrics:
            self.train_metrics[metric].append(train_metrics[metric])
            self.val_metrics[metric].append(val_metrics[metric])
        
        # 更新DataFrame
        epoch_data = {
            'epoch': epoch + 1,
            'time': current_time,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_specificity': train_metrics['specificity'],
            'train_auc': train_metrics['auc'],
            'train_f1': train_metrics['f1'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_specificity': val_metrics['specificity'],
            'val_auc': val_metrics['auc'],
            'val_f1': val_metrics['f1']
        }
        
        self.metrics_df = pd.concat([
            self.metrics_df, 
            pd.DataFrame([epoch_data])
        ], ignore_index=True)
        
        # 保存CSV文件
        self.metrics_df.to_csv('results/training_metrics.csv', index=False)

def main():
    # 创建results文件夹
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 加载和预处理数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 进一步划分训练集为训练集和验证集
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 获取特征数量
    num_features = X_train.shape[1]
    print(f"特征数量: {num_features}")
    
    # 创建模型
    model = GatedTransformerNetwork(
        num_features=num_features,
        num_heads=2,
        ff_dim=64,
        num_transformer_blocks=2,
        mlp_units=[128, 64],
        dropout=0.2
    )
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 保存数据供回调函数使用
    model.train_data = (X_train_final, y_train_final)
    model.val_data = (X_val, y_val)
    
    # 创建自定义指标记录器
    metrics_history = MetricsHistory()
    
    # 训练模型
    history = model.fit(
        X_train_final, y_train_final,
        batch_size=64,
        epochs=NUM_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[
            metrics_history,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
        ]
    )
    
    # 在测试集上评估
    y_pred_proba = model.predict(X_test)
    test_metrics = metrics_history.calculate_metrics(y_test, y_pred_proba)
    
    # 保存实验结果
    results = {
        'times': metrics_history.times,
        'train_metrics': metrics_history.train_metrics,
        'val_metrics': metrics_history.val_metrics,
        'test_metrics': test_metrics,
        'epochs': len(metrics_history.times)
    }
    
    # 保存结果到JSON文件
    with open('results/metrics_history.json', 'w') as f:
        json.dump(results, f)
    
    # 将测试集结果添加到CSV文件
    test_results_df = pd.DataFrame({
        'metric': list(test_metrics.keys()),
        'test_value': list(test_metrics.values())
    })
    test_results_df.to_csv('results/test_metrics.csv', index=False)
    
    print("\n测试集结果:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    main()

