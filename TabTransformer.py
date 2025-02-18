import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time
import os
import json
import pandas as pd

# 全局配置变量
NUM_EPOCHS = 5  # 训练轮数
DATASET_PATH = 'Data/train.csv'  # 数据集路径

class TabTransformer(Model):
    def __init__(self, 
                 num_features,
                 num_classes=2,
                 num_transformer_blocks=4,
                 num_heads=8,
                 embedding_dim=32,
                 mlp_dim=64,
                 mlp_dropout=0.1,
                 attention_dropout=0.1):
        super(TabTransformer, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # 特征嵌入层
        self.feature_embedder = tf.keras.Sequential([
            layers.Dense(embedding_dim),
            layers.Reshape((-1, embedding_dim))
        ])
        
        # 位置编码层
        self.position_embedding = layers.Embedding(
            input_dim=num_features,
            output_dim=embedding_dim
        )
        
        # Transformer块
        self.transformer_blocks = []
        for _ in range(num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout
                )
            )
        
        # 输出层
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(mlp_dropout)
        self.out = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=False):
        # 输入特征嵌入
        x = self.feature_embedder(inputs)  # [batch_size, num_features, embedding_dim]
        
        # 生成位置编码
        positions = tf.range(start=0, limit=self.num_features, delta=1)
        positions = tf.expand_dims(positions, 0)  # [1, num_features]
        position_embeddings = self.position_embedding(positions)  # [1, num_features, embedding_dim]
        
        # 广播位置编码到所有批次
        position_embeddings = tf.tile(
            position_embeddings, 
            [tf.shape(inputs)[0], 1, 1]
        )  # [batch_size, num_features, embedding_dim]
        
        # 将特征嵌入和位置编码相加
        x = x + position_embeddings  # [batch_size, num_features, embedding_dim]
        
        # Transformer块处理
        for transformer_block in self.transformer_blocks:
            x = transformer_block(inputs=x, training=training)
        
        # 输出层处理
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        return self.out(x)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, mlp_dropout=0.1, attention_dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=attention_dropout
        )
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(mlp_dropout),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(mlp_dropout)
        self.dropout2 = layers.Dropout(mlp_dropout)
        
    def call(self, inputs, training=False):
        # 多头自注意力机制
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # 前馈神经网络
        mlp_output = self.mlp(out1)
        mlp_output = self.dropout2(mlp_output, training=training)
        return self.layernorm2(out1 + mlp_output)

class MetricsHistory(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=None):
        super(MetricsHistory, self).__init__()
        self.validation_data = validation_data
        self.train_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'specificity': [], 'auc': [], 'f1': []
        }
        self.val_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'specificity': [], 'auc': [], 'f1': []
        }
        self.times = []
        self.start_time = None
        self.metrics_df = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        # 初始化DataFrame用于存储所有指标
        self.metrics_df = pd.DataFrame(columns=[
            'epoch', 'time',
            'train_accuracy', 'train_precision', 'train_recall', 
            'train_specificity', 'train_auc', 'train_f1',
            'val_accuracy', 'val_precision', 'val_recall', 
            'val_specificity', 'val_auc', 'val_f1'
        ])
        
    def on_epoch_end(self, epoch, logs=None):
        def calculate_metrics(y_true, y_pred, y_pred_proba):
            y_pred = np.argmax(y_pred, axis=1)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            f1 = f1_score(y_true, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'auc': auc,
                'f1': f1
            }
        
        # 计算训练集指标
        train_pred = self.model.predict(self.model.train_data[0], verbose=0)
        train_metrics = calculate_metrics(
            self.model.train_data[1],
            train_pred,
            train_pred
        )
        
        # 计算验证集指标
        if self.validation_data:
            val_pred = self.model.predict(self.validation_data[0], verbose=0)
            val_metrics = calculate_metrics(
                self.validation_data[1],
                val_pred,
                val_pred
            )
        
        # 记录指标
        current_time = time.time() - self.start_time
        self.times.append(current_time)
        
        # 更新历史记录
        for metric in train_metrics:
            self.train_metrics[metric].append(train_metrics[metric])
            if self.validation_data:
                self.val_metrics[metric].append(val_metrics[metric])
        
        # 将当前轮次的所有指标添加到DataFrame
        metrics_dict = {
            'epoch': epoch + 1,
            'time': current_time
        }
        # 添加训练集指标
        for metric, value in train_metrics.items():
            metrics_dict[f'train_{metric}'] = value
        # 添加验证集指标
        if self.validation_data:
            for metric, value in val_metrics.items():
                metrics_dict[f'val_{metric}'] = value
        
        # 将新的一行添加到DataFrame
        self.metrics_df = pd.concat([
            self.metrics_df, 
            pd.DataFrame([metrics_dict])
        ], ignore_index=True)
        
        # 每轮都保存CSV文件
        self.metrics_df.to_csv('results/metrics.csv', index=False)

def plot_metrics(history, save_dir='results'):
    """绘制评价指标图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
    fig, axes = plt.subplots(2, 1, figsize=(15, 20))
    
    # 按轮数绘制
    ax = axes[0]
    for metric in metrics:
        ax.plot(history.train_metrics[metric], label=f'Train {metric}')
        ax.plot(history.val_metrics[metric], label=f'Val {metric}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Metrics by Epoch')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    
    # 按时间绘制
    ax = axes[1]
    for metric in metrics:
        ax.plot(history.times, history.train_metrics[metric], label=f'Train {metric}')
        ax.plot(history.times, history.val_metrics[metric], label=f'Val {metric}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Score')
    ax.set_title('Metrics by Time')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'), bbox_inches='tight')
    plt.close()
    
    # 保存指标数据
    metrics_data = {
        'train_metrics': history.train_metrics,
        'val_metrics': history.val_metrics,
        'times': history.times
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

def create_tabtransformer_model(num_features):
    model = TabTransformer(
        num_features=num_features,
        num_classes=2,
        num_transformer_blocks=4,
        num_heads=8,
        embedding_dim=32,
        mlp_dim=64,
        mlp_dropout=0.1,
        attention_dropout=0.1
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    from data_processor import DataProcessor
    
    # 加载和处理数据
    processor = DataProcessor(DATASET_PATH)
    processor.load_data()
    processor.preprocess_data()
    processor.split_data(test_size=0.2)
    
    # 创建和训练模型
    model = create_tabtransformer_model(processor.X_train.shape[1])
    
    # 创建验证集
    val_split = 0.2
    val_size = int(len(processor.X_train) * val_split)
    X_val = processor.X_train[-val_size:]
    y_val = processor.y_train[-val_size:]
    X_train = processor.X_train[:-val_size]
    y_train = processor.y_train[:-val_size]
    
    # 创建指标记录器
    metrics_history = MetricsHistory(validation_data=(X_val, y_val))
    model.train_data = (X_train, y_train)
    
    # 训练模型（移除早停机制）
    history = model.fit(
        X_train,
        y_train,
        epochs=NUM_EPOCHS,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[metrics_history]
    )
    
    # 评估模型并保存结果
    test_loss, test_accuracy = model.evaluate(processor.X_test, processor.y_test)
    print(f"\n测试集准确率: {test_accuracy:.4f}")
    
    # 绘制并保存评价指标图
    plot_metrics(metrics_history)
    
    # 在测试集上计算并保存最终指标
    test_pred = model.predict(processor.X_test)
    test_pred_classes = np.argmax(test_pred, axis=1)
    
    final_metrics = {
        'accuracy': test_accuracy,
        'precision': precision_score(processor.y_test, test_pred_classes),
        'recall': recall_score(processor.y_test, test_pred_classes),
        'specificity': (confusion_matrix(processor.y_test, test_pred_classes).ravel()[0] /
                       (confusion_matrix(processor.y_test, test_pred_classes).ravel()[0] +
                        confusion_matrix(processor.y_test, test_pred_classes).ravel()[1])),
        'auc': roc_auc_score(processor.y_test, test_pred[:, 1]),
        'f1': f1_score(processor.y_test, test_pred_classes)
    }
    
    # 保存最终测试集指标
    with open('results/test_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print("\n最终测试集评价指标:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

