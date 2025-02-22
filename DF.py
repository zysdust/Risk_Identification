import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# 全局变量
N_EPOCHS = 100  # 训练轮数
DATASET_PATH = "Data/Tianchi/train.csv"  # 数据集路径

class DeepForest:
    def __init__(self, n_estimators=100, n_cascades=2, n_forests=2):
        """
        初始化深度森林模型
        Args:
            n_estimators: 每个森林的树的数量
            n_cascades: 级联层数
            n_forests: 每层的森林数量（随机森林 + 完全随机森林）
        """
        self.n_estimators = n_estimators
        self.n_cascades = n_cascades
        self.n_forests = n_forests
        self.forests = []
        
        # 用于记录训练过程中的指标
        self.history = {
            'epoch': [],  # 添加轮数记录
            'train_accuracy': [], 'test_accuracy': [],
            'train_precision': [], 'test_precision': [],
            'train_recall': [], 'test_recall': [],
            'train_specificity': [], 'test_specificity': [],
            'train_auc': [], 'test_auc': [],
            'train_f1': [], 'test_f1': [],
            'training_time': []
        }
        
    def _create_forest_layer(self, n_classes):
        """创建一层森林（包含随机森林和完全随机森林）"""
        layer = []
        for _ in range(self.n_forests):
            # 随机森林
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_features='sqrt',
                random_state=42
            )
            # 完全随机森林
            crf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_features=1,
                random_state=42
            )
            layer.extend([rf, crf])
        return layer
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """计算所有评估指标"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # 计算特异度 (Specificity)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        f1 = f1_score(y_true, y_pred)
        
        return accuracy, precision, recall, specificity, auc, f1
    
    def fit(self, X_train, y_train, X_test, y_test):
        """训练深度森林模型并记录训练过程"""
        print("开始训练深度森林模型...")
        start_time = time.time()
        
        for epoch in range(N_EPOCHS):
            print(f"\n训练轮数 {epoch + 1}/{N_EPOCHS}")
            
            n_classes = len(np.unique(y_train))
            n_samples = X_train.shape[0]
            
            self.forests = []
            current_features_train = X_train
            current_features_test = X_test
            
            for cascade in range(self.n_cascades):
                print(f"  训练第 {cascade + 1}/{self.n_cascades} 层级...")
                
                layer = self._create_forest_layer(n_classes)
                layer_predictions_train = np.zeros((n_samples, len(layer) * n_classes))
                layer_predictions_test = np.zeros((X_test.shape[0], len(layer) * n_classes))
                
                for i, forest in enumerate(layer):
                    print(f"    训练第 {i + 1}/{len(layer)} 个森林...")
                    forest.fit(current_features_train, y_train)
                    
                    predictions_train = forest.predict_proba(current_features_train)
                    predictions_test = forest.predict_proba(current_features_test)
                    
                    start_idx = i * n_classes
                    end_idx = (i + 1) * n_classes
                    layer_predictions_train[:, start_idx:end_idx] = predictions_train
                    layer_predictions_test[:, start_idx:end_idx] = predictions_test
                
                self.forests.append(layer)
                
                if cascade < self.n_cascades - 1:
                    current_features_train = np.hstack([current_features_train, layer_predictions_train])
                    current_features_test = np.hstack([current_features_test, layer_predictions_test])
            
            # 计算当前轮次的预测结果
            train_pred = self.predict(X_train)
            train_pred_proba = self.predict_proba(X_train)
            test_pred = self.predict(X_test)
            test_pred_proba = self.predict_proba(X_test)
            
            # 计算并记录所有指标
            train_metrics = self._calculate_metrics(y_train, train_pred, train_pred_proba)
            test_metrics = self._calculate_metrics(y_test, test_pred, test_pred_proba)
            
            current_time = time.time() - start_time
            
            # 更新历史记录
            self.history['epoch'].append(epoch + 1)
            metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
            for i, metric in enumerate(metric_names):
                self.history[f'train_{metric}'].append(train_metrics[i])
                self.history[f'test_{metric}'].append(test_metrics[i])
            self.history['training_time'].append(current_time)
            
            print(f"  当前轮次训练完成:")
            print(f"    训练集准确率: {train_metrics[0]:.4f}")
            print(f"    测试集准确率: {test_metrics[0]:.4f}")
            print(f"    累计训练时间: {current_time:.2f}秒")
    
    def predict_proba(self, X):
        """预测类别概率"""
        n_samples = X.shape[0]
        n_classes = len(self.forests[0][0].classes_)
        current_features = X
        
        # 对每一层进行预测
        for layer in self.forests:
            layer_predictions = np.zeros((n_samples, len(layer) * n_classes))
            
            for i, forest in enumerate(layer):
                predictions = forest.predict_proba(current_features)
                start_idx = i * n_classes
                end_idx = (i + 1) * n_classes
                layer_predictions[:, start_idx:end_idx] = predictions
            
            if layer is not self.forests[-1]:
                current_features = np.hstack([current_features, layer_predictions])
        
        # 返回最后一层的平均预测概率
        final_predictions = layer_predictions.reshape((n_samples, len(layer), n_classes))
        return np.mean(final_predictions, axis=1)
    
    def predict(self, X):
        """预测类别"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def plot_metrics(self, save_path='results'):
        """绘制并保存评估指标图"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc', 'f1']
        fig, axes = plt.subplots(2, 1, figsize=(15, 20))
        
        # 绘制随训练轮数变化的指标
        for metric in metrics:
            axes[0].plot(range(1, self.n_cascades + 1), 
                        self.history[f'train_{metric}'], 
                        marker='o', 
                        label=f'Train {metric.capitalize()}')
            axes[0].plot(range(1, self.n_cascades + 1), 
                        self.history[f'test_{metric}'], 
                        marker='s', 
                        linestyle='--',
                        label=f'Test {metric.capitalize()}')
        
        axes[0].set_xlabel('训练轮数')
        axes[0].set_ylabel('指标值')
        axes[0].set_title('评估指标随训练轮数的变化')
        axes[0].grid(True)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 绘制随时间变化的指标
        for metric in metrics:
            axes[1].plot(self.history['training_time'], 
                        self.history[f'train_{metric}'], 
                        marker='o', 
                        label=f'Train {metric.capitalize()}')
            axes[1].plot(self.history['training_time'], 
                        self.history[f'test_{metric}'], 
                        marker='s', 
                        linestyle='--',
                        label=f'Test {metric.capitalize()}')
        
        axes[1].set_xlabel('训练时间 (秒)')
        axes[1].set_ylabel('指标值')
        axes[1].set_title('评估指标随训练时间的变化')
        axes[1].grid(True)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(save_path, f'metrics_plot_{timestamp}.png'), 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()
        
        # 保存指标数据
        pd.DataFrame(self.history).to_csv(
            os.path.join(save_path, f'metrics_data_{timestamp}.csv'),
            index=False
        )
        print(f"\n实验结果已保存至 {save_path} 目录")

if __name__ == "__main__":
    # 加载数据
    print("加载数据...")
    data = pd.read_csv(DATASET_PATH)
    X = data.drop("Label", axis=1)
    y = data["Label"]
    
    # 数据预处理
    print("数据预处理...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    print("划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建并训练深度森林模型
    print("\n初始化深度森林模型...")
    df = DeepForest(n_estimators=100, n_cascades=3, n_forests=2)
    
    # 训练模型并记录指标
    df.fit(X_train, y_train, X_test, y_test)
    
    # 绘制并保存评估指标图
    print("\n绘制评估指标图...")
    df.plot_metrics()
    
    # 输出最终评估结果
    print("\n最终评估结果:")
    y_pred = df.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)
    print(f"最终测试集准确率: {final_accuracy:.4f}")

