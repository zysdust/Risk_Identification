import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
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
        self.data = pd.read_csv(self.data_path)
        print(f"数据集形状: {self.data.shape}")
        print("\n数据基本信息:")
        print(self.data.info())
        print("\n缺失值统计:")
        print(self.data.isnull().sum())
        return self
        
    def analyze_data(self):
        """数据分析"""
        print("\n数据基本统计描述:")
        print(self.data.describe())
        
        # 查看标签分布
        print("\n标签分布:")
        print(self.data['Label'].value_counts(normalize=True))
        
        # 保存标签分布图
        plt.figure(figsize=(8, 6))
        self.data['Label'].value_counts().plot(kind='bar')
        plt.title('标签分布')
        plt.xlabel('类别')
        plt.ylabel('数量')
        plt.savefig('label_distribution.png')
        plt.close()
        
        return self
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n开始数据预处理...")
        
        # 分离特征和标签
        self.X = self.data.drop(['ID', 'Label'], axis=1)
        self.y = self.data['Label']
        
        # 标准化特征
        scaler = StandardScaler()
        self.X = pd.DataFrame(
            scaler.fit_transform(self.X),
            columns=self.X.columns
        )
        
        print("特征标准化完成")
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
    
    def save_processed_data(self):
        """保存处理后的数据"""
        print("\n保存处理后的数据...")
        
        # 保存训练集
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        # train_data.to_csv('processed_train.csv', index=False)
        
        # 保存测试集
        test_data = pd.concat([self.X_test, self.y_test], axis=1)
        # test_data.to_csv('processed_test.csv', index=False)
        
        print("数据保存完成")
        return self

if __name__ == "__main__":
    # 创建数据处理器实例
    processor = DataProcessor('Data/train_mini.csv')
    
    # 执行数据处理流程
    (processor
     .load_data()
     .analyze_data()
     .preprocess_data()
     .split_data()
     .save_processed_data()
    )
