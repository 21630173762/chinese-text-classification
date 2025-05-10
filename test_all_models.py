import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import joblib
from transformers import AutoTokenizer
from models.base_model import (
    BERTModel, TextCNN, LSTMModel, GRUModel,
    BiLSTMModel, BiGRUModel, TransformerModel,
    HANModel, DPCNNModel, RCNNModel
)
from models.traditional_models import TraditionalModel
from utils.lstm_tokenizer import LSTMTokenizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    texts = []
    labels = []
    # 定义类别到索引的映射
    class_to_idx = {
        '体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4,
        '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            texts.append(text)
            # 使用映射将中文标签转换为数字
            labels.append(class_to_idx[label])
    return texts, labels

def create_data_loader(texts, labels, tokenizer, batch_size):
    # 对文本进行编码
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    
    # 创建数据集
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['label'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = TextDataset(encodings, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontproperties='Microsoft YaHei')
    plt.ylabel('真实标签', fontproperties='Microsoft YaHei')
    plt.xlabel('预测标签', fontproperties='Microsoft YaHei')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_accuracy(y_true, y_pred, class_names, model_name, save_path):
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(len(class_names)):
        mask = (y_true == i)
        if sum(mask) > 0:
            accuracy = sum((y_true[mask] == y_pred[mask])) / sum(mask)
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0)
    
    # 创建柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracies)
    
    # 在柱子上添加具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    plt.title(f'{model_name}模型各类别准确率', fontproperties='Microsoft YaHei')
    plt.xlabel('类别', fontproperties='Microsoft YaHei')
    plt.ylabel('准确率', fontproperties='Microsoft YaHei')
    plt.ylim(0, 1.1)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, data_loader, device, class_names, model_name, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, f'{model_name}混淆矩阵', cm_path)
    
    # 绘制类别准确率柱状图
    accuracy_path = os.path.join(save_dir, f'{model_name}_accuracy.png')
    plot_class_accuracy(np.array(all_labels), np.array(all_preds), class_names, model_name, accuracy_path)
    
    # 打印分类报告
    print(f"\n{model_name}模型评估结果：")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_labels

def evaluate_traditional_model(model, texts, labels, class_names, model_name, save_dir):
    # 预处理文本
    processed_texts = [preprocess_text(text) for text in texts]
    
    # 预测
    preds = model.predict(processed_texts)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    cm_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, f'{model_name}混淆矩阵', cm_path)
    
    # 绘制类别准确率柱状图
    accuracy_path = os.path.join(save_dir, f'{model_name}_accuracy.png')
    plot_class_accuracy(np.array(labels), np.array(preds), class_names, model_name, accuracy_path)
    
    # 打印分类报告
    print(f"\n{model_name}模型评估结果：")
    print(classification_report(labels, preds, target_names=class_names))
    
    return preds, labels

def preprocess_text(text):
    # 简单的文本预处理
    # 这里可以根据需要添加更多的预处理步骤
    return text.strip()

def main():
    # 设置参数
    data_path = "D:/c/AICode/work/data/processed/cnews.val.processed.txt"
    model_dir = "D:/c/AICode/work/models/saved"
    save_dir = os.path.join(model_dir, "images")
    os.makedirs(save_dir, exist_ok=True)
    
    # 类别名称
    class_names = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    texts, labels = load_data(data_path)
    
    # 加载tokenizers
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
    lstm_tokenizer = LSTMTokenizer(vocab_size=50000)
    lstm_tokenizer.fit(texts)  # 使用训练数据构建LSTM的词表
    
    # 定义深度学习模型配置
    dl_model_configs = {
        'bert': {'type': BERTModel, 'params': {}},
        'textcnn': {'type': TextCNN, 'params': {'vocab_size': bert_tokenizer.vocab_size, 'embedding_dim': 300}},
        'gru': {'type': GRUModel, 'params': {'vocab_size': bert_tokenizer.vocab_size, 'embedding_dim': 300, 'hidden_dim': 256}},
        'bigru': {'type': BiGRUModel, 'params': {'vocab_size': bert_tokenizer.vocab_size, 'embedding_dim': 300, 'hidden_dim': 256}},
        'transformer': {'type': TransformerModel, 'params': {'vocab_size': bert_tokenizer.vocab_size, 'embedding_dim': 256}},
        'han': {'type': HANModel, 'params': {'vocab_size': bert_tokenizer.vocab_size, 'embedding_dim': 300, 'hidden_dim': 256}},
        'dpcnn': {'type': DPCNNModel, 'params': {'vocab_size': bert_tokenizer.vocab_size, 'embedding_dim': 300}},
        'rcnn': {'type': RCNNModel, 'params': {'vocab_size': bert_tokenizer.vocab_size, 'embedding_dim': 300, 'hidden_dim': 256}}
    }
    
    # 定义传统机器学习模型配置
    traditional_models = ['naive_bayes', 'svm', 'logistic', 'random_forest']
    
    # 测试所有神经网络模型
    print("\n开始测试神经网络模型:")
    for model_type in dl_model_configs.keys():
        print(f"\n开始测试 {model_type} 模型...")
        
        # 根据模型类型选择tokenizer
        if model_type in ['lstm', 'bilstm']:
            tokenizer = lstm_tokenizer
            print(f"使用 LSTM tokenizer 处理 {model_type} 模型的数据")
        else:
            tokenizer = bert_tokenizer
            print(f"使用 BERT tokenizer 处理 {model_type} 模型的数据")
        
        # 创建数据加载器
        data_loader = create_data_loader(texts, labels, tokenizer, batch_size=32)
        print(f"数据加载器创建完成，数据集大小: {len(data_loader.dataset)}")
        
        # 初始化模型
        try:
            model_config = dl_model_configs[model_type]
            model = model_config['type'](**model_config['params']).to(device)
            print(f"{model_type} 模型初始化完成")
            
            # 加载模型权重
            model_path = os.path.join(model_dir, f'best_model_{model_type}.pt')
            if os.path.exists(model_path):
                model.load(model_path)
                model.to(device)
                
                # 评估模型
                evaluate_model(model, data_loader, device, class_names, model_type, save_dir)
            else:
                print(f"模型文件 {model_path} 不存在，跳过测试")
                
        except Exception as e:
            print(f"测试 {model_type} 模型时出错：{str(e)}")
            continue
    
    # 测试传统机器学习模型
    print("\n开始测试传统机器学习模型...")
    for model_name in traditional_models:
        try:
            print(f"\n正在测试 {model_name} 模型...")
            
            # 加载模型
            model_path = os.path.join(model_dir, f'best_model_{model_name}.pkl')
            if os.path.exists(model_path):
                try:
                    # 首先尝试使用joblib加载
                    model = joblib.load(model_path)
                except:
                    try:
                        # 如果失败，尝试使用pickle加载
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                    except Exception as e:
                        print(f"无法加载模型文件 {model_path}：{str(e)}")
                        continue
                
                # 评估模型
                evaluate_traditional_model(model, texts, labels, class_names, model_name, save_dir)
            else:
                print(f"模型文件 {model_path} 不存在，跳过测试")
                
        except Exception as e:
            print(f"测试 {model_name} 模型时出错：{str(e)}")
            continue

if __name__ == '__main__':
    main() 