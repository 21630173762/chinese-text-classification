import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm as tqdm_base
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.data_processor import load_data, create_data_loaders, preprocess_text
from models.base_model import (
    BERTModel, TextCNN, LSTMModel, GRUModel,
    BiLSTMModel, BiGRUModel, TransformerModel,
    HANModel, DPCNNModel, RCNNModel
)
from models.traditional_models import TraditionalModel
from transformers import AutoTokenizer
from utils.lstm_tokenizer import LSTMTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 配置日志
def setup_logging(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 获取当前时间并格式化为文件名
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{current_time}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 自定义tqdm类，将输出重定向到日志
class TqdmLoggingHandler(tqdm_base):
    def __init__(self, *args, logger=None, **kwargs):
        self.logger = logger
        self.has_logged = False
        super().__init__(*args, **kwargs)
    
    def display(self, msg=None, pos=None):
        # 只在进度条完成时记录一次日志
        if self.n == self.total and self.logger and not self.has_logged:
            self.has_logged = True
            self.logger.info(msg if msg is not None else self.__str__())
        super().display(msg, pos)

def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def train_neural_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=2e-5, savedpath=None, logger=None):
    if logger is None:
        logger = setup_logging(os.path.join(savedpath, 'logs') if savedpath else 'logs')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 用于存储训练过程中的损失和准确率
    train_losses = []
    train_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = TqdmLoggingHandler(
            train_loader, 
            desc=f'Epoch {epoch+1}/{num_epochs}',
            logger=logger
        )
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        # 计算平均训练损失和准确率
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs} 完成:')
        logger.info(f'训练损失: {avg_train_loss:.4f}, 训练准确率: {avg_train_acc:.2f}%')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    
    # 保存图像
    if savedpath:
        image_dir = os.path.join(savedpath, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        plt.savefig(os.path.join(image_dir, f'{model.__class__.__name__}_loss_acc.png'))
        plt.close()
    else:
        plt.show()
    
    return train_accs[-1]  # 返回最后一个epoch的训练准确率

def evaluate_neural_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(classification_report(all_labels, all_preds))
    return accuracy

def train_traditional_model(model, texts, labels):
    # 预处理文本
    processed_texts = [preprocess_text(text) for text in texts]
    
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model.train(train_texts, train_labels)
    
    # 验证
    val_preds = model.predict(val_texts)
    val_acc = 100 * np.mean(val_preds == val_labels)
    print(f'Validation Accuracy: {val_acc:.2f}%')
    print(classification_report(val_labels, val_preds))
    
    # 保存模型
    model.save(f'best_model_{model.model_type}.pkl')
    
    return val_acc

def test_all_models(data_path, savedpath, batch_size=32, num_epochs=30, learning_rate=2e-5, 
                   embedding_dim=300, hidden_dim=256):
    logger = setup_logging(os.path.join(savedpath, 'logs'))
    logger.info("开始测试所有模型")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"保存路径: {savedpath}")
    logger.info(f"批次大小: {batch_size}, 学习率: {learning_rate}")
    
    # 加载数据
    logger.info("正在加载数据...")
    texts, labels = load_data(data_path)
    logger.info(f"加载完成，共 {len(texts)} 条数据")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 定义所有模型类型
    neural_models = [
         'lstm',
    ]
    
    traditional_models = [
        'naive_bayes', 'svm', 'logistic', 'random_forest'
    ]
    
    #创建保存路径
    if not os.path.exists(savedpath):
        os.makedirs(savedpath)
        logger.info(f"创建保存目录: {savedpath}")
    
    # 加载tokenizers
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
    lstm_tokenizer = LSTMTokenizer(vocab_size=50000)
    lstm_tokenizer.fit(texts)  # 使用训练数据构建LSTM的词表
    
    # 测试所有神经网络模型
    logger.info("\n开始测试神经网络模型:")
    for model_type in neural_models:
        logger.info(f"\n开始测试 {model_type} 模型...")
        
        # 根据模型类型选择tokenizer和epoch数
        if model_type == 'bert':
            tokenizer = bert_tokenizer
            current_epochs = 5
            logger.info(f"使用 BERT tokenizer 处理 {model_type} 模型的数据，训练轮数: {current_epochs}")
        elif model_type in ['lstm', 'bilstm']:
            tokenizer = lstm_tokenizer
            current_epochs = 30
            logger.info(f"使用 LSTM tokenizer 处理 {model_type} 模型的数据，训练轮数: {current_epochs}")
        else:
            tokenizer = bert_tokenizer
            current_epochs = 30
            logger.info(f"使用 BERT tokenizer 处理 {model_type} 模型的数据，训练轮数: {current_epochs}")
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(texts, labels, tokenizer, batch_size)
        logger.info(f"数据加载器创建完成，训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
        
        # 初始化模型
        try:
            if model_type == 'bert':
                model = BERTModel().to(device)
            elif model_type == 'textcnn':
                model = TextCNN(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim).to(device)
            elif model_type == 'lstm':
                model = LSTMModel(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, 
                                 hidden_dim=hidden_dim).to(device)
            elif model_type == 'gru':
                model = GRUModel(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, 
                                hidden_dim=hidden_dim).to(device)
            elif model_type == 'bilstm':
                model = BiLSTMModel(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, 
                                   hidden_dim=hidden_dim).to(device)
            elif model_type == 'bigru':
                model = BiGRUModel(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, 
                                  hidden_dim=hidden_dim).to(device)
            elif model_type == 'transformer':
                model = TransformerModel(
                    vocab_size=tokenizer.vocab_size, 
                    embedding_dim=256,
                    num_heads=8
                ).to(device)
            elif model_type == 'han':
                model = HANModel(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, 
                                hidden_dim=hidden_dim).to(device)
            elif model_type == 'dpcnn':
                model = DPCNNModel(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim).to(device)
            elif model_type == 'rcnn':
                model = RCNNModel(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, 
                                 hidden_dim=hidden_dim).to(device)
            
            logger.info(f"{model_type} 模型初始化完成")
            
            # 训练模型
            logger.info(f"开始训练 {model_type} 模型...")
            train_neural_model(model, train_loader, val_loader, device, current_epochs, learning_rate, savedpath, logger)
            
            # 评估模型
            val_acc = evaluate_neural_model(model, val_loader, device)
            logger.info(f"{model_type} 模型验证准确率: {val_acc:.2f}%")
            
            # 保存模型
            model_path = os.path.join(savedpath, f'best_model_{model_type}.pt')
            model.save(model_path)
            logger.info(f"模型已保存到: {model_path}")
            
        except Exception as e:
            logger.error(f"{model_type} 模型训练过程中出现错误: {e}")
            continue
    
    # 测试所有传统机器学习模型
    logger.info("\n开始测试传统机器学习模型:")
    for model_type in traditional_models:
        logger.info(f"\n开始测试 {model_type} 模型...")
        try:
            model = TraditionalModel(model_type=model_type)
            val_acc = train_traditional_model(model, texts, labels)
            logger.info(f"{model_type} 模型验证准确率: {val_acc:.2f}%")
            
            # 保存模型
            model_path = os.path.join(savedpath, f'best_model_{model_type}.pkl')
            model.save(model_path)
            logger.info(f"模型已保存到: {model_path}")
            
        except Exception as e:
            logger.error(f"{model_type} 模型训练过程中出现错误: {e}")
            continue
    
    logger.info("所有模型测试完成")

def main():
    # 设置默认参数
    data_path = "D:/c/AICode/work/data/processed/cnews.train.processed.txt"
    savedpath = "D:/c/AICode/work/models/saved"
    batch_size = 32
    num_epochs = 1
    learning_rate = 2e-5
    embedding_dim = 300
    hidden_dim = 256
    
    # 直接运行所有模型测试
    test_all_models(
        data_path,
        savedpath,
        batch_size,
        num_epochs,
        learning_rate,
        embedding_dim,
        hidden_dim
    )

if __name__ == '__main__':
    main() 