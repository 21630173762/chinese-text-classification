import os
import jieba
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def clean_text(text):
    """清洗文本"""
    # 去除特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data(data_path):
    """加载cnews数据集"""
    texts = []
    labels = []
    label_dict = {
        '体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4,
        '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9
    }
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            text = clean_text(text)
            texts.append(text)
            labels.append(label_dict[label])
    
    return texts, labels

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(texts, labels, tokenizer, batch_size=32, val_ratio=0.2):
    """创建数据加载器"""
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_ratio, random_state=42, stratify=labels
    )
    
    # 创建数据集
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def preprocess_text(text):
    """预处理单个文本"""
    text = clean_text(text)
    return ' '.join(jieba.cut(text)) 