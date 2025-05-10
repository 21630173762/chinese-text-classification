import jieba
from collections import Counter
import logging
import torch

# 设置jieba的日志级别为WARNING，这样就不会显示DEBUG信息
jieba.setLogLevel(logging.WARNING)

class LSTMTokenizer:
    def __init__(self, vocab_size=50000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def fit(self, texts):
        # 使用结巴分词
        for text in texts:
            words = jieba.lcut(text)
            self.word_freq.update(words)
        
        # 构建词表
        for word, freq in self.word_freq.most_common(self.vocab_size - 2):
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text, max_length=512):
        words = jieba.lcut(text)
        # 截断或填充到max_length
        if len(words) > max_length:
            words = words[:max_length]
        else:
            words = words + ['<PAD>'] * (max_length - len(words))
        
        # 转换为索引
        indices = [self.word2idx.get(word, 1) for word in words]  # 1是<UNK>的索引
        return indices
    
    def decode(self, indices):
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]
        
    def __call__(self, text, max_length=512, padding=True, truncation=True, return_tensors=None):
        # 编码文本
        indices = self.encode(text, max_length)
        
        # 创建attention mask
        attention_mask = [1] * len(indices)
        if padding:
            attention_mask = attention_mask + [0] * (max_length - len(indices))
        
        # 创建返回字典
        result = {
            'input_ids': indices,
            'attention_mask': attention_mask
        }
        
        # 如果需要返回tensor
        if return_tensors == 'pt':
            result = {
                'input_ids': torch.tensor(result['input_ids']),
                'attention_mask': torch.tensor(result['attention_mask'])
            }
            
        return result 