# 中文文本分类模型训练框架

这是一个用于中文文本分类的模型训练框架，支持多种深度学习模型和传统机器学习模型。

## 支持的模型

### 深度学习模型
- BERT
- TextCNN
- LSTM
- GRU
- BiLSTM
- BiGRU
- Transformer
- HAN (Hierarchical Attention Network)
- DPCNN
- RCNN

### 传统机器学习模型
- 朴素贝叶斯 (Naive Bayes)
- 支持向量机 (SVM)
- 逻辑回归 (Logistic Regression)
- 随机森林 (Random Forest)

## 环境要求

- Python 3.8+
- 剩下的详见requirements.txt

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/your-username/chinese-text-classification.git
cd chinese-text-classification
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构


```
.
├── models/                 # 模型目录
│   ├── base_model.py      # 基础模型定义
│   └── traditional_models.py  # 传统机器学习模型
├── utils/                  # 工具函数
│   ├── data_processor.py  # 数据处理工具
│   ├── lstm_tokenizer.py  # RNN系列的模型分词器
│   └── __init__.py        # 工具包初始化文件
├── test_all_models.py     # 模型测试脚本
├── train.py               # 训练脚本
├── preprocess.py          # 数据预处理脚本
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明
```

## 使用方法

1. 数据预处理：
```bash
python preprocess.py
```

2. 训练模型：
```bash
python train.py
```

## 配置说明

在 `train.py` 中可以修改以下参数：
- `data_path`: 训练数据路径
- `savedpath`: 模型保存路径
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `embedding_dim`: 词向量维度
- `hidden_dim`: 隐藏层维度

## 日志和结果

- 训练日志保存在 `models/saved/logs` 目录下
- 训练好的模型保存在 `models/saved` 目录下
- 训练过程的可视化结果保存在 `models/saved/images` 目录下

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
