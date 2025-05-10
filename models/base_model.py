import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification

class BaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("子类必须实现forward方法")
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))

class BERTModel(BaseModel):
    def __init__(self, num_classes=10):
        super(BERTModel, self).__init__(num_classes)
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class TextCNN(BaseModel):
    def __init__(self, vocab_size, embedding_dim, num_classes=10):
        super(TextCNN, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, embedding_dim)) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class LSTMModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=10, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # 词嵌入
        x = self.embedding(input_ids)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0, float('-inf')
            )
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 多层感知机
        features = self.mlp(context)
        
        # 分类层
        logits = self.fc(features)
        return logits

class GRUModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=10):
        super(GRUModel, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out)
        return logits

class BiLSTMModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=10, num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # 词嵌入
        x = self.embedding(input_ids)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0, float('-inf')
            )
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 多层感知机
        features = self.mlp(context)
        
        # 分类层
        logits = self.fc(features)
        return logits

class BiGRUModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=10):
        super(BiGRUModel, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out)
        return logits

class TransformerModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, num_heads=8, num_layers=6, num_classes=10):
        super(TransformerModel, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  
        x = self.transformer_encoder(x)  
        x = x.mean(dim=1)  
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

class HANModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=10):
        super(HANModel, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.word_gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.sentence_gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.sentence_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        
        x = self.embedding(input_ids)  
        word_output, _ = self.word_gru(x)  
        word_attention = F.softmax(self.word_attention(word_output), dim=1)  
        sentence_vector = torch.sum(word_output * word_attention, dim=1)  
        
        sentence_vector = sentence_vector.unsqueeze(1)
        sentence_output, _ = self.sentence_gru(sentence_vector)  
        sentence_attention = F.softmax(self.sentence_attention(sentence_output), dim=1)  
        document_vector = torch.sum(sentence_output * sentence_attention, dim=1)  
        
        document_vector = self.dropout(document_vector)
        logits = self.fc(document_vector)
        return logits

class DPCNNModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, num_classes=10):
        super(DPCNNModel, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_region = nn.Conv2d(1, 250, (3, embedding_dim), stride=1)
        self.conv = nn.Conv2d(250, 250, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(250, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.unsqueeze(1)
        x = self.conv_region(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        
        while x.size()[2] > 2:
            x = self._block(x)
            
        x = x.squeeze()
        x = self.fc(x)
        return x
        
    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px
        return x

class RCNNModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=10):
        super(RCNNModel, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim * 2 + embedding_dim, 100, kernel_size=3)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  
        lstm_out, _ = self.lstm(x)  
        x = torch.cat([x, lstm_out], dim=2)  
        x = x.permute(0, 2, 1)  
        x = F.relu(self.conv(x))  
        x = self.max_pool(x)  
        x = x.mean(dim=2)  
        x = self.dropout(x)
        logits = self.fc(x)
        return logits