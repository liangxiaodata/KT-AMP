import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('prot_t5_xl_half_uniref50-enc', do_lower_case=False)

train_sequence = pd.read_csv('data/ACP_dataset/tsv/AB_train.csv',sep='\t')['text'].tolist()
test_sequence = pd.read_csv('data/ACP_dataset/tsv/AB_test.csv',sep='\t')['text'].tolist()

train_label = pd.read_csv('data/ACP_dataset/tsv/AB_train.csv',sep='\t')['label'].tolist()
test_label = pd.read_csv('data/ACP_dataset/tsv/AB_test.csv',sep='\t')['label'].tolist()

train_sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in train_sequence]
ids = tokenizer.batch_encode_plus(train_sequence_examples, add_special_tokens=True, padding="longest")
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

test_sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in test_sequence]
ids_test = tokenizer.batch_encode_plus(test_sequence_examples, add_special_tokens=True, padding="longest")
input_ids_test = torch.tensor(ids_test['input_ids']).to(device)
attention_mask_test = torch.tensor(ids_test['attention_mask']).to(device)

from torch.utils.data import Dataset, DataLoader

# 定义一个自定义数据集
class MyDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# 创建数据集实例
dataset = MyDataset(input_ids, attention_mask, train_label)
dataset_test = MyDataset(input_ids_test, attention_mask_test, test_label)# 替换 labels 为你的标签数据

# 定义批处理大小和是否随机打乱数据
batch_size = 8
shuffle = True

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)



import torch.nn as nn

class ImprovedBinaryClassificationMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256, 128, 64, 32], output_dim=2):
        super(ImprovedBinaryClassificationMLP, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())

        # 中间层，增加残差连接
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            # if i < len(hidden_dims) - 2:  # 不在最后一层使用残差连接
            #     self.layers.append(nn.Identity())

        # 输出层
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Identity):
                # 残差连接，将前一层的输出与当前层的输出相加
                x = x + layer(x)
            else:
                x = layer(x)

        return x



import torch
import torch.nn as nn

class PretrainedAndClassifierModel(nn.Module):
    def __init__(self):
        super(PretrainedAndClassifierModel, self).__init__()

        # 实例化预训练模型和分类头
        #self.pretrained_model = T5EncoderModel.from_pretrained("prot_t5_xl_half_uniref50-enc")
        self.pretrained_model = T5EncoderModel.from_pretrained("finetune_peptide_model_with_best_performance")
        self.classifier_model = ImprovedBinaryClassificationMLP(input_dim=1024, hidden_dims=[512, 256, 128, 64, 32], output_dim=2)

        #for param in self.pretrained_model.parameters():
        #        param.requires_grad = False

        for name, param in self.pretrained_model.named_parameters():
            if not name.startswith('encoder.block.23.') and not name.startswith('encoder.final_layer_norm'):
                param.requires_grad = False

        # 自适应平均池化
        self.average_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, attention_mask):
        # 前向传播到预训练模型
        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask)
        emb_0 = outputs.last_hidden_state
        # 平均池化
        pooled_sequence = self.average_pooling(emb_0.permute(0, 2, 1)).squeeze(2)
        

        # 前向传播到分类头
        proc_feat = self.classifier_model(pooled_sequence)

        return proc_feat

# 创建整合的模型


integrated_model = PretrainedAndClassifierModel().to(device)

import torch.optim as optim
from sklearn.metrics import accuracy_score, matthews_corrcoef,roc_auc_score,roc_curve
from tqdm import tqdm
import numpy as np



optimizer = optim.Adam(integrated_model.parameters(), lr=0.00005)

# 定义损失函数，例如交叉熵损失
criterion = nn.CrossEntropyLoss()
for epoch in range(50):
    running_loss = 0.0
    print('epoch:', epoch + 1)
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = integrated_model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    with torch.no_grad():
        integrated_model.eval()  # 设置模型为评估模式
        total_labels = []
        total_probabilities = []

        for test_batch in test_dataloader:
            test_input_ids = test_batch['input_ids'].to(device)
            test_attention_mask = test_batch['attention_mask'].to(device)
            test_labels = test_batch['labels'].to(device)

            test_outputs = integrated_model(test_input_ids, attention_mask=test_attention_mask)
            # 获取模型的预测概率
            probabilities = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()

            total_labels.extend(test_labels.cpu().numpy())
            total_probabilities.extend(probabilities)

        # 计算 AUC
        test_auc = roc_auc_score(total_labels, total_probabilities)
        accuracy = accuracy_score(total_labels, [1 if prob >= 0.5 else 0 for prob in total_probabilities])

        # 计算 Matthews相关系数
        mcc = matthews_corrcoef(total_labels, [1 if prob >= 0.5 else 0 for prob in total_probabilities])
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}, Test AUC: {test_auc:.4f}, Test Accuracy: {accuracy:.4f}, Test MCC: {mcc:.4f}')
    


