# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        # config["hidden_size"] = 128
        hidden_size = config["hidden_size"]
        # config["vocab_size"] chars.txt 字符集的个数 = 4622  vocab_size=4622+1
        vocab_size = config["vocab_size"] + 1
        # config["max_length"] = 20
        max_length = config["max_length"]
        # index = padding_idx 的行用0补充，这里padding_idx=0 即第一行全为0，若padding_idx=1，则index=1的行全为0
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)  # padding_idx=0 意思是下标为0 的行设为0
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        sentence_length = torch.sum(x.gt(0), dim=-1)
        x = self.embedding(x)
        #使用lstm可以使用以下部分
        # x = pack_padded_sequence(x, sentence_length, batch_first=True, enforce_sorted=False)
        # x, _ = self.layer(x)
        # x, _ = pad_packed_sequence(x, batch_first=True)
        #使用线性层   self.layer = nn.Linear(hidden_size, hidden_size)
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x

# 孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    # torch.nn.functional.normalize(tensor1, dim=-1) tensor1 的每个值除以范数
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        # print("tensor1", tensor1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        # print("tensor2", tensor2) torch.mul 对两个张量进行逐个元素相乘
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        # print("cosine", cosine)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        # p为与a同一类的样本，n为与a不同类的样本。要使triplet_loss最小化，则要将 a与p的距离最小化，a与n的距离越大越好，ap-an负的越大越好
        # 损失函数为 diff大于0的值的平均数， 大于0的都是损失，diff越小于0越好，即ap<<an
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        # gt: greate than
        print("diff", diff)
        return torch.mean(diff[diff.gt(0)])

    #sentence: (batch_size, max_length)
    def forward(self,sentence1, sentence2=None, sentence3=None):
        if sentence2 is not None and sentence3 is not None:
            sentence1 = SentenceEncoder(sentence1)
            sentence2 = SentenceEncoder(sentence2)
            sentence3 = SentenceEncoder(sentence3)
            return self.cosine_triplet_loss(sentence1, sentence2, sentence3)
        else:
            return self.sentence_encode(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    # print("model.parameters", list(model.parameters()))
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, l)
    print("y",y)
    # print(model.state_dict())
