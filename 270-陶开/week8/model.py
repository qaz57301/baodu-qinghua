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
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
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

    def forward(self, sentence1, sentence2=None, target=None):
        print("sentence1", sentence1)
        print("sentence2", sentence2)
        print("target", target)
        #同时传入两个句子
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)
            # print("vector1", vector1)
            # print("vector1.shape", vector1.shape)
            vector2 = self.sentence_encoder(sentence2)
            # print("vector2", vector2)
            # print("vector2.shape", vector2.shape)
            #如果有标签，则计算loss,
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())
            #如果无标签，计算余弦距离
            else:
                return self.cosine_distance(vector1, vector2)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)


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
