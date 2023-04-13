# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam, SGD
from loader import load_data
from transformers import BertModel



"""
建立网络模型
1、TorchModel
hidden_size
vocab_size
max_length
class_num
embedding
layer
classify
crf_layer
use_crf
loss
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        train_data = load_data(config["train_data_path"],config,shuffle=False)
        vocab_size = train_data.dataset.vocab_size+1
        # vocab_size = config["vocab_size"]+1
        max_length = config["max_length"]
        class_num = config["class_num"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True,bidirectional=True,num_layers=1)
        self.layer = BertModel.from_pretrained(r"bert-base-chinese",return_dict=False)
        self.classify = nn.Linear(hidden_size*2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        # loss 采用交叉熵损失
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # 当输入真实标签，返回loss值； 无真实标签，返回预测值
    def forward(self, x, target=None):
        # x shape(batch_size, sen_len)
        print("x.shape", x.shape)
        # x = self.embedding(x)
        # 经过embedding后，shape = batch_size,sen_len, hidden_size, hidden_size=LSTM.hidden_size(第二个参数)
        # 线性层
        x, _ = self.layer(x)
        print("bert_x.shape", x.shape)
        print(type(x))
        print(len(x))
        print("layer_x", x.shape)
        # print(type(_))
        # print(len(_))
        # 线性层，输出shape batch_size, sen_len, class_num
        predict = self.classify(x)
        print("predict_shape", predict.shape)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                print("mask", mask)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]),target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict

def choose_optimizer(config,model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    print(model.parameters())
    if optimizer=="Adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer=="SGD":
        return SGD(model.parameters(),lr=learning_rate)



if __name__ == "__main__":
    from config import Config
    from loader import load_data
    data_path = "ner_data/train.txt"
    train_data = load_data(data_path, config=Config, shuffle=False)
    model = TorchModel(Config)
    for index, batch_data in enumerate(train_data):
        print("index", index)
        input_ids, labels = batch_data
        loss = model(input_ids, labels)
        print("loss", loss)

