import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.linear_layer = nn.Linear(self.bert.config.hidden_size, config["class_num"])
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None):
        x, _ = self.bert(x)
        x = self.linear_layer(x)
        if y is not None:
            return self.loss(x.view(-1, x.shape[-1]), y.view(-1))
        else:
            return x


def choose_optimizer(config, model):
    lr = config["learning_rate"]
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)